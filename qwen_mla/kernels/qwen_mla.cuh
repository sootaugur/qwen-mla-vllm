/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MLA_FA2_CUH_
#define FLASHINFER_MLA_FA2_CUH_
#include <cooperative_groups.h>

#include <cstdint>
#include <cuda/std/limits>
#include <sstream>

#include "../profiler.cuh"
#include "mla_params.cuh"
#include "prefill.cuh"
#include "variant_helper.cuh"


#ifndef QWEN_MLA_CTA_TILE_Q_HELPER
#define QWEN_MLA_CTA_TILE_Q_HELPER
#include <cuda_runtime.h>
// Shared by MLAPlan (scheduler.cuh) and the launcher (mla.cuh): both MUST derive the same q tile.
// On devices with < 128 KB shared memory per SM (consumer/workstation Blackwell sm_120: 99 KB)
// a 32-row q tile cannot fit next to a 768-wide latent tile, so up to 32 heads use 16-row tiles
// there (q is tiled over packed (token, head) rows, as the stock kernel does for 128 heads).
// Large-smem devices (H100 227 KB) keep the measured 16/32/64 rule unchanged.
static inline uint32_t mla_cta_tile_q(uint32_t num_heads) {
  static const bool small_smem = [] {
    int dev = 0, smem = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    return smem < 128 * 1024;
  }();
  if (num_heads <= 16 || (num_heads <= 32 && small_smem)) return 16;
  return num_heads <= 32 ? 32 : 64;
}
#endif

namespace flashinfer {

namespace mla {

struct StandardAttention : AttentionVariantBase {
  float sm_scale_log2;
  // Per-tensor symmetric-quantization scales for the FP8 KV path. Both 1.0 on
  // the BF16/FP16 path so they are no-ops there.
  float ckv_scale;
  float kpe_scale;

  PROFILER_CLOSURE_PARAMS_DECL

  template <typename Params>
  __device__ __host__ StandardAttention(const Params& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    sm_scale_log2 = params.sm_scale * math::log2e;
    ckv_scale = params.ckv_scale;
    kpe_scale = params.kpe_scale;
  }
};

template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, uint32_t RMS_ELEMS, uint32_t QK_RED_ELEMS, typename DTypeQ,
          typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem_nope[CTA_TILE_Q * HEAD_DIM_CKV];
      alignas(16) DTypeQ q_smem_pe[CTA_TILE_Q * HEAD_DIM_KPE];
      alignas(16) DTypeKV ckv_smem[NUM_STAGES][CTA_TILE_KV * HEAD_DIM_CKV];
      alignas(16) DTypeKV
          kpe_p_smem[NUM_STAGES]
                    [CTA_TILE_KV * (HEAD_DIM_KPE > CTA_TILE_Q ? HEAD_DIM_KPE : CTA_TILE_Q)];
      // [kv_local][q_row] for the current tile; RMS_ELEMS is 1 when the rms tail is off.
      alignas(16) DTypeKV rms_smem[NUM_STAGES][RMS_ELEMS];
      // Partial q*k scores, [warpgroup][warp][lane][reg]. QK_RED_ELEMS is 1 when the D split
      // is off, so this costs nothing in the stock configuration.
      alignas(16) float qk_red[2][4][32][QK_RED_ELEMS];
      union {
        alignas(16) float m_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
        alignas(16) float d_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
      };
    };
    alignas(16) DTypeO o_smem[CTA_TILE_Q * HEAD_DIM_CKV];
  };
};

template <bool CAUSAL_, uint32_t NUM_STAGES_, bool QK_SHARD_, uint32_t HEAD_DIM_CKV_,
          uint32_t HEAD_DIM_KPE_, uint32_t CTA_TILE_Q_, uint32_t CTA_TILE_KV_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename IdType_>
struct KernelTraits {
  static constexpr bool CAUSAL = CAUSAL_;
  static constexpr uint32_t NUM_STAGES = NUM_STAGES_;
  // NOTE(Zihao): whether to shard Q*K computation across warpgroups
  // if true, each warpgroup will compute a subset of Q*K (sharded on the KV dimension)
  // if false, each warpgroup will compute the full Q*K, which is duplicated across warpgroups
  static constexpr bool QK_SHARD = QK_SHARD_;
  static constexpr uint32_t NUM_MMA_KV = CTA_TILE_KV_ / 16;
  static constexpr uint32_t HEAD_DIM_CKV = HEAD_DIM_CKV_;
  static constexpr uint32_t HEAD_DIM_KPE = HEAD_DIM_KPE_;
  static constexpr uint32_t HEAD_DIM_ALL = HEAD_DIM_CKV + HEAD_DIM_KPE;
  static constexpr uint32_t NUM_MMA_D_CKV = HEAD_DIM_CKV / 16;
  static constexpr uint32_t NUM_MMA_D_KPE = HEAD_DIM_KPE / 16;
  static constexpr uint32_t NUM_THREADS = 256;
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = CTA_TILE_KV_;

  // MLA: how the 8 warps (2 warpgroups x 4) are split between q rows and the CKV dimension.
  // Each q-carrying warp owns 16 q rows, so Q_WARPS = CTA_TILE_Q / 16; every warp freed from
  // the q dimension is spent sharding CKV for the P*V accumulation instead, which is what
  // keeps o_frag inside the register budget at HEAD_DIM_CKV > 512.
  //   CTA_TILE_Q 64 -> Q_WARPS 4, D_SHARDS 2  (stock flashinfer, bit-identical)
  //   CTA_TILE_Q 32 -> Q_WARPS 2, D_SHARDS 4  (TP=1, 24 heads)
  //   CTA_TILE_Q 16 -> Q_WARPS 1, D_SHARDS 8  (TP=2, 12 heads)
  static constexpr uint32_t WARPS_PER_WG = 4;
  static constexpr uint32_t Q_WARPS = CTA_TILE_Q / 16;
  static constexpr uint32_t D_SHARDS = 2 * (WARPS_PER_WG / Q_WARPS);
  // Ways the q*k reduction dimension is split across the warps of a warpgroup. 1 when every
  // warp already owns distinct q rows (stock); otherwise they would duplicate each other.
  //
  // Only worth it when q*k is big enough to amortise the cross-warp sum. MEASURED, ms/layer at
  // batch 64 / ctx 32k / 12 heads:
  //   rank  768 (NUM_MMA_D_CKV 48): 1.75 -> 1.08   split wins
  //   rank 1792 (112):              3.95 -> 2.41   split wins
  //   rank  256 (16):               0.71 -> 0.83   split LOSES
  // At rank 256 a warp's share of q*k is 6 mma_d, fewer ldmatrix than the reduction costs.
  // The threshold sits between 16 and 48; 32 is the power of two between them.
  static constexpr uint32_t QD_SHARDS =
      (NUM_MMA_D_CKV >= 32) ? (WARPS_PER_WG / Q_WARPS) : 1;
  static_assert(NUM_MMA_D_CKV % QD_SHARDS == 0 && NUM_MMA_D_KPE % QD_SHARDS == 0,
                "q*k D split must divide both the latent and the rope tile counts");
  static constexpr uint32_t NUM_MMA_D_CKV_PER_WARP = NUM_MMA_D_CKV / D_SHARDS;
  static_assert(CTA_TILE_Q == 16 || CTA_TILE_Q == 32 || CTA_TILE_Q == 64,
                "CTA_TILE_Q must be 16, 32 or 64 (Q_WARPS = CTA_TILE_Q/16 warps of 16 rows)");
  static_assert(NUM_MMA_D_CKV % D_SHARDS == 0,
                "HEAD_DIM_CKV must be a multiple of 16 * D_SHARDS");

  static constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_Q_PE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_CKV = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KPE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_P =
      CTA_TILE_KV >= 64 ? SwizzleMode::k128B : SwizzleMode::k64B;
  static constexpr SwizzleMode SWIZZLE_MODE_O = SwizzleMode::k128B;
  static constexpr uint32_t UPCAST_STRIDE_Q_NOPE = HEAD_DIM_CKV / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_Q_PE = HEAD_DIM_KPE / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_CKV = HEAD_DIM_CKV / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_KPE = HEAD_DIM_KPE / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_FINAL_O = HEAD_DIM_CKV / upcast_size<DTypeO_>();
  static constexpr uint32_t UPCAST_STRIDE_P = CTA_TILE_KV / upcast_size<DTypeKV_>();

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

#ifdef QWEN_MLA_RMS_TAIL
  // MLA only: divide each score by rms(key, head) from the cache row's tail. The macro's
  // VALUE is the element gap between the end of the latent and the first rms slot, i.e. the
  // full packed-rope width (256) -- NOT HEAD_DIM_KPE, which may be narrowed to just the rope
  // groups this TP rank owns. Anchoring on ckv keeps the two independent.
  static constexpr bool RMS_TAIL = true;
  static constexpr uint32_t RMS_GAP = (QWEN_MLA_RMS_TAIL);
#else
  static constexpr bool RMS_TAIL = false;
  static constexpr uint32_t RMS_GAP = 0;
#endif
  // The rms tail is staged with 16-byte cp.async, which needs a 16-byte-aligned source. When it
  // starts mid-chunk (partial rope at TP=2: latent + 116), start RMS_SHIFT elements earlier and
  // index past them; RMS_SHIFT is 0 for every layout whose rms start is already aligned.
  static constexpr uint32_t RMS_SHIFT = RMS_TAIL ? (HEAD_DIM_CKV + RMS_GAP) % 8 : 0;
  static constexpr uint32_t RMS_ROW = ((CTA_TILE_Q + RMS_SHIFT + 7) / 8) * 8;
  static constexpr uint32_t RMS_SMEM_PER_STAGE = RMS_TAIL ? CTA_TILE_KV * RMS_ROW : 1;

  static constexpr uint32_t S_FRAG_REGS = (QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV) * 8;
  // Holding q's fragments across the KV loop costs 4 registers per MMA step this warp owns.
  // Worth it while that fits; past it the spill costs more than the shared reads it saves.
  // MEASURED, ms/layer: rank 768 (56 regs) 1.09 -> 0.99, rank 256 (96) 0.70 -> 0.56,
  // rank 1792 (120) 2.40 -> 3.18. So 96 still pays and 120 does not; the cliff is between.
  static constexpr uint32_t Q_FRAG_REGS = ((NUM_MMA_D_CKV + NUM_MMA_D_KPE) / QD_SHARDS) * 4;
  static constexpr bool HOIST_Q = Q_FRAG_REGS <= 96;
  static constexpr uint32_t Q_NOPE_FRAGS = HOIST_Q ? NUM_MMA_D_CKV / QD_SHARDS : 1;
  static constexpr uint32_t Q_PE_FRAGS = HOIST_Q ? NUM_MMA_D_KPE / QD_SHARDS : 1;
  static constexpr uint32_t QK_RED_ELEMS = QD_SHARDS > 1 ? S_FRAG_REGS : 1;

  using SharedStorage =
      SharedStorageQKVO<NUM_STAGES, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_CKV, HEAD_DIM_KPE,
                        RMS_SMEM_PER_STAGE, QK_RED_ELEMS, DTypeQ, DTypeKV, DTypeO>;
  using AttentionVariant = StandardAttention;

  static constexpr DTypeQKAccum MaskFillValue = -math::inf;
};

// First of this warp's 16 q rows within the CTA tile. Stock (Q_WARPS==4) this is
// warp_idx_in_wg * 16; when warps are spent on D instead, several warps share the same rows and
// redundantly compute the same q*k -- cheaper than a cross-warp reduction, and this kernel is
// bandwidth-bound anyway.
template <typename KTraits>
__device__ __forceinline__ uint32_t mla_q_row_base() {
  return (threadIdx.y % KTraits::Q_WARPS) * 16;
}

// Which slice of CKV this warp accumulates for P*V. Stock (Q_WARPS==4) this reduces to
// warpgroup_idx, i.e. the original 2-way split across warpgroups.
template <typename KTraits>
__device__ __forceinline__ uint32_t mla_d_shard() {
  return threadIdx.z * (KTraits::WARPS_PER_WG / KTraits::Q_WARPS) +
         threadIdx.y / KTraits::Q_WARPS;
}

// Same slice expressed as a starting column, in units of 8 elements (one b128_t), which is what
// get_permuted_offset takes. One mma_d step is 16 elements = 2 columns.
template <typename KTraits>
__device__ __forceinline__ uint32_t mla_d_col_base() {
  return mla_d_shard<KTraits>() * (2 * KTraits::NUM_MMA_D_CKV_PER_WARP);
}

// True for the warps that own q rows, i.e. the ones that must publish s/m/d to shared memory.
// The others hold identical copies; letting them write too would be benign but wasteful.
template <typename KTraits>
__device__ __forceinline__ bool mla_is_q_warp() {
  return threadIdx.y < KTraits::Q_WARPS;
}

template <typename KTraits>
__device__ __forceinline__ void init_states_(float (*o_frag)[8], typename KTraits::DTypeQKAccum* m,
                                             float* d) {
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = 0.f;
    }
  }

#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    m[j] = typename KTraits::DTypeQKAccum(-math::inf);
    d[j] = 1.f;
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeQ* q_nope,
    typename KTraits::DTypeQ* q_pe, const uint32_t q_nope_stride_n, const uint32_t q_nope_stride_h,
    const uint32_t q_pe_stride_n, const uint32_t q_pe_stride_h, const uint32_t q_len,
    const uint32_t packed_offset, const uint_fastdiv& num_heads) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;

  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q_smem_nope(smem_storage->q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem_pe);

  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

#pragma unroll 1
  for (uint32_t idx = tid; idx < CTA_TILE_Q * UPCAST_STRIDE_Q_NOPE;
       idx += KTraits::NUM_THREADS) {
    const uint32_t row = idx / UPCAST_STRIDE_Q_NOPE, col = idx % UPCAST_STRIDE_Q_NOPE;
    uint32_t q, r;
    num_heads.divmod(packed_offset + row, q, r);
    DTypeQ* ptr = q_nope + q * q_nope_stride_n + r * q_nope_stride_h +
                  col * upcast_size<DTypeQ>();
    q_smem_nope.load_128b_async<SharedMemFillMode::kFillZero>(
        q_smem_nope.template get_permuted_offset<UPCAST_STRIDE_Q_NOPE>(row, col), ptr, q < q_len);
  }

#pragma unroll 1
  for (uint32_t idx = tid; idx < CTA_TILE_Q * UPCAST_STRIDE_Q_PE; idx += KTraits::NUM_THREADS) {
    const uint32_t row = idx / UPCAST_STRIDE_Q_PE, col = idx % UPCAST_STRIDE_Q_PE;
    uint32_t q, r;
    num_heads.divmod(packed_offset + row, q, r);
    DTypeQ* ptr = q_pe + q * q_pe_stride_n + r * q_pe_stride_h + col * upcast_size<DTypeQ>();
    q_smem_pe.load_128b_async<SharedMemFillMode::kFillZero>(
        q_smem_pe.template get_permuted_offset<UPCAST_STRIDE_Q_PE>(row, col), ptr, q < q_len);
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_kv(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeKV* ckv,
    typename KTraits::DTypeKV* kpe, typename KTraits::IdType* indices, const uint32_t ckv_stride_n,
    const uint32_t ckv_stride_page, const uint32_t kpe_stride_n, const uint32_t kpe_stride_page,
    const uint32_t packed_kv_bound, const uint32_t packed_block_iter_base,
    const uint_fastdiv& block_size, const uint32_t stage_idx) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);

  if constexpr (KTraits::NUM_MMA_KV == 1) {
    if (warpgroup_idx == 0) {
      uint32_t q, r;
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + warp_idx_in_wg * 4;
      block_size.divmod(packed_block_iter, q, r);

      // Cast page index to int64_t before multiplying to avoid overflow.
      DTypeKV* ckv_ptr =
          ckv +
          static_cast<int64_t>(packed_block_iter < packed_kv_bound ? indices[q] : 0) *
              ckv_stride_page +
          r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr =
          kpe +
          static_cast<int64_t>(packed_block_iter < packed_kv_bound ? indices[q] : 0) *
              kpe_stride_page +
          r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            warp_idx_in_wg * 4 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
        ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(ckv_smem_offset_w, ckv_ptr,
                                                               packed_block_iter < packed_kv_bound);
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
            warp_idx_in_wg * 4 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
        kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(kpe_smem_offset_w, kpe_ptr,
                                                               packed_block_iter < packed_kv_bound);
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      uint32_t q, r;
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 +
                                   (warpgroup_idx + mma_kv * 2) * 16 + warp_idx_in_wg * 4;
      block_size.divmod(packed_block_iter, q, r);

      // See comment above: widen to int64_t to avoid 32-bit overflow when indices[q] is large.
      DTypeKV* ckv_ptr =
          ckv +
          static_cast<int64_t>(packed_block_iter < packed_kv_bound ? indices[q] : 0) *
              ckv_stride_page +
          r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr =
          kpe +
          static_cast<int64_t>(packed_block_iter < packed_kv_bound ? indices[q] : 0) *
              kpe_stride_page +
          r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            32 * mma_kv + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8,
            8 * mma_d + lane_idx % 8);
        ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(ckv_smem_offset_w, ckv_ptr,
                                                               packed_block_iter < packed_kv_bound);
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
            32 * mma_kv + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8,
            8 * mma_d + lane_idx % 8);
        kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(kpe_smem_offset_w, kpe_ptr,
                                                               packed_block_iter < packed_kv_bound);
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  }
}

// Preload this warp's slice of q into MMA fragments. Called once per work item; the KV loop
// then reuses them instead of re-reading q from shared memory every tile.
template <typename KTraits, uint32_t NUM_MMA_D_QK, uint32_t UPCAST_STRIDE_Q,
          SwizzleMode SWIZZLE_MODE_Q>
__device__ __forceinline__ void mla_load_q_frags_(smem_t<SWIZZLE_MODE_Q> q_smem,
                                                   uint32_t (*q_frags)[4]) {
  if constexpr (!KTraits::HOIST_Q) {
    return;
  } else {
    constexpr uint32_t D_STEP = NUM_MMA_D_QK / KTraits::QD_SHARDS;
    const uint32_t lane_idx = threadIdx.x;
    // see compute_qk_: the modulo is required when the size gate forces QD_SHARDS below
    // WARPS_PER_WG / Q_WARPS.
    const uint32_t d_first = ((threadIdx.y / KTraits::Q_WARPS) % KTraits::QD_SHARDS) * D_STEP;
    const uint32_t row = mla_q_row_base<KTraits>() + lane_idx % 16;
#pragma unroll
    for (uint32_t d_i = 0; d_i < D_STEP; ++d_i) {
      uint32_t off = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
          row, (d_first + d_i) * 2 + lane_idx / 16);
      q_smem.ldmatrix_m8n8x4(off, q_frags[d_i]);
    }
  }
}

template <bool init, typename KTraits, uint32_t NUM_MMA_D_QK, uint32_t UPCAST_STRIDE_Q,
          uint32_t UPCAST_STRIDE_K, SwizzleMode SWIZZLE_MODE_Q, SwizzleMode SWIZZLE_MODE_KV>
__device__ __forceinline__ void compute_qk_(smem_t<SWIZZLE_MODE_Q> q_smem,
                                            uint32_t (*q_frags)[4],
                                            smem_t<SWIZZLE_MODE_KV> k_smem,
                                            typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  alignas(16) uint32_t k_frag[4], q_frag_smem[4];
  // Each warp covers only its share of the reduction dimension; mla_qk_reduce_ sums the
  // partials afterwards. QD_SHARDS == 1 restores the full sweep.
  constexpr uint32_t D_STEP = NUM_MMA_D_QK / KTraits::QD_SHARDS;
  // The % QD_SHARDS matters: warp y's D shard is y / Q_WARPS only while QD_SHARDS equals
  // WARPS_PER_WG / Q_WARPS. The size gate can force QD_SHARDS to 1 with Q_WARPS still 1, and
  // without the modulo warps 1-3 then index y * NUM_MMA_D_QK -- past the end of their q and k
  // tiles, reading whatever follows in shared memory.
  const uint32_t d_first = ((threadIdx.y / KTraits::Q_WARPS) % KTraits::QD_SHARDS) * D_STEP;
  // compute q*k^T
#pragma unroll
  for (uint32_t d_i = 0; d_i < D_STEP; ++d_i) {
    const uint32_t mma_d = d_first + d_i;
    uint32_t* q_frag;
    if constexpr (KTraits::HOIST_Q) {
      q_frag = q_frags[d_i];
    } else {
      q_smem.ldmatrix_m8n8x4(q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
                                 mla_q_row_base<KTraits>() + lane_idx % 16,
                                 mma_d * 2 + lane_idx / 16),
                             q_frag_smem);
      q_frag = q_frag_smem;
    }

    if constexpr (KTraits::QK_SHARD) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
            (warpgroup_idx * (KTraits::NUM_MMA_KV / 2) + mma_kv) * 16 + 8 * (lane_idx / 16) +
                lane_idx % 8,
            2 * mma_d + (lane_idx % 16) / 8);

        k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag);

        if (init && d_i == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    } else {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
            mma_kv * 16 + 8 * (lane_idx / 16) + lane_idx % 8, 2 * mma_d + (lane_idx % 16) / 8);

        k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag);

        if (init && d_i == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void logits_mask_(const uint32_t qo_packed_idx_base,
                                             const uint32_t kv_idx_base, const uint32_t qo_len,
                                             const uint32_t kv_len, const uint32_t kv_end,
                                             const uint_fastdiv num_heads,
                                             typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  uint32_t q[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    q[j] = (qo_packed_idx_base + mla_q_row_base<KTraits>() + lane_idx / 4 + 8 * j) /
           num_heads;
  }

  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[(reg_id % 4) / 2],
                       kv_idx = kv_idx_base + warpgroup_idx * (NUM_MMA_KV / 2) * 16 + mma_kv * 16 +
                                2 * (lane_idx % 4) + 8 * (reg_id / 4) + reg_id % 2;
        const bool mask =
            (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                               : kv_idx >= kv_end));
        s_frag[mma_kv][reg_id] = (mask) ? s_frag[mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                             2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                                                             reg_id % 2;
        const bool mask =
            (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                               : kv_idx >= kv_end));
        s_frag[mma_kv][reg_id] = (mask) ? s_frag[mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states_(typename KTraits::SharedStorage* smem_storage,
                                                   const uint32_t stage_idx,
                                                   typename KTraits::AttentionVariant variant,
                                                   typename KTraits::DTypeQKAccum (*s_frag)[8],
                                                   float (*o_frag)[8],
                                                   typename KTraits::DTypeQKAccum* m, float* d) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t warpgroup_idx = threadIdx.z, lane_idx = threadIdx.x, warp_idx_in_wg = threadIdx.y;
  float m_prev[2];
  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m_prev[j] = m[j];
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        float m_local = max(max(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
                            max(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
        m[j] = max(m[j], m_local);
      }
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x2));
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x1));
      if (lane_idx % 4 == 0 && mla_is_q_warp<KTraits>()) {
        smem_storage->m_wg[warpgroup_idx][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4] =
            m[j];
      }
    }

    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[j] = max(smem_storage->m_wg[0][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4],
                 smem_storage->m_wg[1][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4]);
      float o_scale = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);
      d[j] *= o_scale;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;
        o_frag[mma_d][j * 2 + 1] *= o_scale;
        o_frag[mma_d][j * 2 + 4] *= o_scale;
        o_frag[mma_d][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        s_frag[mma_kv][j * 2 + 0] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 0] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 1] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 1] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 4] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 4] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 5] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 5] * sm_scale - m[j] * sm_scale);
      }
    }
  } else {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m_prev[j] = m[j];
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        float m_local = max(max(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
                            max(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
        m[j] = max(m[j], m_local);
      }
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x2));
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x1));
    }

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float o_scale = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);
      d[j] *= o_scale;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;
        o_frag[mma_d][j * 2 + 1] *= o_scale;
        o_frag[mma_d][j * 2 + 4] *= o_scale;
        o_frag[mma_d][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        s_frag[mma_kv][j * 2 + 0] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 0] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 1] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 1] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 4] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 4] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 5] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 5] * sm_scale - m[j] * sm_scale);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_rms_(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeKV* ckv,
    typename KTraits::IdType* indices, const uint32_t ckv_stride_n,
    const uint32_t ckv_stride_page, const uint32_t packed_kv_bound,
    const uint32_t packed_block_iter_base, const uint_fastdiv& block_size,
    const uint32_t stage_idx, const uint32_t qo_packed_idx_base,
    const uint_fastdiv& num_heads) {
  if constexpr (!KTraits::RMS_TAIL) {
    return;
  } else {
    using DTypeKV = typename KTraits::DTypeKV;
    constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    // 128-bit chunks per row: 8 heads each. CTA_TILE_Q is 16/32/64, so this divides.
    constexpr uint32_t SHIFT = KTraits::RMS_SHIFT;
    constexpr uint32_t CHUNKS = KTraits::RMS_ROW / 8;
    static_assert(CTA_TILE_Q % 8 == 0, "rms rows are staged in 8-head 128-bit chunks");

    // cp.async, not a synchronous read: this is issued between load_kv and its
    // commit_group(), so it joins the same async group and is covered by the same
    // wait_group. Staging it synchronously instead put a dependent global load in the
    // pipeline's critical path and cost 30-80% (rank 768: 1.80 -> 2.37 ms/layer).
    smem_t<SwizzleMode::k128B> rms_smem(smem_storage->rms_smem[stage_idx]);
    const uint32_t nh = static_cast<uint32_t>(num_heads);
    // First head of this q tile. Decode packs q as (token, head) with one token per request, so a
    // tile starting at packed row qo_packed_idx_base covers heads [head0, head0 + CTA_TILE_Q). With
    // one tile per request (heads <= CTA_TILE_Q, the H100 rule) head0 is 0; with 16-row tiles for 24
    // heads (small-smem devices) the second tile starts at head 16 and must read THOSE heads' rms.
    const uint32_t head0 = qo_packed_idx_base % nh;
    const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
#pragma unroll 1
    for (uint32_t idx = tid; idx < CTA_TILE_KV * CHUNKS; idx += KTraits::NUM_THREADS) {
      const uint32_t kv_local = idx / CHUNKS, c = idx % CHUNKS;
      uint32_t page, slot;
      const uint32_t it = packed_block_iter_base + kv_local;
      block_size.divmod(it, page, slot);
      DTypeKV* src = ckv +
                     static_cast<int64_t>(it < packed_kv_bound ? indices[page] : 0) *
                         ckv_stride_page +
                     slot * ckv_stride_n + KTraits::HEAD_DIM_CKV + KTraits::RMS_GAP - SHIFT + head0 +
                     c * 8;
      // Chunks past the head count are skipped, which is also what keeps the read inside the
      // row: the cache tail has one rms slot per TP=1 head (24), so the last enabled chunk
      // ends at ceil(num_heads/8)*8 <= 24 for both 12 and 24 local heads. kFillZero makes
      // those lanes 0, and the divide below treats a non-positive rms as 1.
      const bool pred = (it < packed_kv_bound) && (head0 + c * 8 < nh + SHIFT);
      rms_smem.load_128b_async<SharedMemFillMode::kFillZero>(kv_local * CHUNKS + c, src, pred);
    }
    // The staged layout is [kv_local][q_local], unswizzled -- mla_rms_divide_ indexes it with
    // the same flat expression; q_local is the head relative to head0. A multi-token query (a tile
    // spanning two tokens) would need a per-row divmod and a non-contiguous gather.
  }
}

// Divide the raw q.k by rms(key, head), BEFORE sm_scale (which update_mdo_states_ applies) and
// before masking -- the same order as the Triton fork's `qk = qk / rms; qk *= sm_scale`.
// The fragment indexing mirrors logits_mask_ exactly; that function already derives the
// (q row, kv index) pair each register holds.
template <typename KTraits>
__device__ __forceinline__ void mla_rms_divide_(typename KTraits::SharedStorage* smem_storage,
                                                 const uint32_t stage_idx,
                                                 typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  if constexpr (!KTraits::RMS_TAIL) {
    return;
  } else {
    constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z;
    const typename KTraits::DTypeKV* rms = smem_storage->rms_smem[stage_idx];
    const uint32_t q_row_base = mla_q_row_base<KTraits>();
    uint32_t qrow[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      qrow[j] = q_row_base + lane_idx / 4 + 8 * j;
    }
    constexpr uint32_t KV_PER_WG = KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KV_PER_WG; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_local = qrow[(reg_id % 4) / 2];
        const uint32_t kv_local =
            (KTraits::QK_SHARD ? warpgroup_idx * KV_PER_WG * 16 : 0) + mma_kv * 16 +
            2 * (lane_idx % 4) + 8 * (reg_id / 4) + reg_id % 2;
        const float r = float(rms[kv_local * KTraits::RMS_ROW + q_local + KTraits::RMS_SHIFT]);
        s_frag[mma_kv][reg_id] /= (r > 0.f ? r : 1.f);
      }
    }
  }
}

// Sum the per-warp partial q*k across the warps that split the reduction dimension. Every
// warp ends up with the full score fragment, which it needs for its own slice of P*V.
template <typename KTraits>
__device__ __forceinline__ void mla_qk_reduce_(typename KTraits::SharedStorage* smem_storage,
                                                float (*s_frag)[8]) {
  if constexpr (KTraits::QD_SHARDS == 1) {
    return;
  } else {
    constexpr uint32_t N = KTraits::S_FRAG_REGS;
    // warp y owns (q_group = y % Q_WARPS, d_shard = y / Q_WARPS), so the partials to sum are
    // the warps sharing this q_group: q_group + k * Q_WARPS. Summing 0..QD_SHARDS-1 instead
    // would add warps that hold DIFFERENT q rows -- correct only when Q_WARPS == 1.
    const uint32_t wg = threadIdx.z, w = threadIdx.y, lane = threadIdx.x;
    const uint32_t q_group = w % KTraits::Q_WARPS;
    float* mine = &smem_storage->qk_red[wg][w][lane][0];
    float* flat = reinterpret_cast<float*>(s_frag);
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
      mine[i] = flat[i];
    }
    // A warpgroup-scoped named barrier (bar.sync 1/2, 128 threads) is the correct scope here
    // and measured exactly neutral, so it is not worth the inline PTX: the barrier stall is in
    // the p_smem all-gather and the m_wg/d_wg exchange, not in this one.
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
      float acc = 0.f;
#pragma unroll
      for (uint32_t k = 0; k < KTraits::QD_SHARDS; ++k) {
        acc += smem_storage->qk_red[wg][q_group + k * KTraits::Q_WARPS][lane][i];
      }
      flat[i] = acc;
    }
    // No barrier here: the buffer is not rewritten until the next tile's compute_mla_qk, and
    // update_mdo_states_ and compute_mla_pv both sync before then.
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               uint32_t (*q_nope_frags)[4],
                                               uint32_t (*q_pe_frags)[4],
                                               typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q_smem_nope(smem_storage->q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem_pe);
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  compute_qk_</*init=*/true, KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
              KTraits::UPCAST_STRIDE_KPE, KTraits::SWIZZLE_MODE_Q_PE>(
      q_smem_pe, q_pe_frags, kpe_smem, s_frag);
  compute_qk_</*init=*/false, KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV, KTraits::SWIZZLE_MODE_Q_NOPE>(
      q_smem_nope, q_nope_frags, ckv_smem, s_frag);
  mla_qk_reduce_<KTraits>(smem_storage, s_frag);
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[8],
                                               typename KTraits::DTypeQKAccum* d,
                                               float (*o_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  constexpr uint32_t NUM_MMA_D_CKV_PER_WARP = KTraits::NUM_MMA_D_CKV_PER_WARP;
  const uint32_t d_col_base = mla_d_col_base<KTraits>();
  const uint32_t q_row_base = mla_q_row_base<KTraits>();
  if constexpr (KTraits::QK_SHARD) {
    // shard s_frag computation on KV dimension across warpgroups, need allgather
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV / 2][8];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::cast<8>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }

    __syncthreads();
    smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
    constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          q_row_base + lane_idx % 16, warpgroup_idx * NUM_MMA_KV + mma_kv * 2 + lane_idx / 16);
      // Warps that share q rows hold identical p; only the q-owning ones publish it.
      if (mla_is_q_warp<KTraits>()) {
        p_smem.stmatrix_m8n8x4(p_smem_offset_w, (uint32_t*)p_f16[mma_kv]);
      }
#else
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          q_row_base + lane_idx / 4, warpgroup_idx * NUM_MMA_KV + mma_kv * 2);
      if (mla_is_q_warp<KTraits>())
      {
        ((uint32_t*)(p_smem.base + p_smem_offset_w))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][0];
        ((uint32_t*)(p_smem.base + p_smem_offset_w + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][2];
        ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1)))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][4];
        ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][6];
      }
#endif
    }
    // wait for p_smem to be filled
    __syncthreads();

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t p_frag[4];
      uint32_t p_smem_offset_r = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          q_row_base + lane_idx % 16, mma_kv * 2 + lane_idx / 16);
      p_smem.ldmatrix_m8n8x4(p_smem_offset_r, p_frag);

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
        uint32_t v_frag[4];
        uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            mma_kv * 16 + lane_idx % 16, d_col_base + mma_d * 2 + lane_idx / 16);
        ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag,
                                                                             v_frag);
      }
    }
  } else {
    // no need to store p_smem because all warpgroups are working on the same p
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV][8];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::cast<8>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
        uint32_t v_frag[4];
        uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            mma_kv * 16 + lane_idx % 16, d_col_base + mma_d * 2 + lane_idx / 16);
        ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
            o_frag[mma_d], (uint32_t*)p_f16[mma_kv], v_frag);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d_(typename KTraits::SharedStorage* smem_storage,
                                             const uint32_t stage_idx, float (*o_frag)[8],
                                             typename KTraits::DTypeQKAccum* m, float* d) {
  const uint32_t warpgroup_idx = threadIdx.z, lane_idx = threadIdx.x, warp_idx_in_wg = threadIdx.y;
  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (lane_idx % 4 == 0 && mla_is_q_warp<KTraits>()) {
        smem_storage->d_wg[warpgroup_idx][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4] =
            d[j];
      }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d[j] = smem_storage->d_wg[0][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4] +
             smem_storage->d_wg[1][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4];
    }
  }

  float d_rcp[2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    d_rcp[j] = (m[j] != typename KTraits::DTypeQKAccum(-math::inf)) ? math::ptx_rcp(d[j]) : 0.f;
  }

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = o_frag[mma_d][reg_id] * d_rcp[(reg_id % 4) / 2];
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m_(typename KTraits::AttentionVariant variant,
                                            typename KTraits::DTypeQKAccum* m) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (m[j] != typename KTraits::DTypeQKAccum(-math::inf)) {
        m[j] *= variant.sm_scale_log2;
      }
    }
  }
}

template <typename KTraits>
__device__ void DevicePersistentMergeStates(
    typename KTraits::IdType* merge_packed_offset_start,
    typename KTraits::IdType* merge_packed_offset_end,
    typename KTraits::IdType* merge_partial_packed_offset_start,
    typename KTraits::IdType* merge_partial_packed_offset_end,
    typename KTraits::IdType* merge_partial_stride, typename KTraits::DTypeO* partial_o,
    float* partial_lse, typename KTraits::DTypeO* final_o, float* final_lse,
    const uint32_t o_stride_n, const uint32_t o_stride_h, const uint_fastdiv& num_heads,
    const bool& return_lse_base_on_e) {
  constexpr uint32_t VEC_SIZE = 8;  // partial o has data type float
  constexpr uint32_t NUM_THRS_PER_ROW = KTraits::HEAD_DIM_CKV / VEC_SIZE;
  constexpr uint32_t ROWS_PER_ITERATION = (KTraits::NUM_THREADS) / NUM_THRS_PER_ROW;
  const uint32_t cta_idx = (gridDim.x * blockIdx.y + blockIdx.x);
  const uint32_t thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t offset_start = merge_packed_offset_start[cta_idx];
  const uint32_t len = merge_packed_offset_end[cta_idx] - offset_start;
  const uint32_t partial_offset_start = merge_partial_packed_offset_start[cta_idx];
  const uint32_t partial_offset_end = merge_partial_packed_offset_end[cta_idx];
  const uint32_t stride = merge_partial_stride[cta_idx];
#pragma unroll 1
  for (uint32_t local_packed_offset = thread_id / NUM_THRS_PER_ROW; local_packed_offset < len;
       local_packed_offset += ROWS_PER_ITERATION) {
    uint32_t final_packed_offset = offset_start + local_packed_offset;
    uint32_t q, r;
    num_heads.divmod(final_packed_offset, q, r);
    state_t<VEC_SIZE> st;
#pragma unroll 8
    for (uint32_t partial_packed_offset = partial_offset_start + local_packed_offset;
         partial_packed_offset < partial_offset_end; partial_packed_offset += stride) {
      vec_t<float, VEC_SIZE> o_partial;
      float lse_partial;
      o_partial.cast_load(partial_o + partial_packed_offset * KTraits::HEAD_DIM_CKV +
                          (thread_id % NUM_THRS_PER_ROW) * VEC_SIZE);
      lse_partial = partial_lse[partial_packed_offset];
      st.merge(o_partial, lse_partial, 1);
    }
    st.normalize();
    st.o.cast_store(final_o +
                    (q * o_stride_n + r * o_stride_h + (thread_id % NUM_THRS_PER_ROW) * VEC_SIZE));
    if (final_lse) {
      final_lse[q * num_heads + r] = st.get_lse();
      if (return_lse_base_on_e) {
        final_lse[q * num_heads + r] *= math::loge2;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_empty_o(typename KTraits::DTypeO* final_o, float* final_lse,
                                              typename KTraits::DTypeO* partial_o,
                                              float* partial_lse, const uint32_t o_stride_n,
                                              const uint32_t o_stride_h, const uint32_t q_len,
                                              const uint32_t packed_offset,
                                              const uint_fastdiv& num_heads) {
  const uint32_t thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t total_packed = q_len * static_cast<uint32_t>(num_heads);
  const uint32_t remaining_packed =
      (packed_offset < total_packed) ? total_packed - packed_offset : 0;
  const uint32_t num_valid_packed = remaining_packed < static_cast<uint32_t>(KTraits::CTA_TILE_Q)
                                        ? remaining_packed
                                        : static_cast<uint32_t>(KTraits::CTA_TILE_Q);
  const uint32_t num_elems = num_valid_packed * KTraits::HEAD_DIM_CKV;

  if (partial_o != nullptr) {
    const uint32_t partial_packed_offset = blockIdx.x * KTraits::CTA_TILE_Q;
    for (uint32_t elem_idx = thread_id; elem_idx < num_elems; elem_idx += KTraits::NUM_THREADS) {
      const uint32_t packed_idx = elem_idx / KTraits::HEAD_DIM_CKV;
      const uint32_t dim_idx = elem_idx - packed_idx * KTraits::HEAD_DIM_CKV;
      partial_o[(partial_packed_offset + packed_idx) * KTraits::HEAD_DIM_CKV + dim_idx] =
          typename KTraits::DTypeO(0.f);
    }
    for (uint32_t packed_idx = thread_id; packed_idx < num_valid_packed;
         packed_idx += KTraits::NUM_THREADS) {
      partial_lse[partial_packed_offset + packed_idx] =
          -cuda::std::numeric_limits<float>::infinity();
    }
  } else {
    for (uint32_t elem_idx = thread_id; elem_idx < num_elems; elem_idx += KTraits::NUM_THREADS) {
      const uint32_t packed_idx = elem_idx / KTraits::HEAD_DIM_CKV;
      const uint32_t dim_idx = elem_idx - packed_idx * KTraits::HEAD_DIM_CKV;
      uint32_t q, r;
      num_heads.divmod(packed_offset + packed_idx, q, r);
      final_o[q * o_stride_n + r * o_stride_h + dim_idx] = typename KTraits::DTypeO(0.f);
    }
    if (final_lse) {
      for (uint32_t packed_idx = thread_id; packed_idx < num_valid_packed;
           packed_idx += KTraits::NUM_THREADS) {
        uint32_t q, r;
        num_heads.divmod(packed_offset + packed_idx, q, r);
        final_lse[q * num_heads + r] = -cuda::std::numeric_limits<float>::infinity();
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o(typename KTraits::SharedStorage* smem_storage,
                                        typename KTraits::DTypeO* final_o, float* final_lse,
                                        typename KTraits::DTypeO* partial_o, float* partial_lse,
                                        float (*o_frag)[8], typename KTraits::DTypeQKAccum* m,
                                        float* d, const uint32_t o_stride_n,
                                        const uint32_t o_stride_h, const uint32_t q_len,
                                        const uint32_t packed_offset, const uint_fastdiv& num_heads,
                                        const bool& return_lse_base_on_e) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t UPCAST_STRIDE_FINAL_O = KTraits::UPCAST_STRIDE_FINAL_O;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t q_row_base = mla_q_row_base<KTraits>();
  const uint32_t d_col_base = mla_d_col_base<KTraits>();
  smem_t<KTraits::SWIZZLE_MODE_O> o_smem(smem_storage->o_smem);

  // step 0. o_frag (this warp's q rows x this warp's CKV slice) -> o_smem
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
    uint32_t o_frag_f16[8 / 2];
    vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_d]);
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
    uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
        q_row_base + lane_idx % 16, d_col_base + mma_d * 2 + lane_idx / 16);
    o_smem.template stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
    uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
        q_row_base + lane_idx / 4, d_col_base + mma_d * 2);
    ((uint32_t*)(o_smem.base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
    ((uint32_t*)(o_smem.base + o_smem_offset_w + 8 * UPCAST_STRIDE_FINAL_O))[lane_idx % 4] =
        o_frag_f16[1];
    ((uint32_t*)(o_smem.base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
    ((uint32_t*)(o_smem.base + (o_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_FINAL_O))[lane_idx % 4] =
        o_frag_f16[3];
#endif
  }

  // The [CTA_TILE_Q, HEAD_DIM_CKV] tile is now written by ALL warps (each owning a CKV slice)
  // and read back cooperatively below, so the two phases need a barrier. Stock flashinfer needs
  // none here because every warp read back exactly the rows it had written.
  __syncthreads();

  // lse: one value per q row. Warps sharing q rows hold identical m/d, so only the q-owning
  // ones write.
  if (mla_is_q_warp<KTraits>() && lane_idx % 4 == 0) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      const uint32_t row = q_row_base + 8 * j + lane_idx / 4;
      const float lse = (m[j] == typename KTraits::DTypeQKAccum(-math::inf))
                            ? -cuda::std::numeric_limits<float>::infinity()
                            : math::ptx_log2(d[j]) + float(m[j]);
      if (partial_o != nullptr) {
        if ((packed_offset + row) / num_heads < q_len) {
          partial_lse[blockIdx.x * CTA_TILE_Q + row] = lse;
        }
      } else if (final_lse != nullptr) {
        uint32_t q, r;
        num_heads.divmod(packed_offset + row, q, r);
        if (q < q_len) {
          final_lse[q * num_heads + r] = return_lse_base_on_e ? lse * math::loge2 : lse;
        }
      }
    }
  }

  // step 1. o_smem -> gmem. One 128b store moves 8 elements, so the tile is
  // CTA_TILE_Q * UPCAST_STRIDE_FINAL_O stores, spread over every thread. Doing it
  // cooperatively rather than per-warp keeps this independent of how CKV was sharded, and
  // HEAD_DIM_CKV/8 need not divide the warp count (it is 96 at rank 768).
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
#pragma unroll 1
  for (uint32_t idx = tid; idx < CTA_TILE_Q * UPCAST_STRIDE_FINAL_O;
       idx += KTraits::NUM_THREADS) {
    const uint32_t row = idx / UPCAST_STRIDE_FINAL_O, col = idx % UPCAST_STRIDE_FINAL_O;
    const uint32_t o_smem_offset_r =
        o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(row, col);
    if (partial_o != nullptr) {
      if ((packed_offset + row) / num_heads < q_len) {
        o_smem.template store_128b(o_smem_offset_r,
                                   partial_o + (blockIdx.x * CTA_TILE_Q + row) * HEAD_DIM_CKV +
                                       col * upcast_size<DTypeO>());
      }
    } else {
      uint32_t q, r;
      num_heads.divmod(packed_offset + row, q, r);
      if (q < q_len) {
        o_smem.template store_128b(
            o_smem_offset_r,
            final_o + q * o_stride_n + r * o_stride_h + col * upcast_size<DTypeO>());
      }
    }
  }
}

template <typename KTraits, typename Params>
#ifndef QWEN_MLA_MIN_CTAS
#define QWEN_MLA_MIN_CTAS 1
#endif
__global__ __launch_bounds__(KTraits::NUM_THREADS, QWEN_MLA_MIN_CTAS)
void BatchMLAPagedAttentionKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  extern __shared__ __align__(alignof(typename KTraits::SharedStorage)) uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  typename KTraits::AttentionVariant variant(params, blockIdx.y, smem);

  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = KTraits::SWIZZLE_MODE_Q_NOPE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_PE = KTraits::SWIZZLE_MODE_Q_PE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_CKV = KTraits::SWIZZLE_MODE_CKV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KPE = KTraits::SWIZZLE_MODE_KPE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::CAUSAL;

  DTypeQ* q_nope = params.q_nope;
  DTypeQ* q_pe = params.q_pe;
  DTypeKV* ckv = params.ckv;
  DTypeKV* kpe = params.kpe;
  IdType* kv_indices = params.kv_indices;
  DTypeO* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  float s_frag[KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV][8];
  alignas(16) uint32_t q_nope_frags[KTraits::Q_NOPE_FRAGS][4];
  alignas(16) uint32_t q_pe_frags[KTraits::Q_PE_FRAGS][4];
  alignas(16) float o_frag[KTraits::NUM_MMA_D_CKV_PER_WARP][8];
  float m[2];
  float d[2];

  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t q_nope_stride_n = params.q_nope_stride_n;
  const uint32_t q_nope_stride_h = params.q_nope_stride_h;
  const uint32_t q_pe_stride_n = params.q_pe_stride_n;
  const uint32_t q_pe_stride_h = params.q_pe_stride_h;
  const uint32_t ckv_stride_page = params.ckv_stride_page;
  const uint32_t ckv_stride_n = params.ckv_stride_n;
  const uint32_t kpe_stride_page = params.kpe_stride_page;
  const uint32_t kpe_stride_n = params.kpe_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    const uint32_t q_indptr = params.q_indptr[work_idx];
    const uint32_t kv_indptr = params.kv_indptr[work_idx];
    const int32_t partial_indptr = params.partial_indptr[work_idx];
    const uint32_t q_len = params.q_len[work_idx];
    const uint32_t kv_len = params.kv_len[work_idx];
    const uint32_t packed_qo_start = params.q_start[work_idx];
    const uint32_t kv_start = params.kv_start[work_idx];
    const uint32_t kv_end = params.kv_end[work_idx];

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
    const uint32_t qo_upperbound =
        min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

    init_states_<KTraits>(o_frag, m, d);

    __syncthreads();
    load_q<KTraits>(&smem_storage, q_nope + q_indptr * q_nope_stride_n,
                    q_pe + q_indptr * q_pe_stride_n, q_nope_stride_n, q_nope_stride_h,
                    q_pe_stride_n, q_pe_stride_h, qo_upperbound, qo_packed_idx_base,
                    params.num_heads);

    if (kv_end <= kv_start) {
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();
      write_empty_o<KTraits>(
          final_o + q_indptr * o_stride_n, final_lse ? final_lse + q_indptr * num_heads : nullptr,
          (partial_indptr == -1) ? nullptr : partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
          (partial_indptr == -1) ? nullptr : partial_lse + partial_indptr, o_stride_n, o_stride_h,
          qo_upperbound, qo_packed_idx_base, num_heads);
      continue;
    }

    int kv_tile_idx =
        ceil_div(
            (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                    : kv_end),
            CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    int mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / num_heads) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    uint32_t block_iter_base = kv_indptr * block_size + kv_start;
    // last kv tile
    __syncthreads();
    uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;
    load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                     kpe_stride_n, kpe_stride_page, packed_kv_bound,
                     block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size,
                     kv_tile_idx % NUM_STAGES);
    load_rms_<KTraits>(&smem_storage, ckv, kv_indices, ckv_stride_n, ckv_stride_page,
                       packed_kv_bound, block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size,
                       kv_tile_idx % NUM_STAGES, qo_packed_idx_base, num_heads);
    cp_async::commit_group();
#pragma unroll
    for (int stage_idx = 1; stage_idx < NUM_STAGES; ++stage_idx) {
      if (kv_tile_idx - stage_idx >= 0) {
        load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                         kpe_stride_n, kpe_stride_page, packed_kv_bound,
                         block_iter_base + (kv_tile_idx - stage_idx) * CTA_TILE_KV, block_size,
                         (kv_tile_idx - stage_idx) % NUM_STAGES);
    load_rms_<KTraits>(&smem_storage, ckv, kv_indices, ckv_stride_n, ckv_stride_page,
                       packed_kv_bound, block_iter_base + (kv_tile_idx - stage_idx) * CTA_TILE_KV, block_size,
                       (kv_tile_idx - stage_idx) % NUM_STAGES, qo_packed_idx_base, num_heads);
        cp_async::commit_group();
      }
    }

    // q is in shared memory now (its cp.async group was committed before the KV ones, so the
    // first wait below covers it). Pull this warp's slice into registers once, rather than
    // re-reading it from shared memory on every KV tile.
    cp_async::wait_group<NUM_STAGES - 1>();
    __syncthreads();
    mla_load_q_frags_<KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
                       KTraits::SWIZZLE_MODE_Q_NOPE>(
        smem_t<KTraits::SWIZZLE_MODE_Q_NOPE>(smem_storage.q_smem_nope), q_nope_frags);
    mla_load_q_frags_<KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
                       KTraits::SWIZZLE_MODE_Q_PE>(
        smem_t<KTraits::SWIZZLE_MODE_Q_PE>(smem_storage.q_smem_pe), q_pe_frags);

    // loop with mask
#pragma unroll 1
    for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {
      cp_async::wait_group<NUM_STAGES - 1>();
      __syncthreads();

      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frags,
                              q_pe_frags, s_frag);
      mla_rms_divide_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag);

      // logits mask
      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);

      if (kv_tile_idx - NUM_STAGES >= 0) {
        __syncthreads();
        load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                         kpe_stride_n, kpe_stride_page, packed_kv_bound,
                         block_iter_base + (kv_tile_idx - NUM_STAGES) * CTA_TILE_KV, block_size,
                         (kv_tile_idx - NUM_STAGES) % NUM_STAGES);
    load_rms_<KTraits>(&smem_storage, ckv, kv_indices, ckv_stride_n, ckv_stride_page,
                       packed_kv_bound, block_iter_base + (kv_tile_idx - NUM_STAGES) * CTA_TILE_KV, block_size,
                       (kv_tile_idx - NUM_STAGES) % NUM_STAGES, qo_packed_idx_base, num_heads);
        cp_async::commit_group();
      }
    }

    // loop without mask
#pragma unroll 1
    for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
      cp_async::wait_group<NUM_STAGES - 1>();
      __syncthreads();

      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frags,
                              q_pe_frags, s_frag);
      mla_rms_divide_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);
      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);

      __syncthreads();
      load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                       kpe_stride_n, kpe_stride_page, packed_kv_bound,
                       block_iter_base + (kv_tile_idx - NUM_STAGES) * CTA_TILE_KV, block_size,
                       (kv_tile_idx - NUM_STAGES) % NUM_STAGES);
    load_rms_<KTraits>(&smem_storage, ckv, kv_indices, ckv_stride_n, ckv_stride_page,
                       packed_kv_bound, block_iter_base + (kv_tile_idx - NUM_STAGES) * CTA_TILE_KV, block_size,
                       (kv_tile_idx - NUM_STAGES) % NUM_STAGES, qo_packed_idx_base, num_heads);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    // last tiles
#pragma unroll
    for (; kv_tile_idx >= 0; --kv_tile_idx) {
      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frags,
                              q_pe_frags, s_frag);
      mla_rms_divide_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag);

      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
    }

    __syncthreads();

    // normalize and write back
    normalize_d_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, o_frag, m, d);

    finalize_m_<KTraits>(variant, m);

    write_o<KTraits>(
        &smem_storage, final_o + q_indptr * o_stride_n,
        final_lse ? final_lse + q_indptr * num_heads : nullptr,
        (partial_indptr == -1) ? nullptr : partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
        (partial_indptr == -1) ? nullptr : partial_lse + partial_indptr, o_frag, m, d, o_stride_n,
        o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads, params.return_lse_base_on_e);
  }

  auto grid = cg::this_grid();
  grid.sync();

  // the second stage, merge partial outputs
  DevicePersistentMergeStates<KTraits>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, partial_o, partial_lse, final_o, final_lse, o_stride_n,
      o_stride_h, num_heads, params.return_lse_base_on_e);
}

// Both the host scheduler (MLAPlan) and this launcher must derive the SAME cta_tile_q, or the
// plan's q tiling and the kernel's q tiling disagree and the output is silently wrong. Keyed on
// num_heads because that is the only quantity both sides have.
#ifndef QWEN_MLA_CTA_TILE_Q_FOR
#define QWEN_MLA_CTA_TILE_Q_FOR(num_heads) mla_cta_tile_q(num_heads)
#endif

#include <cstdio>
#include <cstdlib>

template <MaskMode MASK_MODE, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename Params>
cudaError_t BatchMLAPagedAttention(Params params, uint32_t num_blks_x, uint32_t num_blks_y,
                                   cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(32, 4, 2);

  int device;
  int smem_limit_per_sm;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&smem_limit_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  const uint32_t cta_tile_q_rt = QWEN_MLA_CTA_TILE_Q_FOR(static_cast<uint32_t>(params.num_heads));
  bool launched = false;

  // QWEN_MLA_TILE="<stages>,<cta_tile_kv>" pins one config instead of taking the first that fits,
  // so the ladder can be swept without recompiling. Unset in production.
  static const int mla_want_s = [] {
    const char* e = std::getenv("QWEN_MLA_TILE");
    int a = -1, b = -1;
    if (e) sscanf(e, "%d,%d", &a, &b);
    return a;
  }();
  static const int mla_want_kv = [] {
    const char* e = std::getenv("QWEN_MLA_TILE");
    int a = -1, b = -1;
    if (e) sscanf(e, "%d,%d", &a, &b);
    return b;
  }();

// Stock DISPATCH_SMEM_CONFIG chose (NUM_STAGES, CTA_TILE_KV) from the device's shared memory
// and never looked at HEAD_DIM_CKV, which is why 768 died in cudaFuncSetAttribute with a bare
// "invalid argument". Select on the size this instantiation actually needs.
#define QWEN_MLA_LAUNCH(CTQ, STAGES, CTKV)                                                        \
  {                                                                                           \
    using KTraits =                                                                           \
        KernelTraits<CAUSAL, STAGES, ((CTKV) >= 32), HEAD_DIM_CKV, HEAD_DIM_KPE, CTQ, CTKV,   \
                     DTypeQ, DTypeKV, DTypeO, IdType>;                                        \
    constexpr size_t smem_size = sizeof(typename KTraits::SharedStorage);                     \
    if (!launched && smem_size <= static_cast<size_t>(smem_limit_per_sm) &&                   \
        (mla_want_s < 0 || (mla_want_s == (STAGES) && mla_want_kv == (CTKV)))) {           \
      auto kernel = BatchMLAPagedAttentionKernel<KTraits, Params>;                            \
      void* args[] = {(void*)&params};                                                        \
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                                              \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));                   \
      FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args,     \
                                                       smem_size, stream));                   \
      launched = true;                                                                        \
    }                                                                                         \
  }
// Largest KV tile first, two cp.async stages preferred, then one stage at the same width
// before narrowing. Widths >= 32 take the QK_SHARD path; CTA_TILE_KV 16 is last because
// upstream can only reach it on a device with < 92 KB of shared memory, so it is effectively
// untested -- it faults with an invalid __shared__ write at HEAD_DIM_CKV 1792. Every rank MLA
// uses fits at >= 32, so nothing selects it in practice; it stays only as a backstop and the
// numerics gate would catch it.
// Widest KV tile first, 2 stages before 3. MEASURED at batch 64 / ctx 32k / 12 heads / rope
// window 128, sweeping with QWEN_MLA_TILE (ms/layer):
//   rank  256: (2,64) 0.79  (3,64) 0.78  (2,32) 0.98  (1,32) 1.34
//   rank  768: (2,32) 1.93  (3,32) 1.98  (1,32) 2.39   -- nothing at KV 64 fits
//   rank 1792: (1,32) 4.37                             -- nothing else fits
// A third stage never pays: the kernel is persistent (grid == SM count, 1 CTA/SM, 8 warps), so
// the extra buffer cannot buy occupancy and only adds shared memory and scheduling pressure.
#define QWEN_MLA_LAUNCH_SET(CTQ)                                                     \
  if (cta_tile_q_rt == (CTQ)) {                                                  \
    QWEN_MLA_LAUNCH(CTQ, 2, 64) QWEN_MLA_LAUNCH(CTQ, 3, 64) QWEN_MLA_LAUNCH(CTQ, 2, 32)      \
    QWEN_MLA_LAUNCH(CTQ, 3, 32) QWEN_MLA_LAUNCH(CTQ, 1, 64) QWEN_MLA_LAUNCH(CTQ, 1, 32)      \
    QWEN_MLA_LAUNCH(CTQ, 2, 16) QWEN_MLA_LAUNCH(CTQ, 1, 16)                              \
  }

  QWEN_MLA_LAUNCH_SET(16)
  QWEN_MLA_LAUNCH_SET(32)
  QWEN_MLA_LAUNCH_SET(64)

#undef QWEN_MLA_LAUNCH_SET
#undef QWEN_MLA_LAUNCH

  if (!launched) {
    std::ostringstream err;
    err << "MLA: no MLA tile config fits HEAD_DIM_CKV=" << HEAD_DIM_CKV
        << " HEAD_DIM_KPE=" << HEAD_DIM_KPE << " cta_tile_q=" << cta_tile_q_rt
        << " smem_limit=" << smem_limit_per_sm;
    FLASHINFER_ERROR(err.str());
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_FA2_CUH_
