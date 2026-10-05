#!/usr/bin/env python3
"""Derive kernels/qwen_mla.cuh + qwen_scheduler.cuh from flashinfer's installed headers.

WHY THIS EXISTS. flashinfer's fa2 MLA decode kernel is JIT-templated on head_dim_ckv /
head_dim_kpe, so unlike FlashMLA (Hopper-only) and CutlassMLA (hardcoded 512/64) it can be
instantiated at MLA's ranks. Two things stop it being fast there, both measured:

  1. DISPATCH_SMEM_CONFIG picks (NUM_STAGES, CTA_TILE_KV) from the DEVICE's shared memory and
     never looks at HEAD_DIM_CKV. At 768 the (2, 64) branch asks ~330 KB, cudaFuncSetAttribute
     refuses, and the launch dies with a bare "invalid argument".

  2. CTA_TILE_Q is hardcoded 64 because the 4 warps of a warpgroup each own 16 q rows -- right
     for DeepSeek's 128 heads, wrong for our 12 (TP=2) or 24 (TP=1). The P*V accumulator
     o_frag is (CTA_TILE_Q/4 rows) x (HEAD_DIM_CKV / D_SHARDS) per warp, so with D sharded
     only 2 ways it is 192 registers/thread at HEAD_DIM_CKV=768 -- over budget, and it spills.
     Measured on this box (batch 64 / ctx 32k / 12 heads), stock kernel:
         ckv 256 kpe 256 -> o_frag  64 regs -> 2.29 TB/s
         ckv 512 kpe  64 -> o_frag 128 regs -> 2.00 TB/s
         ckv 768 kpe 256 -> o_frag 192 regs -> 0.89 TB/s   <- spilling
     The Triton fork hits the same wall from the other side (255 regs, 102% spill overhead),
     so the accumulator is the binding constraint in both kernels independently.

THE CHANGE. Make CTA_TILE_Q a real parameter and spend the freed warps on the D dimension
instead of on q rows:

    Q_WARPS  = CTA_TILE_Q / 16          # warps carrying q rows: 4 stock, 1 for us
    D_SHARDS = 2 * (4 / Q_WARPS)        # ways CKV is split for P*V: 2 stock, 8 for us
    q_row_base = (warp_idx_in_wg % Q_WARPS) * 16
    d_shard    = warpgroup_idx * (4 / Q_WARPS) + warp_idx_in_wg / Q_WARPS

CTA_TILE_Q=64 reproduces the stock kernel exactly (Q_WARPS=4 -> D_SHARDS=2, q_row_base =
warp_idx_in_wg*16, d_shard = warpgroup_idx), so this is a generalisation rather than a fork:
the DeepSeek path is unchanged and stays available as a control. At CTA_TILE_Q=16 the
accumulator drops 192 -> 48 registers/thread.

MLAPlan derives cta_tile_q from num_heads with the SAME expression as the launcher. Both sides
must agree or the host's q tiling and the kernel's q tiling disagree and the output is silently
wrong, so the expression lives in one macro used by both.

Offsets are recomputed with get_permuted_offset() rather than carried by
advance_offset_by_column/row: the incremental form needs a "- NUM_MMA_D_CKV" correction tuned to
the stock loop bounds, which is exactly the kind of thing that breaks quietly when the bounds
change. get_permuted_offset is i*stride + (j ^ (i%8)) -- cheap, and this is not the hot loop.

Re-run after upgrading flashinfer; it starts from the installed headers every time.
"""
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable   # was the H100 box venv; run this script with the serving venv


def fi_include() -> pathlib.Path:
    out = subprocess.check_output(
        [PY, "-c", "import flashinfer.jit.env as e; print(e.FLASHINFER_INCLUDE_DIR)"],
        text=True,
    ).strip()
    return pathlib.Path(out)


def sub(s: str, old: str, new: str, what: str, count: int = 1) -> str:
    n = s.count(old)
    if n != count:
        raise SystemExit(f"PATCH FAILED [{what}]: expected {count} match(es), found {n}.\n"
                         f"flashinfer's header probably changed; re-read it and update this "
                         f"script.\n--- looking for ---\n{old[:400]}")
    return s.replace(old, new)


def patch_mla(s: str) -> str:
    # ---- 1. KernelTraits: q/D warp split constants -------------------------------------
    s = sub(s, """  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = CTA_TILE_KV_;""",
        """  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
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
  static constexpr uint32_t NUM_MMA_D_CKV_PER_WARP = NUM_MMA_D_CKV / D_SHARDS;
  static_assert(CTA_TILE_Q == 16 || CTA_TILE_Q == 32 || CTA_TILE_Q == 64,
                "CTA_TILE_Q must be 16, 32 or 64 (Q_WARPS = CTA_TILE_Q/16 warps of 16 rows)");
  static_assert(NUM_MMA_D_CKV % D_SHARDS == 0,
                "HEAD_DIM_CKV must be a multiple of 16 * D_SHARDS");""",
        "traits constants")

    # helpers, right after the KernelTraits struct
    s = sub(s, """  static constexpr DTypeQKAccum MaskFillValue = -math::inf;
};""",
        """  static constexpr DTypeQKAccum MaskFillValue = -math::inf;
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
}""",
        "warp-split helpers")

    # ---- 2. init_states_ ----------------------------------------------------------------
    s = sub(s, """  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = 0.f;""",
        """  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = 0.f;""",
        "init_states_")

    # ---- 3. compute_qk_: q rows ---------------------------------------------------------
    s = sub(s, """    uint32_t q_smem_offset_r = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
        warp_idx_in_wg * 16 + lane_idx % 16, mma_d * 2 + lane_idx / 16);""",
        """    uint32_t q_smem_offset_r = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
        mla_q_row_base<KTraits>() + lane_idx % 16, mma_d * 2 + lane_idx / 16);""",
        "compute_qk_ q rows")

    # ---- 4. logits_mask_: q index -------------------------------------------------------
    s = sub(s, """    q[j] = (qo_packed_idx_base + warp_idx_in_wg * 16 + lane_idx / 4 + 8 * j) / num_heads;""",
        """    q[j] = (qo_packed_idx_base + mla_q_row_base<KTraits>() + lane_idx / 4 + 8 * j) /
           num_heads;""",
        "logits_mask_ q index")

    # ---- 5. update_mdo_states_ ----------------------------------------------------------
    s = sub(s, """      if (lane_idx % 4 == 0) {
        smem_storage->m_wg[warpgroup_idx][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = m[j];
      }""",
        """      if (lane_idx % 4 == 0 && mla_is_q_warp<KTraits>()) {
        smem_storage->m_wg[warpgroup_idx][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4] =
            m[j];
      }""",
        "m_wg write")
    s = sub(s, """      m[j] = max(smem_storage->m_wg[0][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4],
                 smem_storage->m_wg[1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4]);""",
        """      m[j] = max(smem_storage->m_wg[0][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4],
                 smem_storage->m_wg[1][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4]);""",
        "m_wg read")
    # both branches rescale o_frag over the warp's own D slice
    s = sub(s, """      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;""",
        """      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;""",
        "o_frag rescale", count=2)

    # ---- 6. normalize_d_ ---------------------------------------------------------------
    s = sub(s, """      if (lane_idx % 4 == 0) {
        smem_storage->d_wg[warpgroup_idx][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = d[j];
      }""",
        """      if (lane_idx % 4 == 0 && mla_is_q_warp<KTraits>()) {
        smem_storage->d_wg[warpgroup_idx][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4] =
            d[j];
      }""",
        "d_wg write")
    s = sub(s, """      d[j] = smem_storage->d_wg[0][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] +
             smem_storage->d_wg[1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4];""",
        """      d[j] = smem_storage->d_wg[0][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4] +
             smem_storage->d_wg[1][mla_q_row_base<KTraits>() + j * 8 + lane_idx / 4];""",
        "d_wg read")
    s = sub(s, """  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = o_frag[mma_d][reg_id] * d_rcp[(reg_id % 4) / 2];""",
        """  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV_PER_WARP; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = o_frag[mma_d][reg_id] * d_rcp[(reg_id % 4) / 2];""",
        "normalize_d_ o_frag")

    # ---- 7. main kernel: o_frag declaration -------------------------------------------
    s = sub(s, """  alignas(16) float o_frag[NUM_MMA_D_CKV / 2][8];""",
        """  alignas(16) float o_frag[KTraits::NUM_MMA_D_CKV_PER_WARP][8];""",
        "o_frag decl")
    return s


def patch_mla_pv(s: str) -> str:
    """compute_mla_pv: shard CKV across all warps, and address ckv/p explicitly.

    The stock loops carry the smem offset with advance_offset_by_column/row and then undo the
    net column advance with a literal "- NUM_MMA_D_CKV". That correction is only right when the
    inner loop count is a multiple of 4 (the XOR swizzle bits cancel every 4 steps). Stock runs
    NUM_MMA_D_CKV/2 = 16 iterations at HEAD_DIM_CKV 512, so it holds there; our
    NUM_MMA_D_CKV_PER_WARP is 6 at 768, so it does not. Recompute from get_permuted_offset
    instead -- one extra imad next to an ldmatrix + mma.
    """
    # --- QK_SHARD branch -----------------------------------------------------------------
    s = sub(s, """  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
      lane_idx % 16, warpgroup_idx * NUM_MMA_D_CKV + lane_idx / 16);""",
        """  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  constexpr uint32_t NUM_MMA_D_CKV_PER_WARP = KTraits::NUM_MMA_D_CKV_PER_WARP;
  const uint32_t d_col_base = mla_d_col_base<KTraits>();
  const uint32_t q_row_base = mla_q_row_base<KTraits>();""",
        "pv ckv offset base")

    s = sub(s, """#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx % 16,
          warpgroup_idx * NUM_MMA_KV + mma_kv * 2 + lane_idx / 16);
      p_smem.stmatrix_m8n8x4(p_smem_offset_w, (uint32_t*)p_f16[mma_kv]);
#else
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx / 4, warpgroup_idx * NUM_MMA_KV + mma_kv * 2);""",
        """#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          q_row_base + lane_idx % 16, warpgroup_idx * NUM_MMA_KV + mma_kv * 2 + lane_idx / 16);
      // Warps that share q rows hold identical p; only the q-owning ones publish it.
      if (mla_is_q_warp<KTraits>()) {
        p_smem.stmatrix_m8n8x4(p_smem_offset_w, (uint32_t*)p_f16[mma_kv]);
      }
#else
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          q_row_base + lane_idx / 4, warpgroup_idx * NUM_MMA_KV + mma_kv * 2);
      if (mla_is_q_warp<KTraits>())""",
        "p_smem write")

    # the non-stmatrix fallback body is 4 statements; wrap them in the guard opened above
    s = sub(s, """      ((uint32_t*)(p_smem.base + p_smem_offset_w))[lane_idx % 4] = *(uint32_t*)&p_f16[mma_kv][0];
      ((uint32_t*)(p_smem.base + p_smem_offset_w + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
          *(uint32_t*)&p_f16[mma_kv][2];
      ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1)))[lane_idx % 4] =
          *(uint32_t*)&p_f16[mma_kv][4];
      ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
          *(uint32_t*)&p_f16[mma_kv][6];
#endif""",
        """      {
        ((uint32_t*)(p_smem.base + p_smem_offset_w))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][0];
        ((uint32_t*)(p_smem.base + p_smem_offset_w + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][2];
        ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1)))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][4];
        ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
            *(uint32_t*)&p_f16[mma_kv][6];
      }
#endif""",
        "p_smem write fallback body")

    s = sub(s, """    uint32_t p_smem_offset_r = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
        warp_idx_in_wg * 16 + lane_idx % 16, lane_idx / 16);

    // wait for p_smem to be filled
    __syncthreads();

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t p_frag[4];
      p_smem.ldmatrix_m8n8x4(p_smem_offset_r, p_frag);
      p_smem_offset_r = p_smem.template advance_offset_by_column<2>(p_smem_offset_r, mma_kv);

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
        uint32_t v_frag[4];
        ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag,
                                                                             v_frag);
        ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
      }
      ckv_smem_offset_r =
          ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r) -
          NUM_MMA_D_CKV;
    }""",
        """    // wait for p_smem to be filled
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
    }""",
        "pv QK_SHARD inner loops")

    # --- non-QK_SHARD branch -------------------------------------------------------------
    s = sub(s, """#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
        uint32_t v_frag[4];
        ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
            o_frag[mma_d], (uint32_t*)p_f16[mma_kv], v_frag);
        ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
      }
      ckv_smem_offset_r =
          ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r) -
          NUM_MMA_D_CKV;
    }""",
        """#pragma unroll
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
    }""",
        "pv non-shard inner loops")
    return s


_NEW_WRITE_O = r"""
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
"""


def patch_write_o(s: str) -> str:
    """Replace write_o wholesale: o_frag is now a CKV slice, not a q-row slice."""
    start = s.index("__device__ __forceinline__ void write_o(")
    end = s.index("template <typename KTraits, typename Params>\n__global__", start)
    # keep the trailing blank line structure tidy
    return s[:start] + _NEW_WRITE_O.strip() + "\n\n" + s[end:]


_NEW_LAUNCHER = r"""
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
"""


def patch_launcher(s: str) -> str:
    """Replace DISPATCH_SMEM_CONFIG + BatchMLAPagedAttention with a shape-aware dispatch."""
    start = s.index("#define DISPATCH_SMEM_CONFIG(")
    end = s.index("}  // namespace mla", start)
    return s[:start] + _NEW_LAUNCHER.strip() + "\n\n" + s[end:]


QWEN_MLA_HELPER = r'''
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
'''


def add_helper(s: str) -> str:
    """Define mla_cta_tile_q() at global scope, before the first namespace, in each header."""
    return sub(s, "\nnamespace flashinfer {", "\n" + QWEN_MLA_HELPER + "\nnamespace flashinfer {", "cta_tile_q helper")


def patch_scheduler(s: str) -> str:
    """MLAPlan must tile q with the same cta_tile_q the kernel uses (see QWEN_MLA_CTA_TILE_Q_FOR)."""
    s = sub(s, """  uint32_t num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;
  const int cta_tile_q = 64;""",
        """  uint32_t num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;
  // MUST match QWEN_MLA_CTA_TILE_Q_FOR in attention/mla.cuh -- the kernel derives its q tile from
  // num_heads the same way, and a disagreement silently mistiles the query.
  const int cta_tile_q = static_cast<int>(mla_cta_tile_q(static_cast<uint32_t>(num_heads)));""",
        "MLAPlan cta_tile_q")
    return s


def patch_load_q(s: str) -> str:
    """load_q writes a hardcoded 64 q rows; make it write CTA_TILE_Q rows.

    The row index is 32*mma_q + warpgroup_idx*16 + warp_idx_in_wg*4 + lane_idx/8, which always
    spans 0..63 no matter what CTA_TILE_Q says. Harmless upstream, where CTA_TILE_Q is always
    64. At CTA_TILE_Q=16 it overruns q_smem_nope by 4x: for small head dims the spill lands in
    ckv_smem (which the kv loader overwrites before use, so the answer still came out right),
    but at HEAD_DIM_CKV=1792 it runs past the whole SharedStorage allocation and faults with
    "invalid __shared__ write of size 16 bytes".

    Replaced by a flat cooperative loop over CTA_TILE_Q * (HEAD_DIM/8) 128-bit chunks. Global
    reads stay contiguous (consecutive threads take consecutive columns of a row) and the
    per-thread iteration count is unchanged in the stock CTA_TILE_Q=64 case.
    """
    start = s.index("__device__ __forceinline__ void load_q(")
    end = s.index("\ntemplate <", start) + 1
    new = r"""__device__ __forceinline__ void load_q(
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

"""
    return s[:start] + new + s[end:]


def patch_rms(s: str) -> str:
    """Add MLA's per-(key, head) rms divide, behind the QWEN_MLA_RMS_TAIL compile flag.

    Qwen3.5 applies k_norm to the ASSEMBLED [rope ; nope] key after up-projection, so
        q . k_norm(key_h) = (q_nope . c_n + q_pe . r_n) / rms(n, h)
    The numerator absorbs; the denominator is per (key token, HEAD) while the latent is shared
    across heads, so it cannot be folded into the cache or into q. It has to be an elementwise
    divide on the score.

    No new params field, no binding change, no Python wrapper change: MLA's cache row is
    [latent | rope | rms | pad] CONTIGUOUS, and this kernel already takes ckv and kpe as
    separate pointers with separate strides. Pass ckv = row base and kpe = row base + Lv with
    the same row stride, and rms(n, h) sits at kpe_row(n) + HEAD_DIM_KPE + h.

    Staged through shared memory rather than read per fragment element: a thread needs 8 values
    per mma_kv at (2 q rows x 4 kv), all 2-byte scattered reads. Staging is
    CTA_TILE_KV * CTA_TILE_Q loads per tile over NUM_THREADS (2 each at our tiles) and the
    num_heads values for one token are contiguous.

    The staging load is synchronous while the KV load is cp.async, but it needs NO extra
    barrier: it is issued immediately after each load_kv, so the __syncthreads() that already
    precedes load_kv retires the stage's previous occupant, and the __syncthreads() after
    wait_group publishes the writes before the divide reads them.
    """
    # --- shared storage -------------------------------------------------------------------
    s = sub(s, """template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {""",
        """template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, uint32_t RMS_ELEMS, typename DTypeQ, typename DTypeKV,
          typename DTypeO>
struct SharedStorageQKVO {""",
        "SharedStorage template params")

    s = sub(s, """      alignas(16) DTypeKV
          kpe_p_smem[NUM_STAGES]
                    [CTA_TILE_KV * (HEAD_DIM_KPE > CTA_TILE_Q ? HEAD_DIM_KPE : CTA_TILE_Q)];""",
        """      alignas(16) DTypeKV
          kpe_p_smem[NUM_STAGES]
                    [CTA_TILE_KV * (HEAD_DIM_KPE > CTA_TILE_Q ? HEAD_DIM_KPE : CTA_TILE_Q)];
      // [kv_local][q_row] for the current tile; RMS_ELEMS is 1 when the rms tail is off.
      alignas(16) DTypeKV rms_smem[NUM_STAGES][RMS_ELEMS];""",
        "rms_smem member")

    s = sub(s, """  using SharedStorage = SharedStorageQKVO<NUM_STAGES, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_CKV,
                                          HEAD_DIM_KPE, DTypeQ, DTypeKV, DTypeO>;""",
        """#ifdef QWEN_MLA_RMS_TAIL
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

  using SharedStorage =
      SharedStorageQKVO<NUM_STAGES, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_CKV, HEAD_DIM_KPE,
                        RMS_SMEM_PER_STAGE, DTypeQ, DTypeKV, DTypeO>;""",
        "SharedStorage alias")

    # --- the two new device functions, placed just before compute_mla_qk ------------------
    anchor = "template <typename KTraits>\n__device__ __forceinline__ void compute_mla_qk("
    new_fns = r"""template <typename KTraits>
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

"""
    idx = s.index(anchor)
    s = s[:idx] + new_fns + s[idx:]

    # --- call sites -----------------------------------------------------------------------
    import re

    def add_rms_load(mtext: str) -> str:
        args = [a.strip() for a in mtext[mtext.index("(") + 1:-2].split(",")]
        if len(args) != 12:
            raise SystemExit(f"PATCH FAILED [load_kv args]: got {len(args)}: {args}")
        it, stage = args[9], args[11]
        return (mtext + "\n    load_rms_<KTraits>(&smem_storage, ckv, kv_indices, ckv_stride_n, "
                f"ckv_stride_page,\n                       packed_kv_bound, {it}, block_size,\n"
                f"                       {stage}, qo_packed_idx_base, num_heads);")

    s, n = re.subn(r"load_kv<KTraits>\([^;]*?\);", lambda m: add_rms_load(m.group(0)), s)
    if n != 4:
        raise SystemExit(f"PATCH FAILED [load_kv sites]: expected 4, patched {n}")

    s, n = re.subn(
        r"compute_mla_qk<KTraits>\(&smem_storage, ([^,]+), s_frag\);",
        lambda m: m.group(0) + "\n      mla_rms_divide_<KTraits>(&smem_storage, "
                  f"{m.group(1)}, s_frag);",
        s,
    )
    if n != 3:
        raise SystemExit(f"PATCH FAILED [compute_mla_qk sites]: expected 3, patched {n}")
    return s


def patch_min_ctas(s: str) -> str:
    """Let the launch bounds ask for N CTAs/SM, so the register cap can be traded for occupancy.

    ncu on the shipped kernel: zero spilling, 210 registers/thread, ~145 KB of shared memory --
    which pins it to 1 CTA/SM, 8 warps, 12.5% occupancy, with 82.6% of cycles having no eligible
    warp and DRAM at 17%. Base FlashAttention reaches 4.84 TB/s on this box against our 2.15.

    __launch_bounds__(NUM_THREADS, 2) tells ptxas to fit 128 registers/thread instead of 210.
    Whether that is a win is a genuine question: it doubles warps/SM only if the compiler can
    get there without spilling more than the extra occupancy buys. Paired with NUM_STAGES=1 the
    shared memory also drops to ~87 KB, which is under the 116 KB two CTAs need.

    -DQWEN_MLA_MIN_CTAS=N; 1 is the default and leaves ptxas unconstrained.
    """
    return sub(s, "__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPagedAttentionKernel(",
        """#ifndef QWEN_MLA_MIN_CTAS
#define QWEN_MLA_MIN_CTAS 1
#endif
__global__ __launch_bounds__(KTraits::NUM_THREADS, QWEN_MLA_MIN_CTAS)
void BatchMLAPagedAttentionKernel(""",
        "launch bounds min CTAs")


def patch_qk_dshard(s: str) -> str:
    """Stop the 4 warps of a warpgroup recomputing the same q*k.

    MEASURED: DRAM reads exactly the must-read 3.76 GB, but shared memory is read 333x more
    than it is written -- 66.06M ldmatrix instructions, ~33.8 GB, a 9x amplification over DRAM.
    L1/TEX is the busiest unit at 57% while DRAM sits at 16%, which is why neither more warps
    (4 warpgroups) nor more CTAs (launch bounds) moved the needle: the bottleneck is the shared
    path, not occupancy or bandwidth.

    Counting ldmatrix per warp per KV tile at rank 768: q*k is 112 of 126. And with Q_WARPS=1
    all four warps in a warpgroup hold the SAME q rows and the SAME kv half, so they issue those
    112 identically -- 4x redundant.

    Fix: split q*k's D dimension across those warps (each does NUM_MMA_D_QK / QD_SHARDS steps)
    and sum the partial scores through shared memory. Per warp that is 112 -> 28 + a reduction,
    i.e. ~3x less shared traffic. The reduction is 8 floats per thread per KV tile against the
    84 ldmatrix it removes.

    QD_SHARDS = WARPS_PER_WG / Q_WARPS, so CTA_TILE_Q=64 gives 1 and reproduces stock exactly.
    """
    # ---- traits ---------------------------------------------------------------------------
    s = sub(s, """  static constexpr uint32_t D_SHARDS = 2 * (WARPS_PER_WG / Q_WARPS);""",
        """  static constexpr uint32_t D_SHARDS = 2 * (WARPS_PER_WG / Q_WARPS);
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
                "q*k D split must divide both the latent and the rope tile counts");""",
        "QD_SHARDS trait")

    # scratch for the cross-warp score reduction, in the same union arm as ckv_smem
    s = sub(s, """      // [kv_local][q_row] for the current tile; RMS_ELEMS is 1 when the rms tail is off.
      alignas(16) DTypeKV rms_smem[NUM_STAGES][RMS_ELEMS];""",
        """      // [kv_local][q_row] for the current tile; RMS_ELEMS is 1 when the rms tail is off.
      alignas(16) DTypeKV rms_smem[NUM_STAGES][RMS_ELEMS];
      // Partial q*k scores, [warpgroup][warp][lane][reg]. QK_RED_ELEMS is 1 when the D split
      // is off, so this costs nothing in the stock configuration.
      alignas(16) float qk_red[2][4][32][QK_RED_ELEMS];""",
        "qk_red scratch")
    s = sub(s, """template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, uint32_t RMS_ELEMS, typename DTypeQ, typename DTypeKV,
          typename DTypeO>
struct SharedStorageQKVO {""",
        """template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, uint32_t RMS_ELEMS, uint32_t QK_RED_ELEMS, typename DTypeQ,
          typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {""",
        "SharedStorage QK_RED param")
    s = sub(s, """  using SharedStorage =
      SharedStorageQKVO<NUM_STAGES, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_CKV, HEAD_DIM_KPE,
                        RMS_SMEM_PER_STAGE, DTypeQ, DTypeKV, DTypeO>;""",
        """  static constexpr uint32_t S_FRAG_REGS = (QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV) * 8;
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
                        RMS_SMEM_PER_STAGE, QK_RED_ELEMS, DTypeQ, DTypeKV, DTypeO>;""",
        "SharedStorage alias with QK_RED")

    # ---- compute_qk_: each warp walks only its slice of the reduction dimension -------------
    s = sub(s, """  alignas(16) uint32_t q_frag[4], k_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d) {""",
        """  alignas(16) uint32_t q_frag[4], k_frag[4];
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
    const uint32_t mma_d = d_first + d_i;""",
        "compute_qk_ D slice")
    s = sub(s, """        if (init && mma_d == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    } else {""",
        """        if (init && d_i == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    } else {""",
        "compute_qk_ init sharded (QK_SHARD branch)")
    s = sub(s, """        if (init && mma_d == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    }
  }
}""",
        """        if (init && d_i == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    }
  }
}""",
        "compute_qk_ init sharded (non-shard branch)")

    # ---- the reduction, run at the end of compute_mla_qk -----------------------------------
    s = sub(s, """  compute_qk_</*init=*/false, KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV>(q_smem_nope, ckv_smem, s_frag);
}""",
        """  compute_qk_</*init=*/false, KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV>(q_smem_nope, ckv_smem, s_frag);
  mla_qk_reduce_<KTraits>(smem_storage, s_frag);
}""",
        "call the reduction")

    anchor = "template <typename KTraits>\n__device__ __forceinline__ void compute_mla_qk("
    reduce_fn = r"""// Sum the per-warp partial q*k across the warps that split the reduction dimension. Every
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

"""
    idx = s.index(anchor)
    s = s[:idx] + reduce_fn + s[idx:]
    return s


def patch_hoist_q(s: str) -> str:
    """Load q's MMA fragments once per work item instead of once per KV tile.

    q is loop-invariant but compute_qk_ re-issues its ldmatrix for every KV tile. After the D
    split the shared-memory model is exact -- 42 ldmatrix per warp per tile, and the measured
    total (42 x 8 warps x 65536 tile-instances = 22,020,096) matches ncu to the instruction --
    and q is 14 of those 42, i.e. a third of what is left. L1/TEX is still the busiest unit at
    49.7% against DRAM's 26.2%, so removing a third of the shared traffic is the next move.

    Cost is D_STEP*4 registers per warp: 48 for the latent plus 8 for the rope at rank 768.
    """
    s = sub(s, """template <bool init, typename KTraits, uint32_t NUM_MMA_D_QK, uint32_t UPCAST_STRIDE_Q,
          uint32_t UPCAST_STRIDE_K, SwizzleMode SWIZZLE_MODE_Q, SwizzleMode SWIZZLE_MODE_KV>
__device__ __forceinline__ void compute_qk_(smem_t<SWIZZLE_MODE_Q> q_smem,
                                            smem_t<SWIZZLE_MODE_KV> k_smem,
                                            typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  alignas(16) uint32_t q_frag[4], k_frag[4];""",
        """// Preload this warp's slice of q into MMA fragments. Called once per work item; the KV loop
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
  alignas(16) uint32_t k_frag[4], q_frag_smem[4];""",
        "compute_qk_ takes preloaded q frags")

    s = sub(s, """    const uint32_t mma_d = d_first + d_i;
    uint32_t q_smem_offset_r = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
        mla_q_row_base<KTraits>() + lane_idx % 16, mma_d * 2 + lane_idx / 16);
    q_smem.ldmatrix_m8n8x4(q_smem_offset_r, q_frag);""",
        """    const uint32_t mma_d = d_first + d_i;
    uint32_t* q_frag;
    if constexpr (KTraits::HOIST_Q) {
      q_frag = q_frags[d_i];
    } else {
      q_smem.ldmatrix_m8n8x4(q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
                                 mla_q_row_base<KTraits>() + lane_idx % 16,
                                 mma_d * 2 + lane_idx / 16),
                             q_frag_smem);
      q_frag = q_frag_smem;
    }""",
        "use the preloaded fragment")

    # compute_mla_qk forwards the fragments
    s = sub(s, """__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[8]) {""",
        """__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               uint32_t (*q_nope_frags)[4],
                                               uint32_t (*q_pe_frags)[4],
                                               typename KTraits::DTypeQKAccum (*s_frag)[8]) {""",
        "compute_mla_qk signature")
    s = sub(s, """  compute_qk_</*init=*/true, KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
              KTraits::UPCAST_STRIDE_KPE>(q_smem_pe, kpe_smem, s_frag);
  compute_qk_</*init=*/false, KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV>(q_smem_nope, ckv_smem, s_frag);""",
        """  compute_qk_</*init=*/true, KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
              KTraits::UPCAST_STRIDE_KPE, KTraits::SWIZZLE_MODE_Q_PE>(
      q_smem_pe, q_pe_frags, kpe_smem, s_frag);
  compute_qk_</*init=*/false, KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV, KTraits::SWIZZLE_MODE_Q_NOPE>(
      q_smem_nope, q_nope_frags, ckv_smem, s_frag);""",
        "compute_mla_qk forwards frags")

    # declare + fill the fragments in the kernel, and pass them at the three call sites
    s = sub(s, """  float s_frag[KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV][8];""",
        """  float s_frag[KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV][8];
  alignas(16) uint32_t q_nope_frags[KTraits::Q_NOPE_FRAGS][4];
  alignas(16) uint32_t q_pe_frags[KTraits::Q_PE_FRAGS][4];""",
        "q fragment storage")

    s = sub(s, """    // loop with mask
#pragma unroll 1""",
        """    // q is in shared memory now (its cp.async group was committed before the KV ones, so the
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
#pragma unroll 1""",
        "hoisted q fragment load")

    import re
    s, n = re.subn(r"compute_mla_qk<KTraits>\(&smem_storage, ([^,]+), s_frag\);",
                   lambda m: f"compute_mla_qk<KTraits>(&smem_storage, {m.group(1)}, q_nope_frags,\n"
                             f"                              q_pe_frags, s_frag);", s)
    if n != 3:
        raise SystemExit(f"PATCH FAILED [compute_mla_qk calls]: expected 3, patched {n}")
    return s


if __name__ == "__main__":
    inc = fi_include()
    src_mla = inc / "flashinfer" / "attention" / "mla.cuh"
    shutil.copy(src_mla, HERE / "qwen_mla.cuh.orig")
    s = src_mla.read_text()
    s = patch_mla(s)
    s = patch_load_q(s)
    s = patch_mla_pv(s)
    s = patch_write_o(s)
    s = patch_launcher(s)
    s = patch_rms(s)
    s = patch_min_ctas(s)
    s = patch_qk_dshard(s)
    s = patch_hoist_q(s)
    s = add_helper(s)
    (HERE / "qwen_mla.cuh").write_text(s)

    src_sched = inc / "flashinfer" / "attention" / "scheduler.cuh"
    shutil.copy(src_sched, HERE / "qwen_scheduler.cuh.orig")
    (HERE / "qwen_scheduler.cuh").write_text(add_helper(patch_scheduler(src_sched.read_text())))
    # The runtime overlay (fi_mla.ensure_overlay) refuses to pair these headers with any other
    # flashinfer version, since the rest of the include tree is the installed one.
    ver = subprocess.check_output([PY, "-c", "import flashinfer; print(flashinfer.__version__)"], text=True).strip()
    (HERE / "FLASHINFER_VERSION").write_text(ver + "\n")
    print(f"wrote {HERE / 'qwen_scheduler.cuh'} (from {src_sched})")
    print(f"wrote {HERE / 'qwen_mla.cuh'} (from {src_mla})")
