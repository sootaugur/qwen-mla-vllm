# FORK of vllm/v1/attention/ops/triton_decode_attention.py
#
# ONE CHANGE: the grouped (MLA) kernel divides the attention scores by a per-(key-token, head)
# scalar before softmax.
#
# WHY. Qwen3.5 applies k_norm to the ASSEMBLED [rope ; nope] key AFTER up-projection, so
#     q . k_norm(key_h) = ([ (q_nope*(1+w)) @ W_UK_h ] . c + rope_term) / rms(key_h)
# The numerator absorbs fine -- (1+w) and W_UK fold into the query, exactly as DeepSeek MLA
# does. The denominator does not: rms is per (key-token, HEAD) while the cached latent is
# shared across all 24 heads, so it cannot be folded into the cache. DeepSeek avoids this by
# normalising the LATENT (kv_a_layernorm), which keeps K linear in c.
#
# The divisor has exactly the shape of `qk` ([BLOCK_H, BLOCK_N]), so applying it here is one
# load and one divide. The rms values live INSIDE K_Buffer, appended after [latent | rope], so
# vLLM's paging, eviction and prefix caching move them with the latent automatically rather
# than needing a second cache kept in sync.
#
# RMS_OFFSET < 0 disables the divide, making this kernel bit-identical to upstream.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# Changes:
# - Add support for page size >= 1.

# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size >= 1.
"""

import logging

import torch
from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

is_hip_ = current_platform.is_rocm()

logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if version.parse(triton.__version__) < version.parse("3.2.0"):
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored."
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


def _page_stride(buf, page_size):
    # Stride between pages. 4D buffers have a page dim; 3D buffers pack pages
    # along the token dim, so split it out first. Read the real stride (a
    # cross-layer view has gaps), don't assume PAGE_SIZE * token stride.
    if buf.ndim == 3:
        buf = buf.unflatten(-3, (-1, page_size))
    return buf.stride(-4)


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vpbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            ).to(tl.int64)  # page_number * page stride overflows int32
            kv_in_page = offs_n % PAGE_SIZE
            offs_buf_k = (
                (kv_page_number * stride_buf_kpbs + kv_in_page * stride_buf_kbs)[
                    :, None
                ]
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                (kv_page_number * stride_buf_vpbs + kv_in_page * stride_buf_vbs)[
                    :, None
                ]
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )
            if v.dtype.is_fp8():
                v = (v.to(tl.float32) * vs).to(q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
):
    BLOCK = 64 if not is_hip_ else 8

    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4
    if kv_group_num != 1:
        num_warps = 1 if is_hip_ else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        _page_stride(k_buffer, page_size),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        _page_stride(v_buffer, page_size),
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vpbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    RMS_OFFSET: tl.constexpr,
    PAGE_ALIGNED: tl.constexpr,
    RMS_BF16: tl.constexpr,
    RMS_LOG8: tl.constexpr,
    MLA_RELOAD_V: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    IS_MLA: tl.constexpr = False,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    # nope region is the LATENT only. Upstream uses Lk here, which is safe only when
    # BLOCK_DMODEL == Lv (true for DeepSeek's power-of-two 512, false for our 768) -- otherwise
    # the rope dims leak into the nope dot product.
    mask_d = offs_d < Lv
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=(mask_h[:, None]) & (mask_d[None, :]),
        other=0.0,
        cache_modifier=".ca",
    )

    if BLOCK_DPE > 0:
        # rope starts at Lv. vLLM builds the absorbed query as [latent | rope] contiguously,
        # so BLOCK_DMODEL (a power-of-two round-up of Lv) is only the right offset when Lv is
        # already a power of two.
        offs_dpe = Lv + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe,
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
            cache_modifier=".ca",
        )

    # Round the per-split length up to a whole number of BLOCK_N tiles. Every tile then starts
    # at a multiple of BLOCK_N, which (with PAGE_SIZE % BLOCK_N == 0) means no tile straddles a
    # page and the page number is a SCALAR per tile -- see PAGE_ALIGNED below. Splits stay
    # correct because the end is still clamped to the sequence length and an empty split is
    # skipped; _fwd_kernel_stage2 recomputes these same three lines and so agrees on the
    # boundaries.
    if PAGE_ALIGNED:
        kv_len_per_split = (
            tl.cdiv(tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS), BLOCK_N) * BLOCK_N
        )
    else:
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if PAGE_ALIGNED:
                # The whole tile lives in one page, so this is ONE scalar load instead of a
                # BLOCK_N-wide gather, and kv_off_k becomes `uniform_base + arange * stride`.
                # That matters far more than the saved instructions: with a vector gather the
                # K/V addresses are data-dependent per element, Triton cannot prove them affine,
                # and the software pipeliner declines to issue async copies -- which is why
                # num_stages measured *exactly* no effect (4.41 ms at 1, 2 and 3 stages).
                kv_page_number = tl.load(
                    Req_to_tokens
                    + stride_req_to_tokens_b * cur_batch_req_idx
                    + start_n // PAGE_SIZE,
                    cache_modifier=".ca",
                ).to(tl.int64)  # page_number * page stride overflows int32
                kv_off_k = kv_page_number * stride_buf_kpbs + (
                    (start_n % PAGE_SIZE) + tl.arange(0, BLOCK_N)
                ) * stride_buf_kbs
            else:
                kv_page_number = tl.load(
                    Req_to_tokens
                    + stride_req_to_tokens_b * cur_batch_req_idx
                    + offs_n // PAGE_SIZE,
                    mask=offs_n < split_kv_end,
                    other=0,
                    cache_modifier=".ca",
                ).to(tl.int64)  # page_number * page stride overflows int32
                kv_off_k = (
                    kv_page_number * stride_buf_kpbs
                    + (offs_n % PAGE_SIZE) * stride_buf_kbs
                )

            # explicitly facilitate overlapping load/compute
            offs_buf_k = kv_off_k[None, :] + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
                cache_modifier=".cg",
            )

            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_off_k[None, :] + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                    cache_modifier=".cg",
                )
                if kpe.dtype.is_fp8():
                    kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))

            if RMS_OFFSET >= 0:
                # rms(key_h) lives in K_Buffer at [RMS_OFFSET + head] for each key token, so it
                # is fetched with the same paged offsets as k. Shape [BLOCK_H, BLOCK_N] matches
                # qk exactly. Guard against zero for masked-out lanes.
                if RMS_LOG8:
                    # FP8 cache, compact row: rms as a uint8 code of log2(rms) over [-3, 3] (mla_impl)
                    rms_row = (K_Buffer + kv_off_k + cur_kv_head * stride_buf_kh + RMS_OFFSET).to(
                        tl.pointer_type(tl.uint8))
                    code = tl.load(
                        rms_row[None, :] + cur_head[:, None],
                        mask=mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                        other=128,
                    ).to(tl.float32)
                    rms = tl.exp2(-3.0 + code * (6.0 / 255.0))
                elif RMS_BF16:
                    # FP8 cache: the rms tail is kept as raw bf16 (2 bytes per head) -- e4m3's 3-bit
                    # mantissa would put up to 6% error on every logit. Reinterpret the row's tail.
                    rms_row = (K_Buffer + kv_off_k + cur_kv_head * stride_buf_kh + RMS_OFFSET).to(
                        tl.pointer_type(tl.bfloat16))
                    rms = tl.load(
                        rms_row[None, :] + cur_head[:, None],
                        mask=mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                        other=1.0,
                    ).to(tl.float32)
                else:
                    offs_buf_rms = (
                        kv_off_k[None, :]
                        + cur_kv_head * stride_buf_kh
                        + (RMS_OFFSET + cur_head)[:, None]
                    )
                    rms_raw = tl.load(
                        K_Buffer + offs_buf_rms,
                        mask=mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                        other=1.0,
                    )
                    if rms_raw.dtype.is_fp8():          # compact FP8 row: rms is e4m3 x k_scale
                        rms = rms_raw.to(tl.float32) * ks
                    else:
                        rms = rms_raw.to(tl.float32)
                qk = qk / tl.where(rms > 0, rms, 1.0)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            if not IS_MLA:
                if PAGE_ALIGNED:
                    kv_off_v = kv_page_number * stride_buf_vpbs + (
                        (start_n % PAGE_SIZE) + tl.arange(0, BLOCK_N)
                    ) * stride_buf_vbs
                else:
                    kv_off_v = (
                        kv_page_number * stride_buf_vpbs
                        + (offs_n % PAGE_SIZE) * stride_buf_vbs
                    )
                offs_buf_v = kv_off_v[:, None] + base_offs_v
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                if v.dtype.is_fp8():
                    v = (v.to(tl.float32) * vs).to(q.dtype)
            elif MLA_RELOAD_V:
                # Upstream reuses the K tile via tl.trans, reasoning that a second load of the
                # same c_kv is unnecessary. But tl.trans of a [BLOCK_DMODEL, BLOCK_N] tile into
                # the [BLOCK_N, BLOCK_DV] layout the second dot wants is a real shuffle, and it
                # keeps both orientations live at once. Re-loading hits L1/L2 (the lines were
                # just read) and costs no extra DRAM. Measured, not assumed.
                offs_buf_v2 = kv_off_k[:, None] + cur_kv_head * stride_buf_vh + offs_dv[None, :]
                v = tl.load(
                    K_Buffer + offs_buf_v2,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                    cache_modifier=".cg",
                )
            else:
                # MLA uses a single c_kv.
                # loading the same c_kv to interpret it as v is not necessary.
                # transpose the existing c_kv (aka k) for the dot product.
                v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# -------------------------------------------------------------------------------------------------
# TWO-PASS MLA DECODE for very wide latents on small-shared-memory devices (sm_120: 99 KB/SM).
#
# The single-pass kernel holds a [BLOCK_H, Lv] q tile and a [Lv, BLOCK_N] K tile in shared memory and
# a [BLOCK_H, Lv] fp32 accumulator in registers. At Lv 1792 that only fits with BLOCK_H 4, so 24 query
# heads re-read the whole latent cache 6 times (measured 8.1 ms/layer at batch 38, 16k context).
# Splitting the work lets every key be read about twice instead:
#   pass A  scores S[b, h, n] = (q . k / rms) * sm_scale for ALL heads per program, looping the
#           latent in DC-wide chunks so shared memory stays small; each K row is read once.
#   pass B  per (batch, kv split, DVC-wide output slice): online softmax over S and P . V for all
#           heads; each V column is read once.
# Both write the single-pass kernel's split-K partial format, so _fwd_kernel_stage2 merges unchanged.
# -------------------------------------------------------------------------------------------------
@triton.jit
def _mla2p_scores_kernel(
    Q, K_Buffer, S, sm_scale, Req_to_tokens, B_Seqlen,
    stride_req_b, stride_qb, stride_qh, stride_kp, stride_kt, stride_sb, stride_sh, stride_ss,
    chunk_start, k_scale,
    H: tl.constexpr, Lv: tl.constexpr, Lk: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    DC: tl.constexpr, BLOCK_DPE: tl.constexpr, NUM_KV_SPLITS: tl.constexpr, PAGE_SIZE: tl.constexpr,
    RMS_OFFSET: tl.constexpr, logit_cap: tl.constexpr, CHUNK: tl.constexpr, RMS_BF16: tl.constexpr,
    RMS_LOG8: tl.constexpr,
):
    ks = tl.load(k_scale)
    b = tl.program_id(0)
    split = tl.program_id(1)
    heads = tl.arange(0, BLOCK_H)
    mh = heads < H
    seq = tl.load(B_Seqlen + b)
    per = tl.cdiv(tl.cdiv(seq, NUM_KV_SPLITS), BLOCK_N) * BLOCK_N     # same bounds as stage 1/2
    split_start = per * split
    split_end = tl.minimum(split_start + per, seq)
    start = split_start + chunk_start                                    # this round's chunk
    end = tl.minimum(start + CHUNK, split_end)
    for start_n in tl.range(start, end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mn = offs_n < end
        page = tl.load(Req_to_tokens + stride_req_b * b + start_n // PAGE_SIZE).to(tl.int64)
        kv_off = page * stride_kp + ((start_n % PAGE_SIZE) + tl.arange(0, BLOCK_N)) * stride_kt
        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        for d0 in tl.range(0, Lv, DC):
            offs_d = d0 + tl.arange(0, DC)
            md = offs_d < Lv
            q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
                        mask=mh[:, None] & md[None, :], other=0.0, cache_modifier=".ca")
            k = tl.load(K_Buffer + kv_off[None, :] + offs_d[:, None],
                        mask=mn[None, :] & md[:, None], other=0.0, cache_modifier=".cg")
            if k.dtype.is_fp8():
                k = k.to(tl.float32) * ks
            qk += tl.dot(q, k.to(q.dtype))
        if BLOCK_DPE > 0:
            offs_p = Lv + tl.arange(0, BLOCK_DPE)
            mp = offs_p < Lk
            qpe = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + offs_p[None, :],
                          mask=mh[:, None] & mp[None, :], other=0.0, cache_modifier=".ca")
            kpe = tl.load(K_Buffer + kv_off[None, :] + offs_p[:, None],
                          mask=mn[None, :] & mp[:, None], other=0.0, cache_modifier=".cg")
            if kpe.dtype.is_fp8():
                kpe = kpe.to(tl.float32) * ks
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
        if RMS_OFFSET >= 0:
            if RMS_LOG8:
                rms_row = (K_Buffer + kv_off + RMS_OFFSET).to(tl.pointer_type(tl.uint8))
                code = tl.load(rms_row[None, :] + heads[:, None],
                               mask=mh[:, None] & mn[None, :], other=128).to(tl.float32)
                rms = tl.exp2(-3.0 + code * (6.0 / 255.0))
            elif RMS_BF16:
                rms_row = (K_Buffer + kv_off + RMS_OFFSET).to(tl.pointer_type(tl.bfloat16))
                rms = tl.load(rms_row[None, :] + heads[:, None],
                              mask=mh[:, None] & mn[None, :], other=1.0).to(tl.float32)
            else:
                rms_raw = tl.load(K_Buffer + kv_off[None, :] + (RMS_OFFSET + heads)[:, None],
                                  mask=mh[:, None] & mn[None, :], other=1.0)
                if rms_raw.dtype.is_fp8():
                    rms = rms_raw.to(tl.float32) * ks
                else:
                    rms = rms_raw.to(tl.float32)
            qk = qk / tl.where(rms > 0, rms, 1.0)
        qk *= sm_scale
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)
        tl.store(S + b * stride_sb + heads[:, None] * stride_sh + split * stride_ss + (offs_n - start)[None, :],
                 qk, mask=mh[:, None] & mn[None, :])


@triton.jit
def _mla2p_pv_kernel(
    S, K_Buffer, Att_Out, Req_to_tokens, B_Seqlen, Lse_In, Lse_Out,
    stride_req_b, stride_kp, stride_kt, stride_sb, stride_sh, stride_ss, stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, chunk_start, k_scale,
    H: tl.constexpr, Lv: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, DVC: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr, PAGE_SIZE: tl.constexpr, CHUNK: tl.constexpr, MERGE: tl.constexpr,
    LSE_TO_OUT: tl.constexpr,
):
    b = tl.program_id(0)
    split = tl.program_id(1)
    dvc = tl.program_id(2)
    heads = tl.arange(0, BLOCK_H)
    mh = heads < H
    offs_dv = dvc * DVC + tl.arange(0, DVC)
    mdv = offs_dv < Lv
    seq = tl.load(B_Seqlen + b)
    per = tl.cdiv(tl.cdiv(seq, NUM_KV_SPLITS), BLOCK_N) * BLOCK_N
    split_start = per * split
    split_end = tl.minimum(split_start + per, seq)
    start = split_start + chunk_start
    end = tl.minimum(start + CHUNK, split_end)
    lse_off = b * stride_lb + heads * stride_lh + split
    if MERGE:
        if end <= start:              # this split has no keys in this round: carry its LSE forward
            if dvc == 0:
                tl.store(Lse_Out + lse_off, tl.load(Lse_In + lse_off, mask=mh), mask=mh)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, DVC], dtype=tl.float32)
    if end > start:
        for start_n in tl.range(start, end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mn = offs_n < end
            sc = tl.load(S + b * stride_sb + heads[:, None] * stride_sh + split * stride_ss
                         + (offs_n - start)[None, :], mask=mh[:, None] & mn[None, :], other=float("-inf"))
            page = tl.load(Req_to_tokens + stride_req_b * b + start_n // PAGE_SIZE).to(tl.int64)
            kv_off = page * stride_kp + ((start_n % PAGE_SIZE) + tl.arange(0, BLOCK_N)) * stride_kt
            v = tl.load(K_Buffer + kv_off[:, None] + offs_dv[None, :],
                        mask=mn[:, None] & mdv[None, :], other=0.0, cache_modifier=".cg")
            if v.dtype.is_fp8():
                v = (v.to(tl.float32) * tl.load(k_scale)).to(tl.bfloat16)
            n_max = tl.maximum(tl.max(sc, 1), e_max)
            re = tl.exp(e_max - n_max)
            pr = tl.exp(sc - n_max[:, None])
            acc = acc * re[:, None] + tl.dot(pr.to(v.dtype), v)
            e_sum = e_sum * re + tl.sum(pr, 1)
            e_max = n_max
        base = b * stride_ob + heads * stride_oh + split * stride_os
        o = acc / e_sum[:, None]
        lse = e_max + tl.log(e_sum)
        if MERGE:                     # fold this chunk into the partial from earlier rounds
            lse_p = tl.load(Lse_In + lse_off, mask=mh, other=float("-inf"))
            o_p = tl.load(Att_Out + base[:, None] + offs_dv[None, :], mask=mh[:, None] & mdv[None, :], other=0.0)
            m = tl.maximum(lse_p, lse)
            wp = tl.exp(lse_p - m)
            wc = tl.exp(lse - m)
            o = (o_p * wp[:, None] + o * wc[:, None]) / (wp + wc)[:, None]
            lse = m + tl.log(wp + wc)
        tl.store(Att_Out + base[:, None] + offs_dv[None, :], o, mask=mh[:, None] & mdv[None, :])
        if dvc == 0:
            if LSE_TO_OUT:
                tl.store(Att_Out + base + Lv, lse, mask=mh)
            else:
                tl.store(Lse_Out + lse_off, lse, mask=mh)


_SMALL_SMEM = {}


def _small_smem(device) -> bool:
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _SMALL_SMEM:
        p = torch.cuda.get_device_properties(i)
        _SMALL_SMEM[i] = getattr(p, "shared_memory_per_multiprocessor", 228 * 1024) < 128 * 1024
    return _SMALL_SMEM[i]


def _decode_mla_two_pass(q, k_buffer, att_out, req_to_token, b_seq_len, num_kv_splits, sm_scale,
                         page_size, rms_offset, logit_cap, Lk, Lv, k_scale=None, rms_bf16=False,
                         rms_log8=False):
    """Wide-latent MLA decode as scores pass + P.V pass (see the comment above the kernels).

    Scores are fp32 (bf16 would distort exp() at |logit| ~ 30) in a scratch of [batch, heads, splits,
    CHUNK]. When a whole split fits the budget (QWEN_MLA_2P_SCRATCH_MB, default 256) there is one round, as
    before; otherwise each split is processed in a static number of rounds of CHUNK keys, and each round
    folds its partial into the running one with the standard LSE merge (LSEs ping-pong between two small
    buffers so no program reads an LSE another is overwriting). The round count comes from the block
    table's shape, not from sequence lengths, so it is fixed per CUDA-graph capture.
    """
    import os as _o
    B, H = q.shape[0], q.shape[1]
    BLOCK_N = 16                       # tl.dot K >= 16, and PAGE_SIZE % 16 == 0 keeps tiles in one page
    assert page_size % BLOCK_N == 0, "two-pass decode needs page_size % 16 == 0"
    BLOCK_H = max(16, triton.next_power_of_2(H))
    # Defaults measured at batch 38 / ctx 16.6k / 24 heads / 32 splits on RTX PRO 6000: 3.92 ms/layer vs
    # 7.63 single-pass (DC 128/DVC 256: 4.26; DC 64/DVC 128: 4.51; warps/stages: flat).
    DC = int(_o.environ.get("QWEN_MLA_2P_DC", 256))          # latent chunk of the scores pass
    DVC = int(_o.environ.get("QWEN_MLA_2P_DVC", 512))        # output slice per P.V program
    WA = int(_o.environ.get("QWEN_MLA_2P_WARPS_A", 4)); WB = int(_o.environ.get("QWEN_MLA_2P_WARPS_B", 4))
    SA = int(_o.environ.get("QWEN_MLA_2P_STAGES_A", 2)); SB = int(_o.environ.get("QWEN_MLA_2P_STAGES_B", 2))
    BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0

    max_tokens = req_to_token.shape[1] * page_size                       # static per capture
    cdiv = lambda a, b: -(-a // b)
    per_max = cdiv(cdiv(max_tokens, num_kv_splits), BLOCK_N) * BLOCK_N   # longest possible split
    budget = int(float(_o.environ.get("QWEN_MLA_2P_SCRATCH_MB", 256)) * (1 << 20))
    fit = budget // (4 * B * H * num_kv_splits)
    CHUNK = per_max if fit >= per_max else max(BLOCK_N, (fit // BLOCK_N) * BLOCK_N)
    rounds = -(-per_max // CHUNK)
    S = torch.empty(B, H, num_kv_splits, CHUNK, dtype=torch.float32, device=q.device)
    if rounds > 1:
        lse_bufs = [torch.empty(B, H, num_kv_splits, dtype=torch.float32, device=q.device) for _ in range(2)]
    else:
        lse_bufs = [att_out, att_out]                                    # unused (LSE_TO_OUT)

    kp, kt = _page_stride(k_buffer, page_size), k_buffer.stride(-3)
    if k_scale is None:
        k_scale = torch.ones((), dtype=torch.float32, device=q.device)
    for r in range(rounds):
        cs = r * CHUNK
        _mla2p_scores_kernel[(B, num_kv_splits)](
            q, k_buffer, S, sm_scale, req_to_token, b_seq_len,
            req_to_token.stride(0), q.stride(0), q.stride(1), kp, kt, S.stride(0), S.stride(1), S.stride(2),
            cs, k_scale,
            H=H, Lv=Lv, Lk=Lk, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, DC=DC, BLOCK_DPE=BLOCK_DPE,
            NUM_KV_SPLITS=num_kv_splits, PAGE_SIZE=page_size, RMS_OFFSET=rms_offset, logit_cap=logit_cap,
            CHUNK=CHUNK, RMS_BF16=rms_bf16, RMS_LOG8=rms_log8, num_warps=WA, num_stages=SA)
        lin, lout = lse_bufs[r % 2], lse_bufs[(r + 1) % 2]
        _mla2p_pv_kernel[(B, num_kv_splits, triton.cdiv(Lv, DVC))](
            S, k_buffer, att_out, req_to_token, b_seq_len, lin, lout,
            req_to_token.stride(0), kp, kt, S.stride(0), S.stride(1), S.stride(2),
            att_out.stride(0), att_out.stride(1), att_out.stride(2),
            lin.stride(0) if rounds > 1 else 0, lin.stride(1) if rounds > 1 else 0, cs, k_scale,
            H=H, Lv=Lv, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, DVC=DVC, NUM_KV_SPLITS=num_kv_splits,
            PAGE_SIZE=page_size, CHUNK=CHUNK, MERGE=r > 0, LSE_TO_OUT=rounds == 1,
            num_warps=WB, num_stages=SB)
    if rounds > 1:
        att_out[..., Lv].copy_(lse_bufs[rounds % 2])                     # final LSE into vLLM's format
    return BLOCK_N                     # stage 2 recomputes the tile-aligned split bounds from this


_FIT_CACHE = {}   # (shape, device) -> (BLOCK_H, BLOCK_N, warps, stages) that fits shared memory


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
    is_mla=False,
    rms_offset=-1,
    rms_bf16=False,
    rms_log8=False,
):
    # with is_mla there is only a single c_kv in smem.
    # could increase BLOCK or num_stages.
    # Lk is the ATTENTION width, not the buffer width. The rms scalars live in a tail appended
    # to K_Buffer, and deriving Lk from k_buffer.shape[-1] silently folded them into the q.k dot
    # product -- rel error 8.9e-2 against a control of 1.9e-3, i.e. degraded but plausible.
    Lk = k_buffer.shape[-1] if rms_offset < 0 else rms_offset
    Lv = v_buffer.shape[-1]

    # Wide MLA latents (1792) cannot fit the single-pass tiles in 99 KB of shared memory without
    # BLOCK_H 4, which re-reads the latent once per 4 heads; use the two-pass kernels instead.
    # Large-smem devices (H100) keep the single-pass path. QWEN_MLA_TWO_PASS=0 disables it.
    import os as _os2
    if (is_mla and not is_hip_ and Lv >= 1536 and _small_smem(q.device)
            and _os2.environ.get("QWEN_MLA_TWO_PASS", "1") != "0" and page_size % 16 == 0):
        return _decode_mla_two_pass(q, k_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits,
                                    sm_scale, page_size, rms_offset, logit_cap, Lk, Lv,
                                    k_scale=k_scale, rms_bf16=rms_bf16, rms_log8=rms_log8)

    # Align tile dimensions with latent rank for MLA to avoid shape mismatch.
    if is_mla:
        if not is_hip_ and Lk == 576:
            BLOCK_DMODEL = 512
            BLOCK_DPE = 64
        elif not is_hip_ and Lk == 288:
            BLOCK_DMODEL = 256
            BLOCK_DPE = 32
        else:
            BLOCK_DMODEL = triton.next_power_of_2(Lv)
            BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    BLOCK = 32
    if is_hip_:
        BLOCK = 16

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    # TUNING KNOBS (env, default = upstream values). The fp32 accumulator [BLOCK_H, BLOCK_DV]
    # lives in registers, and BLOCK_DV is next_pow2(kv_lora_rank): 512 for DeepSeek, but 1024
    # for our rank-768 layers and 2048 for rank 1792 -- 128 and 256 registers per thread at 4
    # warps, against a hardware maximum of 255. Spilled registers become local-memory traffic,
    # which is why this kernel can measure single-digit-percent HBM efficiency on our shapes.
    # Lowering BLOCK_H shrinks the accumulator but re-reads the shared latent once per head
    # block, so the optimum is measured (bench_decode.py), not assumed.
    # MEASURED defaults (bench_decode.py, batch 64 / ctx 32k / 12 heads = TP=2 local):
    #     rank  768 (BLOCK_DV 1024):  BN32/w4 7.00 ms | BN32/w8 5.19 | BN16/w4 4.42  <-
    #     rank 1792 (BLOCK_DV 2048):  BN32/w4 53.92    | BN32/w8 OOM-shmem | BN16/w8 8.36  <-
    # Upstream's BLOCK_N=32 is tuned for DeepSeek's BLOCK_DV=512. The K tile is
    # (BLOCK_DMODEL + BLOCK_DPE) * BLOCK_N * 2 bytes, so at our BLOCK_DMODEL of 1024/2048 it is
    # 80/144 KB against ~99 KB of shared memory -- BLOCK_N=32 either spills or fails outright,
    # and halving BLOCK_N is worth more than any BLOCK_H change. (An earlier reading of this
    # kernel concluded BLOCK_H=4 helped at 2048; that was an artefact of only testing BN=32.)
    # Wider accumulators want more threads to spread the registers over, hence warps by BLOCK_DV.
    #
    # BLOCK_H stays 16 and BLOCK_N stays 16 for MLA. Two tempting variants MEASURED WORSE or
    # did not fit, so they are recorded here rather than re-derived:
    #   BLOCK_N=32 is upstream's default, tuned for DeepSeek's BLOCK_DV=512; at our 1024/2048 the
    #     K tile (double-buffered) exceeds this device's 227 KB of shared memory outright.
    #   BLOCK_H=32 would let 24 query heads (TP=1) share one block instead of re-reading the
    #     whole cache twice -- but it only beats BLOCK_H=16 when paired with BLOCK_N=32
    #     (3.83 ms vs 4.48), which is the combination that does not fit. At BLOCK_N=16 it is
    #     slower (4.99 vs 4.48), so growing it buys nothing and can fail the launch at rank 1792
    #     where the q tile scales with BLOCK_H * BLOCK_DMODEL.
    import os as _os
    if is_mla:
        BLOCK = 16
        _num_warps = 8 if BLOCK_DV >= 2048 else 4
    else:
        _num_warps = 4
    BLOCK_H = int(_os.environ.get("QWEN_MLA_BLOCK_H", BLOCK_H))
    BLOCK = int(_os.environ.get("QWEN_MLA_BLOCK_N", BLOCK))
    _num_warps = int(_os.environ.get("QWEN_MLA_NUM_WARPS", _num_warps))

    # SHARED-MEMORY FIT (added for Blackwell sm_120: 99 KB/block, vs 227 KB on H100 where the
    # defaults above were measured). The K tile is (BLOCK_DMODEL + BLOCK_DPE) * BLOCK_N * 2 B and
    # the q tile BLOCK_H * BLOCK_DMODEL * 2 B, so at rank 1792 the H100 defaults need ~137 KB.
    # Try the measured defaults first, then step down -- num_stages, then BLOCK_N, then BLOCK_H --
    # and remember what fit per (shape, device) so the fallback costs one failed compile, once.
    # Explicit QWEN_MLA_* env overrides are honoured as the only candidate.
    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-triton-kernel-performance-optimization
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        # Avoid shared memory overflow on NVIDIA when BLOCK_DMODEL is large
        # like non-MLA D_QK=576, BLOCK_DMODEL=1024, BLOCK_H=16
        # exceeds 101376 bytes limit
        num_stages = 1
    _env = any(k in _os.environ for k in ("QWEN_MLA_BLOCK_H", "QWEN_MLA_BLOCK_N", "QWEN_MLA_NUM_WARPS", "QWEN_MLA_NUM_STAGES"))
    num_stages = int(_os.environ.get("QWEN_MLA_NUM_STAGES", num_stages))
    _key = (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, kv_group_num, q.device.index, page_size)
    if _key in _FIT_CACHE:
        _cands = [_FIT_CACHE[_key]]
    elif _env:
        _cands = [(BLOCK_H, BLOCK, _num_warps, num_stages)]
    else:
        _cands = [(BLOCK_H, BLOCK, _num_warps, num_stages)]
        # BLOCK_N is the K dimension of tl.dot(p, v) and must stay >= 16, so only the stage count
        # and BLOCK_H (the q tile) can shrink; BLOCK_H 4 re-reads the latent more often but fits.
        for _bh, _bn, _ns in ((BLOCK_H, BLOCK, 1), (max(4, BLOCK_H // 2), BLOCK, 1),
                              (max(4, BLOCK_H // 4), BLOCK, 1)):
            if (_bh, _bn, _num_warps, _ns) not in _cands:
                _cands.append((_bh, _bn, _num_warps, _ns))
    _err = None
    for BLOCK_H, BLOCK, _num_warps, num_stages in _cands:
        # A BLOCK_N tile lies entirely inside one page iff the page divides evenly into tiles; the
        # kernel then rounds each split up to a whole tile so tile starts are BLOCK_N-aligned too.
        # Both hold for our 400-token page at BLOCK_N 16, and the scalar page lookup that unlocks
        # is what lets Triton pipeline the K loads. QWEN_MLA_PAGE_ALIGNED=0 restores the gather.
        PAGE_ALIGNED = (page_size % BLOCK == 0) and (
            _os.environ.get("QWEN_MLA_PAGE_ALIGNED", "1") == "1"
        )
        NUM_KV_SPLITS = num_kv_splits
        grid = (
            batch,
            triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
            NUM_KV_SPLITS,
        )

        try:
            _fwd_grouped_kernel_stage1[grid](
                q,
                k_buffer,
                v_buffer,
                sm_scale,
                Req_to_tokens,
                B_Seqlen,
                att_out,
                Req_to_tokens.stride(0),
                q.stride(0),
                q.stride(1),
                _page_stride(k_buffer, page_size),
                k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
                k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
                _page_stride(v_buffer, page_size),
                v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
                v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
                att_out.stride(0),
                att_out.stride(1),
                att_out.stride(2),
                k_scale,
                v_scale,
                kv_group_num=kv_group_num,
                q_head_num=head_num,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DPE=BLOCK_DPE,
                BLOCK_DV=BLOCK_DV,
                BLOCK_N=BLOCK,
                BLOCK_H=BLOCK_H,
                NUM_KV_SPLITS=NUM_KV_SPLITS,
                PAGE_SIZE=page_size,
                logit_cap=logit_cap,
                RMS_OFFSET=rms_offset,
                PAGE_ALIGNED=PAGE_ALIGNED,
                RMS_BF16=rms_bf16,
                RMS_LOG8=rms_log8,
                # At BLOCK_DV 2048 the tl.trans of the K tile is large enough that re-loading V from
                # L1/L2 beats transposing it (8.31 -> 7.87 ms at rank 1792); at 1024 the transpose still
                # wins (3.70 vs 5.31), so this is on only for the widest layer.
                MLA_RELOAD_V=_os.environ.get("QWEN_MLA_RELOAD_V", "1" if BLOCK_DV >= 2048 else "0") == "1",
                num_warps=_num_warps,
                num_stages=num_stages,
                Lk=Lk,
                Lv=Lv,
                IS_MLA=is_mla,
                **extra_kargs,
            )
        except triton.runtime.errors.OutOfResources as e:
            _err = e
            continue
        if _key not in _FIT_CACHE:
            _FIT_CACHE[_key] = (BLOCK_H, BLOCK, _num_warps, num_stages)
            if (BLOCK_H, BLOCK, _num_warps, num_stages) != _cands[0]:
                print(f"[qwen-mla-decode] shared-memory fit for DMODEL {BLOCK_DMODEL}/DV {BLOCK_DV}: "
                      f"BLOCK_H {BLOCK_H}, BLOCK_N {BLOCK}, warps {_num_warps}, stages {num_stages}", flush=True)
        # stage 2 must recompute the same split boundaries, so it needs the tile width.
        return BLOCK if PAGE_ALIGNED else 0
    raise _err


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    ALIGN_N: tl.constexpr = 0,
    OUTPUT_FP16: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        # MUST match _fwd_grouped_kernel_stage1 exactly: stage 1 rounds the split length up to a
        # whole number of BLOCK_N tiles when PAGE_ALIGNED, so ALIGN_N carries that BLOCK_N here.
        # Disagreeing on the boundary silently reads partials stage 1 never wrote.
        if ALIGN_N > 0:
            kv_len_per_split = (
                tl.cdiv(tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS), ALIGN_N) * ALIGN_N
            )
        else:
            kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    result = acc / e_sum
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        result,
        mask=mask_d,
    )
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse + cur_batch * stride_lse_bs + cur_head,
        lse_val,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    lse,
    v_buffer,
    b_seq_len,
    num_kv_splits,
    align_n=0,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        lse,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        ALIGN_N=align_n,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
    rms_offset=-1,
    rms_bf16=False,
    rms_log8=False,
):
    align_n = _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
        is_mla=is_mla,
        rms_offset=rms_offset,
        rms_bf16=rms_bf16,
        rms_log8=rms_log8,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits, align_n=align_n
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
    rms_offset=-1,
    rms_bf16=False,
    rms_log8=False,
):
    assert num_kv_splits == attn_logits.shape[2]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            k_scale,
            v_scale,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            k_scale,
            v_scale,
            is_mla=is_mla,
            rms_offset=rms_offset,
            rms_bf16=rms_bf16,
            rms_log8=rms_log8,
        )
