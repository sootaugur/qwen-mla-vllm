"""Make vLLM's DCP output-correction kernel work at a non-power-of-two head dim.

Decode context parallelism shards the KV cache across ranks by sequence, so each rank attends
over its own slice and the results are combined with the all-gathered LSEs. That combine is
`vllm/v1/attention/ops/common.py::_correct_attn_cp_out_kernel`, which does

    d_offsets = tl.arange(0, HEAD_DIM)          # HEAD_DIM = D, the value head dim

`tl.arange` requires a power of two. For DeepSeek MLA D is 512 and it never comes up; MLA's
per-layer latent ranks are 256 / 768 / 1792, so two of the three fail at compile time with
"arange's range must be a power of 2" -- during cudagraph capture, before a single token is
generated.

Same fix the decode kernel needed: iterate a padded power-of-two range and mask the loads and
stores back to the real width. Nothing else about the kernel changes, and at D=512 the padding
is a no-op, so the DeepSeek path is unaffected.

Patched as a module attribute on vllm.v1.attention.ops.common, not by editing the installed
package: cp_lse_ag_out_rs / cp_lse_ag_out_ar resolve `correct_attn_out` as a module global at
call time, so replacing it there reaches every caller.
"""
from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    # padded so tl.arange is legal at HEAD_DIM 768 / 1792; masked back below
    d_offsets = tl.arange(0, HEAD_DIM_PADDED)
    d_mask = d_offsets < HEAD_DIM
    num_n_offsets = tl.arange(0, N_ROUNDED)

    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets, mask=d_mask, other=0.0)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)
    tl.store(new_output_ptr + output_offsets, output, mask=d_mask)


def _mla_correct_attn_out(out, lses, cp_rank, ctx, is_lse_base_on_e: bool = True):
    from vllm.v1.attention.ops.common import CPTritonContext

    if ctx is None:
        ctx = CPTritonContext()

    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"
    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, f"expected lses [N,B,H], got {tuple(lses.shape)}"

    B, H, D = out.shape
    N = lses.shape[0]
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    lse = torch.empty_strided((B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype)

    regular_args = (out, out, lses, lse, o_sB, o_sH, o_sD, l_sN, l_sB, l_sH, cp_rank)
    const_args = {
        "HEAD_DIM": D,
        "HEAD_DIM_PADDED": triton.next_power_of_2(D),
        "N_ROUNDED": N,
        "IS_BASE_E": is_lse_base_on_e,
    }
    ctx.call_kernel(_mla_correct_attn_cp_out_kernel, (B, H, 1), *regular_args, **const_args)
    return out, lse


def install() -> None:
    """Swap in the padded kernel. Idempotent."""
    import vllm.v1.attention.ops.common as common

    if getattr(common, "_mla_dcp_patched", False):
        return
    common.correct_attn_out = _mla_correct_attn_out
    common._mla_dcp_patched = True
    print("[qwen-mla] DCP output correction patched for non-power-of-two head dims",
          file=sys.stderr, flush=True)


def skip_fa4_warmup() -> None:
    """Skip the FA4 CuteDSL prefill warmup, which use_mla=True drags us into.

    With use_mla True, kernel_warmup walks the MLA prefill backends and asks FA4 for warmup
    keys. It fails at our geometry -- first for kv_lora_rank, then with "SM100 forward with
    head_dim=256 does not support SplitKV" -- while warming a backend we never run: MLA prefill
    is pinned to FLASH_ATTN (see the serve script; TRTLLM_RAGGED was never validated against our
    narrowed (192, 64, 256) tensors).

    Warmup only moves JIT cost off the first request, so skipping it is a startup-latency
    trade, not a correctness one.
    """
    import sys

    import vllm.model_executor.warmup.kernel_warmup as kw

    if getattr(kw, "_mla_fa4_warmup_skipped", False):
        return
    kw.fa4_cutedsl_warmup = lambda *a, **k: None
    kw._mla_fa4_warmup_skipped = True
    print("[qwen-mla] FA4 CuteDSL prefill warmup skipped (MLA prefill is FLASH_ATTN)",
          file=sys.stderr, flush=True)
