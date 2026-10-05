"""Re-align the prefill VALUE tensor, which partial rope leaves 4 bytes short of a TMA boundary.

THE BUG THIS WORKS AROUND IS vLLM's, NOT OURS. MLACommonImpl builds k and v from one fused
projection and hands the halves straight to the backend:

    kv_nope = kv_b_proj(c).view(-1, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, v = kv_nope.split([qk_nope_head_dim, v_head_dim], -1)   # mla_attention.py:2611

`v` is therefore a VIEW whose base pointer sits `qk_nope_head_dim` elements into the buffer. At
DeepSeek's 128 and Qwen3.5's 192 that offset is 256 / 384 bytes -- both multiples of 16, so the
alignment holds by luck of the head-dim arithmetic and no one ever had to think about it.

Partial rope moves six dims out of the rope block and into the nope block, so qk_nope_head_dim
becomes 198: offset 396 bytes, 12 (mod 16). The trtllm-gen FMHA kernel builds a TMA descriptor
on that pointer and hard-fails:

    kernelParams.h:589  Check failed: (reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0

Nothing else is wrong -- q and k are freshly allocated by _mla_narrow_q / _concat_k_nope_k_pe
and land on allocator boundaries; only the split view is offset. So the whole partial-rope
geometry (rope 58, nope 198, tail 4*58+24 = 256) is sound and needs no padding and no fourth
dropped subspace; it just needs `v` copied to a fresh allocation.

The check is on the ADDRESS, not on contiguity, and it is applied unconditionally rather than
only when nope_dim is odd-aligned: a tensor can be contiguous and still misaligned, so a bare
.contiguous() is not sufficient on its own and the result is re-checked.

COST. One strided copy of [prefill_tokens, local_heads, 256] bf16 per layer per chunk -- 12 KB
per token, ~0.4 GB for a 32k chunk at TP=1, under a millisecond of HBM against a prefill step
that is tens of milliseconds of matmul. Measured impact is inside the noise; the alternative
(rope_dim = 0 mod 8) costs a fourth dropped subspace AND 8 dims of dead tail per cached token
for every request, forever.
"""
from __future__ import annotations

import functools

import torch

_ALIGN = 16

# Where `v` sits in each entry point's POSITIONAL signature, for the case a future vLLM stops
# passing it by keyword. Both current call sites use keywords (mla_attention.py:2618, :2384),
# so these indices are a guard rather than the live path -- but silently aligning the wrong
# argument would be worse than the crash, so they are named rather than guessed from the end.
_V_POS = {"run_prefill_new_tokens": 2,        # (q, k, v, return_softmax_lse, out, output_scale)
          "run_prefill_context_chunk": 3}     # (chunk_idx, q, k, v)


def align16(t):
    """A 16-byte-aligned tensor with the same values; the input itself when already aligned."""
    if t is None or t.data_ptr() % _ALIGN == 0:
        return t
    out = t.contiguous()
    if out.data_ptr() % _ALIGN:          # was already contiguous, just offset -- force a copy
        out = t.clone(memory_format=torch.contiguous_format)
    return out


def patch_prefill_backend(backend) -> None:
    """Wrap this backend CLASS's two entry points so `v` is aligned before the kernel sees it.

    Patched on the class, once, guarded by a sentinel: the backend object is rebuilt per layer
    but the class is shared, so per-instance wrapping would stack 16 layers of closures. Both
    entry points need it -- run_prefill_new_tokens takes the split `v` (mla_attention.py:2611)
    and run_prefill_context_chunk takes another one from the chunked-context loop (:2384, :2542).
    """
    cls = type(backend)
    if getattr(cls, "_mla_v_aligned", False):
        return
    for name in ("run_prefill_new_tokens", "run_prefill_context_chunk"):
        orig = getattr(cls, name, None)
        if orig is None:
            continue

        def _wrap(fn, pos):
            @functools.wraps(fn)
            def inner(self, *a, **kw):
                if "v" in kw:
                    kw["v"] = align16(kw["v"])
                elif len(a) > pos:
                    a = a[:pos] + (align16(a[pos]),) + a[pos + 1:]
                return fn(self, *a, **kw)
            return inner

        setattr(cls, name, _wrap(orig, _V_POS[name]))
    cls._mla_v_aligned = True
