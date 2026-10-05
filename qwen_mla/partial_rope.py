"""Partial-RoPE helpers shared by the materialised and absorbed serving paths.

WHY PARTIAL RoPE EXISTS HERE. FlashInfer's MLA decode kernel requires
`kv_lora_rank % 128 == 0` at TP=2 (fi_overlay/.../mla.cuh:124, D_SHARDS=8), while page
unification wants `rank + tail = 2^k`. Together those force `tail % 128 == 0`. The tail is
`rope_span * rope_dim + rms_scalars`; at TP=1 that is `4*64 + 24 = 280 = 24 (mod 128)` -- illegal
at every rank, which is why the shipped checkpoint pads to rows 1024/2048/4096 and serves 1.00x
on a single GPU.

Deleting the 24 rms scalars (DeepSeek canon) reaches 256 but was measured and REJECTED: msmarco
NDCG@10 60.27 -> 23.98, Recall@10 24 -> 7.1. Shrinking the ROPE block reaches the same 256 with
every normalisation intact: `4*58 + 24 = 256 = 2*128`, so ranks 256/768/1792 land exactly on rows
512/1024/2048.

The dropped subspaces do not vanish -- they are permuted to the front of the nope block at export
and carried by the latent un-rotated (MHA2MLA's partial-RoPE, arXiv 2502.14837). Subspaces
29/30/31 were selected on measurement: they rotate 3.4 / 2.1 / 1.2 degrees across 131k tokens, so
they encode content rather than position, and they carry below-average magnitude so the latent
inherits little extra reconstruction burden. Measured cost: 0.486% of attention logits at 131k.

A NAIVE 2-NORM CRITERION IS A TRAP. Ranking subspaces by ||q||*||k|| puts the FASTEST ones last
and would drop those -- which is MHA2MLA's S_low, their worst strategy at -5.25%. Magnitude is
not positional information.
"""
from __future__ import annotations


def rope_keep_from_config(hf) -> list[int] | None:
    """Kept subspace indices, or None for full rope (the original behaviour)."""
    keep = getattr(hf, "mla_rope_keep", None)
    if keep is None:
        return None
    keep = [int(x) for x in keep]
    if not keep:
        raise ValueError("mla_rope_keep is empty")
    return keep


def model_rope_dim(hf) -> int:
    """The BASE model's rotary width, independent of any partial-rope narrowing."""
    rp = getattr(hf, "rope_parameters", None) or {}
    prf = rp.get("partial_rotary_factor", getattr(hf, "partial_rotary_factor", 0.25))
    return int(hf.head_dim * prf)


def effective_rope_dim(hf) -> int:
    """Per-group rope width actually cached: 2*len(keep) under partial rope, else the model's."""
    keep = rope_keep_from_config(hf)
    return model_rope_dim(hf) if keep is None else 2 * len(keep)


def patch_partial_rope(rotary_emb, keep, mrd: int, theta: float) -> None:
    """Rewrite a rotary cache so the kept subspaces use their ORIGINAL frequencies.

    vLLM derives inv_freq from whatever rotary width it was built with, so a 58-wide rope would
    silently use `base^(-2i/58)` instead of the kept subspaces' `base^(-2i/64)`. Every angle
    would be wrong -- and wrong in a way that still produces fluent text, so nothing would fail
    loudly.

    The cache is [max_pos, rotary_dim] laid out as cat([cos(freqs), sin(freqs)], -1) over
    rotary_dim//2 frequencies. Angles are accumulated in FLOAT32 to match vLLM's own
    _compute_cos_sin_cache: float64 is more accurate but diverges by up to 2.7e-4 at these
    positions (t*inv_freq reaches thousands of radians), and matching the convention makes this
    cache BIT-IDENTICAL to slicing the full-width one -- verified with torch.equal.
    """
    import torch

    cache = rotary_emb.cos_sin_cache
    n = cache.shape[0]
    inv = 1.0 / (theta ** (2.0 * torch.tensor(keep, dtype=torch.float32) / mrd))
    freqs = torch.outer(torch.arange(n, dtype=torch.float32), inv)
    new = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(cache.dtype).to(cache.device)
    if new.shape != cache.shape:
        raise ValueError(f"partial-rope cache {tuple(new.shape)} != {tuple(cache.shape)}; "
                         f"rotary was built at the wrong width for keep={len(keep)}")
    cache.copy_(new)
