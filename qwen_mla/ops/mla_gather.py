"""Torch fallback for gather_and_maybe_dequant_cache at non-DeepSeek cache widths.

    RuntimeError: gather_and_maybe_dequant_cache only support the head_dim to
                  320 or 576 for better performance

The CUDA kernel hard-rejects any entry size outside DeepSeek's two. Our rows are
kv_lora_rank + tail (524 / 1048 / 2096 at TP=2), so every chunked-context prefill dies -- which
is every request long enough to matter, and none of the short ones we validated on.

SEMANTICS, read off the metadata builder rather than guessed (mla_attention.py ~1646):
    chunk_starts[chunk][seq]  -> seq_starts : where this chunk begins IN THE SEQUENCE
    cu_seq_lens[chunk]        -> [0, cumsum(chunk_seq_lens)] : where each seq begins IN dst
    token_to_seq[chunk]       -> repeat_interleave(seq_idx, chunk_seq_lens)
so for dst row j:  s = token_to_seq[j];  tok = seq_starts[s] + (j - cu_seq_lens[s])
and tok indexes the sequence, which the block table maps into the paged cache.

test_gather_torch.py checks this against the CUDA op itself at width 576, so the reading is
verified rather than asserted -- a plausible-but-wrong reading here would degrade long-context
attention while leaving the output fluent.
"""
from __future__ import annotations

import torch


def gather_cache_torch(*, src_cache: torch.Tensor, dst: torch.Tensor,
                       block_table: torch.Tensor, cu_seq_lens: torch.Tensor,
                       token_to_seq: torch.Tensor, num_tokens: int,
                       kv_cache_dtype: str, scale: torch.Tensor,
                       seq_starts: torch.Tensor | None = None) -> None:
    if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise NotImplementedError(
            f"MLA gather fallback has no dequant path for kv_cache_dtype={kv_cache_dtype}; "
            f"this build serves bf16 deliberately")
    if num_tokens <= 0:
        return

    block_size = src_cache.shape[1]
    j = torch.arange(num_tokens, device=dst.device)
    s = token_to_seq[:num_tokens].long()
    tok = j - cu_seq_lens.long()[s]
    if seq_starts is not None:
        tok = tok + seq_starts.long()[s]
    blk = block_table[s, tok // block_size].long()
    dst[:num_tokens].copy_(src_cache[blk, tok % block_size].reshape(num_tokens, -1))
