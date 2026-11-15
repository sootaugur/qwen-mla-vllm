"""Free superseded GDN (Mamba) state blocks during prefill under --mamba-cache-mode align.

In align mode every scheduled chunk writes the recurrent state at its end into a fresh block, and
the request's block table interleaves those blocks with null placeholders (align skips the blocks
it has no state for). Two paths are meant to free the states a request has moved past, and with
async scheduling neither does:

  * MambaManager.remove_skipped_blocks frees one block, `last_state_block_idx` (the state from two
    steps ago), and only if the processed-token count has already passed it. With a step in flight
    the processed count lags by that step, the check fails, and the next allocation overwrites
    `last_state_block_idx` -- that block is never revisited.
  * The base class's _remove_blocks_in_range walks back from the boundary and stops at the first
    null block, which in align mode is always right there.

So each prefill chunk leaks one state block per GDN layer until the request finishes. On GLA-g2
(48 GDN layers, 1.69 MiB pages, 3136-token chunks) that is 81 MiB per chunk, ~2.5 GiB for a 100k
prompt: three times the request's attention KV, so the pool held 1.7 such requests where the
capacity estimate said 4, and the rest were preempted and re-prefilled.

The fix applies vLLM's own condition -- a state block below the processed boundary
((processed - 1) // block_size, the same boundary both paths use) is no longer needed -- to every
block below it rather than one. Freed states keep their prefix-cache hashes: they move from pinned
to evictable, exactly as they would have at request end.
"""
from __future__ import annotations


def patch_mamba_align_free():
    import os
    if os.environ.get("QWEN_MLA_ALIGN_FREE", "1") == "0":
        return
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    if getattr(MambaManager, "_qwen_mla_align_free", False):
        return
    orig = MambaManager.remove_skipped_blocks

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        orig(self, request_id, processed_computed_tokens, num_prompt_tokens)
        if self.mamba_cache_mode != "align":
            return
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        end = min((processed_computed_tokens - 1) // self.block_size, len(blocks))
        stale = []
        for i in range(end):
            if not blocks[i].is_null:
                stale.append(blocks[i])
                blocks[i] = self._null_block
        if stale:
            self.block_pool.free_blocks(stale)

    MambaManager.remove_skipped_blocks = remove_skipped_blocks
    MambaManager._qwen_mla_align_free = True
