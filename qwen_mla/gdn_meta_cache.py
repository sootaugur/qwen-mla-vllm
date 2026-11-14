"""Build the GDN attention metadata once per step instead of once per linear-attention layer.

vLLM groups layers into KV-cache groups of equal size. The MTP draft layer is a group of one, so every
layer ends up in its own group: 48 GDN groups on Qwen3.8-27B. The model runner builds attention
metadata per group, and reuses one build across groups only when the builder implements
`update_block_table`. GDNAttentionMetadataBuilder does not, so `build()` runs 48 times per step,
each one a few dozen small host-side torch ops plus, in `--mamba-cache-mode align` (prefix caching),
a block-table gather with its own arange / index / host-to-device copies.

Without prefix caching that host work overlaps the previous step's GPU work (async scheduling), but in
align mode the runner waits for the previous step's accepted-token counts before preparing the next
step, so the host time lands directly on every decode step. Measured on 2x RTX 5060 Ti, MTP k=3: GPUs
63% busy with prefix caching vs 95% without; 57 vs 66 tok/s at one request.

This adds `update_block_table`: the first GDN group builds as before; the others reuse that metadata
and recompute only their state indices (the one thing that differs between groups: each layer's state
lives in its own blocks).

FULL CUDA graphs read each builder's persistent buffers at captured addresses, so a cached build must
still refresh this builder's buffers. The state-index buffers stay per builder and are refreshed here.
The other persistent buffers (spec masks, token indices, query start locations, accepted counts) hold
the same values for every group, so all GDN builders on a device share one set, which the group that
builds fills once per step.

When it is on: by default only with speculative decoding AND `--mamba-cache-mode align`, the one
combination where it was measured to help (2x RTX 5060 Ti, MTP k=3: decode +17% / +15% / +16% / +9% at
1 / 2 / 3 / 4 concurrent requests). Without MTP it measured ~5% slower (within run-to-run noise, but
consistently), and without prefix caching the host work already overlaps the GPU, so there is nothing
to win. QWEN_MLA_GDN_META_CACHE=1 forces it on, =0 off.
"""
from __future__ import annotations

import copy
import os

_SHARED_BUFFERS = ("spec_sequence_masks", "spec_token_indx", "non_spec_token_indx",
                   "spec_query_start_loc", "non_spec_query_start_loc", "num_accepted_tokens")


def _wanted(vllm_config) -> bool:
    env = os.environ.get("QWEN_MLA_GDN_META_CACHE", "")
    if env in ("0", "1"):
        return env == "1"
    return (vllm_config.speculative_config is not None
            and vllm_config.cache_config.mamba_cache_mode == "align")


def patch_gdn_metadata_cache():
    if os.environ.get("QWEN_MLA_GDN_META_CACHE") == "0":
        return
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
    B = gdn_attn.GDNAttentionMetadataBuilder
    if getattr(B, "_qwen_mla_meta_cache", False):
        return
    NULL = gdn_attn.NULL_BLOCK_ID
    orig_init, orig_build = B.__init__, B.build
    shared: dict = {}

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *a, **k):
        orig_init(self, kv_cache_spec, layer_names, vllm_config, device, *a, **k)
        self.supports_update_block_table = _wanted(vllm_config)
        if not self.supports_update_block_table:
            return
        if not shared:
            print("[qwen-mla] GDN metadata: built once per step and shared across the linear-attention "
                  "layers (QWEN_MLA_GDN_META_CACHE=0 to disable)", flush=True)
        key = (str(device), self.decode_cudagraph_max_bs, self.num_spec)
        bufs = shared.setdefault(key, {n: getattr(self, n) for n in _SHARED_BUFFERS})
        for n in _SHARED_BUFFERS:
            setattr(self, n, bufs[n])

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                        num_decode_draft_tokens_cpu, fast_build)
        if not self.supports_update_block_table:
            return md
        mask_cpu = None
        if md.num_spec_decodes > 0:
            mask_cpu = num_decode_draft_tokens_cpu >= 0
        sp, ns = md.spec_state_indices_tensor, md.non_spec_state_indices_tensor
        md._qwen_mla_recipe = dict(
            seq_lens=common_attn_metadata.seq_lens,
            mask_cpu=mask_cpu,
            batch_size=common_attn_metadata.num_reqs,
            graph_spec=sp is not None and sp.data_ptr() == self.spec_state_indices_tensor.data_ptr(),
            graph_nonspec=ns is not None and ns.data_ptr() == self.non_spec_state_indices_tensor.data_ptr(),
        )
        return md

    def update_block_table(self, metadata, blk_table, slot_mapping):
        r = getattr(metadata, "_qwen_mla_recipe", None)
        if r is None:                       # cached by a builder with the cache off: cannot happen
            raise RuntimeError("GDN metadata without a qwen-mla recipe")
        bt = mamba_get_block_table_tensor(blk_table, r["seq_lens"], self.kv_cache_spec,
                                          self.vllm_config.cache_config.mamba_cache_mode)
        new = copy.copy(metadata)
        bs, mask = r["batch_size"], r["mask_cpu"]
        if mask is None:                    # no speculative rows: plain decodes and/or prefills
            ns = bt[:, 0]
            if metadata.num_prefills > 0:
                new.prefill_state_indices = ns[metadata.num_decodes:] if metadata.num_decodes > 0 else ns
            if r["graph_nonspec"]:
                buf = self.non_spec_state_indices_tensor
                buf[:metadata.num_decodes].copy_(ns, non_blocking=True)
                ns = buf[:bs]
                ns[metadata.num_decodes:].fill_(NULL)
            new.non_spec_state_indices_tensor = ns
        else:
            sp = bt[mask, : self.num_spec + 1]
            if metadata.num_prefills == 0 and metadata.num_decodes == 0:
                ns = None
            else:
                ns = bt[~mask, 0]
            if metadata.num_prefills > 0:
                new.prefill_state_indices = ns
            if r["graph_spec"]:
                buf = self.spec_state_indices_tensor
                buf[:metadata.num_spec_decodes].copy_(sp, non_blocking=True)
                sp = buf[:bs]
                sp[metadata.num_spec_decodes:].fill_(NULL)
            new.spec_state_indices_tensor = sp
            new.non_spec_state_indices_tensor = ns
        return new

    B.__init__ = __init__
    B.build = build
    B.update_block_table = update_block_table
    B._qwen_mla_meta_cache = True
