"""MLA impl for Qwen3.5-geometry MLA: absorbed decode + exact prefill k_norm.

WHAT DIFFERS FROM DeepSeek MLA, AND WHY IT NEEDS A SUBCLASS.
DeepSeek normalises the LATENT (kv_a_layernorm), so K stays linear in c and W_UK folds into the
query with nothing left over. Qwen3.5 applies k_norm to the ASSEMBLED [rope ; nope] key AFTER
up-projection:

    q . k_norm(key_h) = ( [ (q_nope*(1+w)) @ W_UK_h ] . c  +  rope_term ) / rms(key_h)

The numerator absorbs exactly as usual. The denominator is a per-(key-token, HEAD) scalar and
the latent is shared across heads, so it cannot be folded into the cache. Two consequences:

  decode  : the forked Triton kernel divides the scores by rms, read from a tail appended to
            the KV cache (see ops/mla_decode_attention.py).
  prefill : k/v are materialised from the latent anyway, so k_norm is applied literally --
            exact, no trick needed.

WHERE (1+w) LIVES. It is folded into kv_b_proj's K rows at build time, which serves BOTH paths:
prefill reads kv_b_proj directly, and vLLM derives the decode-time W_UK_T from the same weight.
rms must be computed from the RAW up-projection, so the model keeps an unscaled copy.

CACHE LAYOUT  [ latent | rope 256 | rms (local heads) | page pad ]  -- 1048 dims at TP=1
The rms tail rides inside the declared qk_rope_head_dim (256+24=280) so vLLM's compiled
concat_and_cache_mla writes it, and paging / eviction / prefix-caching move it for free. The
kernel never reads it as attention because Lk is pinned to rms_offset.

PREFILL RUNS AT THE STOCK (192, 64, 256) GEOMETRY, AND THAT IS THE WHOLE TRICK.
Only DECODE needs the packed layout: there the latent is shared across heads, so all four rope
keys must be resident and the kernel picks per head. Prefill materialises k per head from the
latent regardless, so each head can take its OWN 64-wide rope slice -- which is exactly the
DeepSeek dimension triple every vLLM MLA prefill backend already supports. So we narrow q and k
from the packed 280 down to 64 at the two funnels below, and prefill needs no custom kernel and
no backend of our own. Without this the key would be 192+256=448 wide and no backend takes it.

TP CORRECTNESS -- TWO DIFFERENT INDEX SPACES, WHICH IS EASY TO CONFLATE.
k_rope_proj is Replicated, so every rank holds all four ROPE groups and must pick its head's
group by GLOBAL index (mla_head_offset + i): using the local index serves rank 1 groups 0,1
where 2,3 are correct -- wrong keys, still-fluent output, nothing raised.
The RMS tail is the opposite. Query heads are sharded, so each rank stores rms for its OWN
heads only and indexes them LOCALLY. Sizing that tail globally makes k_pe narrower than the
cache row and kills concat_and_cache_mla inside the CUDA kernel.
"""
from __future__ import annotations

import torch
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backends.mla.triton_mla import (TritonMLABackend, TritonMLAImpl,
                                                       TritonMLAMetadataBuilder)

from vllm.distributed import get_tensor_model_parallel_rank

from . import fi_decode
from .ops.mla_decode_attention import decode_attention_fwd as mla_decode_attention_fwd
from .ops.mla_gather import gather_cache_torch
from .prefill_align import patch_prefill_backend


def chunked_workspace_rows(vllm_config) -> int:
    """Rows of the chunked-prefill context workspace: vLLM's rule without its one-page-per-sequence floor.

    vLLM sizes it min(max(8 x max_model_len, 4 x max_num_seqs x page), 64k) and then floors it at
    max_num_seqs x page, so every prefill can get a page-aligned chunk. Our pages are 784 tokens, so
    at the default 1024 sequences that floor is 802,816 rows: 3.3 GB of workspace for the widest
    group, and a 16 GB simulated up-projection in the profile run, which OOMs before serving. Our
    gather (ops/mla_gather.py) handles chunk starts at any token, so build() drops the page
    alignment and one row per prefill is the real floor.
    """
    sched, model = vllm_config.scheduler_config, vllm_config.model_config
    return max(min(8 * model.max_model_len, 64 * 1024), sched.max_num_seqs)


def splitk_cap(batch: int, heads: int, lse_dim: int, splits: int) -> int:
    """Largest split count <= `splits` whose fp32 split-K scratch fits QWEN_MLA_SPLITK_SCRATCH_MB.

    vLLM sizes the Triton decode accumulator [batch, heads, splits, rank + 1] for the worst case
    (max_num_seqs, and splits from max_model_len) and reserves it before KV allocation. With
    defaults on a 96 GB part (1024 seqs, 262k context -> 376 splits) and our widest latent (1792)
    that is 66 GB, so the server OOMs before it starts. Halving the splits only trades parallelism
    over keys: the default 512 MB leaves the measured configurations (16k context, up to 166
    sequences) at their natural split count.
    """
    import os
    budget = int(float(os.environ.get("QWEN_MLA_SPLITK_SCRATCH_MB", 512)) * (1 << 20))
    while splits > 1 and batch * heads * splits * lse_dim * 4 > budget:
        splits //= 2
    return max(1, splits)


def splitk_reserve_bytes(batch: int, heads: int, lse_dim: int, splits: int) -> int:
    """Bytes that cover every splitk_cap()-ed request with batch <= `batch`."""
    import os
    budget = int(float(os.environ.get("QWEN_MLA_SPLITK_SCRATCH_MB", 512)) * (1 << 20))
    full = batch * heads * splits * lse_dim * 4
    return min(full, max(budget, batch * heads * lse_dim * 4))


def mla_base_tail(hf, tp_size: int = 1) -> int:
    """Live cache tail beyond the latent: packed rope (4x64) + one rms scalar per LOCAL head.

    The rope block is replicated (k_rope_proj is ReplicatedLinear, so every rank holds all four
    groups), but rms is per query head and the query heads are sharded -- rank 1 needs the rms
    of heads 12-23 and nothing else. Sizing the tail at the GLOBAL head count reserves 24 slots
    while forward writes 12, so k_pe comes out narrower than the cache row and
    concat_and_cache_mla fails inside the CUDA kernel with no usable message.
    """
    # EFFECTIVE width, not the model's: under partial rope only 2*len(mla_rope_keep) dims per
    # group are cached, which is the entire mechanism by which this tail becomes a multiple of
    # 128 (4*58 + 24 = 256). Reading the model's 64 here would silently restore the old
    # 280-dim tail and undo it.
    from .partial_rope import effective_rope_dim
    rope = effective_rope_dim(hf)
    return mla_rope_span(hf, tp_size) * rope + hf.num_attention_heads // tp_size


def mla_rope_span(hf, tp_size: int = 1) -> int:
    """How many rope GROUPS a rank's heads span -- i.e. how many are worth caching.

    k_rope_proj is replicated so every rank computes all four, but a rank only reads the groups
    its own query heads belong to: at TP=2 rank 0 spans groups 0-1 and rank 1 spans 2-3. Caching
    all four wastes half the rope block. The span is identical on every rank (heads split
    evenly), which is what lets one row width describe them all.
    """
    group = hf.num_attention_heads // hf.num_key_value_heads
    n_local = hf.num_attention_heads // tp_size
    return max((r * n_local + n_local - 1) // group - (r * n_local) // group + 1
               for r in range(tp_size))


def mla_gla_split(hf, tp_size: int) -> int:
    """Latent groups each TP rank's slice is divided by: G when the GLA groups align with TP.

    A GLA checkpoint (mla_latent_groups = G > 1) has block-diagonal up-projections: query heads of
    group g read only latent columns [g*r/G, (g+1)*r/G). When tp_size is a multiple of G every rank's
    heads lie in ONE group, so the rank needs -- and caches -- only that group's r/G latent dims.
    Returns 1 (no split: full latent per rank) otherwise, or when QWEN_MLA_GLA_SHARD=0.
    """
    import os
    g = int(getattr(hf, "mla_latent_groups", 1) or 1)
    if g > 1 and tp_size % g == 0 and os.environ.get("QWEN_MLA_GLA_SHARD", "1") != "0":
        return g
    return 1


def mla_local_ranks(hf, tp_size: int = 1) -> dict[int, int]:
    """Per-layer latent width as CACHED on one TP rank (see mla_gla_split)."""
    g = mla_gla_split(hf, tp_size)
    return {int(k): int(v) // g for k, v in hf.mla_ranks.items()}


def mla_padded_head_size(hf, kv_lora_rank: int, tp_size: int = 1) -> int:
    """Cache row width for a layer, padded so every layer's PAGE divides the largest.

    vLLM unifies differing page sizes by scaling block_size -- but only when the largest page is
    an exact multiple of the others (kv_cache_utils.unify_kv_cache_spec_page_size). Otherwise it
    pads every page to the maximum. Our allocation gives rows of 536 / 1048 / 2072, whose ratios
    are 67:131:259, so nothing divides and ALL 16 layers get padded to 2072 -- against the base
    model's own 2048 dims/token/layer. The 2.000x compression the checkpoint was trained for
    would then be 0.99x in production, measured (5.93x over the materialised path where 11.73x
    was predicted; 12288/2072 = 5.93 identifies the cause exactly).

    Rounding each row up to a power-of-two multiple of the SMALLEST row makes the ratios
    1:2:4, so unification is by block size and the only cost is the rounding itself: rows become
    536 / 1072 / 2144, a mean of 1072 against the true 1048 -- 2.3% overhead to avoid 98%.

    The padding sits past the rms tail and is never read: the decode kernel bounds its
    attention reads at rms_offset, and prefill slices rope/rms at explicit offsets.
    """
    tail = mla_base_tail(hf, tp_size)
    base = min(mla_local_ranks(hf, tp_size).values()) + tail
    # NOTE on 16-byte alignment. The row width is also the per-token stride, so at TP=2 the
    # smallest rank gives 256 + 268 = 524 elements = 1048 bytes and every odd token row is only
    # 8-byte aligned -- which the flashinfer decode path cannot use, because a 128-bit cp.async
    # on it faults. Rounding base up to 528 fixes that and keeps the 1:2:4 ratio, but widening
    # the row makes concat_and_cache_mla reject the write, so it is NOT done here; the
    # flashinfer path checks alignment and routes those two layers to Triton instead
    # (fi_decode._cp_async_safe). Worth revisiting with the cache-write path in hand.
    raw = kv_lora_rank + tail

    def _pow2(x: int) -> int:
        w = 1
        while w < x:
            w *= 2
        return w

    # TWO LEGAL SCHEMES, and which one wins depends on TP. vLLM only requires
    # `max_page % layer_page == 0`; powers of two are sufficient, not necessary.
    #
    #   pow2      : row = next_pow2(rank + tail)
    #   base-mult : smallest power-of-two MULTIPLE of a 16-aligned base >= rank + tail
    #               (this is the scheme the docstring above describes; the code had drifted
    #                to plain pow2)
    #
    # At TP=1 the tail is 280, so rank 256 gives raw 536 -- barely over 512 -- and pow2 rounds
    # the whole allocation to 1024/2048/4096, mean 2048, which is EXACTLY the uncompressed KV
    # size (2 x 4 kv_heads x 256 head_dim). MEASURED: official125-absorbed serves 11,984 tok/GiB
    # at TP=1 against 23,967 at TP=2 -- a single-GPU user gets no compression at all. base-mult
    # gives 544/1088/2176, mean 1088 = 1.88x. It costs nothing at TP>=2, where pow2 already wins
    # (512/1024/2048 = 2.00x) and is therefore selected, so behaviour there is unchanged.
    #
    # Both schemes keep every row a multiple of 16 elements: that satisfies the 128-bit cp.async
    # in the flashinfer decode path (the alignment failure noted above was 524 elements = 1048
    # bytes, not a multiple of 8) and keeps HEAD_DIM_KPE = row - rank a multiple of 16 for the
    # kernel's NUM_MMA_D_KPE asserts.
    counts: dict[int, int] = {}
    for v in mla_local_ranks(hf, tp_size).values():
        counts[int(v)] = counts.get(int(v), 0) + 1
    ranks = sorted(counts)
    base = ((min(r + tail for r in ranks) + 15) // 16) * 16
    cand = []
    for rows in ({r: _pow2(r + tail) for r in ranks},
                 {r: base * _pow2(-(-(r + tail) // base)) for r in ranks}):
        mx = max(rows.values())
        if any(mx % v for v in rows.values()):
            continue                                   # pages would not unify
        if any(v % 16 for v in rows.values()):
            continue                                   # cp.async / NUM_MMA_D_KPE alignment
        cand.append((sum(rows[r] * counts[r] for r in ranks) / sum(counts.values()), rows))
    if cand:
        rows = min(cand, key=lambda t: t[0])[1]
        if kv_lora_rank in rows:
            return rows[kv_lora_rank]
    return _pow2(raw)


def mla_rank_by_head_size(hf, tp_size: int = 1) -> dict[int, int]:
    """padded head_size -> kv_lora_rank. Max on collision: the value sizes a workspace."""
    out: dict[int, int] = {}
    for v in mla_local_ranks(hf, tp_size).values():
        r = int(v)
        hs = mla_padded_head_size(hf, r, tp_size)
        out[hs] = max(out.get(hs, 0), r)
    return out


class QwenMLATritonImpl(TritonMLAImpl):
    """TritonMLAImpl with the rms divisor wired into decode and k_norm into prefill."""

    # set by the model definition before the first forward
    rms_offset: int = -1
    mla_local_heads: int = 0        # rms scalars in the k_pe tail = this rank's LOCAL heads
    mla_rope_dim: int = 0         # per-head rope width: 64, or 58 under partial rope
    mla_rope_group: int = 0       # query heads per KV head (6)
    mla_head_offset: int = 0      # this rank's first GLOBAL head index
    mla_rope_first: int = 0       # first GLOBAL rope group this rank caches
    mla_packed_rope: int = 0      # width of the packed rope block (256), EXCLUDING rms + pad

    def _mla_kernel(self, *args, **kwargs):
        kwargs["rms_offset"] = self.rms_offset
        return mla_decode_attention_fwd(*args, **kwargs)

    def _mla_group_of_local_head(self, n_local: int, device) -> torch.Tensor:
        """Rope group owned by each rank-local head. See TP CORRECTNESS in the module docstring."""
        gh = self.mla_head_offset + torch.arange(n_local, device=device)
        # Relative to the FIRST cached group: the cache holds only this rank's window, so a
        # global index would run off the end of it and read the rms tail as rope.
        return gh // self.mla_rope_group - self.mla_rope_first

    def _concat_k_nope_k_pe(self, k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        """Assemble the PREFILL key at stock width: [ k_nope 192 | this head's rope 64 ] / rms.

        k_pe arrives from the cache as [ packed rope 256 | rms 24 ] because the rms tail rides
        inside the declared qk_rope_head_dim -- so prefill gets the divisor for free, with no
        extra plumbing and no risk of it drifting out of sync with the cache.

        Selecting each head's own rope group here (rather than broadcasting all 256, which is
        what the parent does) is what keeps the key 256 wide and inside FLASH_ATTN's supported
        (192, 64, 256).

        (1+w) is NOT applied here: it is already folded into kv_b_proj's K rows, so k_nope
        arrives pre-scaled. Applying it again would square it. Only the division by rms remains.
        """
        if self.rms_offset < 0:
            return super()._concat_k_nope_k_pe(k_nope, k_pe)

        t, n_local = k_nope.shape[0], k_nope.shape[-2]
        # Slice at EXPLICIT offsets, never from the end: the row is padded past the rms tail so
        # that page sizes unify by block size (see mla_padded_head_size), and measuring back
        # from k_pe.shape[-1] would read padding as rms.
        n_rope = self.mla_packed_rope
        flat = k_pe.reshape(t, -1)
        g = self._mla_group_of_local_head(n_local, k_nope.device)

        rope = flat[:, :n_rope].view(t, -1, self.mla_rope_dim)[:, g, :]   # [t, local, 64|58]
        rms = flat[:, n_rope:n_rope + n_local]        # [t, local] -- tail is this rank's heads

        k = torch.empty((t, n_local, k_nope.shape[-1] + self.mla_rope_dim),
                        dtype=k_nope.dtype, device=k_nope.device)
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1]:] = rope.to(k_nope.dtype)
        return k / rms.unsqueeze(-1).to(k.dtype).clamp(min=1e-6)

    def _mla_narrow_q(self, q: torch.Tensor) -> torch.Tensor:
        """Narrow q from the packed 280-wide pe to this head's own 64.

        q is [ nope 192 | packed rope 256 | rms pad 24 ]. Every slot outside the head's own rope
        group is zero by construction, so this drops nothing -- it just stops handing the backend
        448 dims of which 192 are structurally zero.
        """
        t, n_local = q.shape[0], q.shape[-2]
        nope = q[..., : self.qk_nope_head_dim]
        pe = q[..., self.qk_nope_head_dim:]
        g = self._mla_group_of_local_head(n_local, q.device)
        h = torch.arange(n_local, device=q.device)
        packed = pe[..., : self.mla_packed_rope]      # explicit: rms + row padding follow it
        sel = packed.view(t, n_local, -1, self.mla_rope_dim)[:, h, g, :]  # [t, local, 64|58]
        return torch.cat([nope, sel], dim=-1)

    def forward_mha(self, q: torch.Tensor, *args, **kwargs):
        if self.rms_offset < 0:
            return super().forward_mha(q, *args, **kwargs)
        q = self._mla_narrow_q(q)
        # Partial rope makes qk_nope_head_dim 198, so the `v` the parent splits out of the fused
        # kv_b projection starts 396 bytes in -- 12 (mod 16) -- and the trtllm-gen prefill kernel
        # rejects the TMA descriptor built on it. Only reached when nope is not 8-aligned, so a
        # full-rope deployment runs the unpatched backend exactly as before. See prefill_align.
        if self.qk_nope_head_dim % 8:
            md = kwargs.get("attn_metadata") or args[3]
            if md.prefill is not None and md.prefill.prefill_backend is not None:
                patch_prefill_backend(md.prefill.prefill_backend)
        # The chunked-CONTEXT gather is a CUDA kernel that hard-rejects entry sizes outside
        # DeepSeek's 320/576, so every prefill with cached context dies on our rows. Swap in the
        # torch equivalent for the duration of the call (verified bit-identical to the kernel at
        # 576 -- test_gather_torch.py). Patch the module attribute, not a local import: the
        # parent resolves `ops.gather_and_maybe_dequant_cache` at call time.
        import vllm._custom_ops as _ops
        orig = _ops.gather_and_maybe_dequant_cache
        _ops.gather_and_maybe_dequant_cache = gather_cache_torch
        try:
            return super().forward_mha(q, *args, **kwargs)
        finally:
            _ops.gather_and_maybe_dequant_cache = orig

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        # QWEN_MLA_DECODE_BACKEND=flashinfer routes decode to the patched flashinfer fa2 MLA kernel
        # (~1.9x the Triton fork on the attention term -- docs/06-flashinfer-path.md). Falls
        # through to Triton for anything that path does not cover, so the fast path never has to
        # guess: multi-token decode rows and the non-causal DSpark block both keep the fork.
        if not (fi_decode.enabled() and self.rms_offset >= 0 and attn_metadata.causal
                and attn_metadata.num_decode_tokens == attn_metadata.num_decodes):
            fi_decode._dbg("gate failed", enabled=fi_decode.enabled(), rms=self.rms_offset >= 0,
                                causal=getattr(attn_metadata, "causal", None),
                                ndt=getattr(attn_metadata, "num_decode_tokens", None), nd=getattr(attn_metadata, "num_decodes", None))
        if (fi_decode.enabled() and self.rms_offset >= 0
                and attn_metadata.causal
                and attn_metadata.num_decode_tokens == attn_metadata.num_decodes):
            out = fi_decode.forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata,
                                             layer)
            if out is not None:          # None = layout the fast path declined; use Triton
                return out
        # Swap the module-global the parent's body resolves at call time. Patching the module
        # attribute (not re-importing the name) is what makes the substitution take effect --
        # `from x import f` in the parent would bind the original at import time.
        import vllm.v1.attention.backends.mla.triton_mla as _t
        q0 = q[0] if isinstance(q, tuple) else q
        B, H, lse_dim = q0.shape[0], q0.shape[1], self.kv_lora_rank + 1
        orig, orig_splits = _t.decode_attention_fwd, _t._compute_num_kv_splits
        _t.decode_attention_fwd = self._mla_kernel
        _t._compute_num_kv_splits = lambda msl, sm: splitk_cap(B, H, lse_dim, orig_splits(msl, sm))
        try:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        finally:
            _t.decode_attention_fwd, _t._compute_num_kv_splits = orig, orig_splits


class MLAMetadata(MLACommonMetadata):
    """MLACommonMetadata without the DeepSeek-only head-size guard.

    __post_init__ rejects any head_dim outside [320, 576] -- DeepSeek's 256+64 and 512+64. Our
    cache rows are kv_lora_rank + 280, i.e. 1048 / 1536 / 2072 across the allocation, so the
    guard fires on every layer. It is the field's ONLY consumer (nothing else in vLLM reads
    metadata.head_dim -- checked), and the guard calls MLACommonBackend.supports_head_size by
    NAME rather than through cls, so overriding it on our backend has no effect. Dropping the
    check here is therefore the narrowest available intervention rather than a workaround.
    """

    def __post_init__(self):
        return


class MLATritonMetadataBuilder(TritonMLAMetadataBuilder):
    """Per-LAYER latent ranks, which vLLM's MLA metadata assumes is a single model-wide number.

    Two things break otherwise, both sized from the model config rather than from the layer:

      kv_lora_rank : get_mla_dims() reads hf_text_config.kv_lora_rank, which does not exist on a
                     Qwen config -- and could not, since our allocation gives layers 256, 768,
                     1792, ... The builder is constructed per KV-cache group, and every layer in
                     a group shares a head_size, so the rank is recoverable exactly from the
                     group's own spec. It feeds lse_dim for the split-KV decode accumulator.

      workspace    : the chunked-prefill scratch is allocated model_config.get_head_size() wide.
                     For a non-DeepSeek model_type that returns head_dim (256), but the context
                     gather writes whole CACHE ROWS into it (kv_lora_rank + 280 = 1048 here) and
                     MLACommonImpl then splits them on exactly that boundary. Too narrow by 4x,
                     and varying per group, so no single model-level value can be right.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        import torch as _torch
        hf = vllm_config.model_config.hf_text_config
        # Rows are padded for page unification, so head_size - tail OVERSTATES the rank. Look it
        # up from the allocation instead; the group's head_size identifies it exactly.
        tp = vllm_config.parallel_config.tensor_parallel_size
        rank = mla_rank_by_head_size(hf, tp).get(kv_cache_spec.head_size)
        if rank is None:
            raise ValueError(f"no MLA rank maps to head_size {kv_cache_spec.head_size} "
                             f"(known: {sorted(mla_rank_by_head_size(hf, tp))})")

        had = hasattr(hf, "kv_lora_rank")
        prev = getattr(hf, "kv_lora_rank", None)
        hf.kv_lora_rank = rank                    # read by get_mla_dims during super().__init__
        try:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        finally:
            if had:
                hf.kv_lora_rank = prev
            else:
                try:
                    delattr(hf, "kv_lora_rank")
                except AttributeError:
                    pass

        self.metadata_cls = MLAMetadata

        w = self.chunked_prefill_workspace
        rows, dt, dev = chunked_workspace_rows(vllm_config), w.dtype, w.device
        del w
        self.chunked_prefill_workspace = None            # release the parent's before allocating
        self.chunked_prefill_workspace_size = rows
        self.chunked_prefill_workspace = _torch.empty((rows, kv_cache_spec.head_size), dtype=dt, device=dev)

        # ---- decode context parallelism ------------------------------------------------
        # The config gates are removable (see qwen_mla/__init__.py), and with them removed
        # DCP RUNS -- but it is numerically wrong, so refuse it rather than serve it.
        # DCP shards the cache by token and all-gathers q, so every rank must serve all
        # GLOBAL heads for the tokens it owns. Our cache row cannot: it stores only the rope
        # groups this TP rank's heads span and one rms slot per LOCAL head. Rank 0 asked for
        # head 20's rope reads groups 0,1 and its rms reads page padding. The output stays
        # fluent -- measured 56.2 ms/step and coherent text that quietly diverges from the
        # non-DCP run -- which is exactly why this raises instead of warning.
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "MLA does not support decode context parallelism yet. DCP requires every rank "
                "to serve all global heads for its token shard, but the cache row stores only "
                "this rank's rope window and its local heads' rms. Fixing it means caching all "
                "4 rope groups and all global rms slots (tail 140 -> 280), computing rms for "
                "every head on every rank, and planning the decode kernel at the all-gathered "
                "head count. See docs/03-ruled-out.md item 7."
            )

        # ---- flashinfer decode path -------------------------------------------------------
        # Planning must happen here, not in forward_mqa: it is host-side (D2H copies) and
        # forward_mqa runs INSIDE the captured CUDA graph. build() runs outside it, once per
        # step, and one builder exists per KV-cache group, so a step plans once per rank.
        self._mla_fi_plan = None
        self._mla_fi_ok = False
        # The CACHED rope width per KV group: 58 under partial rope (mla_qk_rope_head_dim), 64 for
        # full rope. Deriving it from partial_rotary_factor alone gave 64 for partial-rope students,
        # the plan's window (256) never matched the decode call's (232), and every layer silently
        # fell back to Triton.
        rope_dim = int(getattr(hf, "mla_qk_rope_head_dim", 0)
                       or hf.head_dim * getattr(hf, "partial_rotary_factor", 0.25))
        n_local = hf.num_attention_heads // tp
        rope_group = hf.num_attention_heads // hf.num_key_value_heads
        head_offset = get_tensor_model_parallel_rank() * n_local
        off, win = fi_decode.rope_window(head_offset, n_local, rope_group, rope_dim)
        itemsize = _torch.tensor([], dtype=kv_cache_spec.dtype).element_size()
        page = kv_cache_spec.block_size
        # flashinfer's rope tile count must divide by the q*k D split (up to 4 x 16-wide MMA tiles), so
        # head_dim_kpe is padded to a multiple of 64: 232 / 116 (partial rope at TP=1 / 2) -> 256 / 128,
        # which ends exactly at the end of the cache row. The extra columns are the rms scalars that
        # follow the rope block (finite) and meet zero-padded q, so the scores are unchanged. The rms
        # tail offset (win) stays the true width.
        kpe = -(-win // 64) * 64
        # The kernel stages the rms tail with 16-byte cp.async; a start that is only 8-byte aligned
        # (partial rope at TP=2: rank + 116) is handled in-kernel by RMS_SHIFT. Anything coarser is not.
        rms_aligned = ((rank + win) * itemsize) % 8 == 0
        if (fi_decode.cp_async_safe(kv_cache_spec.head_size, page, rank, off, itemsize)
                and rank + off + kpe <= kv_cache_spec.head_size and rms_aligned):
            self._mla_fi_ok = True
            sched = vllm_config.scheduler_config
            # Speculative decoding puts ONE DECODE ROW PER PROPOSED TOKEN, so the decode
            # batch is seqs*(1+k). Sized at max_num_seqs the plan rejects every spec step and
            # silently falls back to Triton -- which measured 2x SLOWER end to end than not
            # speculating at all (629.7 vs 1254.8 tok/s at 32 concurrent).
            spec = getattr(vllm_config, "speculative_config", None)
            qmax = 1 + (getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 1
            max_batch = sched.max_num_seqs * qmax
            max_pages = max_batch * (
                -(-vllm_config.model_config.max_model_len // page) + 1)
            self._mla_fi_cfg = dict(device=device, rank=rank, win=win, kpe=kpe, n_local=n_local, spec=qmax > 1,
                                     page_size=page, max_batch=max_batch,
                                     max_pages=max_pages)
            self._mla_fi_scale = (hf.head_dim ** -0.5)
            self._mla_fi_qdtype = vllm_config.model_config.dtype
            self._mla_fi_kvdtype = kv_cache_spec.dtype

    def _reserve_attn_logits_workspace(self) -> None:
        # The parent reserves the split-K accumulator at max_num_seqs x max-context splits; see
        # splitk_cap for why that cannot stand. forward_mqa caps its splits by the same budget.
        from vllm.v1.attention.backends.mla.triton_mla import _compute_num_kv_splits
        from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized
        from vllm.platforms import current_platform
        if not is_workspace_manager_initialized():
            return
        B = self.vllm_config.scheduler_config.max_num_seqs
        if getattr(self, "non_causal_multi_token_decode", False):
            B *= self.reorder_batch_threshold
        H = self.num_heads * self.dcp_world_size
        splits = _compute_num_kv_splits(self.model_config.max_model_len, current_platform.num_compute_units())
        n = splitk_reserve_bytes(B, H, self.mla_dims.kv_lora_rank + 1, splits)
        current_workspace_manager().get_simultaneous(((n // 4,), torch.float32))

    def build(self, *args, **kwargs):
        # Unaligned context chunks (see chunked_workspace_rows); the parent hard-codes alignment.
        import vllm.model_executor.layers.attention.mla_attention as _m
        orig = _m.build_mla_chunked_context_metadata
        _m.build_mla_chunked_context_metadata = lambda **kw: orig(**{**kw, "align_chunk_to_block": False})
        try:
            md = super().build(*args, **kwargs)
        finally:
            _m.build_mla_chunked_context_metadata = orig
        common = kwargs.get("common_attn_metadata", args[1] if len(args) > 1 else None)
        fi_decode.plan_for_step(self, md, common)
        return md


class QwenMLATritonBackend(TritonMLABackend):
    @staticmethod
    def get_name() -> str:
        return "QWEN_MLA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[QwenMLATritonImpl]:
        return QwenMLATritonImpl

    @staticmethod
    def get_builder_cls() -> type[MLATritonMetadataBuilder]:
        return MLATritonMetadataBuilder
