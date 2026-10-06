"""Qwen3.5/3.8 + MLA on the stock RoPE base, with ABSORBED attention and a latent KV cache.

Milestone 2. The sibling qwen3_5_mla_materialized.py reconstructs 24 K/V heads per token and caches
12,288 dims/token/layer -- 6x MORE than the base model it compresses (vLLM measured 3.95x
concurrency at 131k against the base's 23.43x). This caches

    [ latent 768 | rope 4x64 | rms 24 ] = 1,048 dims/token/layer

which is 11.7x smaller than the materialised path and 1.95x smaller than the base, finally
matching the 2.000x the conversion claims.

WE DO OUR OWN ROTARY. vLLM's MLA wrapper rotates only the leading qk_rope_head_dim of k_pe.
Our four per-KV-head rope keys are packed into one 256-wide k_pe, so vLLM would rotate group 0
and leave groups 1-3 unrotated -- silently, and only for 3/4 of the heads. rotary_emb is
therefore left out of MLAModules and applied here, per group, before the tensors are handed over.

ORDER OF OPERATIONS IS LOAD-BEARING. k_norm scales the assembled key by (1+w) and THEN rope is
applied, so w_rope must multiply k_rope BEFORE rotation -- rotation mixes dimensions, so a
per-dim scale does not commute with it. The 1/rms factor is a scalar and does commute, which is
what lets it move to the logits (decode) or the assembled key (prefill).

(1+w_nope) is folded into kv_b_proj's K rows once, at build time. Prefill reads kv_b_proj
directly and vLLM derives the decode-time W_UK_T from the same weight, so both paths get it and
neither applies it twice. rms needs the RAW up-projection, so an unscaled copy is kept.
"""
from __future__ import annotations

import re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope

from .partial_rope import patch_partial_rope, rope_keep_from_config

from .mla_impl import QwenMLATritonBackend, chunked_workspace_rows, mla_gla_split, mla_padded_head_size


class Qwen3_5MLAAttention(nn.Module):
    def __init__(self, config, kv_lora_rank: int, cache_config=None, quant_config=None,
                 prefix: str = "") -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp == 0
        self.num_heads = self.total_num_heads // tp
        self.head_dim = config.head_dim or (config.hidden_size // self.total_num_heads)
        self.num_kv_heads = config.num_key_value_heads
        # GLA per-group cache: when the latent groups align with TP, this rank's heads read only
        # their group's slice of the latent, so the rank projects and caches just that slice.
        self.global_kv_lora_rank = kv_lora_rank
        self.latent_slice = None
        g = mla_gla_split(config, tp)
        if g > 1:
            tp_rank = get_tensor_model_parallel_rank()
            group = tp_rank // (tp // g)
            heads_per_group = self.total_num_heads // g
            first, last = tp_rank * self.num_heads, (tp_rank + 1) * self.num_heads - 1
            if first // heads_per_group != group or last // heads_per_group != group:
                raise RuntimeError(f"GLA: rank {tp_rank} heads {first}-{last} span groups; cannot shard")
            rg = kv_lora_rank // g
            self.latent_slice = slice(group * rg, (group + 1) * rg)
            kv_lora_rank = rg
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.head_dim ** -0.5

        rp = getattr(config, "rope_parameters", None) or {}
        prf = rp.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.25))
        # PARTIAL RoPE. `mla_rope_keep` names the 2-D frequency subspaces that still rotate;
        # the rest were permuted to the front of the nope block at export and ride in the latent
        # un-rotated. This is what makes the cache tail a multiple of 128 -- 4*58 + 24 rms = 256
        # -- so ranks 256/768/1792 are legal for FlashInfer's MLA kernel AND land exactly on
        # rows 512/1024/2048 (2.00x at TP=1, no padding). Absent => full rope, as before.
        self.model_rope_dim = int(self.head_dim * prf)    # 64, the BASE rotary width
        self.rope_keep = rope_keep_from_config(config)
        self.rope_dim = (self.model_rope_dim if self.rope_keep is None
                         else 2 * len(self.rope_keep))    # 64, or 58 under partial rope
        self.nope_dim = self.head_dim - self.rope_dim     # 192, or 198
        self.rope_group = self.total_num_heads // self.num_kv_heads
        # rms rides in the k_pe tail so vLLM's compiled cache-write moves it for us
        # Cache only the rope groups THIS RANK's heads span (mla_rope_span). k_rope_proj is
        # replicated so all four are computed, but rank r reads only its own -- at TP=2 that
        # halves the rope block (256 -> 128) and, with the tail at 140, makes the padded row a
        # power of two (512/1024/2048). That is what lets the MTP draft's 1024 dims/token divide
        # the row, and it also brings every row to 16-byte alignment so no layer falls back off
        # the flashinfer path. Mean row 1048 -> 1024, i.e. cheaper as well.
        self.head_offset = get_tensor_model_parallel_rank() * self.num_heads
        self.rope_first = self.head_offset // self.rope_group
        _last = (self.head_offset + self.num_heads - 1) // self.rope_group
        self.rope_span = _last - self.rope_first + 1
        self.packed_rope = self.rope_span * self.rope_dim             # 128 at TP=2
        self.rms_offset = self.kv_lora_rank + self.packed_rope
        # Declared tail = rope + rms + PAGE PADDING. The padding is inert (nothing reads past
        # rms_offset + num_heads) and exists only so this layer's cache page divides the
        # largest layer's -- without it vLLM pads every row to the widest and the served
        # compression collapses to 1x. See mla_padded_head_size.
        self.head_size = mla_padded_head_size(config, kv_lora_rank, tp)
        self.declared_rope = self.head_size - self.kv_lora_rank
        self.rope_pad = self.declared_rope - self.packed_rope - self.num_heads
        assert self.rope_pad >= 0, f"padded head_size {self.head_size} too small"

        self.q_proj = ColumnParallelLinear(config.hidden_size,
                                           self.total_num_heads * self.head_dim * 2,
                                           bias=False, quant_config=quant_config,
                                           prefix=f"{prefix}.q_proj")
        self.kv_a_proj = ReplicatedLinear(config.hidden_size, kv_lora_rank, bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.kv_a_proj")
        if self.latent_slice is not None:
            # The checkpoint holds the full [r, hidden] projection; keep this group's rows. (The latent
            # projections are excluded from quantization, so this is a plain bf16 weight.)
            if not hasattr(self.kv_a_proj, "weight") or self.kv_a_proj.weight.shape[0] != kv_lora_rank:
                raise NotImplementedError("GLA latent sharding needs an unquantized kv_a_proj")
            sl = self.latent_slice

            def _load_group_rows(param, loaded_weight, *args, **kwargs):
                param.data.copy_(loaded_weight[sl].to(param.dtype))
            self.kv_a_proj.weight.weight_loader = _load_group_rows
        self.k_rope_proj = ReplicatedLinear(config.hidden_size,
                                            self.num_kv_heads * self.rope_dim, bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.k_rope_proj")
        # kv_b_proj: latent -> per-head [k_nope | v]. vLLM splits and derives W_UK_T/W_UV.
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, self.total_num_heads * (self.nope_dim + self.head_dim),
            bias=False, quant_config=quant_config, prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size, bias=False,
                                        quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.eps = config.rms_norm_eps
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rp_clean = {k: v for k, v in rp.items() if not k.startswith("mrope")}
        rp_clean.setdefault("partial_rotary_factor", prf)
        if self.rope_keep is not None:
            # Request the NARROWER width, then rewrite the table: get_rope would otherwise
            # derive base^(-2i/58) rather than the kept subspaces' original base^(-2i/64) --
            # wrong angles that still produce fluent text.
            rp_clean["partial_rotary_factor"] = self.rope_dim / self.head_dim
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
                                   is_neox_style=True, rope_parameters=rp_clean)
        # M-RoPE (multimodal serving): positions arrive as [3, T] (time, height, width) and each
        # rotary frequency takes ONE of the three, by the teacher's own section layout. Text tokens
        # have t == h == w, so this matters only for image/video tokens. The table maps each column
        # of our (kept-frequency) cos/sin cache to its section, by ORIGINAL frequency index.
        self.mrope_section = rp.get("mrope_section")
        if self.mrope_section:
            s0, s1, s2 = self.mrope_section
            inter = bool(rp.get("mrope_interleaved", False))

            def _sec(f):                       # = vLLM's apply_interleaved_rope / chunked split
                if inter:
                    return 1 if (f % 3 == 1 and f < 3 * s1) else (2 if (f % 3 == 2 and f < 3 * s2) else 0)
                return 0 if f < s0 else (1 if f < s0 + s1 else 2)
            keep = self.rope_keep if self.rope_keep is not None else list(range(self.model_rope_dim // 2))
            self.register_buffer("_mrope_col_sec", torch.tensor([_sec(int(f)) for f in keep] * 2),
                                 persistent=False)
        if self.rope_keep is not None:
            patch_partial_rope(self.rotary_emb, self.rope_keep, self.model_rope_dim,
                               float(rp_clean.get("rope_theta", 1e7)))

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads, scale=self.scaling,
            qk_nope_head_dim=self.nope_dim, qk_rope_head_dim=self.declared_rope,
            v_head_dim=self.head_dim, q_lora_rank=None, kv_lora_rank=kv_lora_rank,
            kv_b_proj=self.kv_b_proj, cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn", attn_backend=QwenMLATritonBackend)
        # The profile run simulates the context up-projection at this many rows; match the builder.
        self.mla_attn._chunked_prefill_workspace_size = chunked_workspace_rows(self.mla_attn._vllm_config)
        self.mla_attn.impl.rms_offset = self.rms_offset
        self.mla_attn.impl.mla_local_heads = self.num_heads   # rms tail is rank-local
        self.mla_attn.impl.mla_rope_dim = self.rope_dim
        self.mla_attn.impl.mla_packed_rope = self.packed_rope
        self.mla_attn.impl.mla_rope_group = self.rope_group
        # kv_b_proj is ColumnParallel, so this rank owns GLOBAL heads [offset, offset+num_heads).
        # k_pe (rope + rms) is replicated in full, so both must be indexed globally.
        self.mla_attn.impl.mla_head_offset = self.head_offset
        self.mla_attn.impl.mla_rope_first = self.rope_first

    def register_raw_up(self, k_up_weight: torch.Tensor) -> None:
        """Keep the UNSCALED k_up_proj for computing rms.

        kv_b_proj carries (1+w_nope) folded in, so it cannot be used here: rms is the norm of
        the RAW assembled key, and using the scaled weight would fold w in twice.

        Kept in the model dtype, not fp32: the rms this feeds is cast to bf16 to ride in the
        cache tail, so an fp32 weight buys precision that is immediately discarded while
        forgoing tensor cores and doubling the weight read.
        """
        self.register_buffer(
            "_k_up_raw",
            k_up_weight.detach().to(self.kv_a_proj.weight.dtype),
            persistent=False,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        nh, hd, rd = self.num_heads, self.head_dim, self.rope_dim
        w = (1.0 + self.k_norm.weight.float())            # Qwen3_5RMSNorm is (1 + weight)
        # Rope group per RANK-LOCAL head, by global index. k_rope_proj is Replicated (all 4
        # groups on every rank) while the q heads are sharded, so this is the one mapping that
        # ties them together; it is used for rms, for the q slot, and by the impl for prefill.
        g_of_head = (self.head_offset + torch.arange(nh, device=hidden_states.device)) \
            // self.rope_group

        qg, _ = self.q_proj(hidden_states)
        q, gate = torch.chunk(qg.view(n, nh, hd * 2), 2, dim=-1)
        q = self.q_norm(q)

        c, _ = self.kv_a_proj(hidden_states)               # [n, r] -- the cached latent
        kr_raw, _ = self.k_rope_proj(hidden_states)
        kr_raw = kr_raw.view(n, self.num_kv_heads, rd).float()

        # rms(key_h) from the RAW up-projection and RAW rope key, per (token, head).
        # A norm splits over the concatenation, so the rope and nope halves are reduced
        # separately and added -- no [n, nh, rd] gather and no [n, nh, head_dim] concat, both of
        # which were pure materialisation. The rope sum is [n, num_kv_heads] and only then
        # expands to heads. Measured 47.4 -> 15.9 us/layer, 0.76 -> 0.25 ms over 16 layers.
        # The up-projection runs in bf16, not fp32: rms is CAST TO BF16 to ride in the cache
        # tail a few lines below, so the extra precision is discarded anyway -- and fp32 here
        # gets no tensor cores (the run log even warns TF32 is available but off). The
        # reductions stay in fp32, which is where a norm actually needs the range.
        # 20.8 -> 8.6 us/layer.
        k_nope_raw = torch.nn.functional.linear(c, self._k_up_raw).view(n, nh, self.nope_dim)
        rope_sq = kr_raw.pow(2).sum(-1)                                # [n, num_kv_heads], fp32
        sumsq = k_nope_raw.float().pow(2).sum(-1) + rope_sq[:, g_of_head]   # [n, nh]
        rms = sumsq.div(hd).add(self.eps).sqrt()                       # [n, nh]

        # ONE rotary call for everything. vLLM rotates the leading rotary_dim of EVERY head, so
        # laying the rope block at the head-local offset 0 rotates all 24 query heads and all 4
        # rope keys in a single call -- no per-head loop, and no risk of rotating only group 0
        # the way passing a packed 256-wide k_pe to vLLM's MLA wrapper would.
        # new_empty, not new_zeros: rotary only touches the leading rotary_dim and passes the
        # rest through, and both results are sliced back to [..., :rd] immediately below, so the
        # tail is never read. Zeroing it was 256 dims written per head to use 64.
        if positions.ndim == 2:                            # M-RoPE positions [3, T]
            q_rot, k_rot = self._mrope_rotate(positions, q[..., :rd],
                                              (kr_raw * w[:rd]).to(q.dtype))
        else:
            q_pad = q.new_empty(n, nh, hd); q_pad[..., :rd] = q[..., :rd]
            k_pad = q.new_empty(n, self.num_kv_heads, hd)
            k_pad[..., :rd] = (kr_raw * w[:rd]).to(q.dtype)   # w BEFORE rotation: rotation mixes dims
            q_rot, k_rot = self.rotary_emb(positions, q_pad.reshape(n, -1), k_pad.reshape(n, -1))
            q_rot = q_rot.view(n, nh, hd)[..., :rd]
            k_rot = k_rot.view(n, self.num_kv_heads, hd)[..., :rd]

        # q = [ nope (absorbed by vLLM via W_UK_T) | packed rope in this head's group slot | pad ]
        # The slot is chosen by GLOBAL head index: under TP this rank's local head i is global
        # head head_offset+i, so a local-index slice would put rank 1's heads in groups 0,1
        # instead of 2,3 -- wrong keys, but still fluent output, so nothing would flag it.
        q_pe = q.new_zeros(n, nh, self.declared_rope)
        slot = ((g_of_head - self.rope_first)[:, None] * rd
                + torch.arange(rd, device=q.device)[None, :])
        q_pe.scatter_(2, slot.unsqueeze(0).expand(n, -1, -1), q_rot)
        k_win = k_rot[:, self.rope_first:self.rope_first + self.rope_span]
        parts = [k_win.reshape(n, self.packed_rope), rms.to(k_rot.dtype)]
        if self.rope_pad:                                  # page-alignment padding, never read
            parts.append(k_rot.new_zeros(n, self.rope_pad))
        k_pe = torch.cat(parts, dim=-1).unsqueeze(1)

        attn_out = self.mla_attn(torch.cat([q[..., rd:], q_pe], dim=-1), c, k_pe,
                                 output_shape=(n, nh * hd))
        attn_out = attn_out * torch.sigmoid(gate.reshape(n, -1))
        out, _ = self.o_proj(attn_out)
        return out


    def _mrope_rotate(self, positions, q_r, k_r):
        """Neox rotation of [n, heads, rd] q and k with per-frequency (t, h, w) positions."""
        if not self.mrope_section:
            raise ValueError("3-row (M-RoPE) positions but the config has no mrope_section")
        n, rd = q_r.shape[0], q_r.shape[-1]
        cs = self.rotary_emb.cos_sin_cache.to(q_r.dtype)[positions]           # [3, n, rd]
        cs = cs.gather(0, self._mrope_col_sec.view(1, 1, rd).expand(1, n, rd))[0]
        cos, sin = cs.chunk(2, dim=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]

        def rot(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rot(q_r), rot(k_r)

    def build_kv_b_from_up_projections(self, k_up: torch.Tensor, v_up: torch.Tensor) -> None:
        """kv_b_proj = per-head [ (1+w_nope) * W_UK ; W_UV ], stacked over heads.

        vLLM's MLA splits kv_b_proj into W_UK / W_UV and derives the decode-time W_UK_T from it,
        so folding (1+w_nope) here reaches BOTH the prefill reconstruction and the absorbed
        decode query -- once, in one place. The raw k_up is retained separately for rms.

        Layout must be head-major [head][nope+v_head_dim, r]; interleaving them the other way
        loads without error and silently permutes which head sees which projection.

        k_up/v_up arrive UNSHARDED (all heads), but kv_b_proj is ColumnParallel and holds only
        this rank's heads, so the head slice happens here. It is not vLLM's usual sharded load:
        we are writing the weight ourselves, so nothing else would apply it.
        """
        nope, hd, r = self.nope_dim, self.head_dim, self.kv_lora_rank
        tot, nh = self.total_num_heads, self.num_heads
        R = self.global_kv_lora_rank
        sl = slice(self.head_offset, self.head_offset + nh)
        # The checkpoint iterator yields CPU tensors; k_norm/kv_b_proj already live on device.
        # Slice to this rank's heads BEFORE the transfer -- at TP=2 that halves what moves.
        dev = self.kv_b_proj.weight.device
        k = k_up.reshape(tot, nope, R)[sl].to(dev).float()
        v = v_up.reshape(tot, hd, R)[sl].to(dev).float()
        if self.latent_slice is not None:
            # GLA: these heads must read only their group's latent columns. Verify the block
            # structure before dropping the rest -- a checkpoint that is not block-diagonal would
            # otherwise lose real weight mass silently.
            ls = self.latent_slice
            on = k[..., ls].abs().sum() + v[..., ls].abs().sum()
            off = (k.abs().sum() + v.abs().sum()) - on
            if off > 1e-4 * on:
                raise RuntimeError(f"GLA: up-projections are not block-diagonal for heads {sl} "
                                   f"(off-group mass {float(off):.3e} vs {float(on):.3e})")
            k, v = k[..., ls].contiguous(), v[..., ls].contiguous()
        w_nope = (1.0 + self.k_norm.weight.float())[self.rope_dim:]      # [nope]
        kw = k * w_nope[None, :, None]
        merged = torch.cat([kw, v], dim=1).reshape(nh * (nope + hd), r)
        if merged.shape != tuple(self.kv_b_proj.weight.shape):
            raise RuntimeError(f"kv_b_proj shard mismatch: built {tuple(merged.shape)} vs "
                               f"{tuple(self.kv_b_proj.weight.shape)} (heads {sl})")
        with torch.no_grad():
            self.kv_b_proj.weight.copy_(merged.to(self.kv_b_proj.weight.dtype))
        self.register_raw_up(k.reshape(nh * nope, r))


# ---------------------------------------------------------------------------------------------
# Decoder / model / ForCausalLM wrappers. Same module-global swap trick as the materialised
# sibling: Qwen3_5Model builds layers from the MODULE-GLOBAL Qwen3_5DecoderLayer and
# Qwen3_5ForCausalLM builds Qwen3_5Model by name, so subclassing alone silently constructs stock
# layers while logging our architecture name.
# ---------------------------------------------------------------------------------------------
from vllm.model_executor.models.qwen3_5 import (Qwen3_5DecoderLayer,     # noqa: E402
                                                Qwen3_5ForCausalLM, Qwen3_5Model)
from vllm.model_executor.models.utils import (WeightsMapper,             # noqa: E402
                                              extract_layer_index)

_BASE_STACKED = {k: v for k, v in Qwen3_5Model.hf_to_vllm_mapper.orig_to_new_stacked.items()
                 if k not in (".q_proj", ".k_proj", ".v_proj")}
# k_up_proj / v_up_proj are CONSUMED into kv_b_proj at load time, so they are mapped to None
# rather than left to fail as unexpected keys.
_ABSORBED_MAPPER = WeightsMapper(orig_to_new_substr={"language_model.": ""},
                                 orig_to_new_stacked=_BASE_STACKED)


class Qwen3_5MLADecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "", *,
                 ranks: dict[int, int] | None = None) -> None:
        super().__init__(vllm_config, layer_type=layer_type, prefix=prefix)
        if layer_type == "full_attention":
            idx = extract_layer_index(prefix)
            cfg = vllm_config.model_config.hf_text_config
            vllm_config.compilation_config.static_forward_context.pop(
                f"{prefix}.self_attn.attn", None)
            del self.self_attn
            self.self_attn = Qwen3_5MLAAttention(
                cfg, kv_lora_rank=(ranks or {})[idx],
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.self_attn")


class Qwen3_5MLAModel(Qwen3_5Model):
    hf_to_vllm_mapper = _ABSORBED_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        import vllm.model_executor.models.qwen3_5 as _q35
        cfg = vllm_config.model_config.hf_text_config
        types = getattr(cfg, "layer_types", None) or []
        full_idx = [i for i, t in enumerate(types) if t == "full_attention"]
        raw = getattr(cfg, "mla_ranks", None)
        if not raw:
            raise RuntimeError("checkpoint config has no `mla_ranks`")
        ranks = {int(k): int(v) for k, v in raw.items()}
        if sorted(ranks) != sorted(full_idx):
            raise RuntimeError(f"mla_ranks {sorted(ranks)} != full-attn {sorted(full_idx)}")

        def _factory(vllm_config, layer_type, prefix=""):
            return Qwen3_5MLADecoderLayer(vllm_config, layer_type=layer_type,
                                                   prefix=prefix, ranks=ranks)
        orig = _q35.Qwen3_5DecoderLayer
        _q35.Qwen3_5DecoderLayer = _factory
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5DecoderLayer = orig


class Qwen3_5MLAForCausalLM(Qwen3_5ForCausalLM):
    hf_to_vllm_mapper = _ABSORBED_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        import vllm.model_executor.models.qwen3_5 as _q35
        orig = _q35.Qwen3_5Model
        _q35.Qwen3_5Model = Qwen3_5MLAModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5Model = orig

    def load_weights(self, weights):
        """Consume k_up_proj / v_up_proj into kv_b_proj instead of loading them directly."""
        # Keyed by (layer index, kind) parsed from the name rather than by a full name built
        # from an assumed prefix: this checkpoint's tensors are model.language_model.layers.N.*,
        # and matching "model.layers.N.*" found nothing while looking like a missing-weight bug.
        held: dict[tuple[int, str], torch.Tensor] = {}
        pat = re.compile(r"\.layers\.(\d+)\.self_attn\.(k_up_proj|v_up_proj)\.weight$")

        def _filter(ws):
            for name, w in ws:
                m = pat.search(name)
                if m:
                    held[(int(m.group(1)), m.group(2))] = w
                    continue
                yield name, w

        loaded = super().load_weights(_filter(weights))
        built = 0
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if not isinstance(attn, Qwen3_5MLAAttention):
                continue
            i = extract_layer_index(attn.mla_attn.layer_name.rsplit(".attn", 1)[0])
            k, v = held.get((i, "k_up_proj")), held.get((i, "v_up_proj"))
            if k is None or v is None:
                raise RuntimeError(f"layer {i}: missing k_up/v_up in the checkpoint "
                                   f"(held layers: {sorted({j for j, _ in held})}); "
                                   f"kv_b_proj cannot be built")
            attn.build_kv_b_from_up_projections(k.to(torch.float32), v.to(torch.float32))
            built += 1
        if built == 0:
            raise RuntimeError("no MLA layers received kv_b_proj weights")
        print(f"[qwen-mla] built kv_b_proj for {built} layers from k_up/v_up", flush=True)
        return loaded
