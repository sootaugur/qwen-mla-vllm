"""vLLM model definition for Qwen3.5/3.8 + MLA on the STOCK RoPE base (decoupled rope keys).

Sibling of qwen3_5_mla_nope.py, which is NoPE-only. That module omits rotary entirely and carries
no rope key, so it cannot represent this model: here the base keeps its rotary and each layer
holds a decoupled rope key alongside the latent.

WHY A SEPARATE CLASS RATHER THAN A FLAG. The two differ in module inventory (k_rope_proj
exists and is loaded rather than dropped), in k_up_proj's width (qk_nope_head_dim=192 per head,
not the full 256), and in the forward's op order. A flag threading through all three is how a
NoPE checkpoint ends up quietly served as a RoPE one -- the exact failure that made the phase-1
export bench as its own teacher. Two classes cannot be confused by a config typo.

ROPE MODE IS per_kv_head, AND THAT IS NOT A DEFAULT. Averaging the 4 KV heads' rope
projections into one shared key ('shared', the canon's function default) costs +0.4984 NLL on
this model at ZERO compression -- the mean is ~87% off each head it stands for. per_kv_head
copies each KV head's projection verbatim and measures +0.0019. This module therefore assumes
rope_kv_heads == num_key_value_heads and asserts it, rather than accepting a mode it cannot
serve correctly.

OP ORDER MIRRORS mla_retrofit.mla.MLAAttention EXACTLY, because divergence is silent:
    q_norm(q) -> reconstruct [k_rope ; k_nope] -> k_norm over the FULL 256 -> rotary -> attn
k_norm lands on the assembled key, before rotary. Normalising the nope part alone, or rotating
before normalising, changes the result while raising no error.

M-ROPE IS DROPPED, DELIBERATELY. The 27B ships as a VL checkpoint with
mrope_section=[11,11,10] and mrope_interleaved. M-RoPE computes freqs = inv_freq * pos with a
different pos per section; for TEXT-ONLY input all three sections share one position, so it
reduces exactly to standard rope regardless of how frequencies are partitioned or interleaved
among sections. The export drops mrope_section (vLLM's `uses_mrope` otherwise asserts), and
this module builds plain rope at rotary_dim = partial_rotary_factor * head_dim = 64. That
equivalence is an argument, not a measurement, so it MUST be checked against the HF path on the
same checkpoint before any number from here is trusted -- two engines on one checkpoint is what
caught the gamma=0 export bug.

MILESTONE 1: SPEED, NOT MEMORY -- same as the NoPE sibling. K/V are materialised per query
head and handed to ordinary paged attention, so the cache is 24 heads wide rather than the
latent. This buys vLLM's batching and kernels for benchmarking (our eval is ~65k prefill
against ~100 decode); it does not yet realise the cache saving.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope

from .partial_rope import patch_partial_rope, rope_keep_from_config
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    extract_layer_index,
    make_layers,
)

# Unlike the NoPE mapper, ".k_rope_proj." is KEPT -- here it is a real weight, not a
# zero-element artefact. Dropping it would leave the rope key randomly initialised and produce
# a model that loads cleanly and scores like noise.
_BASE_STACKED = {
    k: v
    for k, v in Qwen3_5Model.hf_to_vllm_mapper.orig_to_new_stacked.items()
    if k not in (".q_proj", ".k_proj", ".v_proj")
}
_ROPE_QWEN_MLA_MAPPER = WeightsMapper(
    orig_to_new_substr={"language_model.": ""},
    orig_to_new_stacked=_BASE_STACKED,
)


def _ranks_from_config(config, full_idx: list[int]) -> dict[int, int]:
    """Per-layer kv_lora_rank, read from the CHECKPOINT CONFIG rather than the environment.

    The NoPE sibling takes ranks from env because one registered arch served many allocations
    during the sweep. That is a liability here: vLLM v1 runs the engine in a separate process,
    and this project has already shipped a guard that consulted os.environ across that boundary
    and could therefore never fire. The export writes `mla_ranks` into config.json, which
    travels with the weights and cannot disagree with them.
    """
    raw = getattr(config, "mla_ranks", None)
    if not raw:
        raise RuntimeError(
            "checkpoint config has no `mla_ranks`; re-export with export_rope_mla_hf.py"
        )
    ranks = {int(k): int(v) for k, v in raw.items()}
    if sorted(ranks) != sorted(full_idx):
        raise RuntimeError(
            f"mla_ranks covers {sorted(ranks)} but full-attention layers are {sorted(full_idx)}"
        )
    return ranks



class Qwen3_5MLAMaterializedAttention(nn.Module):
    """Gated attention: low-rank latent for the nope dims + a decoupled per-KV-head rope key."""

    def __init__(
        self,
        config,
        kv_lora_rank: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp == 0
        self.num_heads = self.total_num_heads // tp
        self.head_dim = config.head_dim or (config.hidden_size // self.total_num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.head_dim**-0.5

        rope_params = getattr(config, "rope_parameters", None) or {}
        prf = rope_params.get("partial_rotary_factor",
                              getattr(config, "partial_rotary_factor", 0.25))
        self.model_rope_dim = int(self.head_dim * prf)
        # PARTIAL RoPE. `mla_rope_keep` lists the 2-D frequency subspaces that still rotate; the
        # rest were permuted to the front of the nope block at export and are carried by the
        # latent un-rotated. This is what makes the cache tail a multiple of 128
        # (4*58 + 24 rms = 256) so ranks 256/768/1792 are legal for FlashInfer's MLA kernel AND
        # land exactly on rows 512/1024/2048. Absent => full rope, original behaviour.
        self.rope_keep = rope_keep_from_config(config)
        if self.rope_keep is not None:
            self.qk_rope_head_dim = 2 * len(self.rope_keep)
        else:
            self.qk_rope_head_dim = self.model_rope_dim
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim

        # One rope key per ORIGINAL KV head, broadcast over that head's query group -- the GQA
        # grouping of the base model. `shared` (a single averaged key) is not supported here;
        # see the module docstring.
        assert self.total_num_heads % self.num_kv_heads == 0
        self.rope_group = self.total_num_heads // self.num_kv_heads

        self.q_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_heads * self.head_dim * 2, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.kv_a_proj = ReplicatedLinear(
            config.hidden_size, kv_lora_rank, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.kv_a_proj")
        # k_up_proj emits only the NOPE dims (192/head); the rope dims come from k_rope_proj.
        self.k_up_proj = ColumnParallelLinear(
            kv_lora_rank, self.total_num_heads * self.qk_nope_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.k_up_proj")
        self.v_up_proj = ColumnParallelLinear(
            kv_lora_rank, self.total_num_heads * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.v_up_proj")
        # Replicated: it is small (num_kv_heads*64 rows) and every TP rank needs whichever KV
        # heads its own query heads belong to. Sharding it would cut across the GQA grouping.
        self.k_rope_proj = ReplicatedLinear(
            config.hidden_size, self.num_kv_heads * self.qk_rope_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.k_rope_proj")
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rms_eps = config.rms_norm_eps
        self.canon_norm = bool(getattr(config, "mla_canon_norm", False))

        # get_rope takes a rope_parameters DICT; it derives rotary_dim from
        # partial_rotary_factor (0.25 * 256 = 64) and base from rope_theta. mrope keys are
        # stripped HERE as well as in the export: relying on the export alone means a stale or
        # hand-edited config silently selects the M-RoPE path, and this module's forward passes
        # 1-D positions that path does not expect.
        rp_clean = {k: v for k, v in rope_params.items() if not k.startswith("mrope")}
        rp_clean.setdefault("partial_rotary_factor", prf)
        if self.rope_keep is not None:
            # Ask for the NARROWER rotary width, then overwrite its table: get_rope would derive
            # inv_freq = base^(-2i/58), but the kept subspaces must keep their ORIGINAL
            # base^(-2i/64) frequencies or every rotation is subtly wrong.
            rp_clean["partial_rotary_factor"] = self.qk_rope_head_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rp_clean,
        )
        if self.rope_keep is not None:
            patch_partial_rope(self.rotary_emb, self.rope_keep, self.model_rope_dim,
                               float(rp_clean.get("rope_theta", 1e7)))
        assert self.rotary_emb.rotary_dim == self.qk_rope_head_dim, (
            f"rotary_dim {self.rotary_emb.rotary_dim} != qk_rope_head_dim "
            f"{self.qk_rope_head_dim}; the rope key and the rotated slice must agree")

        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling,
            num_kv_heads=self.num_heads,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        qg, _ = self.q_proj(hidden_states)
        q, gate = torch.chunk(qg.view(n, self.num_heads, self.head_dim * 2), 2, dim=-1)
        q = self.q_norm(q)

        c, _ = self.kv_a_proj(hidden_states)
        if self.canon_norm:
            # CANON: normalise the LATENT (DeepSeek kv_a_layernorm) and drop the per-head key
            # norm below. That removes the 24-scalar rms tail, which is the ONLY thing in the
            # cache row that is not a multiple of 128 -- and the row must be, because
            # FlashInfer's MLA kernel requires kv_lora_rank % 128 == 0 at TP=2 while page
            # alignment wants rank + tail = 2^k. tail = 4*64 rope + 0 = 256 = 2*128 makes ranks
            # 256/768/1792 land exactly on 512/1024/2048.
            c = (c.float() * torch.rsqrt(c.float().pow(2).mean(-1, keepdim=True)
                                         + self.rms_eps)).to(c.dtype)
        k_nope, _ = self.k_up_proj(c)
        v, _ = self.v_up_proj(c)
        k_nope = k_nope.view(n, self.num_heads, self.qk_nope_head_dim)

        # Decoupled rope key: [n, num_kv_heads, 64] -> broadcast over each KV head's query
        # group -> take this rank's contiguous head slice. ColumnParallelLinear splits
        # k_up_proj/v_up_proj into contiguous head ranges, so the same slice keeps rope and
        # nope dims describing the SAME head.
        kr, _ = self.k_rope_proj(hidden_states)
        kr = kr.view(n, self.num_kv_heads, self.qk_rope_head_dim)
        kr = kr.repeat_interleave(self.rope_group, dim=1)
        h0 = self.tp_rank * self.num_heads
        kr = kr[:, h0:h0 + self.num_heads, :]

        # Base-model dim order [rope(0:64) ; nope(64:256)], then k_norm over the FULL head_dim.
        # Under canon the key is used RAW: the latent was normalised above, the rope gains and
        # the mean rms divisor are folded into k_rope_proj at export, and (1+w_nope) into
        # k_up_proj. Applying k_norm here as well would normalise a second time.
        k = torch.cat([kr, k_nope], dim=-1)
        if not self.canon_norm:
            k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q.reshape(n, -1), k.reshape(n, -1))
        attn_out = self.attn(q, k, v.reshape(n, -1))
        attn_out = attn_out * torch.sigmoid(gate.reshape(n, -1))
        out, _ = self.o_proj(attn_out)
        return out


class Qwen3_5MLAMaterializedDecoderLayer(Qwen3_5DecoderLayer):
    """Stock Qwen3.5 layer, except full-attention layers get the rope-MLA module."""

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "", *,
                 ranks: dict[int, int] | None = None) -> None:
        super().__init__(vllm_config, layer_type=layer_type, prefix=prefix)
        if layer_type == "full_attention":
            idx = extract_layer_index(prefix)
            config = vllm_config.model_config.hf_text_config
            # The stock Attention built by super().__init__ registers itself in
            # static_forward_context; deleting the module leaves the registration behind and
            # the replacement then collides on the same name.
            vllm_config.compilation_config.static_forward_context.pop(
                f"{prefix}.self_attn.attn", None)
            del self.self_attn
            self.self_attn = Qwen3_5MLAMaterializedAttention(
                config,
                kv_lora_rank=(ranks or {})[idx],
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
            )


class Qwen3_5MLAMaterializedModel(Qwen3_5Model):
    hf_to_vllm_mapper = _ROPE_QWEN_MLA_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        import vllm.model_executor.models.qwen3_5 as _q35

        cfg = vllm_config.model_config.hf_text_config
        types = getattr(cfg, "layer_types", None) or []
        full_idx = [i for i, t in enumerate(types) if t == "full_attention"]
        ranks = _ranks_from_config(cfg, full_idx)
        print(f"[qwen-mla] pid {os.getpid()}: kv_lora ranks "
              f"{[ranks[i] for i in sorted(ranks)]}", flush=True)

        def _factory(vllm_config, layer_type, prefix=""):
            return Qwen3_5MLAMaterializedDecoderLayer(
                vllm_config, layer_type=layer_type, prefix=prefix, ranks=ranks)

        # Qwen3_5Model.__init__ builds layers from the MODULE-GLOBAL Qwen3_5DecoderLayer, not
        # from a class attribute, so subclassing alone changes nothing -- the parent silently
        # constructs stock layers and the run reports success while testing none of this code.
        orig = _q35.Qwen3_5DecoderLayer
        _q35.Qwen3_5DecoderLayer = _factory
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5DecoderLayer = orig


class Qwen3_5MLAMaterializedForCausalLM(Qwen3_5ForCausalLM):
    hf_to_vllm_mapper = _ROPE_QWEN_MLA_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        import vllm.model_executor.models.qwen3_5 as _q35

        # Same module-global swap, one level up: the base ForCausalLM builds Qwen3_5Model by
        # name rather than through a class attribute.
        orig = _q35.Qwen3_5Model
        _q35.Qwen3_5Model = Qwen3_5MLAMaterializedModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5Model = orig

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        # A rope key that never loaded is the failure this model is most exposed to: it would
        # leave k_rope_proj at its init, load "cleanly", and destroy positional information --
        # the same shape of silent failure as serving a DroPE checkpoint as its RoPE teacher.
        want = {n for n, _ in self.named_parameters() if "k_rope_proj" in n}
        missed = sorted(want - set(loaded))
        if missed:
            raise RuntimeError(
                f"{len(missed)} k_rope_proj weights were not loaded (e.g. {missed[:3]}) — "
                f"the rope key would be random. Check the weight mapper.")
        return loaded
