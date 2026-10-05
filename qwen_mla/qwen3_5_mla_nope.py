"""vLLM model definition for Qwen3.5 + MLA (NoPE, variable per-layer latent rank).

    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3_5MLANoPEForCausalLM",
                                 "qwen3_5_mla_nope:Qwen3_5MLANoPEForCausalLM")

WHAT THIS IS. Qwen3.5 is a hybrid: 48 gated-delta-net layers and 16 full-attention layers.
MLA replaces the KV path of the 16 full-attention layers with a low-rank latent, per layer,
on a gamma=0 (NoPE) base. Everything else — GDN layers, MLPs, norms, embeddings — is stock
Qwen3.5, so this subclasses the in-tree model and swaps exactly one module.

MILESTONE 1: SPEED, NOT MEMORY. This materialises K and V from the latent and hands them to
the ordinary paged Attention. That is quality-identical to the PyTorch path and gives vLLM's
batching, scheduling and kernels — which is where the benchmarking speedup lives, since our
eval workload is ~65536 tokens of prefill against ~100 tokens of decode, and every MLA
implementation materialises K/V during prefill anyway.

It does NOT reproduce the cache saving: MLA reconstructs one K/V per QUERY head (24), where
base Qwen3.5 is GQA with 4 KV heads, so the cache is 24/4 = 6x the base rather than 3.2x
smaller. Milestone 2 (cache the latent itself) is what recovers that, and it needs one extra
trick, recorded here so it is not rediscovered:

    ABSORPTION WITH POST-UP-PROJECTION QK-NORM.
    MLA's decode path works by absorbing W_UK into the query: q·Kᵀ = (q·W_UK)·cᵀ, which needs
    K linear in the latent c. DeepSeek normalises the LATENT (kv_a_layernorm), which stays
    linear. Qwen3.5 normalises the reconstructed KEY per head, which does not — so this model
    looks unabsorbable. It is not:

        k_h            = W_UK_h · c
        k_norm_h       = (k_h / rms(k_h)) ⊙ w
        q_h · k_norm_hᵀ = [(q_h ⊙ w) · W_UK_h] · cᵀ / rms(k_h)

    The norm weight folds into the QUERY, and the only non-linear term is rms(k_h) — one
    scalar per (token, head). Caching [latent | 24 rms scalars] restores absorption at ~3.75%
    cache overhead, and the scalars are free to compute during prefill.

NO ROTARY. The base is DroPE'd (gamma=0), so the rotary transform is the identity. Rather
than applying an identity rotation, this omits rotary entirely for the converted layers —
cheaper, and it makes the NoPE assumption explicit rather than implicit in a gamma value that
could silently drift.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
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

# Weight-name mapping for the exported MLA checkpoint.
#
# 1. "language_model." -> "": the export descends from a VL base
#    (Qwen3_5ForConditionalGeneration) whose text weights nest under model.language_model.
#    Written WITHOUT a leading "model." because the mapper runs after the model prefix has
#    already been stripped.
# 2. ".k_rope_proj." -> None: a zero-element artefact of the training-side module (NoPE keeps
#    the attribute for shape compatibility). It has no counterpart here, so drop it rather
#    than fail on an unexpected key.
# 3. The inherited stacked mapping fuses .q_proj/.k_proj/.v_proj into .qkv_proj. MLA has no
#    k_proj/v_proj and its q_proj (query + output gate) stands alone, so those three entries
#    are REMOVED while the GDN in_proj fusions are kept — those still apply to the 48
#    linear-attention layers, which are untouched by MLA.
_BASE_STACKED = {
    k: v
    for k, v in Qwen3_5Model.hf_to_vllm_mapper.orig_to_new_stacked.items()
    if k not in (".q_proj", ".k_proj", ".v_proj")
}
_QWEN_MLA_MAPPER = WeightsMapper(
    orig_to_new_substr={"language_model.": "", ".k_rope_proj.": None},
    orig_to_new_stacked=_BASE_STACKED,
)

# Per-layer ranks come from the measured allocation, not the config: the whole point of MLA
# is that layers differ. Supplied by env so the same registered arch serves any allocation.
_RANKS_ENV = "QWEN_MLA_RANKS_JSON"
_RATIO_ENV = "QWEN_MLA_RATIO"
_UNIFORM_ENV = "QWEN_MLA_UNIFORM_RANK"


def _load_ranks(full_idx: list[int]) -> dict[int, int]:
    uni = int(os.environ.get(_UNIFORM_ENV, "0"))
    if uni:
        return {i: uni for i in full_idx}
    path = os.environ.get(_RANKS_ENV, "")
    if not path:
        raise RuntimeError(
            f"set {_RANKS_ENV}=<allocation json> (+ {_RATIO_ENV}) or {_UNIFORM_ENV}=<r>"
        )
    alloc = json.loads(open(path).read())["allocations"][os.environ.get(_RATIO_ENV, "3.2")]
    ranks = {int(k): int(v) for k, v in alloc["ranks"].items()}
    if sorted(ranks) != sorted(full_idx):
        raise RuntimeError(
            f"allocation covers {sorted(ranks)} but full-attention layers are {sorted(full_idx)}"
        )
    return ranks


class Qwen3_5MLANoPEAttention(nn.Module):
    """Qwen3.5 gated attention whose K/V come from a per-layer low-rank latent.

    Mirrors `mla_retrofit.mla.MLAAttention` (the training-side module) exactly in structure and
    order of operations, because any divergence shows up as a silent quality gap rather than an
    error: q_proj carries the fused query+gate, q_norm/k_norm are per-head, k_norm is applied
    AFTER the up-projection, and the output is gated by sigmoid before o_proj.
    """

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
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp == 0
        self.num_heads = self.total_num_heads // tp
        self.head_dim = config.head_dim or (config.hidden_size // self.total_num_heads)
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.head_dim**-0.5

        # Query AND gate: Qwen3.5's q_proj emits 2x head_dim per head (attn_output_gate).
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        # Down-projection to the shared latent. Replicated: it is small and every TP rank
        # needs the whole latent to build its own head slice.
        self.kv_a_proj = ReplicatedLinear(
            config.hidden_size,
            kv_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj",
        )
        self.k_up_proj = ColumnParallelLinear(
            kv_lora_rank,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_up_proj",
        )
        self.v_up_proj = ColumnParallelLinear(
            kv_lora_rank,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_up_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # GemmaRMSNorm == Qwen3NextRMSNorm: the (1 + weight) convention. Using the same class
        # the in-tree model uses means the checkpoint weights load unchanged.
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # num_kv_heads == num_heads: MLA reconstructs one K/V per QUERY head, so this is MHA,
        # not the base model's GQA. See the milestone-1 note at the top about cache size.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        qg, _ = self.q_proj(hidden_states)
        qg = qg.view(n, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        q = self.q_norm(q)

        c, _ = self.kv_a_proj(hidden_states)
        k, _ = self.k_up_proj(c)
        v, _ = self.v_up_proj(c)
        k = self.k_norm(k.view(n, self.num_heads, self.head_dim))

        # NoPE: no rotary. gamma=0 makes it the identity, so it is omitted entirely.
        attn_out = self.attn(
            q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)
        )
        attn_out = attn_out * torch.sigmoid(gate.reshape(n, -1))
        out, _ = self.o_proj(attn_out)
        return out


class Qwen3_5MLANoPEDecoderLayer(Qwen3_5DecoderLayer):
    """Stock Qwen3.5 layer, except full-attention layers get the MLA module."""

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "", *,
                 ranks: dict[int, int] | None = None) -> None:
        super().__init__(vllm_config, layer_type=layer_type, prefix=prefix)
        if layer_type == "full_attention":
            idx = extract_layer_index(prefix)
            config = vllm_config.model_config.hf_text_config
            # super().__init__ already built the stock GQA attention, and vLLM's Attention
            # registers itself in compilation_config.static_forward_context under its prefix.
            # Dropping the module is not enough — the registration survives and the MLA
            # attention then collides on the same name ("Duplicate layer name"). Remove the
            # entry as well so the slot is genuinely free.
            vllm_config.compilation_config.static_forward_context.pop(
                f"{prefix}.self_attn.attn", None
            )
            del self.self_attn
            self.self_attn = Qwen3_5MLANoPEAttention(
                config,
                kv_lora_rank=(ranks or {})[idx],
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
            )


class Qwen3_5MLANoPEModel(Qwen3_5Model):
    hf_to_vllm_mapper = _QWEN_MLA_MAPPER

    """Stock Qwen3.5 model whose full-attention layers are MLA.

    `Qwen3_5Model.__init__` constructs layers through a closure that resolves
    `Qwen3_5DecoderLayer` from the qwen3_5 module globals at CALL time. Rather than duplicate
    that long __init__ (and inherit the maintenance burden of keeping the copy in sync), the
    global is swapped for the duration of construction and restored in a finally. The same
    late-binding property that makes this work is the one that makes `from x import y` patches
    silently fail — see TRAPS #12.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        from vllm.model_executor.models import qwen3_5 as _q35

        config = vllm_config.model_config.hf_text_config
        full_idx = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]
        ranks = _load_ranks(full_idx)
        per_layer_kv = 2 * config.num_key_value_heads * config.head_dim
        base_total = per_layer_kv * len(full_idx)
        print(
            f"[qwen-mla] {len(full_idx)} full-attn layers; ranks "
            f"{[ranks[i] for i in full_idx]}; latent {sum(ranks.values())} vs base "
            f"{base_total} dims/token = {base_total / sum(ranks.values()):.3f}x",
            flush=True,
        )
        self._mla_ranks = ranks
        # Proof-of-construction for the smoke test. A FILE, not an env var: the engine runs
        # in a subprocess, so anything written to its environment is invisible to the caller —
        # which is what made the first version of this guard always fail.
        marker = os.environ.get("QWEN_MLA_BUILT_MARKER")
        if marker:
            open(marker, "w").write(f"{len(full_idx)} layers\n")

        def _factory(vllm_config, layer_type, prefix=""):
            return Qwen3_5MLANoPEDecoderLayer(
                vllm_config, layer_type=layer_type, prefix=prefix, ranks=ranks
            )

        orig = _q35.Qwen3_5DecoderLayer
        _q35.Qwen3_5DecoderLayer = _factory
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5DecoderLayer = orig


class Qwen3_5MLANoPEForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.5 causal LM whose inner model is the MLA variant.

    `Qwen3_5ForCausalLMBase.__init__` does `self.model = Qwen3_5Model(...)`, resolving the name
    from the qwen3_5 module globals — so subclassing alone leaves the STOCK model in place and
    every MLA layer silently disappears. That failure is invisible: vLLM logs
    "Resolved architecture: Qwen3_5MLANoPEForCausalLM", loads, and generates, while running none
    of this code. Swap the global for the duration of construction, same as the model does for
    its decoder layer.
    """

    # The base packs q/k/v into a fused qkv_proj. MLA has no k_proj/v_proj at all, and its
    # q_proj stands alone, so that mapping must not apply or the loader will look for weights
    # that do not exist.
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    # Qwen3_5ForCausalLM itself defines no mapper (the inner model carries it), so combine
    # with whatever it inherits rather than assuming the attribute exists.
    hf_to_vllm_mapper = _QWEN_MLA_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        from vllm.model_executor.models import qwen3_5 as _q35

        orig = _q35.Qwen3_5Model
        _q35.Qwen3_5Model = Qwen3_5MLANoPEModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5Model = orig


class _NoRotary(nn.Module):
    """Identity stand-in for vLLM's rotary embedding on a DroPE (gamma=0) model."""

    def forward(self, positions, query, key, *a, **kw):
        return query, key


class Qwen3_5DroPEForCausalLM(Qwen3_5ForCausalLM):
    """Uncompressed DroPE phase-1: stock Qwen3.5 with rotary DISABLED.

    WHY THIS CLASS HAS TO EXIST. DroPE sets gamma=0 by rescaling the rotary `inv_freq` buffer,
    which HF registers NON-PERSISTENTLY — `save_pretrained` does not write it. Worse, vLLM
    never reads it either: it constructs rotary from `rope_parameters` via `get_rope`. So an
    exported DroPE checkpoint loaded under the stock architecture silently becomes the RoPE
    TEACHER, and it looks entirely healthy — it generates fluent text and posts plausible
    benchmark numbers. It was caught only by scoring the same checkpoint on both engines:
    msmarco NDCG@10 read 58.01 under vLLM against 40.73 under PyTorch.
    ANY consumer of these checkpoints that builds rotary from config has the same problem.

    Replacing the rotary module with an identity is the faithful representation: at gamma=0
    the rotation IS the identity, so this computes the same function while making the
    assumption explicit rather than hiding it in a buffer that does not survive a save.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        n = 0
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "rotary_emb"):
                attn.rotary_emb = _NoRotary()
                # the fused qk-norm+rope+gate kernel bakes rotary in; force the eager path
                if hasattr(attn, "use_fused_qk_norm_rope_gate"):
                    attn.use_fused_qk_norm_rope_gate = False
                n += 1
        print(f"[drope-vllm] rotary disabled on {n} attention layers (gamma=0)", flush=True)
