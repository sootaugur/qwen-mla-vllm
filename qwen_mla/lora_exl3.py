"""LoRA adapters on the plugin's models: EXL3 base layers, the output head, and GLA's sharded latent.

* EXL3 layers. vLLM's LoRA wrappers find a base layer's device through a fixed list of weight
  attribute names (weight, qweight, weight_packed, ...). cuda-exl3's layers hold `trellis` instead,
  so every wrapped linear layer failed at startup with "Unsupported base layer". The LoRA path
  needs only the device: the adapter's delta is added to whatever the base layer's quant method
  returns. When vLLM's lookup fails, fall back to the layer's first parameter or buffer.

* Output head. The models declare `lm_head` as a LoRA embedding module, so an adapter that targets
  it (e.g. a decision head trained as an lm_head LoRA) is applied rather than skipped. vLLM builds
  the wrapper from `lm_head.weight`'s dtype and device; an EXL3 head has no `weight`, so take the
  device from its first parameter and the dtype from the LoRA config.

* GLA latent. With the latent split across tensor-parallel ranks, each rank's kv_a_proj holds only
  its group's rows (qwen3_5_mla.py tags the layer with that row slice). It is a ReplicatedLinear, so
  vLLM hands every rank the adapter's full lora_B; keep the rank's rows.

Checked on GLA-g2-EXL3-4.0bpw at TP=2 with rank-16 adapters on every MLP projection: a zero adapter
matches the base within run-to-run noise, a small random one shifts the logprobs slightly, a large one
garbles the output, and base and adapter requests batched together each get their own weights. With
AutoTrust-Decider's decision adapter (lm_head included, k/v folded by scripts/fold_lora_kv.py) it matched
the base-model EXL3 build serving the original adapter (README, LoRA adapters); scaling only the lm_head
LoRA by 20 changed the decisions, so that part is applied.
"""
from __future__ import annotations


def _first_tensor(module):
    for t in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return t
    return None


def patch_lora_on_exl3():
    from vllm.lora.layers import base_linear
    from vllm.lora.layers import utils as lu
    if getattr(lu, "_qwen_mla_lora_dev", False):
        return
    orig = lu._get_lora_device

    def _get_lora_device(base_layer):
        try:
            return orig(base_layer)
        except ValueError:
            t = _first_tensor(base_layer)
            if t is None:
                raise
            return t.device

    lu._get_lora_device = _get_lora_device
    base_linear._get_lora_device = _get_lora_device      # imported by name there
    lu._qwen_mla_lora_dev = True
    _patch_lm_head()
    _patch_latent_rows()


def _patch_lm_head():
    import torch
    import vllm.lora.model_manager as mm
    import vllm.lora.utils as u
    from vllm.lora.layers import LogitsProcessorWithLoRA
    orig = u.from_layer_logits_processor

    def from_layer_logits_processor(layer, lm_head, max_loras, lora_config, model_config=None):
        if hasattr(lm_head, "weight"):
            return orig(layer, lm_head, max_loras, lora_config, model_config)
        t = _first_tensor(lm_head)
        dtype = lora_config.lora_dtype if isinstance(lora_config.lora_dtype, torch.dtype) else torch.bfloat16
        ret = LogitsProcessorWithLoRA(layer, lm_head.embedding_dim, dtype, t.device,
                                      lm_head.get_sharded_to_full_mapping())
        ret.create_lora_weights(max_loras, lora_config, model_config)
        return ret

    u.from_layer_logits_processor = from_layer_logits_processor
    mm.from_layer_logits_processor = from_layer_logits_processor   # imported by name there
    # vLLM's own Qwen3.5 classes (the stock EXL3 / INT4 builds served next to this plugin) declare no
    # embedding modules either, so an lm_head LoRA was skipped there as well.
    from vllm.model_executor.models import qwen3_5 as q
    for cls in (q.Qwen3_5ForCausalLM, q.Qwen3_5ForConditionalGeneration):
        if "lm_head" not in (cls.embedding_modules or {}):
            cls.embedding_modules = {**(cls.embedding_modules or {}), "lm_head": "output_embeddings"}


def _patch_latent_rows():
    from vllm.lora.layers import ReplicatedLinearWithLoRA
    orig = ReplicatedLinearWithLoRA.set_lora

    def set_lora(self, index, lora_a, lora_b):
        rows = getattr(self.base_layer, "_qwen_mla_lora_rows", None)
        if rows is not None and lora_b.shape[0] != self.output_size:
            lora_b = lora_b[rows]
        return orig(self, index, lora_a, lora_b)

    ReplicatedLinearWithLoRA.set_lora = set_lora
