"""Let --enable-lora serve the EXL3 builds.

vLLM's LoRA wrappers find a base layer's device through a fixed list of weight attribute names
(weight, qweight, weight_packed, ...). cuda-exl3's layers hold `trellis` instead, so every wrapped
linear layer fails at startup with "Unsupported base layer". The LoRA path needs only the device: the
adapter's delta is added to whatever the base layer's quant method returns. When vLLM's lookup fails,
fall back to the layer's first parameter or buffer.

Checked on GLA-g2-EXL3-4.0bpw at TP=2 with rank-16 adapters on every MLP projection: a zero adapter
matches the base within run-to-run noise, a small random one shifts the logprobs slightly, a large one
garbles the output, and base and adapter requests batched together each get their own weights.
"""
from __future__ import annotations


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
            for t in list(base_layer.parameters(recurse=False)) + list(base_layer.buffers(recurse=False)):
                return t.device
            raise

    lu._get_lora_device = _get_lora_device
    base_linear._get_lora_device = _get_lora_device      # imported by name there
    lu._qwen_mla_lora_dev = True
