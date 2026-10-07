"""Qwen3_5MLAForConditionalGeneration: the base model's vision tower in front of the MLA language model.

The retrofit changes only the 16 full-attention layers of the language model, so the vision tower,
its merger and the multimodal processor are the teacher's, unchanged. This reuses vLLM's
Qwen3_5ForConditionalGeneration wholesale and swaps only the language model it builds. M-RoPE (the
[3, T] time/height/width positions multimodal requests carry) is handled in Qwen3_5MLAAttention.

Checkpoint: the teacher's multimodal config (model_type "qwen3_5", vision_config, preprocessor configs)
with the student's text_config (mla_* fields) and architectures ["Qwen3_5MLAForConditionalGeneration"];
weights are the student's plus the teacher's model.visual.* tensors.
"""
from __future__ import annotations

import vllm.model_executor.models.qwen3_5 as _q35
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY

from .qwen3_5_mla import Qwen3_5MLAForCausalLM


# The processor registry is keyed by class, so a subclass must be registered itself.
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MLAForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        # The parent constructs `Qwen3_5ForCausalLM` by module-global name; substitute ours for the
        # duration of its __init__ so everything else (vision tower, processors, M-RoPE positions,
        # weight mapping) stays the parent's own code.
        orig = _q35.Qwen3_5ForCausalLM
        _q35.Qwen3_5ForCausalLM = Qwen3_5MLAForCausalLM
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _q35.Qwen3_5ForCausalLM = orig
        # The MTP proposer finds the image token by CLASS NAME (Qwen3_5ForConditionalGeneration ->
        # config.image_token_id); any other name falls through to config.image_token_index, which Qwen
        # configs lack. Provide it so speculative decoding works on the multimodal model.
        if not hasattr(self.config, "image_token_index"):
            self.config.image_token_index = self.config.image_token_id

    def process_weights_after_loading(self):
        # vLLM calls the model-level hook on the TOP module only; the MLA language model checks there
        # that every layer's kv_b_proj was built (its load_weights may be called in several pieces).
        hook = getattr(self.language_model, "process_weights_after_loading", None)
        if hook is not None:
            hook()
        getattr(super(), "process_weights_after_loading", lambda: None)()
