"""vLLM general plugin registering the MLA architecture.

REGISTERED VIA ENTRY POINT, NOT A DIRECT CALL. vLLM v1 runs the engine in a SEPARATE PROCESS,
so `ModelRegistry.register_model(...)` executed in the parent never reaches the worker — the
model silently resolves to the stock architecture and the run "succeeds" while testing nothing.
The `vllm.general_plugins` entry point is loaded in every process, which is the only way the
registration is actually in effect where the model is built.
"""


def _skip_cudagraph_memory_profiling_if_unwanted():
    """Don't COMPUTE an estimate that VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 will discard.

    gpu_worker.determine_available_memory() calls profile_cudagraph_memory() whenever cudagraphs
    are enabled, and only then zeroes the result if the env var is off -- so the flag saves no
    work and, more importantly, does not avoid the code path.

    That path (_init_minimal_kv_cache_for_profiling) allocates a minimal KV cache sized for
    UNPADDED pages and then reshapes it through a strided view that assumes the PADDED page
    stride. With any padded-page layer present it overruns:

        setStorage: sizes [832, 2, 32, 512] ... requiring 1449223168 ... storage of size 111607808

    MLA pads its cache rows on purpose (so per-layer pages divide evenly instead of every layer
    inflating to the widest -- without it the served compression collapses to ~1x), and adding
    the MTP draft's standard-attention layer introduces a third page shape that triggers it.

    Scoped to the env var: with the flag unset this patch does nothing, so the default path is
    untouched. Skipping the estimate only makes the KV budget more conservative.
    """
    import os
    if os.environ.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") != "0":
        return
    import sys
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner, "_mla_profiling_patched", False):
        return
    GPUModelRunner.profile_cudagraph_memory = lambda self: 0
    GPUModelRunner._mla_profiling_patched = True
    print("[qwen-mla] cudagraph memory profiling skipped "
          "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0)", file=sys.stderr, flush=True)


def _enable_use_mla_if_requested():
    """Let MLA answer True to model_config.use_mla, which is what gates DCP.

    Decode context parallelism shards the KV cache across ranks by SEQUENCE instead of
    replicating it, so per-GPU KV traffic halves at TP=2. For MLA that is exactly the right
    thing -- the latent is identical on every rank today, so every rank reads the whole batch's
    cache. vLLM refuses it for us with

        Decode context parallelism for GQA/MQA requires --tensor-parallel-size (2) to be
        greater than the model's total number of KV heads (4)

    because that check (config/model.py, `if dcp > 1 and not self.use_mla`) takes the GQA branch:
    `use_mla` is `is_deepseek_mla and not VLLM_MLA_DISABLE`, and `is_deepseek_mla` is a
    model_type whitelist we are not on. We ARE MLA in every way that matters -- MLAAttention,
    an MLACommonImpl subclass, one latent "head" in the cache.

    Only `use_mla` is overridden, NOT `is_deepseek_mla`: `get_head_size()` keys off the latter
    and would then look for `kv_lora_rank` on the hf config, which does not exist and could not
    (ours is per layer). The other things `use_mla` reaches are correct for us anyway --
    `get_num_kv_heads()` returns 1, which is what our cache has.

    Opt-in via QWEN_MLA_ENABLE_DCP=1 so the default path is untouched.
    """
    import os
    import sys
    if os.environ.get("QWEN_MLA_ENABLE_DCP") != "1":
        return
    from vllm.config.model import ModelConfig
    if getattr(ModelConfig, "_mla_use_mla_patched", False):
        return
    _orig = ModelConfig.use_mla

    def use_mla(self):
        try:
            archs = self.architectures or []
        except Exception:                      # accessed before architectures is resolved
            archs = []
        if any(a in ALL_ARCHS for a in archs):
            # get_mla_dims() reads hf_text_config.kv_lora_rank, which a Qwen config does not
            # have and could not -- ours is per layer. Nothing on the hot path calls it (the
            # attention layer is handed its dims explicitly), but with use_mla True the FA4
            # prefill warmup asks for warmup KEYS through it. Publish the largest rank: it is
            # the right shape to warm, and get_head_size() is unaffected because that keys off
            # is_deepseek_mla, which stays False.
            htc = getattr(self, "hf_text_config", None)
            if htc is not None and not hasattr(htc, "kv_lora_rank"):
                ranks = getattr(htc, "mla_ranks", None)
                if ranks:
                    htc.kv_lora_rank = max(int(v) for v in ranks.values())
            return True
        return _orig.fget(self)

    ModelConfig.use_mla = property(use_mla)
    ModelConfig._mla_use_mla_patched = True

    # The DCP combine kernel cannot compile at our latent ranks without this.
    from . import dcp_fix
    dcp_fix.install()
    dcp_fix.skip_fa4_warmup()
    print("[qwen-mla] use_mla forced True for MLA architectures (QWEN_MLA_ENABLE_DCP=1)",
          file=sys.stderr, flush=True)


# Architecture name -> "module:class". The first group is canonical; LEGACY_ARCHS keeps checkpoints
# exported under the earlier "MVLA" naming loadable (the v1-preview uploads used them).
ARCHS = {
    "Qwen3_5MLAForCausalLM": "qwen_mla.qwen3_5_mla:Qwen3_5MLAForCausalLM",              # serving (latent cache)
    "Qwen3_5MLAMaterializedForCausalLM":
        "qwen_mla.qwen3_5_mla_materialized:Qwen3_5MLAMaterializedForCausalLM",           # per-head K/V reference
    "Qwen3_5MLANoPEForCausalLM": "qwen_mla.qwen3_5_mla_nope:Qwen3_5MLANoPEForCausalLM",  # NoPE research variant
    "Qwen3_5DroPEForCausalLM": "qwen_mla.qwen3_5_mla_nope:Qwen3_5DroPEForCausalLM",
}
LEGACY_ARCHS = {
    "Qwen3_5MVLAAbsorbedForCausalLM": "Qwen3_5MLAForCausalLM",
    "Qwen3_5RoPEMVLAForCausalLM": "Qwen3_5MLAMaterializedForCausalLM",
    "Qwen3_5MVLAForCausalLM": "Qwen3_5MLANoPEForCausalLM",
}
ALL_ARCHS = frozenset(ARCHS) | frozenset(LEGACY_ARCHS)


def _alias_legacy_env():
    """MVLA_* environment variables (pre-rename) are honoured as QWEN_MLA_* when the new name is unset.

    Runs in every process before any kernel reads its knobs, so the C++ getenv() in the patched
    headers sees the copied values too.
    """
    import os
    for k, v in list(os.environ.items()):
        if k.startswith("MVLA_"):
            os.environ.setdefault("QWEN_MLA_" + k[len("MVLA_"):], v)


def _alias_legacy_config_keys():
    """Checkpoint configs written before the rename use mvla_* keys; mirror them to mla_* on load.

    Wraps the get_config ModelConfig calls (resolved through vllm.config.model's globals), and only
    fills keys that are absent, so a config carrying both is read as its mla_* values.
    """
    import vllm.config.model as _m
    if getattr(_m.get_config, "_qwen_mla_aliased", False):
        return
    orig = _m.get_config

    def _mirror(cfg):
        if cfg is None:
            return
        for k in list(vars(cfg)):
            if k.startswith("mvla_") and not hasattr(cfg, "mla_" + k[len("mvla_"):]):
                setattr(cfg, "mla_" + k[len("mvla_"):], getattr(cfg, k))

    def get_config(*args, **kwargs):
        cfg = orig(*args, **kwargs)
        _mirror(cfg)
        _mirror(getattr(cfg, "text_config", None))
        return cfg
    get_config._qwen_mla_aliased = True
    _m.get_config = get_config


_REGISTERED = False


def register():
    import os, sys
    global _REGISTERED
    if _REGISTERED:              # two entry points (qwen_mla, legacy mvla) may both call this
        return
    _REGISTERED = True
    from vllm import ModelRegistry

    print(f"[qwen-mla] register() called in pid {os.getpid()}", file=sys.stderr, flush=True)
    _alias_legacy_env()
    # QWEN_MLA_GLA_SHARD=0 (keep the full latent on every rank, for A/B) changes tensor shapes, but vLLM's
    # torch.compile cache key does not include plugin env vars: a graph compiled for the sharded
    # shapes would be reused and fail ("expected size 256==128"). Bypass the cache for such runs.
    if os.environ.get("QWEN_MLA_GLA_SHARD") == "0" and "VLLM_DISABLE_COMPILE_CACHE" not in os.environ:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        print("[qwen-mla] QWEN_MLA_GLA_SHARD=0: compile cache disabled for this run", file=sys.stderr, flush=True)
    _alias_legacy_config_keys()
    _fingerprint_compile_cache()
    _enable_use_mla_if_requested()
    # Each architecture is a SEPARATE name so a config cannot silently select the wrong cache behaviour
    # (e.g. serve a RoPE checkpoint through the NoPE class with rotary omitted).
    for name, target in ARCHS.items():
        ModelRegistry.register_model(name, target)
    for old, new in LEGACY_ARCHS.items():
        ModelRegistry.register_model(old, ARCHS[new])
    _skip_cudagraph_memory_profiling_if_unwanted()
    _inherit_qwen3_5_config_hook()


def _inherit_qwen3_5_config_hook():
    """Give the plugin architectures the teacher's Qwen3.5 config hook.

    vLLM applies per-model config fixes by looking the ARCHITECTURE NAME up in
    MODELS_CONFIG_MAP. Qwen3.5's hook copies the checkpoint's `mamba_ssm_dtype: float32` into
    cache_config.mamba_ssm_cache_dtype. Our architectures were not in the map, so the hook
    never ran and the 48 untouched linear-attention layers kept their recurrent state in
    bf16 while the teacher keeps it in float32 -- a precision difference that has nothing to
    do with the retrofit. The dtype-only variant is used (the one the teacher's own
    Qwen3_5ForConditionalGeneration gets); the text-only variant also rewrites
    rope_parameters, which the plugin handles itself.
    """
    from vllm.model_executor.models.config import (MODELS_CONFIG_MAP,
                                                   Qwen3_5ForConditionalGenerationConfig)
    for arch in ALL_ARCHS:
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)


def _fingerprint_compile_cache():
    """Fold the plugin's code and the checkpoint's mla_* config into vLLM's compile-cache key.

    vLLM's ModelConfig.compute_hash covers its own fields (model path, dtype, ...) but not the HF config
    contents or plugin source, so a torch.compile / AOT artifact built for one config or plugin version
    was silently reused for another at the same model path -- measured: a materialised reference served
    from a stale artifact differed by 0.03 mean |dlogprob| until recompiled. For our architectures only,
    the hash now also covers a digest of every qwen_mla source file, the architecture name and the
    mla_* config values; other models' hashes are untouched.
    """
    import hashlib, json, pathlib
    import vllm.config.model as _m
    if getattr(_m.ModelConfig.compute_hash, "_qwen_mla", False):
        return
    src = hashlib.sha256()
    for f in sorted(pathlib.Path(__file__).resolve().parent.rglob("*.py")):
        src.update(f.read_bytes())
    code_digest = src.hexdigest()
    orig = _m.ModelConfig.compute_hash

    def compute_hash(self):
        h = orig(self)
        try:
            archs = list(getattr(self, "architectures", None) or [])
        except Exception:
            archs = []
        if not any(a in ALL_ARCHS for a in archs):
            return h
        cfg = getattr(self, "hf_text_config", None) or getattr(self, "hf_config", None)
        keys = {k: repr(v) for k, v in sorted(vars(cfg).items()) if k.startswith("mla_")} if cfg is not None else {}
        extra = json.dumps({"archs": archs, "mla": keys, "code": code_digest}, sort_keys=True)
        return hashlib.sha256((h + extra).encode()).hexdigest()
    compute_hash._qwen_mla = True
    _m.ModelConfig.compute_hash = compute_hash
