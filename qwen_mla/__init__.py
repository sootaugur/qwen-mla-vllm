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
    import sys
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner, "_mla_profiling_patched", False):
        return
    GPUModelRunner._mla_profiling_patched = True
    if os.environ.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS") != "0":
        # Automatic fallback: with an FP8 KV cache plus the MTP draft's standard-attention layer, the
        # minimal profiling cache overruns exactly as described above (measured: "setStorage: sizes
        # [12800, 64, 4, 512] ... out of bounds"). Recover the way vLLM's own no-graphs branch does --
        # release the profiling cache and report no estimate -- for that error only.
        orig = GPUModelRunner.profile_cudagraph_memory

        def profile_cudagraph_memory(self):
            try:
                return orig(self)
            except RuntimeError as e:
                if "setStorage" not in str(e):
                    raise
                self._cleanup_profiling_kv_cache()
                print("[qwen-mla] CUDA-graph memory estimate skipped (mixed padded KV page shapes); "
                      "KV budget computed without it", file=sys.stderr, flush=True)
                return 0
        GPUModelRunner.profile_cudagraph_memory = profile_cudagraph_memory
        return
    GPUModelRunner.profile_cudagraph_memory = lambda self: 0
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
    "Qwen3_5MLAForConditionalGeneration":
        "qwen_mla.qwen3_5_mla_vl:Qwen3_5MLAForConditionalGeneration",                    # + vision tower
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
    # flashinfer JIT-compiles kernels at startup (vLLM's own sampler included) by running the `ninja`
    # EXECUTABLE that pip installs into the environment's bin/. When the server is started as
    # .venv/bin/vllm without activating the venv, that directory is not on PATH and the engine dies with
    # FileNotFoundError: 'ninja'. register() runs first in every vLLM process, so fix PATH here.
    _bindir = os.path.dirname(sys.executable)
    if _bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")
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
    _profile_prefill_context_scratch()
    _inherit_qwen3_5_config_hook()
    _enable_mtp_draft()
    _mtp_draft_attention_backend()
    _spec_decode_graph_capture_fix()
    _fp8_kv_default_from_checkpoint()
    _prefix_cache_hit_granularity()
    _prefix_cache_scheduler_fixes()
    from .mamba_align_free import patch_mamba_align_free
    patch_mamba_align_free()


def _profile_prefill_context_scratch():
    """Make the profile run see the whole per-chunk scratch of a long-context prefill.

    vLLM sizes the KV cache from the peak memory of a profile forward, in which every MLA layer
    stands in for the context-chunk path by allocating the kv_b_proj output alone:
    [workspace rows, heads, nope + v]. The real path holds more at once: the assembled key
    [rows, heads, nope + rope], the per-head rope selection feeding it [rows, heads, rope], and
    FLASH_ATTN's copy of v padded to the key width [rows, heads, nope + rope]. Measured on a GLA-g2
    rank with 64k rows: 1.89 GB real vs 0.68 GB simulated -- the KV cache took the difference and the
    first prompt past ~50k tokens ran out of memory, on any GPU whose budget was tight enough to notice
    (16 GB cards first). Allocating those (plus one FP8 dequant slice and a measured margin) alongside
    the parent's stand-in makes the profiled peak cover the real one, so the KV budget is right by
    construction rather than by --kv-cache-memory-bytes.
    """
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    if getattr(MLAAttention.forward_impl, "_qwen_mla", False):
        return
    orig = MLAAttention.forward_impl

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *args, **kwargs):
        if attn_metadata is not None or getattr(self.impl, "rms_offset", -1) < 0:
            return orig(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *args, **kwargs)
        import torch
        rows, heads = self.chunked_prefill_workspace_size, self.num_heads
        kw = self.qk_nope_head_dim + getattr(self.impl, "mla_rope_dim", self.qk_rope_head_dim)
        dev, dt = k_c_normed.device, k_c_normed.dtype
        held = [torch.empty((rows, heads, kw), device=dev, dtype=dt),            # assembled key
                torch.empty((rows, heads, kw - self.qk_nope_head_dim), device=dev, dtype=dt),  # rope sel
                torch.empty((rows, heads, max(kw, self.v_head_dim)), device=dev, dtype=dt)]   # padded v
        if getattr(self.impl, "mla_kv_fp8", False):
            # One FP8 dequant slice of the gather (fp32 intermediates, ~10 bytes per element). It is not
            # live at the same moment as the key above; counting both keeps the estimate an upper bound.
            from .ops.mla_gather import _DEQUANT_SLICE
            held.append(torch.empty((min(rows, _DEQUANT_SLICE), self.head_size * 10), device=dev, dtype=torch.uint8))
        # Margin for the query-side and indexing temporaries of the same call (prefill output, softmax lse,
        # merge buffers, gather indices): measured peak 499 MiB vs 445 MiB from the terms above at the
        # widest GLA-g2 layer, 15k rows.
        sim = rows * heads * (self.qk_nope_head_dim + self.v_head_dim) * dt.itemsize
        held.append(torch.empty(int(0.2 * (sim + sum(t.numel() * t.element_size() for t in held))),
                                device=dev, dtype=torch.uint8))
        try:
            return orig(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *args, **kwargs)
        finally:
            del held
    forward_impl._qwen_mla = True
    MLAAttention.forward_impl = forward_impl


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


_ORIG_HF_CONFIG_OVERRIDE = None


def _mtp_hf_config_override(hf_config):
    """Module-level (not a closure): vLLM stores this callable in its config, and data-parallel startup
    pickles the config into each engine process."""
    archs = getattr(hf_config, "architectures", None) or []
    if getattr(hf_config, "model_type", None) == "qwen3_5_text" and any(a in ALL_ARCHS for a in archs):
        hf_config.model_type = "qwen3_5_mtp"
        hf_config.update({"n_predict": getattr(hf_config, "mtp_num_hidden_layers", None),
                          "architectures": ["Qwen3_5MTP"]})
        return hf_config
    return _ORIG_HF_CONFIG_OVERRIDE(hf_config)


def _enable_mtp_draft():
    """Let `--speculative-config '{"method": "mtp"}'` find the Qwen3.5 MTP head.

    SpeculativeConfig.hf_config_override turns a target config into its draft config by
    model_type, and recognises Qwen3.5 only as "qwen3_5" (the multimodal wrapper). Text-only
    checkpoints are model_type "qwen3_5_text", so without this the draft keeps the target's
    architecture and vLLM would build a second full model as the "draft". The MTP head itself is
    the base model's, unmodified: standard attention, its own KV cache.
    """
    global _ORIG_HF_CONFIG_OVERRIDE
    from vllm.config.speculative import SpeculativeConfig
    if getattr(SpeculativeConfig, "_qwen_mla_mtp", False):
        return
    _ORIG_HF_CONFIG_OVERRIDE = SpeculativeConfig.hf_config_override
    SpeculativeConfig.hf_config_override = staticmethod(_mtp_hf_config_override)
    SpeculativeConfig._qwen_mla_mtp = True


def _mtp_draft_attention_backend():
    """Keep FULL decode CUDA graphs with MTP: put the draft's attention on Triton when vLLM would pick FlashInfer.

    The MTP head has one standard-attention layer, and vLLM auto-selects its backend independently of
    the target's (`speculative_config.attention_backend`, NOT the global --attention-backend). With an
    FP8 KV cache outside Hopper, FLASH_ATTN is ruled out and FLASHINFER wins -- whose CUDA-graph support
    with spec decode is single-token only, and vLLM lowers the whole engine to that common denominator:

        CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
        FlashInferBackend ...; setting cudagraph_mode=PIECEWISE

    so every attention call of the TARGET runs outside the graph, every step. TRITON_ATTN supports full
    graphs with spec decode; for one draft layer its kernel speed is immaterial. Measured on 2x RTX 5060 Ti
    (GLA-g2 EXL3, TP=2, MTP x3, 9k-token prompt): 47 -> 60 tok/s decode, identical acceptance.

    Only that case is changed: bf16 KV (FLASH_ATTN available) and Hopper are left to vLLM. An explicit
    `attention_backend` in --speculative-config always wins; QWEN_MLA_MTP_DRAFT_BACKEND=auto disables
    this, any other value names the backend to force.
    """
    import os, sys
    from vllm.v1.spec_decode import llm_base_proposer as _p
    cls = _p.SpecDecodeBaseProposer
    if getattr(cls._create_draft_vllm_config, "_qwen_mla", False):
        return
    orig = cls._create_draft_vllm_config

    def _create_draft_vllm_config(self):
        spec = self.speculative_config
        want = os.environ.get("QWEN_MLA_MTP_DRAFT_BACKEND", "").strip()
        if spec is not None and spec.attention_backend is None and want.lower() != "auto":
            archs = list(getattr(self.vllm_config.model_config, "architectures", None) or [])
            if any(a in ALL_ARCHS for a in archs) and getattr(spec, "method", None) == "mtp":
                pick = want or None
                if pick is None:
                    from vllm.platforms import current_platform
                    fp8 = str(self.vllm_config.cache_config.cache_dtype).startswith("fp8")
                    if fp8 and not current_platform.is_device_capability_family(90):
                        pick = "TRITON_ATTN"
                if pick:
                    from vllm.v1.attention.backends.registry import AttentionBackendEnum
                    spec.attention_backend = AttentionBackendEnum[pick.upper()]
                    print(f"[qwen-mla] MTP draft attention -> {pick.upper()} (keeps FULL decode CUDA graphs; "
                          "QWEN_MLA_MTP_DRAFT_BACKEND=auto to leave it to vLLM)", file=sys.stderr, flush=True)
        return orig(self)
    _create_draft_vllm_config._qwen_mla = True
    cls._create_draft_vllm_config = _create_draft_vllm_config


def _spec_decode_graph_capture_fix():
    """Capture spec-decode CUDA graphs with the linear-attention layers in spec-decode mode.

    GDN's metadata builder tells speculative rows (1 + k tokens each) from prefills by
    `num_decode_draft_tokens_cpu`, which the model runner fills from the scheduler -- but only for real
    steps. Capture goes through _dummy_run, which never touches the buffer, so it reads all zeros: the
    uniform (1 + k)-token decode batch being captured is classified as PREFILLS. The builder then takes
    the chunked-prefill branch, including a host-to-device copy that is illegal inside stream capture:

        torch.AcceleratorError: CUDA error: invalid argument   (gdn_attn.py build -> async_tensor_h2d)

    and when capture does go through, the replayed graph holds prefill-path kernels and buffers for what
    are spec-decode rows at run time -- measured as an intermittent startup crash and a mid-generation
    "unspecified launch failure". Reached whenever a GDN model is served with spec decode under FULL
    decode graphs, i.e. any attention backend with full spec-decode graph support (FLASH_ATTN with a bf16
    cache, or the Triton draft above). vLLM's stock Qwen3.5 has the same exposure; this is scoped to our
    architectures.

    Fix: before each dummy run, describe its rows the way the scheduler would -- a uniform-decode
    batch of (1 + k)-token rows is k drafts per request, one accepted token -- and everything else
    as non-speculative (-1).
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner._dummy_run, "_qwen_mla_spec", False):
        return
    orig = GPUModelRunner._dummy_run

    def _dummy_run(self, num_tokens, *args, **kwargs):
        if self.speculative_config is not None and hasattr(self, "num_decode_draft_tokens"):
            archs = list(getattr(self.model_config, "architectures", None) or [])
            if any(a in ALL_ARCHS for a in archs):
                uniform = kwargs.get("uniform_decode", args[2] if len(args) > 2 else False)
                qlen = self.uniform_decode_query_len
                self.num_decode_draft_tokens.np[:] = -1
                if uniform and qlen > 1:
                    reqs = min(self.scheduler_config.max_num_seqs, -(-num_tokens // qlen))
                    self.num_decode_draft_tokens.np[:reqs] = qlen - 1
                    if num_tokens % qlen:
                        self.num_decode_draft_tokens.np[reqs - 1] = num_tokens % qlen - 1
                    self.num_accepted_tokens.np[:reqs] = 1
                    self.num_accepted_tokens.copy_to_gpu()
        return orig(self, num_tokens, *args, **kwargs)
    _dummy_run._qwen_mla_spec = True
    GPUModelRunner._dummy_run = _dummy_run

    # The converse case at run time. vLLM picks the FULL "uniform decode" graph from the batch SHAPE
    # alone (every request scheduled exactly 1 + k tokens), so a prefill chunk of exactly 1 + k tokens
    # -- the tail of a prompt -- replays the graph captured above for spec-decode rows, and GDN runs its
    # verification kernels on prompt tokens: garbage from the first output token. Measured with MTP k=3
    # on a 20-token prompt split 16 + 4 by prefix caching; any prompt whose last chunk is 4 tokens
    # (e.g. 3204 tokens at --max-num-batched-tokens 3200) hits it too. Such batches now take the
    # mixed (PIECEWISE) path, where GDN's metadata builder sees the row as a prefill.
    orig_prepare = GPUModelRunner._prepare_inputs
    orig_determine = GPUModelRunner._determine_batch_execution_and_padding

    def _prepare_inputs(self, scheduler_output, *args, **kwargs):
        if self.speculative_config is not None:
            archs = list(getattr(self.model_config, "architectures", None) or [])
            if any(a in ALL_ARCHS for a in archs):
                sd = scheduler_output.scheduled_spec_decode_tokens
                self._qwen_mla_all_spec = all(len(sd.get(r, ())) + 1 == n
                                              for r, n in scheduler_output.num_scheduled_tokens.items())
        return orig_prepare(self, scheduler_output, *args, **kwargs)

    def _determine_batch_execution_and_padding(self, *args, **kwargs):
        if kwargs.get("force_uniform_decode") is None and getattr(self, "_qwen_mla_all_spec", True) is False:
            kwargs["force_uniform_decode"] = False
        self._qwen_mla_all_spec = True           # one-shot: dummy and capture runs never see a stale flag
        return orig_determine(self, *args, **kwargs)
    GPUModelRunner._prepare_inputs = _prepare_inputs
    GPUModelRunner._determine_batch_execution_and_padding = _determine_batch_execution_and_padding


def _prefix_cache_hit_granularity():
    """Default --prefix-match-unit for prefix caching (--mamba-cache-mode align) on our architectures.

    GLA-g2's MLA layers have different latent widths, and vLLM equalises their page sizes by giving each
    width its own block size: 1x, 2x and 4x the cache block (1600/3200/6400 tokens with MTP, 1568/3136/
    6272 without). With the default hash unit (the GCD = the cache block, which is also the linear-
    attention block) vLLM sees no unit finer than a GDN block and rounds every hit to the LCM -- 4 cache
    blocks. A 16k-token second turn reused 12.5k tokens without MTP and 6.4k with it (one 6400-token
    block lost to the Eagle drop, see below). A unit finer than the GDN block switches on vLLM's partial
    hits: full-attention groups match inside a block, and the scheduler ends one prefill chunk at the
    prompt's last unit boundary so the GDN state there is kept. The unit only sets where hits may land
    (one hash per unit), not how often state is stored. An explicit --prefix-match-unit wins.
    """
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig
    if getattr(HybridAttentionMambaModelConfig.verify_and_update_config, "_qwen_mla_pmu", False):
        return
    orig = HybridAttentionMambaModelConfig.verify_and_update_config.__func__

    def verify_and_update_config(cls, vllm_config):
        orig(cls, vllm_config)
        cc = vllm_config.cache_config
        archs = list(getattr(vllm_config.model_config, "architectures", None) or [])
        if (any(a in ALL_ARCHS for a in archs) and cc.enable_prefix_caching
                and cc.mamba_cache_mode == "align" and cc.prefix_match_unit is None):
            import sys
            cc.prefix_match_unit = 16        # hybrid block sizes (set later, at KV init) are multiples of 16
            print("[qwen-mla] prefix caching: hits land on 16-token boundaries", file=sys.stderr, flush=True)
    verify_and_update_config._qwen_mla_pmu = True
    HybridAttentionMambaModelConfig.verify_and_update_config = classmethod(verify_and_update_config)


def _prefix_cache_scheduler_fixes():
    """Two fixes to vLLM's hybrid prefix-cache lookup, applied to the scheduler of our architectures.

    1. Trim every full-attention group to the reconciled hit. vLLM looks a hit up group by group and
       shrinks it to what all groups can serve (the GDN layers only have state at chunk ends), then trims
       only the FIRST full-attention group's block list. GLA-g2 has four full-attention groups (three MLA
       latent widths with their own block sizes, plus the MTP layer), so the others kept blocks past the
       hit: the request's block table then holds shared cached blocks where it will write fresh KV, and
       with partial hits (see _prefix_cache_hit_granularity) the copy-on-write redirect asserts on the
       wrong tail block and kills the engine (single_type_kv_cache_manager._apply_cow).

    2. With MTP, keep the Eagle drop from shortening every hit. vLLM drops the last matched unit of a hit
       because the draft layer's KV at its last position was computed from the token AFTER it, which a
       new request may not share. When no KV-cache group is flagged as the drafter's (only DeepSeek-V4's
       is), vLLM applies the drop to EVERY group. Here that pushed the hit below the GDN state at the
       prompt tail and lost 2-3 prefill chunks per turn (16k turn 2: 12.8k reused instead of 15.6k), and
       before the finer hit unit a whole 6400-token block. The drop guards draft quality only: the MTP
       layer's KV feeds proposals that the target model verifies, so a stale draft entry at one position
       can cost some acceptance at worst, never change output; and the drafter computes its KV for every
       scheduled position in the step that schedules it, so a hit never reads uncomputed draft KV.
    """
    from vllm.v1.core.sched.scheduler import Scheduler
    if getattr(Scheduler.__init__, "_qwen_mla_pc", False):
        return
    orig = Scheduler.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        archs = list(getattr(self.vllm_config.model_config, "architectures", None) or [])
        coord = getattr(getattr(self, "kv_cache_manager", None), "coordinator", None)
        if coord is None or not hasattr(coord, "attention_groups") or not any(a in ALL_ARCHS for a in archs):
            return
        # In align mode every prefill chunk but a prompt's last is floored to the cache block, whose size
        # depends on the KV dtype, TP size and MTP (1600 tokens on 2 GPUs with FP8 KV and MTP). A budget
        # that is not a multiple of it leaves the remainder of every step's budget unused.
        bs, mnst = self.cache_config.block_size, self.max_num_scheduled_tokens
        if getattr(self, "need_mamba_block_aligned_split", False) and mnst > bs and mnst % bs:
            lo = mnst // bs * bs
            print(f"[qwen-mla] prefix caching: prefill chunks are multiples of the {bs}-token cache block, so "
                  f"--max-num-batched-tokens {mnst} prefills only {lo} tokens per step; use {lo} or {lo + bs}",
                  flush=True)
        from vllm.utils.math_utils import cdiv
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        full = [gid for g in coord.attention_groups if isinstance(g.spec, FullAttentionSpec) for gid in g.group_ids]
        find = coord.find_longest_cache_hit

        def find_longest_cache_hit(*a, **k):
            out = find(*a, **k)
            blocks, hit_length = out[0], out[1]
            for gid in full:
                del blocks[gid][cdiv(hit_length, coord.single_type_managers[gid].block_size):]
            return out
        coord.find_longest_cache_hit = find_longest_cache_hit

        spec = self.vllm_config.speculative_config
        if spec is not None and getattr(spec, "method", None) == "mtp":
            self.kv_cache_manager.use_eagle = False
            if hasattr(coord, "eagle_group_ids"):
                coord.eagle_group_ids = set()
            coord.attention_groups = [g._replace(use_eagle=False) for g in coord.attention_groups]
            for m in coord.single_type_managers:
                m.use_eagle = False
    __init__._qwen_mla_pc = True
    Scheduler.__init__ = __init__


def _fp8_kv_default_from_checkpoint():
    """`--kv-cache-dtype auto` -> fp8 for our checkpoints that ship calibrated FP8 KV.

    The quantized students declare an 8-bit float kv_cache_scheme and carry their own scales in
    `mla_kv_cache_scales`, meaning "serve with an FP8 cache unless told otherwise". vLLM honours such a
    declaration only for some quantization formats (compressed-tensors does, cuda-exl3 does not), so
    resolve it here for our architectures, whatever the weight format. An explicit --kv-cache-dtype always
    wins. BF16 parents carry scales but no kv_cache_scheme, so they stay BF16 by default.
    """
    import vllm.utils.torch_utils as tu
    if getattr(tu, "_qwen_mla_kv_default", False):
        return
    orig = tu.resolve_kv_cache_dtype_string

    def resolve_kv_cache_dtype_string(kv_cache_dtype, model_config):
        out = orig(kv_cache_dtype, model_config)
        if kv_cache_dtype != "auto" or out != "auto":
            return out
        hf = getattr(model_config, "hf_config", None)
        archs = getattr(hf, "architectures", None) or []
        if not any(a in ALL_ARCHS for a in archs):
            return out
        qc = getattr(hf, "quantization_config", None) or {}
        scheme = qc.get("kv_cache_scheme") if isinstance(qc, dict) else None
        tc = getattr(hf, "text_config", hf)
        if (isinstance(scheme, dict) and scheme.get("num_bits") == 8 and scheme.get("type") == "float"
                and getattr(tc, "mla_kv_cache_scales", None)):
            return "fp8"
        return out

    tu.resolve_kv_cache_dtype_string = resolve_kv_cache_dtype_string
    import vllm.engine.arg_utils as au        # imported the name directly
    au.resolve_kv_cache_dtype_string = resolve_kv_cache_dtype_string
    tu._qwen_mla_kv_default = True


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
