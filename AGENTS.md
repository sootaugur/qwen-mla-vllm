# AGENTS.md — installing and serving qwen-mla-vllm

Instructions for coding agents setting this up on a user's machine. Follow them in order.

## 1. Preconditions

```bash
nvidia-smi                      # an NVIDIA GPU and driver must be present
python3 --version               # 3.12 expected
nvcc --version || echo "no nvcc: the server will work, using the slower Triton decode path"
nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit --format=csv
```

RTX 50-series GPUs (compute capability 12.0) need `nvcc` from CUDA 12.8 or newer. If the system
toolkit is missing or older and you cannot install one system-wide, install a user-space toolkit from
pip as shown in the README ("No system CUDA toolkit" under "16 GB GPUs"). It is required for EXL3 builds
and for the fast decode path, and `CUDA_HOME`/`PATH` must stay set when serving.

Use a fresh virtual environment. This package pins `vllm==0.27.1`; do not install it into an
environment that needs a different vLLM version.

## 2. Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "git+https://github.com/sootaugur/qwen-mla-vllm"
```

## 3. Serve

One GPU:

```bash
vllm serve TelperionAI/Qwen3.8-27B-MLA --reasoning-parser qwen3
```

Two GPUs:

```bash
vllm serve TelperionAI/Qwen3.8-27B-GLA-g2 --tensor-parallel-size 2 --reasoning-parser qwen3
```

* Optional: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` (MTP speculative decoding).
  Measured 1.6-2.1x decode on 2x RTX 5060 Ti (GLA-g2-EXL3) and 1.3-1.7x for bf16/FP8 builds on an
  RTX PRO 6000. It costs KV capacity (the draft has its own cache). Recommend it for single-user,
  latency-bound serving; measure first for high-concurrency serving.
* Tool calling (agent harnesses): `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
* Agent loops, chat, or a long shared system prompt: add `--enable-prefix-caching --mamba-cache-mode align`
  so repeated prefixes are not prefilled again (a 37.7k-token second turn: 52 s -> 0.6 s to first token).
  Costs ~7% of KV capacity; with MTP, no measurable decode speed since 0.6.5 (measured on 2x RTX 5060 Ti). Give `--max-num-batched-tokens` a multiple of
  the cache block, which varies with KV dtype, TP size and MTP (1600 on 2 GPUs, FP8 KV, MTP); if it is not, the plugin prints
  `[qwen-mla] prefix caching: prefill chunks are multiples of ...` with the values to use. Do not set `--prefix-match-unit`;
  the plugin picks it.
* The models accept images like the base model. If the user only needs text, add
  `--limit-mm-per-prompt '{"image": 0, "video": 0}'`.
* Pick the weight format by VRAM per GPU (MLA for 1 GPU, GLA-g2 for 2):
  bf16 `Qwen3.8-27B-MLA` (55.7 GB), `-MLA-FP8` (31.5 GB), `-MLA-INT4` (22.9 GB),
  `-MLA-EXL3-4.0bpw` (17.2 GB). Same names with `GLA-g2` for two GPUs. FP8 needs Hopper or Blackwell.
* The quantized builds default to an FP8 KV cache (calibrated); bf16 builds default to bf16 KV and take
  `--kv-cache-dtype fp8` for about 2x the capacity at long context.
* **EXL3 builds need a second plugin**, cuda-exl3 (https://github.com/Zeuss5/cuda-exl3). Install it after
  this one: `pip install --no-build-isolation "git+https://github.com/Zeuss5/cuda-exl3"` (compiles CUDA
  code; needs `nvcc`; takes a few minutes).
* **Two 16 GB GPUs** (e.g. RTX 5060 Ti): `TelperionAI/Qwen3.8-27B-GLA-g2-EXL3-4.0bpw` at TP=2 serves the full
  262k context with vision and MTP using the exact command in the README section "16 GB GPUs"
  (`--max-num-seqs 8 --max-num-batched-tokens 3200 --gpu-memory-utilization 0.88`, prefix caching on). Do not add
  `--kv-cache-memory-bytes`; the plugin sizes the cache correctly.
* **24 GB GPU:** use `TelperionAI/Qwen3.8-27B-MLA-EXL3-4.0bpw` text-only:
  `--limit-mm-per-prompt '{"image": 0, "video": 0}' --max-model-len 32768`, no `--speculative-config`.
  **32 GB:** `-MLA-INT4` with the same flags.
* If the GPUs are connected only over PCIe (no NVLink: RTX PRO, GeForce), add
  `--disable-custom-all-reduce`. Without it, multi-GPU startup can hang.
* The first start JIT-compiles a CUDA kernel and can take several minutes. Do not treat a slow
  first start as a hang; watch the log.
* The models are ~51 GB (bf16) and download from Hugging Face on first use.

## 4. Verify

Wait for `Application startup complete` in the log, then:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "TelperionAI/Qwen3.8-27B-MLA", "messages": [{"role": "user", "content": "What is 17*23?"}]}'
```

The answer should contain `391`. In the server log, check for:

* `[qwen-mla] register() called` — the plugin loaded.
* `MLA decode: flashinfer MLA kernel active` — the fast path is on. If instead you see
  `flashinfer path unavailable (...)`, report the reason in parentheses to the user; serving still works.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `Model architectures ['Qwen3_5MLAForCausalLM'] are not supported` | the plugin is not installed in the environment running `vllm`; reinstall, and do not set `VLLM_PLUGINS` to a list that excludes `qwen_mla` |
| hang or `custom_all_reduce ... invalid argument` at TP>1 | add `--disable-custom-all-reduce` |
| `flashinfer path unavailable (... nvcc ...)` | install the CUDA toolkit (matching the driver) and `ninja`, then restart; or accept the Triton path |
| `the shipped kernel headers target flashinfer X, installed is Y` | a different flashinfer was installed; reinstall in a fresh venv |
| out of memory at startup | lower `--max-model-len` or `--gpu-memory-utilization`; on 16-24 GB GPUs add `--max-num-seqs 8 --max-num-batched-tokens 2048` (with prefix caching, a multiple of the cache block instead, e.g. 1600) |
| `To serve at least one request with the model's max seq len ... KV cache is needed` | the KV cache cannot hold one `--max-model-len` sequence; lower it, drop MTP, or set `--limit-mm-per-prompt '{"image": 0, "video": 0}'` |
| `FlashInfer requires GPUs with sm75 or higher` on an RTX 50-series GPU | misleading: FlashInfer's JIT found an `nvcc` too old for sm_120. Put CUDA >= 12.8 first on `PATH` and set `CUDA_HOME` |
| `ptxas ... Unsupported .version` while building cuda-exl3 | mixed CUDA versions in the pip toolkit (`nvidia-nvvm` newer than `nvidia-cuda-nvcc`); pin all to the same minor version |
| `Xid 79, GPU has fallen off the bus` in `journalctl -k`, server dies mid-generation | hardware: power or PCIe link. Ask the user to cap power (`sudo nvidia-smi -pl 150` on an RTX 5060 Ti) or force the slot to PCIe 4.0; do not retry heavy load until they have |
| `AssertionError` in `_apply_cow` (engine dies on a prefix-cache hit) | plugin older than 0.6.1; upgrade |
| with MTP, an occasional prompt answers with garbage from the first token (`!!!!`, stray words) | plugin older than 0.6.1 (a final prefill chunk of exactly 1 + k tokens replayed the decode graph); upgrade |
| MTP decode falls off sharply at long context (2x RTX 5060 Ti, k=3: 64 tok/s at 1k, 44 at 81k) | plugin older than 0.6.3 with an FP8 cache: every draft token re-read the latent cache. Upgrade; 0.6.3 gives 55 at 81k |
| prefix caching on: long prompts are preempted (`GPU KV cache usage` hits 100%) with far fewer tokens resident than `Maximum concurrency` at startup promised | plugin older than 0.6.4: every prefill chunk leaked linear-attention state blocks until the request finished; upgrade |
| `ValueError: Unsupported base layer` at startup with `--enable-lora` on an EXL3 build | plugin older than 0.6.6; upgrade |
| prefix caching on but `Prefix cache hit rate` stays low across turns of one conversation | the harness changes earlier messages (timestamps, reordering) so the prefix differs; nothing to fix server-side |
| tok/s reported by a client is about half the server log's `Avg generation throughput` with MTP | the client counts stream chunks, and MTP sends several tokens per chunk; trust the server log |
