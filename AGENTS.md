# AGENTS.md — installing and serving qwen-mla-vllm

Instructions for coding agents setting this up on a user's machine. Follow them in order.

## 1. Preconditions

```bash
nvidia-smi                      # an NVIDIA GPU and driver must be present
python3 --version               # 3.12 expected
nvcc --version || echo "no nvcc: the server will work, using the slower Triton decode path"
```

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
  On an RTX PRO 6000 it sped up the bf16 and FP8 builds (1.3-1.7x at low concurrency) but not the INT4 or
  EXL3 builds. The benefit depends on the GPU (likely larger on lower-bandwidth cards), so leave it off
  unless the user asks, or measure first.
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
| out of memory at startup | lower `--max-model-len` or `--gpu-memory-utilization` |
