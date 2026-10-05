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

* FP8 weights (about 30 GB instead of 54 GB; same KV cache): use `TelperionAI/Qwen3.8-27B-MLA-FP8` or
  `TelperionAI/Qwen3.8-27B-GLA-g2-FP8` with the same flags. They need a GPU with FP8 support (Hopper or
  Blackwell).
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
