# qwen-mla-vllm

> **Experimental.** Verified on RTX PRO 6000 (Blackwell) at TP=1 and TP=2 with vLLM 0.27.1;
> little real-world use yet. Bug reports welcome.

A [vLLM](https://github.com/vllm-project/vllm) plugin that serves the **training-free MLA retrofits
of Qwen3.8-27B** with a compressed **latent KV cache**:

Two retrofits -- **MLA** (one shared latent; for 1 GPU) and **GLA-g2** (latent split into 2 head groups,
so it also halves per-GPU KV at TP=2; for 2 GPUs) -- each in four weight formats:

| weights | MLA (TP=1) | GLA-g2 (TP=2) | size | KV cache default |
|---|---|---|---|---|
| bf16 | [Qwen3.8-27B-MLA](https://huggingface.co/TelperionAI/Qwen3.8-27B-MLA) | [Qwen3.8-27B-GLA-g2](https://huggingface.co/TelperionAI/Qwen3.8-27B-GLA-g2) | 55.7 GB | bf16 (FP8 calibrated, opt-in) |
| FP8 | [-MLA-FP8](https://huggingface.co/TelperionAI/Qwen3.8-27B-MLA-FP8) | [-GLA-g2-FP8](https://huggingface.co/TelperionAI/Qwen3.8-27B-GLA-g2-FP8) | 31.5 GB | FP8 |
| INT4 (AWQ+GPTQ) | [-MLA-INT4](https://huggingface.co/TelperionAI/Qwen3.8-27B-MLA-INT4) | [-GLA-g2-INT4](https://huggingface.co/TelperionAI/Qwen3.8-27B-GLA-g2-INT4) | 22.9 GB | FP8 |
| EXL3 4.0 bpw | [-MLA-EXL3-4.0bpw](https://huggingface.co/TelperionAI/Qwen3.8-27B-MLA-EXL3-4.0bpw) | [-GLA-g2-EXL3-4.0bpw](https://huggingface.co/TelperionAI/Qwen3.8-27B-GLA-g2-EXL3-4.0bpw) | 17.2 GB | FP8 |

All cache **half the KV bytes per token** of the base model (a quarter with FP8 KV), and all keep the
base model's **vision tower** and **MTP head** (for speculative decoding). Installing the plugin is all
vLLM needs to load them; the EXL3 builds also need [cuda-exl3](https://github.com/Zeuss5/cuda-exl3)
(see [EXL3](#exl3-builds)).

## Quick start

```bash
pip install "git+https://github.com/sootaugur/qwen-mla-vllm"      # also installs vllm==0.27.1
vllm serve TelperionAI/Qwen3.8-27B-MLA --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

Two GPUs, for the grouped model:

```bash
vllm serve TelperionAI/Qwen3.8-27B-GLA-g2 --tensor-parallel-size 2 --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
# PCIe-only GPUs (no NVLink: RTX PRO, GeForce): add --disable-custom-all-reduce
```

`--speculative-config` turns on MTP speculative decoding (optional; see [below](#speculative-decoding-mtp)).
Images work as with the base model (OpenAI `image_url` content). For text-only serving, add
`--limit-mm-per-prompt '{"image": 0, "video": 0}'` to skip reserving memory for the vision encoder.

Test request:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "TelperionAI/Qwen3.8-27B-MLA", "messages": [{"role": "user", "content": "What is 17*23?"}]}'
```

The first start JIT-compiles the fast decode kernel (a few minutes); later starts reuse it.

## FP8 KV cache

Every checkpoint ships calibrated per-layer FP8 KV scales (absmax over a 3.7M-token corpus, measured in
the served model). The quantized builds use an FP8 cache by default; the bf16 builds use bf16 and switch
with `--kv-cache-dtype fp8`. Either default can be overridden (`--kv-cache-dtype bfloat16` / `fp8`).

* **Quality:** indistinguishable from run-to-run noise in teacher-forced tests (top-1 disagreement with
  the bf16-KV run equals that between two bf16-KV runs). The per-head score divisor stored with each row
  is kept as a 1-byte log2 code (0.8% max error) rather than e4m3 (6%).
* **Capacity** (one 96 GB GPU, 262k-token requests, vision on): bf16 base 1.9 sequences; MLA bf16 with
  bf16 / FP8 KV 3.7 / 7.3 (3.8x the base); MLA-FP8 12.7, -INT4 14.5, -EXL3 15.6. At shorter requests the
  gain is smaller (16k: about 1.5x from FP8 KV) because each sequence also holds a fixed linear-attention state.
* **Speed:** FP8-cache decode runs on the plugin's Triton kernels (FlashInfer's FP8 MLA kernel is SM90 +
  DeepSeek dims only), about 15-25% slower at moderate concurrency than the bf16 FlashInfer path.

## EXL3 builds

The EXL3 checkpoints are graft-built from TelperionAI/Qwen3.8-27B-EXL3-4.0bpw-v2 and run on
**[cuda-exl3](https://github.com/Zeuss5/cuda-exl3)**, a separate vLLM plugin providing the EXL3 runtime
(batched trellis kernels). Install it next to this one; it compiles CUDA code on install (needs `nvcc`,
a few minutes):

```bash
pip install "git+https://github.com/sootaugur/qwen-mla-vllm"
pip install --no-build-isolation "git+https://github.com/Zeuss5/cuda-exl3"
vllm serve TelperionAI/Qwen3.8-27B-MLA-EXL3-4.0bpw --reasoning-parser qwen3
```

On a **24 GB GPU** serve it text-only (`--limit-mm-per-prompt '{"image": 0, "video": 0}'`, FP8 KV is the
default, no MTP): about 106k tokens of KV at 32k max context, measured under an equivalent memory budget.
Vision and MTP need more memory than a 24 GB card leaves. The INT4 builds fit a 32 GB card text-only
(about 250k tokens of KV at 32k context).

## Requirements

* **NVIDIA GPU** with vLLM 0.27.1 support. This version is tested on RTX PRO 6000
  (Blackwell, sm_120); earlier versions ran on H100 (sm_90).
* **vLLM 0.27.1** exactly (installed as a dependency). The plugin hooks vLLM internals that change
  between releases.
* **For the fast decode path: the CUDA toolkit (`nvcc`) and `ninja`**, used to JIT-compile a patched
  FlashInfer MLA kernel on first use. Without them the plugin logs one warning and decodes with its
  Triton kernels instead: correct, somewhat slower.
* Python 3.12.

## How it works

The checkpoints declare the architecture `Qwen3_5MLAForConditionalGeneration` (base model's vision tower in
front of the MLA language model; text-only checkpoints use `Qwen3_5MLAForCausalLM`); this plugin registers it through
vLLM's `vllm.general_plugins` entry point, so vLLM finds it automatically. The 16 full-attention
layers cache a per-layer latent (ranks 256 / 768 / 1792) plus a small RoPE and normalisation tail,
instead of per-head keys and values. The other 48 layers (Gated DeltaNet linear attention), the MLPs
and the embeddings are the unmodified Qwen3.8-27B.

Decode kernels, chosen automatically per layer:

* **FlashInfer MLA (patched)**: the fast path, JIT-compiled from headers shipped in this package.
* **Two-pass Triton**: the widest latent (1792) on GPUs with 99 KB of shared memory per SM
  (consumer and workstation Blackwell), where a single-pass kernel cannot fit.
* **Triton**: the fallback for anything else.

For **GLA-g2 at TP=2**, each GPU caches only its own head group's half of the latent, so per-GPU KV
is halved under tensor parallelism as well. (Plain MLA replicates its latent on every GPU.)

## Performance

RTX PRO 6000 (Blackwell), bf16 weights and KV, vLLM 0.27.1, 16k-token prompts, 256 decode steps.
"Max" is each model's own largest batch that fits in KV memory.

**TP=1**

| | max concurrent 16k sequences | batch 1 (ms/token) | batch 16 (ms/step) | throughput at max batch |
|---|---|---|---|---|
| Qwen3.8-27B (base) | 24 | 38.4 | 54.4 | 378 tok/s |
| **Qwen3.8-27B-MLA** | **40** | 40.0 | 56.1 | **505 tok/s** |

**TP=2**

| | max concurrent 16k sequences | batch 1 (ms/token) | batch 92 (ms/step) | throughput at max batch |
|---|---|---|---|---|
| Qwen3.8-27B (base) | 92 | 22.0 | 88.2 | 1,043 tok/s |
| **Qwen3.8-27B-GLA-g2** | **166** | 23.5 | 75.2 | **1,424 tok/s** |

Prefill throughput matches the base model (within 3%).

### Speculative decoding (MTP)

The models keep the base model's MTP head unchanged, and it drafts for the retrofit as well as it does
for the base model: the same mean accepted tokens per step (k=3: 1.86 vs 1.82). Decode throughput,
tokens/s at 1 / 8 / 32 concurrent requests (64 mixed prompts, thinking on, temperature 0.6, top-p 0.95, top-k 20,
bf16):

| | no MTP | MTP, k=3 | speedup |
|---|---|---|---|
| Qwen3.8-27B, TP=1 | 26 / 155 / 547 | 57 / 326 / 972 | 2.16× / 2.10× / 1.78× |
| **Qwen3.8-27B-MLA, TP=1** | 25 / 129 / 548 | 47 / 227 / 858 | 1.85× / 1.76× / 1.57× |
| Qwen3.8-27B, TP=2 | 45 / 240 / 850 | 91 / 431 / 1,393 | 2.03× / 1.79× / 1.64× |
| **Qwen3.8-27B-GLA-g2, TP=2** | 42 / 217 / 846 | 70 / 341 / 1,180 | 1.65× / 1.57× / 1.39× |

The speedup is smaller than the base model's because a verify step currently reads the latent cache once
per draft token; a single causal pass per request is planned. Greedy output with MTP differs from greedy
output without it no more than it does for the base model.

### Vision

Paired against the base model on the same items, greedy, thinking off (300 ChartQA test questions, relaxed
accuracy; 300 DocVQA validation questions, ANLS):

| | ChartQA | DocVQA |
|---|---|---|
| Qwen3.8-27B | 89.7 | 97.1 |
| **Qwen3.8-27B-MLA** | 88.0 (−1.7 [−4.3, +1.0]) | 97.3 (+0.1 [−1.0, +1.3]) |
| **Qwen3.8-27B-GLA-g2** (TP=2) | 87.7 (−2.0 [−5.3, +1.0]) | 96.6 (−0.5 [−2.2, +1.1]) |

No difference is significant; 84–95% of answers are identical to the base model's.

## Configuration

Nothing is required. Optional environment variables:

| variable | default | effect |
|---|---|---|
| `QWEN_MLA_DECODE_BACKEND` | `flashinfer` | `triton` disables the FlashInfer fast path |
| `QWEN_MLA_TWO_PASS` | `1` | `0` disables the two-pass kernel for the 1792 latent |
| `QWEN_MLA_2P_SCRATCH_MB` | `256` | scratch budget of the two-pass kernel; very long contexts process keys in rounds when exceeded |
| `QWEN_MLA_FP8_RMS` | `log8` | FP8 cache: format of the per-head score divisor (`log8`, `bf16` = wider rows and no MTP, `e4m3`) |
| `QWEN_MLA_SPLITK_SCRATCH_MB` | `512` | cap on the Triton decode's split-K scratch; fewer key splits at very large batch x context |
| `QWEN_MLA_GLA_SHARD` | `1` | `0` keeps the full latent on every GPU for grouped models (A/B testing) |
| `QWEN_MLA_CACHE` | `~/.cache/qwen-mla` | where the kernel overlay and compiled modules live |
| `QWEN_MLA_FI_DEBUG` | unset | `1` logs which decode kernel each layer uses, and why |

Earlier `MVLA_*` names are still accepted, as are checkpoints that use the earlier architecture
name `Qwen3_5MVLAAbsorbedForCausalLM` and `mvla_*` config keys.

## Verifying an install

During startup the server log should contain `[qwen-mla] register() called` and, once the decode
kernels are set up (CUDA-graph capture),

```
MLA decode: flashinfer MLA kernel active (overlay ...)
```

If it says `flashinfer path unavailable (...); using the Triton kernel`, the reason in parentheses
says why (usually a missing `nvcc`). Generation still works.

Kernel correctness tests (need a GPU):

```bash
python tests/test_fi_partial_rope.py --layout tp1 --rms-spread --page 784 --shuffle --ragged
python tests/test_two_pass_1792.py
python tests/test_decode_numerics.py
python tests/test_mrope.py          # CPU only
```

## Limitations

* vLLM 0.27.1 only, for now.
* At TP=2, startup can sit in CUDA-graph capture for several minutes (the log repeats "No available
  shared memory broadcast block"); it completes. Starting two TP=2 servers at the same moment on one
  machine has hung during capture; start them one after another.
* Decode context parallelism (DCP) is not supported.
* FP8 KV decode runs on Triton kernels (slower than the bf16 FlashInfer path).
* The INT4 and EXL3 builds reuse the base model's quantized tensors (their calibration saw the base
  model's activations); only the 16 attention layers' latent projections come from the retrofit, in bf16.
* Tested at TP=1 and TP=2. The GLA per-group cache engages when the tensor-parallel size is a multiple
  of the group count (2); at TP=1 GLA-g2 runs like plain MLA. TP=4 should work (each pair of GPUs
  shares a group) but has not been tested.

## License

Apache-2.0. The models are derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(Apache-2.0). Independent project, not affiliated with the Qwen team.
