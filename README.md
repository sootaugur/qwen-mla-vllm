# qwen-mla-vllm

A [vLLM](https://github.com/vllm-project/vllm) plugin that serves the **training-free MLA retrofits
of Qwen3.8-27B** with a compressed **latent KV cache**:

| model | what it is | best for |
|---|---|---|
| [TelperionAI/Qwen3.8-27B-MLA](https://huggingface.co/TelperionAI/Qwen3.8-27B-MLA) | shared latent (Multi-head Latent Attention) | 1 GPU (TP=1) |
| [TelperionAI/Qwen3.8-27B-GLA-g2](https://huggingface.co/TelperionAI/Qwen3.8-27B-GLA-g2) | latent split into 2 head groups | 2 GPUs (TP=2) |

Both cache **half the KV bytes per token** of the base model. Installing the plugin is all vLLM needs
to load them; no fork, no flags.

## Quick start

```bash
pip install "git+https://github.com/sootaugur/qwen-mla-vllm"      # also installs vllm==0.27.1
vllm serve TelperionAI/Qwen3.8-27B-MLA --reasoning-parser qwen3
```

Two GPUs, for the grouped model:

```bash
vllm serve TelperionAI/Qwen3.8-27B-GLA-g2 --tensor-parallel-size 2 --reasoning-parser qwen3
# PCIe-only GPUs (no NVLink: RTX PRO, GeForce): add --disable-custom-all-reduce
```

Test request:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "TelperionAI/Qwen3.8-27B-MLA", "messages": [{"role": "user", "content": "What is 17*23?"}]}'
```

The first start JIT-compiles the fast decode kernel (a few minutes); later starts reuse it.

## Requirements

* **NVIDIA GPU** with vLLM 0.27.1 support. Tested on RTX PRO 6000 (Blackwell, sm_120) and H100 (sm_90).
* **vLLM 0.27.1** exactly (installed as a dependency). The plugin hooks vLLM internals that change
  between releases.
* **For the fast decode path: the CUDA toolkit (`nvcc`) and `ninja`**, used to JIT-compile a patched
  FlashInfer MLA kernel on first use. Without them the plugin logs one warning and decodes with its
  Triton kernels instead: correct, somewhat slower.
* Python 3.12.

## How it works

The checkpoints declare the architecture `Qwen3_5MLAForCausalLM`; this plugin registers it through
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

## Configuration

Nothing is required. Optional environment variables:

| variable | default | effect |
|---|---|---|
| `QWEN_MLA_DECODE_BACKEND` | `flashinfer` | `triton` disables the FlashInfer fast path |
| `QWEN_MLA_TWO_PASS` | `1` | `0` disables the two-pass kernel for the 1792 latent |
| `QWEN_MLA_2P_SCRATCH_MB` | `256` | scratch budget of the two-pass kernel; very long contexts process keys in rounds when exceeded |
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
```

## Limitations

* vLLM 0.27.1 only, for now.
* Decode context parallelism (DCP) is not supported.
* Tested at TP=1 and TP=2. The GLA per-group cache engages when the tensor-parallel size is a multiple
  of the group count (2); at TP=1 GLA-g2 runs like plain MLA. TP=4 should work (each pair of GPUs
  shares a group) but has not been tested.

## License

Apache-2.0. The models are derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(Apache-2.0). Independent project, not affiliated with the Qwen team.
