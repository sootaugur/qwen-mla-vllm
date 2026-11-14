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

`--speculative-config` turns on MTP speculative decoding (optional\*; see [below](#speculative-decoding-mtp)).
Images work as with the base model (OpenAI `image_url` content). For text-only serving, add
`--limit-mm-per-prompt '{"image": 0, "video": 0}'` to skip reserving memory for the vision encoder.
For tool calling (agent harnesses), add `--enable-auto-tool-choice --tool-call-parser qwen3_coder`: the
chat template emits the base model's `<tool_call><function=...>` XML.

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

## 16 GB GPUs

Two 16 GB GeForce cards (tested: 2x RTX 5060 Ti, PCIe, no NVLink) serve the grouped EXL3 build at the
**full 262k context with vision and MTP**:

```bash
vllm serve TelperionAI/Qwen3.8-27B-GLA-g2-EXL3-4.0bpw --tensor-parallel-size 2 --disable-custom-all-reduce \
  --reasoning-parser qwen3 --max-model-len 262144 --gpu-memory-utilization 0.88 \
  --max-num-seqs 8 --max-num-batched-tokens 3200 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --enable-prefix-caching --mamba-cache-mode align
```

* `--max-num-seqs 8 --max-num-batched-tokens 3200` shrink the activation memory vLLM reserves before
  sizing the KV cache (40% more KV than vLLM's defaults in a text-only test). 3200 is two 1600-token
  cache blocks (the block size on this setup): with prefix caching, prefill chunks end on block
  boundaries, so a budget that is not a multiple of the block wastes the remainder of every chunk.
* `--enable-prefix-caching --mamba-cache-mode align`: see [Prefix caching](#prefix-caching). Makes repeat
  turns of an agent loop nearly free, for about 7% of the KV cache. With MTP, decode costs nothing measurable
  since 0.6.5 (it was ~15% before); see [Prefix caching](#prefix-caching).
* `--gpu-memory-utilization 0.88` leaves room for a desktop on the GPU that drives the display.
* Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment (less fragmentation).

Measured (FP8 KV, the default for this build; 8.7 GiB of weights per GPU; decode without prefix caching):

| | KV cache | 262k sequences | decode, short prompt | decode, 9k prompt + 2k-token story | decode at 29k context |
|---|---|---|---|---|---|
| no MTP | 463k tokens (432k with prefix caching) | 1.77 | 36 tok/s | 36 tok/s | 34 tok/s |
| MTP, k=3 | 305k tokens (283k with prefix caching) | 1.16 | 68-75 tok/s | 56-58 tok/s | 62-64 tok/s (with prefix caching) |

With MTP and prefix caching, decode at 81k context runs at 55 tok/s greedy (50-54 with the default
sampling). Before 0.6.3 a verify step read the latent cache once per draft token and it was 44; reading it
once per request costs about 1-3% at short context, inside run-to-run noise. For comparison, cyankiwi's
AWQ-INT4 quant of the base model on the same cards and vLLM, with MTP k=3 and prefix caching, decodes at
57 / 63 / 56 tok/s at 1k / 29k / 81k context, but fits only 100k tokens of context, text-only.

A 251k-token prompt with an image attached ran to completion (needle and image both answered correctly),
peaking at 14.8 / 14.1 GB per GPU. Prefill is about 750 tokens/s and slows with context (251k tokens took
9 minutes). With prefix caching, the next turn of a conversation only prefills what is new: a 37.7k-token
second turn started answering after 0.6 s instead of 52 s.

Home-hardware notes:

* **Prefill speed is bound by the link between the GPUs.** GeForce cards cannot reach each other's memory
  directly, so tensor parallelism all-reduces through host memory over PCIe. With one card in a chipset
  slot (PCIe 3.0 x4 on the tested B660 board, 3.6 GB/s) the all-reduce took ~48% of prefill time;
  the GPUs drew ~100 W of their 150 W cap and raising the cap to 180 W changed nothing. A second slot
  wired to the CPU (x8/x8 boards, or an M.2-to-PCIe adapter in a CPU-attached M.2 slot) is the fix;
  `--max-num-batched-tokens` above 3200 did not help.
* **Tool calling** for agent harnesses: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
* **Streaming with MTP** delivers one chunk per decode step, carrying all tokens accepted in it (about 2-3).
  A client that counts chunks as tokens reports the step rate, roughly half the real speed.
* **No system CUDA toolkit, or one older than 12.8?** Both the cuda-exl3 build and FlashInfer's JIT need
  an `nvcc` that knows your GPU (sm_120 for RTX 50-series). A user-space toolkit from pip works, matching
  the CUDA version torch was built for (`python -c "import torch; print(torch.version.cuda)"`, 13.0 for
  vLLM 0.27.1):

  ```bash
  pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-cuda-runtime==13.0.*"
  export CUDA_HOME=$(python -c "import nvidia, os; print(os.path.join(nvidia.__path__[0], 'cu13'))")
  export PATH=$CUDA_HOME/bin:$PATH
  ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so; ln -sfn lib $CUDA_HOME/lib64
  TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation "git+https://github.com/Zeuss5/cuda-exl3"
  ```

  Keep `CUDA_HOME` and `PATH` set when serving. All four packages must be the same minor version (a newer
  `nvidia-nvvm` produces PTX the older `ptxas` rejects: `Unsupported .version`).
* **`Xid 79, GPU has fallen off the bus`** under sustained MTP load (seen at the 180 W stock limit on an
  RTX 5060 Ti in a PCIe 5.0 slot) is a power or link-integrity event, not a software fault. Capping power
  (`sudo nvidia-smi -pl 150`, reapplied at boot) stopped it; decode speed was unaffected. Forcing the slot to
  PCIe 4.0 in the BIOS is the other usual fix.

## Requirements

* **NVIDIA GPU** with vLLM 0.27.1 support. Tested on RTX PRO 6000 and RTX 5060 Ti
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

### Speculative decoding (MTP)\*

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

The speedup in this table is smaller than the base model's because a verify step read the latent cache once
per draft token. Since 0.6.3 the Triton decode kernels, used with an FP8 cache, read it once per request
(at TP=1, once per two draft tokens), which keeps MTP decode from falling off at long context; see
[16 GB GPUs](#16-gb-gpus). The FlashInfer decode path, used with a bf16 cache, still reads it once per
draft token. Greedy output with MTP differs from greedy output without it no more than it does for the
base model.

\*The table was measured on one GPU type (RTX PRO 6000, very high memory bandwidth) with a bf16 KV cache.
With the FP8 KV cache there, k=3 gave 1.6x (bf16 build) and 1.3x (FP8 build) at one concurrent request
and did not speed up the INT4 or EXL3 builds. Those FP8-cache numbers predate 0.6.0: an FP8 cache sent
the draft's attention layer to FlashInfer, which made vLLM drop the whole engine to piecewise CUDA graphs
with spec decode. 0.6.0 keeps full graphs (the draft layer runs on Triton), and on 2x RTX 5060 Ti the
GLA-g2-EXL3 build now gains 1.6-2.1x from k=3 (see [16 GB GPUs](#16-gb-gpus)). The RTX PRO 6000 FP8-cache
numbers have not been re-measured; measure on your hardware.

Plugin versions 0.6.0 and earlier with MTP give garbage output for a prompt whose last prefill chunk is
exactly 1 + k tokens (k=3: e.g. 3204 tokens at `--max-num-batched-tokens 3200`) when it runs alone: vLLM
picks the full decode CUDA graph by batch shape and replays spec-decode kernels on prompt tokens.
Upgrade to 0.6.1.

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
| `QWEN_MLA_TWO_PASS` | `1` | `0` disables the two-pass kernel for the 1792 latent (spec-verify steps use it regardless) |
| `QWEN_MLA_2P_SCRATCH_MB` | `256` (`64` for spec-verify steps) | scratch budget of the two-pass kernel; very long contexts process keys in rounds when exceeded |
| `QWEN_MLA_GROUPED_VERIFY` | `1` | Triton decode with MTP: `0` gives each draft token its own kernel row, re-reading the latent cache per token. Grouping was measured on 99 KB-shared-memory cards (RTX 50-series); on others, compare both |
| `QWEN_MLA_VERIFY_MAX_ROWS` | `64` | most query rows (draft tokens x local heads) one grouped program serves; above it a request's tokens split into smaller groups |
| `QWEN_MLA_FP8_RMS` | `log8` | FP8 cache: format of the per-head score divisor (`log8`, `bf16` = wider rows and no MTP, `e4m3`) |
| `QWEN_MLA_SPLITK_SCRATCH_MB` | `512` | cap on the Triton decode's split-K scratch; fewer key splits at very large batch x context |
| `QWEN_MLA_GLA_SHARD` | `1` | `0` keeps the full latent on every GPU for grouped models (A/B testing) |
| `QWEN_MLA_ALIGN_FREE` | `1` | `0` keeps vLLM's linear-attention state blocks for a whole prefill under `--mamba-cache-mode align` (see [Prefix caching](#prefix-caching)) |
| `QWEN_MLA_GDN_META_CACHE` | on with MTP + `--mamba-cache-mode align` | build the linear-attention metadata once per step instead of once per layer (see [Prefix caching](#prefix-caching)); `1` forces it on, `0` off |
| `QWEN_MLA_PREFILL_WS_ROWS` | 1024 per GiB of GPU memory, max 64k | context rows per long-prompt prefill chunk; scratch is ~30 KB per row |
| `QWEN_MLA_MTP_DRAFT_BACKEND` | Triton with an FP8 cache (not Hopper) | attention backend of the MTP draft layer; `auto` leaves it to vLLM (see [MTP](#speculative-decoding-mtp)) |
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

With MTP and an FP8 cache, also expect `[qwen-mla] MTP draft attention -> TRITON_ATTN`, and
`Capturing CUDA graphs (decode, FULL)`. If instead the log says `setting cudagraph_mode=PIECEWISE`,
decode runs well below its speed (an explicit `attention_backend` in `--speculative-config` overrides
the plugin's choice).

With prefix caching, expect `[qwen-mla] prefix caching: hits land on 16-token boundaries`; the server's
periodic stats line reports `Prefix cache hit rate`.

Kernel correctness tests (need a GPU):

```bash
python tests/test_fi_partial_rope.py --layout tp1 --rms-spread --page 784 --shuffle --ragged
python tests/test_two_pass_1792.py
python tests/test_decode_numerics.py
python tests/test_mrope.py          # CPU only
```

## Prefix caching

`--enable-prefix-caching --mamba-cache-mode align` reuses the KV cache and linear-attention state of a
prompt prefix that an earlier request already computed: the system prompt, and in an agent loop or chat,
everything up to the newest message. vLLM marks this mode experimental for hybrid models. The plugin fixes
four problems vLLM 0.27.1 has with it on these architectures (all automatic):

* GLA-g2's MLA layers come in three latent widths, which vLLM gives different block sizes (1x, 2x, 4x the
  cache block). Hits were rounded down to the largest, so a 16k-token second turn reused only 12.5k tokens.
  The plugin sets `--prefix-match-unit 16`, which lets hits land inside a block and makes the scheduler
  keep the linear-attention state at the end of every prompt.
* With MTP, vLLM shortened every hit by a block in every layer group (6.4k of 15.6k tokens reused).
  The plugin turns that off: it protects only the drafter's proposals, which the model verifies anyway.
* With several full-attention groups and a hit shortened by the linear-attention layers, vLLM trimmed
  only one group's blocks; the next partial hit crashed the engine (`AssertionError` in `_apply_cow`).
* During prefill vLLM kept the linear-attention state of every chunk until the request finished: each
  step's free of the superseded state ran while that step was still in flight, failed, and was never
  retried. On GLA-g2 that is 81 MiB per 3136-token chunk, ~2.5 GiB for a 100k prompt, three times its
  KV cache. With 4 x 100k-token prompts on 2x RTX 5060 Ti the pool filled and requests were preempted
  and prefilled again, although the startup log promised 4.1x concurrency at 131k; with the fix the same
  load peaks at 35% of the pool. `QWEN_MLA_ALIGN_FREE=0` restores vLLM's behavior.

Measured on 2x RTX 5060 Ti, second turn of a two-turn conversation:

| | prompt | reused | time to first token |
|---|---|---|---|
| MTP k=3, before | 15.7k | 6.4k | 12.4 s |
| MTP k=3 | 15.7k | 15.6k | 2.1 s |
| MTP k=3 | 37.7k | 37.7k | 0.6 s (turn 1: 52 s) |
| no MTP | 15.7k | 15.6k | 0.4 s (was 4.3 s) |

Cached and uncached greedy answers to the same prompts match token for token up to near-ties (first-token
logprobs within 0.06 of each other; two uncached runs differ by as much). When a new prompt extends an
earlier one, the hit reaches to within 16 tokens of the earlier prompt's end. Give `--max-num-batched-tokens` a multiple of the cache block: prefill chunks are floored to it, so 3000
on a 1600-token block prefills 1600 per step. The block size depends on the KV dtype, the TP size and MTP
(1600 on two GPUs with FP8 KV and MTP, 1568 without MTP); the worker log prints it (`Setting attention block
size to N tokens`), and the plugin warns at startup when the budget is not a multiple.

Decode speed: with MTP, align mode waits for each step's accepted-token counts before preparing the next
step, so host-side work stops overlapping the GPU. vLLM builds the linear-attention metadata once per KV-cache
group, and GLA-g2's single 1792-latent layer makes every layer its own group (vLLM sizes groups by the
smallest set of same-spec layers): 48 builds per step. Before 0.6.5
that cost about 15% of decode (2x RTX 5060 Ti, MTP k=3: GPUs 63% busy vs 95% without prefix caching). The
plugin now builds it once per step and gives each layer only its own state indices
(`QWEN_MLA_GDN_META_CACHE`). Measured there, decode tok/s at 1 / 2 / 3 / 4 concurrent requests:

| MTP k=3 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| prefix caching, 0.6.3 | 57 | 96-103 | 135-137 | 153-156 |
| prefix caching, 0.6.5 | 67-68 | 115-120 | 158-159 | 168-169 |
| no prefix caching | 66-69 | 111-126 | 146-161 | 169-194 |

It is on only with MTP and align mode, the combination it was measured to help: without MTP it measured about
5% slower (within run-to-run noise, but at every concurrency), and without prefix caching the host work
already overlaps the GPU. Without MTP, prefix caching's decode cost has not been re-measured. Prefix caching
pays off whenever prompts repeat (agent loops, chat, a long system prompt).

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
