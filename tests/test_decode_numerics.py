#!/usr/bin/env python3
"""Does the retuned decode kernel still compute the right thing?

BLOCK_H / BLOCK_N / num_warps are pure tiling, so changing them MUST NOT change the result --
which is exactly the kind of assumption that has been wrong here before (the BLOCK_H=4 rule was
adopted from a sweep that only covered BLOCK_N=32). This checks the tuned defaults against an
explicit torch reference, and separately checks that different tile shapes agree with each
other, at every latent rank the allocation actually uses.

Reference follows the kernel's order exactly: qk = (q_latent.k_latent + q_rope.k_rope) / rms,
then * sm_scale, then softmax over keys. rms is per (key token, query head).
"""
from __future__ import annotations
import os
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

PAGE = 400
PACKED_ROPE = 256


def reference(q, kv, block_table, seq_lens, rank, n_heads, scale):
    B, H, _ = q.shape
    out = torch.zeros(B, H, rank, dtype=torch.float32, device=q.device)
    qk_dim = rank + PACKED_ROPE
    for b in range(B):
        L = int(seq_lens[b])
        rows = []
        for t in range(L):
            pg = int(block_table[b, t // PAGE])
            rows.append(kv[pg, t % PAGE, 0])
        K = torch.stack(rows).float()                     # [L, entry]
        k_qk = K[:, :qk_dim]                              # latent + packed rope
        v = K[:, :rank]
        rms = K[:, qk_dim:qk_dim + n_heads]               # [L, H]
        for h in range(H):
            s = (q[b, h].float() @ k_qk.T) / rms[:, h].clamp(min=1e-6)
            s = s * scale
            p = torch.softmax(s, dim=-1)
            out[b, h] = p @ v
    return out


def run_kernel(q, kv, block_table, seq_lens, rank, splits, bh=None, bn=None, warps=None):
    for k, v in (("QWEN_MLA_BLOCK_H", bh), ("QWEN_MLA_BLOCK_N", bn), ("QWEN_MLA_NUM_WARPS", warps)):
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = str(v)
    import importlib
    import qwen_mla.ops.mla_decode_attention as m
    importlib.reload(m)
    B, H, _ = q.shape
    o = torch.zeros(B, H, rank, dtype=torch.bfloat16, device=q.device)
    lse = torch.zeros(B, H, dtype=torch.bfloat16, device=q.device)
    logits = torch.zeros(B, H, splits, rank + 1, dtype=torch.float32, device=q.device)
    m.decode_attention_fwd(q, kv, kv[..., :rank], o, lse, block_table, seq_lens, logits,
                           splits, 1.0 / (256 ** 0.5), PAGE, is_mla=True,
                           rms_offset=rank + PACKED_ROPE)
    return o.float()


def case(rank, n_heads, ctx=900, B=2, splits=8):
    dev = torch.device("cuda:0")
    torch.manual_seed(rank + n_heads)
    entry = rank + PACKED_ROPE + n_heads
    ppq = (ctx + PAGE - 1) // PAGE
    kv = torch.randn(B * ppq + 4, PAGE, 1, entry, dtype=torch.bfloat16, device=dev) * 0.05
    kv[..., rank + PACKED_ROPE:] = kv[..., rank + PACKED_ROPE:].abs() + 0.3   # rms > 0
    bt = torch.arange(B * ppq, device=dev, dtype=torch.int32).reshape(B, ppq)
    sl = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    q = torch.randn(B, n_heads, rank + PACKED_ROPE, dtype=torch.bfloat16, device=dev) * 0.05

    ref = reference(q, kv, bt, sl, rank, n_heads, 1.0 / (256 ** 0.5))
    tuned = run_kernel(q, kv, bt, sl, rank, splits)
    alt = run_kernel(q, kv, bt, sl, rank, splits, bh=8, bn=16, warps=4)

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()
    r_t, r_a = rel(tuned, ref), rel(alt, ref)
    ok = r_t < 5e-2 and r_a < 5e-2
    print(f"  rank {rank:>5} heads {n_heads:>3}   tuned vs ref {r_t:.2e}   "
          f"alt-tiles vs ref {r_a:.2e}   tuned vs alt {rel(tuned, alt):.2e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


print("decode kernel numerics after retuning (bf16 reference tolerance 5e-2)")
good = True
for rank in (256, 768, 1792):
    for heads in (12, 24):
        good &= case(rank, heads)
print("\nPASS" if good else "\nFAIL")
sys.exit(0 if good else 1)
