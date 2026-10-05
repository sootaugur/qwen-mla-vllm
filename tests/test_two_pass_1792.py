#!/usr/bin/env python3
"""Two-pass wide-latent decode (rank 1792) vs the single-pass Triton kernel vs an fp32 reference.

Varying rms, 784-token pages, shuffled tables, ragged lengths; 24 heads (TP=1) and 12 (TP=2), partial
and full rope. Run on a small-shared-memory device (sm_120), where the two-pass path is active.
"""
import os, sys
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
import test_fi_partial_rope as T
T.PAGE, T.SHUFFLE, T.RAGGED, T.RMS_SPREAD = 784, True, True, True
dev = torch.device("cuda:0")
for heads, rope in ((24, 232), (12, 116), (24, 256)):
    for ctx, batch in ((4096, 4), (16384, 6)):
        rank = 1792
        cache, q, pps = T.build(batch, heads, ctx, rank, rope, dev, seed=7 + heads + ctx)
        bt, lens = T.layout(batch, ctx, pps, dev)
        ref = T.ref_out(cache, q, batch, heads, ctx, rank, rope, pps, bt, lens)
        os.environ["QWEN_MLA_TWO_PASS"] = "1"; two = T.triton_out(cache, q, batch, heads, ctx, rank, rope, pps, dev, bt, lens)
        os.environ["QWEN_MLA_TWO_PASS"] = "0"; one = T.triton_out(cache, q, batch, heads, ctx, rank, rope, pps, dev, bt, lens)
        a, b, c = T.rel(two, ref), T.rel(one, ref), T.rel(two, one)
        ok = a[0] < max(3 * b[0], 1e-2)
        print(f"rank 1792 heads {heads:>2} rope {rope} ctx {ctx:>5}: two-pass vs ref {a[0]:.2e}/{a[1]:.2e} | single-pass vs ref {b[0]:.2e}/{b[1]:.2e} | two vs single {c[0]:.2e}  {'ok' if ok else 'FAIL'}")
