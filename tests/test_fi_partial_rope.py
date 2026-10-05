#!/usr/bin/env python3
"""Partial-rope MLA decode: patched flashinfer (padded rope window) vs Triton vs fp32 reference.

The released students cache [latent | packed rope (58 per KV group) | one rms per local head]:
TP=1 (24 heads) -> rope 232 + rms 24 = 256; TP=2 (12 heads) -> rope 116 + rms 12 = 128. flashinfer
needs head_dim_kpe divisible into 64-wide tiles, so the plugin hands it the rope window padded to
256 / 128 -- which reads the rms scalars as extra "rope" columns -- and zero-pads q over them. The rms
tail offset given to the kernel stays the true rope width. This checks that the padding is exact.

`triton vs ref` is the accuracy bar of the shipped kernel; `fi vs ref` must be in the same league.

  python serve/test_fi_partial_rope.py --layout tp1|tp2 [--ctx 2048] [--batch 4]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

PAGE = 400            # overridden by --page (the live model uses 784)
SHUFFLE = False       # --shuffle: non-contiguous page tables, as vLLM allocates them
RAGGED = False        # --ragged: per-sequence lengths differ
RMS_SPREAD = False    # --rms-spread: rms varies strongly per (key, head)


def build(batch, heads, ctx, rank, rope, dev, seed=0):
    torch.manual_seed(seed)
    entry = rank + rope + heads + (-(rope + heads)) % 8      # rows padded to 16 bytes, as real caches are
    pps = (ctx + PAGE - 1) // PAGE
    cache = torch.randn(batch * pps, PAGE, 1, entry, dtype=torch.bfloat16, device=dev) * 0.05
    if RMS_SPREAD:   # rms varying per (key, head) over ~e^+-2, like the live model; ~0.5 +- 0.05 hides indexing bugs
        cache[..., rank + rope:] = torch.exp(torch.randn_like(cache[..., rank + rope:].float())).to(cache.dtype)
    else:
        cache[..., rank + rope:] = cache[..., rank + rope:].abs() + 0.5      # rms > 0 (padding too: harmless)
    q = torch.randn(batch, heads, rank + rope, dtype=torch.bfloat16, device=dev) * 0.05
    return cache, q, pps


def layout(batch, ctx, pps, dev):
    """Block table [batch, pps] and per-sequence lengths, contiguous/full unless --shuffle/--ragged."""
    g = torch.Generator().manual_seed(1)
    order = torch.randperm(batch * pps, generator=g) if SHUFFLE else torch.arange(batch * pps)
    bt = order.reshape(batch, pps).to(torch.int32).to(dev)
    lens = [ctx - (int(torch.randint(0, PAGE + 37, (1,), generator=g)) if RAGGED and b else 0) for b in range(batch)]
    return bt, torch.tensor(lens, dtype=torch.int32, device=dev)


def ref_out(cache, q, batch, heads, ctx, rank, rope, pps, bt, lens):
    outs = []
    for b in range(batch):
        flat = cache[bt[b].long(), :, 0, :].reshape(pps * PAGE, -1)[: int(lens[b])].float()
        c, r, rms = flat[..., :rank], flat[..., rank:rank + rope], flat[..., rank + rope:rank + rope + heads]
        qf = q[b].float()
        s = qf[..., :rank] @ c.T + qf[..., rank:] @ r.T
        s = s / rms.T * (1.0 / 256 ** 0.5)
        outs.append(torch.softmax(s, -1) @ c)
    return torch.stack(outs)


def fi_out(cache, q, batch, heads, ctx, rank, rope, kpe_w, pps, dev, bt, lens):
    import flashinfer
    flat = cache[:, :, 0, :]
    ckv, kpe = flat[..., :rank], flat[..., rank:rank + kpe_w]                 # padded window: reads rms cols
    q_pe = torch.nn.functional.pad(q[..., rank:rank + rope], (0, kpe_w - rope)).contiguous()
    i32 = dict(dtype=torch.int32, device=dev)
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.empty(256 << 20, dtype=torch.int8, device=dev), backend="fa2")
    pages = (lens + PAGE - 1) // PAGE
    kv_indptr = torch.zeros(batch + 1, **i32); kv_indptr[1:] = torch.cumsum(pages, 0)
    kv_indices = torch.cat([bt[b, : int(pages[b])] for b in range(batch)]).to(torch.int32)
    w.plan(torch.arange(batch + 1, **i32), kv_indptr, kv_indices, lens, heads, rank, kpe_w, PAGE, False, 1.0 / 256 ** 0.5,
           torch.bfloat16, torch.bfloat16)
    return w.run(q[..., :rank].contiguous(), q_pe, ckv, kpe)


def triton_out(cache, q, batch, heads, ctx, rank, rope, pps, dev, bt, lens):
    from qwen_mla.ops.mla_decode_attention import decode_attention_fwd
    o = torch.zeros(batch, heads, rank, dtype=torch.bfloat16, device=dev)
    lse = torch.zeros(batch, heads, dtype=torch.bfloat16, device=dev)
    logits = torch.zeros(batch, heads, 64, rank + 1, dtype=torch.float32, device=dev)
    decode_attention_fwd(q, cache, cache[..., :rank], o, lse, bt, lens,
                         logits, 64, 1.0 / 256 ** 0.5, PAGE, is_mla=True, rms_offset=rank + rope)
    return o


def rel(a, b):
    d = (a.float() - b.float()).abs(); s = b.float().abs().max()
    return d.max().item() / s.item(), (d.mean() / s).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--layout", choices=["tp1", "tp2", "tp1full", "tp2full"], default="tp1",
                    help="tp1: 24 heads, rope 232; tp2: 12 heads, rope 116 (one rms gap per process)")
    ap.add_argument("--page", type=int, default=400); ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--ragged", action="store_true"); ap.add_argument("--rms-spread", action="store_true")
    a = ap.parse_args()
    global PAGE, SHUFFLE, RAGGED, RMS_SPREAD
    PAGE, SHUFFLE, RAGGED, RMS_SPREAD = a.page, a.shuffle, a.ragged, a.rms_spread
    dev = torch.device("cuda:0")
    from qwen_mla import fi_mla
    print(f"{'rank':>5}{'heads':>6}{'rope':>5}{'kpe':>5}   triton vs ref (max/mean)   fi vs ref (max/mean)   fi vs triton")
    bad = 0
    for heads, rope in {"tp1": ((24, 232),), "tp2": ((12, 116),), "tp1full": ((24, 256),), "tp2full": ((12, 256),)}[a.layout]:
        fi_mla.use_mla_headers()
        fi_mla.enable_rms_tail(rope)           # the rms gap is baked into the JIT build: one per process
        kpe_w = -(-rope // 64) * 64
        for rank in (256, 768):
            cache, q, pps = build(a.batch, heads, a.ctx, rank, rope, dev, seed=rank + heads)
            bt, lens = layout(a.batch, a.ctx, pps, dev)
            ref = ref_out(cache, q, a.batch, heads, a.ctx, rank, rope, pps, bt, lens)
            tri = triton_out(cache, q, a.batch, heads, a.ctx, rank, rope, pps, dev, bt, lens)
            try:
                fi = fi_out(cache, q, a.batch, heads, a.ctx, rank, rope, kpe_w, pps, dev, bt, lens)
            except RuntimeError as e:
                print(f"{rank:>5}{heads:>6}{rope:>5}{kpe_w:>5}   FI ERROR {str(e)[:90]}"); bad += 1; continue
            t, f, ft = rel(tri, ref), rel(fi, ref), rel(fi, tri)
            ok = f[0] < max(3 * t[0], 1e-2)
            bad += not ok
            print(f"{rank:>5}{heads:>6}{rope:>5}{kpe_w:>5}   {t[0]:.2e} / {t[1]:.2e}      {f[0]:.2e} / {f[1]:.2e}     "
                  f"{ft[0]:.2e}  {'ok' if ok else 'FAIL'}")
    print("PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
