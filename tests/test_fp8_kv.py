#!/usr/bin/env python3
"""FP8 KV cache: Triton decode (single-pass + two-pass) and the prefill gather vs exact references.

FP8 row: [latent e4m3 | packed rope e4m3 | rms as raw bf16, 2 bytes per head | pad]. For each layout:

  kernel vs ref(dequantized cache)  -- must be tight: isolates kernel bugs (wrong offsets, scale, rms bytes)
  fp8 vs bf16 reference             -- the quantization error itself, for information
  gather                            -- the prefill context gather decodes rows back to the bf16 layout

  python serve/test_fp8_kv.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

PAGE = 784
dev = torch.device("cuda:0")
F8 = torch.float8_e4m3fn


def make(batch, heads, ctx, rank, rope, seed):
    torch.manual_seed(seed)
    pps = (ctx + PAGE - 1) // PAGE
    lat = torch.randn(batch * pps, PAGE, rank, device=dev) * 2.0
    lat[..., :4] *= 8                                                    # a few outlier channels
    rp = torch.randn(batch * pps, PAGE, rope, device=dev) * 1.5
    rms = torch.exp(torch.randn(batch * pps, PAGE, heads, device=dev) * 0.6)  # ~0.3 .. 3, like the model
    q = torch.randn(batch, heads, rank + rope, dtype=torch.bfloat16, device=dev) * 0.05
    return lat, rp, rms, q, pps


def rms_bytes(rms, scale, mode):
    if mode == "bf16":
        return rms.to(torch.bfloat16).contiguous().view(torch.uint8)
    if mode == "e4m3":
        return (rms / scale).clamp(-448, 448).to(F8).view(torch.uint8)
    return ((rms.log2() + 3.0) * (255.0 / 6.0)).round().clamp(0, 255).to(torch.uint8)


def rms_decoded(rms, scale, mode):
    if mode == "bf16":
        return rms.to(torch.bfloat16).float()
    if mode == "e4m3":
        return (rms / scale).clamp(-448, 448).to(F8).float() * scale
    return torch.exp2(-3.0 + ((rms.log2() + 3.0) * (255.0 / 6.0)).round().clamp(0, 255) * (6.0 / 255.0))


def fp8_cache(lat, rp, rms, scale, width, mode):
    P, T = lat.shape[:2]
    row = torch.cat([(lat / scale).clamp(-448, 448).to(F8).view(torch.uint8),
                     (rp / scale).clamp(-448, 448).to(F8).view(torch.uint8),
                     rms_bytes(rms, scale, mode)], -1)
    c = torch.zeros(P, T, 1, width, dtype=torch.uint8, device=dev)
    c[:, :, 0, :row.shape[-1]] = row
    return c.view(F8)


def deq(lat, rp, scale):
    return ((lat / scale).clamp(-448, 448).to(F8).float() * scale,
            (rp / scale).clamp(-448, 448).to(F8).float() * scale)


def ref(q, lat, rp, rms, bt, lens, rank, rms_exact=False):
    out = []
    for b in range(q.shape[0]):
        L = int(lens[b])
        c = lat[bt[b].long()].reshape(-1, rank)[:L]
        r = rp[bt[b].long()].reshape(-1, rp.shape[-1])[:L]
        m = rms[bt[b].long()].reshape(-1, rms.shape[-1])[:L]
        m = m.float() if rms_exact else m.to(torch.bfloat16).float()
        qf = q[b].float()
        s = (qf[:, :rank] @ c.T + qf[:, rank:] @ r.T) / m.T / 16.0
        out.append(torch.softmax(s, -1) @ c)
    return torch.stack(out)


def run(batch, heads, ctx, rank, rope, mode):
    from qwen_mla.ops.mla_decode_attention import decode_attention_fwd
    lat, rp, rms, q, pps = make(batch, heads, ctx, rank, rope, seed=rank + heads)
    scale = torch.tensor(float(max(lat.abs().max(), rp.abs().max())) / 448, device=dev)
    width = rank + rope + (2 if mode == "bf16" else 1) * heads
    width += (-width) % 16
    cache = fp8_cache(lat, rp, rms, scale, width, mode)
    g = torch.Generator().manual_seed(1)
    bt = torch.randperm(batch * pps, generator=g).reshape(batch, pps).to(torch.int32).to(dev)
    lens = torch.tensor([ctx - (37 * b) % PAGE for b in range(batch)], dtype=torch.int32, device=dev)
    o = torch.zeros(batch, heads, rank, dtype=torch.bfloat16, device=dev)
    lse = torch.zeros(batch, heads, dtype=torch.bfloat16, device=dev)
    logits = torch.zeros(batch, heads, 32, rank + 1, dtype=torch.float32, device=dev)
    decode_attention_fwd(q, cache, cache[..., :rank], o, lse, bt, lens, logits, 32, 1 / 16.0, PAGE,
                         k_scale=scale, v_scale=scale, is_mla=True, rms_offset=rank + rope,
                         rms_bf16=mode == "bf16", rms_log8=mode == "log8")
    dl, dr = deq(lat, rp, scale)
    rq = rms_decoded(rms, scale, mode)
    r_deq = ref(q, dl, dr, rq, bt, lens, rank, rms_exact=True)
    r_bf = ref(q, lat, rp, rms, bt, lens, rank)
    rel = lambda a, b: ((a.float() - b).abs().max() / b.abs().max()).item()
    e_k, e_q = rel(o, r_deq), rel(r_deq, r_bf)

    # gather: decode a few rows back to the bf16 layout
    from qwen_mla.ops.mla_gather import gather_cache_torch
    n = 300
    dst = torch.zeros(n, width, dtype=torch.bfloat16, device=dev)
    cu = torch.tensor([0, n], device=dev)
    gather_cache_torch(src_cache=cache, dst=dst, block_table=bt[:1], cu_seq_lens=cu,
                       token_to_seq=torch.zeros(n, dtype=torch.int32, device=dev), num_tokens=n,
                       kv_cache_dtype="fp8", scale=scale, seq_starts=torch.tensor([123], device=dev),
                       mla_rms=(rank + rope, heads, mode))
    toks = torch.arange(123, 123 + n, device=dev)
    pages, offs = bt[0, toks // PAGE].long(), toks % PAGE
    exp = torch.cat([dl[pages, offs], dr[pages, offs], rq[pages, offs]], -1)
    e_g = ((dst[:, :rank + rope + heads].float() - exp).abs().max() / exp.abs().max()).item()
    ok = e_k < 2e-2 and e_g < 1e-2
    print(f"{mode:4s} rank {rank:4d} heads {heads:2d} rope {rope:3d}: kernel vs deq-ref {e_k:.2e} | "
          f"fp8 quant error {e_q:.2e} | gather {e_g:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = True
    for mode in ("log8", "bf16", "e4m3"):
        for heads, rope in ((24, 232), (12, 116)):        # TP=1 and TP=2 rank-local layouts
            for rank in (256, 768, 1792):
                ok &= run(4, heads, 3000, rank, rope, mode)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
