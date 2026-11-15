#!/usr/bin/env python3
"""Fold a base-model LoRA's k_proj/v_proj into an MLA/GLA checkpoint's latent projections.

A PEFT adapter trained on the base Qwen3.5/3.6/3.8 model targets self_attn.k_proj / v_proj, which the
MLA/GLA builds replace (vLLM then refuses the adapter: "expected target modules in [...] but received
[...k_proj...]"). Everything else in such an adapter -- q_proj, o_proj, the linear-attention and MLP
projections, lm_head -- applies to these builds unchanged: the retrofit keeps those weights (the EXL3 and
INT4 builds reuse the base model's quantized tensors as they are).

The checkpoint's key for query head h is, in the retrofit's per-head dim order `perm`,
    [ k_rope_proj x  (rope dims of KV head j = h // group)  ;  k_up[h] @ kv_a x  (nope dims) ]
and its value v_up[h] @ kv_a x. A LoRA delta on the base model's k_proj/v_proj therefore maps to
  * a LoRA on k_rope_proj for the rope rows -- exact, the rope key is not compressed;
  * a LoRA on kv_a_proj for the nope rows and v: dc = U^+ d(x), with U = [k_up; v_up] restricted to each
    latent group's heads and columns. This is the per-token least-squares optimum given the fixed
    up-projections, so the delta rides in the cached latent and the KV compression is unchanged.
    Its rank is the k and v ranks added (32 for a rank-16 adapter); vLLM needs --max-lora-rank >= it.

Per layer the script prints the share of the k/v delta that lies inside the latent's span
(isotropic inputs). On autotrust/AutoTrust-Decider-Pro-Vision-27B with GLA-g2-EXL3-4.0bpw that was
15-29% in the rank-256 layers, 37-54% at 768 and 99% at 1792, and the folded adapter matched the
base-model EXL3 build with the original adapter on BANKING77 / CLINC150 (64 options: 76.0 / 92.7%
vs 76.2 / 92.7%, identical log-loss); dropping k/v instead gave 75.7 / 92.5%.

usage: fold_lora_kv.py --adapter <PEFT dir> --model <MLA/GLA checkpoint dir> --out <dir> [--rank N]
Files other than the adapter weights and config (e.g. a decision head) are copied to --out.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def rope_perm(model_rope_dim, head_dim, keep):
    """Same as qwen_mla.partial_rope.rope_perm: kept rope pairs first, then dropped, then nope."""
    half = model_rope_dim // 2
    keep = [int(k) for k in keep]
    drop = [s for s in range(half) if s not in keep]
    return keep + [s + half for s in keep] + drop + [s + half for s in drop] + list(range(model_rope_dim, head_dim))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0,
                    help="truncate the kv_a_proj fold to this rank (default: keep it whole, lossless)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    if "mla_ranks" not in tc:
        raise SystemExit(f"{args.model} is not an MLA/GLA checkpoint (no mla_ranks in its config)")
    if tc.get("mla_canon_norm"):
        raise SystemExit("mla_canon_norm checkpoints normalise the latent; a kv_a_proj LoRA cannot be folded exactly")
    if tc.get("mla_rope_mode", "per_kv_head") != "per_kv_head":
        raise SystemExit(f"mla_rope_mode {tc['mla_rope_mode']!r} is not supported (per_kv_head only)")
    nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    groups = int(tc.get("mla_latent_groups", 1) or 1)
    model_rd = int(tc.get("mla_model_rope_dim") or hd * tc["rope_parameters"].get("partial_rotary_factor", 0.25))
    keep = tc.get("mla_rope_keep")
    perm = rope_perm(model_rd, hd, keep) if keep is not None else list(range(hd))
    rd = 2 * len(keep) if keep is not None else model_rd
    nope, hpg, grp = hd - rd, nh // groups, nh // nkv

    acfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
    if acfg.get("use_dora") or acfg.get("use_rslora"):
        raise SystemExit("DoRA / rsLoRA adapters are not supported")
    r = int(acfg["r"])
    scale = float(acfg["lora_alpha"]) / r       # vLLM applies alpha / r to every module
    ad = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    wm = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]

    def ckpt(name):
        with safe_open(os.path.join(args.model, wm[name]), "pt") as f:
            return f.get_tensor(name).double()

    def key(layer, mod, ab):
        hits = [k for k in ad if k.endswith(f"layers.{layer}.self_attn.{mod}.lora_{ab}.weight")]
        return hits[0] if hits else None

    new, dropped = {}, []
    for layer in sorted(int(k) for k in tc["mla_ranks"]):
        ka, va = key(layer, "k_proj", "A"), key(layer, "v_proj", "A")
        if ka is None and va is None:
            continue
        pre = f"model.language_model.layers.{layer}.self_attn."
        if pre + "kv_a_proj.weight" not in wm:
            pre = f"model.layers.{layer}.self_attn."
        k_up, v_up = ckpt(pre + "k_up_proj.weight"), ckpt(pre + "v_up_proj.weight")
        rtot = ckpt(pre + "kv_a_proj.weight").shape[0]
        rg = rtot // groups
        zero = lambda rows, cols: torch.zeros(rows, cols, dtype=torch.float64)
        hidden = ad[ka or va].shape[1]
        Ak = ad[ka].double() if ka else zero(0, hidden)
        Bk = ad[ka.replace("lora_A", "lora_B")].double() * scale if ka else zero(nkv * hd, 0)
        Av = ad[va].double() if va else zero(0, hidden)
        Bv = ad[va.replace("lora_A", "lora_B")].double() * scale if va else zero(nkv * hd, 0)
        dropped += [k for k in (ka, va) if k] + [k.replace("lora_A", "lora_B") for k in (ka, va) if k]
        base = (ka or va).split(".self_attn.")[0] + ".self_attn."

        if ka:   # rope rows: exact
            rows = torch.tensor([j * hd + perm[i] for j in range(nkv) for i in range(rd)])
            new[base + "k_rope_proj.lora_A.weight"] = Ak
            new[base + "k_rope_proj.lora_B.weight"] = Bk[rows] / scale

        # nope rows + v through the latent, one least-squares solve per latent group
        A2 = torch.cat([Ak, Av], 0)
        rk, rv = Ak.shape[0], Av.shape[0]
        Bf, Rs, share = [], [], []
        for g in range(groups):
            heads = range(g * hpg, (g + 1) * hpg)
            cols = slice(g * rg, (g + 1) * rg)
            krows = torch.tensor([h * nope + i for h in heads for i in range(nope)])
            vrows = torch.tensor([h * hd + d for h in heads for d in range(hd)])
            kt = torch.tensor([(h // grp) * hd + perm[rd + i] for h in heads for i in range(nope)])
            vt = torch.tensor([(h // grp) * hd + d for h in heads for d in range(hd)])
            U = torch.cat([k_up[krows][:, cols], v_up[vrows][:, cols]], 0)
            Bt = zero(U.shape[0], rk + rv)
            Bt[: len(kt), :rk] = Bk[kt]
            Bt[len(kt):, rk:] = Bv[vt]
            sol = torch.linalg.lstsq(U, Bt).solution
            resid = U @ sol - Bt                                       # least squares: residual is orthogonal to U
            assert (U.T @ resid).norm() <= 1e-6 * (U.norm() * Bt.norm() + 1e-30), "least-squares fold failed"
            Q, R = torch.linalg.qr(U)
            D = Bt @ A2
            share.append(((Q.T @ D).norm() ** 2 / D.norm().clamp_min(1e-30) ** 2).item())
            Bf.append(sol)
            Rs.append(R)
        Bf = torch.cat(Bf, 0)
        Rb = torch.block_diag(*Rs)                                     # measure the fold where it lands: keys/values
        Us, S, Vh = torch.linalg.svd(Rb @ Bf @ A2, full_matrices=False)
        k = int((S > S[0] * 1e-6).sum()) if S.numel() and S[0] > 0 else 0
        if args.rank:
            k = min(k, args.rank)
        lost = (S[k:].pow(2).sum() / S.pow(2).sum().clamp_min(1e-30)).item()
        Bn = torch.linalg.solve_triangular(Rb, Us[:, :k] * S[:k].sqrt(), upper=True)
        new[base + "kv_a_proj.lora_A.weight"] = S[:k].sqrt()[:, None] * Vh[:k]
        new[base + "kv_a_proj.lora_B.weight"] = Bn / scale
        print(f"layer {layer:2d}  latent {rtot:4d}  k/v delta inside the latent span: "
              + " ".join(f"{s:.0%}" for s in share) + f"  kv_a_proj rank {k}" + (f" (lost {lost:.1%})" if lost > 1e-9 else ""))

    if not dropped:
        raise SystemExit("the adapter has no k_proj/v_proj weights; it needs no folding")
    os.makedirs(args.out, exist_ok=True)
    out = {k: v for k, v in ad.items() if k not in set(dropped)}
    out.update({k: v.to(ad[dropped[0]].dtype).contiguous() for k, v in new.items()})
    save_file(out, os.path.join(args.out, "adapter_model.safetensors"))
    acfg["target_modules"] = [m for m in acfg["target_modules"] if m not in ("k_proj", "v_proj")] + ["kv_a_proj", "k_rope_proj"]
    json.dump(acfg, open(os.path.join(args.out, "adapter_config.json"), "w"), indent=2)
    for f in os.listdir(args.adapter):
        if f not in ("adapter_model.safetensors", "adapter_config.json") and os.path.isfile(os.path.join(args.adapter, f)):
            shutil.copy(os.path.join(args.adapter, f), args.out)
    need = max(v.shape[0] for k, v in out.items() if k.endswith("lora_A.weight"))
    print(f"wrote {args.out}: {len(out)} tensors; serve with --enable-lora --max-lora-rank {need} or more")


if __name__ == "__main__":
    main()
