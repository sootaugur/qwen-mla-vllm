#!/usr/bin/env python3
"""M-RoPE path of Qwen3_5MLAAttention vs vLLM's own rotary (CPU, no model).

1. t == h == w (text tokens in a multimodal request): must equal the 1-D rotary the text model uses.
2. distinct t/h/w (image tokens): must equal vLLM's MRotaryEmbedding at FULL 64-wide rope, restricted
   to the kept frequencies -- i.e. exactly what the teacher computes on the dimensions we keep.
"""
import sys, types, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from vllm.model_executor.layers.rotary_embedding import get_rope
from qwen_mla.partial_rope import patch_partial_rope
from qwen_mla.qwen3_5_mla import Qwen3_5MLAAttention

HD, THETA, SEC = 256, 1e7, [11, 11, 10]
KEEP = [i for i in range(32) if i not in (29, 30, 31)]          # the released students' 58/64
RD = 2 * len(KEEP)
torch.manual_seed(0)
from vllm.config import VllmConfig, set_current_vllm_config
_ctx = set_current_vllm_config(VllmConfig()); _ctx.__enter__()

def sec_of(f):
    return 1 if (f % 3 == 1 and f < 3 * SEC[1]) else (2 if (f % 3 == 2 and f < 3 * SEC[2]) else 0)

rot = get_rope(HD, max_position=4096, is_neox_style=True,
               rope_parameters={"rope_theta": THETA, "rope_type": "default", "partial_rotary_factor": RD / HD},
               dtype=torch.float32)
patch_partial_rope(rot, KEEP, 64, THETA)
self = types.SimpleNamespace(mrope_section=SEC, rotary_emb=rot,
                             _mrope_col_sec=torch.tensor([sec_of(f) for f in KEEP] * 2))
n, nh, nkv = 37, 24, 4
q, k = torch.randn(n, nh, RD), torch.randn(n, nkv, RD)

# 1. equal positions
p = torch.randint(0, 4000, (n,))
qm, km = Qwen3_5MLAAttention._mrope_rotate(self, p.expand(3, n), q, k)
qp = torch.zeros(n, nh, HD); qp[..., :RD] = q; kp = torch.zeros(n, nkv, HD); kp[..., :RD] = k
q1, k1 = rot.forward_native(p, qp.reshape(n, -1), kp.reshape(n, -1))
e1 = max((qm - q1.view(n, nh, HD)[..., :RD]).abs().max().item(), (km - k1.view(n, nkv, HD)[..., :RD]).abs().max().item())

# 2. distinct positions vs the teacher's full-width interleaved M-RoPE
full = get_rope(HD, max_position=4096, is_neox_style=True, dtype=torch.float32,
                rope_parameters={"rope_theta": THETA, "rope_type": "default", "partial_rotary_factor": 0.25,
                                 "mrope_section": SEC, "mrope_interleaved": True})
p3 = torch.randint(0, 4000, (3, n))
qm, km = Qwen3_5MLAAttention._mrope_rotate(self, p3, q, k)
cols = KEEP + [32 + f for f in KEEP]                              # kept pairs inside the 64-wide block
qf, kf = torch.zeros(n, nh, HD), torch.zeros(n, nkv, HD)
qf[..., cols] = q; kf[..., cols] = k
q2, k2 = full.forward_native(p3, qf.reshape(n, -1), kf.reshape(n, -1))
e2 = max((qm - q2.view(n, nh, HD)[..., cols]).abs().max().item(), (km - k2.view(n, nkv, HD)[..., cols]).abs().max().item())
print(f"equal t/h/w vs 1-D rotary: max abs err {e1:.2e}\ndistinct t/h/w vs teacher M-RoPE (kept dims): max abs err {e2:.2e}")
ok = e1 < 1e-4 and e2 < 1e-4
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
