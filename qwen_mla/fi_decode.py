"""FlashInfer-backed MLA decode, as a drop-in for the forked Triton kernel.

Default decode path. QWEN_MLA_DECODE_BACKEND=triton restores the forked Triton kernel, which stays
in the tree as the fallback for layouts this path declines and as the numerics reference.

WHY. The Triton fork tops out near 1.9 TB/s because its fp32 accumulator [BLOCK_H, BLOCK_DV]
puts 255 registers on every thread and spills (ncu: 102% spill overhead, 12.5% occupancy).
flashinfer's fa2 MLA kernel is JIT-templated on head_dim_ckv/head_dim_kpe, so it can be
instantiated at MLA's 256/768/1792 -- and once its CKV dimension is sharded across all 8 warps
instead of 2, it stops spilling entirely.

THREE THINGS MAKE THIS FIT MLA WITHOUT COPYING OR RESHAPING THE CACHE:

  1. The kernel takes ckv and kpe as separate pointers with separate strides, read straight off
     the tensors. MLA's row is [latent | packed rope | rms | pad] contiguous, so two strided
     VIEWS of the one cache are all it needs.
  2. The patched kernel reads rms at ckv_row + HEAD_DIM_CKV + RMS_GAP + head, anchored on ckv,
     so the rope window can be narrowed independently.
  3. rope_mode='per_kv_head' leaves a query head zero in the three rope slots that are not its
     own group, so this rank reads only the contiguous window its heads span -- 128 of 256 dims
     at TP=2. Drops only structurally-zero lanes, so it is exact up to fp32 reduction order
     (tests/test_fi_rope_narrow.py), at 12.5% fewer bytes per key.

CUDA GRAPHS. plan() is host-side and does D2H copies, so it cannot run inside a captured graph.
It runs in the metadata builder's build(), which vLLM calls per step OUTSIDE the graph, and the
captured region only replays run(). For the replay to be correct, every device buffer run()
reads must keep its address across steps, so the indptr/indices/len arrays are preallocated
once per rank and filled IN PLACE; plan()'s non-cuda-graph path does `.to(device)`, which is
identity for a tensor already there, so those exact allocations are what the kernel sees.

Planning is per KV-cache group (one rank each), so a step plans three times, not sixteen.
"""
from __future__ import annotations

import os
import pathlib
import sys

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _announce(msg: str) -> None:
    """Say which decode path is live, on a channel that actually reaches the server log.

    vLLM configures logging for the "vllm" logger namespace; a plugin logger under
    "qwen_mla.*" has no handler attached, so logger.info here is swallowed and an operator
    grepping for it sees nothing whether the fast path is on or off. The rest of the plugin
    already prints startup facts to stderr for the same reason.
    """
    logger.info("%s", msg)
    print(f"[qwen-mla] {msg}", file=sys.stderr, flush=True)

_rms_gap = None             # cached rope width = offset of the rms tail from the latent
_ready: bool | None = None  # None = not tried, True = usable, False = permanently disabled

def enabled() -> bool:
    return (os.environ.get("QWEN_MLA_DECODE_BACKEND", "flashinfer").lower() == "flashinfer"
            and _ready is not False)


def _disable(why: str) -> bool:
    global _ready
    _ready = False
    hint = (" The fast path JIT-compiles a CUDA kernel on first use and needs the CUDA toolkit (nvcc) "
            "and ninja." if any(w in why for w in ("nvcc", "Ninja", "ninja", "No such file")) else "")
    _announce(f"MLA decode: flashinfer path unavailable ({why}); using the Triton kernel.{hint}")
    return False


def _init_once(rms_gap: int) -> bool:
    """Point flashinfer's JIT at our patched headers. Must precede any flashinfer import.

    rms_gap is the CACHED rope width, i.e. where the rms tail sits relative to the latent. It is
    baked into the JIT module URI, so it cannot change once a module is built; every layer on a
    rank shares it, since the rope span follows the head split rather than the layer.

    Never raises: a missing overlay or a broken flashinfer install degrades to the Triton
    kernel with one warning rather than taking the server down.
    """
    global _ready, _rms_gap
    if _ready is not None:
        if _ready and rms_gap != _rms_gap:
            return _disable(f"rms gap changed after the JIT module was built "
                            f"({_rms_gap} -> {rms_gap}); the compiled kernel reads rms at a "
                            f"fixed offset")
        return _ready
    _rms_gap = rms_gap
    try:
        from . import fi_mla

        overlay = fi_mla.use_mla_headers()
        fi_mla.enable_rms_tail(rms_gap)
    except Exception as e:                          # noqa: BLE001 - must not kill the server
        return _disable(f"{type(e).__name__}: {e}")
    _ready = True
    _announce(f"MLA decode: flashinfer MLA kernel active (overlay {overlay})")
    return True


def rope_window(head_offset: int, n_local: int, rope_group: int, rope_dim: int):
    """The contiguous rope slice this rank's heads span: (offset, width) in elements.

    Rope groups are indexed GLOBALLY (k_rope_proj is replicated), so the window comes from the
    rank's global head range -- local indices would serve rank 1 groups 0,1 where 2,3 are
    correct: wrong keys, fluent output, nothing raised.
    """
    first = head_offset // rope_group
    last = (head_offset + n_local - 1) // rope_group
    # Offset 0: the cache now stores ONLY this rank's window (mla_rope_span), so it starts
    # where the rope block starts. It was `first * rope_dim` when all four groups were cached.
    return 0, (last - first + 1) * rope_dim


def cp_async_safe(head_size: int, page_size: int, rank: int, off: int, itemsize: int) -> bool:
    """Every address the kernel touches with a 128-bit cp.async must be 16-byte aligned.

    head_size is also the per-token stride, so a row that is only 8-byte aligned leaves every
    odd token misaligned and the kernel faults with CUDA "misaligned address" -- reported
    asynchronously, usually at an unrelated synchronize. That is exactly the smallest rank at
    TP=2: 256 + 268 = 524 elements = 1048 bytes. Those layers stay on Triton.
    """
    a = 16 // itemsize
    return (head_size % a == 0 and (page_size * head_size) % a == 0
            and rank % a == 0 and off % a == 0)


class _RankPlan:
    """One flashinfer wrapper per KV-cache group, with buffers stable across steps."""

    def __init__(self, device, rank, win, n_local, page_size, max_batch, max_pages, kpe=None, spec=False):
        import flashinfer

        self.rank, self.win, self.n_local = rank, win, n_local
        self.kpe = kpe or win            # rope width given to flashinfer (win padded to a multiple of 16)
        self.page_size = page_size
        # Backs the split-k partial outputs. MLAPlan allocates exactly
        #   partial_o   2 * num_sm * cta_tile_q * 2 B * head_dim_ckv
        #   partial_lse 2 * num_sm * cta_tile_q * 4 B
        # (num_clusters * cluster_tile_q == num_sm * cta_tile_q), and its allocator raises rather than
        # overrun if the buffer is short. Sized from that with 25% margin -- ~11 MB for latent 768 on a
        # 188-SM part -- instead of flashinfer's suggested 128 MB, which cost KV capacity per group.
        self._ws = torch.empty(self._workspace_bytes(device, rank, n_local), dtype=torch.int8, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.qo_indptr = torch.arange(max_batch + 1, **i32)
        self.kv_indptr = torch.zeros(max_batch + 1, **i32)
        self.kv_indices = torch.zeros(max_pages, **i32)
        self.kv_len = torch.zeros(max_batch, **i32)
        # CUDA-graph mode: plan() copies into THESE buffers (stable addresses for the captured run)
        # and does its CPU-side scheduling from whatever host tensors it is given. Planning from CPU
        # lengths therefore needs no device->host sync (see plan_cpu).
        self.w = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self._ws, use_cuda_graph=True, qo_indptr=self.qo_indptr, kv_indptr=self.kv_indptr,
            kv_indices=self.kv_indices, kv_len_arr=self.kv_len, backend="fa2")
        # Pinned host staging, a small ring so an async H2D copy still in flight is never overwritten
        # by the next step's plan (async scheduling lets the CPU run ahead).
        self._ring = [dict(qo=torch.arange(max_batch + 1, dtype=torch.int32).pin_memory(),
                           indptr=torch.zeros(max_batch + 1, dtype=torch.int32).pin_memory(),
                           lens=torch.zeros(max_batch, dtype=torch.int32).pin_memory(),
                           rows=torch.zeros(max_pages, dtype=torch.int64).pin_memory(),
                           cols=torch.zeros(max_pages, dtype=torch.int64).pin_memory()) for _ in range(4)]
        self._ring_i = 0
        self.max_pages = max_pages
        self.planned_batch = -1
        self.planned_scale = None
        self.fits = self._probe(device)

    @staticmethod
    def _workspace_bytes(device, rank, n_local) -> int:
        props = torch.cuda.get_device_properties(device)
        smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
        # Same rule as mla_cta_tile_q() in the patched headers.
        ctq = 16 if (n_local <= 16 or (n_local <= 32 and smem < 128 * 1024)) else (32 if n_local <= 32 else 64)
        need = 2 * props.multi_processor_count * ctq * (2 * rank + 4) + 4096
        if os.environ.get("QWEN_MLA_FI_WS_MB"):                 # explicit override (debugging / A-B tests)
            return int(float(os.environ["QWEN_MLA_FI_WS_MB"]) * (1 << 20))
        return int(need * 1.25) + (1 << 20)

    def _probe(self, device) -> bool:
        """One tiny decode call, outside graph capture, to learn whether any tile config fits.

        On small-shared-memory devices (sm_120, 99 KB/SM) the widest latent (1792) fits no
        (stages, CTA_TILE_KV) config and the launcher raises "no MLA tile config fits". That is
        a host-side check before any launch, so probing is safe; a group that fails stays on the
        Triton kernel while the others keep the fast path.
        """
        try:
            bf = dict(dtype=torch.bfloat16, device=device)
            ckv = torch.zeros(1, self.page_size, self.rank, **bf)
            kpe = torch.zeros(1, self.page_size, self.kpe, **bf)
            nb = self.kv_len.shape[0]
            qo = torch.ones(nb + 1, dtype=torch.int32); qo[0] = 0
            ip = torch.ones(nb + 1, dtype=torch.int32); ip[0] = 0
            ln = torch.zeros(nb, dtype=torch.int32); ln[0] = 1
            self.w.plan(qo, ip, torch.zeros(1, dtype=torch.int32, device=device), ln, self.n_local,
                        self.rank, self.kpe, self.page_size, False, 1.0, torch.bfloat16, torch.bfloat16)
            self.w.run(torch.zeros(1, self.n_local, self.rank, **bf),
                       torch.zeros(1, self.n_local, self.kpe, **bf), ckv, kpe)
            torch.cuda.synchronize(device)
            return True
        except RuntimeError as e:
            if "no MLA tile config fits" not in str(e):
                raise
            _announce(f"MLA decode: latent {self.rank} x {self.n_local} heads fits no flashinfer tile on "
                      f"this device; those layers use the Triton kernel")
            return False

    def plan(self, block_table, seq_lens, scale, q_dtype, kv_dtype):
        B = seq_lens.shape[0]
        if B > self.kv_len.shape[0]:
            raise ValueError(f"decode batch {B} exceeds the preallocated "
                             f"{self.kv_len.shape[0]}")
        pages = (seq_lens + self.page_size - 1) // self.page_size
        self.kv_len[:B].copy_(seq_lens)
        self.kv_indptr[0] = 0
        torch.cumsum(pages, 0, dtype=torch.int32, out=self.kv_indptr[1:B + 1])
        keep = (torch.arange(block_table.shape[1], device=block_table.device)[None, :]
                < pages[:, None])
        flat = block_table[keep].to(torch.int32)
        if flat.numel() > self.max_pages:
            raise ValueError(f"{flat.numel()} pages exceeds the preallocated {self.max_pages}")
        self.kv_indices[:flat.numel()].copy_(flat)
        # Graph mode: full-length buffers, rows past B empty (device-side fills, no sync of their own).
        self.qo_indptr[:B + 1] = torch.arange(B + 1, dtype=torch.int32, device=self.qo_indptr.device)
        self.qo_indptr[B + 1:] = B
        self.kv_indptr[B + 1:] = self.kv_indptr[B]
        self.kv_len[B:] = 0
        self.w.plan(self.qo_indptr, self.kv_indptr, self.kv_indices, self.kv_len, self.n_local, self.rank,
                    self.kpe, self.page_size, False, scale, q_dtype, kv_dtype)
        self.planned_batch = B
        self.planned_scale = scale

    def plan_cpu(self, block_table, seq_lens, lens_cpu, scale, q_dtype, kv_dtype):
        """plan() from CPU sequence lengths: no device->host synchronisation.

        lens_cpu are the decode rows' lengths, already on the host (vLLM keeps them there). Page
        pointers, lengths and the (row, col) gather indices are built on the CPU in pinned memory;
        the page ids are gathered on the device with integer indices (a boolean mask would sync).
        """
        B = int(lens_cpu.shape[0])
        if B > self.kv_len.shape[0]:
            raise ValueError(f"decode batch {B} exceeds the preallocated {self.kv_len.shape[0]}")
        r = self._ring[self._ring_i]; self._ring_i = (self._ring_i + 1) % len(self._ring)
        lens = lens_cpu.to(torch.int64)
        pages = (lens + self.page_size - 1) // self.page_size
        r["lens"][:B].copy_(lens_cpu)
        r["indptr"][0] = 0
        torch.cumsum(pages, 0, out=pages); r["indptr"][1:B + 1].copy_(pages)
        total = int(pages[-1]) if B else 0
        if total > self.max_pages:
            raise ValueError(f"{total} pages exceeds the preallocated {self.max_pages}")
        starts = r["indptr"][:B].to(torch.int64)
        counts = r["indptr"][1:B + 1].to(torch.int64) - starts
        rows = torch.repeat_interleave(torch.arange(B), counts, output_size=total)
        r["rows"][:total].copy_(rows)
        r["cols"][:total].copy_(torch.arange(total) - torch.repeat_interleave(starts, counts, output_size=total))
        dev = block_table.device
        flat = block_table[r["rows"][:total].to(dev, non_blocking=True),
                           r["cols"][:total].to(dev, non_blocking=True)].to(torch.int32)
        # Graph-mode plan() copies into the fixed buffers whole, so pass full-length arrays; rows past
        # B are empty requests (no q rows, no pages).
        r["qo"][:B + 1].copy_(torch.arange(B + 1, dtype=torch.int32)); r["qo"][B + 1:] = B
        r["indptr"][B + 1:] = total
        r["lens"][B:] = 0
        self.w.plan(r["qo"], r["indptr"], flat, r["lens"], self.n_local, self.rank,
                    self.kpe, self.page_size, False, scale, q_dtype, kv_dtype)
        self.planned_batch = B
        self.planned_scale = scale


def _decode_lens_cpu(common, nd):
    """The decode rows' sequence lengths on the host, without forcing a device->host copy.

    vLLM's seq_lens_cpu property syncs when the cached copy is absent (async scheduling), so read the
    cached tensor, else the host upper bound -- exact for plain decode, optimistic only under
    speculative decoding. None means "not available without a sync"; the caller then uses plan().
    """
    if common is None:
        return None
    c = getattr(common, "_seq_lens_cpu", None)
    if c is None:
        c = getattr(common, "seq_lens_cpu_upper_bound", None)
    if c is None:
        return None
    c = torch.as_tensor(c)
    return c[:nd].to(torch.int32) if c.device.type == "cpu" else None


def plan_for_step(builder, md, common=None) -> None:
    """Called from the metadata builder, outside any CUDA graph capture.

    Anything that goes wrong here disables the fast path for the rest of the process rather
    than propagating: forward_mqa then finds no plan on the metadata and uses Triton.
    """
    if not enabled() or md.decode is None:
        return
    if not getattr(builder, "_mla_fi_ok", False):
        return
    if not _init_once(builder._mla_fi_cfg["win"]):
        return
    try:
        plan = builder._mla_fi_plan
        if plan is None:
            plan = _RankPlan(**builder._mla_fi_cfg)
            builder._mla_fi_plan = plan
        if not plan.fits:
            return
        lens_cpu = (_decode_lens_cpu(common, md.decode.seq_lens.shape[0])
                    if not builder._mla_fi_cfg.get("spec") else None)
        if lens_cpu is not None and os.environ.get("QWEN_MLA_FI_SYNC_PLAN") != "1":
            _dbg("plan path", path="cpu (no sync)", rank=builder._mla_fi_cfg["rank"])
            plan.plan_cpu(md.decode.block_table, md.decode.seq_lens, lens_cpu, builder._mla_fi_scale,
                          builder._mla_fi_qdtype, builder._mla_fi_kvdtype)
        else:
            _dbg("plan path", path="device (syncs)", rank=builder._mla_fi_cfg["rank"], have_cpu=lens_cpu is not None)
            plan.plan(md.decode.block_table, md.decode.seq_lens, builder._mla_fi_scale,
                      builder._mla_fi_qdtype, builder._mla_fi_kvdtype)
    except Exception as e:                          # noqa: BLE001 - must not kill the server
        _disable(f"plan failed: {type(e).__name__}: {e}")
        return
    object.__setattr__(md, "_mla_fi_plan", plan)


_DBG_SEEN = set()


def _dbg(reason, **kw):
    if os.environ.get("QWEN_MLA_FI_DEBUG") and (reason, tuple(sorted(kw.items()))) not in _DBG_SEEN:
        _DBG_SEEN.add((reason, tuple(sorted(kw.items()))))
        print(f"[qwen-mla-fi-debug] {reason} {kw}", flush=True)


def forward_mqa(impl, q, kv_c_and_k_pe_cache, attn_metadata, layer):
    """Returns (out, lse|None), or None if this step has no usable plan (use Triton)."""
    plan = getattr(attn_metadata, "_mla_fi_plan", None)
    if plan is None:
        _dbg("no plan on metadata", rank=getattr(impl, "kv_lora_rank", None))
        return None
    if isinstance(q, tuple):
        q = torch.cat(q, dim=-1)
    if q.shape[0] != plan.planned_batch or q.shape[1] != plan.n_local:
        _dbg("batch/heads mismatch", q=tuple(q.shape[:2]), planned=plan.planned_batch, n_local=plan.n_local)
        return None
    # sm_scale is baked into the plan, so a layer disagreeing with it would be silently wrong.
    if impl.scale != plan.planned_scale:
        _dbg("scale mismatch", impl=impl.scale, plan=plan.planned_scale)
        return None

    cache = kv_c_and_k_pe_cache
    if cache.dim() == 4:                       # [pages, page_size, 1, head_size]
        cache = cache[:, :, 0, :]
    rank = plan.rank
    off, win = rope_window(impl.mla_head_offset, q.shape[1], impl.mla_rope_group,
                           impl.mla_rope_dim)
    if win != plan.win or rank != impl.kv_lora_rank:
        _dbg("window/rank mismatch", win=win, plan_win=plan.win, rank=rank, impl_rank=impl.kv_lora_rank)
        return None
    _dbg("FI used", rank=rank, batch=q.shape[0])
    if os.environ.get("QWEN_MLA_FI_DEBUG") and not torch.cuda.is_current_stream_capturing():
        tail = q[..., rank + off + win:]
        _dbg("q layout", rank=rank, q_width=q.shape[-1], off=off, win=win, kpe=plan.kpe,
             tail_width=tail.shape[-1], tail_absmax=float(tail.abs().max()) if tail.numel() else 0.0,
             rope_absmax=float(q[..., rank + off:rank + off + win].abs().max()),
             cache_width=cache.shape[-1], impl_rms_offset=impl.rms_offset, fi_rms_gap=_rms_gap,
             packed_rope=getattr(impl, "mla_packed_rope", None), rope_dim=impl.mla_rope_dim,
             rope_first=impl.mla_rope_first, rope_group=impl.mla_rope_group, heads=impl.mla_local_heads)

    ckv = cache[..., :rank]
    kpe = cache[..., rank + off:rank + off + plan.kpe]
    q_nope = q[..., :rank].contiguous()
    q_pe = q[..., rank + off:rank + off + win]
    if plan.kpe != win:                     # zero lanes over the rms columns read by the padded kpe
        q_pe = torch.nn.functional.pad(q_pe, (0, plan.kpe - win))
    q_pe = q_pe.contiguous()

    if getattr(impl, "need_to_return_lse_for_decode", False):
        o, lse = plan.w.run(q_nope, q_pe, ckv, kpe, return_lse=True)
        return o, lse
    return plan.w.run(q_nope, q_pe, ckv, kpe), None
