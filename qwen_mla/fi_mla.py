"""Run flashinfer's fa2 MLA decode kernel from OUR patched headers.

flashinfer resolves kernel sources through module-level paths in flashinfer.jit.env, not
through env vars, so the hook is to rebind them before the first JIT build:

  * FLASHINFER_INCLUDE_DIR -> an overlay built on first use under ~/.cache/qwen-mla: a symlink
    mirror of flashinfer's include tree with attention/mla.cuh and attention/scheduler.cuh replaced
    by the copies shipped in qwen_mla/kernels/ (see ensure_overlay, kernels/patch_mla.py).
  * FLASHINFER_WORKSPACE_BASE -> a separate cache root. The JIT cache key is the module URI,
    which encodes dtypes and head dims but NOT the header contents -- pointing at the stock
    cache would silently reuse a module built from stock sources and quietly measure nothing.

Why bother: FlashMLA is Hopper-only and CutlassMLA hardcodes 512/64, but flashinfer's fa2 MLA
kernel is JIT-templated on head_dim_ckv/head_dim_kpe, so it can be instantiated at MLA's
256/768/1792. The patched headers add the shape-aware tile dispatch and the q/D warp split that
those ranks need. See patch_mla.py for the reasoning and the measurements.
"""
from __future__ import annotations

import os
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_HEADERS = _HERE / "kernels"           # patched mla.cuh / scheduler.cuh shipped with the package
_CACHE = pathlib.Path(os.environ.get("QWEN_MLA_CACHE", str(pathlib.Path.home() / ".cache" / "qwen-mla")))
# Compiled-module cache, separate from flashinfer's own: its cache key is the module URI, which
# encodes dtypes and head dims but NOT header contents or compile flags, so sharing the stock cache
# would risk loading a module built from stock sources. Not derived from this file's location, so
# reinstalling the plugin does not force a multi-minute recompile.
_WORKSPACE = pathlib.Path(os.environ.get("QWEN_MLA_JIT_CACHE", str(_CACHE / "jit")))

_patched = None


def ensure_overlay() -> pathlib.Path:
    """Build (once) the include-path overlay flashinfer's JIT compiles our kernel from.

    A mirror of the INSTALLED flashinfer include tree -- symlinks, so it tracks the wheel -- with
    attention/mla.cuh and attention/scheduler.cuh replaced by the patched copies shipped in
    qwen_mla/kernels/. Those were generated from one flashinfer version (kernels/FLASHINFER_VERSION)
    and pairing them with another's surrounding headers would compile a mixture, so a mismatch raises
    (the caller then falls back to the Triton kernel). Built under a temporary name and renamed into
    place, so TP workers starting together cannot see a half-built tree.
    """
    import hashlib, shutil
    import flashinfer
    import flashinfer.jit.env as jit_env

    want = (_HEADERS / "FLASHINFER_VERSION").read_text().strip()
    if flashinfer.__version__ != want:
        raise RuntimeError(f"the shipped kernel headers target flashinfer {want}, "
                           f"installed is {flashinfer.__version__}")
    ours = {"mla.cuh": _HEADERS / "qwen_mla.cuh", "scheduler.cuh": _HEADERS / "qwen_scheduler.cuh"}
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in ours.values())).hexdigest()[:12]
    out = _CACHE / f"fi_overlay-{want}-{digest}"
    if (out / ".complete").exists():
        return out
    src = pathlib.Path(jit_env.FLASHINFER_INCLUDE_DIR)
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "flashinfer" / "attention").mkdir(parents=True)
    for p in src.iterdir():                                    # top of the include tree
        if p.name != "flashinfer":
            (tmp / p.name).symlink_to(p)
    for p in (src / "flashinfer").iterdir():                   # flashinfer/* except attention/
        if p.name != "attention":
            (tmp / "flashinfer" / p.name).symlink_to(p)
    for p in (src / "flashinfer" / "attention").iterdir():     # attention/* except ours
        if p.name not in ours:
            (tmp / "flashinfer" / "attention" / p.name).symlink_to(p)
    # COPIES, not symlinks: the headers use relative includes ("../profiler.cuh"), which resolve
    # against the file's real location.
    for name, f in ours.items():
        shutil.copyfile(f, tmp / "flashinfer" / "attention" / name)
    (tmp / ".flashinfer-version").write_text(flashinfer.__version__)
    (tmp / ".complete").write_text("")
    try:
        os.replace(tmp, out)
    except OSError:                                            # another worker finished first
        shutil.rmtree(tmp, ignore_errors=True)
        if not (out / ".complete").exists():
            raise
    return out


def use_mla_headers() -> pathlib.Path:
    """Point flashinfer's JIT at our overlay. Call before any flashinfer JIT build."""
    global _patched
    if _patched is not None:
        return _patched
    overlay = ensure_overlay()
    # flashinfer's JIT runs the `ninja` EXECUTABLE, which pip installs into the environment's bin/.
    # When the server is started as .venv/bin/vllm without activating the venv, that directory is
    # not on PATH and every JIT build fails; put the running interpreter's bin/ first.
    import sys
    bindir = str(pathlib.Path(sys.executable).parent)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    # must be set before flashinfer.jit.env is imported, so it lands in the module constants
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(_WORKSPACE))
    _WORKSPACE.mkdir(parents=True, exist_ok=True)

    import flashinfer.jit.env as jit_env

    jit_env.FLASHINFER_INCLUDE_DIR = overlay
    _patched = overlay
    return overlay


def workspace_dir() -> pathlib.Path:
    return _WORKSPACE


_RMS_GAP = 256
_MIN_CTAS = int(os.environ.get("QWEN_MLA_MIN_CTAS", "1"))


def enable_rms_tail(rms_gap: int = 256) -> None:
    """Build the MLA module with -DQWEN_MLA_RMS_TAIL=<gap> (see patch_mla.py::patch_rms).

    `rms_gap` is the element distance from the end of the latent to the first rms slot, i.e.
    the FULL packed-rope width (256), independent of how much of the rope this rank actually
    reads. The kernel anchors the rms address on ckv for exactly that reason.

    flashinfer's JIT cache key is the module URI, which encodes dtypes and head dims but not
    compile flags, so the URI is suffixed too -- otherwise a module already built without the
    rms divide would be reused and the divide would silently not happen.

    gen_batch_mla_module() ends in gen_jit_spec(uri, sources, extra_cuda_cflags=...); the
    smallest safe hook is to swap gen_jit_spec for the duration of that one call rather than
    reimplement the source generation, which would drift from flashinfer's.
    """
    global _RMS_GAP
    _RMS_GAP = rms_gap

    import flashinfer.jit.attention.modules as modules
    import flashinfer.mla._core as mla_core

    if getattr(mla_core.gen_batch_mla_module, "_mla_rms", False):
        return
    inner = mla_core.gen_batch_mla_module

    def wrapped(backend, *args):
        orig_gen_jit_spec = modules.gen_jit_spec

        def gen_jit_spec(uri, sources, **kw):
            cflags = list(kw.pop("extra_cuda_cflags", None) or [])
            cflags.append(f"-DQWEN_MLA_RMS_TAIL={_RMS_GAP}")
            cflags.append(f"-DQWEN_MLA_MIN_CTAS={_MIN_CTAS}")

            return orig_gen_jit_spec(uri + f"_mla_rms{_RMS_GAP}_c{_MIN_CTAS}", sources,
                                     extra_cuda_cflags=cflags, **kw)

        modules.gen_jit_spec = gen_jit_spec
        try:
            return inner(backend, *args)
        finally:
            modules.gen_jit_spec = orig_gen_jit_spec

    wrapped._mla_rms = True
    mla_core.gen_batch_mla_module = wrapped


def mla_cache_views(cache, kv_lora_rank: int, packed_rope: int = 256):
    """Split an MLA cache into the (ckv, kpe) pair the MLA kernel expects -- as VIEWS.

    MLA stores one row per token: [latent (kv_lora_rank) | packed rope (256) | rms | pad].
    The kernel takes ckv and kpe as separate pointers with separate strides (batch_mla_run.cu
    reads them straight off the tensors), so two strided views of the same rows work with no
    copy and no cache reformatting -- and, because the rows stay contiguous, rms(n, h) lands at
    kpe_row(n) + 256 + h, exactly where the patched kernel looks for it.

    `cache` is [num_pages, page_size, 1, entry] as vLLM allocates it.
    """
    if cache.dim() == 4:
        cache = cache[:, :, 0, :]
    ckv = cache[..., :kv_lora_rank]
    kpe = cache[..., kv_lora_rank:kv_lora_rank + packed_rope]
    return ckv, kpe
