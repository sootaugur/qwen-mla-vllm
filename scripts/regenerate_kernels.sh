#!/usr/bin/env bash
# MAINTAINER ONLY: regenerate qwen_mla/kernels/*.cuh from the installed flashinfer (after a vLLM /
# flashinfer upgrade), then clear the runtime caches. Users never need this.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
python "$ROOT/qwen_mla/kernels/patch_mla.py"
rm -rf "${QWEN_MLA_CACHE:-$HOME/.cache/qwen-mla}"
echo "headers regenerated for flashinfer $(cat "$ROOT/qwen_mla/kernels/FLASHINFER_VERSION")"
