#!/usr/bin/env bash
# Build the CausalNET method environment on a GPU box for the seq2cause-bench external arm
# (plans/causalnet.md §4; algos/causalnet-zhu-2024/RUN.md): a venv with the cu12 torch pin, the
# pinned clone of github.com/CGCL-codes/CausalNET at aaccce6 and the lab's two patches applied to
# its working tree (uncommitted, as CASCADE's patch is handled: RUN.md D-CA-1 there, D-CN-1/2 here).
#
#   bash scripts/envs/causalnet-env.sh /opt/causalnet
#
# leaves /opt/causalnet/venv/bin/python (--method-python) and /opt/causalnet/CausalNET (--method-src).
# Run from the harness directory (benchmarks/trace-bench): the patches are read from
# scripts/envs/causalnet/. Idempotent: an existing clone is reset to the pin before patching.
set -euo pipefail
PREFIX="${1:?usage: causalnet-env.sh <prefix>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN=aaccce6dbd25eb4baa3711c250405aa1d8d6e7b1
REPO=https://github.com/CGCL-codes/CausalNET.git

mkdir -p "$PREFIX"
if [ ! -x "$PREFIX/venv/bin/python" ]; then
  python3 -m venv "$PREFIX/venv"
fi
"$PREFIX/venv/bin/python" -m pip install --upgrade pip
"$PREFIX/venv/bin/python" -m pip install -r "$HERE/causalnet-requirements.txt"

if [ ! -d "$PREFIX/CausalNET/.git" ]; then
  git clone "$REPO" "$PREFIX/CausalNET"
fi
cd "$PREFIX/CausalNET"
git fetch -q origin
git checkout -q --detach "$PIN"
git checkout -q -- .            # drop any earlier patch application
git clean -q -fd -- model       # and any stray files under model/
git apply --check "$HERE/causalnet/0001-collapse-decay.patch"
git apply "$HERE/causalnet/0001-collapse-decay.patch"
git apply --check "$HERE/causalnet/0002-drivable-entry.patch"
git apply "$HERE/causalnet/0002-drivable-entry.patch"
echo "CausalNET at $(git rev-parse HEAD) with patches: 0001-collapse-decay 0002-drivable-entry"
git status --short

"$PREFIX/venv/bin/python" - <<'PY'
import os, sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
"$PREFIX/venv/bin/python" -m pip freeze > "$PREFIX/causalnet-pip-freeze.txt"
echo "method_env written to $PREFIX/causalnet-pip-freeze.txt"
