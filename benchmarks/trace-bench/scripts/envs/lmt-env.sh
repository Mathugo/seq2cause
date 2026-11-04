#!/usr/bin/env bash
# Build the LMT external arm's method environment on a box (plans/lmt.md §2; D-SB-29): a python
# 3.11 venv with scripts/envs/lmt-requirements.txt and a clone of github.com/xx987/LMT at the
# pinned commit. The clone carries no licence file: it is cloned and imported, never copied into
# this repository. The harness drives the fit as a subprocess under the venv's interpreter:
#
#   python -m seq2causebench.externalarm --method lmt ... \
#       --method-python "$LMT_ENV/bin/python" --method-src "$LMT_SRC"
#
# Usage: scripts/envs/lmt-env.sh [<env dir>] [<clone dir>]   (defaults: ~/envs/lmt, ~/src/LMT)
# Needs python3.11 on PATH (or PYTHON=… ), git, network for pip and the clone.
set -euo pipefail
LMT_ENV="${1:-$HOME/envs/lmt}"
LMT_SRC="${2:-$HOME/src/LMT}"
LMT_REPO="https://github.com/xx987/LMT"
LMT_COMMIT="6f4143dc0668187f69505bc206e87ae2084d47d3"   # HEAD of main on 2026-11-03
PYTHON="${PYTHON:-python3.11}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if [ ! -x "$LMT_ENV/bin/python" ]; then
  "$PYTHON" -m venv "$LMT_ENV"
fi
"$LMT_ENV/bin/python" -m pip install --upgrade pip >/dev/null
"$LMT_ENV/bin/python" -m pip install -r "$HERE/lmt-requirements.txt"

if [ ! -d "$LMT_SRC/.git" ]; then
  git clone --quiet "$LMT_REPO" "$LMT_SRC"
fi
git -C "$LMT_SRC" fetch --quiet origin
git -C "$LMT_SRC" checkout --quiet "$LMT_COMMIT"
test "$(git -C "$LMT_SRC" rev-parse HEAD)" = "$LMT_COMMIT"
test -f "$LMT_SRC/chemical_1000_event_repro/text_causal/graph_prior.py"

# the import the runner makes, under the venv
"$LMT_ENV/bin/python" -I -c "
import sys; sys.path.insert(0, '$LMT_SRC/chemical_1000_event_repro')
import torch, numpy, pandas
from text_causal import graph_prior
print('lmt env ok: torch', torch.__version__, 'numpy', numpy.__version__)
"
echo "LMT_ENV=$LMT_ENV"
echo "LMT_SRC=$LMT_SRC  ($LMT_COMMIT)"
