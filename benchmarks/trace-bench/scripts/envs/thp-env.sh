#!/usr/bin/env bash
# Build the THP method environment from scratch (plans/thp.md §2; the `thp` conda environment's
# twin on a box without conda): a venv under <dir>, gCastle's pinned dependencies from
# thp-requirements.txt, gCastle itself at the pinned commit, then the lab's patches under
# scripts/envs/thp/ onto the installed files. Used verbatim by a replica's
# setup_command. The runner's `--method-python` is <dir>/venv/bin/python and its `--method-src`
# the venv's site-packages directory (printed at the end).
#
#     scripts/envs/thp-env.sh <dir> [python]      (python defaults to python3.11)
set -euo pipefail
dir=${1:?usage: thp-env.sh <dir> [python]}
base_python=${2:-python3.11}
here=$(cd "$(dirname "$0")" && pwd)
GCASTLE_REPO=https://github.com/huawei-noah/trustworthyAI
GCASTLE_COMMIT=8675af0bc532c7f138158ded62e8cf303e9af655

"$base_python" -m venv "$dir/venv"
py="$dir/venv/bin/python"
"$py" -m pip install --upgrade "pip==26.2.2" "setuptools==84.2.1" "wheel==0.48.0"
"$py" -m pip install -r "$here/thp-requirements.txt"
# gcastle's setup.py imports `castle` (and so numpy) while building: no build isolation, and no
# dependency resolution of its own (every dependency is pinned above)
"$py" -m pip install --no-build-isolation --no-deps \
    "gcastle @ git+${GCASTLE_REPO}@${GCASTLE_COMMIT}#subdirectory=gcastle"

# the lab's patches onto the installed files (plans/thp.md §2, §9; RUN.md D-TH-7): the
# install's direct_url.json keeps the upstream commit, method_sha256 hashes the patched bytes
site=$("$py" -c "import castle, os; print(os.path.dirname(os.path.dirname(os.path.abspath(castle.__file__))))")
for p in "$here"/thp/0*.patch; do
    patch -p1 -d "$site" --dry-run < "$p" > /dev/null
    patch -p1 -d "$site" < "$p"
    echo "applied $(basename "$p")"
done
find "$site/castle/algorithms/ttpm" -name '*.pyc' -delete

"$py" - <<'EOF'
import glob, inspect, json, os, sys
import castle
from castle.algorithms import TTPM
from castle.common.priori_knowledge import PrioriKnowledge  # noqa: F401
assert "priori_knowledge" in inspect.signature(TTPM.__init__).parameters, "TTPM lacks priori_knowledge"
assert hasattr(TTPM, "_em_type") and hasattr(TTPM, "_hill_climb_shipped"), "0001-decomposable-score not applied"
site = os.path.dirname(os.path.dirname(os.path.abspath(castle.__file__)))
url = glob.glob(os.path.join(site, "gcastle-*.dist-info", "direct_url.json"))
assert url, "no direct_url.json: the install is not from the git URL"
commit = json.load(open(url[0], encoding="utf-8"))["vcs_info"]["commit_id"]
print(f"thp env ok: python={sys.executable} gcastle={castle.__version__} commit={commit}")
print(f"method-src={site}")
EOF
