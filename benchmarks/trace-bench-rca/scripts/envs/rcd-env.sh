#!/usr/bin/env bash
# Build the pinned RCD environment `rcaeval-rcd-259ea41` end to end (plans/rca-arms.md, addendum "2026-10-25, later"; RUN.md
# D-TBR-7): the conda environment from environment-rcd.yml (the lock's versions of the packages RCD imports), RCAEval at the
# pin installed WITHOUT its dependencies, and the suite's four patched causal-learn files (lib/causallearn at the pin) copied
# over the installed causal-learn 0.1.2.3 — the step script/link.sh does in the suite. The copies' sha256 are asserted against
# constants.RCD_OVERLAY_SHA256 before the script exits 0; rcd_runner.py refuses an environment without them.
# Written 2026-11-09 after the first l apply failed in setup: the yml's pip list carried `--no-deps …`, which pip rejects inside a
# requirements file, and the overlay was only described in the yml's comments. Idempotent: an existing environment is kept.
#   bash scripts/envs/rcd-env.sh        # from benchmarks/trace-bench-rca, with conda on PATH and the harness's python as `python`
#   HARNESS_PYTHON=<harness env>/bin/python bash scripts/envs/rcd-env.sh   # locally, when `python` is not the harness env
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
ENV_NAME=rcaeval-rcd-259ea41
PIN=259ea4167160256a74ad30004aa3d96a998c846e
PREFIX="$(conda info --base)/envs/$ENV_NAME"
if [ ! -x "$PREFIX/bin/python" ]; then
  conda env create -f "$HERE/environment-rcd.yml"
fi
"$PREFIX/bin/python" -m pip install --no-deps "RCAEval @ git+https://github.com/phamquiluan/RCAEval@$PIN"
TMP=$(mktemp -d)
git init -q "$TMP"
git -C "$TMP" fetch -q --depth 1 https://github.com/phamquiluan/RCAEval "$PIN"
git -C "$TMP" checkout -q FETCH_HEAD
SITE=$("$PREFIX/bin/python" -c "import causallearn, os; print(os.path.dirname(causallearn.__file__))")
for f in utils/PCUtils/SkeletonDiscovery.py utils/Fas.py search/ConstraintBased/FCI.py graph/GraphClass.py; do
  cp "$TMP/lib/causallearn/$f" "$SITE/$f"
done
rm -rf "$TMP"
# the digests are asserted by the harness's interpreter (tracebenchrca.constants imports tracebench, which the RCD environment
# does not carry): `python` of the calling environment — the box's `train` env, or HARNESS_PYTHON locally
PYTHONPATH="$HERE/src" "${HARNESS_PYTHON:-python}" - "$SITE" <<'PY'
import hashlib, os, sys
from tracebenchrca.constants import RCD_OVERLAY_SHA256
site = sys.argv[1]
bad = []
for f, want in RCD_OVERLAY_SHA256.items():
    got = hashlib.sha256(open(os.path.join(site, f), "rb").read()).hexdigest()
    print(f"{got[:16]}  {f}")
    if not got.startswith(want):
        bad.append(f)
if bad:
    sys.exit(f"overlay digests differ from constants.RCD_OVERLAY_SHA256: {bad}")
PY
"$PREFIX/bin/python" -c "from causallearn.utils.PCUtils import SkeletonDiscovery; assert hasattr(SkeletonDiscovery, 'local_skeleton_discovery'), 'the overlay did not take'; import RCAEval; print('rcd environment ready:', '$SITE')"
