#!/usr/bin/env bash
# Build the harness's own method environment `trace-bench-rca` end to end on a box (RUN.md "Environment", plans/rca-arms.md
# addendum 2026-10-21): the conda environment from environment.yml (Python 3.12 — at the RCAEval pin, `RCAEval/e2e/__init__.py`
# imports its method modules leniently only on 3.10 / 3.12 / 3.14; on 3.11, the provisioning template's `train` environment,
# it imports `rcd`, `mmrcd` and `torai` unconditionally and `RCAEval.e2e.baro` cannot be imported), the package as an editable
# install with the `baro` and `rcaeval` extras (fse-baro and RCAEval at their pinned commits), then the checks: Python 3.12, the
# two installed commits against constants.BARO_COMMIT / RCAEVAL_COMMIT, the benchmark package at the commit requirements.txt
# pins, and the imports `rca/baro` makes (`RCAEval.e2e.baro`, `baro.root_cause_analysis`).
# Written 2026-11-09 after the second l apply stopped at `rank --arm rca/baro` on every box: the chain ran in the `train`
# environment (Python 3.11, the package installed with `--no-deps -e .`, no extras), where RCAEval was not installed — the local
# environment had carried the extras since 2026-10-21 and no replica had ever run `rca/baro` on a box. The job scripts run that
# one command under this environment's interpreter (`$BP`), as `rankrcd` drives the RCD environment through `--rcd-python`.
# Idempotent: an existing environment is kept and the install step re-run (a no-op when nothing changed).
#   bash scripts/envs/rca-env.sh                                  # from benchmarks/trace-bench-rca, with conda on PATH
#   RCA_ENV_NAME=<throwaway name> bash scripts/envs/rca-env.sh    # a from-scratch build beside an existing local environment
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
ENV_NAME=${RCA_ENV_NAME:-trace-bench-rca}
PREFIX="$(conda info --base)/envs/$ENV_NAME"
if [ ! -x "$PREFIX/bin/python" ]; then
  conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi
"$PREFIX/bin/python" -m pip install -e "$HERE[baro,rcaeval]"
"$PREFIX/bin/python" - "$HERE" <<'PY'
import importlib, importlib.metadata as md, json, re, sys
here = sys.argv[1]
assert sys.version_info[:2] == (3, 12), f"python {sys.version.split()[0]}: the RCAEval pin imports leniently only on 3.10 / 3.12 / 3.14"
from tracebenchrca.constants import BARO_COMMIT, RCAEVAL_COMMIT

def commit_of(dist):
    d = md.distribution(dist)
    f = next((p for p in d.files or [] if p.name == "direct_url.json"), None)
    assert f is not None, f"{dist} is not a VCS install"
    return json.loads(d.locate_file(f).read_text())["vcs_info"]["commit_id"]

want_tb = re.search(r"trace-bench @ git\+\S+@([0-9a-f]{40})", open(f"{here}/requirements.txt").read()).group(1)
got = {"RCAEval": commit_of("RCAEval"), "fse-baro": commit_of("fse-baro"), "trace-bench": commit_of("trace-bench")}
want = {"RCAEval": RCAEVAL_COMMIT, "fse-baro": BARO_COMMIT, "trace-bench": want_tb}
for k in want:
    print(f"{got[k][:16]}  {k}")
    assert got[k] == want[k], f"{k} installed at {got[k][:16]}, the pin is {want[k][:16]}"
baro = importlib.import_module("RCAEval.e2e.baro").baro
importlib.import_module("baro.root_cause_analysis").robust_scorer
import tracebench
print("rca environment ready:", sys.executable, "python", sys.version.split()[0], "tracebench", tracebench.__version__, "baro", baro.__module__)
PY
