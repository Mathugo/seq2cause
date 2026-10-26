"""Run the suite's RCD once, in the pinned RCD environment (D-TBR-6, D-TBR-7; `plans/rca-arms.md`,
addendum 2026-10-25, later).

    <pinned python> rcd_runner.py --input input.csv --inject-time <UNIX s> --seed <run index> --out ranking.json

The input is the case's filled table as the method side wrote it (`time` plus every `<service>_<metric>`
column). This file runs under the suite's RCD stack (Python 3.10, causal-learn 0.1.2.3 with the suite's
four patched files) and imports nothing from `tracebenchrca`. It refuses an environment whose
causal-learn lacks `local_skeleton_discovery`, calls `RCAEval.e2e.rcd.rcd(table, inject_time,
dataset="trace-bench", seed=<run index>, gamma=5, localized=True, bins=5, dk_select_useful=False)` —
the dataset name selects the suite's named-dataset preprocessing (time dropped, constant columns
dropped per segment, columns matched) — and writes the returned metric ranking with the facts of the
run: the columns that preprocessing kept (recomputed with the suite's own functions, for the count
only), the sha256 of the four overlaid causal-learn files, versions and wall clock.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings

DATASET = "trace-bench"
KWARGS = {"gamma": 5, "localized": True, "bins": 5, "dk_select_useful": False}
OVERLAY = ("utils/PCUtils/SkeletonDiscovery.py", "utils/Fas.py", "search/ConstraintBased/FCI.py", "graph/GraphClass.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--inject-time", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    import causallearn
    import numpy as np
    import pandas as pd
    from causallearn.utils.PCUtils import SkeletonDiscovery

    if not hasattr(SkeletonDiscovery, "local_skeleton_discovery"):
        print(
            "causal-learn lacks local_skeleton_discovery: the suite's lib/causallearn overlay is not in place",
            file=sys.stderr,
        )
        return 3
    import RCAEval
    from RCAEval.e2e.rcd import _match_columns, rcd
    from RCAEval.io.time_series import drop_constant, drop_time

    root = os.path.dirname(causallearn.__file__)
    overlay = {f: sha256(os.path.join(root, f)) for f in OVERLAY}
    df = pd.read_csv(args.input)
    normal, anomal = df[df["time"] < args.inject_time], df[df["time"] >= args.inject_time]
    n_df, a_df = _match_columns(drop_constant(drop_time(normal)), drop_constant(drop_time(anomal)))
    kept = list(n_df.columns)
    t0 = time.time()
    out = rcd(df, args.inject_time, dataset=DATASET, seed=args.seed, **KWARGS)
    wall = time.time() - t0
    ranks = [str(c) for c in out["ranks"]]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ranks": ranks,
                "seed": args.seed,
                "dataset": DATASET,
                "kwargs": KWARGS,
                "n_columns_in": int(df.shape[1] - 1),
                "n_columns_kept": len(kept),
                "n_rows": int(len(df)),
                "n_pre_rows": int(len(normal)),
                "wall_clock_s": round(wall, 3),
                "overlay_sha256": overlay,
                "python": platform.python_version(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "rcaeval_file": os.path.abspath(RCAEval.__file__),
            },
            f,
            indent=1,
            sort_keys=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
