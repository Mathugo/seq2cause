"""Execute CausalRCA's substituted `train_latency.py` in the pinned environment (D-TBR-5;
`plans/rca-arms.md`, addenda 2026-10-22, later, and 2026-10-23).

    cd <training dir> && <pinned python> causalrca_runner.py --mode train --src <pinned checkout> --seed <run index>
    cd <training dir> && <pinned python> causalrca_runner.py --mode convert --src <pinned checkout> --seed <run index>

The training directory already holds `train_latency_substituted.py` and `input.npy`, written by
`tracebenchrca.causalrca` on the method side. This file runs under the authors' Python 3.9 stack and
imports nothing from `tracebenchrca`: numpy first (under Rosetta 2, importing torch first faults),
then torch; it seeds both with the run index, puts the pinned checkout on the import path for the
script's `from utils import *` / `from modules import *` / `from config import CONFIG`, sets the
registered command line and executes the source (`--mode train`; it writes `runner.json` and prints
nothing of its own, so the last line of standard output is the script's ranking). The script opens its
graph pickle without closing it, so the pickle is complete only once the training process has exited;
a second invocation (`--mode convert`) then converts it to `adj.npy` in the environment that wrote it.
"""

import argparse
import json
import os
import pickle
import platform
import sys
import time

import numpy as np

SOURCE = "train_latency_substituted.py"
GRAPH_PKL = "cpu-hog_front-end_adj.pkl"
ADJ_NPY = "adj.npy"
RUNNER_JSON = "runner.json"
ARGV = ["train_latency.py", "--gamma", "0.25", "--eta", "10"]
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "MPLBACKEND", "PYTHONDONTWRITEBYTECODE")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("train", "convert"))
    p.add_argument("--src", required=True, help="the pinned CausalRCA checkout (its directory goes on sys.path)")
    p.add_argument("--seed", required=True, type=int, help="the run index; seeds torch and numpy")
    args = p.parse_args(argv)
    if args.mode == "convert":
        with open(GRAPH_PKL, "rb") as f:
            adj = pickle.load(f)
        np.save(ADJ_NPY, np.asarray(adj, dtype=np.float64))
        return 0

    import torch  # after numpy (module top)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, os.path.abspath(args.src))
    with open(SOURCE, encoding="utf-8") as f:
        source = f.read()
    sys.argv = list(ARGV)
    t0 = time.time()
    exec(compile(source, SOURCE, "exec"), {"__name__": "__main__"})
    wall = time.time() - t0
    meta = {
        "seed": args.seed,
        "wall_clock_s": round(wall, 3),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "torch_threads": int(torch.get_num_threads()),
        "env": {k: os.environ.get(k) for k in THREAD_ENV},
    }
    with open(RUNNER_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
