"""Freeze CausalRCA's graph-axis edge threshold for one rung and series from its seed-0 validation read
(`plans/rca-arms.md`, addenda 2026-10-22 (Q16) and 2026-10-23).

    python -m tracebenchrca.graphfreeze --val-read <arm folder>/<series>/graph-val --freezes-dir freezes \
        --output-folder <arm folder>/<series>/graph-freeze

τ = the argmax over the registered grid of the mean directed F1 over the seed-0 cases, ties → the
larger τ; a τ at either end of the grid is flagged `at_grid_edge` (recorded, never refused). Writes
`freezes/<date>-<rung>-rca-causalrca-<series>-graph.json` and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .constants import ARM_CAUSALRCA, CAUSALRCA_GRAPH_TAUS, RESULTS_JSON, RUN_DIR
from .graphaxis import SCORE_SERVICE_JSON
from .graphaxis import STAGE as AXIS_STAGE
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json

STAGE = "graph-freeze"
SCHEMA = "tracebenchrca/graph-freeze@1"


def choose(mean_over_cases, grid):
    """(τ, its mean F1): the largest F1, ties to the larger τ."""
    scored = [(mean_over_cases[repr(float(t))]["f1"], float(t)) for t in grid]
    best = max(f for f, _ in scored)
    tau = max(t for f, t in scored if f == best)
    return tau, best


def freeze(val_read, freezes_dir):
    val_read = Path(val_read)
    rec = read_json(val_read / RUN_DIR / RESULTS_JSON)
    if rec.get("status") != "ok" or rec.get("stage") != AXIS_STAGE or rec.get("split") != "val":
        raise ValueError(f"{val_read} is not a completed graph-axis validation read")
    doc = read_json(val_read / SCORE_SERVICE_JSON)
    if sha256_file(val_read / SCORE_SERVICE_JSON) != rec["score_service_sha256"]:
        raise ValueError(f"{val_read / SCORE_SERVICE_JSON} differs from the file its record hashed")
    grid = [float(t) for t in CAUSALRCA_GRAPH_TAUS]
    if doc["taus"] != grid or doc["seed"] != 0:
        raise ValueError("the validation read is not the seed-0 read over the registered grid")
    tau, f1 = choose(doc["mean_over_cases"], grid)
    name = f"{dt.date.today().isoformat()}-{doc['rung']}-{ARM_CAUSALRCA.replace('/', '-')}-{doc['series']}-graph.json"
    path = Path(freezes_dir) / name
    if path.exists():
        raise FileExistsError(f"{path} exists; a freeze is never overwritten")
    write_json(
        path,
        {
            "schema": SCHEMA,
            "arm": ARM_CAUSALRCA,
            "rung": doc["rung"],
            "series": doc["series"],
            "validation_corpus": doc["corpus_id"],
            "tau": tau,
            "val_mean_f1": f1,
            "grid": grid,
            "at_grid_edge": tau in (grid[0], grid[-1]),
            "mean_f1_by_tau": {k: v["f1"] for k, v in doc["mean_over_cases"].items()},
            "val_read_sha256": sha256_file(val_read / RUN_DIR / RESULTS_JSON),
            "rule": "argmax of the mean directed F1 over the seed-0 cases at the service view's default floor; ties to the larger tau",
        },
    )
    return {
        "stage": STAGE,
        "freeze": str(path),
        "freeze_sha256": sha256_file(path),
        "rung": doc["rung"],
        "series": doc["series"],
        "tau": tau,
        "val_mean_f1": f1,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-read", required=True, help="the seed-0 graph-axis validation read of the rung and series")
    p.add_argument("--freezes-dir", required=True)
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = freeze(args.val_read, args.freezes_dir)
        log({"event": "graph_freeze", **{k: res[k] for k in ("rung", "series", "tau")}})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
