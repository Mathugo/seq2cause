"""Rank the deployed services of every case of one export with RCD, ten seeded runs per case (method side).

    python -m tracebenchrca.rankrcd --export-record <export stage folder> --rung xs \
        --rcd-python <pinned env>/bin/python --workers 8 --work-dir <scratch> \
        --prior-rung-record smallest-rung --staging-ledger ../trace-bench/plans/staging-ledger.json \
        --output-folder <run-dir>

`plans/rca-arms.md`, addendum 2026-10-25, later (D-TBR-6, D-TBR-7). Per case the filled table
(`cases.read_table`, D-TBR-2) is written once as `input.csv` into the case's work directory; the pinned
environment's interpreter then runs `rcd_runner.py` once per run (`--seed r`, r = 0 … 9), in parallel
processes, and each run's metric ranking is collapsed to services by first occurrence (the suite's
coarse collapse). A run that exits non-zero or returns no list is a failed run (no hit; counted with
its reason; never re-run). Services absent from a run's ranking are misses for that run, counted as
`n_services_unranked_per_run`.

Writes `ranking-positional.json` — `{"cases": {"case-00": {"services": [...], "rankings": [[...] | null, ...]}}}`
— keyed by case directory (D-TBR-3), the shape `scorercd` (and CausalRCA's score side) restores the
case ids from. The record names the pinned environment (`pip freeze`), the overlay digests the runner
reported, every run's kept-column count, wall clock and failure reason.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .arms import check_input, code_of
from .cases import collapse_to_services, list_cases, read_services, read_table, read_window
from .constants import (
    ARM_RCD,
    INPUT_SERIES,
    INPUT_SERVICES,
    INPUT_WINDOW,
    RCD_DATASET,
    RCD_KWARGS,
    RCD_OVERLAY_SHA256,
    RCD_RUNS,
    RUNGS,
)
from .log import log
from .rank import POSITIONAL_JSON, STAGE, check_prior_arm, read_export_record
from .rankcausalrca import pinned_environment
from .record import RunRecord, sha256_file, write_json
from .staging import require_prior_rung

RUNNER = Path(__file__).with_name("rcd_runner.py")
INPUT_CSV = "input.csv"


def _spawn(cmd):
    """The runner process (a seam for tests; the record's own git calls use `subprocess` directly)."""
    return subprocess.run(cmd, capture_output=True, text=True)


def run_once(python, work, inject_time, seed):
    """One RCD run in the pinned environment; returns the runner's JSON or a failure reason."""
    out = work / f"ranking-{seed}.json"
    cmd = [
        str(python),
        str(RUNNER),
        "--input",
        str(work / INPUT_CSV),
        "--inject-time",
        str(inject_time),
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    t0 = time.time()
    proc = _spawn(cmd)
    if proc.returncode != 0 or not out.exists():
        return None, f"exit code {proc.returncode}: {proc.stderr.strip()[-300:]}", time.time() - t0
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc.get("ranks"), list):
        return None, "no ranking returned", time.time() - t0
    return doc, None, time.time() - t0


def rank_case(case_dir, python, work, workers):
    check_input(ARM_RCD, INPUT_SERIES)
    check_input(ARM_RCD, INPUT_SERVICES)
    check_input(ARM_RCD, INPUT_WINDOW)
    df, n_filled = read_table(case_dir, fill=True)
    services = [s for s, _ in read_services(case_dir)]
    inject_time = read_window(case_dir)["inject_time"]
    work.mkdir(parents=True, exist_ok=True)
    df.to_csv(work / INPUT_CSV, index=False)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(lambda r: run_once(python, work, inject_time, r), range(RCD_RUNS)))
    runs, rankings = [], []
    for r, (doc, failure, wall) in enumerate(results):
        if doc is None:
            runs.append({"run": r, "failure": failure, "wall_clock_s": round(wall, 3)})
            rankings.append(None)
            continue
        ranked = collapse_to_services([str(c) for c in doc["ranks"]])
        check = {k: v for k, v in doc["overlay_sha256"].items()}
        bad = [f for f, want in RCD_OVERLAY_SHA256.items() if not check.get(f, "").startswith(want)]
        if bad:
            raise RuntimeError(
                f"the pinned environment's causal-learn overlay differs from the registered files: {bad}"
            )
        runs.append(
            {
                "run": r,
                "failure": None,
                "n_metrics_ranked": len(doc["ranks"]),
                "n_services_ranked": len(ranked),
                "n_services_unranked": len(set(services) - set(ranked)),
                "n_columns_in": doc["n_columns_in"],
                "n_columns_kept": doc["n_columns_kept"],
                "wall_clock_s": doc["wall_clock_s"],
                "overlay_sha256": doc["overlay_sha256"],
            }
        )
        rankings.append(ranked)
    return {
        "case_dir": case_dir.name,
        "services": services,
        "rankings": rankings,
        "n_rows": int(len(df)),
        "n_pre_rows": int((df["time"] < inject_time).sum()),
        "n_filled": n_filled,
        "n_runs": RCD_RUNS,
        "n_failed_runs": sum(1 for x in rankings if x is None),
        "n_services_unranked_per_run": [x["n_services_unranked"] for x in runs if x["failure"] is None],
        "runs": runs,
    }


def rank(export_record, rung, python, workers, work_dir, prior, ledger, out):
    export_res, export_sha = read_export_record(export_record, rung)
    staging = check_prior_arm(require_prior_rung(prior, rung, STAGE, ledger), ARM_RCD)
    t0 = time.time()
    cases = []
    for d in list_cases(export_res["export_dir"]):
        row = rank_case(d, python, Path(work_dir) / d.name, workers)
        cases.append(row)
        log({"event": "rank_case", "arm": ARM_RCD, "case_dir": d.name, "failed_runs": row["n_failed_runs"]})
    out = Path(out)
    write_json(
        out / POSITIONAL_JSON,
        {"cases": {c["case_dir"]: {"services": c["services"], "rankings": c["rankings"]} for c in cases}},
    )
    runs = [r for c in cases for r in c["runs"]]
    return {
        "stage": STAGE,
        "arm": ARM_RCD,
        "code": {**code_of(ARM_RCD), "dataset": RCD_DATASET, "kwargs": dict(RCD_KWARGS), "runner": RUNNER.name},
        "pinned_environment": {"python": str(python), "pip_freeze": pinned_environment(python)},
        "workers": workers,
        "rung": rung,
        "corpus_id": export_res["corpus_id"],
        "export_record_sha256": export_sha,
        "staging": staging,
        "ranking_file": POSITIONAL_JSON,
        "ranking_sha256": sha256_file(out / POSITIONAL_JSON),
        "n_cases": len(cases),
        "n_runs_per_case": RCD_RUNS,
        "n_failed_runs": sum(c["n_failed_runs"] for c in cases),
        "failures_by_reason": {
            f: sum(1 for r in runs if r["failure"] == f) for f in sorted({r["failure"] for r in runs if r["failure"]})
        },
        "n_unranked_cases": sum(1 for c in cases if c["n_failed_runs"] == c["n_runs"]),
        "misses_by_reason": {},
        "n_empty_rankings": sum(1 for c in cases for x in c["rankings"] if x == []),
        "n_filled_cells": sum(c["n_filled"] for c in cases),
        "n_metrics_lt_min_pre": 0,
        "wall_clock_s": {
            "runs_sum": round(sum(r["wall_clock_s"] for r in runs), 1),
            "stage": round(time.time() - t0, 1),
        },
        "cases": [{k: v for k, v in c.items() if k != "rankings"} for c in cases],
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-record", required=True, help="the export stage's output folder (its run/ record)")
    p.add_argument("--rung", required=True, choices=RUNGS, help="must equal the export's rung")
    p.add_argument("--rcd-python", required=True, help="the pinned RCD environment's interpreter (environment-rcd.yml)")
    p.add_argument("--workers", required=True, type=int, help="runs of one case in parallel processes")
    p.add_argument("--work-dir", required=True, help="scratch directory for the per-case inputs and per-run outputs")
    p.add_argument(
        "--prior-rung-record",
        required=True,
        help="smallest-rung | the previous rung's rankrcd folder | its ledger sha256 | an owner-dated exception (.md)",
    )
    p.add_argument("--staging-ledger", required=True, help="the shared staging ledger (constants.STAGING_LEDGER)")
    p.add_argument("--output-folder", required=True, help="where the rankings and this stage's run record are written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = rank(
            args.export_record,
            args.rung,
            Path(args.rcd_python),
            args.workers,
            args.work_dir,
            args.prior_rung_record,
            args.staging_ledger,
            args.output_folder,
        )
        log(
            {
                "event": "rank",
                "arm": ARM_RCD,
                "corpus_id": res["corpus_id"],
                "n_cases": res["n_cases"],
                "n_failed_runs": res["n_failed_runs"],
            }
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
