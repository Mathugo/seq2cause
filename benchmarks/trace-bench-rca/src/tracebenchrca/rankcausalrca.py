"""Rank the deployed services of every case of one export with CausalRCA, one series (method side).

    python -m tracebenchrca.rankcausalrca --export-record <export folder> --rung xs --series latency-95 \
        --causalrca-src <pinned checkout> --causalrca-python <pinned env>/bin/python --device cpu --workers 8 \
        --work-dir out/causalrca/xs-s0/latency-95 --prior-rung-record smallest-rung \
        --staging-ledger ../trace-bench/plans/staging-ledger.json --output-folder <arm folder>/<series>/rank

The `rank` stage of `rca/causalrca` (`plans/rca-arms.md`, addenda 2026-10-22, later (D-TBR-5) and
2026-10-23): `CAUSALRCA_RUNS` trainings per case of the authors' script with its data block substituted,
in the pinned environment, each in a directory of its own under `--work-dir` (git-ignored; the inputs,
substituted source, removed block and the script's outputs stay there, hashed in this record). Writes
`ranking-positional.json` — per case directory the services, every run's ranking (`null` for a failed
run) and the mean `|adj|` over the successful runs — keyed by case directory, never by case id
(D-TBR-3); `tracebenchrca.scorecausalrca` restores the ids on the score side. Staging names the previous
rung's rank record of the same arm and series.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from .arms import code_of
from .cases import list_cases
from .causalrca import rank_export
from .constants import (
    ARM_CAUSALRCA,
    CAUSALRCA_ARGV,
    CAUSALRCA_EPOCHS,
    CAUSALRCA_RUNS,
    CAUSALRCA_SERIES,
    RESULTS_JSON,
    RUN_DIR,
    RUNGS,
)
from .log import log
from .rank import POSITIONAL_JSON, STAGE, check_prior_arm, read_export_record
from .record import RunRecord, read_json, sha256_file, write_json
from .staging import StagingRefusal, require_prior_rung


def check_prior_series(staging, series):
    if staging["form"] == "record":
        prior = read_json(Path(staging["path"]) / RUN_DIR / RESULTS_JSON).get("series")
    elif staging["form"] == "ledger":
        prior = staging["entry"].get("series")
    else:
        return staging
    if prior != series:
        raise StagingRefusal(f"the prior-rung record is a {prior!r} read; this read is {series!r}")
    return staging


def pinned_environment(python):
    out = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    return sorted(ln for ln in out.splitlines() if ln.strip())


def rank(export_record, rung, series, src, python, device, workers, work_dir, prior, ledger, out):
    export_res, export_sha = read_export_record(export_record, rung)
    staging = check_prior_series(check_prior_arm(require_prior_rung(prior, rung, STAGE, ledger), ARM_CAUSALRCA), series)
    t0 = time.time()
    src_info, cases = rank_export(
        list_cases(export_res["export_dir"]), series, src, python, Path(work_dir), workers, device == "gpu"
    )
    out = Path(out)
    write_json(
        out / POSITIONAL_JSON,
        {
            "series": series,
            "cases": {
                c["case_dir"]: {"services": c["services"], "rankings": c["rankings"], "mean_abs_adj": c["mean_abs_adj"]}
                for c in cases
            },
        },
    )
    runs = [r for c in cases for r in c["runs"]]
    return {
        "stage": STAGE,
        "arm": ARM_CAUSALRCA,
        "series": series,
        "code": {**code_of(ARM_CAUSALRCA), **src_info, "epochs": CAUSALRCA_EPOCHS, "argv": list(CAUSALRCA_ARGV)},
        "pinned_environment": {"python": str(python), "pip_freeze": pinned_environment(python)},
        "device": device,
        "workers": workers,
        "rung": rung,
        "corpus_id": export_res["corpus_id"],
        "export_record_sha256": export_sha,
        "staging": staging,
        "ranking_file": POSITIONAL_JSON,
        "ranking_sha256": sha256_file(out / POSITIONAL_JSON),
        "n_cases": len(cases),
        "n_runs_per_case": CAUSALRCA_RUNS,
        "n_failed_runs": sum(c["n_failed_runs"] for c in cases),
        "failures_by_reason": {
            f: sum(1 for r in runs if r["failure"] == f) for f in sorted({r["failure"] for r in runs if r["failure"]})
        },
        "n_unranked_cases": sum(1 for c in cases if c["n_failed_runs"] == c["n_runs"]),
        "n_filled_cells": sum(c["input"]["n_filled"] for c in cases),
        "n_metrics_lt_min_pre": 0,  # no pre-fault fit: the method reads the fault window only
        "training_wall_clock_s": {
            "sum": round(sum((r.get("runner") or {}).get("wall_clock_s", 0.0) for r in runs), 1),
            "max": max([0.0] + [(r.get("runner") or {}).get("wall_clock_s", 0.0) for r in runs]),
            "stage": round(time.time() - t0, 1),
        },
        "cases": [{k: v for k, v in c.items() if k not in ("rankings", "mean_abs_adj")} for c in cases],
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-record", required=True, help="the export stage's output folder (its run/ record)")
    p.add_argument("--rung", required=True, choices=RUNGS, help="must equal the export's rung")
    p.add_argument("--series", required=True, choices=CAUSALRCA_SERIES)
    p.add_argument(
        "--causalrca-src", required=True, help="the pinned checkout (commit, clean tree, file hashes checked)"
    )
    p.add_argument("--causalrca-python", required=True, help="the pinned environment's interpreter (Python 3.9)")
    p.add_argument("--device", required=True, choices=("cpu", "gpu"), help="cpu: single-threaded trainings in parallel")
    p.add_argument("--workers", required=True, type=int, help="trainings run in parallel processes")
    p.add_argument(
        "--work-dir", required=True, help="git-ignored root of the per-training directories (must not exist)"
    )
    p.add_argument(
        "--prior-rung-record",
        required=True,
        help="smallest-rung | the previous rung's rank folder of the same arm and series | its ledger sha256",
    )
    p.add_argument("--staging-ledger", required=True, help="the shared staging ledger (constants.STAGING_LEDGER)")
    p.add_argument("--output-folder", required=True, help="where the rankings and this stage's run record are written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if Path(args.work_dir).exists():
        raise SystemExit(f"{args.work_dir} exists: every training directory is new")
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = rank(
            args.export_record,
            args.rung,
            args.series,
            args.causalrca_src,
            args.causalrca_python,
            args.device,
            args.workers,
            args.work_dir,
            args.prior_rung_record,
            args.staging_ledger,
            args.output_folder,
        )
        log(
            {
                "event": "rank",
                "arm": ARM_CAUSALRCA,
                "series": args.series,
                "corpus_id": res["corpus_id"],
                "n_cases": res["n_cases"],
                "n_failed_runs": res["n_failed_runs"],
            }
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
