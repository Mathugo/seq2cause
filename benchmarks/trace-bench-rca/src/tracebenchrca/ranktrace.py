"""Probe every case of one export with TRACE's own root-cause rule (method side) — the case-level ranking axis or the
per-trace axis.

    python -m tracebenchrca.ranktrace --arm rca/trace --axis trace --select post-nonok --cap 0 \
        --export-record <export stage folder> --corpus-dir <pulled corpus> --model-dir <the seed's frozen backbone> \
        --freeze ../trace-bench/freezes/<date>-xs-latent-s0.json --rung xs --seed 0 \
        --trace-python <seq2cause-trace-bench python> --harness-root <the repository root> --device cpu \
        --work-dir <scratch> --prior-rung-record smallest-rung \
        --staging-ledger ../trace-bench/plans/staging-ledger.json --output-folder <run-dir>

`plans/rca-arms.md`, addendum 2026-10-30 (D-TBR-13), and the per-trace axis plan of 2026-10-31. The arm reads each
case's `window.json` and `services.json` through the case door and the corpus's end-ordered request view (the
declared `views` input) through the benchmark's method accessor inside the runner — never `data.csv`, never the
labels, graphs or topology. The reference harness's environment runs `trace_runner.py` once per corpus and arm (the
backbone loaded once; the `rankrcd` pattern): per case, the traces `--select` names (`window`: start in
`[inject_time, end_time)`, the landed case-level runs; `post-nonok`: rule R, the whole trace inside the window with
a non-client non-OK span), all three splits, ordered by start time then trace id, the first `--cap` (0 = all), are
probed with the arm's frozen cell (`TRACE_CELLS`: the cell key of the seed's committed freeze, whose `model_sha256`
the backbone must match) and cut at its frozen tau.

- `--axis ranking` (stage `rank`): the rule (`tracerule.rank_case`) ranks the deployed services per case — the number
  of probed traces in which a service is upstream of a non-OK position, ties by mean hop, then `services.json` order;
  a case without evidence is a counted miss. Writes `ranking-positional.json` — `{"cases": {"case-00": [service, …]}}`,
  keyed by case directory (D-TBR-3), the shape `tracebenchrca.score` restores the case ids from.
- `--axis trace` (stage `rank-trace`): the rule (`tracerule.rank_trace`) ranks, inside every probed trace, the
  services and window positions upstream of its non-OK positions under both readings (`hop1`: upstream only; `hop0`:
  plus the failed position itself). Writes `ranking-trace.json` (schema `ranking-trace@1`): the cell's `c`, the
  length, selection and cap, the export's service names (name order) and, per case, the selected and skipped trace
  ids and per probed trace the reached service indices and window positions of each reading (`s`/`p`, `s0`/`p0`).
  `tracebench.score_rca_trace` scores it against the benchmark's per-trace labels.

The record names the cell and its `(c, N, g, tau)`, the model and freeze hashes, the runner's environment
(`pip freeze`), import paths and commit, the device, the selection and cap, and per case the window, candidate,
probed, skipped, truncated and folded counts, the non-OK and upstream counts, and (ranking axis) the per-service
counts, mean hops, the hop-0 diagnostic and the miss reason, or (trace axis) the mean reached-set sizes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from .arms import check_input, code_of
from .cases import list_cases, read_services, read_window
from .constants import (
    ARM_TRACE,
    ARM_TRACE_CORE,
    AXIS_RANKING,
    AXIS_TRACE,
    INPUT_SERVICES,
    INPUT_VIEWS,
    INPUT_WINDOW,
    RANKING_TRACE_JSON,
    RUNGS,
    SEEDS,
    STAGE_RANK_TRACE,
    TRACE_CELLS,
    TRACE_DEVICES,
    TRACE_MAX_LEN,
    TRACE_NONOK_OUTCOMES,
    TRACE_READINGS,
    TRACE_SELECTS,
    VARIANT,
)
from .export import corpus_id_of
from .log import log
from .rank import POSITIONAL_JSON, STAGE, check_prior_arm, read_export_record
from .rankcausalrca import pinned_environment
from .record import RunRecord, read_json, sha256_file, write_json
from .staging import require_prior_rung
from .tracerule import rank_case, rank_trace

RUNNER = Path(__file__).with_name("trace_runner.py")
TRACE_ARMS = (ARM_TRACE, ARM_TRACE_CORE)
AXES = (AXIS_RANKING, AXIS_TRACE)
STAGES = {AXIS_RANKING: STAGE, AXIS_TRACE: STAGE_RANK_TRACE}
RANKING_TRACE_SCHEMA = "ranking-trace@1"
CASES_JSON = "cases.json"
SUMMARY_JSON = "summary.json"
HARNESS_SRC = ("benchmarks/trace-bench/src", "src")  # seq2causebench and the shipped seq2cause, from one commit
DOC_COUNTS = (  # the runner's per-case counts both axes carry into the record
    "n_traces_window",
    "n_traces_candidates",
    "n_traces_selected",
    "n_traces_probed",
    "n_skipped_short",
    "n_truncated",
    "n_folded",
    "n_nonok_folded",
    "tok_pos",
    "probe_wall_clock_s",
)
RANKING_DOC_COUNTS = (  # the ranking axis's record rows, as landed (2026-10-30)
    "n_traces_window",
    "n_traces_selected",
    "n_skipped_short",
    "n_truncated",
    "n_folded",
    "tok_pos",
    "probe_wall_clock_s",
)


def _spawn(cmd, env):
    """The runner process (a seam for tests)."""
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def runner_paths(harness_root):
    root = Path(harness_root).resolve()
    paths = [root / rel for rel in HARNESS_SRC]
    missing = [str(p) for p in paths if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"--harness-root {root} lacks {missing}")
    return [str(p) for p in paths]


def run_runner(
    python, harness_root, corpus_dir, cases_file, model_dir, freeze, cell, select, cap, device, seed, out_dir
):
    paths = runner_paths(harness_root)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    cmd = [
        str(python),
        str(RUNNER),
        "--corpus-dir",
        str(corpus_dir),
        "--cases",
        str(cases_file),
        "--model",
        str(model_dir),
        "--freeze",
        str(freeze),
        "--cell",
        cell,
        "--select",
        select,
        "--cap",
        str(cap),
        "--max-len",
        str(TRACE_MAX_LEN),
        "--device",
        device,
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    t0 = time.time()
    proc = _spawn(cmd, env)
    wall = time.time() - t0
    if proc.returncode != 0 or not (Path(out_dir) / SUMMARY_JSON).exists():
        raise RuntimeError(f"the runner exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    return read_json(Path(out_dir) / SUMMARY_JSON), paths, wall


def _mean(values):
    return sum(values) / len(values) if values else None


def trace_axis_case(doc, names):
    """One case of the trace axis: the compact per-trace entries (indices into `names`) and the record row."""
    index = {s: i for i, s in enumerate(names)}
    entries, sizes1, sizes0 = {}, [], []
    n_nonok = n_with_nonok = n_with_upstream = n_client = 0
    for tr in doc["traces"]:
        h1 = rank_trace(tr, names, TRACE_NONOK_OUTCOMES, hop0=False)
        h0 = rank_trace(tr, names, TRACE_NONOK_OUTCOMES, hop0=True)
        entries[tr["trace_id"]] = {
            "s": [index[s] for s in h1["services"]],
            "p": h1["positions"],
            "s0": [index[s] for s in h0["services"]],
            "p0": h0["positions"],
        }
        sizes1.append(len(h1["services"]))
        sizes0.append(len(h0["services"]))
        n_nonok += h1["n_nonok"]
        n_with_nonok += h1["n_nonok"] > 0
        n_with_upstream += bool(h1["services"])
        n_client += h1["n_client_dropped"]
    row = {
        "case_dir": doc["case_dir"],
        "select": doc["select"],
        **{k: doc[k] for k in DOC_COUNTS},
        "n_nonok_positions": n_nonok,
        "n_traces_with_nonok": n_with_nonok,
        "n_traces_with_upstream": n_with_upstream,
        "n_client_positions_dropped": n_client,
        "mean_set_size_hop1": _mean(sizes1),
        "mean_set_size_hop0": _mean(sizes0),
    }
    case = {"selected": list(doc["selected_trace_ids"]), "skipped": list(doc["skipped_trace_ids"]), "traces": entries}
    return case, row


def rank(
    arm,
    export_record,
    corpus_dir,
    model_dir,
    freeze,
    rung,
    seed,
    python,
    harness_root,
    device,
    work_dir,
    prior,
    ledger,
    out,
    *,
    axis,
    select,
    cap,
):
    if arm not in TRACE_ARMS:
        raise ValueError(f"arm {arm!r} is not a TRACE root-cause arm; these are {TRACE_ARMS}")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    if select not in TRACE_SELECTS:
        raise ValueError(f"select must be one of {TRACE_SELECTS}, got {select!r}")
    if cap < 0:
        raise ValueError("cap must be >= 0 (0 = every selected trace)")
    stage = STAGES[axis]
    export_res, export_sha = read_export_record(export_record, rung)
    corpus_id = corpus_id_of(corpus_dir)
    if corpus_id != export_res["corpus_id"]:
        raise ValueError(f"the export is of {export_res['corpus_id']!r}; the corpus is {corpus_id!r}")
    if corpus_id != f"{rung}/{VARIANT}/seed={seed}":
        raise ValueError(f"--seed {seed} does not name the corpus {corpus_id!r}")
    staging = check_prior_arm(require_prior_rung(prior, rung, stage, ledger), arm)
    for what in (INPUT_VIEWS, INPUT_WINDOW, INPUT_SERVICES):
        check_input(arm, what)
    dirs = list_cases(export_res["export_dir"])
    windows = [{"case_dir": d.name, **read_window(d)} for d in dirs]
    services = {d.name: [s for s, _ in read_services(d)] for d in dirs}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cases_file = work / CASES_JSON
    write_json(cases_file, windows)
    t0 = time.time()
    summary, paths, runner_wall = run_runner(
        python,
        harness_root,
        corpus_dir,
        cases_file,
        model_dir,
        freeze,
        TRACE_CELLS[arm],
        select,
        cap,
        device,
        seed,
        work,
    )
    if summary["cell"] != TRACE_CELLS[arm] or summary["corpus_id"] != corpus_id:
        raise ValueError(f"the runner's summary names {summary['cell']!r} on {summary['corpus_id']!r}")
    if summary.get("select") != select or summary.get("cap") != cap:
        raise ValueError(
            f"the runner ran select={summary.get('select')!r} cap={summary.get('cap')!r}, not {select}/{cap}"
        )
    docs = {d.name: read_json(work / f"{d.name}.json") for d in dirs}
    for name, doc in docs.items():
        if doc.get("select") != select:
            raise ValueError(f"{name}: the runner's document says select={doc.get('select')!r}, not {select!r}")
    out = Path(out)
    common = {
        "stage": stage,
        "axis": axis,
        "arm": arm,
        "code": {**code_of(arm), "cell": TRACE_CELLS[arm], "runner": RUNNER.name},
        "rung": rung,
        "seed": seed,
        "corpus_id": corpus_id,
        "export_record_sha256": export_sha,
        "staging": staging,
        "select": select,
        "cap": cap,
        "freeze": {"path": str(freeze), "sha256": sha256_file(freeze)},
        "model_dir": str(model_dir),
        "device": device,
        "max_len": TRACE_MAX_LEN,
        "nonok_outcomes": list(TRACE_NONOK_OUTCOMES),
        "runner": summary,
        "pinned_environment": {"python": str(python), "pythonpath": paths, "pip_freeze": pinned_environment(python)},
    }
    if axis == AXIS_RANKING:
        return _rank_ranking_axis(arm, dirs, docs, services, common, out, runner_wall, t0)
    return _rank_trace_axis(arm, dirs, docs, services, common, summary, out, runner_wall, t0)


def _rank_ranking_axis(arm, dirs, docs, services, common, out, runner_wall, t0):
    cases, positional = [], {}
    for d in dirs:
        doc = docs[d.name]
        row = {
            "case_dir": d.name,
            **{k: doc[k] for k in RANKING_DOC_COUNTS},
            **rank_case(doc["traces"], services[d.name], TRACE_NONOK_OUTCOMES),
            "n_filled": 0,
        }
        if row["services"] is not None:
            positional[d.name] = row["services"]
        cases.append(row)
        log(
            {
                "event": "rank_case",
                "arm": arm,
                "case_dir": d.name,
                "ranked": row["services"] is not None,
                "n_probed": row["n_traces_probed"],
                "n_reached": row["n_services_reached"],
            }
        )
    write_json(out / POSITIONAL_JSON, {"cases": positional})
    return {
        **common,
        "ranking_file": POSITIONAL_JSON,
        "ranking_sha256": sha256_file(out / POSITIONAL_JSON),
        "n_cases": len(cases),
        "n_unranked_cases": len(cases) - len(positional),
        "misses_by_reason": {
            r: sum(1 for c in cases if c.get("miss") == r) for r in sorted({c["miss"] for c in cases if c.get("miss")})
        },
        "n_empty_rankings": 0,
        "n_filled_cells": 0,
        "n_metrics_lt_min_pre": 0,
        "n_traces_window": sum(c["n_traces_window"] for c in cases),
        "n_traces_probed": sum(c["n_traces_probed"] for c in cases),
        "wall_clock_s": {"runner": round(runner_wall, 1), "stage": round(time.time() - t0, 1)},
        "cases": cases,
    }


def _rank_trace_axis(arm, dirs, docs, services, common, summary, out, runner_wall, t0):
    names = services[dirs[0].name]
    for d in dirs:
        if services[d.name] != names:
            raise ValueError(f"{d.name}/services.json differs from {dirs[0].name}'s; one service order per export")
    cases, rows = {}, []
    for d in dirs:
        case, row = trace_axis_case(docs[d.name], names)
        cases[d.name] = case
        rows.append(row)
        log(
            {
                "event": "rank_trace_case",
                "arm": arm,
                "case_dir": d.name,
                "n_candidates": row["n_traces_candidates"],
                "n_probed": row["n_traces_probed"],
                "n_skipped": row["n_skipped_short"],
                "n_with_upstream": row["n_traces_with_upstream"],
            }
        )
    ranking = {
        "schema": RANKING_TRACE_SCHEMA,
        "arm": arm,
        "cell": TRACE_CELLS[arm],
        "c": summary["c"],
        "max_len": common["max_len"],
        "select": common["select"],
        "cap": common["cap"],
        "device": common["device"],
        "services": names,
        "cases": cases,
    }
    write_json(out / RANKING_TRACE_JSON, ranking)
    return {
        **common,
        "ranking_file": RANKING_TRACE_JSON,
        "ranking_sha256": sha256_file(out / RANKING_TRACE_JSON),
        "readings": list(TRACE_READINGS),
        "services": names,
        "n_cases": len(rows),
        "n_traces_window": sum(r["n_traces_window"] for r in rows),
        "n_traces_candidates": sum(r["n_traces_candidates"] for r in rows),
        "n_traces_probed": sum(r["n_traces_probed"] for r in rows),
        "n_skipped_short": sum(r["n_skipped_short"] for r in rows),
        "n_nonok_folded": sum(r["n_nonok_folded"] for r in rows),
        "wall_clock_s": {"runner": round(runner_wall, 1), "stage": round(time.time() - t0, 1)},
        "cases": rows,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=TRACE_ARMS)
    p.add_argument(
        "--axis", required=True, choices=AXES, help="ranking: the case-level stage; trace: the per-trace axis"
    )
    p.add_argument(
        "--select", required=True, choices=TRACE_SELECTS, help="window: the landed case-level rule; post-nonok: rule R"
    )
    p.add_argument("--cap", required=True, type=int, help="selected traces probed per case, in start order; 0 = all")
    p.add_argument("--export-record", required=True, help="the export stage's output folder (its run/ record)")
    p.add_argument("--corpus-dir", required=True, help="the pulled corpus (tier rca) the export was made from")
    p.add_argument(
        "--model-dir", required=True, help="the seed's frozen backbone directory (the freeze's model_sha256)"
    )
    p.add_argument("--freeze", required=True, help="the reference harness's committed freeze record of this corpus")
    p.add_argument("--rung", required=True, choices=RUNGS, help="must equal the export's rung")
    p.add_argument("--seed", required=True, type=int, choices=SEEDS, help="the corpus seed")
    p.add_argument("--trace-python", required=True, help="the reference harness environment's interpreter")
    p.add_argument("--harness-root", required=True, help="the repository root (seq2causebench and seq2cause sources)")
    p.add_argument("--device", required=True, choices=TRACE_DEVICES)
    p.add_argument("--work-dir", required=True, help="scratch directory for the runner's inputs and per-case outputs")
    p.add_argument(
        "--prior-rung-record",
        required=True,
        help="smallest-rung | the previous rung's ranktrace folder of the same arm and axis | its ledger sha256 | an owner-dated exception (.md)",
    )
    p.add_argument("--staging-ledger", required=True, help="the shared staging ledger (constants.STAGING_LEDGER)")
    p.add_argument("--output-folder", required=True, help="where the ranking and this stage's run record are written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGES[args.axis], vars(args)) as rec:
        res = rank(
            args.arm,
            args.export_record,
            args.corpus_dir,
            args.model_dir,
            args.freeze,
            args.rung,
            args.seed,
            Path(args.trace_python),
            args.harness_root,
            args.device,
            args.work_dir,
            args.prior_rung_record,
            args.staging_ledger,
            args.output_folder,
            axis=args.axis,
            select=args.select,
            cap=args.cap,
        )
        log(
            {
                "event": res["stage"],
                "arm": args.arm,
                "axis": args.axis,
                "corpus_id": res["corpus_id"],
                "n_cases": res["n_cases"],
                "n_unranked": res.get("n_unranked_cases"),
                "n_traces_probed": res["n_traces_probed"],
            }
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
