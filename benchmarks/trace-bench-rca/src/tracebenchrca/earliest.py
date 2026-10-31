"""`control/trace-earliest` — the earliest-error control (`plans/controls.md` §3, addendum 2026-10-31; D-TBR-19):
"the first thing that failed", on the case-level ranking axis and the per-trace axis. Method side, label-free: it
reads each case's `window.json` and `services.json` through the case door and the corpus's end-ordered request view
(the declared `views` input: `views/end-request/sequences/split=*/…/part-*.parquet` and `model-vocab.json`) —
never the labels, the graphs or the view's call-tree links.

    python -m tracebenchrca.earliest --axis trace --select post-nonok --cap 0 --export-record <export folder> \\
        --corpus-dir <pulled corpus> --rung xs --seed 0 --prior-rung-record smallest-rung \\
        --staging-ledger ../trace-bench/plans/staging-ledger.json --output-folder <run-dir>

Per case the traces `--select` names (`window`: start in `[inject_time, end_time)`; `post-nonok`: rule R — the
whole trace inside the window and a non-client position with a raw non-OK outcome), all three splits, ordered by
`(start_ns, trace_id)`, the first `--cap` (0 = all). Per selected trace the non-client services ordered by the start
of their earliest non-OK span (`start_ns + offset`), ties by name, the trace's remaining non-client services
appended by name; positions likewise (the non-OK non-client positions by span start, then the rest by index;
`c = 1`, so a window position is the ops index). Case level (`--axis ranking`, stage `rank`): a service's score is
the number of selected traces in which it holds the trace's earliest non-OK span (every tied service), ties by
`services.json` order, every deployed service ranked; a case with no selected trace or no non-OK span is a counted
miss → `ranking-positional.json`, scored by `tracebenchrca.score`. Per trace (`--axis trace`, stage `rank-trace`):
`ranking-trace.json` (schema `ranking-trace@2`, one reading `earliest` under the keys `s` / `p`), scored by
`tracebenchrca.scoretrace --rows arm-only`.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import pyarrow.parquet as pq
from tracebench.constants import KIND_CLIENT, OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW

from .arms import check_input
from .cases import list_cases, read_services, read_window
from .constants import (
    ARM_CONTROL_TRACE_EARLIEST,
    AXIS_RANKING,
    AXIS_TRACE,
    INPUT_SERVICES,
    INPUT_VIEWS,
    INPUT_WINDOW,
    RANKING_TRACE_JSON,
    RUNGS,
    SEEDS,
    STAGE_RANK_TRACE,
    TRACE_MAX_LEN,
    TRACE_MISS_NO_NONOK,
    TRACE_MISS_NO_TRACE,
    TRACE_READING_EARLIEST,
    TRACE_SELECT_WINDOW,
    TRACE_SELECTS,
    VARIANT,
)
from .export import corpus_id_of
from .log import log
from .rank import POSITIONAL_JSON, STAGE, check_prior_arm, read_export_record
from .record import RunRecord, read_json, sha256_file, write_json
from .staging import require_prior_rung

ARM = ARM_CONTROL_TRACE_EARLIEST
AXES = (AXIS_RANKING, AXIS_TRACE)
STAGES = {AXIS_RANKING: STAGE, AXIS_TRACE: STAGE_RANK_TRACE}
NONOK_IDS = (OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW)  # = constants.TRACE_NONOK_OUTCOMES by name
VIEW = "end-request"
COLUMNS = ("trace_id", "ops", "outcomes", "offsets", "durations", "start_ns")
SPLITS = ("train", "val", "test")
RANKING_TRACE_SCHEMA = "ranking-trace@2"


def load_ops(corpus_dir):
    """`{op id: (service, is_client)}` from the view's vocabulary (a method-readable file)."""
    vocab = read_json(Path(corpus_dir) / "views" / VIEW / "model-vocab.json")
    return {int(o["id"]): (str(o["service"]), int(o["kind"]) == KIND_CLIENT) for o in vocab["base_ops"]}


def view_rows(corpus_dir):
    """Every row of the three splits of the end-ordered request view (the declared `views` input)."""
    root = Path(corpus_dir) / "views" / VIEW / "sequences"
    for split in SPLITS:
        files = sorted(glob.glob(str(root / f"split={split}" / "**" / "*.parquet"), recursive=True))
        for f in files:
            yield from pq.read_table(f, columns=list(COLUMNS)).to_pylist()


def trace_end_ns(row):
    ends = [int(o) + int(d) for o, d in zip(row["offsets"], row["durations"], strict=True)]
    return int(row["start_ns"]) + (max(ends) if ends else 0)


def nonok_spans(row, ops):
    """`[(start_ns_of_span, index, service)]` of the non-client positions with a non-OK outcome."""
    out = []
    for i, (op, o, off) in enumerate(zip(row["ops"], row["outcomes"], row["offsets"], strict=True)):
        service, client = ops[int(op)]
        if int(o) in NONOK_IDS and not client:
            out.append((int(row["start_ns"]) + int(off), i, service))
    return out


def selected(row, lo_ns, hi_ns, select, ops):
    start = int(row["start_ns"])
    if select == TRACE_SELECT_WINDOW:
        return lo_ns <= start < hi_ns
    return lo_ns <= start and trace_end_ns(row) < hi_ns and bool(nonok_spans(row, ops))


def rank_one(row, ops, services):
    """One trace: `(services ranked, positions ranked, earliest services, n_nonok)` — the non-client services by
    the start of their earliest non-OK span (ties by name), the rest by name; positions by span start, the rest by
    index."""
    spans = sorted(nonok_spans(row, ops))
    order = {s: i for i, s in enumerate(services)}
    earliest = {}
    for t, _, svc in spans:
        if svc not in earliest:
            earliest[svc] = t
    for svc in earliest:
        if svc not in order:
            raise ValueError(f"the view names a service outside services.json: {svc!r}")
    ranked = sorted(earliest, key=lambda s: (earliest[s], s))
    present = sorted({ops[int(op)][0] for op in row["ops"] if not ops[int(op)][1]} - set(earliest))
    nonok_pos = [i for _, i, _ in spans]
    rest_pos = [i for i in range(len(row["ops"])) if i not in set(nonok_pos) and not ops[int(row["ops"][i])][1]]
    first_t = spans[0][0] if spans else None
    tied = [svc for t, _, svc in spans if t == first_t] if spans else []
    return ranked + present, nonok_pos + rest_pos, sorted(set(tied), key=lambda s: order[s]), len(spans)


def run(axis, export_record, corpus_dir, rung, seed, select, cap, prior, ledger, out):
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
    staging = check_prior_arm(require_prior_rung(prior, rung, stage, ledger), ARM)
    for what in (INPUT_VIEWS, INPUT_WINDOW, INPUT_SERVICES):
        check_input(ARM, what)
    dirs = list_cases(export_res["export_dir"])
    windows = [(d.name, read_window(d)) for d in dirs]
    services = {d.name: [s for s, _ in read_services(d)] for d in dirs}
    names = services[dirs[0].name]
    if any(services[d.name] != names for d in dirs):
        raise ValueError("one service order per export")
    ops = load_ops(corpus_dir)
    t0 = time.time()
    chosen = {name: [] for name, _ in windows}
    n_rows = 0
    for row in view_rows(corpus_dir):
        n_rows += 1
        for name, w in windows:
            lo, hi = int(w["inject_time"]) * 10**9, int(w["end_time"]) * 10**9
            if selected(row, lo, hi, select, ops):
                chosen[name].append((int(row["start_ns"]), str(row["trace_id"]), row))
    out = Path(out)
    cases_rows, positional, trace_cases = [], {}, {}
    for name, _ in windows:
        rows = sorted(chosen[name], key=lambda r: (r[0], r[1]))
        rows = rows[:cap] if cap > 0 else rows
        counts = dict.fromkeys(names, 0)
        entries, n_nonok_traces, n_nonok = {}, 0, 0
        for _, tid, row in rows:
            svcs, poss, tied, k = rank_one(row, ops, names)
            n_nonok += k
            n_nonok_traces += k > 0
            for svc in tied:
                counts[svc] += 1
            entries[tid] = {"s": [names.index(s) for s in svcs], "p": poss}
        order = {s: i for i, s in enumerate(names)}
        ranked = sorted(names, key=lambda s: (-counts[s], order[s]))
        miss = TRACE_MISS_NO_TRACE if not rows else (TRACE_MISS_NO_NONOK if n_nonok == 0 else None)
        if miss is None:
            positional[name] = ranked
        cases_rows.append(
            {
                "case_dir": name,
                "select": select,
                "n_traces_selected": len(rows),
                "n_traces_probed": len(rows),
                "n_traces_with_nonok": n_nonok_traces,
                "n_nonok_positions": n_nonok,
                "counts": {s: counts[s] for s in names if counts[s] > 0},
                "services": ranked if miss is None else None,
                "miss": miss,
            }
        )
        trace_cases[name] = {"selected": [tid for _, tid, _ in rows], "skipped": [], "traces": entries}
        log({"event": "earliest_case", "case_dir": name, "n_selected": len(rows), "miss": miss})
    common = {
        "stage": stage,
        "axis": axis,
        "arm": ARM,
        "reading": TRACE_READING_EARLIEST,
        "code": None,
        "rung": rung,
        "seed": seed,
        "corpus_id": corpus_id,
        "export_record_sha256": export_sha,
        "staging": staging,
        "select": select,
        "cap": cap,
        "max_len": TRACE_MAX_LEN,
        "n_rows_read": n_rows,
        "n_cases": len(cases_rows),
        "n_traces_probed": sum(c["n_traces_probed"] for c in cases_rows),
        "wall_clock_s": round(time.time() - t0, 1),
        "cases": cases_rows,
    }
    if axis == AXIS_RANKING:
        write_json(out / POSITIONAL_JSON, {"cases": positional})
        return {
            **common,
            "ranking_file": POSITIONAL_JSON,
            "ranking_sha256": sha256_file(out / POSITIONAL_JSON),
            "n_unranked_cases": len(cases_rows) - len(positional),
            "misses_by_reason": {
                r: sum(1 for c in cases_rows if c["miss"] == r)
                for r in sorted({c["miss"] for c in cases_rows if c["miss"]})
            },
            "n_empty_rankings": 0,
            "n_filled_cells": 0,
            "n_metrics_lt_min_pre": 0,
        }
    ranking = {
        "schema": RANKING_TRACE_SCHEMA,
        "arm": ARM,
        "cell": "none",
        "readings": [TRACE_READING_EARLIEST],
        "primary_reading": TRACE_READING_EARLIEST,
        "c": 1,
        "max_len": TRACE_MAX_LEN,
        "select": select,
        "cap": cap,
        "device": "cpu",
        "services": names,
        "cases": trace_cases,
    }
    write_json(out / RANKING_TRACE_JSON, ranking)
    return {
        **common,
        "ranking_file": RANKING_TRACE_JSON,
        "ranking_sha256": sha256_file(out / RANKING_TRACE_JSON),
        "readings": [TRACE_READING_EARLIEST],
        "services": names,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", required=True, choices=AXES)
    p.add_argument("--select", required=True, choices=TRACE_SELECTS, help="window: the case rule; post-nonok: rule R")
    p.add_argument("--cap", required=True, type=int, help="selected traces per case, in start order; 0 = all")
    p.add_argument("--export-record", required=True, help="the export stage's output folder (its run/ record)")
    p.add_argument("--corpus-dir", required=True, help="the pulled corpus the export was made from")
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument(
        "--prior-rung-record", required=True, help="smallest-rung | the previous rung's record | its ledger sha256"
    )
    p.add_argument("--staging-ledger", required=True)
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGES[args.axis], vars(args)) as rec:
        res = run(
            args.axis,
            args.export_record,
            args.corpus_dir,
            args.rung,
            args.seed,
            args.select,
            args.cap,
            args.prior_rung_record,
            args.staging_ledger,
            args.output_folder,
        )
        log({"event": res["stage"], "arm": ARM, "axis": args.axis, "n_cases": res["n_cases"]})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
