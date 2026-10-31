"""Probe every case of one export with TRACE's own root-cause rule (method side) — the case-level ranking axis or the
per-trace axis.

    python -m tracebenchrca.ranktrace --arm rca/trace --probe-arms rca/trace --reading hop0 --phase post \
        --subtract none --pre-record none --axis trace --select post-nonok --cap 0 \
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

Since 2026-10-31 (`plans/rca-arms.md`, addendum "2026-10-31, later"; D-TBR-16 … 18): `--reading` (`hop1` the
landed upstream-closure reading; `hop0` the primary reading — the failed position's own service at hop 0; `source`
the author's source rule) selects the case-level rule and names the per-trace primary; the trace axis writes every
reading (`s`/`p`, `s0`/`p0`, `ss`/`ps`; schema `ranking-trace@2`). `--probe-arms` names the arms whose cells share
this arm's forward (one layout: the information-gain and CMI cells): the runner probes every trace once for all of
them and a later `ranktrace` of another of those arms, pointed at the same `--work-dir`, reuses the runner's files
when `summary.json` matches (cells, selection, cap, phase, seed, corpus) instead of probing again — the same path
re-ranks the landed arms' work files under the new readings (`--from-work-dir` of the plan), their sha256
recorded. `--phase pre` probes the pre-fault stretch (`cases.json` gains `pre_start_time`: the preceding case's
end in injection order, or the corpus start for the first case, `pre_source` recorded); `--subtract pre` (ranking
axis, phase post) ranks by the post credit rate minus the pre credit rate of the `--pre-record` run (the same arm,
reading and cell, phase pre), both counts and denominators recorded, `n_cases_no_pre` counted.

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
    AXIS_RANKING,
    AXIS_TRACE,
    INPUT_SERVICES,
    INPUT_VIEWS,
    INPUT_WINDOW,
    RANKING_TRACE_JSON,
    RESULTS_JSON,
    RUN_DIR,
    RUNGS,
    SEEDS,
    STAGE_RANK_TRACE,
    TRACE_ARMS_ALL,
    TRACE_CELLS,
    TRACE_DEVICES,
    TRACE_MAX_LEN,
    TRACE_NONOK_OUTCOMES,
    TRACE_PHASE_POST,
    TRACE_PHASE_PRE,
    TRACE_PHASES,
    TRACE_PRE_SOURCE_CORPUS_START,
    TRACE_PRE_SOURCE_PREVIOUS,
    TRACE_READING_HOP0,
    TRACE_READING_HOP1,
    TRACE_READING_SOURCE,
    TRACE_READINGS,
    TRACE_RULE_CLOSURE,
    TRACE_RULE_SOURCE,
    TRACE_SELECT_WINDOW,
    TRACE_SELECTS,
    TRACE_SUBTRACT_NONE,
    TRACE_SUBTRACT_PRE,
    TRACE_SUBTRACTS,
    VARIANT,
)
from .export import corpus_id_of
from .log import log
from .rank import POSITIONAL_JSON, STAGE, check_prior_arm, read_export_record
from .rankcausalrca import pinned_environment
from .record import RunRecord, read_json, sha256_file, write_json
from .staging import require_prior_rung
from .tracerule import rank_case, rank_trace, subtract_rates

RUNNER = Path(__file__).with_name("trace_runner.py")
TRACE_ARMS = TRACE_ARMS_ALL
AXES = (AXIS_RANKING, AXIS_TRACE)
STAGES = {AXIS_RANKING: STAGE, AXIS_TRACE: STAGE_RANK_TRACE}
RANKING_TRACE_SCHEMA = "ranking-trace@2"  # @1 + the source reading (`ss` / `ps`) and the `readings` list
READING_KEYS = {TRACE_READING_HOP1: ("s", "p"), TRACE_READING_HOP0: ("s0", "p0"), TRACE_READING_SOURCE: ("ss", "ps")}
RULE_OF_READING = {
    TRACE_READING_HOP1: (TRACE_RULE_CLOSURE, False),
    TRACE_READING_HOP0: (TRACE_RULE_CLOSURE, True),
    TRACE_READING_SOURCE: (TRACE_RULE_SOURCE, False),
}
PRE_RECORD_NONE = "none"
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
    python, harness_root, corpus_dir, cases_file, model_dir, freeze, cells, select, cap, device, seed, out_dir, phase
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
        "--cells",
        *cells,
        "--phase",
        phase,
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


def cell_traces(doc, cell):
    """The runner's traces of one document with this cell's cut edges (`edges_by_cell` since 2026-10-31; the landed
    documents carry one cell's `edges`)."""
    out = []
    for tr in doc["traces"]:
        by_cell = tr.get("edges_by_cell")
        if by_cell is not None:
            if cell not in by_cell:
                raise ValueError(f"{doc['case_dir']}: the runner's document holds no edges for cell {cell}")
            edges = by_cell[cell]
        else:
            if doc.get("cells", [doc.get("cell")]) not in ([cell], [None]):
                raise ValueError(f"{doc['case_dir']}: the runner's document is of another cell")
            edges = tr["edges"]
        out.append({**tr, "edges": edges})
    return out


def summary_matches(summary, cell, select, cap, phase, seed, corpus_id):
    """Whether a work directory's runner summary serves this read (the dedup of one layout's cells and the re-rank
    of the landed work files): the cell among its cells, the same selection, cap, phase, seed and corpus."""
    cells = summary.get("cells") or [summary.get("cell")]
    return (
        cell in cells
        and summary.get("select") == select
        and summary.get("cap") == cap
        and summary.get("phase", TRACE_PHASE_POST) == phase
        and summary.get("seed", seed) == seed
        and summary.get("corpus_id") == corpus_id
    )


def pre_starts(windows):
    """`{case_dir: (pre_start_time, pre_source)}`: the preceding case's end in injection order, or the corpus start
    (0 s) for the first case (D-TBR-18)."""
    out, prev_end = {}, None
    for w in sorted(windows, key=lambda w: (w["inject_time"], w["case_dir"])):
        if prev_end is None:
            out[w["case_dir"]] = (0, TRACE_PRE_SOURCE_CORPUS_START)
        else:
            out[w["case_dir"]] = (prev_end, TRACE_PRE_SOURCE_PREVIOUS)
        prev_end = w["end_time"]
    return out


def check_layout(freeze, cells):
    """Every cell of `--probe-arms` must share one pass in the freeze (probe, noise, c, N, g, context_mode)."""
    if len(cells) < 2:
        return None
    layouts = {}
    for key in cells:
        c = freeze.get("cells", {}).get(key)
        if c is None:
            raise ValueError(f"the freeze has no cell {key}")
        layouts[key] = (c["probe"], c["noise"], c["c"], c["N"], c["g"], c.get("context_mode"))
    if len(set(layouts.values())) != 1:
        raise ValueError(f"--probe-arms must share one layout; the freeze says {layouts}")
    return dict(zip(("probe", "noise", "c", "N", "g", "context_mode"), next(iter(layouts.values())), strict=True))


def read_pre_record(pre_record, arm, reading, cell, corpus_id, select, cap):
    """The completed phase-`pre` rank record this subtraction reads (same arm, reading, cell, corpus, selection)."""
    path = Path(pre_record) / RUN_DIR / RESULTS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{pre_record} holds no rank record ({RUN_DIR}/{RESULTS_JSON})")
    res = read_json(path)
    want = {
        "status": "ok",
        "stage": STAGE,
        "arm": arm,
        "reading": reading,
        "phase": TRACE_PHASE_PRE,
        "subtract": TRACE_SUBTRACT_NONE,
        "corpus_id": corpus_id,
        "select": select,
        "cap": cap,
    }
    got = {k: res.get(k) for k in want}
    if got != want:
        raise ValueError(f"{path} is not the pre-phase record this subtraction needs: {got} != {want}")
    if res["code"]["cell"] != cell:
        raise ValueError(f"{path} reads cell {res['code']['cell']}, not {cell}")
    return res, sha256_file(path)


def trace_axis_case(doc, names):
    """One case of the trace axis: the compact per-trace entries (indices into `names`) under every reading and the
    record row."""
    index = {s: i for i, s in enumerate(names)}
    entries, sizes = {}, {r: [] for r in TRACE_READINGS}
    n_nonok = n_with_nonok = n_with_upstream = n_client = n_sources = 0
    for tr in doc["traces"]:
        entry = {}
        for reading in TRACE_READINGS:
            rule, hop0 = RULE_OF_READING[reading]
            h = rank_trace(tr, names, TRACE_NONOK_OUTCOMES, hop0=hop0, rule=rule)
            s_key, p_key = READING_KEYS[reading]
            entry[s_key] = [index[s] for s in h["services"]]
            entry[p_key] = h["positions"]
            sizes[reading].append(len(h["services"]))
            if reading == TRACE_READING_HOP1:
                n_nonok += h["n_nonok"]
                n_with_nonok += h["n_nonok"] > 0
                n_with_upstream += bool(h["services"])
                n_client += h["n_client_dropped"]
            if reading == TRACE_READING_SOURCE:
                n_sources += h["n_sources"]
        entries[tr["trace_id"]] = entry
    row = {
        "case_dir": doc["case_dir"],
        "select": doc["select"],
        **{k: doc[k] for k in DOC_COUNTS},
        "n_nonok_positions": n_nonok,
        "n_traces_with_nonok": n_with_nonok,
        "n_traces_with_upstream": n_with_upstream,
        "n_client_positions_dropped": n_client,
        "n_sources": n_sources,
        "mean_set_size_hop1": _mean(sizes[TRACE_READING_HOP1]),
        "mean_set_size_hop0": _mean(sizes[TRACE_READING_HOP0]),
        "mean_set_size_source": _mean(sizes[TRACE_READING_SOURCE]),
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
    reading,
    phase,
    subtract,
    pre_record,
    probe_arms,
):
    if arm not in TRACE_ARMS:
        raise ValueError(f"arm {arm!r} is not a TRACE root-cause arm; these are {TRACE_ARMS}")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    if select not in TRACE_SELECTS:
        raise ValueError(f"select must be one of {TRACE_SELECTS}, got {select!r}")
    if cap < 0:
        raise ValueError("cap must be >= 0 (0 = every selected trace)")
    if reading not in TRACE_READINGS:
        raise ValueError(f"reading must be one of {TRACE_READINGS}, got {reading!r}")
    if phase not in TRACE_PHASES:
        raise ValueError(f"phase must be one of {TRACE_PHASES}, got {phase!r}")
    if subtract not in TRACE_SUBTRACTS:
        raise ValueError(f"subtract must be one of {TRACE_SUBTRACTS}, got {subtract!r}")
    probe_arms = list(probe_arms)
    if arm not in probe_arms or any(a not in TRACE_ARMS for a in probe_arms):
        raise ValueError(f"--probe-arms must name {arm!r} and TRACE arms only; got {probe_arms}")
    if phase == TRACE_PHASE_PRE and (select != TRACE_SELECT_WINDOW or axis != AXIS_RANKING):
        raise ValueError("phase pre is the case-level selection (--axis ranking --select window; D-TBR-18)")
    if subtract == TRACE_SUBTRACT_PRE and (axis != AXIS_RANKING or phase != TRACE_PHASE_POST):
        raise ValueError("--subtract pre applies to the ranking axis of the post phase (D-TBR-18)")
    if (subtract == TRACE_SUBTRACT_PRE) != (pre_record != PRE_RECORD_NONE):
        raise ValueError("--pre-record names the pre-phase record exactly when --subtract pre")
    stage = STAGES[axis]
    cell = TRACE_CELLS[arm]
    cells = [TRACE_CELLS[a] for a in probe_arms]
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
    pre = pre_starts(windows)
    for w in windows:
        w["pre_start_time"], w["pre_source"] = pre[w["case_dir"]]
    services = {d.name: [s for s, _ in read_services(d)] for d in dirs}
    layout = check_layout(read_json(freeze), cells)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cases_file = work / CASES_JSON
    t0 = time.time()
    summary_path = work / SUMMARY_JSON
    reused = summary_path.exists() and summary_matches(
        read_json(summary_path), cell, select, cap, phase, seed, corpus_id
    )
    if reused:
        summary, runner_wall = read_json(summary_path), 0.0
        paths = runner_paths(harness_root)
        work_files = {
            SUMMARY_JSON: sha256_file(summary_path),
            **{f"{d.name}.json": sha256_file(work / f"{d.name}.json") for d in dirs},
        }
        log({"event": "runner_reused", "arm": arm, "work_dir": str(work), "cells": summary.get("cells")})
    else:
        write_json(cases_file, windows)
        summary, paths, runner_wall = run_runner(
            python,
            harness_root,
            corpus_dir,
            cases_file,
            model_dir,
            freeze,
            cells,
            select,
            cap,
            device,
            seed,
            work,
            phase,
        )
        work_files = None
    if not summary_matches(summary, cell, select, cap, phase, seed, corpus_id):
        raise ValueError(
            f"the runner's summary (cells {summary.get('cells') or [summary.get('cell')]}, select "
            f"{summary.get('select')!r}, cap {summary.get('cap')!r}, phase {summary.get('phase', TRACE_PHASE_POST)!r}) "
            f"does not serve {cell} / {select} / {cap} / {phase} on {corpus_id!r}"
        )
    docs = {d.name: read_json(work / f"{d.name}.json") for d in dirs}
    for name, doc in docs.items():
        if doc.get("select") != select or doc.get("phase", TRACE_PHASE_POST) != phase:
            raise ValueError(
                f"{name}: the runner's document says select={doc.get('select')!r} phase="
                f"{doc.get('phase', TRACE_PHASE_POST)!r}, not {select!r} / {phase!r}"
            )
    docs = {name: {**doc, "traces": cell_traces(doc, cell)} for name, doc in docs.items()}
    out = Path(out)
    common = {
        "stage": stage,
        "axis": axis,
        "arm": arm,
        "reading": reading,
        "phase": phase,
        "subtract": subtract,
        "probe_arms": probe_arms,
        "layout": layout,
        "code": {**code_of(arm), "cell": cell, "cells": cells, "runner": RUNNER.name},
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
        "pre_starts": {
            w["case_dir"]: {"pre_start_time": w["pre_start_time"], "pre_source": w["pre_source"]} for w in windows
        },
        "runner": summary,
        "runner_reused": bool(reused),
        "work_files_sha256": work_files,
        "pinned_environment": {"python": str(python), "pythonpath": paths, "pip_freeze": pinned_environment(python)},
    }
    if axis == AXIS_RANKING:
        pre_res = None
        if subtract == TRACE_SUBTRACT_PRE:
            pre_res, pre_sha = read_pre_record(pre_record, arm, reading, cell, corpus_id, select, cap)
            common["pre_record"] = {"path": str(pre_record), "sha256": pre_sha}
        return _rank_ranking_axis(arm, dirs, docs, services, common, out, runner_wall, t0, reading, pre_res)
    return _rank_trace_axis(arm, dirs, docs, services, common, summary, out, runner_wall, t0)


def _rank_ranking_axis(arm, dirs, docs, services, common, out, runner_wall, t0, reading, pre_res=None):
    rule, hop0 = RULE_OF_READING[reading]
    pre_rows = {c["case_dir"]: c for c in pre_res["cases"]} if pre_res is not None else None
    cases, positional, n_no_pre = [], {}, 0
    for d in dirs:
        doc = docs[d.name]
        row = {
            "case_dir": d.name,
            **{k: doc[k] for k in RANKING_DOC_COUNTS},
            **rank_case(doc["traces"], services[d.name], TRACE_NONOK_OUTCOMES, hop0=hop0, rule=rule),
            "n_filled": 0,
        }
        if pre_rows is not None:  # the pre-fault subtraction (D-TBR-18)
            pre_row = pre_rows.get(d.name)
            sub = subtract_rates(row, pre_row, services[d.name])
            n_no_pre += sub["no_pre"]
            row["subtraction"] = sub
            row["services"] = sub["services"]
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
        "n_cases_no_pre": n_no_pre if pre_rows is not None else None,
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
        "readings": list(TRACE_READINGS),
        "primary_reading": common["reading"],
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
        "--probe-arms",
        required=True,
        nargs="+",
        choices=TRACE_ARMS,
        help="the arms whose cells share this arm's forward (one layout; must include --arm): probed once (D-TBR-16)",
    )
    p.add_argument(
        "--reading",
        required=True,
        choices=TRACE_READINGS,
        help="hop1: the landed upstream-closure reading; hop0: the primary reading; source: the author's source rule",
    )
    p.add_argument(
        "--phase", required=True, choices=TRACE_PHASES, help="post: the fault window; pre: the pre-fault stretch"
    )
    p.add_argument(
        "--subtract",
        required=True,
        choices=TRACE_SUBTRACTS,
        help="pre: rank by the post credit rate minus the pre credit rate of --pre-record (ranking axis, phase post)",
    )
    p.add_argument(
        "--pre-record",
        required=True,
        help="the pre-phase rank record of the same arm and reading (with --subtract pre), else 'none'",
    )
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
            reading=args.reading,
            phase=args.phase,
            subtract=args.subtract,
            pre_record=args.pre_record,
            probe_arms=args.probe_arms,
        )
        log(
            {
                "event": res["stage"],
                "arm": args.arm,
                "axis": args.axis,
                "reading": args.reading,
                "phase": args.phase,
                "subtract": args.subtract,
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
