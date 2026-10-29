"""Score one TRACE arm's per-trace rankings of one corpus on the per-trace axis, with every row of the axis (score side).

    python -m tracebenchrca.scoretrace --corpus-dir <pulled corpus, tiers rca + trace> \
        --rank-record <the arm's rank-trace folder> --ranking-trace <its ranking-trace.json> \
        --broadcast-results-dir <results root holding the landed case rankings> \
        --bootstrap-b 2000 --bootstrap-seed 0 --output-folder <the arm's trace folder>

`plans/rca-arms.md`, addendum 2026-10-31 (D-TBR-14 / D-TBR-15); scorer `tracebench.score_rca_trace`
(`docs/scorers-service-rca.md` §4 in the benchmark). One call scores, on the scored set of the rank read's
selection (the traces in which the fault took), every row of the axis for this corpus:

- the arm itself under both readings — `<arm>/hop1` (the author's rule, hop >= 1) and `<arm>/hop0` — from
  the rank read's per-trace reached services and positions (mode `arm`);
- `floor/trace-topology/hop1` and `/hop0` — the rule on the trace's true call tree (mode `topology`);
- `control/trace-random` — the analytic random expectation over the trace's services (mode `random`);
- `control/trace-callees/hop1` and `/hop0` — the analytic expectation over the failed spans' descendant
  services, the axis's predict-all audit (mode `callees`);
- `broadcast/<arm>` for every landed case-level ranking (`constants.TRACE_BROADCAST_ARMS`; `rca/causalrca`
  once per series): the case ranking filtered to the trace's own services (mode `broadcast`).

Writes `score-trace.json` (`{"rows": {row_id: result}}`, every result without its per-trace rows) and the
gitignored `score-trace-traces.json` (`{row_id: [per-trace rows]}`, hashed in the record), and the stage's run
record. The floor, control and broadcast rows depend on the corpus and the selection only, so the two TRACE arms'
cells carry identical copies; `reporttrace` reads them once and checks the copies agree.

This module reads labels; it runs after the method process has exited and never hands anything back to a method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.export_rca import case_dirs
from tracebench.score_rca import load_cases
from tracebench.score_rca_trace import load_inputs, services_sorted
from tracebench.score_rca_trace import score as score_trace_axis

from .arms import arm_slug
from .constants import (
    ARM_CAUSALRCA,
    ARM_CONTROL_TRACE_CALLEES,
    ARM_CONTROL_TRACE_RANDOM,
    ARM_FLOOR_TRACE_TOPOLOGY,
    ARM_TRACE,
    ARM_TRACE_CORE,
    CAUSALRCA_SERIES,
    RANKING_JSON,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_TRACE_JSON,
    SCORE_TRACE_TRACES_JSON,
    STAGE_RANK_TRACE,
    STAGE_SCORE_TRACE,
    TRACE_BROADCAST_ARMS,
    TRACE_READINGS,
    VARIANT,
)
from .export import corpus_id_of
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json

STAGE = STAGE_SCORE_TRACE
HOP1, HOP0 = TRACE_READINGS
READING_KEYS = {HOP1: ("s", "p"), HOP0: ("s0", "p0")}
TRACE_ARMS = (ARM_TRACE, ARM_TRACE_CORE)


def read_rank_trace_record(rank_record, ranking_trace):
    """The completed `rank-trace` record and its ranking file (named explicitly: the file is not tracked and may
    sit elsewhere than beside the record); the file must hash to what the record bound."""
    rank_record = Path(rank_record)
    path = rank_record / RUN_DIR / RESULTS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{rank_record} holds no rank record ({RUN_DIR}/{RESULTS_JSON})")
    res = read_json(path)
    if res.get("status") != "ok" or res.get("stage") != STAGE_RANK_TRACE:
        raise ValueError(f"{path} is not a completed {STAGE_RANK_TRACE} record")
    ranking_trace = Path(ranking_trace)
    if sha256_file(ranking_trace) != res["ranking_sha256"]:
        raise ValueError(f"{ranking_trace} differs from the ranking its record hashed")
    doc = read_json(ranking_trace)
    if doc.get("arm") != res["arm"]:
        raise ValueError(f"{ranking_trace} is an {doc.get('arm')!r} file; the record is {res['arm']!r}")
    return res, sha256_file(path), doc


def cell_of(doc):
    """The selection header the scorer checks against its own recomputation of rule R."""
    return {
        "c": int(doc["c"]),
        "max_len": int(doc["max_len"]),
        "select": doc["select"],
        "cap": int(doc["cap"]),
        "selected": {d: list(v["selected"]) for d, v in doc["cases"].items()},
        "skipped": {d: list(v["skipped"]) for d, v in doc["cases"].items()},
    }


def arm_prediction(doc, reading):
    s_key, p_key = READING_KEYS[reading]
    return {
        "services": list(doc["services"]),
        "cases": {
            d: {tid: {"s": list(t[s_key]), "p": list(t[p_key])} for tid, t in v["traces"].items()}
            for d, v in doc["cases"].items()
        },
    }


def broadcast_sources(results_dir, rung, seed):
    """`[(row_id, ranking.json path)]` of the landed case rankings this corpus's broadcast rows read."""
    root = Path(results_dir) / rung / VARIANT / f"seed={seed}"
    out = []
    for arm in TRACE_BROADCAST_ARMS:
        if arm == ARM_CAUSALRCA:
            for series in CAUSALRCA_SERIES:
                out.append((f"broadcast/{arm}/{series}", root / arm_slug(arm) / series / RANKING_JSON))
        else:
            out.append((f"broadcast/{arm}", root / arm_slug(arm) / RANKING_JSON))
    missing = [str(p) for _, p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"landed case rankings missing for the broadcast rows: {missing}")
    return out


def rows_of(corpus_dir, arm, doc, broadcast_results_dir, rung, seed, b, boot_seed):
    """Every row of the axis for this cell: `{row_id: result}` with the per-trace rows still inside."""
    cell = cell_of(doc)
    inputs = load_inputs(corpus_dir, cell)  # the view and the labels read once per cell

    def one(prediction, mode, hop0):
        return score_trace_axis(corpus_dir, cell, prediction, mode, hop0, b, boot_seed, inputs=inputs)

    rows = {}
    for reading in TRACE_READINGS:
        rows[f"{arm}/{reading}"] = one(arm_prediction(doc, reading), "arm", reading == HOP0)
    for reading in TRACE_READINGS:
        rows[f"{ARM_FLOOR_TRACE_TOPOLOGY}/{reading}"] = one(None, "topology", reading == HOP0)
    rows[ARM_CONTROL_TRACE_RANDOM] = one(None, "random", False)
    for reading in TRACE_READINGS:
        rows[f"{ARM_CONTROL_TRACE_CALLEES}/{reading}"] = one(None, "callees", reading == HOP0)
    for row_id, path in broadcast_sources(broadcast_results_dir, rung, seed):
        rows[row_id] = one(read_json(path), "broadcast", False)
        rows[row_id]["source_ranking_sha256"] = sha256_file(path)
    return rows


def check_case_dirs(corpus_dir, doc):
    """The file's case directories must be exactly the labelled cases' positional names (D-TBR-3)."""
    names = {name for name, _ in case_dirs(load_cases(corpus_dir))}
    if set(doc["cases"]) != names:
        raise ValueError(
            f"the rank read's case directories {sorted(doc['cases'])} are not the labelled cases' {sorted(names)}"
        )


def score(corpus_dir, rank_record, ranking_trace, broadcast_results_dir, b, boot_seed, out):
    rank_res, rank_sha, doc = read_rank_trace_record(rank_record, ranking_trace)
    arm = rank_res["arm"]
    if arm not in TRACE_ARMS:
        raise ValueError(f"{arm!r} has no per-trace axis; the TRACE arms are {TRACE_ARMS}")
    corpus_id = corpus_id_of(corpus_dir)
    if corpus_id != rank_res["corpus_id"]:
        raise ValueError(f"the rank read is of {rank_res['corpus_id']!r}; this corpus is {corpus_id!r}")
    expected = services_sorted(corpus_dir)
    if list(doc["services"]) != expected:
        raise ValueError("the rank read's services are not the corpus's deployed services in name order")
    check_case_dirs(corpus_dir, doc)
    rows = rows_of(corpus_dir, arm, doc, broadcast_results_dir, rank_res["rung"], rank_res["seed"], b, boot_seed)
    traces = {row_id: r.pop("traces") for row_id, r in rows.items()}
    out = Path(out)
    write_json(out / SCORE_TRACE_JSON, {"rows": rows})
    write_json(out / SCORE_TRACE_TRACES_JSON, traces)
    primary = rows[f"{arm}/{HOP1}"]
    return {
        "stage": STAGE,
        "axis": primary["axis"],
        "arm": arm,
        "rung": rank_res["rung"],
        "seed": rank_res["seed"],
        "corpus_id": corpus_id,
        "rank_record_sha256": rank_sha,
        "ranking_trace_sha256": rank_res["ranking_sha256"],
        "score_trace_sha256": sha256_file(out / SCORE_TRACE_JSON),
        "score_trace_traces_file": SCORE_TRACE_TRACES_JSON,
        "score_trace_traces_sha256": sha256_file(out / SCORE_TRACE_TRACES_JSON),
        "select": doc["select"],
        "cap": int(doc["cap"]),
        "c": int(doc["c"]),
        "max_len": int(doc["max_len"]),
        "device": doc.get("device"),
        "rows": sorted(rows),
        "metrics": {row_id: r["metrics"] for row_id, r in rows.items()},
        "random": primary["random"],
        "counts": primary["counts"],
        "n_cases": primary["metrics"]["n_cases"],
        "n_traces": primary["metrics"]["n_traces"],
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True, help="the pulled corpus (tiers rca and trace: the labels parquet)")
    p.add_argument("--rank-record", required=True, help="the arm's rank-trace stage folder (its run/ record)")
    p.add_argument(
        "--ranking-trace", required=True, help="the arm's ranking-trace.json (not tracked; hashed by the record)"
    )
    p.add_argument(
        "--broadcast-results-dir",
        required=True,
        help="the results root holding the landed case rankings (ranking.json)",
    )
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples over cases (registered: 2000)")
    p.add_argument("--bootstrap-seed", required=True, type=int, help="bootstrap seed (registered: 0)")
    p.add_argument("--output-folder", required=True, help="the arm's trace folder: score-trace.json, the sidecar, run/")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = score(
            args.corpus_dir,
            args.rank_record,
            args.ranking_trace,
            args.broadcast_results_dir,
            args.bootstrap_b,
            args.bootstrap_seed,
            args.output_folder,
        )
        log(
            {
                "event": STAGE,
                "arm": res["arm"],
                "corpus_id": res["corpus_id"],
                "n_traces": res["n_traces"],
                **{k: v.get("s@1") for k, v in res["metrics"].items() if k.startswith(res["arm"])},
            }
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
