"""The per-trace axis's score side (contract-per-trace-axis.md; plan of 2026-10-31): `scoretrace` scores every row of
the axis for one TRACE arm's cell end to end on a hand-written corpus with known answers — the arm under both readings,
the call-tree floor, the two analytic controls and every landed case ranking broadcast to the traces — binds the two
files to the record, and refuses a ranking that does not hash to its record, a non-TRACE arm, a reordered services
list, a missing landed ranking and a selection the benchmark cannot reproduce; `reporttrace` pools the cells per rung,
reads the shared rows once and checks the copies agree across the TRACE arms' cells, and scores the four hypotheses
mechanically (checked by hand on the pooled numbers and on hand-built summary rows)."""

import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from tracebench.constants import (
    KIND_BFF,
    KIND_CLIENT,
    KIND_SERVICE,
    OUTCOME_4XX,
    OUTCOME_5XX,
    OUTCOME_ERR,
    OUTCOME_OK,
    OUTCOME_SLOW,
    PHASE_NONE,
    PHASE_POST,
    PHASE_PRE,
    PHASE_STRADDLE,
)
from tracebench.correlate.views import ARROW_SCHEMA, LABELS_SCHEMA
from tracebench.score_rca_trace import case_windows_ns, load_ops, select_traces, services_sorted
from tracebench.score_rca_trace import score as score_trace_axis

from tracebenchrca import reporttrace, scoretrace
from tracebenchrca.arms import arm_slug
from tracebenchrca.constants import (
    ARM_BARO,
    ARM_CAUSALRCA,
    ARM_CONTROL_TRACE_CALLEES,
    ARM_CONTROL_TRACE_RANDOM,
    ARM_FLOOR_TRACE_TOPOLOGY,
    ARM_RCD,
    ARM_TRACE,
    ARM_TRACE_CORE,
    CAUSALRCA_SERIES,
    RANKING_JSON,
    RANKING_TRACE_JSON,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_TRACE_JSON,
    SCORE_TRACE_TRACES_JSON,
    STAGE_RANK_TRACE,
    STAGE_SCORE_TRACE,
    TRACE_AXIS_DIR,
    TRACE_BROADCAST_ARMS,
    TRACE_CELLS,
    TRACE_READINGS,
    TRACE_SELECT_POST_NONOK,
)
from tracebenchrca.record import read_json, sha256_file, write_json

# --- the benchmark test's hand corpus (tests/test_score_rca_trace.py §8, copied so this file is self-contained) ------

EPOCH = 1767603600  # 2026-01-05T09:00:00Z
MS = 1_000_000
SERVICES = ["svc-a", "svc-b", "svc-c"]  # name order = index order of an arm's "s" lists
CASES = [
    {
        "case_id": "hand_svc-c_crash_0",
        "system": "hand",
        "root_cause_component": "svc-c",
        "root_cause_endpoints": ["/c"],
        "fault_kind": "crash",
        "inject_time_s": 600.0,
        "end_time_s": 900.0,
    },
    {
        "case_id": "hand_svc-b_degrade_1",
        "system": "hand",
        "root_cause_component": "svc-b",
        "root_cause_endpoints": ["/b1"],
        "fault_kind": "degrade",
        "inject_time_s": 1200.0,
        "end_time_s": 1500.0,
    },
]
CASE_IDS = [c["case_id"] for c in CASES]
BASE_OPS = [
    {"id": 0, "service": "svc-a", "name": "/a", "kind": KIND_BFF},
    {"id": 1, "service": "svc-b", "name": "/b1", "kind": KIND_SERVICE},
    {"id": 2, "service": "svc-b", "name": "/b2", "kind": KIND_SERVICE},
    {"id": 3, "service": "svc-c", "name": "/c", "kind": KIND_SERVICE},
    {"id": 4, "service": "web", "name": "/page", "kind": KIND_CLIENT},
]
RC_OP = {0: 3, 1: 1}  # fault index -> root-cause op
OK, X4, X5, ERR, SLOW = OUTCOME_OK, OUTCOME_4XX, OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW
# (trace_id, start_s, split, ops, outcomes, parent_pos, offsets_ms, durations_ms); end-ordered (callee first)
TRACES = [
    ("ta", 610, "train", [1, 3, 0, 4], [OK, X5, OK, OK], [2, 2, 3, -1], [2, 3, 1, 0], [10, 20, 90, 100]),
    ("tb", 620, "val", [3, 2, 0, 4], [X4, X5, OK, OK], [1, 2, 3, -1], [3, 2, 1, 0], [20, 50, 90, 100]),  # rc 4xx only
    (
        "tc",
        630,
        "train",
        [3, 1, 0, 4],
        [ERR, ERR, OK, OK],
        [1, 2, 3, -1],
        [3, 2, 1, 0],
        [20, 50, 90, 100],
    ),  # rc in context
    ("td", 640, "val", [3, 4], [X5, OK], [1, -1], [1, 0], [10, 20]),  # skipped by the method
    (
        "te",
        650,
        "train",
        [1, 2, 0, 0, 1, 3, 4],
        [OK, OK, OK, OK, X5, X5, OK],
        [2, 2, 6, 6, 3, 4, -1],
        [2, 2, 1, 1, 2, 3, 0],
        [5, 5, 40, 40, 20, 10, 100],
    ),  # inverted link 5 -> 4; rc past max_len 5
    ("tf", 1210, "test", [3, 1, 0, 4], [OK, SLOW, SLOW, OK], [1, 2, 3, -1], [3, 2, 1, 0], [20, 50, 90, 100]),  # case 1
    ("tg", 1220, "test", [2, 0, 4], [X5, X5, OK], [1, 2, -1], [2, 1, 0], [10, 50, 60]),  # /b2 is not an rc endpoint
    ("th", 1100, "train", [3, 0, 4], [X5, X5, OK], [1, 2, -1], [2, 1, 0], [10, 50, 60]),  # outside both windows
    ("ti", 700, "train", [3, 0, 4], [OK, OK, OK], [1, 2, -1], [2, 1, 0], [10, 50, 60]),  # POST but healthy
    ("tj", 1490, "test", [1, 0, 4], [X5, X5, OK], [1, 2, -1], [1, 1, 0], [100, 19000, 20000]),  # straddles the end
    ("tk", 660, "val", [3, 4], [OK, ERR], [1, -1], [1, 0], [10, 20]),  # only the client span failed
]
C, MAX_LEN = 2, 5
SELECTED = {"case-00": ["ta", "tb", "tc", "td", "te"], "case-01": ["tf", "tg"]}  # rule R, cap 0
SKIPPED = {"case-00": ["td"], "case-01": []}
# the scored set C = {ta, tc, td, te} (case-00, root cause svc-c) and {tf} (case-01, root cause svc-b); tb's root-cause
# span is 4xx only and tg carries no root-cause position. Every trace in C holds the root-cause service; td's only
# non-client service is svc-c.
TRACE_HOP1 = {
    "case-00": {"ta": {"s": [2, 1], "p": [1, 0]}, "tc": {"s": [2], "p": [0]}, "te": {"s": [1], "p": [3]}},
    "case-01": {"tf": {"s": [2, 1, 0], "p": [0, 1]}},
}
TRACE_HOP0 = {
    "case-00": {"ta": {"s": [2], "p": [0]}, "tc": {"s": [1, 2], "p": [1, 0]}, "te": {"s": [2, 1], "p": [4, 3]}},
    "case-01": {"tf": {"s": [1], "p": [0]}},
}
# hit@1 of `rca/trace/hop1` by hand: case-00 ta svc-c first (1), tc svc-c (1), td skipped (0), te svc-b (0) -> 0.5;
# case-01 tf svc-c first, root cause svc-b (0) -> 0.0; the mean over cases 0.25. hop0: ta 1, tc svc-b first 0, td 0,
# te svc-c first 1 -> 0.5; tf svc-b 1 -> 1.0; mean 0.75.
TRACE_S1 = {"hop1": 0.25, "hop0": 0.75}
# the second TRACE arm names svc-a first everywhere: hit@1 0 at both readings
CORE_HOP1 = {
    "case-00": {"ta": {"s": [0], "p": [2]}, "tc": {"s": [0], "p": [1]}, "te": {"s": [0], "p": [2]}},
    "case-01": {"tf": {"s": [0], "p": [2]}},
}
CORE_HOP0 = CORE_HOP1
# case-level rankings broadcast to the traces (`filter_broadcast`: the order kept, absent services appended);
# hit@1 by hand over C — the four traces of case-00 then tf:
BROADCAST = {
    "cases": {CASE_IDS[0]: ["svc-b", "svc-c"], CASE_IDS[1]: ["svc-a", "svc-c", "svc-b"]}
}  # 0,0,1,0 | 0 -> 0.125
PERFECT = {"cases": {CASE_IDS[0]: ["svc-c", "svc-b", "svc-a"], CASE_IDS[1]: ["svc-b", "svc-a", "svc-c"]}}  # -> 1.0
SIDEWAYS = {
    "cases": {CASE_IDS[0]: ["svc-a", "svc-c", "svc-b"], CASE_IDS[1]: ["svc-b", "svc-a", "svc-c"]}
}  # 0,0,1,0 | 1 -> 0.625
RUN_AVERAGED = {"cases": {CASE_IDS[0]: [["svc-c", "svc-b"], None], CASE_IDS[1]: [["svc-b"], None]}}  # every trace 0.5
BROADCAST_S1 = {
    f"broadcast/{ARM_BARO}": 1.0,
    f"broadcast/{ARM_RCD}": 0.5,
    f"broadcast/{ARM_CAUSALRCA}/latency-50": 0.125,
    f"broadcast/{ARM_CAUSALRCA}/latency-95": 0.625,
}
BROADCAST_ROWS = [
    f"broadcast/{arm}/{series}" if arm == ARM_CAUSALRCA else f"broadcast/{arm}"
    for arm in TRACE_BROADCAST_ARMS
    for series in (CAUSALRCA_SERIES if arm == ARM_CAUSALRCA else (None,))
]
LANDED_READINGS = TRACE_READINGS[:2]  # hop1, hop0: the readings of a schema-@1 ranking file (the landed cells)
SHARED_ROWS = [f"{ARM_FLOOR_TRACE_TOPOLOGY}/{r}" for r in LANDED_READINGS] + [ARM_CONTROL_TRACE_RANDOM]
SHARED_ROWS += [f"{ARM_CONTROL_TRACE_CALLEES}/{r}" for r in LANDED_READINGS] + BROADCAST_ROWS
SHARED_S1 = {  # the benchmark test's hand answers for the floor and the controls
    f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop1": 0.25,
    f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop0": 1.0,
    ARM_CONTROL_TRACE_RANDOM: pytest.approx(5 / 12),
    f"{ARM_CONTROL_TRACE_CALLEES}/hop1": 0.5,
    f"{ARM_CONTROL_TRACE_CALLEES}/hop0": pytest.approx(13 / 24),
    **{row: BROADCAST_S1.get(row, 0.125) for row in BROADCAST_ROWS},
}
CORPUS_ID = "xs/latent/seed=0"
RUNG, SEED = "xs", 0


def _phase(start_s, end_s, lo, hi):
    if end_s < lo:
        return PHASE_PRE
    if start_s >= hi:
        return PHASE_NONE
    if start_s >= lo and end_s < hi:
        return PHASE_POST
    return PHASE_STRADDLE


def _view_row(tid, start_s, ops, outcomes, parent, offs_ms, durs_ms):
    return {
        "trace_id": tid,
        "ops": ops,
        "outcomes": outcomes,
        "offsets": [o * MS for o in offs_ms],
        "durations": [d * MS for d in durs_ms],
        "parent_pos": parent,
        "healthy": all(o == OK for o in outcomes),
        "n_err_spans": sum(1 for o in outcomes if o != OK),
        "n_spans": len(ops),
        "truncated": False,
        "scenario_hash": "h",
        "scenario_sid": -1,
        "start_ns": (EPOCH + start_s) * 10**9,
        "attribution_level": 0,
        "sequence_kind": "request",
    }


def hand_corpus(d):
    """The benchmark test's hand corpus written at `d`: config, instantiation, cases, vocabulary, one parquet part per
    split of the end-ordered request view and the request-grain labels (plus never-read session-grain rows)."""
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(
        yaml.safe_dump({"run": {"window": {"start": "2026-01-05T09:00:00Z", "duration_hours": 1.0}}})
    )
    kinds = [("svc-a", KIND_BFF), ("svc-b", KIND_SERVICE), ("svc-c", KIND_SERVICE), ("web", KIND_CLIENT)]
    services = [{"index": i, "name": n, "kind": k} for i, (n, k) in enumerate(kinds)]
    write_json(d / "instantiation.json", {"topology": {"services": services}})
    write_json(d / "labels" / "cases.json", {"cases": CASES})
    write_json(d / "views" / "end-request" / "model-vocab.json", {"base_ops": BASE_OPS})
    by_split, labels = {}, []
    for tid, start_s, split, ops, outcomes, parent, offs, durs in TRACES:
        by_split.setdefault(split, []).append(_view_row(tid, start_s, ops, outcomes, parent, offs, durs))
        end_s = start_s + max(o + du for o, du in zip(offs, durs, strict=True)) / 1000.0
        for fi, case in enumerate(CASES):
            rc = [i for i, op in enumerate(ops) if op == RC_OP[fi]]
            labels.append(
                {
                    "trace_id": tid,
                    "grain": "request",
                    "fault_index": fi,
                    "phase": _phase(start_s, end_s, case["inject_time_s"], case["end_time_s"]),
                    "rc_positions_end": rc,
                    "rc_positions_start": rc,
                    "rc_took": any(outcomes[i] != OK for i in rc),
                }
            )
        labels.append(
            {
                "trace_id": tid,
                "grain": "session",
                "fault_index": 0,
                "phase": PHASE_NONE,
                "rc_positions_end": [],
                "rc_positions_start": [],
                "rc_took": False,
            }
        )
    for split, rows in by_split.items():
        part = d / "views" / "end-request" / "sequences" / f"split={split}" / "date=2026-01-05" / "part-0000.parquet"
        part.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA), part)
    pq.write_table(pa.Table.from_pylist(labels, schema=LABELS_SCHEMA), d / "labels" / "sequence-labels.parquet")
    return d


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The hand corpus under the dataset host's layout, so `corpus_id_of` reads `xs/latent/seed=0`."""
    return hand_corpus(tmp_path_factory.mktemp("hand") / "instances" / RUNG / "latent" / f"seed={SEED}")


# --- the harness-side inputs: a rank-trace record with its ranking file, the landed case rankings ---------------------


def _ranking_trace(arm, hop1, hop0, selected=SELECTED, skipped=SKIPPED, services=SERVICES):
    """A `ranking-trace.json` in the contract's shape (reached-only indices into `services`, both readings)."""
    cases = {}
    for case_dir in selected:
        traces = {
            tid: {**hop1[case_dir][tid], "s0": hop0[case_dir][tid]["s"], "p0": hop0[case_dir][tid]["p"]}
            for tid in hop1.get(case_dir, {})
        }
        cases[case_dir] = {
            "selected": list(selected[case_dir]),
            "skipped": list(skipped.get(case_dir, [])),
            "traces": traces,
        }
    return {
        "schema": "ranking-trace@1",
        "arm": arm,
        "cell": TRACE_CELLS.get(arm, "none"),
        "c": C,
        "max_len": MAX_LEN,
        "select": TRACE_SELECT_POST_NONOK,
        "cap": 0,
        "device": "cpu",
        "services": list(services),
        "cases": cases,
    }


def _rank_record(folder, doc, **overrides):
    """A completed `rank-trace` record binding `ranking-trace.json` by its hash; returns the ranking file's path."""
    ranking = folder / RANKING_TRACE_JSON
    write_json(ranking, doc)
    rec = {
        "status": "ok",
        "stage": STAGE_RANK_TRACE,
        "arm": doc["arm"],
        "rung": RUNG,
        "seed": SEED,
        "corpus_id": CORPUS_ID,
        "ranking_sha256": sha256_file(ranking),
        **overrides,
    }
    write_json(folder / RUN_DIR / RESULTS_JSON, rec)
    return ranking


def _broadcast_prediction(row_id):
    if row_id == f"broadcast/{ARM_BARO}":
        return PERFECT
    if row_id == f"broadcast/{ARM_RCD}":
        return RUN_AVERAGED
    if row_id == f"broadcast/{ARM_CAUSALRCA}/latency-95":
        return SIDEWAYS
    return BROADCAST


def _broadcast_root(root, skip=()):
    """The landed case rankings of every broadcast arm: `<root>/xs/latent/seed=0/<slug>[/<series>]/ranking.json`."""
    cell = root / RUNG / "latent" / f"seed={SEED}"
    for row_id in BROADCAST_ROWS:
        if row_id in skip:
            continue
        parts = row_id.split("/")[1:]  # ["rca", "causalrca", "latency-50"] or ["rca", "baro"]
        arm = "/".join(parts[:2])
        path = cell / arm_slug(arm)
        if arm == ARM_CAUSALRCA:
            path = path / parts[2]
        write_json(path / RANKING_JSON, _broadcast_prediction(row_id))
    return root


def _score_argv(corpus, rank_folder, ranking, broadcast, out, b=50):
    return [
        "--corpus-dir",
        str(corpus),
        "--rank-record",
        str(rank_folder),
        "--ranking-trace",
        str(ranking),
        "--broadcast-results-dir",
        str(broadcast),
        "--rows",
        "all",
        "--bootstrap-b",
        str(b),
        "--bootstrap-seed",
        "0",
        "--output-folder",
        str(out),
    ]


# --- 1. scoretrace end to end -------------------------------------------------------------------------------------


def test_scoretrace_end_to_end_on_a_hand_corpus(corpus, tmp_path):
    assert services_sorted(corpus) == SERVICES
    ops, client_ops = load_ops(corpus)
    windows = case_windows_ns(corpus)
    sel = select_traces(corpus, windows, client_ops, TRACE_SELECT_POST_NONOK, 0)
    assert {cd: [r["trace_id"] for r in rows] for cd, rows in sel.items()} == SELECTED  # the runner's list = rule R
    doc = _ranking_trace(ARM_TRACE, TRACE_HOP1, TRACE_HOP0)
    rank_folder = tmp_path / "rank-trace"
    ranking = _rank_record(rank_folder, doc)
    broadcast = _broadcast_root(tmp_path / "results")
    out = tmp_path / "trace"
    assert scoretrace.main(_score_argv(corpus, rank_folder, ranking, broadcast, out)) == 0

    score = read_json(out / SCORE_TRACE_JSON)
    rows = score["rows"]
    expected_rows = [f"{ARM_TRACE}/{r}" for r in LANDED_READINGS] + SHARED_ROWS
    assert set(score) == {"rows"} and sorted(rows) == sorted(expected_rows) and len(expected_rows) == 17
    assert all("traces" not in r for r in rows.values())
    traces = read_json(out / SCORE_TRACE_TRACES_JSON)
    assert sorted(traces) == sorted(expected_rows)
    for row_id, per in traces.items():
        assert [t["trace_id"] for t in per] == ["ta", "tc", "td", "te", "tf"], row_id  # the scored set C, every row
        assert rows[row_id]["metrics"]["n_traces"] == 5 and rows[row_id]["metrics"]["n_cases"] == 2
        assert rows[row_id]["axis"] == "trace" and rows[row_id]["select"] == TRACE_SELECT_POST_NONOK
        assert rows[row_id]["bootstrap"]["b"] == 50 and rows[row_id]["bootstrap"]["seed"] == 0
    # the arm's two readings: mode arm with the chosen reading's lists; hit@1 as derived by hand above
    hop1, hop0 = rows[f"{ARM_TRACE}/hop1"], rows[f"{ARM_TRACE}/hop0"]
    assert (hop1["mode"], hop1["hop0"], hop0["mode"], hop0["hop0"]) == ("arm", 0, "arm", 1)
    assert hop1["metrics"]["s@1"] == TRACE_S1["hop1"] and hop0["metrics"]["s@1"] == TRACE_S1["hop0"]
    assert hop1["metrics"] == {
        "s@1": 0.25,
        "s@3": 0.75,
        "p@1": 0.5,
        "p@3": 0.625,
        "exact": 0.125,
        "mean_set_size": 2.0,
        "n_cases": 2,
        "n_traces": 5,
    }
    assert (
        hop1["counts"]["n_unprobed"] == 1 and hop1["counts"]["n_selected"] == 7 and hop1["counts"]["n_candidates"] == 5
    )
    assert [t["probed"] for t in traces[f"{ARM_TRACE}/hop1"]] == [True, True, False, True, True]  # td was skipped
    # the floor, the controls and the broadcast rows: modes and hop-0 flags by row, hit@1 the hand answers
    modes = {row_id: (rows[row_id]["mode"], rows[row_id]["hop0"]) for row_id in SHARED_ROWS}
    assert modes[f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop1"] == ("topology", 0) and modes[
        f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop0"
    ] == ("topology", 1)
    assert modes[ARM_CONTROL_TRACE_RANDOM] == ("random", 0)
    assert modes[f"{ARM_CONTROL_TRACE_CALLEES}/hop1"] == ("callees", 0) and modes[
        f"{ARM_CONTROL_TRACE_CALLEES}/hop0"
    ] == ("callees", 1)
    assert all(modes[row_id] == ("broadcast", 0) for row_id in BROADCAST_ROWS)
    assert {row_id: rows[row_id]["metrics"]["s@1"] for row_id in SHARED_ROWS} == SHARED_S1
    for row_id in BROADCAST_ROWS:
        assert rows[row_id]["metrics"]["p@1"] is None and rows[row_id]["metrics"]["exact"] is None
        parts = row_id.split("/")[1:]
        path = broadcast / RUNG / "latent" / f"seed={SEED}" / arm_slug("/".join(parts[:2]))
        path = (path / parts[2] if len(parts) == 3 else path) / RANKING_JSON
        assert rows[row_id]["source_ranking_sha256"] == sha256_file(path)
    assert all(
        "source_ranking_sha256" not in rows[row_id] for row_id in expected_rows if not row_id.startswith("broadcast/")
    )
    # the record binds both files, the rank record and every row's metrics
    rec = read_json(out / RUN_DIR / RESULTS_JSON)
    assert rec["status"] == "ok" and rec["stage"] == STAGE_SCORE_TRACE == "score-trace" and rec["axis"] == "trace"
    assert rec["arm"] == ARM_TRACE and rec["rung"] == RUNG and rec["seed"] == SEED and rec["corpus_id"] == CORPUS_ID
    assert rec["score_trace_sha256"] == sha256_file(out / SCORE_TRACE_JSON)
    assert rec["score_trace_traces_sha256"] == sha256_file(out / SCORE_TRACE_TRACES_JSON)
    assert rec["score_trace_traces_file"] == SCORE_TRACE_TRACES_JSON
    assert rec["rank_record_sha256"] == sha256_file(rank_folder / RUN_DIR / RESULTS_JSON)
    assert rec["ranking_trace_sha256"] == sha256_file(ranking)
    assert rec["rows"] == sorted(expected_rows) and set(rec["metrics"]) == set(expected_rows)
    assert all(rec["metrics"][row_id] == rows[row_id]["metrics"] for row_id in expected_rows)
    assert (rec["select"], rec["cap"], rec["c"], rec["max_len"], rec["device"]) == (
        TRACE_SELECT_POST_NONOK,
        0,
        C,
        MAX_LEN,
        "cpu",
    )
    assert (
        rec["n_cases"] == 2
        and rec["n_traces"] == 5
        and rec["counts"] == hop1["counts"]
        and rec["random"] == hop1["random"]
    )
    # the same cell under the second TRACE arm carries identical copies of the shared rows (what reporttrace checks)
    core_folder = tmp_path / "rank-trace-core"
    core_ranking = _rank_record(core_folder, _ranking_trace(ARM_TRACE_CORE, CORE_HOP1, CORE_HOP0))
    core_out = tmp_path / "trace-core"
    assert scoretrace.main(_score_argv(corpus, core_folder, core_ranking, broadcast, core_out)) == 0
    core_rows = read_json(core_out / SCORE_TRACE_JSON)["rows"]
    assert sorted(core_rows) == sorted([f"{ARM_TRACE_CORE}/{r}" for r in LANDED_READINGS] + SHARED_ROWS)
    assert all(core_rows[row_id] == rows[row_id] for row_id in SHARED_ROWS)
    assert (
        core_rows[f"{ARM_TRACE_CORE}/hop1"]["metrics"]["s@1"] == 0.0
        and core_rows[f"{ARM_TRACE_CORE}/hop0"]["metrics"]["s@1"] == 0.0
    )


# --- 2. refusals --------------------------------------------------------------------------------------------------


def test_scoretrace_refusals(corpus, tmp_path):
    broadcast = _broadcast_root(tmp_path / "results")
    doc = _ranking_trace(ARM_TRACE, TRACE_HOP1, TRACE_HOP0)

    def run(name, doc_, out=None, broadcast_=broadcast, **overrides):
        folder = tmp_path / name
        ranking = _rank_record(folder, doc_, **overrides)
        return scoretrace.score(corpus, folder, ranking, broadcast_, 0, 0, out or tmp_path / f"{name}-out")

    # (a) the ranking file does not hash to what its record bound
    with pytest.raises(ValueError, match="differs from the ranking"):
        run("sha", doc, ranking_sha256="0" * 64)
    # (b) a record of an arm without a per-trace axis (through main: a failed record is left behind)
    with pytest.raises(ValueError, match="no per-trace axis"):
        run("baro", {**doc, "arm": ARM_BARO})
    folder = tmp_path / "baro-main"
    ranking = _rank_record(folder, {**doc, "arm": ARM_BARO})
    with pytest.raises(ValueError, match="no per-trace axis"):
        scoretrace.main(_score_argv(corpus, folder, ranking, broadcast, tmp_path / "baro-main-out", b=0))
    assert read_json(tmp_path / "baro-main-out" / RUN_DIR / RESULTS_JSON)["status"] == "failed"
    # the ranking file names another arm than its record
    with pytest.raises(ValueError, match="file; the record is"):
        run("arm-mismatch", doc, arm=ARM_TRACE_CORE)
    # (c) the services list in another order than the corpus's deployed services in name order
    reordered = _ranking_trace(ARM_TRACE, TRACE_HOP1, TRACE_HOP0, services=["svc-c", "svc-b", "svc-a"])
    with pytest.raises(ValueError, match="name order"):
        run("services", reordered)
    # a rank read of another corpus
    with pytest.raises(ValueError, match="rank read is of"):
        run("corpus", doc, corpus_id="s/latent/seed=0")
    # the case directories are not the labelled cases' positional names
    renamed = {**doc, "cases": {"case-00": doc["cases"]["case-00"], "case-02": doc["cases"]["case-01"]}}
    with pytest.raises(ValueError, match="case directories"):
        run("case-dirs", renamed)
    # (d) a landed case ranking missing for one broadcast row
    partial = _broadcast_root(tmp_path / "results-partial", skip=(f"broadcast/{ARM_CAUSALRCA}/latency-95",))
    with pytest.raises(FileNotFoundError, match="latency-95"):
        run("broadcast", doc, broadcast_=partial)
    # (e) a selected list the benchmark's rule R does not reproduce (one extra id): the scorer's selection check
    extra = _ranking_trace(
        ARM_TRACE, TRACE_HOP1, TRACE_HOP0, selected={**SELECTED, "case-00": SELECTED["case-00"] + ["tk"]}
    )
    with pytest.raises(ValueError, match="case-00"):
        run("selected", extra)
    # a skipped id outside the selected list
    skipped = _ranking_trace(ARM_TRACE, TRACE_HOP1, TRACE_HOP0, skipped={**SKIPPED, "case-00": ["td", "tz"]})
    with pytest.raises(ValueError, match="skipped traces not in the selected list"):
        run("skipped", skipped)
    # a record that is not a completed rank-trace record, and no record at all
    with pytest.raises(ValueError, match="not a completed"):
        run("rank-stage", doc, stage="rank")
    with pytest.raises(ValueError, match="not a completed"):
        run("failed", doc, status="failed")
    with pytest.raises(FileNotFoundError):
        scoretrace.score(
            corpus, tmp_path / "nowhere", tmp_path / "sha" / RANKING_TRACE_JSON, broadcast, 0, 0, tmp_path / "x"
        )
    assert not any(p.name == SCORE_TRACE_JSON for p in tmp_path.rglob("*"))  # no refusal wrote a score file


# --- 3. reporttrace over a fake results tree ----------------------------------------------------------------------

SEEDS5 = (0, 1, 2, 3, 4)
ARMS2 = (ARM_TRACE, ARM_TRACE_CORE)


def _cell(cell_dict, arm_pred, mode, hop0, corpus, prediction=None):
    return score_trace_axis(corpus, cell_dict, prediction if prediction is not None else arm_pred, mode, hop0, 0, 0)


CELL = {
    "c": C,
    "max_len": MAX_LEN,
    "select": TRACE_SELECT_POST_NONOK,
    "cap": 0,
    "selected": SELECTED,
    "skipped": SKIPPED,
}


@pytest.fixture(scope="module")
def cell_rows(corpus):
    """`{arm: {row_id: result}}` — the rows of one corpus's cell per TRACE arm, in `score-trace.json`'s shape (the
    benchmark's `score` minus `traces`), the shared rows computed once and copied into both arms' cells."""
    return _cell_rows(corpus, CELL)


def _cell_rows(corpus, cell):
    def strip(res):
        res = dict(res)
        res.pop("traces")
        return res

    shared = {}
    for reading, hop0 in (("hop1", 0), ("hop0", 1)):
        shared[f"{ARM_FLOOR_TRACE_TOPOLOGY}/{reading}"] = strip(
            score_trace_axis(corpus, cell, None, "topology", hop0, 0, 0)
        )
        shared[f"{ARM_CONTROL_TRACE_CALLEES}/{reading}"] = strip(
            score_trace_axis(corpus, cell, None, "callees", hop0, 0, 0)
        )
    shared[ARM_CONTROL_TRACE_RANDOM] = strip(score_trace_axis(corpus, cell, None, "random", 0, 0, 0))
    for row_id in BROADCAST_ROWS:
        shared[row_id] = strip(score_trace_axis(corpus, cell, _broadcast_prediction(row_id), "broadcast", 0, 0, 0))
    out = {}
    for arm, (hop1, hop0) in {ARM_TRACE: (TRACE_HOP1, TRACE_HOP0), ARM_TRACE_CORE: (CORE_HOP1, CORE_HOP0)}.items():
        rows = dict(shared)
        rows[f"{arm}/hop1"] = strip(
            score_trace_axis(corpus, cell, {"services": SERVICES, "cases": hop1}, "arm", 0, 0, 0)
        )
        rows[f"{arm}/hop0"] = strip(
            score_trace_axis(corpus, cell, {"services": SERVICES, "cases": hop0}, "arm", 1, 0, 0)
        )
        out[arm] = rows
    return out


def _write_cell(d, arm, rows, rung=RUNG):
    """One landed cell: `score-trace.json`, its `score-trace` record and the `rank-trace` record it names."""
    rank_rec = d / "rank" / RUN_DIR / RESULTS_JSON
    write_json(
        rank_rec, {"status": "ok", "stage": STAGE_RANK_TRACE, "arm": arm, "rung": rung, "ranking_sha256": "x" * 64}
    )
    write_json(d / SCORE_TRACE_JSON, {"rows": rows})
    write_json(
        d / RUN_DIR / RESULTS_JSON,
        {
            "status": "ok",
            "stage": STAGE_SCORE_TRACE,
            "arm": arm,
            "rung": rung,
            "score_trace_sha256": sha256_file(d / SCORE_TRACE_JSON),
            "rank_record_sha256": sha256_file(rank_rec),
        },
    )


def _results_tree(root, cell_rows, seeds=SEEDS5, arms=ARMS2):
    for k in seeds:
        for arm in arms:
            _write_cell(reporttrace.cell_dir(root, RUNG, k, arm), arm, cell_rows[arm])
    return root


def _report_argv(results, out, arms=ARMS2, seeds=SEEDS5):
    argv = ["--results-dir", str(results), "--rungs", RUNG, "--seeds", *map(str, seeds), "--arms", *arms]
    return argv + ["--bootstrap-b", "50", "--bootstrap-seed", "0", "--output-folder", str(out)]


def test_reporttrace_pools_and_scores_hypotheses(corpus, cell_rows, tmp_path):
    assert (
        reporttrace.cell_dir("r", RUNG, 3, ARM_TRACE_CORE).as_posix()
        == f"r/xs/latent/seed=3/rca-trace-core/{TRACE_AXIS_DIR}"
    )
    results = _results_tree(tmp_path / "results", cell_rows)
    out = tmp_path / "tables"
    assert reporttrace.main(_report_argv(results, out)) == 0
    summary = read_json(out / "summary.json")
    rows, hyp = summary["rows"], summary["hypotheses"]
    arm_rows = [f"{arm}/{r}" for arm in ARMS2 for r in LANDED_READINGS]
    assert sorted(r["row"] for r in rows) == sorted(SHARED_ROWS + arm_rows)  # shared rows once, arm rows per arm
    assert all(r["rung"] == RUNG and r["n_seeds"] == 5 for r in rows) and len(rows) == 19
    by = {r["row"]: r for r in rows}
    for row_id, r in by.items():
        assert set(r) == {
            "rung",
            "row",
            "mode",
            "hop0",
            "metrics",
            "bootstrap",
            "random",
            "by_fault_kind",
            "counts",
            "n_seeds",
        }
        pooled = read_json(out / f"pooled-{RUNG}-{row_id.replace('/', '-')}.json")
        assert pooled["metrics"] == r["metrics"] and pooled["n_cases_per_result"] == [2] * 5 and "traces" not in pooled
        assert (
            pooled["bootstrap"]["b"] == 50
            and pooled["bootstrap"]["seed"] == 0
            and r["bootstrap"] == pooled["bootstrap"]
        )
        assert r["metrics"]["n_cases"] == 10 and r["metrics"]["n_traces"] == 25  # five identical corpora pooled
        assert (
            r["counts"]["n_candidates"] == 25
            and r["counts"]["n_selected"] == 35
            and r["counts"]["mean_n_services"] == pytest.approx(2.6)
        )
        assert (r["mode"], r["hop0"]) == (
            cell_rows[ARM_TRACE].get(row_id, cell_rows[ARM_TRACE_CORE].get(row_id))["mode"],
            cell_rows[ARM_TRACE].get(row_id, cell_rows[ARM_TRACE_CORE].get(row_id))["hop0"],
        )
    # the pooled points are the cells' (identical copies): the hand answers of the per-cell test
    s1 = {row_id: by[row_id]["metrics"]["s@1"] for row_id in SHARED_ROWS}
    assert s1 == SHARED_S1
    assert by[f"{ARM_TRACE}/hop1"]["metrics"]["s@1"] == 0.25 and by[f"{ARM_TRACE}/hop0"]["metrics"]["s@1"] == 0.75
    assert (
        by[f"{ARM_TRACE_CORE}/hop1"]["metrics"]["s@1"] == 0.0 and by[f"{ARM_TRACE_CORE}/hop0"]["metrics"]["s@1"] == 0.0
    )
    assert by[f"{ARM_TRACE}/hop1"]["random"]["s@1"] == pytest.approx(5 / 12)
    # the hypotheses by hand on those numbers
    assert set(hyp) == {
        "H-trace-perseq-random",
        "H-trace-perseq-topology",
        "H-trace-perseq-broadcast",
        "H-trace-perseq-self",
        "H-trace-earliest",
    }
    assert hyp["H-trace-earliest"]["verdict"] == "not run"  # no control/trace-earliest cell in this tree
    assert all(h["verdict"] in ("pass", "fail") for k, h in hyp.items() if k != "H-trace-earliest")
    # random: the hit@1 of every trace-reading row (0.25, 0.0, 0.25) is below the random expectation 5/12; no interval
    # can lie above it -> fail; the per-entry booleans are low > random
    rnd = hyp["H-trace-perseq-random"]
    assert rnd["verdict"] == "fail" and set(rnd["per_rung_and_row"]) == {f"{RUNG} {a}/hop1" for a in ARMS2} | {
        f"{RUNG} {ARM_FLOOR_TRACE_TOPOLOGY}/hop1"
    }
    for key, verdict in rnd["per_rung_and_row"].items():
        r = by[key.split(" ", 1)[1]]
        assert verdict == (r["bootstrap"]["intervals"]["s@1"]["low"] > r["random"]["s@1"]) and verdict is False
    # topology: the floor's hit@1 (0.25 at hop1, 1.0 at hop0) >= each arm's (0.25 / 0.75; 0.0 / 0.0) -> pass
    topo = hyp["H-trace-perseq-topology"]
    assert topo["verdict"] == "pass" and all(topo["per_rung_and_row"].values())
    assert topo["gap_arm_minus_floor_hit1"] == {
        f"{RUNG} {ARM_TRACE}/hop1": 0.0,
        f"{RUNG} {ARM_TRACE}/hop0": -0.25,
        f"{RUNG} {ARM_TRACE_CORE}/hop1": -0.25,
        f"{RUNG} {ARM_TRACE_CORE}/hop0": -1.0,
    }
    # broadcast: rca/baro's perfect case ranking scores 1.0 > 0.25 -> fail for both arms; it is the best broadcast row
    bc = hyp["H-trace-perseq-broadcast"]
    assert bc["verdict"] == "fail" and bc["per_rung_and_arm"] == {f"{RUNG} {a}": False for a in ARMS2}
    assert bc["best_broadcast"] == {f"{RUNG} {a}": {"row": f"broadcast/{ARM_BARO}", "s@1": 1.0} for a in ARMS2}
    # self: (arm - best broadcast) is negative at both readings for both arms -> no sign change -> pass
    self_ = hyp["H-trace-perseq-self"]
    assert self_["verdict"] == "pass" and self_["sign_arm_minus_best_broadcast"] == {
        f"{RUNG} {a}": {"hop1": -1, "hop0": -1} for a in ARMS2
    }
    # the markdown names every row and every hypothesis; the record binds the files
    md = (out / "rca-trace.md").read_text(encoding="utf-8")
    assert all(f"`{row_id}`" in md for row_id in SHARED_ROWS + arm_rows) and all(
        f"**{h}**: {v['verdict']}" in md for h, v in hyp.items()
    )
    rec = read_json(out / RUN_DIR / RESULTS_JSON)
    assert rec["status"] == "ok" and rec["stage"] == "report-trace" and rec["rows"] == sorted(SHARED_ROWS + arm_rows)
    assert (
        rec["hypotheses"] == {k: v["verdict"] for k, v in hyp.items()}
        and rec["arms"] == list(ARMS2)
        and rec["seeds"] == list(SEEDS5)
    )
    assert set(rec["files"]) == {"summary.json", "rca-trace.md"} | {
        f"pooled-{RUNG}-{row_id.replace('/', '-')}.json" for row_id in by
    }
    assert rec["files"]["summary.json"] == sha256_file(out / "summary.json")
    assert not any(name.startswith("pooled-xs-s") for name in rec["files"])  # one rung: no across-rung table
    # a single named arm reports its own rows and the shared rows only
    out1 = tmp_path / "tables-core"
    assert reporttrace.main(_report_argv(results, out1, arms=(ARM_TRACE_CORE,))) == 0
    assert read_json(out1 / RUN_DIR / RESULTS_JSON)["rows"] == sorted(
        SHARED_ROWS + [f"{ARM_TRACE_CORE}/{r}" for r in LANDED_READINGS]
    )
    # refusal: a shared row differs between the two arms' cells of one corpus
    differing = _results_tree(tmp_path / "results-differ", cell_rows)
    d = reporttrace.cell_dir(differing, RUNG, 2, ARM_TRACE_CORE)
    changed = dict(cell_rows[ARM_TRACE_CORE])
    changed[f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop1"] = {
        **changed[f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop1"],
        "metrics": {**changed[f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop1"]["metrics"], "s@1": 0.3},
    }
    shutil.rmtree(d)
    _write_cell(d, ARM_TRACE_CORE, changed)
    with pytest.raises(ValueError, match="differs between"):
        reporttrace.main(_report_argv(differing, tmp_path / "tables-differ"))
    assert read_json(tmp_path / "tables-differ" / RUN_DIR / RESULTS_JSON)["status"] == "failed"
    # refusal: a changed score file, and a seed missing from the tree
    changed_file = _results_tree(tmp_path / "results-changed", cell_rows)
    path = reporttrace.cell_dir(changed_file, RUNG, 1, ARM_TRACE) / SCORE_TRACE_JSON
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the file"):
        reporttrace.main(_report_argv(changed_file, tmp_path / "tables-changed"))
    missing = _results_tree(tmp_path / "results-missing", cell_rows)
    shutil.rmtree(reporttrace.cell_dir(missing, RUNG, 4, ARM_TRACE_CORE))
    with pytest.raises((FileNotFoundError, ValueError)):
        reporttrace.main(_report_argv(missing, tmp_path / "tables-missing"))
    # the flags are required and constrained
    for bad in (["--rungs", "xl"], ["--seeds", "7"], ["--arms", ARM_BARO]):
        with pytest.raises(SystemExit):
            reporttrace.main(
                _report_argv(results, tmp_path / "tables-bad")[:2]
                + bad
                + ["--bootstrap-b", "1", "--bootstrap-seed", "0", "--output-folder", "x"]
            )


def test_reporttrace_shared_rows_across_context_lengths(corpus, cell_rows, tmp_path):
    """The TRACE arms run at their own context length (xs seed 0: c = 3 and 2), so the shared rows' window
    coordinates differ between the arms' cells; the copies agree on every other field, the report runs, and the
    shared rows are the first named arm's."""
    short = _cell_rows(corpus, {**CELL, "c": C - 1, "skipped": {"case-00": [], "case-01": []}})[ARM_TRACE_CORE]
    for row_id in SHARED_ROWS:
        assert short[row_id] != cell_rows[ARM_TRACE][row_id]  # `c` at least moved with the window
        assert reporttrace.without_window_fields(short[row_id]) == reporttrace.without_window_fields(
            cell_rows[ARM_TRACE][row_id]
        )
    results = tmp_path / "results"
    for k in SEEDS5:
        _write_cell(reporttrace.cell_dir(results, RUNG, k, ARM_TRACE), ARM_TRACE, cell_rows[ARM_TRACE])
        _write_cell(reporttrace.cell_dir(results, RUNG, k, ARM_TRACE_CORE), ARM_TRACE_CORE, short)
    out, ref = tmp_path / "tables", tmp_path / "tables-ref"
    assert reporttrace.main(_report_argv(results, out)) == 0
    assert reporttrace.main(_report_argv(_results_tree(tmp_path / "results-ref", cell_rows), ref)) == 0
    for row_id in SHARED_ROWS:
        name = f"pooled-{RUNG}-{row_id.replace('/', '-')}.json"
        assert read_json(out / name) == read_json(ref / name)
    # a service-level difference still refuses
    bad = {**short, ARM_CONTROL_TRACE_RANDOM: {**short[ARM_CONTROL_TRACE_RANDOM], "counts": {"n_candidates": -1}}}
    d = reporttrace.cell_dir(results, RUNG, 3, ARM_TRACE_CORE)
    shutil.rmtree(d)
    _write_cell(d, ARM_TRACE_CORE, bad)
    with pytest.raises(ValueError, match="differs between"):
        reporttrace.main(_report_argv(results, tmp_path / "tables-bad"))


# --- 4. the hypotheses on hand-built summary rows ------------------------------------------------------------------


def _row(row_id, s1, low=None, random=0.3, rung=RUNG):
    low = s1 - 0.1 if low is None else low
    return {
        "rung": rung,
        "row": row_id,
        "metrics": {"s@1": s1, "s@3": min(1.0, s1 + 0.2)},
        "bootstrap": {"intervals": {"s@1": {"low": low, "high": min(1.0, s1 + 0.1)}}},
        "random": {"s@1": random},
    }


def test_reporttrace_hypotheses_by_hand():
    floor1, floor0 = f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop1", f"{ARM_FLOOR_TRACE_TOPOLOGY}/hop0"
    baro, rule = f"broadcast/{ARM_BARO}", "broadcast/control/rule"
    # every hypothesis passes: intervals above random, the floor above both arms at both readings, both arms above
    # every broadcast row at hop1, and hop0 keeps the sign of (arm - best broadcast)
    rows = [
        _row(f"{ARM_TRACE}/hop1", 0.7, low=0.6),
        _row(f"{ARM_TRACE}/hop0", 0.8),
        _row(f"{ARM_TRACE_CORE}/hop1", 0.6, low=0.5),
        _row(f"{ARM_TRACE_CORE}/hop0", 0.65),
        _row(floor1, 0.9, low=0.8),
        _row(floor0, 0.95),
        _row(baro, 0.5),
        _row(rule, 0.4),
    ]
    hyp = reporttrace.hypotheses(rows, (RUNG,), ARMS2)
    assert [h["verdict"] for h in hyp.values()] == ["pass"] * 4 + ["not run"]
    assert list(hyp) == [
        "H-trace-perseq-random",
        "H-trace-perseq-topology",
        "H-trace-perseq-broadcast",
        "H-trace-perseq-self",
        "H-trace-earliest",  # not run without the control's row (addendum "2026-10-31, later")
    ]
    assert hyp["H-trace-perseq-random"]["per_rung_and_row"] == {
        f"xs {ARM_TRACE}/hop1": True,
        f"xs {ARM_TRACE_CORE}/hop1": True,
        f"xs {floor1}": True,
    }
    assert hyp["H-trace-perseq-topology"]["gap_arm_minus_floor_hit1"] == pytest.approx(
        {
            f"xs {ARM_TRACE}/hop1": -0.2,
            f"xs {ARM_TRACE}/hop0": -0.15,
            f"xs {ARM_TRACE_CORE}/hop1": -0.3,
            f"xs {ARM_TRACE_CORE}/hop0": -0.3,
        }
    )
    assert hyp["H-trace-perseq-broadcast"]["best_broadcast"] == {
        f"xs {ARM_TRACE}": {"row": baro, "s@1": 0.5},
        f"xs {ARM_TRACE_CORE}": {"row": baro, "s@1": 0.5},
    }
    assert hyp["H-trace-perseq-self"]["sign_arm_minus_best_broadcast"] == {
        f"xs {a}": {"hop1": 1, "hop0": 1} for a in ARMS2
    }
    # every hypothesis fails: rca/trace's low (0.25) is below random (0.3); the floor's hop1 hit@1 (0.4) is under
    # both arms' (0.45, 0.6) while its hop0 (0.95) stays above; rca/baro (0.5) is above rca/trace at hop1 (not above
    # rca/trace-core); rca/trace's hop0 (0.6) flips its sign against baro, rca/trace-core leads at both readings
    rows = [
        _row(f"{ARM_TRACE}/hop1", 0.45, low=0.25),
        _row(f"{ARM_TRACE}/hop0", 0.6),
        _row(f"{ARM_TRACE_CORE}/hop1", 0.6, low=0.5),
        _row(f"{ARM_TRACE_CORE}/hop0", 0.65),
        _row(floor1, 0.4, low=0.35),
        _row(floor0, 0.95),
        _row(baro, 0.5),
        _row(rule, 0.4),
    ]
    hyp = reporttrace.hypotheses(rows, (RUNG,), ARMS2)
    assert [h["verdict"] for h in hyp.values()] == ["fail"] * 4 + ["not run"]
    assert hyp["H-trace-perseq-random"]["per_rung_and_row"][f"xs {ARM_TRACE}/hop1"] is False
    assert hyp["H-trace-perseq-random"]["per_rung_and_row"][f"xs {ARM_TRACE_CORE}/hop1"] is True
    assert hyp["H-trace-perseq-topology"]["per_rung_and_row"] == {
        f"xs {ARM_TRACE}/hop1": False,
        f"xs {ARM_TRACE}/hop0": True,
        f"xs {ARM_TRACE_CORE}/hop1": False,
        f"xs {ARM_TRACE_CORE}/hop0": True,
    }
    assert hyp["H-trace-perseq-topology"]["gap_arm_minus_floor_hit1"][f"xs {ARM_TRACE}/hop1"] == pytest.approx(0.05)
    assert hyp["H-trace-perseq-broadcast"]["per_rung_and_arm"] == {
        f"xs {ARM_TRACE}": False,
        f"xs {ARM_TRACE_CORE}": True,
    }
    assert hyp["H-trace-perseq-self"]["sign_arm_minus_best_broadcast"][f"xs {ARM_TRACE}"] == {"hop1": -1, "hop0": 1}
    assert hyp["H-trace-perseq-self"]["per_rung_and_arm"] == {f"xs {ARM_TRACE}": False, f"xs {ARM_TRACE_CORE}": True}
    # self: a tie (arm = best broadcast) at one reading and a lead at the other is a sign change too; equal signs at
    # both readings, whichever they are, is none
    tie = [_row(f"{ARM_TRACE}/hop1", 0.5), _row(f"{ARM_TRACE}/hop0", 0.6), _row(baro, 0.5)]
    assert reporttrace.hypotheses(tie, (RUNG,), (ARM_TRACE,))["H-trace-perseq-self"][
        "sign_arm_minus_best_broadcast"
    ] == {f"xs {ARM_TRACE}": {"hop1": 0, "hop0": 1}}
    assert reporttrace.hypotheses(tie, (RUNG,), (ARM_TRACE,))["H-trace-perseq-self"]["verdict"] == "fail"
    below = [_row(f"{ARM_TRACE}/hop1", 0.3), _row(f"{ARM_TRACE}/hop0", 0.4), _row(baro, 0.5), _row(rule, 0.45)]
    assert reporttrace.hypotheses(below, (RUNG,), (ARM_TRACE,))["H-trace-perseq-self"]["verdict"] == "pass"
    # a rung is judged on every named rung: a second rung failing one entry fails the hypothesis
    two = rows[:0] + [
        _row(f"{ARM_TRACE}/hop1", 0.7, low=0.6),
        _row(floor1, 0.9, low=0.8),
        _row(f"{ARM_TRACE}/hop1", 0.7, low=0.2, rung="s"),
        _row(floor1, 0.9, low=0.8, rung="s"),
    ]
    rnd = reporttrace.hypotheses(two, ("xs", "s"), (ARM_TRACE,))["H-trace-perseq-random"]
    assert rnd["verdict"] == "fail" and rnd["per_rung_and_row"] == {
        f"xs {ARM_TRACE}/hop1": True,
        f"xs {floor1}": True,
        f"s {ARM_TRACE}/hop1": False,
        f"s {floor1}": True,
    }
    # without the rows a hypothesis needs it cannot pass
    empty = reporttrace.hypotheses([], (RUNG,), ARMS2)
    assert all(h["verdict"] == "fail" for k, h in empty.items() if k != "H-trace-earliest")
    assert empty["H-trace-earliest"]["verdict"] == "not run"
    assert reporttrace._sign(0.0) == 0 and reporttrace._sign(-0.2) == -1 and reporttrace._sign(1e-9) == 1
