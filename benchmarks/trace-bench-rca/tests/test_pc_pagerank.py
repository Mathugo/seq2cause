"""`rca/pc-pagerank` (`plans/rca-arms.md`, addendum 2026-10-28): the adapter passes the port only the table and the
registered dataset name (D-TBR-12), never opens `window.json` (the arm declares no window), counts the port's
fallback as a miss (D-TBR-8's rule), refuses a ranking that still holds the clock, records the PageRank tie
structure in the export's column order, the score side locates the root cause's tie group from the labels, the
report scores the per-arm hypotheses, and — when the pinned port and scikit-network are installed — the arm ranks
the fixture export end to end."""

import numpy as np
import pytest
from fixture_corpus import SERVICES, fixture_corpus
from test_rank_score import _export_record, _rank, _score

from tracebenchrca import cases, report
from tracebenchrca import rank as rank_mod
from tracebenchrca.constants import (
    ARM_CONTROL_RULE,
    ARM_PC_PAGERANK,
    METRICS,
    PC_PAGERANK_DATASET,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_RCA_JSON,
)
from tracebenchrca.record import read_json

NAMES = [s for s, _ in SERVICES]
COLUMNS = [f"{s}_{m}" for s in NAMES for m in METRICS]


def _port(calls, seen):
    def pc_pagerank(data, **kwargs):
        seen.append({"kwargs": kwargs, "columns": list(data.columns)})
        return next(calls)

    return pc_pagerank


def test_tie_groups_follow_the_column_order_and_refuse_a_foreign_order():
    scores = [0.5, 0.5, 0.2, 0.1, 0.1, 0.1] + [0.05] * 6
    names = COLUMNS
    ranks = [n for n, _ in sorted(zip(names, scores, strict=True), key=lambda x: x[1], reverse=True)]
    assert ranks[:2] == COLUMNS[:2]  # a stable sort: ties keep the export's column order
    n_distinct, groups = rank_mod.tie_groups(names, scores, ranks)
    assert n_distinct == 4
    assert [g["n_metrics"] for g in groups] == [2, 1, 3, 6]
    assert groups[0]["services_first_ranked"] == ["tb-edge"] and groups[2]["services_first_ranked"] == ["tb-beta"]
    assert groups[3]["services_first_ranked"] == ["ext-gamma"]
    with pytest.raises(ValueError):
        rank_mod.tie_groups(names, scores, list(reversed(ranks)))


def test_the_adapter_passes_only_the_dataset_name_never_opens_the_window_and_counts_the_fallback(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    seen = []
    adj = np.zeros((len(COLUMNS), len(COLUMNS)))
    adj[0, 5] = 1  # tb-edge_count -> tb-beta_error
    calls = iter(
        [
            {"adj": [], "node_names": COLUMNS, "ranks": COLUMNS},  # the wrapper's fallback: unranked column order
            {"adj": adj, "node_names": COLUMNS, "ranks": None},  # ranks filled below from the port's own PageRank
        ]
    )
    scores = rank_mod._pagerank_scores(adj)
    port_ranks = [n for n, _ in sorted(zip(COLUMNS, scores, strict=True), key=lambda x: x[1], reverse=True)]
    calls = iter([next(calls), {"adj": adj, "node_names": COLUMNS, "ranks": port_ranks}])
    monkeypatch.setattr(rank_mod, "_pc_pagerank", lambda: _port(calls, seen))

    def boom(*_a, **_k):
        raise AssertionError("rca/pc-pagerank must never open window.json")

    monkeypatch.setattr(rank_mod, "read_window", boom)
    monkeypatch.setattr(cases, "read_window", boom)
    out = tmp_path / "pcpr/rank"
    assert _rank(ARM_PC_PAGERANK, rec, out) == 0
    assert all(c["kwargs"] == {"dataset": PC_PAGERANK_DATASET} for c in seen)
    assert all("time" in c["columns"] for c in seen)  # the table goes to the port unmodified; the port drops the clock
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert res["n_unranked_cases"] == 1 and res["misses_by_reason"] == {"port fallback": 1}
    assert res["cases"][0]["miss"] == "port fallback" and res["cases"][0]["services"] is None
    c1 = res["cases"][1]
    # PageRank walks against the edges, toward causes: the cause tb-edge_count takes the mass (CausalRCA's reading)
    assert list(pos) == ["case-01"] and pos["case-01"][0] == "tb-edge" and sorted(pos["case-01"]) == sorted(NAMES)
    assert c1["n_edges"] == 1 and c1["time_rank"] is None and c1["n_services_unranked"] == 0
    assert c1["n_metrics_dropped"] == 0 and c1["n_pre_rows"] is None
    # the cause takes the mass, the ten isolated (dangling) metrics tie, the effect with only an out-arc sits last
    assert c1["n_distinct_scores"] == 3 and [g["n_metrics"] for g in c1["tie_groups"]] == [1, 10, 1]
    assert c1["tie_groups"][1]["services_first_ranked"] == ["tb-beta", "ext-gamma"]  # the tail tie keeps column order


def test_a_ranking_that_keeps_the_clock_fails_the_run(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    calls = iter(
        [{"adj": np.zeros((len(COLUMNS) + 1,) * 2), "node_names": ["time"] + COLUMNS, "ranks": ["time"] + COLUMNS}]
    )
    monkeypatch.setattr(rank_mod, "_pc_pagerank", lambda: _port(calls, []))
    with pytest.raises(ValueError):
        rank_mod.rank(ARM_PC_PAGERANK, rec, "xs", "smallest-rung", "unused.json", tmp_path / "pcpr2/rank")


def test_the_score_side_locates_the_root_causes_tie_group(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    adj = np.zeros((len(COLUMNS), len(COLUMNS)))
    adj[0, 5] = 1  # one edge; every other metric ties in the tail group (an empty graph makes the port itself raise)
    scores = rank_mod._pagerank_scores(adj)
    port_ranks = [n for n, _ in sorted(zip(COLUMNS, scores, strict=True), key=lambda x: x[1], reverse=True)]
    calls = iter([{"adj": adj, "node_names": COLUMNS, "ranks": port_ranks}] * 2)
    monkeypatch.setattr(rank_mod, "_pc_pagerank", lambda: _port(calls, []))
    out = tmp_path / "pcpr3/rank"
    assert _rank(ARM_PC_PAGERANK, rec, out) == 0
    corpus = fixture_corpus()
    assert _score(corpus, out, tmp_path / "pcpr3") == 0
    res = read_json(tmp_path / "pcpr3" / RUN_DIR / RESULTS_JSON)
    assert res["n_cases_root_in_tie_group"] == 2  # tb-beta (case 0) and ext-gamma (case 1) both sit in the tail tie
    assert all(
        t["root_tie_group_metrics"] == len(COLUMNS) - 2 and t["root_tie_group_services"] == 2
        for t in res["root_tie_groups"]
    )
    assert (tmp_path / "pcpr3" / SCORE_RCA_JSON).exists()


def test_real_port_on_the_fixture_export(tmp_path):
    """The pinned port with scikit-network: the named-dataset path drops the fixture's whole-table constant columns
    before PC, so every remaining metric is a node; a service all of whose columns are constant is unranked and
    counted; `time` is never ranked; the run is deterministic."""
    pytest.importorskip("RCAEval.e2e.pc_pagerank")
    pytest.importorskip("sknetwork")
    rec, _ = _export_record(tmp_path)
    out = tmp_path / "real/rank"
    assert _rank(ARM_PC_PAGERANK, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    for c in res["cases"]:
        if c.get("miss"):
            assert c["miss"] == "port fallback" and c["port_stderr"]
            continue
        assert c["time_rank"] is None and c["n_metrics_dropped"] >= 0 and c["n_edges"] >= 0
        assert set(pos[c["case_dir"]]) <= set(NAMES) and c["n_services_unranked"] == len(NAMES) - len(
            pos[c["case_dir"]]
        )
        assert sum(g["n_metrics"] for g in c["tie_groups"]) == c["n_metrics_in"] - c["n_metrics_dropped"]
    out2 = tmp_path / "real2/rank"
    assert _rank(ARM_PC_PAGERANK, rec, out2) == 0
    assert read_json(out2 / rank_mod.POSITIONAL_JSON) == read_json(out / rank_mod.POSITIONAL_JSON)


def _row(rung, arm, avg, low, random=0.6):
    return {
        "rung": rung,
        "arm": arm,
        "metrics": {"avg@5": avg, "ac@1": avg, "ac@3": avg, "ac@5": avg, "n_cases": 10},
        "bootstrap": {"intervals": {"avg@5": {"low": low, "high": avg + 0.1}}},
        "random": {"avg@5": random},
        "n_missing": 0,
    }


def test_report_scores_the_per_arm_hypotheses():
    rows = [_row("xs", ARM_PC_PAGERANK, 0.7, 0.65), _row("xs", ARM_CONTROL_RULE, 0.75, 0.6)]
    hyp = report.hypotheses(rows, ("xs",))
    assert hyp["H-rca-pc-pagerank-random"]["verdict"] == "pass" and hyp["H-rca-pc-pagerank-rule"]["verdict"] == "fail"
    assert hyp["H-rca-pc-pagerank-rule"]["gap_arm_minus_rule"]["xs"] == pytest.approx(-0.05)
    assert ARM_PC_PAGERANK in report.REPORT_ARMS
