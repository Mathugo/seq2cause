"""The two CIRCA arms (`plans/rca-arms.md`, addendum 2026-10-25): the port's fallback is a counted miss
and never a ranking (D-TBR-8); the structural graph of Algorithm 1 is built from the case's `topology.json`
in the encoding the port's head reads, so that inside the port's own scoring each effect's parents are
exactly its registered causes (D-TBR-9); the topology door refuses bad edges; the report scores the
per-arm hypotheses and the gain hypothesis; and, when the pinned port is installed, both arms rank every
service of the fixture export end to end."""

import json

import numpy as np
import pytest
from fixture_corpus import SERVICES, TOPOLOGY_EDGES, fixture_topology
from test_rank_score import _export_record, _rank

from tracebenchrca import cases, report
from tracebenchrca import rank as rank_mod
from tracebenchrca.constants import ARM_CIRCA, ARM_CIRCA_TOPOLOGY, ARM_CONTROL_RULE, METRICS, RESULTS_JSON, RUN_DIR
from tracebenchrca.record import read_json

NAMES = [s for s, _ in SERVICES]
COLUMNS = [f"{s}_{m}" for s in NAMES for m in METRICS]


def _col(service, metric):
    return f"{service}_{metric}"


def test_structural_graph_follows_algorithm_1_on_the_four_series():
    adj, n_edges = rank_mod.structural_graph(COLUMNS, TOPOLOGY_EDGES)
    idx = {c: i for i, c in enumerate(COLUMNS)}

    def causes(effect):
        return {c for c in COLUMNS if adj[idx[c], idx[effect]] == 1 and adj[idx[effect], idx[c]] == -1}

    # within a service: Traffic -> each Latency, Traffic -> Errors; nothing into Traffic from inside
    assert causes(_col("tb-beta", "latency-50")) >= {_col("tb-beta", "count")}
    assert causes(_col("tb-beta", "error")) >= {_col("tb-beta", "count")}
    # across tb-edge -> tb-beta -> ext-gamma: caller T -> callee T; callee L -> caller L; callee E -> caller E
    assert causes(_col("tb-beta", "count")) == {_col("tb-edge", "count")}
    assert causes(_col("ext-gamma", "count")) == {_col("tb-beta", "count")}
    assert causes(_col("tb-beta", "latency-50")) == {
        _col("tb-beta", "count"),
        _col("ext-gamma", "latency-50"),
        _col("ext-gamma", "latency-95"),
    }
    assert causes(_col("tb-edge", "error")) == {_col("tb-edge", "count"), _col("tb-beta", "error")}
    assert causes(_col("tb-edge", "count")) == set()  # the BFF calls nobody above it
    # every edge is encoded once, antisymmetrically; the count matches
    assert n_edges == int((adj == 1).sum()) == int((adj == -1).sum())
    assert not ((adj == 1) & (adj.T == 1)).any()


def test_structural_graph_refuses_mutual_causes():
    with pytest.raises(ValueError):
        rank_mod.structural_graph(COLUMNS, [("tb-edge", "tb-beta", 1.0), ("tb-beta", "tb-edge", 1.0)])


def test_the_ports_head_sees_the_registered_parents():
    """Through the port's own graph construction (edges added, `reverse()`, `MemoryGraph.parents`)."""
    rht_mod = pytest.importorskip("RCAEval.graph_heads.rht")
    import pandas as pd

    adj, _ = rank_mod.structural_graph(COLUMNS, TOPOLOGY_EDGES)
    rows = 40
    df = pd.DataFrame({"time": np.arange(rows) * 30}, dtype=float)
    rng = np.random.default_rng(0)
    for c in COLUMNS:
        df[c] = rng.normal(size=rows)
    seen = {}
    orig = rht_mod.RHTScorer.score_node

    def spy(self, graph, series, node, data):
        seen[f"{node.entity}_{node.metric}"] = {f"{p.entity}_{p.metric}" for p in graph.parents(node)}
        return orig(self, graph, series, node, data)

    rht_mod.RHTScorer.score_node = spy
    try:
        rht_mod.rht(adj, inject_time=30 * 30, data=df)
    finally:
        rht_mod.RHTScorer.score_node = orig
    assert seen[_col("tb-beta", "count")] == {_col("tb-edge", "count")}
    assert seen[_col("tb-beta", "latency-95")] == {
        _col("tb-beta", "count"),
        _col("ext-gamma", "latency-50"),
        _col("ext-gamma", "latency-95"),
    }
    assert seen[_col("tb-edge", "error")] == {_col("tb-edge", "count"), _col("tb-beta", "error")}
    assert seen[_col("ext-gamma", "latency-50")] == {_col("ext-gamma", "count")}


def test_topology_door(tmp_path):
    rec, exp = _export_record(tmp_path)
    case = sorted(exp.glob("case-*"))[0]
    with pytest.raises(FileNotFoundError):
        cases.read_topology(case)  # the extra is absent from a three-file export
    fixture_topology(exp)
    assert cases.read_topology(case) == [(a, b, p) for a, b, p in TOPOLOGY_EDGES]
    for bad in (
        [{"caller": "tb-edge", "callee": "tb-edge", "p_call": 1}],
        [{"caller": "nope", "callee": "tb-beta", "p_call": 1}],
        [
            {"caller": "tb-edge", "callee": "tb-beta", "p_call": 1},
            {"caller": "tb-edge", "callee": "tb-beta", "p_call": 0.2},
        ],
    ):
        (case / "topology.json").write_text(json.dumps({"edges": bad}))
        with pytest.raises(ValueError):
            cases.read_topology(case)


def test_the_ports_fallback_is_a_counted_miss_and_never_a_ranking(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    calls = iter(
        [
            {"adj": [], "node_names": COLUMNS, "ranks": COLUMNS},  # the wrapper's fallback: unranked column order
            {"adj": np.zeros((len(COLUMNS), len(COLUMNS))), "ranks": [_col("ext-gamma", "error")] + COLUMNS},
        ]
    )

    def circa(data, inject_time=None, dataset=None):
        assert dataset is None and inject_time is not None and "time" in data.columns
        return next(calls)

    monkeypatch.setattr(rank_mod, "_circa", lambda: circa)
    out = tmp_path / "circa/rank"
    assert _rank(ARM_CIRCA, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert list(pos) == ["case-01"] and pos["case-01"][0] == "ext-gamma" and sorted(pos["case-01"]) == sorted(NAMES)
    assert res["n_unranked_cases"] == 1 and res["misses_by_reason"] == {"port fallback": 1}
    assert res["cases"][0]["miss"] == "port fallback" and res["cases"][0]["services"] is None
    assert res["cases"][1]["n_edges"] == 0 and res["cases"][1]["time_rank"] is None


def test_the_topology_arm_declares_and_needs_the_topology(tmp_path, monkeypatch):
    rec, exp = _export_record(tmp_path)
    monkeypatch.setattr(
        rank_mod, "_rht", lambda: lambda adj, inject_time, data: [(c, float(i)) for i, c in enumerate(COLUMNS)]
    )
    with pytest.raises(FileNotFoundError):
        _rank(ARM_CIRCA_TOPOLOGY, rec, tmp_path / "ct0/rank")
    fixture_topology(exp)
    out = tmp_path / "ct/rank"
    assert _rank(ARM_CIRCA_TOPOLOGY, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert all(sorted(v) == sorted(NAMES) for v in pos.values())
    assert pos["case-00"][0] == "ext-gamma"  # the port's re-sort: the largest score first
    assert all(c["n_call_edges"] == 2 and c["n_edges"] > 0 for c in res["cases"])


def test_real_port_on_the_fixture_export(tmp_path):
    """The pinned port: the fixture's constant columns make PC's correlation matrix singular on case 0, so the
    port falls back (D-TBR-8: a counted miss with the wrapper's line kept); on case 1 the head ranks only the
    metrics that carry a PC edge and are not constant inside its window (`CaseData(prune=True)`), so the other
    services are unranked and counted. The structural-graph arm runs no PC; its only loss is the pruning."""
    pytest.importorskip("RCAEval.e2e.circa")
    pytest.importorskip("causallearn")
    rec, exp = _export_record(tmp_path)
    fixture_topology(exp)
    out = tmp_path / "circa"
    assert _rank(ARM_CIRCA, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert res["misses_by_reason"] == {"port fallback": 1} and "singular" in res["cases"][0]["port_stderr"]
    assert list(pos) == ["case-01"] and set(pos["case-01"]) <= set(NAMES) and len(pos["case-01"]) >= 1
    assert res["cases"][1]["n_services_unranked"] == len(NAMES) - len(pos["case-01"])
    assert res["cases"][1]["time_rank"] is None and res["cases"][1]["n_edges"] >= 1
    out = tmp_path / "circa-topology"
    assert _rank(ARM_CIRCA_TOPOLOGY, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert res["n_unranked_cases"] == 0 and sorted(pos) == ["case-00", "case-01"]
    assert all(set(v) <= set(NAMES) and len(v) >= 1 for v in pos.values())
    assert all(c["n_services_unranked"] == len(NAMES) - len(pos[c["case_dir"]]) for c in res["cases"])
    assert all(c["time_rank"] is None and c["n_call_edges"] == 2 and c["n_edges"] == 21 for c in res["cases"])


def _row(rung, arm, avg, low, random=0.6):
    return {
        "rung": rung,
        "arm": arm,
        "metrics": {"avg@5": avg, "ac@1": avg, "ac@3": avg, "ac@5": avg, "n_cases": 10},
        "bootstrap": {"intervals": {"avg@5": {"low": low, "high": avg + 0.1}}},
        "random": {"avg@5": random},
        "n_missing": 0,
    }


def test_report_scores_the_per_arm_and_gain_hypotheses():
    rungs = ("xs",)
    rows = [
        _row("xs", ARM_CIRCA, 0.7, 0.65),
        _row("xs", ARM_CIRCA_TOPOLOGY, 0.8, 0.55),
        _row("xs", ARM_CONTROL_RULE, 0.75, 0.6),
    ]
    hyp = report.hypotheses(rows, rungs)
    assert hyp["H-rca-circa-random"]["verdict"] == "pass" and hyp["H-rca-circa-topology-random"]["verdict"] == "fail"
    assert hyp["H-rca-circa-rule"]["verdict"] == "fail" and hyp["H-rca-circa-topology-rule"]["verdict"] == "pass"
    assert hyp["H-circa-topology-gain"]["verdict"] == "pass" and hyp["H-circa-topology-gain"]["gap_topology_minus_pc"][
        "xs"
    ] == pytest.approx(0.1)
    hyp2 = report.hypotheses([_row("xs", ARM_CIRCA, 0.9, 0.7), _row("xs", ARM_CIRCA_TOPOLOGY, 0.8, 0.7)], rungs)
    assert hyp2["H-circa-topology-gain"]["verdict"] == "fail" and "H-rca-circa-rule" not in hyp2
    assert ARM_CIRCA in report.REPORT_ARMS and ARM_CIRCA_TOPOLOGY in report.REPORT_ARMS
