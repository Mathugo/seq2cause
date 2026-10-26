"""`rca/tracerca` (`plans/rca-arms.md`, addendum 2026-10-25, later (2); D-TBR-10): the spans door, the port's
table built from the exporter's columns in microseconds, the collapse through the spans' own (service,
operation) pairs, the two registered miss rules (KeyError; an empty ranking), the record's counts, and the
pinned port end to end on fixture spans."""

import json

import pytest
from fixture_corpus import SERVICES, fixture_spans
from test_rank_score import _export_record, _rank

from tracebenchrca import cases
from tracebenchrca import rank as rank_mod
from tracebenchrca.constants import ARM_TRACERCA, RESULTS_JSON, RUN_DIR
from tracebenchrca.record import read_json

NAMES = [s for s, _ in SERVICES]


def test_spans_door_and_the_ports_table(tmp_path):
    rec, exp = _export_record(tmp_path)
    case = sorted(exp.glob("case-*"))[0]
    with pytest.raises(FileNotFoundError):
        cases.read_spans(case)
    rows = fixture_spans(exp)
    spans = cases.read_spans(case)
    assert list(spans.columns) == ["trace_id", "service", "operation", "start_us", "duration_us", "parent_pos"]
    assert len(spans) == len(rows[case.name]) and spans["trace_id"].nunique() == 42
    df, service_of = rank_mod.tracerca_frame(spans)
    assert list(df.columns) == ["serviceName", "methodName", "startTime", "duration", "operationName"]
    assert service_of == {
        "tb-edge_GET /v1/a": "tb-edge",
        "tb-beta_GET /v1/b_c": "tb-beta",
        "ext-gamma_POST /v1/g": "ext-gamma",
    }
    inject_us = json.loads((case / "window.json").read_text())["inject_time"] * 1_000_000
    n_abn, n_post, n_pre = rank_mod.abnormal_span_count(df, inject_us)
    assert (n_pre, n_post) == (90, 36) and n_abn == 24  # tb-beta's and its caller's post-fault spans exceed mean + 3 SD
    (case / "spans.csv").write_text("trace_id,service,operation,start_us,duration_us,parent_pos\nt,nope,op,1,1,-1\n")
    with pytest.raises(ValueError):
        cases.read_spans(case)


def test_the_clients_spans_pass_the_door_and_never_reach_the_port(tmp_path, monkeypatch):
    """Correction 2026-10-26: the synthetic client's root spans are dropped before the port and counted."""
    rec, exp = _export_record(tmp_path)
    rows = fixture_spans(exp)
    for d in sorted(exp.glob("case-*")):
        inject = json.loads((d / "window.json").read_text())["inject_time"] * 1_000_000
        with open(d / "spans.csv", "a", newline="", encoding="utf-8") as f:
            for i in range(7):
                f.write(f"{d.name}-c{i},client,/page/x,{inject + (i - 3) * 1_000_000},{500_000},-1\n")
    case = sorted(exp.glob("case-*"))[0]
    spans = cases.read_spans(case)
    assert len(spans) == len(rows[case.name]) + 7 and (spans["service"] == "client").sum() == 7

    def port(data, inject_time=None, dataset=None):
        assert "client" not in set(data["serviceName"]) and len(data) == len(rows[case.name])
        return {"ranks": ["tb-beta_GET /v1/b_c"]}

    monkeypatch.setattr(rank_mod, "_tracerca", lambda: port)
    out = tmp_path / "tracerca/rank"
    assert _rank(ARM_TRACERCA, rec, out) == 0
    c0 = read_json(out / RUN_DIR / RESULTS_JSON)["cases"][0]
    assert (c0["n_spans_read"], c0["n_spans_client"], c0["n_spans"]) == (126 + 7, 7, 126)
    assert c0["n_traces_read"] == 42 + 7 and c0["n_traces"] == 42 and c0["n_abnormal_spans"] == 24


def test_rank_collapses_through_the_pairs_and_counts_the_misses(tmp_path, monkeypatch):
    rec, exp = _export_record(tmp_path)
    fixture_spans(exp)
    calls = iter(
        [
            {"ranks": ["tb-beta_GET /v1/b_c", "tb-edge_GET /v1/a"]},  # ext-gamma unranked
            KeyError("tb-beta_GET /v1/new"),
        ]
    )

    def port(data, inject_time=None, dataset=None):
        assert (
            set(data.columns) >= {"serviceName", "methodName", "startTime", "duration"} and inject_time % 1_000_000 == 0
        )
        out = next(calls)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(rank_mod, "_tracerca", lambda: port)
    out = tmp_path / "tracerca/rank"
    assert _rank(ARM_TRACERCA, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert pos == {"case-00": ["tb-beta", "tb-edge"]}
    assert (
        res["cases"][0]["n_services_unranked"] == 1
        and res["cases"][0]["n_traces"] == 42
        and res["cases"][0]["n_spans"] == 126
    )
    assert res["cases"][1]["miss"] == "operation unseen before the injection" and res["cases"][1]["services"] is None
    assert res["misses_by_reason"] == {"operation unseen before the injection": 1} and res["n_unranked_cases"] == 1
    assert res["cases"][0]["n_abnormal_spans"] == 24 and res["n_filled_cells"] == 0


def test_an_empty_ranking_is_passed_through_as_a_miss(tmp_path, monkeypatch):
    rec, exp = _export_record(tmp_path)
    fixture_spans(exp)
    monkeypatch.setattr(rank_mod, "_tracerca", lambda: lambda data, inject_time=None, dataset=None: {"ranks": []})
    out = tmp_path / "tracerca/rank"
    assert _rank(ARM_TRACERCA, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    assert res["n_empty_rankings"] == 2 and all(
        c["services"] == [] and c["n_services_unranked"] == 3 for c in res["cases"]
    )
    assert read_json(out / rank_mod.POSITIONAL_JSON)["cases"] == {"case-00": [], "case-01": []}


def test_the_port_never_sees_the_series(tmp_path, monkeypatch):
    rec, exp = _export_record(tmp_path)
    fixture_spans(exp)
    for d in exp.glob("case-*"):
        (d / "data.csv").unlink()  # the arm declares spans, window, services — not the series
    monkeypatch.setattr(
        rank_mod, "_tracerca", lambda: lambda data, inject_time=None, dataset=None: {"ranks": ["tb-beta_GET /v1/b_c"]}
    )
    assert _rank(ARM_TRACERCA, rec, tmp_path / "t/rank") == 0


def test_real_port_on_fixture_spans(tmp_path):
    pytest.importorskip("RCAEval.e2e.tracerca")
    rec, exp = _export_record(tmp_path)
    fixture_spans(exp)
    out = tmp_path / "tracerca/rank"
    assert _rank(ARM_TRACERCA, rec, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert res["n_unranked_cases"] == 0 and sorted(pos) == ["case-00", "case-01"]
    # the slow service's operation and its caller's are the flagged ones; ext-gamma has no abnormal span -> dropped
    for v in pos.values():
        assert set(v) == {"tb-beta", "tb-edge"}  # a tie (equal support and confidence) keeps the port's own order
    assert all(c["n_services_unranked"] == 1 and c["n_operations_ranked"] == 2 for c in res["cases"])
