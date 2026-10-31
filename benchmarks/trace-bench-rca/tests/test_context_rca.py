"""The context family's root-cause side (`plans/rca-arms.md`, addendum "2026-10-31, later"; D-TBR-16 … 19): the hop-0
and source readings of the rule on hand graphs, the pre-fault subtraction, the ranking stage's readings / phases /
subtraction / one-layout reuse with the stand-in runner, the earliest-error control end to end on the hand corpus
(case level through `score`, per trace through `scoretrace --rows arm-only`), and the case-level report's series and
hypotheses by hand."""

import json
from pathlib import Path

import pytest
from fixture_corpus import SERVICES, fixture_corpus
from test_rank_score import LEDGER, _export_record, _score
from test_trace import _fake_runner, _ranktrace, pos
from test_trace_axis import _harness
from test_trace_axis_score import (
    EPOCH,
    corpus,  # noqa: F401  (the hand corpus, a module fixture)
)
from test_trace_axis_score import SERVICES as HAND_SERVICES

from tracebenchrca import earliest, report, scoretrace, tracerule
from tracebenchrca import ranktrace as rt
from tracebenchrca.constants import (
    ARM_CONTROL_RULE,
    ARM_CONTROL_TRACE_EARLIEST,
    ARM_RCD,
    ARM_TRACE,
    ARM_TRACE_ATOMIC_CMI_KEEP,
    ARM_TRACE_CMI_OVERWRITE,
    ARM_TRACE_IG_OVERWRITE,
    CONTEXT_TRACE_ARMS,
    RANKING_TRACE_JSON,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_RCA_JSON,
    SCORE_TRACE_JSON,
    TRACE_CELLS,
    TRACE_MISS_NO_SOURCE,
    TRACE_PRE_SOURCE_CORPUS_START,
    TRACE_PRE_SOURCE_PREVIOUS,
    TRACE_READING_EARLIEST,
    TRACE_READINGS,
    TRACE_SELECT_POST_NONOK,
    TRACE_SERIES,
)
from tracebenchrca.record import read_json, write_json

NAMES = [s for s, _ in SERVICES]  # tb-edge, tb-beta, ext-gamma


# --- the rule: hop 0 and the source rule ----------------------------------------------------------------------------


def test_source_rule_on_hand_graphs():
    # chain 0 tb-edge(ok) -> 1 tb-beta(5xx) -> 2 ext-gamma(5xx): position 1 has an OK parent (a source), position 2 a
    # non-OK parent (not one); an isolated 5xx (no parent) is a source too
    chain = {
        "positions": [pos("tb-edge"), pos("tb-beta", "5xx"), pos("ext-gamma", "5xx")],
        "edges": [[0, 1, 1.0], [1, 2, 1.0]],
    }
    h = tracerule.rank_trace(chain, NAMES, rule="source")
    assert h["services"] == ["tb-beta"] and h["positions"] == [1]
    assert h["n_sources"] == 1 and h["n_sources_isolated"] == 0 and h["n_nonok"] == 2
    lone = {"positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": []}
    h = tracerule.rank_trace(lone, NAMES, rule="source")
    assert h["services"] == ["tb-edge"] and h["n_sources"] == 1 and h["n_sources_isolated"] == 1
    # two sources in one trace rank by the number held, then services.json order; a client source is dropped
    two = {
        "positions": [
            pos("client", "5xx", True),
            pos("ext-gamma", "err"),
            pos("ext-gamma", "slow"),
            pos("tb-beta", "5xx"),
        ],
        "edges": [[1, 3, 1.0]],
    }
    h = tracerule.rank_trace(two, NAMES, rule="source")
    assert h["services"] == ["ext-gamma"] and h["positions"] == [1, 2] and h["n_client_dropped"] == 1
    assert h["n_sources"] == 3  # the client's, ext-gamma's two; tb-beta's 5xx has the non-OK parent 1
    with pytest.raises(ValueError):
        tracerule.rank_trace(chain, NAMES, rule="elsewhere")


def test_rank_case_hop0_and_source_readings_and_misses():
    t1 = {"positions": [pos("ext-gamma"), pos("tb-beta"), pos("tb-edge", "5xx")], "edges": [[0, 1, 1.0], [1, 2, 1.0]]}
    t2 = {"positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": []}
    hop1 = tracerule.rank_case([t1, t2], NAMES)
    hop0 = tracerule.rank_case([t1, t2], NAMES, hop0=True)
    assert hop1["services"] == ["tb-beta", "ext-gamma", "tb-edge"] and hop1["counts"] == {"tb-beta": 1, "ext-gamma": 1}
    assert hop1["hop0"] is False and hop1["rule"] == "closure"
    # hop 0: tb-edge credits itself in both traces (2 traces, mean hop 0) and leads
    assert hop0["services"] == ["tb-edge", "tb-beta", "ext-gamma"] and hop0["counts"]["tb-edge"] == 2
    assert hop0["mean_hops"]["tb-edge"] == 0.0 and hop0["hop0"] is True and hop0["self_counts"] == {"tb-edge": 2}
    src = tracerule.rank_case([t1, t2], NAMES, rule="source")
    # t1: position 2 (tb-edge 5xx) has the OK parent 1 -> a source; t2: isolated -> a source: tb-edge holds both
    assert src["services"] == ["tb-edge", "tb-beta", "ext-gamma"] and src["counts"] == {"tb-edge": 2}
    assert src["n_sources"] == 2 and src["n_sources_isolated"] == 1 and src["source_counts"] == {"tb-edge": 2}
    assert src["rule"] == "source" and src["hop0"] is True and src["miss"] is None
    # only a client source: a counted miss under the source rule
    client = {"positions": [pos("tb-beta"), pos("client", "5xx", True)], "edges": [[0, 1, 1.0]]}
    row = tracerule.rank_case([client], NAMES, rule="source")
    assert row["services"] is None and row["miss"] == TRACE_MISS_NO_SOURCE and row["n_client_positions_dropped"] == 1
    # the landed reading of the same traces is unchanged
    assert tracerule.rank_case([client], NAMES)["services"][0] == "tb-beta"


def test_subtract_rates():
    post = {
        "n_traces_probed": 4,
        "counts": {"tb-beta": 4, "tb-edge": 2},
        "services": ["tb-beta", "tb-edge", "ext-gamma"],
    }
    pre = {"n_traces_probed": 2, "counts": {"tb-beta": 2, "ext-gamma": 1}}
    sub = tracerule.subtract_rates(post, pre, NAMES)
    # tb-beta 1.0 - 1.0 = 0; tb-edge 0.5 - 0 = 0.5; ext-gamma 0 - 0.5 = -0.5
    assert sub["scores"] == {"tb-edge": 0.5, "tb-beta": 0.0, "ext-gamma": -0.5}
    assert sub["services"] == ["tb-edge", "tb-beta", "ext-gamma"] and sub["no_pre"] is False
    assert sub["n_post_probed"] == 4 and sub["n_pre_probed"] == 2
    # no pre trace: the post rates stand, flagged; a post miss stays a miss
    none = tracerule.subtract_rates(post, None, NAMES)
    assert (
        none["no_pre"] and none["services"] == ["tb-beta", "tb-edge", "ext-gamma"] and none["pre_rate"]["tb-beta"] == 0
    )
    miss = tracerule.subtract_rates({**post, "services": None}, pre, NAMES)
    assert miss["services"] is None


# --- the ranking stage: readings, one-layout reuse, the pre phase and the subtraction -----------------------------


DOCS = {
    "case-00": [
        {"positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": [[0, 1, 0.7]]},
        {
            "positions": [pos("ext-gamma"), pos("tb-beta", "slow"), pos("tb-edge", "5xx")],
            "edges": [[0, 1, 0.2], [1, 2, 0.9]],
        },
    ],
    "case-01": [{"positions": [pos("tb-beta"), pos("tb-edge")], "edges": [[0, 1, 0.7]]}],
}


def _freeze_with_layout(tmp_path, arms):
    freeze = tmp_path / "freeze.json"
    cells = {
        TRACE_CELLS[a]: {
            "probe": "core",
            "noise": "all",
            "c": 1,
            "N": 8,
            "g": 1,
            "tau": 0.1,
            "context_mode": "overwrite",
        }
        for a in arms
    }
    freeze.write_text(json.dumps({"cells": cells}))
    return freeze


def test_ranktrace_readings_reuse_pre_phase_and_subtraction(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    fx_corpus = fixture_corpus()
    harness_root = _harness(tmp_path)
    seen = []
    monkeypatch.setattr(rt, "_spawn", _fake_runner(DOCS, seen))
    monkeypatch.setattr(rt, "pinned_environment", lambda python: ["stand-in==0"])
    _freeze_with_layout(tmp_path, (ARM_TRACE_IG_OVERWRITE, ARM_TRACE_CMI_OVERWRITE))
    work = tmp_path / "work-overwrite"
    both = [ARM_TRACE_IG_OVERWRITE, ARM_TRACE_CMI_OVERWRITE]
    # 1. the IG arm, hop0, probing both cells of the layout
    out_ig = tmp_path / "ig-hop0"
    assert (
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            out_ig,
            tmp_path,
            harness_root,
            reading="hop0",
            probe_arms=both,
            work_dir=work,
        )
        == 0
    )
    (call,) = seen
    cmd = call["cmd"]
    i = cmd.index("--cells")
    assert cmd[i + 1 : i + 3] == [TRACE_CELLS[a] for a in both] and cmd[cmd.index("--phase") + 1] == "post"
    res = read_json(out_ig / RUN_DIR / RESULTS_JSON)
    assert res["reading"] == "hop0" and res["probe_arms"] == both and res["runner_reused"] is False
    assert res["layout"] == {"probe": "core", "noise": "all", "c": 1, "N": 8, "g": 1, "context_mode": "overwrite"}
    assert res["code"]["cells"] == [TRACE_CELLS[a] for a in both] and res["code"]["commit"].startswith("280ed948")
    posit = read_json(out_ig / rt.POSITIONAL_JSON)["cases"]
    assert posit["case-00"][0] == "tb-edge"  # hop 0: the failed tb-edge credits itself in both traces
    assert res["cases"][0]["hop0"] is True and res["cases"][0]["rule"] == "closure"
    assert res["pre_starts"]["case-00"]["pre_source"] == TRACE_PRE_SOURCE_CORPUS_START
    assert res["pre_starts"]["case-01"]["pre_source"] == TRACE_PRE_SOURCE_PREVIOUS
    windows = read_json(work / rt.CASES_JSON)
    assert windows[0]["pre_start_time"] == 0 and windows[1]["pre_start_time"] == windows[0]["end_time"]
    # 2. the CMI arm of the same layout reuses the work directory: no second runner call, the files' hashes recorded
    out_cmi = tmp_path / "cmi-source"
    assert (
        _ranktrace(
            ARM_TRACE_CMI_OVERWRITE,
            rec,
            fx_corpus,
            out_cmi,
            tmp_path,
            harness_root,
            reading="source",
            probe_arms=both,
            work_dir=work,
        )
        == 0
    )
    assert len(seen) == 1
    res = read_json(out_cmi / RUN_DIR / RESULTS_JSON)
    assert res["runner_reused"] is True and set(res["work_files_sha256"]) == {
        rt.SUMMARY_JSON,
        "case-00.json",
        "case-01.json",
    }
    assert res["reading"] == "source" and res["cases"][0]["rule"] == "source" and res["cases"][0]["n_sources"] == 2
    assert read_json(out_cmi / rt.POSITIONAL_JSON)["cases"]["case-00"][0] == "tb-edge"
    # 3. a work directory of another selection is not reused
    out_other = tmp_path / "ig-hop1-window-cap1"
    assert (
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            out_other,
            tmp_path,
            harness_root,
            reading="hop1",
            probe_arms=both,
            work_dir=work,
            cap=1,
        )
        == 0
    )
    assert len(seen) == 2
    # 4. the pre phase: the runner is asked for the pre-fault stretch of each case
    out_pre = tmp_path / "ig-hop0-pre"
    work_pre = tmp_path / "work-pre"
    assert (
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            out_pre,
            tmp_path,
            harness_root,
            reading="hop0",
            phase="pre",
            probe_arms=both,
            work_dir=work_pre,
        )
        == 0
    )
    cmd = seen[-1]["cmd"]
    assert cmd[cmd.index("--phase") + 1] == "pre"
    pre_res = read_json(out_pre / RUN_DIR / RESULTS_JSON)
    assert pre_res["phase"] == "pre" and pre_res["subtract"] == "none"
    # 5. the subtraction: post − pre under hop0 (the stand-in returns the same traces in both phases: every score 0)
    out_sub = tmp_path / "ig-hop0-subpre"
    assert (
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            out_sub,
            tmp_path,
            harness_root,
            reading="hop0",
            subtract="pre",
            pre_record=out_pre,
            probe_arms=both,
            work_dir=work,
        )
        == 0
    )
    res = read_json(out_sub / RUN_DIR / RESULTS_JSON)
    assert res["subtract"] == "pre" and res["n_cases_no_pre"] == 0 and res["pre_record"]["path"] == str(out_pre)
    sub = res["cases"][0]["subtraction"]
    assert set(sub["scores"].values()) == {0.0} and sub["n_post_probed"] == sub["n_pre_probed"] == 2
    assert read_json(out_sub / rt.POSITIONAL_JSON)["cases"]["case-00"] == posit["case-00"]  # ties by the post rate
    # refusals: the pre phase on the trace axis, a subtraction without a pre record, a pre record of another reading
    with pytest.raises(ValueError):
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            tmp_path / "x1",
            tmp_path,
            harness_root,
            axis="trace",
            select=TRACE_SELECT_POST_NONOK,
            cap=0,
            phase="pre",
            probe_arms=both,
            work_dir=work,
        )
    with pytest.raises(ValueError):
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            tmp_path / "x2",
            tmp_path,
            harness_root,
            reading="hop0",
            subtract="pre",
            probe_arms=both,
            work_dir=work,
        )
    with pytest.raises(ValueError):
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            tmp_path / "x3",
            tmp_path,
            harness_root,
            reading="hop1",
            subtract="pre",
            pre_record=out_pre,
            probe_arms=both,
            work_dir=work,
        )
    with pytest.raises(ValueError):  # --probe-arms must name the arm
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            tmp_path / "x4",
            tmp_path,
            harness_root,
            probe_arms=[ARM_TRACE_CMI_OVERWRITE],
            work_dir=work,
        )


def test_ranktrace_refuses_probe_arms_of_two_layouts(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    fx_corpus = fixture_corpus()
    harness_root = _harness(tmp_path)
    monkeypatch.setattr(rt, "_spawn", _fake_runner(DOCS, []))
    monkeypatch.setattr(rt, "pinned_environment", lambda python: [])
    freeze = _freeze_with_layout(tmp_path, (ARM_TRACE_IG_OVERWRITE, ARM_TRACE_ATOMIC_CMI_KEEP))
    doc = json.loads(freeze.read_text())
    doc["cells"][TRACE_CELLS[ARM_TRACE_ATOMIC_CMI_KEEP]]["probe"] = "cli-atomic-cmi"
    freeze.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="one layout"):
        _ranktrace(
            ARM_TRACE_IG_OVERWRITE,
            rec,
            fx_corpus,
            tmp_path / "o",
            tmp_path,
            harness_root,
            probe_arms=[ARM_TRACE_IG_OVERWRITE, ARM_TRACE_ATOMIC_CMI_KEEP],
        )


# --- the earliest-error control on the hand corpus -------------------------------------------------------------------


@pytest.fixture(scope="module")
def hand_corpus(request):  # the hand corpus pulled by its own fixture name (imported above)
    return request.getfixturevalue("corpus")


def _hand_export(tmp_path, corpus_dir):
    """An export of the hand corpus: two case directories with the window and the services through the case door."""
    exp = tmp_path / "exports/hand"
    kinds = {"svc-a": "bff", "svc-b": "service", "svc-c": "service"}
    cases = read_json(corpus_dir / "labels" / "cases.json")["cases"]
    for i, case in enumerate(cases):
        d = exp / f"case-{i:02d}"
        d.mkdir(parents=True)
        write_json(  # the export's window fields are absolute UNIX seconds (the run window's epoch + the offset)
            d / "window.json",
            {"inject_time": EPOCH + int(case["inject_time_s"]), "end_time": EPOCH + int(case["end_time_s"])},
        )
        write_json(d / "services.json", {"services": [{"name": n, "kind": kinds[n]} for n in HAND_SERVICES]})
    rec = tmp_path / "export-record-hand"
    write_json(
        rec / RUN_DIR / RESULTS_JSON,
        {"status": "ok", "stage": "export", "rung": "xs", "corpus_id": "xs/latent/seed=0", "export_dir": str(exp)},
    )
    return rec


def _earliest(axis, rec, corpus_dir, out, select=TRACE_SELECT_POST_NONOK, cap=0):
    return earliest.main(
        [
            "--axis",
            axis,
            "--select",
            select,
            "--cap",
            str(cap),
            "--export-record",
            str(rec),
            "--corpus-dir",
            str(corpus_dir),
            "--rung",
            "xs",
            "--seed",
            "0",
            "--prior-rung-record",
            "smallest-rung",
            "--staging-ledger",
            LEDGER,
            "--output-folder",
            str(out),
        ]
    )


def test_earliest_control_case_level_and_per_trace(hand_corpus, tmp_path):
    corpus_dir = hand_corpus
    rec = _hand_export(tmp_path, corpus_dir)
    # case level: per trace the service of the earliest non-OK non-client span (by hand on the hand corpus):
    # case-00 ta svc-c, tb svc-b, tc svc-b, td svc-c, te svc-b -> svc-b 3, svc-c 2; case-01 tf svc-a, tg svc-a
    out = tmp_path / "earliest/rank"
    assert _earliest("ranking", rec, corpus_dir, out) == 0
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    posit = read_json(out / earliest.POSITIONAL_JSON)["cases"]
    assert (
        res["stage"] == "rank" and res["arm"] == ARM_CONTROL_TRACE_EARLIEST and res["reading"] == TRACE_READING_EARLIEST
    )
    assert posit == {"case-00": ["svc-b", "svc-c", "svc-a"], "case-01": ["svc-a", "svc-b", "svc-c"]}
    assert res["cases"][0]["counts"] == {"svc-b": 3, "svc-c": 2} and res["cases"][0]["n_traces_probed"] == 5
    assert res["cases"][1]["counts"] == {"svc-a": 2} and res["n_unranked_cases"] == 0
    assert _score(corpus_dir, out, tmp_path / "earliest") == 0
    sc = read_json(tmp_path / "earliest" / SCORE_RCA_JSON)
    assert [c["rank"] for c in sc["cases"]] == [2, 2]  # svc-c second in case 0, svc-b second in case 1
    # per trace: one reading `earliest`; services by the earliest non-OK span, positions likewise
    out_t = tmp_path / "earliest/rank-trace"
    assert _earliest("trace", rec, corpus_dir, out_t) == 0
    doc = read_json(out_t / RANKING_TRACE_JSON)
    assert doc["schema"] == "ranking-trace@2" and doc["readings"] == [TRACE_READING_EARLIEST] and doc["c"] == 1
    assert doc["services"] == HAND_SERVICES and doc["cases"]["case-00"]["selected"] == ["ta", "tb", "tc", "td", "te"]
    t = doc["cases"]["case-00"]["traces"]
    assert t["ta"] == {"s": [2, 0, 1], "p": [1, 0, 2]}  # svc-c failed first; then svc-a, svc-b by name
    assert t["tc"] == {"s": [1, 2, 0], "p": [1, 0, 2]}  # svc-b's err at +2 ms precedes svc-c's at +3 ms
    assert t["te"]["s"] == [1, 2, 0] and t["te"]["p"][:2] == [4, 5]
    assert doc["cases"]["case-01"]["traces"]["tg"] == {"s": [0, 1], "p": [1, 0]}  # svc-c absent from the trace
    # scored through the per-trace scorer's arm-only path: hit@1 by hand over C = {ta, tc, td, te} | {tf}:
    # ta svc-c 1, tc 0, td 1, te 0 -> 0.5; tf svc-a 0 -> 0.0; the mean over cases 0.25
    out_s = tmp_path / "earliest/score"
    assert (
        scoretrace.main(
            [
                "--corpus-dir",
                str(corpus_dir),
                "--rank-record",
                str(out_t),
                "--ranking-trace",
                str(out_t / RANKING_TRACE_JSON),
                "--broadcast-results-dir",
                str(tmp_path / "no-broadcast"),
                "--rows",
                "arm-only",
                "--bootstrap-b",
                "20",
                "--bootstrap-seed",
                "0",
                "--output-folder",
                str(out_s),
            ]
        )
        == 0
    )
    rows = read_json(out_s / SCORE_TRACE_JSON)["rows"]
    assert list(rows) == [f"{ARM_CONTROL_TRACE_EARLIEST}/{TRACE_READING_EARLIEST}"]
    assert rows[f"{ARM_CONTROL_TRACE_EARLIEST}/{TRACE_READING_EARLIEST}"]["metrics"]["s@1"] == pytest.approx(0.25)
    sres = read_json(out_s / RUN_DIR / RESULTS_JSON)
    assert sres["rows_mode"] == "arm-only" and sres["primary_reading"] == TRACE_READING_EARLIEST


# --- the case-level report: series and the family's hypotheses by hand ----------------------------------------------


def _row(rung, arm, avg, reading=None, subtract="none", low=None, random=0.3):
    r = {
        "rung": rung,
        "arm": arm,
        "metrics": {"avg@5": avg, "ac@1": avg, "ac@3": avg, "ac@5": avg, "n_cases": 10},
        "bootstrap": {"intervals": {"avg@5": {"low": avg - 0.1 if low is None else low, "high": avg + 0.1}}},
        "random": {"avg@5": random},
        "n_missing": 0,
    }
    if reading is not None:
        r["reading"], r["subtract"] = reading, subtract
    return r


def test_report_series_and_context_hypotheses_by_hand():
    assert report.series_of(ARM_TRACE) == (None, "hop0", "source")
    assert report.series_of(ARM_TRACE_IG_OVERWRITE) == (
        "hop1",
        "hop0",
        "source",
        "hop1-subpre",
        "hop0-subpre",
        "source-subpre",
    )
    assert report.reading_of(None) == ("hop1", None) and report.reading_of("hop0-subpre") == ("hop0", "pre")
    assert TRACE_SERIES[ARM_TRACE_ATOMIC_CMI_KEEP] == report.series_of(ARM_TRACE_ATOMIC_CMI_KEEP)
    assert ARM_CONTROL_TRACE_EARLIEST in report.REPORT_ARMS and all(
        a in report.PER_ARM_HYPOTHESES for a in CONTEXT_TRACE_ARMS
    )
    g = "xs"
    rows = [
        _row(g, ARM_CONTROL_RULE, 0.5),
        _row(g, ARM_RCD, 0.6),
        _row(g, ARM_TRACE, 0.4),  # the landed row = hop1
        _row(g, ARM_TRACE, 0.55, "hop0"),
        _row(g, ARM_TRACE, 0.45, "source"),
        _row(g, ARM_TRACE_IG_OVERWRITE, 0.5, "hop1"),
        _row(g, ARM_TRACE_IG_OVERWRITE, 0.7, "hop0", low=0.65),
        _row(g, ARM_TRACE_IG_OVERWRITE, 0.52, "source"),
        _row(g, ARM_TRACE_IG_OVERWRITE, 0.75, "hop0", "pre"),
        _row(g, ARM_TRACE_ATOMIC_CMI_KEEP, 0.5, "hop0"),
    ]
    hyp = report.hypotheses(rows, (g,))
    assert hyp["H-rca-hop0"]["verdict"] == "pass" and hyp["H-rca-hop0"]["per_rung_and_arm"] == {
        f"{g} {ARM_TRACE}": True,
        f"{g} {ARM_TRACE_IG_OVERWRITE}": True,
    }
    assert hyp["H-rca-ctx-rule"]["verdict"] == "pass" and hyp["H-rca-ctx-rule"]["gap"] == {
        f"{g} {ARM_TRACE_IG_OVERWRITE}": pytest.approx(0.2),
        f"{g} {ARM_TRACE_ATOMIC_CMI_KEEP}": pytest.approx(0.0),
    }
    assert hyp["H-rca-ctx-rcd"]["verdict"] == "fail"  # the atomic arm's 0.5 < rcd's 0.6
    assert hyp["H-rca-source"]["verdict"] == "pass" and hyp["H-rca-presub"]["verdict"] == "pass"
    assert hyp["H-rca-atomic-cmi"]["verdict"] == "fail"  # 0.5 < rca/trace's hop0 0.55
    # the landed per-arm hypotheses read the family's hop0 row and the landed arm's landed row
    assert hyp["H-rca-trace-ig-overwrite-random"]["verdict"] == "pass"  # low 0.65 > random 0.3
    assert hyp["H-rca-trace-ig-overwrite-rule"]["gap_arm_minus_rule"][g] == pytest.approx(0.2)
    assert hyp["H-rca-trace-rule"]["gap_arm_minus_rule"][g] == pytest.approx(-0.1)  # the landed hop1 row (0.4)
    assert report.hypotheses([_row(g, ARM_TRACE, 0.4)], (g,))["H-rca-hop0"]["verdict"] == "not run"


def test_hop_readings_and_series_constants():
    assert TRACE_READINGS == ("hop1", "hop0", "source")
    for arm in CONTEXT_TRACE_ARMS:
        assert TRACE_CELLS[arm].endswith("/fixed-kl/frozen/request")
    assert "parent_pos" not in Path(earliest.__file__).read_text(encoding="utf-8")  # method side, label-free
