"""The ranking stages: the method side reads only its declared inputs through the case door and
keys its rankings by case directory; the score side restores the case ids by position, scores with
the benchmark's scorer and computes change-point offsets on the export's clock; a missing change
point and a change point after the last row are counted misses, never a fallback; staging and the
record bindings refuse a mismatched read."""

import json

import pytest
from fixture_corpus import EPOCH, SERVICES, WIDTH, fixture_corpus, fixture_export

from tracebenchrca import rank as rank_mod
from tracebenchrca import report as report_mod
from tracebenchrca import score as score_mod
from tracebenchrca.constants import RESULTS_JSON, RUN_DIR, SCORE_RCA_JSON
from tracebenchrca.record import read_json, write_json
from tracebenchrca.staging import StagingRefusal

LEDGER = "unused-at-the-smallest-rung.json"
NAMES = [s for s, _ in SERVICES]


def _export_record(tmp_path, seed=0, rung="xs"):
    exp = tmp_path / f"exports/{rung}-s{seed}"
    fixture_export(exp)
    rec = tmp_path / f"export-record-s{seed}"
    write_json(
        rec / RUN_DIR / RESULTS_JSON,
        {
            "status": "ok",
            "stage": "export",
            "rung": rung,
            "corpus_id": f"xs/latent/seed={seed}",
            "export_dir": str(exp),
        },
    )
    return rec, exp


def _rank(arm, export_rec, out, rung="xs", prior="smallest-rung"):
    return rank_mod.main(
        [
            "--arm",
            arm,
            "--export-record",
            str(export_rec),
            "--rung",
            rung,
            "--prior-rung-record",
            prior,
            "--staging-ledger",
            LEDGER,
            "--output-folder",
            str(out),
        ]
    )


def _score(corpus, rank_out, out):
    return score_mod.main(
        [
            "--corpus-dir",
            str(corpus),
            "--rank-record",
            str(rank_out),
            "--bootstrap-b",
            "50",
            "--bootstrap-seed",
            "0",
            "--output-folder",
            str(out),
        ]
    )


def _stub_baro(monkeypatch, first):
    """`baro()` stand-in: ranks `time` first (as the pinned path can), then the case's named service's metrics."""

    def baro(data, inject_time=None, dataset=None):
        assert dataset is None and inject_time is not None
        cols = [c for c in data.columns if c != "time"]
        top = first[int(inject_time)]
        return {"ranks": ["time"] + sorted(cols, key=lambda c: (not c.startswith(top), c))}

    monkeypatch.setattr(rank_mod, "_baro", lambda: baro)


def test_real_baro_ranks_every_service_and_drops_time(tmp_path):
    pytest.importorskip("RCAEval.e2e.baro")
    rec, _ = _export_record(tmp_path)
    out = tmp_path / "rca-baro/rank"
    assert _rank("rca/baro", rec, out) == 0
    pos = read_json(out / rank_mod.POSITIONAL_JSON)["cases"]
    assert sorted(pos) == ["case-00", "case-01"]
    for ranking in pos.values():
        assert sorted(ranking) == sorted(NAMES) and "time" not in ranking
    assert pos["case-00"][0] == "tb-beta" and pos["case-01"][0] == "ext-gamma"
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    assert res["n_filled_cells"] == 2 and res["n_metrics_lt_min_pre"] == 0
    assert all(c["time_rank"] is not None and c["n_services_unranked"] == 0 for c in res["cases"])


def test_full_pipeline_never_opens_the_window(tmp_path, monkeypatch):
    rec, exp = _export_record(tmp_path)
    for d in exp.glob("case-*"):
        (d / "window.json").unlink()
    monkeypatch.setattr(rank_mod, "_bocpd", lambda: lambda data: [len(data) - 5])
    monkeypatch.setattr(rank_mod, "_robust_scorer", lambda: lambda data, anomalies: {"ranks": list(data.columns)})
    assert _rank("rca/baro-bocpd", rec, tmp_path / "bocpd/rank") == 0
    with pytest.raises(FileNotFoundError):
        _rank("rca/baro", rec, tmp_path / "baro/rank")


def test_no_change_point_and_a_change_point_after_the_last_row_are_counted_misses(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    calls = iter([[], [40]])  # case-00: none; case-01 (40 rows): the run-length array's extra entry
    monkeypatch.setattr(rank_mod, "_bocpd", lambda: lambda data: next(calls))

    def never(*a, **k):
        raise AssertionError("the scorer must not run without a change point inside the table")

    monkeypatch.setattr(rank_mod, "_robust_scorer", lambda: never)
    out = tmp_path / "bocpd/rank"
    assert _rank("rca/baro-bocpd", rec, out) == 0
    assert read_json(out / rank_mod.POSITIONAL_JSON)["cases"] == {}
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    assert res["misses_by_reason"] == {"change point at the table's end": 1, "no change point": 1}
    assert res["n_unranked_cases"] == 2


def test_score_restores_case_ids_by_position_and_offsets_on_the_export_clock(tmp_path, monkeypatch):
    corpus = fixture_corpus(seed=0)
    rec, _ = _export_record(tmp_path)
    monkeypatch.setattr(rank_mod, "_bocpd", lambda: lambda data: [45] if len(data) == 60 else [])
    monkeypatch.setattr(
        rank_mod,
        "_robust_scorer",
        lambda: lambda data, anomalies: {"ranks": ["time", "tb-beta_latency-95", "tb-edge_count"]},
    )
    _rank("rca/baro-bocpd", rec, tmp_path / "arm/rank")
    assert _score(corpus, tmp_path / "arm/rank", tmp_path / "arm") == 0
    ranking = read_json(tmp_path / "arm/ranking.json")["cases"]
    assert list(ranking) == ["xs_tb-beta_degrade_0"]  # case-00 = the earliest injection; case-01 unranked
    scored = read_json(tmp_path / "arm" / SCORE_RCA_JSON)
    assert [(c["case_id"], c["rank"]) for c in scored["cases"]] == [
        ("xs_tb-beta_degrade_0", 1),
        ("xs_ext-gamma_crash_1", None),
    ]
    res = read_json(tmp_path / "arm" / RUN_DIR / RESULTS_JSON)
    assert res["random"]["n_candidates"] == len(SERVICES)
    cp = {c["case_id"]: c for c in res["change_points"]}
    # row 45 of case-00 starts at EPOCH + 45 * WIDTH; the label injects at 1200 s after the corpus start
    assert cp["xs_tb-beta_degrade_0"]["change_point_offset_s"] == 45 * WIDTH - 1200
    assert cp["xs_ext-gamma_crash_1"]["miss"] == "no change point"
    assert EPOCH > 0


def test_score_refuses_another_corpus_and_a_changed_ranking(tmp_path, monkeypatch):
    _stub_baro(monkeypatch, {EPOCH + 40 * WIDTH: "tb-beta", EPOCH + 90 * WIDTH: "ext-gamma"})
    rec, _ = _export_record(tmp_path)
    _rank("rca/baro", rec, tmp_path / "arm/rank")
    with pytest.raises(ValueError, match="this corpus is"):
        _score(fixture_corpus(seed=1), tmp_path / "arm/rank", tmp_path / "arm-other")
    p = tmp_path / "arm/rank" / rank_mod.POSITIONAL_JSON
    p.write_text(json.dumps({"cases": {"case-00": NAMES[::-1]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the ranking"):
        _score(fixture_corpus(seed=0), tmp_path / "arm/rank", tmp_path / "arm-tampered")


def test_staging_and_the_export_binding(tmp_path, monkeypatch):
    _stub_baro(monkeypatch, {EPOCH + 40 * WIDTH: "tb-beta", EPOCH + 90 * WIDTH: "ext-gamma"})
    rec, _ = _export_record(tmp_path)
    with pytest.raises(ValueError, match="rung 'xs' export"):
        _rank("rca/baro", rec, tmp_path / "a/rank", rung="s", prior=str(tmp_path / "none"))
    rec_s, _ = _export_record(tmp_path, seed=2, rung="s")
    with pytest.raises(StagingRefusal):
        _rank("rca/baro", rec_s, tmp_path / "b/rank", rung="s", prior="smallest-rung")
    monkeypatch.setattr(rank_mod, "_bocpd", lambda: lambda data: [])
    _rank("rca/baro-bocpd", rec, tmp_path / "xs-bocpd/rank")
    with pytest.raises(StagingRefusal, match="this read is 'rca/baro'"):
        _rank("rca/baro", rec_s, tmp_path / "c/rank", rung="s", prior=str(tmp_path / "xs-bocpd/rank"))
    _rank("rca/baro", rec, tmp_path / "xs-baro/rank")
    assert _rank("rca/baro", rec_s, tmp_path / "d/rank", rung="s", prior=str(tmp_path / "xs-baro/rank")) == 0
    staging = read_json(tmp_path / "d/rank" / RUN_DIR / RESULTS_JSON)["staging"]
    assert staging["form"] == "record" and staging["previous_rung"] == "xs"


def test_report_pools_five_seeds_and_scores_the_hypotheses(tmp_path, monkeypatch):
    _stub_baro(monkeypatch, {EPOCH + 40 * WIDTH: "tb-beta", EPOCH + 90 * WIDTH: "ext-gamma"})
    monkeypatch.setattr(rank_mod, "_bocpd", lambda: lambda data: [])
    results = tmp_path / "results"
    for k in range(5):
        rec, _ = _export_record(tmp_path, seed=k)
        for arm, slug in (("rca/baro", "rca-baro"), ("rca/baro-bocpd", "rca-baro-bocpd")):
            cell = results / f"xs/latent/seed={k}/{slug}"
            _rank(arm, rec, cell / "rank")
            _score(fixture_corpus(seed=k), cell / "rank", cell)
    out = tmp_path / "tables"
    argv = [
        "--results-dir",
        str(results),
        "--rungs",
        "xs",
        "--arms",
        "rca/baro",
        "rca/baro-bocpd",
        "--bootstrap-b",
        "50",
        "--bootstrap-seed",
        "0",
        "--output-folder",
        str(out),
    ]
    with pytest.raises(SystemExit):
        report_mod.main(argv[:4] + ["--seeds", "0", "1"] + argv[4:])
    assert report_mod.main(argv[:4] + ["--seeds", "0", "1", "2", "3", "4"] + argv[4:]) == 0
    summary = read_json(out / "summary.json")
    baro = next(r for r in summary["rows"] if r["arm"] == "rca/baro")
    assert baro["metrics"]["n_cases"] == 10 and baro["metrics"]["ac@1"] == 1.0
    bocpd = next(r for r in summary["rows"] if r["arm"] == "rca/baro-bocpd")
    assert bocpd["n_missing"] == 10 and bocpd["misses_by_reason"] == {"no change point": 10}
    hyp = summary["hypotheses"]
    assert hyp["H-rca-random"]["verdict"] == "pass" and hyp["H-rca-bocpd"]["verdict"] == "fail"
    assert hyp["H-rca-rule"]["verdict"] == "not run" and "H-rca-scale" not in hyp
    assert (out / "rca-latent.md").read_text(encoding="utf-8").startswith("| rung | arm |")


def test_an_empty_ranking_from_the_pinned_scorer_is_passed_through_and_counted(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    monkeypatch.setattr(rank_mod, "_bocpd", lambda: lambda data: [len(data) - 1])
    monkeypatch.setattr(rank_mod, "_robust_scorer", lambda: lambda data, anomalies: {"ranks": []})
    out = tmp_path / "bocpd/rank"
    assert _rank("rca/baro-bocpd", rec, out) == 0
    assert read_json(out / rank_mod.POSITIONAL_JSON)["cases"] == {"case-00": [], "case-01": []}
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    assert res["n_empty_rankings"] == 2 and res["misses_by_reason"] == {}
