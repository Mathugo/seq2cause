"""RCD's adapter (`plans/rca-arms.md`, addendum 2026-10-25, later; D-TBR-6, D-TBR-7): the method side writes
the filled table once per case and runs the pinned environment's runner ten times with seed = run index;
each run's metric ranking is collapsed to services, a failed run is counted and never refilled, services
absent from a run are misses for that run; the score side averages per-run hits and reproduces the
benchmark's bootstrap when the runs agree; the report pools it run-averaged and scores H-rca-rcd-*. With
the pinned environment present, the real runner ranks the fixture export through the suite's own
named-dataset preprocessing."""

import json
import subprocess
from pathlib import Path

import pytest
from fixture_corpus import SERVICES, fixture_corpus
from test_rank_score import _export_record
from tracebench.score_rca import bootstrap, case_ranks, metrics_of

from tracebenchrca import rankrcd, report, scorercd
from tracebenchrca.constants import (
    ARM_CONTROL_RULE,
    ARM_RCD,
    RCD_ENV,
    RCD_OVERLAY_SHA256,
    RCD_RUNS,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_RCA_JSON,
)
from tracebenchrca.record import read_json

NAMES = [s for s, _ in SERVICES]
PINNED_PYTHON = Path(f"/opt/miniconda3/envs/{RCD_ENV}/bin/python")
LEDGER = "unused-at-the-smallest-rung.json"
REGISTERED_OVERLAY = dict(
    RCD_OVERLAY_SHA256
)  # frozen at import: the fake runner reports these even when a test mutates the registry


def _fake_runner(tmp_path, monkeypatch, plan):
    """Stand-in for the pinned environment: `plan[seed]` is a metric ranking (list) or "fail"; it is written where
    the real runner would write it, with the registered overlay digests."""
    calls = []

    def run(cmd):
        args = dict(zip(cmd[2::2], cmd[3::2], strict=True))
        seed = int(args["--seed"])
        calls.append((Path(args["--input"]), seed, int(args["--inject-time"])))
        assert Path(args["--input"]).exists() and cmd[1].endswith("rcd_runner.py")
        if plan[seed] == "fail":
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        doc = {
            "ranks": plan[seed],
            "n_columns_in": 12,
            "n_columns_kept": 11,
            "wall_clock_s": 0.1,
            "overlay_sha256": {k: v + "0" * 48 for k, v in REGISTERED_OVERLAY.items()},
        }
        Path(args["--out"]).write_text(json.dumps(doc))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(rankrcd, "_spawn", run)
    monkeypatch.setattr(rankrcd, "pinned_environment", lambda python: "fake==0")
    return calls


def _rank(rec, out, work, rung="xs", prior="smallest-rung"):
    return rankrcd.main(
        [
            "--export-record", str(rec), "--rung", rung, "--rcd-python", "/fake/python", "--workers", "2",
            "--work-dir", str(work), "--prior-rung-record", prior, "--staging-ledger", LEDGER, "--output-folder", str(out),
        ]
    )  # fmt: skip


def test_rank_runs_ten_seeds_per_case_collapses_and_counts(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    plan = {r: ["tb-beta_latency-95", "tb-beta_count", "ext-gamma_error"] for r in range(RCD_RUNS)}
    plan[3] = "fail"
    plan[4] = ["tb-edge_error"]  # a partial ranking: two services unranked in this run
    calls = _fake_runner(tmp_path, monkeypatch, plan)
    out = tmp_path / "rca-rcd/rank"
    assert _rank(rec, out, tmp_path / "work") == 0
    assert len(calls) == 2 * RCD_RUNS and sorted({s for _, s, _ in calls}) == list(range(RCD_RUNS))
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rankrcd.POSITIONAL_JSON)["cases"]
    assert sorted(pos) == ["case-00", "case-01"] and res["arm"] == ARM_RCD and res["n_runs_per_case"] == RCD_RUNS
    for c in pos.values():
        assert c["services"] == NAMES and len(c["rankings"]) == RCD_RUNS
        assert (
            c["rankings"][3] is None
            and c["rankings"][4] == ["tb-edge"]
            and c["rankings"][0] == ["tb-beta", "ext-gamma"]
        )
    assert (
        res["n_failed_runs"] == 2
        and res["failures_by_reason"] == {"exit code 1: boom": 2}
        and res["n_unranked_cases"] == 0
    )
    assert (
        res["cases"][0]["n_services_unranked_per_run"] == [1] * 3 + [2] + [1] * 5
    )  # successful runs only (run 3 failed)
    # the method side wrote the whole filled table (time included): preprocessing is the suite's, inside the runner
    header = (tmp_path / "work" / "case-00" / rankrcd.INPUT_CSV).read_text().splitlines()[0].split(",")
    assert header[0] == "time" and len(header) == 1 + 4 * len(NAMES)
    assert res["code"]["dataset"] == "trace-bench" and res["code"]["kwargs"] == {
        "gamma": 5,
        "localized": True,
        "bins": 5,
        "dk_select_useful": False,
    }


def test_rank_refuses_an_environment_whose_overlay_differs(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    plan = {r: ["tb-beta_count"] for r in range(RCD_RUNS)}
    _fake_runner(tmp_path, monkeypatch, plan)
    monkeypatch.setitem(rankrcd.RCD_OVERLAY_SHA256, "utils/Fas.py", "deadbeefdeadbeef")
    with pytest.raises(RuntimeError):
        _rank(rec, tmp_path / "o", tmp_path / "w")


def test_score_averages_runs_and_matches_the_benchmark_when_runs_agree(tmp_path, monkeypatch):
    corpus = fixture_corpus()
    rec, _ = _export_record(tmp_path)
    plan = {r: ["tb-beta_latency-95", "ext-gamma_error", "tb-edge_count"] for r in range(RCD_RUNS)}
    _fake_runner(tmp_path, monkeypatch, plan)
    assert _rank(rec, tmp_path / "rca-rcd/rank", tmp_path / "work") == 0
    assert scorercd.main(
        ["--corpus-dir", str(corpus), "--rank-record", str(tmp_path / "rca-rcd/rank"), "--bootstrap-b", "50",
         "--bootstrap-seed", "0", "--output-folder", str(tmp_path / "rca-rcd")]
    ) == 0  # fmt: skip
    doc = read_json(tmp_path / "rca-rcd" / SCORE_RCA_JSON)
    cases = read_json(corpus / "labels" / "cases.json")["cases"]
    one = {c["case_id"]: ["tb-beta", "ext-gamma", "tb-edge"] for c in cases}
    rows = case_ranks(cases, {"cases": one})
    assert doc["metrics"] == pytest.approx(metrics_of([r["rank"] for r in rows]))
    # every run agrees -> the run-averaged bootstrap equals the benchmark's on the single ranking
    assert doc["bootstrap"]["intervals"] == bootstrap([r["rank"] for r in rows], 50, 0)["intervals"]
    res = read_json(tmp_path / "rca-rcd" / RUN_DIR / RESULTS_JSON)
    assert res["arm"] == ARM_RCD and res["n_failed_runs"] == 0 and "aggregation" in res


def test_report_pools_rcd_run_averaged_and_scores_its_hypotheses():
    assert ARM_RCD in report.RUN_AVERAGED_ARMS and ARM_RCD in report.REPORT_ARMS
    rows = [
        {"rung": "xs", "arm": ARM_RCD, "metrics": {"avg@5": 0.7, "n_cases": 10}, "bootstrap": {"intervals": {"avg@5": {"low": 0.65, "high": 0.8}}}, "random": {"avg@5": 0.6}, "n_missing": 0},
        {"rung": "xs", "arm": ARM_CONTROL_RULE, "metrics": {"avg@5": 0.75, "n_cases": 10}, "bootstrap": {"intervals": {"avg@5": {"low": 0.6, "high": 0.9}}}, "random": {"avg@5": 0.6}, "n_missing": 0},
    ]  # fmt: skip
    hyp = report.hypotheses(rows, ("xs",))
    assert hyp["H-rca-rcd-random"]["verdict"] == "pass" and hyp["H-rca-rcd-rule"]["verdict"] == "fail"


@pytest.mark.skipif(not PINNED_PYTHON.exists(), reason="the pinned RCD environment is not on this machine")
def test_real_runner_in_the_pinned_environment(tmp_path):
    rec, exp = _export_record(tmp_path)
    out = tmp_path / "rca-rcd/rank"
    assert rankrcd.main(
        ["--export-record", str(rec), "--rung", "xs", "--rcd-python", str(PINNED_PYTHON), "--workers", "4", "--work-dir",
         str(tmp_path / "work"), "--prior-rung-record", "smallest-rung", "--staging-ledger", LEDGER, "--output-folder", str(out)]
    ) == 0  # fmt: skip
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    pos = read_json(out / rankrcd.POSITIONAL_JSON)["cases"]
    assert res["n_failed_runs"] == 0 and sorted(pos) == ["case-00", "case-01"]
    for c in res["cases"]:
        for r in c["runs"]:
            assert (
                r["n_columns_kept"] <= r["n_columns_in"] == 4 * len(NAMES)
            )  # time dropped, constants dropped per segment
            assert all(r["overlay_sha256"][k].startswith(v) for k, v in RCD_OVERLAY_SHA256.items())
    assert all(set(x) <= set(NAMES) for c in pos.values() for x in c["rankings"] if x is not None)
    assert "causal-learn==0.1.2.3" in res["pinned_environment"]["pip_freeze"]
