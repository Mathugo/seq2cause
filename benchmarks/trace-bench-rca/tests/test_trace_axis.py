"""The per-trace root-cause axis (plan of 2026-10-31; contract-per-trace-axis.md H3, H5, H8): the runner's source
never names the labels or the call tree; `ranktrace --axis trace` writes the compact `ranking-trace.json` (indices
into the export's services, reached only, both readings) and a `rank-trace` record; `--axis ranking` is the landed
stage with the selection and cap recorded; the GPU/CPU tolerance gate compares edge sets and refuses differing
positions."""

import json
from pathlib import Path

import pytest
from fixture_corpus import SERVICES, fixture_corpus
from test_rank_score import _export_record
from test_trace import _fake_runner, _ranktrace, pos

from tracebenchrca import cases, tolerance
from tracebenchrca import ranktrace as rt
from tracebenchrca.constants import (
    ARGUMENTS_JSON,
    ARM_TRACE,
    ARM_TRACE_CORE,
    RANKING_TRACE_JSON,
    RESULTS_JSON,
    RUN_DIR,
    STAGE_RANK_TRACE,
    TRACE_AXIS_CAP,
    TRACE_CELLS,
    TRACE_MAX_LEN,
    TRACE_READINGS,
    TRACE_SELECT_POST_NONOK,
    TRACE_SELECT_WINDOW,
    TRACE_TOLERANCE_JACCARD,
    TRACE_TRACE_CAP,
)
from tracebenchrca.record import read_json, sha256_file

NAMES = [s for s, _ in SERVICES]  # tb-edge, tb-beta, ext-gamma (the fixture export's services.json order)
RUNNER_SRC = Path(rt.__file__).with_name("trace_runner.py")


def test_runner_source_never_names_labels_or_parent_pos():
    """The method-side runner is label-free and call-tree-free by construction: rule R reads the view's raw outcomes,
    offsets and durations only (the labels parquet and the view's parent column belong to the score side)."""
    text = RUNNER_SRC.read_text(encoding="utf-8")
    assert "sequence-labels" not in text
    assert "parent_pos" not in text
    assert '"offsets", "durations"' in text and "post-nonok" in text


def _harness(tmp_path):
    harness_root = tmp_path / "harness"
    for rel in rt.HARNESS_SRC:
        (harness_root / rel).mkdir(parents=True)
    return harness_root


TRACE_DOCS = {
    "case-00": {
        "traces": [
            {"trace_id": "t-a", "positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": [[0, 1, 0.7]]},
            {
                "trace_id": "t-b",
                "positions": [pos("ext-gamma"), pos("tb-beta", "slow"), pos("tb-edge", "5xx")],
                "edges": [[0, 1, 0.2], [1, 2, 0.9]],
            },
        ],
        "skipped": ["t-skip"],
        "n_nonok_folded": 4,
    },
    "case-01": {"traces": [{"trace_id": "t-c", "positions": [pos("tb-beta"), pos("tb-edge")], "edges": [[0, 1, 0.7]]}]},
}


def test_ranktrace_trace_axis_writes_compact_file_and_record(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    corpus = fixture_corpus()
    harness_root = _harness(tmp_path)
    seen = []
    monkeypatch.setattr(rt, "_spawn", _fake_runner(TRACE_DOCS, seen))
    monkeypatch.setattr(rt, "pinned_environment", lambda python: ["stand-in==0"])

    def boom(*_a, **_k):
        raise AssertionError("rca/trace must never open data.csv")

    monkeypatch.setattr(cases, "read_table", boom)
    out = tmp_path / "trace/rank-trace"
    assert (
        _ranktrace(
            ARM_TRACE, rec, corpus, out, tmp_path, harness_root, axis="trace", select=TRACE_SELECT_POST_NONOK, cap=0
        )
        == 0
    )
    (call,) = seen
    cmd = call["cmd"]
    assert cmd[cmd.index("--select") + 1] == TRACE_SELECT_POST_NONOK and cmd[cmd.index("--cap") + 1] == "0"
    assert cmd[cmd.index("--cell") + 1] == TRACE_CELLS[ARM_TRACE] and TRACE_AXIS_CAP == 0
    # the compact prediction file: header, services, per case the selected / skipped ids and reached-only entries
    ranking = read_json(out / RANKING_TRACE_JSON)
    assert set(ranking) == {"schema", "arm", "cell", "c", "max_len", "select", "cap", "device", "services", "cases"}
    assert ranking["schema"] == "ranking-trace@1" and ranking["arm"] == ARM_TRACE and ranking["c"] == 1
    assert ranking["cell"] == TRACE_CELLS[ARM_TRACE] and ranking["max_len"] == TRACE_MAX_LEN
    assert ranking["select"] == TRACE_SELECT_POST_NONOK and ranking["cap"] == 0 and ranking["device"] == "cpu"
    assert ranking["services"] == NAMES
    assert list(ranking["cases"]) == ["case-00", "case-01"]
    c0 = ranking["cases"]["case-00"]
    assert c0["selected"] == ["t-a", "t-b", "t-skip"] and c0["skipped"] == ["t-skip"]
    assert set(c0["traces"]) == {"t-a", "t-b"}  # the skipped trace has no entry
    # t-a: tb-beta (index 1) upstream of the failed tb-edge at hop 1; hop 0 adds tb-edge (index 0) first (hop 0 < 1)
    assert c0["traces"]["t-a"] == {"s": [1], "p": [0], "s0": [0, 1], "p0": [1, 0]}
    # t-b: ext-gamma (2) upstream of both failures (count 2), tb-beta (1) of one; hop 0 lifts tb-beta to (2, hop 0)
    assert c0["traces"]["t-b"] == {"s": [2, 1], "p": [0, 1], "s0": [1, 2, 0], "p0": [1, 0, 2]}
    for entry in c0["traces"].values():
        assert set(entry["s"]) <= set(entry["s0"]) and set(entry["p"]) <= set(entry["p0"])
        assert all(0 <= i < len(NAMES) for i in entry["s"] + entry["s0"])
    assert ranking["cases"]["case-01"]["traces"] == {"t-c": {"s": [], "p": [], "s0": [], "p0": []}}  # nothing failed
    # the record
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    assert res["stage"] == STAGE_RANK_TRACE and res["axis"] == "trace" and res["arm"] == ARM_TRACE
    assert res["select"] == TRACE_SELECT_POST_NONOK and res["cap"] == 0
    assert res["ranking_file"] == RANKING_TRACE_JSON and res["ranking_sha256"] == sha256_file(out / RANKING_TRACE_JSON)
    assert res["staging"] == {"rung": "xs", "previous_rung": None, "form": "smallest-rung"}
    assert res["readings"] == list(TRACE_READINGS) and res["services"] == NAMES
    assert res["runner"]["select"] == TRACE_SELECT_POST_NONOK and res["code"]["cell"] == TRACE_CELLS[ARM_TRACE]
    assert read_json(out / RUN_DIR / ARGUMENTS_JSON)["command"] == STAGE_RANK_TRACE
    row0, row1 = res["cases"]
    for key in (
        "n_traces_window",
        "n_traces_candidates",
        "n_traces_probed",
        "n_skipped_short",
        "n_truncated",
        "n_folded",
        "n_nonok_folded",
        "n_nonok_positions",
        "n_traces_with_nonok",
        "n_traces_with_upstream",
        "n_client_positions_dropped",
        "mean_set_size_hop1",
        "mean_set_size_hop0",
        "tok_pos",
        "probe_wall_clock_s",
    ):
        assert key in row0, key
    assert row0["n_traces_candidates"] == 3 and row0["n_traces_probed"] == 2 and row0["n_skipped_short"] == 1
    assert row0["n_nonok_folded"] == 4 and row0["n_nonok_positions"] == 3
    assert row0["n_traces_with_nonok"] == 2 and row0["n_traces_with_upstream"] == 2
    assert row0["mean_set_size_hop1"] == pytest.approx(1.5) and row0["mean_set_size_hop0"] == pytest.approx(2.5)
    assert row1["n_traces_with_nonok"] == 0 and row1["mean_set_size_hop1"] == 0.0 and row1["mean_set_size_hop0"] == 0.0
    assert res["n_traces_candidates"] == 4 and res["n_traces_probed"] == 3 and res["n_skipped_short"] == 1
    assert res["n_nonok_folded"] == 4 and "n_unranked_cases" not in res  # no case-level ranking on this axis
    assert not (out / rt.POSITIONAL_JSON).exists()


LANDED_RANKING_KEYS = {  # the ranking axis's record keys as landed on 2026-10-30, plus the record's status
    "status",
    "stage",
    "arm",
    "code",
    "rung",
    "seed",
    "corpus_id",
    "export_record_sha256",
    "staging",
    "ranking_file",
    "ranking_sha256",
    "freeze",
    "model_dir",
    "device",
    "cap",
    "max_len",
    "nonok_outcomes",
    "runner",
    "pinned_environment",
    "n_cases",
    "n_unranked_cases",
    "misses_by_reason",
    "n_empty_rankings",
    "n_filled_cells",
    "n_metrics_lt_min_pre",
    "n_traces_window",
    "n_traces_probed",
    "wall_clock_s",
    "cases",
}


def test_ranktrace_ranking_axis_unchanged(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    corpus = fixture_corpus()
    harness_root = _harness(tmp_path)
    seen = []
    monkeypatch.setattr(rt, "_spawn", _fake_runner(TRACE_DOCS, seen))
    monkeypatch.setattr(rt, "pinned_environment", lambda python: ["stand-in==0"])
    out = tmp_path / "trace-core/rank"
    assert (
        _ranktrace(
            ARM_TRACE_CORE,
            rec,
            corpus,
            out,
            tmp_path,
            harness_root,
            axis="ranking",
            select=TRACE_SELECT_WINDOW,
            cap=500,
        )
        == 0
    )
    (call,) = seen
    cmd = call["cmd"]
    assert cmd[cmd.index("--select") + 1] == TRACE_SELECT_WINDOW and cmd[cmd.index("--cap") + 1] == str(TRACE_TRACE_CAP)
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    assert set(res) == LANDED_RANKING_KEYS | {"axis", "select"}
    assert res["stage"] == "rank" and res["axis"] == "ranking" and res["select"] == TRACE_SELECT_WINDOW
    assert res["cap"] == TRACE_TRACE_CAP and res["ranking_file"] == rt.POSITIONAL_JSON
    assert read_json(out / RUN_DIR / ARGUMENTS_JSON)["command"] == "rank"
    posit = read_json(out / rt.POSITIONAL_JSON)["cases"]
    assert list(posit) == ["case-00"] and posit["case-00"] == ["tb-beta", "ext-gamma", "tb-edge"]
    assert res["n_unranked_cases"] == 1 and res["cases"][0]["n_traces_probed"] == 2
    assert set(res["cases"][0]) >= {
        "n_traces_window",
        "n_traces_selected",
        "counts",
        "mean_hops",
        "self_counts",
        "miss",
    }
    assert not (out / RANKING_TRACE_JSON).exists()
    with pytest.raises(SystemExit):  # the three flags are required
        rt.main(["--arm", ARM_TRACE, "--rung", "xs"])


def _work(root, docs):
    root.mkdir(parents=True, exist_ok=True)
    for case_dir, traces in docs.items():
        (root / f"{case_dir}.json").write_text(json.dumps({"case_dir": case_dir, "traces": traces}))
    return root


def _services_file(tmp_path):
    path = tmp_path / "services.json"
    path.write_text(json.dumps({"services": [{"kind": k, "name": s} for s, k in SERVICES], "source": "fixture"}))
    return path


def test_tolerance_gate_and_identity(tmp_path):
    t1 = {"trace_id": "t1", "positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": [[0, 1, 0.9]]}
    t2 = {
        "trace_id": "t2",
        "positions": [pos("ext-gamma"), pos("tb-beta"), pos("tb-edge", "5xx")],
        "edges": [[0, 2, 0.5], [1, 2, 0.6]],
    }
    t3 = {"trace_id": "t3", "positions": [pos("tb-beta"), pos("ext-gamma", "err")], "edges": []}
    gpu = _work(tmp_path / "gpu", {"case-00": [t1, t2], "case-01": [t3]})
    services = _services_file(tmp_path)

    def run(cpu, out, threshold=TRACE_TOLERANCE_JACCARD):
        argv = ["--gpu-work", str(gpu), "--cpu-work", str(cpu), "--services", str(services)]
        argv += ["--threshold", str(threshold), "--output-folder", str(out)]
        assert tolerance.main(argv) == 0
        return read_json(out / RUN_DIR / RESULTS_JSON)

    # identical sides: every Jaccard 1.0 (the empty-vs-empty pair counts as agreement), the gate passes
    same = _work(tmp_path / "cpu-same", {"case-00": [t1, t2], "case-01": [t3]})
    res = run(same, tmp_path / "out-same")
    assert res["stage"] == "tolerance" and res["mean_jaccard"] == 1.0 and res["n_traces_compared"] == 3
    assert res["gate"] == {"threshold": TRACE_TOLERANCE_JACCARD, "passed": True}
    assert [c["share_identical"] for c in res["per_case"]] == [1.0, 1.0]
    assert res["per_case"][0]["top1_gpu"] == res["per_case"][0]["top1_cpu"] == "tb-beta"
    assert res["per_case"][1]["top1_gpu"] is None  # no upstream service: the rule's counted miss
    assert res["per_case"][0]["n_traces"] == 2 and res["per_case"][0]["n_common"] == 2
    # the CPU cut loses one of t2's two edges: Jaccard 1/2 on t2, the case mean 0.75, overall (1 + 0.5 + 1) / 3
    diff = _work(tmp_path / "cpu-diff", {"case-00": [t1, dict(t2, edges=[[1, 2, 0.6]])], "case-01": [t3]})
    res = run(diff, tmp_path / "out-diff")
    c0 = res["per_case"][0]
    assert c0["mean_jaccard"] == pytest.approx(0.75) and c0["share_identical"] == 0.5
    assert c0["p10_jaccard"] == pytest.approx(0.55) and c0["top1_gpu"] == c0["top1_cpu"] == "tb-beta"
    assert res["mean_jaccard"] == pytest.approx(2.5 / 3) and res["gate"]["passed"] is True
    res = run(diff, tmp_path / "out-strict", threshold=0.9)
    assert res["gate"] == {"threshold": 0.9, "passed": False} and res["status"] == "ok"  # reported, exit 0
    # a trace whose positions differ between the sides is not a tolerance question
    other = _work(
        tmp_path / "cpu-pos",
        {"case-00": [t1, dict(t2, positions=[pos("ext-gamma"), pos("tb-edge"), pos("tb-edge", "5xx")])]},
    )
    with pytest.raises(ValueError):
        run(other, tmp_path / "out-pos")
    assert read_json(tmp_path / "out-pos" / RUN_DIR / RESULTS_JSON)["status"] == "failed"
