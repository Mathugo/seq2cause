"""`rca/trace` and `rca/trace-core` (`plans/rca-arms.md`, addendum 2026-10-30; D-TBR-13): the rule on hand-built sample
graphs (parents, upstream closure, hops, the client dropped, ties, the never-upstream tail, the three miss reasons,
the hop-0 diagnostic), the stage with a stand-in runner (the declared inputs only — never `data.csv`; the runner's
command line carries the registered cap, length and cell; the record's shape scores through `tracebenchrca.score`),
and the report's per-arm hypotheses."""

import json
from pathlib import Path

import pytest
from fixture_corpus import SERVICES, fixture_corpus
from test_rank_score import LEDGER, _export_record, _score

from tracebenchrca import cases, report, tracerule
from tracebenchrca import ranktrace as rt
from tracebenchrca.constants import (
    ARM_CONTROL_RULE,
    ARM_TRACE,
    ARM_TRACE_CORE,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_RCA_JSON,
    TRACE_CELLS,
    TRACE_MAX_LEN,
    TRACE_MISS_NO_NONOK,
    TRACE_MISS_NO_TRACE,
    TRACE_MISS_NO_UPSTREAM,
    TRACE_TRACE_CAP,
)
from tracebenchrca.record import read_json

NAMES = [s for s, _ in SERVICES]  # tb-edge, tb-beta, ext-gamma


def pos(service, outcome="ok", is_client=False):
    return {"op": hash(service) % 97, "outcome": outcome, "service": service, "is_client": is_client}


def test_parents_and_upstream_closure_with_hops():
    # positions 0 ext-gamma(ok) -> 1 tb-beta(ok) -> 2 tb-edge(5xx); and 0 -> 2 directly
    parents = tracerule.parents_of([[0, 1, 0.5], [1, 2, 0.4], [0, 2, 0.1], [0, 1, 0.9]], 3)
    assert parents == [[], [0], [1, 0]]
    assert tracerule.upstream(parents, 2) == {1: 1, 0: 1}  # the direct edge wins: hop 1, not 2
    assert tracerule.upstream(parents, 0) == {}
    with pytest.raises(ValueError):
        tracerule.parents_of([[2, 1, 0.5]], 3)  # not backward in time


def test_rank_case_counts_traces_upstream_of_nonok_positions_and_breaks_ties_by_hop():
    chain = [pos("ext-gamma"), pos("tb-beta"), pos("tb-edge", "5xx")]
    t1 = {"positions": chain, "edges": [[0, 1, 1.0], [1, 2, 1.0]]}  # gamma at hop 2, beta at hop 1
    t2 = {"positions": chain, "edges": [[1, 2, 1.0]]}  # beta only
    t3 = {"positions": [pos("ext-gamma"), pos("tb-edge", "slow")], "edges": [[0, 1, 1.0]]}  # gamma at hop 1
    row = tracerule.rank_case([t1, t2, t3], NAMES)
    assert row["services"] == [
        "tb-beta",
        "ext-gamma",
        "tb-edge",
    ]  # 2 traces vs 2 traces: beta's mean hop 1.0 < gamma's 1.5
    assert row["counts"] == {"tb-beta": 2, "ext-gamma": 2} and row["mean_hops"]["ext-gamma"] == 1.5
    assert row["self_counts"] == {"tb-edge": 3} and row["miss"] is None
    assert row["n_nonok_positions"] == 3 and row["n_traces_with_nonok"] == 3 and row["n_traces_with_upstream"] == 3
    assert row["n_services_reached"] == 2 and row["n_edges"] == 4
    # the never-upstream tail keeps services.json order; the ranking always holds every deployed service
    t4 = {"positions": [pos("tb-beta"), pos("ext-gamma", "err")], "edges": [[0, 1, 1.0]]}
    assert tracerule.rank_case([t4], NAMES)["services"] == ["tb-beta", "tb-edge", "ext-gamma"]


def test_rank_case_drops_the_client_and_counts_the_three_miss_reasons():
    client = {
        "positions": [pos("client", is_client=True), pos("tb-beta"), pos("tb-edge", "5xx")],
        "edges": [[0, 2, 1.0], [1, 2, 1.0]],
    }
    row = tracerule.rank_case([client], NAMES)
    assert row["services"][0] == "tb-beta" and row["n_client_positions_dropped"] == 1 and "client" not in row["counts"]
    assert tracerule.rank_case([], NAMES)["miss"] == TRACE_MISS_NO_TRACE
    ok_only = {"positions": [pos("tb-beta"), pos("tb-edge")], "edges": [[0, 1, 1.0]]}
    assert tracerule.rank_case([ok_only], NAMES)["miss"] == TRACE_MISS_NO_NONOK
    no_parent = {"positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": []}
    row = tracerule.rank_case([no_parent], NAMES)
    assert row["miss"] == TRACE_MISS_NO_UPSTREAM and row["self_counts"] == {"tb-edge": 1} and row["services"] is None
    # a client-only non-OK position still counts as a non-OK position; its own service never enters self_counts
    client_nonok = {"positions": [pos("tb-beta"), pos("client", "5xx", True)], "edges": [[0, 1, 1.0]]}
    row = tracerule.rank_case([client_nonok], NAMES)
    assert row["services"][0] == "tb-beta" and row["self_counts"] == {}
    with pytest.raises(ValueError):
        tracerule.rank_case([{"positions": [pos("elsewhere"), pos("tb-edge", "5xx")], "edges": [[0, 1, 1.0]]}], NAMES)


def test_rank_trace_hop0_and_hop1():
    # 0 client(ok) -> 3; 1 tb-edge(ok) -> 2 tb-beta(ok) -> 3 ext-gamma(5xx); 1 -> 4 tb-beta(slow) <- 2; 5 tb-edge(ok) idle
    trace = {
        "positions": [
            pos("client", is_client=True),
            pos("tb-edge"),
            pos("tb-beta"),
            pos("ext-gamma", "5xx"),
            pos("tb-beta", "slow"),
            pos("tb-edge"),
        ],
        "edges": [[0, 3, 0.9], [1, 2, 0.8], [2, 3, 0.7], [1, 4, 0.6], [2, 4, 0.5]],
    }
    h1 = tracerule.rank_trace(trace, NAMES)
    # both non-OK closures hold positions 1 and 2 (count 2, min hop 1 each): a full tie, broken by services.json
    # order for services (tb-edge before tb-beta) and by index for positions; the client at hop 1 of q=3 is dropped
    assert h1["services"] == ["tb-edge", "tb-beta"] and h1["positions"] == [1, 2]
    assert h1["n_nonok"] == 2 and h1["n_client_dropped"] == 1
    h0 = tracerule.rank_trace(trace, NAMES, hop0=True)
    # hop 0 adds 3 (ext-gamma) and 4 (tb-beta): tb-beta now (2 closures, min hop 0) beats tb-edge (2, hop 1);
    # ext-gamma (1 closure, hop 0) is reached last; positions 3 and 4 join at (1, 0) after 1 and 2 at (2, 1)
    assert h0["services"] == ["tb-beta", "tb-edge", "ext-gamma"] and h0["positions"] == [1, 2, 3, 4]
    assert set(h1["services"]) <= set(h0["services"]) and set(h1["positions"]) <= set(h0["positions"])
    assert h0["n_nonok"] == 2 and h0["n_client_dropped"] == 1
    # reached only: a trace without a non-OK position ranks nothing; one without parents ranks nothing at hop 1
    quiet = {"positions": [pos("tb-beta"), pos("tb-edge")], "edges": [[0, 1, 1.0]]}
    assert tracerule.rank_trace(quiet, NAMES) == {"services": [], "positions": [], "n_nonok": 0, "n_client_dropped": 0}
    lone = {"positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": []}
    assert tracerule.rank_trace(lone, NAMES)["services"] == []
    assert tracerule.rank_trace(lone, NAMES, hop0=True) == {
        "services": ["tb-edge"],
        "positions": [1],
        "n_nonok": 1,
        "n_client_dropped": 0,
    }
    # a client's own non-OK position is dropped (and counted) at hop 0 too
    client_nonok = {"positions": [pos("tb-beta"), pos("client", "5xx", True)], "edges": [[0, 1, 1.0]]}
    assert tracerule.rank_trace(client_nonok, NAMES, hop0=True) == {
        "services": ["tb-beta"],
        "positions": [0],
        "n_nonok": 1,
        "n_client_dropped": 1,
    }
    assert tracerule.rank_trace(client_nonok, NAMES)["n_client_dropped"] == 0
    # the count beats the hop: tb-edge held by both closures at hop 2 outranks tb-beta and ext-gamma, each held by one
    # closure at hop 1 (their tie broken by services.json order)
    two = {
        "positions": [
            pos("tb-edge"),
            pos("tb-beta"),
            pos("ext-gamma", "5xx"),
            pos("ext-gamma"),
            pos("ext-gamma", "err"),
        ],
        "edges": [[0, 1, 1.0], [1, 2, 1.0], [0, 3, 1.0], [3, 4, 1.0]],
    }
    h = tracerule.rank_trace(two, NAMES)
    assert h["services"] == ["tb-edge", "tb-beta", "ext-gamma"] and h["positions"] == [0, 1, 3]
    with pytest.raises(ValueError):
        tracerule.rank_trace({"positions": [pos("elsewhere"), pos("tb-edge", "5xx")], "edges": [[0, 1, 1.0]]}, NAMES)


def _fake_runner(docs, seen):
    """A stand-in for trace_runner.py: records the command line and writes the canned per-case files. `docs[case]` is a
    list of traces, or `{"traces": [...], "skipped": [ids], "n_nonok_folded": k}`; a trace without a `trace_id` gets
    one; `--cap N > 0` keeps the first N canned traces (0 = all) and `--select` is echoed into every document."""

    def spawn(cmd, env):
        seen.append({"cmd": cmd, "env": env})
        out = Path(cmd[cmd.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        windows = json.loads(Path(cmd[cmd.index("--cases") + 1]).read_text())
        select = cmd[cmd.index("--select") + 1]
        cap = int(cmd[cmd.index("--cap") + 1])
        phase = cmd[cmd.index("--phase") + 1]
        i = cmd.index("--cells") + 1
        cells = []
        while i < len(cmd) and not cmd[i].startswith("--"):
            cells.append(cmd[i])
            i += 1
        for w in windows:
            canned = docs[w["case_dir"]]
            spec = canned if isinstance(canned, dict) else {"traces": canned}
            traces = [
                {
                    "trace_id": tr.get("trace_id", f"{w['case_dir']}-t{i:02d}"),
                    **tr,
                    "edges_by_cell": {c: list(tr["edges"]) for c in cells},  # one layout: every cell the same cut
                }
                for i, tr in enumerate(spec["traces"])
            ]
            if cap > 0:
                traces = traces[:cap]
            skipped = list(spec.get("skipped", []))
            selected = [tr["trace_id"] for tr in traces] + skipped
            doc = {
                "case_dir": w["case_dir"],
                "select": select,
                "phase": phase,
                "cells": cells,
                "n_traces_window": 7,
                "n_traces_candidates": len(spec["traces"]) + len(skipped),
                "n_traces_selected": len(selected),
                "n_traces_probed": len(traces),
                "n_skipped_short": len(skipped),
                "n_truncated": 0,
                "n_folded": 0,
                "n_nonok_folded": spec.get("n_nonok_folded", 0),
                "selected_trace_ids": selected,
                "skipped_trace_ids": skipped,
                "tok_pos": 10,
                "probe_wall_clock_s": 0.1,
                "traces": traces,
            }
            (out / f"{w['case_dir']}.json").write_text(json.dumps(doc))
        (out / rt.SUMMARY_JSON).write_text(
            json.dumps(
                {
                    "cell": cells[0],
                    "cells": cells,
                    "phase": phase,
                    "seed": int(cmd[cmd.index("--seed") + 1]),
                    "corpus_id": "xs/latent/seed=0",
                    "model_sha256": "x" * 64,
                    "select": select,
                    "cap": cap,
                    "c": 1,
                }
            )
        )

        class Proc:
            returncode = 0
            stderr = ""
            stdout = ""

        return Proc()

    return spawn


def _ranktrace(
    arm,
    rec,
    corpus,
    out,
    tmp_path,
    harness_root,
    axis="ranking",
    select="window",
    cap=TRACE_TRACE_CAP,
    reading="hop1",
    phase="post",
    subtract="none",
    pre_record="none",
    probe_arms=None,
    work_dir=None,
):
    freeze = tmp_path / "freeze.json"
    if not freeze.exists():
        freeze.write_text("{}")
    return rt.main(
        [
            "--arm",
            arm,
            "--probe-arms",
            *(probe_arms or [arm]),
            "--reading",
            reading,
            "--phase",
            phase,
            "--subtract",
            subtract,
            "--pre-record",
            str(pre_record),
            "--axis",
            axis,
            "--select",
            select,
            "--cap",
            str(cap),
            "--export-record",
            str(rec),
            "--corpus-dir",
            str(corpus),
            "--model-dir",
            str(tmp_path / "model"),
            "--freeze",
            str(freeze),
            "--rung",
            "xs",
            "--seed",
            "0",
            "--trace-python",
            "python-of-the-probe-environment",
            "--harness-root",
            str(harness_root),
            "--device",
            "cpu",
            "--work-dir",
            str(work_dir or (tmp_path / "work" / arm.replace("/", "-"))),
            "--prior-rung-record",
            "smallest-rung",
            "--staging-ledger",
            LEDGER,
            "--output-folder",
            str(out),
        ]
    )


def test_the_stage_reads_only_the_declared_inputs_and_scores_through_the_benchmark(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    corpus = fixture_corpus()
    harness_root = tmp_path / "harness"
    for rel in rt.HARNESS_SRC:
        (harness_root / rel).mkdir(parents=True)
    docs = {
        "case-00": [
            {"positions": [pos("tb-beta"), pos("tb-edge", "5xx")], "edges": [[0, 1, 0.7]]},
            {
                "positions": [pos("ext-gamma"), pos("tb-beta", "slow"), pos("tb-edge", "5xx")],
                "edges": [[0, 1, 0.2], [1, 2, 0.9]],
            },
        ],
        "case-01": [{"positions": [pos("tb-beta"), pos("tb-edge")], "edges": [[0, 1, 0.7]]}],  # nothing failed: a miss
    }
    seen = []
    monkeypatch.setattr(rt, "_spawn", _fake_runner(docs, seen))
    monkeypatch.setattr(rt, "pinned_environment", lambda python: ["stand-in==0"])

    def boom(*_a, **_k):
        raise AssertionError("rca/trace must never open data.csv")

    monkeypatch.setattr(cases, "read_table", boom)
    out = tmp_path / "trace/rank"
    assert _ranktrace(ARM_TRACE, rec, corpus, out, tmp_path, harness_root) == 0
    (call,) = seen
    cmd = call["cmd"]
    assert cmd[1].endswith("trace_runner.py") and cmd[cmd.index("--cells") + 1] == TRACE_CELLS[ARM_TRACE]
    assert cmd[cmd.index("--phase") + 1] == "post"
    assert cmd[cmd.index("--cap") + 1] == str(TRACE_TRACE_CAP) and cmd[cmd.index("--max-len") + 1] == str(TRACE_MAX_LEN)
    assert cmd[cmd.index("--corpus-dir") + 1] == str(corpus) and cmd[cmd.index("--seed") + 1] == "0"
    assert all(str(harness_root / rel) in call["env"]["PYTHONPATH"] for rel in rt.HARNESS_SRC)
    windows = read_json(tmp_path / "work" / "rca-trace" / rt.CASES_JSON)
    assert [w["case_dir"] for w in windows] == ["case-00", "case-01"] and all(
        "inject_time" in w and "end_time" in w for w in windows
    )
    res = read_json(out / RUN_DIR / RESULTS_JSON)
    posit = read_json(out / rt.POSITIONAL_JSON)["cases"]
    assert res["stage"] == "rank" and res["arm"] == ARM_TRACE and res["code"]["cell"] == TRACE_CELLS[ARM_TRACE]
    assert list(posit) == ["case-00"] and posit["case-00"] == ["tb-beta", "ext-gamma", "tb-edge"]
    assert res["n_unranked_cases"] == 1 and res["misses_by_reason"] == {TRACE_MISS_NO_NONOK: 1}
    c0 = res["cases"][0]
    assert c0["counts"] == {"tb-beta": 2, "ext-gamma": 1} and c0["self_counts"] == {"tb-edge": 2, "tb-beta": 1}
    assert c0["n_traces_window"] == 7 and c0["n_traces_probed"] == 2 and c0["n_services_reached"] == 2
    assert res["n_filled_cells"] == 0 and res["pinned_environment"]["pip_freeze"] == ["stand-in==0"]
    assert _score(corpus, out, tmp_path / "trace") == 0
    sc = read_json(tmp_path / "trace" / SCORE_RCA_JSON)
    assert (
        sc["metrics"]["n_cases"] == 2 and sc["n_missing"] == 1
    )  # case 0's root cause tb-beta ranks first; case 1 is a miss
    assert [c["rank"] for c in sc["cases"] if not c["missing"]] == [1]


def test_the_stage_refuses_a_foreign_seed_cell_or_corpus(tmp_path, monkeypatch):
    rec, _ = _export_record(tmp_path)
    corpus = fixture_corpus()
    harness_root = tmp_path / "harness"
    for rel in rt.HARNESS_SRC:
        (harness_root / rel).mkdir(parents=True)
    monkeypatch.setattr(rt, "pinned_environment", lambda python: [])
    freeze = tmp_path / "f.json"
    freeze.write_text("{}")
    args = dict(
        python=Path("py"),
        harness_root=harness_root,
        device="cpu",
        prior="smallest-rung",
        ledger="unused.json",
        axis="ranking",
        select="window",
        cap=TRACE_TRACE_CAP,
        reading="hop1",
        phase="post",
        subtract="none",
        pre_record="none",
    )
    with pytest.raises(ValueError):  # --seed must name the corpus
        rt.rank(
            ARM_TRACE,
            rec,
            corpus,
            tmp_path / "m",
            freeze,
            "xs",
            1,
            work_dir=tmp_path / "w1",
            out=tmp_path / "o1",
            probe_arms=[ARM_TRACE],
            **args,
        )
    with pytest.raises(ValueError):  # a BARO arm has no TRACE stage
        rt.rank(
            "rca/baro",
            rec,
            corpus,
            tmp_path / "m",
            freeze,
            "xs",
            0,
            work_dir=tmp_path / "w2",
            out=tmp_path / "o2",
            probe_arms=["rca/baro"],
            **args,
        )
    monkeypatch.setattr(rt, "_spawn", _fake_runner({"case-00": [], "case-01": []}, []))
    with pytest.raises(FileNotFoundError):  # the harness root must hold both source trees
        rt.rank(
            ARM_TRACE_CORE,
            rec,
            corpus,
            tmp_path / "m",
            freeze,
            "xs",
            0,
            work_dir=tmp_path / "w3",
            out=tmp_path / "o3",
            probe_arms=[ARM_TRACE_CORE],
            **{**args, "harness_root": tmp_path / "nowhere"},
        )
    out = tmp_path / "o4"
    rt.rank(
        ARM_TRACE_CORE,
        rec,
        corpus,
        tmp_path / "m",
        freeze,
        "xs",
        0,
        work_dir=tmp_path / "w4",
        out=out,
        probe_arms=[ARM_TRACE_CORE],
        **args,
    )
    res = read_json(out / rt.POSITIONAL_JSON)
    assert res == {"cases": {}}  # every case a miss: no window trace


def _row(rung, arm, avg, low, random=0.6):
    return {
        "rung": rung,
        "arm": arm,
        "metrics": {"avg@5": avg, "ac@1": avg, "ac@3": avg, "ac@5": avg, "n_cases": 10},
        "bootstrap": {"intervals": {"avg@5": {"low": low, "high": avg + 0.1}}},
        "random": {"avg@5": random},
        "n_missing": 0,
    }


def test_report_scores_the_trace_hypotheses_and_lists_the_arms():
    rows = [
        _row("xs", ARM_TRACE, 0.7, 0.65),
        _row("xs", ARM_TRACE_CORE, 0.5, 0.45),
        _row("xs", ARM_CONTROL_RULE, 0.6, 0.5),
    ]
    hyp = report.hypotheses(rows, ("xs",))
    assert hyp["H-rca-trace-random"]["verdict"] == "pass" and hyp["H-rca-trace-rule"]["verdict"] == "pass"
    assert hyp["H-rca-trace-core-random"]["verdict"] == "fail" and hyp["H-rca-trace-core-rule"]["verdict"] == "fail"
    assert hyp["H-rca-trace-rule"]["gap_arm_minus_rule"]["xs"] == pytest.approx(0.1)
    assert ARM_TRACE in report.REPORT_ARMS and ARM_TRACE_CORE in report.REPORT_ARMS
