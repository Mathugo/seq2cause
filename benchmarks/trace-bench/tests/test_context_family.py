"""The context family end to end on the fixture (plans/reference-arms.md, addendum 2026-10-31;
RUN.md D-SB-20 … 23): a validation sweep of the three probes at the registered N and c,
`scoresweep --arms` on the five arms, a `context` freeze (its own family, file name and grid
filter), test reads of the keep and atomic passes under it through both cuts, the recorded
context mode / window offset / counts source, the identity checks (ii) and (iii) on the
type-level tables, the refusal of a mixed freeze and of a context-mode mismatch."""

import json

import numpy as np
import pytest
from test_m6_pipeline import (  # noqa: F401  (world: the end-view model + freeze, module fixture)
    GRANGER_TAUS,
    PROBE,
    QUANTILES,
    RULE,
    TAUS,
    _git,
    world,
)

from seq2causebench import discover as dc
from seq2causebench import freeze as fz
from seq2causebench import scoresweep as ss
from seq2causebench import sweep as sw
from seq2causebench.constants import (
    CONTEXT_ARMS,
    CONTEXT_FAMILY_GRID,
    RESULTS_JSON,
    RUN_DIR,
    VAL_TABLE_JSON,
)
from seq2causebench.freezecheck import FreezeRefusal, assert_freeze
from seq2causebench.project import read_scores_npz
from seq2causebench.record import read_json


@pytest.fixture(scope="module")
def cworld(request):  # the pipeline test's module fixture, pulled by name (imported above)
    return request.getfixturevalue("world")


KEEP_CELLS = ("trace/core-ig-keep/fixed-kl/frozen", "trace/core-cmi-keep/fixed-kl/frozen")
ATOMIC_ALL = (
    "trace/cli-atomic-cmi-keep/fixed-kl/frozen",
    "trace/cli-atomic-cmi-keep/fixed-kl/shipped",
)
ATOMIC_REAL = ("trace/cli-atomic-cmi-keep/fixed/frozen", "trace/cli-atomic-cmi-keep/fixed/shipped")


def _scoresweep(w, sweep_dir, arms, out):
    return ss.main(
        [
            "--corpus",
            str(w["corpus"]),
            "--sweep-dir",
            str(sweep_dir),
            "--grains",
            "request",
            "--taus",
            *TAUS,
            "--granger-taus",
            *GRANGER_TAUS,
            "--quantiles",
            *QUANTILES,
            "--floor-taus",
            "--arms",
            *arms,
            "--output-folder",
            str(out),
        ]
    )


def _freeze(w, val_table, out):
    return fz.main(
        [
            "--val-tables",
            str(val_table),
            "--pretrain-results",
            str(w["pretrain_results"]),
            "--rung",
            "xs",
            "--variant",
            "latent",
            "--seed",
            "0",
            "--freezes-dir",
            str(w["repo"] / "freezes"),
            "--output-folder",
            str(out),
        ]
    )


def _discover(w, freeze_doc, freeze_path, probe, noise, cells, out):
    fd = freeze_doc["cells"]
    keys = [f"{c}/request" for c in cells]
    specs = [
        f"{k}=" + ("shipped" if k.endswith("/shipped/request") else str(fd[k]["tau"])) for k in keys
    ]
    c, N, g = fd[keys[0]]["c"], fd[keys[0]]["N"], fd[keys[0]]["g"]
    return dc.main(
        [
            "--corpus",
            str(w["corpus"]),
            "--ordering",
            "end",
            "--grain",
            "request",
            "--split",
            "test",
            "--model",
            w["model"],
            "--probe",
            probe,
            "--noise",
            noise,
            "--context",
            str(c),
            "--guidance",
            str(g),
            "--particles",
            str(N),
            "--cells",
            *specs,
            *RULE,
            "--num-sequences",
            "0",
            "--cells-sequences",
            "0",
            "--per-lag-files",
            "3",
            "--freeze",
            str(freeze_path),
            *w["probe"],
            "--output-folder",
            str(out),
        ]
    )


def test_context_family_sweep_freeze_and_reads(cworld, tmp_path):
    w = cworld
    root = tmp_path
    assert (
        sw.main(
            [
                "--corpus",
                str(w["corpus"]),
                "--ordering",
                "end",
                "--grains",
                "request",
                "--split",
                "val",
                "--model",
                w["model"],
                "--probes",
                "core",
                "core-keep",
                "cli-atomic-cmi",
                "--noises",
                "all",
                "real",
                "--contexts",
                "1",
                "--particles-grid",
                "8",
                "--guidance",
                "1",
                "--num-sequences",
                "0",
                "--shapley-sequences",
                "0",
                *RULE,
                *w["probe"],
                "--output-folder",
                str(root / "sweep"),
            ]
        )
        == 0
    )
    r = read_json(root / "sweep" / RUN_DIR / RESULTS_JSON)
    # passes: core all + real, core-keep all (real has no column), atomic all + real
    assert r["status"] == "ok" and r["n_cells"] == 5
    by = {(c["probe"], c["noise"]): c for c in r["cells"]}
    assert set(by) == {
        ("core", "all"),
        ("core", "real"),
        ("core-keep", "all"),
        ("cli-atomic-cmi", "all"),
        ("cli-atomic-cmi", "real"),
    }
    assert by[("core", "all")]["context_mode"] == "overwrite"
    assert by[("core-keep", "all")]["context_mode"] == "none"
    assert by[("cli-atomic-cmi", "all")]["context_mode"] is None
    assert all(c["window_offset"] == 1 and c["g"] == 1 and c["c"] == 1 for c in r["cells"])
    assert by[("core", "all")]["counts_source"] == {"trace/core-cmi-overwrite": "ig-same-pass"}
    assert by[("core-keep", "all")]["counts_source"] == {"trace/core-cmi-keep": "ig-same-pass"}
    assert by[("cli-atomic-cmi", "real")]["counts_source"] == {"trace/cli-atomic-cmi-keep": "none"}
    assert by[("cli-atomic-cmi", "real")]["counts"] == {"fixed": {}}
    assert (
        "shipped_cut" in by[("cli-atomic-cmi", "all")]
        and "shipped_cut" not in by[("core-keep", "all")]
    )
    assert (root / "sweep" / "shippedcut-cli-atomic-cmi-real-c1-N8-request.json").exists()
    # identity checks (ii) and (iii) on the type-level tables of the sweep
    core = read_scores_npz(root / "sweep" / "scores-core-all-c1-N8-request.npz")
    keep = read_scores_npz(root / "sweep" / "scores-core-keep-all-c1-N8-request.npz")
    assert np.array_equal(
        core["max_trace-core__fixed-kl"], core["max_trace-core-ig-overwrite__fixed-kl"]
    )
    assert np.array_equal(core["src"], keep["src"]) and np.array_equal(core["dst"], keep["dst"])
    assert np.array_equal(
        core["max_trace-core-ig-overwrite__fixed-kl"], keep["max_trace-core-ig-keep__fixed-kl"]
    )
    assert np.array_equal(
        core["max_trace-core-cmi-overwrite__fixed-kl"], keep["max_trace-core-cmi-keep__fixed-kl"]
    )
    for col in ("max_trace-core-cmi-overwrite__fixed-kl",):
        assert core[col].min() >= 0 and core[col].max() <= np.log(2) + 1e-6
    atomic = read_scores_npz(root / "sweep" / "scores-cli-atomic-cmi-all-c1-N8-request.npz")
    a = atomic["max_trace-cli-atomic-cmi-keep__fixed-kl"]
    assert a.min() >= 0 and a.max() <= np.log(2) + 1e-6 and len(a) > 0
    # the family's validation table holds the five arms only (D-SB-21)
    assert _scoresweep(w, root / "sweep", CONTEXT_ARMS, root / "scoresweep") == 0
    t = read_json(root / "scoresweep" / VAL_TABLE_JSON)
    assert {row["arm"] for row in t["cells"]} == set(CONTEXT_ARMS) and t["arms"] == list(
        CONTEXT_ARMS
    )
    cuts = {(row["arm"], row["path"], row["cut"]) for row in t["cells"]}
    assert ("trace/cli-atomic-cmi-keep", "fixed", "shipped") in cuts
    assert ("trace/cli-atomic-cmi-keep", "fixed-kl", "frozen") in cuts
    assert ("trace/core-cmi-keep", "fixed-kl", "frozen") in cuts
    assert all(row["tau_source"] in ("grid", "shipped rule") for row in t["cells"])
    # the context freeze: its own family, name and grid; every cell inside the registered grid
    assert _freeze(w, root / "scoresweep" / VAL_TABLE_JSON, root / "freeze-run") == 0
    freeze_path = next((w["repo"] / "freezes").glob("*-xs-latent-s0-context.json"))
    f = read_json(freeze_path)
    assert f["family"] == "context" and f["schema"] == "seq2causebench/freeze@3"
    assert f["family_grid"] == json.loads(json.dumps(CONTEXT_FAMILY_GRID))
    assert set(f["cells"]) == {f"{c}/request" for c in (*KEEP_CELLS, *ATOMIC_ALL, *ATOMIC_REAL)} | {
        "trace/core-ig-overwrite/fixed-kl/frozen/request",
        "trace/core-cmi-overwrite/fixed-kl/frozen/request",
    }
    for key, cell in f["cells"].items():
        assert cell["window_offset"] == 1 and cell["N"] == 8 and cell["c"] == 1 and cell["g"] == 1
        if "keep" in key.split("/")[1] and key.startswith("trace/core"):
            assert cell["context_mode"] == "none" and cell["probe"] == "core-keep"
        elif key.startswith("trace/core"):
            assert cell["context_mode"] == "overwrite" and cell["probe"] == "core"
        else:
            assert cell["context_mode"] is None and cell["probe"] == "cli-atomic-cmi"
        if "inherits" not in cell:
            assert cell["n_rows_outside_family_grid"] == 0 and cell["grid"] == "taus"
    assert f["model_sha256"] == read_json(w["pretrain_results"])["model_sha256"]
    _git(["add", "-A"], w["repo"])
    _git(
        ["commit", "-q", "-m", "context freeze"],
        w["repo"],
        env={
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        },
    )
    # test reads under the context freeze: the keep pass and both atomic passes, both cuts
    for probe, noise, cells in (
        ("core-keep", "all", KEEP_CELLS),
        ("cli-atomic-cmi", "all", ATOMIC_ALL),
        ("cli-atomic-cmi", "real", ATOMIC_REAL),
    ):
        out = root / f"read-{probe}-{noise}"
        assert _discover(w, f, freeze_path, probe, noise, cells, out) == 0
        res = read_json(out / RUN_DIR / RESULTS_JSON)
        assert res["status"] == "ok" and res["freeze"]["sha"] and res["window_offset"] == 1
        assert res["context_mode"] == ("none" if probe == "core-keep" else None)
        for cell in cells:
            arm, path, cut = cell.rsplit("/", 2)
            slug = arm.replace("/", "-")
            assert (out / f"prediction-request-{slug}-{path}-{cut}.json").exists()
            assert res["cells"][f"{cell}/request"]["cut"] == cut
        if probe == "core-keep":
            assert res["counts_source"] == {"trace/core-cmi-keep": "ig-same-pass"}
            assert res["corrupted_cells"]["fixed-kl"]["n_cells"] > 0
        else:
            assert res["counts_source"] == {"trace/cli-atomic-cmi-keep": "none"}
            assert res["corrupted_cells"] == {("fixed-kl" if noise == "all" else "fixed"): {}}
            sc = next(out.glob("shippedcut-request-trace-cli-atomic-cmi-keep-*.json"))
            assert read_json(sc)["rule_is_v019"] is True
    # a table that mixes the family with the landed arms cannot be frozen (D-SB-21)
    assert _scoresweep(w, root / "sweep", [], root / "scoresweep-all") == 0
    t_all = read_json(root / "scoresweep-all" / VAL_TABLE_JSON)
    assert {row["arm"] for row in t_all["cells"]} > set(CONTEXT_ARMS)
    with pytest.raises(fz.FreezeRefusal, match="D-SB-21"):
        _freeze(w, root / "scoresweep-all" / VAL_TABLE_JSON, root / "freeze-all")


def test_discover_refuses_a_keep_probe_with_g_below_c(cworld, tmp_path):
    w = cworld
    with pytest.raises(ValueError, match="g == c"):
        dc.main(
            [
                "--corpus",
                str(w["corpus"]),
                "--ordering",
                "end",
                "--grain",
                "request",
                "--split",
                "val",
                "--model",
                w["model"],
                "--probe",
                "core-keep",
                "--noise",
                "all",
                "--context",
                "2",
                "--guidance",
                "1",
                "--particles",
                "2",
                "--cells",
                "trace/core-ig-keep/fixed-kl/frozen/request=0.1",
                *RULE,
                "--num-sequences",
                "4",
                "--cells-sequences",
                "0",
                "--per-lag-files",
                "1",
                "--freeze",
                "",
                *w["probe"],
                "--output-folder",
                str(tmp_path / "keep-bad"),
            ]
        )


def test_freezecheck_refuses_a_context_mode_mismatch(monkeypatch):
    from seq2causebench import freezecheck

    monkeypatch.setattr(freezecheck, "freeze_commit", lambda p: ("abc", "2026-01-01T00:00:00Z"))
    key = "trace/core-ig-keep/fixed-kl/frozen/request"
    freeze = {
        "cells": {key: {"c": 1, "N": 8, "g": 1, "tau": 0.1, "context_mode": "none"}},
        "model_sha256": "m",
        "corpus_id": "x",
        "ordering": "end",
    }
    args = (freeze, "p", {key: 0.1}, 1, 8, 1, "m", "x", "2026-02-01T00:00:00Z", "end")
    assert assert_freeze(*args, context_mode="none")["sha"] == "abc"
    with pytest.raises(FreezeRefusal, match="D-SB-20"):
        assert_freeze(*args, context_mode="overwrite")
    freeze["cells"][key].pop("context_mode")  # a cell written before D-SB-20 accepts any mode
    assert assert_freeze(*args, context_mode="overwrite")["sha"] == "abc"
