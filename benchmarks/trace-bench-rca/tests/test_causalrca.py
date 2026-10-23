"""CausalRCA's adapter (`plans/rca-arms.md`, addenda 2026-10-22, later (D-TBR-5) and 2026-10-23), on a
fake pinned checkout that carries the registered data block and a torch stand-in: the data block is
replaced and nothing else, the input is the fault window min-max scaled per column in services.json
order, the runner leaves the script's last line as the ranking and converts the graph, a failed run is
counted and never refilled, the ranking axis averages per-run hits and reproduces the benchmark's
bootstrap when runs agree, the graph axis is validated on seed 0, frozen with ties to the larger τ and
tested on seeds 1-4 only, and the report scores H-graph-topology on the four test seeds."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from fixture_corpus import fixture_corpus
from test_control_rule import INJECT, N_ROWS, _case
from test_rank_score import _export_record
from tracebench.score_rca import bootstrap, metrics_of

from tracebenchrca import causalrca as cr
from tracebenchrca import graphaxis, graphfreeze, rankcausalrca, report, scorecausalrca
from tracebenchrca.constants import (
    CAUSALRCA_BLOCK_END,
    CAUSALRCA_BLOCK_START,
    CAUSALRCA_GRAPH_TAUS,
    RESULTS_JSON,
    RUN_DIR,
    SCORE_RCA_JSON,
)
from tracebenchrca.record import read_json

FAKE_SCRIPT = f"""import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from config import CONFIG
from utils import *

{CAUSALRCA_BLOCK_START}
metrics = ['ctn_latency']
idx = 0
all_data = None
name = None
{CAUSALRCA_BLOCK_END}

x = data.to_numpy()
n = x.shape[1]
adj = np.zeros((n, n))
adj[1, 0] = 0.6
if n > 2:
    adj[2, 1] = 0.4 + 0.01 * torch._seed
f = open('cpu-hog_front-end_adj.pkl', 'wb')  # left open, as the authors' script leaves it
pkl.dump(adj, f)
open('metrics_causality.png', 'wb').write(b'png')
print(names[idx])
if torch._seed == 3:
    print('training diverged')
else:
    order = sorted({{i: float(x[:, i].std()) + 1e-9 * i for i in range(n)}}.items(), key=lambda kv: kv[1], reverse=True)
    print(order)
print('epochs', CONFIG.epochs, file=__import__('sys').stderr)
"""

TORCH_STUB = """__version__ = '0.0-stub'
_seed = None
def manual_seed(s):
    global _seed
    _seed = int(s)
def get_num_threads():
    return 1
class cuda:
    @staticmethod
    def is_available():
        return False
"""


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """A fake pinned checkout (git, clean) and the torch stand-in on the runner's import path."""
    src = tmp_path / "causalrca-src"
    src.mkdir()
    (src / "train_latency.py").write_text(FAKE_SCRIPT)
    (src / "config.py").write_text("class CONFIG:\n    epochs = 500\n")
    (src / "utils.py").write_text("")
    (src / "modules.py").write_text("")
    git = ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "pin"], check=True)
    head = subprocess.run(git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    shas = {n: cr._sha(src / n) for n in ("train_latency.py", "modules.py", "utils.py", "config.py")}
    monkeypatch.setattr(cr, "CAUSALRCA_COMMIT", head)
    monkeypatch.setattr(cr, "CAUSALRCA_FILES_SHA256", shas)
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "torch.py").write_text(TORCH_STUB)
    monkeypatch.setenv("PYTHONPATH", str(stub))
    return src


def _corpus(tmp_path, seed):
    """The fixture corpus with a service view that has truth: tb-beta → tb-edge, ext-gamma → tb-beta."""
    d = tmp_path / f"corpora/instances/xs/latent/seed={seed}"
    if not d.exists():
        shutil.copytree(fixture_corpus(seed=seed), d)
        view = {
            "level": "service",
            "grain": "request",
            "default_floor": 0.05,
            "directed": [
                {"src": "svc:1", "dst": "svc:0", "strength": 0.8},
                {"src": "svc:2", "dst": "svc:1", "strength": 0.9},
            ],
            "bidirected": [],
        }
        (d / "graphs/views/service.json").write_text(json.dumps(view))
    return d


def _rank(tmp_path, pinned, seed, series="latency-95", prior="smallest-rung"):
    rec = tmp_path / f"export-record-s{seed}"
    if not rec.exists():  # one export per seed, read once per series
        rec, _ = _export_record(tmp_path, seed=seed)
    out = tmp_path / f"results/xs/latent/seed={seed}/rca-causalrca/{series}"
    argv = ["--export-record", str(rec), "--rung", "xs", "--series", series, "--causalrca-src", str(pinned)]
    argv += ["--causalrca-python", sys.executable, "--device", "cpu", "--workers", "4"]
    argv += ["--work-dir", str(tmp_path / f"work-s{seed}-{series}"), "--prior-rung-record", prior]
    argv += ["--staging-ledger", "unused.json", "--output-folder", str(out / "rank")]
    assert rankcausalrca.main(argv) == 0
    return out


def _score(tmp_path, seed, out):
    argv = ["--corpus-dir", str(_corpus(tmp_path, seed)), "--rank-record", str(out / "rank")]
    assert (
        scorecausalrca.main(argv + ["--bootstrap-b", "50", "--bootstrap-seed", "0", "--output-folder", str(out)]) == 0
    )


def _axis(tmp_path, seed, out, split, freeze, taus):
    argv = ["--corpus-dir", str(_corpus(tmp_path, seed)), "--score-record", str(out), "--split", split]
    argv += ["--freeze", freeze, "--taus", *[str(t) for t in taus], "--output-folder", str(out / f"graph-{split}")]
    return graphaxis.main(argv)


def test_the_data_block_is_replaced_and_nothing_else():
    source = f"head\n{CAUSALRCA_BLOCK_START}\nf = open('x')\n{CAUSALRCA_BLOCK_END}\ntail\n"
    out, removed = cr.substitute(source, ["a", "b"], "latency-95")
    assert out.startswith("head\n") and out.endswith("\ntail\n")
    assert removed == f"{CAUSALRCA_BLOCK_START}\nf = open('x')\n{CAUSALRCA_BLOCK_END}"
    assert "names = ['a', 'b']" in out and "CONFIG.epochs = 1000" in out and "np.load('input.npy')" in out
    assert "columns=['a_latency-95', 'b_latency-95']" in out and "open('x')" not in out
    with pytest.raises(ValueError, match="exactly once"):
        cr.substitute(source + CAUSALRCA_BLOCK_START, ["a"], "latency-95")


def test_the_input_is_the_fault_window_min_max_scaled_in_services_order(tmp_path):
    case = _case(
        tmp_path / "case-00",
        ["svc-b", "svc-a"],
        {("svc-a", "latency-95"): {r: 7.0 for r in range(N_ROWS)}, ("svc-b", "latency-95"): {INJECT + 3: 9.0}},
    )
    arr, names, info = cr.input_array(case, "latency-95")
    assert names == ["svc-b", "svc-a"] and arr.shape == (N_ROWS - INJECT, 2) and info["n_rows"] == N_ROWS - INJECT
    assert arr[:, 1].tolist() == [0.0] * (N_ROWS - INJECT) and info["n_constant_columns"] == 1  # constant → 0
    assert arr[:, 0].min() == 0.0 and arr[:, 0].max() == 1.0 and arr[3, 0] == 1.0


@pytest.mark.parametrize(
    "line,reason",
    [
        ("[(1, 0.5), (0, 0.2), (2, 0.1)]", None),
        ("[(1, 0.5), (0, nan), (2, 0.1)]", "not a list"),
        ("[(1, 0.5), (1, 0.2), (2, 0.1)]", "exactly once"),
        ("training diverged", "not a list"),
    ],
)
def test_parse_ranking(line, reason):
    order, why = cr.parse_ranking("tb-edge\n" + line + "\n", 3)
    assert (order == [1, 0, 2]) if reason is None else (order is None and reason in why)


def test_run_averaged_scores_equal_the_benchmark_when_runs_agree():
    rng = np.random.default_rng(1)
    ranks = [None if v == 0 else int(v) for v in rng.integers(0, 7, size=40)]
    hits = scorecausalrca.hit_matrix([[r] * 10 for r in ranks], 5)
    m = scorecausalrca.run_averaged_metrics(hits)
    ref = metrics_of(ranks)
    assert all(m[k] == pytest.approx(ref[k]) for k in ("ac@1", "ac@3", "ac@5", "avg@5"))
    assert scorecausalrca.run_averaged_bootstrap(hits, 300, 0) == bootstrap(ranks, 300, 0)


def test_the_source_check_refuses_a_dirty_or_changed_checkout(pinned):
    cr.check_source(pinned)
    (pinned / "utils.py").write_text("# edited\n")
    with pytest.raises(ValueError, match="clean"):
        cr.check_source(pinned)


def test_rank_score_and_the_graph_axis_end_to_end(tmp_path, pinned):
    out = _rank(tmp_path, pinned, seed=0)
    res = read_json(out / "rank" / RUN_DIR / RESULTS_JSON)
    assert res["arm"] == "rca/causalrca" and res["series"] == "latency-95" and res["n_runs_per_case"] == 10
    assert res["n_failed_runs"] == 2 and res["failures_by_reason"] == {
        "last line is not a list of (index, score) pairs": 2
    }
    run = res["cases"][0]["runs"][0]
    assert run["runner"]["seed"] == 0 and run["runner"]["env"]["OMP_NUM_THREADS"] == "1"
    assert res["code"]["epochs"] == 1000 and res["code"]["files_sha256"]["train_latency.py"] == cr._sha(
        pinned / "train_latency.py"
    )
    positional = read_json(out / "rank" / "ranking-positional.json")["cases"]
    case = positional["case-00"]
    assert case["rankings"][3] is None and sum(r is None for r in case["rankings"]) == 1
    assert case["mean_abs_adj"][1][0] == pytest.approx(0.6)  # the mean over the nine successful runs
    assert case["mean_abs_adj"][2][1] == pytest.approx(0.4 + 0.01 * np.mean([0, 1, 2, 4, 5, 6, 7, 8, 9]))

    _score(tmp_path, 0, out)
    score = read_json(out / SCORE_RCA_JSON)
    assert score["aggregation"].startswith("per-run") and score["metrics"]["n_cases"] == 2
    assert all(c["n_runs"] == 10 and c["n_failed_runs"] == 1 for c in score["cases"])
    graph = read_json(out / "graph.json")["cases"]
    edges = {(e["src"], e["dst"]) for e in next(iter(graph.values()))["directed"]}
    assert edges == {("svc:1", "svc:0"), ("svc:2", "svc:1")}  # adj[i, j] reads names[i] → names[j]

    assert _axis(tmp_path, 0, out, "val", "none", CAUSALRCA_GRAPH_TAUS) == 0
    fz_dir = tmp_path / "freezes"
    assert (
        graphfreeze.main(
            [
                "--val-read",
                str(out / "graph-val"),
                "--freezes-dir",
                str(fz_dir),
                "--output-folder",
                str(out / "graph-freeze"),
            ]
        )
        == 0
    )
    (fz,) = fz_dir.glob("*-xs-rca-causalrca-latency-95-graph.json")
    frozen = read_json(fz)
    assert frozen["tau"] == 0.4 and frozen["val_mean_f1"] == 1.0  # F1 1 up to 0.4, ties → the larger τ
    with pytest.raises(ValueError, match="seed 0"):
        graphaxis.read(_corpus(tmp_path, 0), out, "val", "none", [0, 0.1])
    with pytest.raises(ValueError, match="seeds 1-4"):
        graphaxis.read(_corpus(tmp_path, 0), out, "test", str(fz), [frozen["tau"]])

    out1 = _rank(tmp_path, pinned, seed=1)
    _score(tmp_path, 1, out1)
    with pytest.raises(ValueError, match="frozen tau"):
        graphaxis.read(_corpus(tmp_path, 1), out1, "test", str(fz), [0.1])
    assert _axis(tmp_path, 1, out1, "test", str(fz), [frozen["tau"]]) == 0
    doc = read_json(out1 / "graph-test" / "score-service.json")
    assert doc["mean_over_cases"]["0.4"]["f1"] == 1.0 and doc["floor_topology"]["f1"] == 0.0


def test_report_scores_h_graph_topology_on_the_four_test_seeds(tmp_path, pinned, monkeypatch):
    monkeypatch.setattr(cr, "CAUSALRCA_RUNS", 2)
    results = tmp_path / "results"
    fz_dir = tmp_path / "freezes"
    for series in ("latency-50", "latency-95"):
        for k in range(5):
            out = _rank(tmp_path, pinned, seed=k, series=series)
            _score(tmp_path, k, out)
        out0 = results / "xs/latent/seed=0/rca-causalrca" / series
        assert _axis(tmp_path, 0, out0, "val", "none", CAUSALRCA_GRAPH_TAUS) == 0
        assert (
            graphfreeze.main(
                [
                    "--val-read",
                    str(out0 / "graph-val"),
                    "--freezes-dir",
                    str(fz_dir),
                    "--output-folder",
                    str(out0 / "graph-freeze"),
                ]
            )
            == 0
        )
        (fz,) = fz_dir.glob(f"*-{series}-graph.json")
        for k in (1, 2, 3, 4):
            assert (
                _axis(
                    tmp_path,
                    k,
                    results / f"xs/latent/seed={k}/rca-causalrca" / series,
                    "test",
                    str(fz),
                    [read_json(fz)["tau"]],
                )
                == 0
            )
    tables = tmp_path / "tables"
    argv = [
        "--results-dir",
        str(results),
        "--rungs",
        "xs",
        "--seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--arms",
        "rca/causalrca",
    ]
    assert report.main(argv + ["--bootstrap-b", "50", "--bootstrap-seed", "0", "--output-folder", str(tables)]) == 0
    summary = read_json(tables / "summary.json")
    assert [r["series"] for r in summary["rows"]] == ["latency-50", "latency-95"]
    assert all(g["seeds"] == [1, 2, 3, 4] and g["n_cases"] == 8 for g in summary["graph_rows"])
    h = summary["hypotheses"]["H-graph-topology"]
    assert h["verdict"] == "fail" and h["seeds"] == [1, 2, 3, 4]  # the learned graph (F1 1) beats an empty topology
    md = (tables / "rca-latent.md").read_text(encoding="utf-8")
    assert "four test seeds (1–4)" in md and "`rca/causalrca` (latency-95)" in md
