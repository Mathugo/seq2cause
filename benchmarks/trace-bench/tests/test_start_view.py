"""The start-ordered-view read (`plans/floors.md`, addenda of 2026-10-09 and 2026-10-22) on the
fixture, CPU, seconds: prepare and pretrain on `start-session`, the blind request sweep,
scoresweep, a `…-start.json` freeze, one test read, annotate and seqscore, every record naming the
view; the freeze refuses tables of one view bound to a backbone trained on the other, and a test
read refuses a freeze swept on the other view. Plus the view-order statistic, by hand and on both
views of the fixture."""

import shutil

import numpy as np
import pytest
from fixture_corpus import fixture_corpus
from test_m6_pipeline import (  # noqa: F401  (world: the end-view model + freeze, module fixture)
    GRANGER_TAUS,
    PRETRAIN,
    PROBE,
    PROBES,
    QUANTILES,
    RULE,
    TAUS,
    _git,
    world,
)

from seq2causebench import annotate as an
from seq2causebench import discover as dc
from seq2causebench import freeze as fz
from seq2causebench import prepare as pp
from seq2causebench import pretrain as pt
from seq2causebench import scoresweep as ss
from seq2causebench import seqscore as sq
from seq2causebench import sweep as sw
from seq2causebench import vieworder as vo
from seq2causebench.constants import ARGUMENTS_JSON, RESULTS_JSON, RUN_DIR, VAL_TABLE_JSON
from seq2causebench.record import read_json, write_json

START_PRETRAIN = [*PRETRAIN]
START_PRETRAIN[START_PRETRAIN.index("--ordering") + 1] = "start"
CORE_CELLS = ("trace/core/shipped/frozen", "trace/core/fixed-kl/frozen")


@pytest.fixture(scope="module")
def sworld(tmp_path_factory):
    """The start-view chain up to and including a committed `…-start.json` freeze."""
    root = tmp_path_factory.mktemp("start")
    corpus = fixture_corpus("latent")
    ledger = root / "staging-ledger.json"
    write_json(ledger, {"schema": "seq2causebench/staging-ledger@1", "entries": []})
    prep = [
        "--corpus",
        str(corpus),
        "--ordering",
        "start",
        "--grain",
        "session",
        "--max-len",
        "24",
        "--entropy-order",
        "1",
        "--output-folder",
        str(root / "prep"),
    ]
    assert pp.main(prep) == 0
    pre = [
        "--corpus",
        str(corpus),
        "--prepare-record",
        str(root / "prep"),
        *START_PRETRAIN,
        "--output-folder",
        str(root / "pre"),
    ]
    assert pt.main(pre) == 0
    model = str(root / "pre" / "model")
    probe = [*PROBE, "--staging-ledger", str(ledger)]
    sweep = [
        "--corpus",
        str(corpus),
        "--ordering",
        "start",
        "--grains",
        "request",
        "--split",
        "val",
        "--model",
        model,
        "--probes",
        *PROBES,
        "--noises",
        "all",
        "real",
        "--contexts",
        "2",
        "3",
        "--particles-grid",
        "2",
        "--guidance",
        "3",
        "--num-sequences",
        "0",
        "--shapley-sequences",
        "0",
        *RULE,
        *probe,
        "--output-folder",
        str(root / "sweep"),
    ]
    assert sw.main(sweep) == 0
    scoresweep = [
        "--corpus",
        str(corpus),
        "--sweep-dir",
        str(root / "sweep"),
        "--grains",
        "request",
        "--taus",
        *TAUS,
        "--granger-taus",
        *GRANGER_TAUS,
        "--quantiles",
        *QUANTILES,
        "--floor-taus",
        "--output-folder",
        str(root / "scoresweep"),
    ]
    assert ss.main(scoresweep) == 0
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    pre_results = root / "pre" / RUN_DIR / RESULTS_JSON
    assert fz.main(_freeze_args(root / "scoresweep", pre_results, repo / "freezes", root)) == 0
    freeze = next((repo / "freezes").glob("*-xs-latent-s0-start.json"))
    _git(["add", "-A"], repo)
    _git(
        ["commit", "-q", "-m", "freeze"],
        repo,
        env={
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        },
    )
    return {
        "root": root,
        "corpus": corpus,
        "model": model,
        "freeze": freeze,
        "freeze_doc": read_json(freeze),
        "probe": probe,
        "pretrain_results": pre_results,
    }


def _freeze_args(scoresweep_dir, pretrain_results, freezes_dir, root):
    return [
        "--val-tables",
        str(scoresweep_dir / VAL_TABLE_JSON),
        "--pretrain-results",
        str(pretrain_results),
        "--rung",
        "xs",
        "--variant",
        "latent",
        "--seed",
        "0",
        "--freezes-dir",
        str(freezes_dir),
        "--output-folder",
        str(root / f"freeze-run-{freezes_dir.name}"),
    ]


def _discover(w, cells, out, ordering):
    fd = w["freeze_doc"]["cells"]
    keys = [f"{c}/request" for c in cells]
    ((c, N, g),) = {(fd[k]["c"], fd[k]["N"], fd[k]["g"]) for k in keys}
    specs = [f"{k}=" + str(fd[k]["tau"]) for k in keys]
    return dc.main(
        [
            "--corpus",
            str(w["corpus"]),
            "--ordering",
            ordering,
            "--grain",
            "request",
            "--split",
            "test",
            "--model",
            w["model"],
            "--probe",
            "core",
            "--noise",
            "all",
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
            str(w["freeze"]),
            *w["probe"],
            "--output-folder",
            str(out),
        ]
    )


def test_every_record_names_the_start_view(sworld):
    root = sworld["root"]
    assert read_json(root / "prep" / RUN_DIR / ARGUMENTS_JSON)["arguments"]["ordering"] == "start"
    assert read_json(root / "pre" / RUN_DIR / ARGUMENTS_JSON)["arguments"]["ordering"] == "start"
    for npz in (root / "sweep").glob("scores-*-request.npz"):
        assert str(np.load(npz)["ordering"]) == "start"
    table = read_json(root / "scoresweep" / VAL_TABLE_JSON)
    assert table["ordering"] == "start" and table["grains"] == ["request"]
    doc = sworld["freeze_doc"]
    assert sworld["freeze"].name.endswith("-xs-latent-s0-start.json")
    assert doc["ordering"] == "start" and doc["family"] == "model"
    assert all(k.endswith("/request") for k in doc["cells"])


def test_freeze_refuses_a_backbone_trained_on_the_other_view(sworld, tmp_path):
    """Same model hash, training view rewritten to `end`: the tables were swept on `start`."""
    pre_run = tmp_path / "pre-run"
    shutil.copytree(sworld["root"] / "pre" / RUN_DIR, pre_run)
    args = read_json(pre_run / ARGUMENTS_JSON)
    args["arguments"]["ordering"] = "end"
    write_json(pre_run / ARGUMENTS_JSON, args)
    with pytest.raises(fz.FreezeRefusal, match="trained on the 'end' view"):
        fz.main(
            _freeze_args(
                sworld["root"] / "scoresweep", pre_run / RESULTS_JSON, tmp_path / "f1", tmp_path
            )
        )
    (pre_run / ARGUMENTS_JSON).unlink()
    with pytest.raises(fz.FreezeRefusal, match="is missing"):
        fz.main(
            _freeze_args(
                sworld["root"] / "scoresweep", pre_run / RESULTS_JSON, tmp_path / "f2", tmp_path
            )
        )


def test_end_view_freeze_still_binds_its_end_backbone(world, tmp_path):  # noqa: F811
    """The check passes on the end-view chain (every landed freeze is end on end)."""
    assert (
        fz.main(
            _freeze_args(
                world["root"] / "scoresweep", world["pretrain_results"], tmp_path / "f", tmp_path
            )
        )
        == 0
    )
    assert next((tmp_path / "f").glob("*-xs-latent-s0.json"))


def test_test_read_refuses_a_freeze_of_the_other_view(sworld, tmp_path):
    with pytest.raises(dc.FreezeRefusal, match="swept on the 'start' view, this run reads 'end'"):
        _discover(sworld, CORE_CELLS, tmp_path / "read-end", "end")


def test_start_view_test_read_annotate_and_seqscore(sworld, tmp_path):
    w, out = sworld, tmp_path / "read"
    assert _discover(w, CORE_CELLS, out, "start") == 0
    r = read_json(out / RUN_DIR / RESULTS_JSON)
    assert r["status"] == "ok" and r["freeze"]["sha"]
    assert read_json(out / RUN_DIR / ARGUMENTS_JSON)["arguments"]["ordering"] == "start"
    assert str(np.load(out / "scores-core-all-request.npz")["ordering"]) == "start"
    for cell in CORE_CELLS:
        key = f"{cell}/request"
        res = tmp_path / "results-start" / cell / "request"
        ann = [
            "--corpus",
            str(w["corpus"]),
            "--run-dir",
            str(out),
            "--cell",
            key,
            "--pretrain-results",
            str(w["pretrain_results"]),
            "--per-lag-files",
            "3",
            "--output-folder",
            str(res),
        ]
        assert an.main(ann) == 0
        assert read_json(res / "annotate.json")["coverage"]["reachable_recall_ceiling"] <= 1
        seq = ["--corpus", str(w["corpus"]), "--run-dir", str(out), "--cell", key]
        assert sq.main([*seq, "--max-lag", "6", "--output-folder", str(res / "seq")]) == 0
        s = read_json(res / "seq" / "seqscore.json")
        assert s["grain"] == "request" and s["predict_all"]["pooled"] is not None
        assert s["n_sequences"] + s["n_skipped_short_by_read"] == 24 == s["n_sequences_split"]


# --- the view-order statistic --------------------------------------------------------------------
class _Seqs:
    def __init__(self, rows):
        self.rows = rows

    def sequences(self, split):
        return iter(self.rows)


def test_view_order_counts_by_hand():
    """Tokens a b a with a → b: positions (0, 1) cause first, (1, 2) cause second, (0, 2) nothing.
    With b → a in the truth too, each pair counts once for each edge."""
    a, b = vo.token_key("3:err"), vo.token_key("7:slow")
    rows = [("t0", [3, 7, 3], [3, 4, 3], 3), ("t1", [3], [3], 1)]
    one = np.array([a * vo.PAIR_BASE + b], dtype=np.int64)
    got = vo.view_order(_Seqs(rows), "test", 0, "head", 0, 64, one)
    assert (got["n_cause_first"], got["n_cooccurrences"], got["share_cause_first"]) == (1, 2, 0.5)
    assert got["n_sequences"] == 2 and got["n_sequences_with_cooccurrence"] == 1
    both = np.sort(np.array([a * vo.PAIR_BASE + b, b * vo.PAIR_BASE + a], dtype=np.int64))
    got = vo.view_order(_Seqs(rows), "test", 0, "head", 0, 64, both)
    assert (got["n_cause_first"], got["n_cooccurrences"]) == (2, 4)
    assert got["n_truth_edges_cooccurring"] == 2 and got["edge_mean_share"] == 0.5
    # the cap reads the first max_len tokens only (D-SB-7)
    got = vo.view_order(_Seqs(rows), "test", 0, "head", 0, 2, one)
    assert (got["n_cause_first"], got["n_cooccurrences"]) == (1, 1)


def test_view_order_on_both_views_of_the_fixture(tmp_path):
    """Truth runs callee → caller; the end view is post-order (callee first), the start view
    pre-order (caller first). Same sequences, so the same co-occurrences, opposite majorities."""
    corpus = fixture_corpus("latent")
    got = {}
    for ordering in ("end", "start"):
        out = tmp_path / f"{ordering}-request"
        argv = [
            "--corpus",
            str(corpus),
            "--ordering",
            ordering,
            "--grain",
            "request",
            "--split",
            "test",
            "--num-sequences",
            "0",
            "--sequence-sample",
            "head",
            "--max-len",
            "64",
            "--seed",
            "0",
            "--output-folder",
            str(out),
        ]
        assert vo.main(argv) == 0
        got[ordering] = read_json(out / vo.VIEWORDER_JSON)
        assert got[ordering]["ordering"] == ordering and got[ordering]["n_sequences"] == 24
    end, start = got["end"], got["start"]
    assert end["n_cooccurrences"] == start["n_cooccurrences"] > 0
    assert end["share_cause_first"] > 0.5 > start["share_cause_first"]
    assert end["n_truth_edges"] == 6
