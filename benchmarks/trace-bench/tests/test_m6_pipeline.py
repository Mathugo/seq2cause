# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""M6 end to end on the fixture, CPU, seconds: prepare → pretrain → blind val sweep → scoresweep
(the benchmark's scorer) → freeze in a git repository → one test read under the freeze →
annotate → seqscore → five-seed report with paired differences and the t-test; plus every refusal
the PRD names (scenarios 3, 7, 9, 13, 22, 23, 28, 29, 35–39, 41, 46, 47) and the freeze
overwrite guard."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fixture_corpus import fixture_corpus
from tracebench.score import score_corpus

from seq2causebench import annotate as an
from seq2causebench import discover as dc
from seq2causebench import freeze as fz
from seq2causebench import prepare as pp
from seq2causebench import pretrain as pt
from seq2causebench import report as rp
from seq2causebench import scoresweep as ss
from seq2causebench import seqscore as sq
from seq2causebench import sweep as sw
from seq2causebench.constants import CLI_SHIPPED_RULE, GRAINS, RESULTS_JSON, RUN_DIR, VAL_TABLE_JSON
from seq2causebench.record import read_json, write_json
from seq2causebench.staging import StagingRefusal

PRETRAIN = [
    "--ordering",
    "end",
    "--grain",
    "session",
    "--max-len",
    "24",
    "--n-layers",
    "1",
    "--d-model",
    "16",
    "--n-heads",
    "2",
    "--ff-mult",
    "2",
    "--dropout",
    "0.0",
    "--rope-theta",
    "10000",
    "--lr",
    "3e-3",
    "--adam-beta1",
    "0.9",
    "--adam-beta2",
    "0.95",
    "--warmup-steps",
    "2",
    "--lr-schedule",
    "cosine",
    "--weight-decay",
    "0.0",
    "--batch-size",
    "16",
    "--steps",
    "8",
    "--grad-clip",
    "1.0",
    "--amp",
    "none",
    "--val-every",
    "4",
    "--val-batches",
    "2",
    "--checkpoint-every",
    "4",
    "--entropy-order",
    "1",
    "--seed",
    "0",
    "--device",
    "cpu",
    "--replica",
    "local-test",
    "--aws-profile-name",
    "none",
]
RULE = [
    "--shipped-method",
    "percentile",
    "--shipped-per-lag",
    "false",
    "--shipped-decay",
    "true",
    "--shipped-decay-type",
    "exponential",
    "--shipped-decay-rate",
    "0.3",
    "--shipped-exponent",
    "0.5",
    "--shipped-floor",
    "none",
    "--shipped-min-group-size",
    "8",
]
PROBE = [
    "--max-len",
    "24",
    "--max-lag",
    "6",
    "--sequence-sample",
    "head",
    "--memory-cap-gb",
    "2",
    "--probe-amp",
    "none",
    "--rung",
    "xs",
    "--prior-rung-record",
    "smallest-rung",
    "--replica",
    "local-test",
    "--aws-profile-name",
    "none",
    "--seed",
    "0",
    "--device",
    "cpu",
]
TAUS = ["1e-6", "1e-4", "1e-2", "1e-1"]
GRANGER_TAUS = ["1e-3", "1e-2", "1e-1"]
QUANTILES = ["0.5", "0.9"]
PROBES = ["core", "cli-full", "saliency"]
CORE_CELLS = (
    "trace/core/shipped/frozen",
    "trace/core/fixed-kl/frozen",
    "baseline/granger/shipped/frozen",
)
CLI_CELLS = (
    "trace/cli/shipped/frozen",
    "trace/cli/shipped/shipped",
    "trace/cli/fixed-kl/frozen",
    "trace/cli/fixed-kl/shipped",
)


def _git(args, cwd, env=None):
    e = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x",
        **(env or {}),
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=e
    ).stdout.strip()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Everything up to and including a committed freeze, shared by the tests below."""
    root = tmp_path_factory.mktemp("m6")
    corpus = fixture_corpus("latent")
    ledger = root / "staging-ledger.json"
    write_json(ledger, {"schema": "seq2causebench/staging-ledger@1", "entries": []})
    assert (
        pp.main(
            [
                "--corpus",
                str(corpus),
                "--ordering",
                "end",
                "--grain",
                "session",
                "--max-len",
                "24",
                "--entropy-order",
                "1",
                "--output-folder",
                str(root / "prep"),
            ]
        )
        == 0
    )
    assert (
        pt.main(
            [
                "--corpus",
                str(corpus),
                "--prepare-record",
                str(root / "prep"),
                *PRETRAIN,
                "--output-folder",
                str(root / "pre"),
            ]
        )
        == 0
    )
    model = str(root / "pre" / "model")
    probe = [*PROBE, "--staging-ledger", str(ledger)]
    assert (
        sw.main(
            [
                "--corpus",
                str(corpus),
                "--ordering",
                "end",
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
                *RULE,
                *probe,
                "--output-folder",
                str(root / "sweep"),
            ]
        )
        == 0
    )
    assert (
        ss.main(
            [
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
                "--output-folder",
                str(root / "scoresweep"),
            ]
        )
        == 0
    )
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    pre_results = root / "pre" / RUN_DIR / RESULTS_JSON
    assert (
        fz.main(
            [
                "--val-tables",
                str(root / "scoresweep" / VAL_TABLE_JSON),
                "--pretrain-results",
                str(pre_results),
                "--alt-val-tables",
                "",
                "--checkpoint-choice",
                "last",
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--seed",
                "0",
                "--freezes-dir",
                str(repo / "freezes"),
                "--output-folder",
                str(root / "freeze-run"),
            ]
        )
        == 0
    )
    freeze = next((repo / "freezes").glob("*-xs-latent-s0.json"))
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
        "repo": repo,
        "ledger": ledger,
        "probe": probe,
        "freeze_doc": read_json(freeze),
        "pretrain_results": pre_results,
    }


def _discover(w, probe, noise, cells, split, out, freeze=None, extra=()):
    fd = w["freeze_doc"]["cells"]
    keys = [f"{c}/request" for c in cells]
    cN = {(fd[k]["c"], fd[k]["N"], fd[k]["g"]) for k in keys}
    assert len(cN) == 1, f"the cells of this pass froze at different (c, N): {cN}"
    c, N, g = cN.pop()
    specs = [
        f"{k}=" + ("shipped" if k.endswith("/shipped/request") else str(fd[k]["tau"])) for k in keys
    ]
    return dc.main(
        [
            "--corpus",
            str(w["corpus"]),
            "--ordering",
            "end",
            "--grain",
            "request",
            "--split",
            split,
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
            str(freeze or ""),
            *w["probe"],
            *extra,
            "--output-folder",
            str(out),
        ]
    )


# --- sweep / scoresweep / freeze --------------------------------------------------------------------
def test_sweep_and_val_table(world):
    r = read_json(world["root"] / "sweep" / RUN_DIR / RESULTS_JSON)
    # core: all + real, cli-full: all + real, saliency: all only -> 5 passes x 2 contexts x 1 N
    assert (
        r["status"] == "ok"
        and r["n_cells"] == 5 * 2
        and all(c["g"] == min(3, c["c"]) for c in r["cells"])
    )
    assert r["shipped_rule"] == CLI_SHIPPED_RULE and r["staging"]["form"] == "smallest-rung"
    assert sorted(Path(world["root"] / "sweep").glob("scores-core-all-c*-N2-request.npz"))
    assert sorted(Path(world["root"] / "sweep").glob("shippedcut-cli-full-all-c*-N2-request.json"))
    assert sorted(Path(world["root"] / "sweep").glob("scores-saliency-all-c*-N0-request.npz"))
    t = read_json(world["root"] / "scoresweep" / VAL_TABLE_JSON)
    assert t["tool_version"] == "0.3.0" and t["config_hash"] == "fixture-latent-0"
    cuts = {(row["arm"], row["path"], row["cut"]) for row in t["cells"]}
    assert ("trace/cli", "shipped", "shipped") in cuts and ("trace/core", "fixed", "frozen") in cuts
    assert ("baseline/saliency", "none", "frozen") in cuts and (
        "baseline/granger",
        "fixed",
        "frozen",
    ) in cuts
    assert all(set(row["directed"]) >= {"precision", "recall", "f1", "shd"} for row in t["cells"])
    sal = [row for row in t["cells"] if row["arm"] == "baseline/saliency"]
    assert (
        all(row["tau_source"].startswith("quantile") for row in sal)
        and len({row["tau"] for row in sal}) >= 1
    )
    cov = next(iter(t["coverage"].values()))
    assert (
        0 <= cov["coverage"]["reachable_recall_ceiling"] <= 1
        and cov["coverage"]["universe_ordered_pairs"] > 0
    )


def test_sweep_refuses_test_and_out_of_order_rung(world, tmp_path):
    with pytest.raises(sw.SweepRefusal):
        sw.main(
            [
                "--corpus",
                str(world["corpus"]),
                "--ordering",
                "end",
                "--grains",
                "request",
                "--split",
                "test",
                "--model",
                world["model"],
                "--probes",
                "core",
                "--noises",
                "all",
                "--contexts",
                "2",
                "--particles-grid",
                "2",
                "--guidance",
                "3",
                "--num-sequences",
                "0",
                *RULE,
                *world["probe"],
                "--output-folder",
                str(tmp_path / "s"),
            ]
        )
    assert (
        read_json(tmp_path / "s" / RUN_DIR / RESULTS_JSON)["status"] == "failed"
    )  # the refusal leaves a record
    probe_s = [x if x != "xs" else "s" for x in world["probe"]]  # rung s without a prior record
    with pytest.raises(StagingRefusal):
        sw.main(
            [
                "--corpus",
                str(world["corpus"]),
                "--ordering",
                "end",
                "--grains",
                "request",
                "--split",
                "val",
                "--model",
                world["model"],
                "--probes",
                "core",
                "--noises",
                "all",
                "--contexts",
                "2",
                "--particles-grid",
                "2",
                "--guidance",
                "3",
                "--num-sequences",
                "0",
                *RULE,
                *probe_s,
                "--output-folder",
                str(tmp_path / "s2"),
            ]
        )


def test_freeze_rule_and_refuses_overwrite(world, tmp_path):
    f = world["freeze_doc"]
    for cell in CORE_CELLS + CLI_CELLS:
        assert f"{cell}/request" in f["cells"], cell
    t = read_json(world["root"] / "scoresweep" / VAL_TABLE_JSON)
    for key, cell in f["cells"].items():
        arm, path, cut, grain = key.rsplit("/", 3)
        if cut == "shipped":
            sib = f["cells"][cell["inherits"]]
            assert (cell["c"], cell["N"], cell["g"]) == (sib["c"], sib["N"], sib["g"]) and cell[
                "tau"
            ] is None
            continue
        rows = [
            r
            for r in t["cells"]
            if (r["arm"], r["path"], r["cut"], r["grain"]) == (arm, path, cut, grain)
        ]
        best = max(r["directed"]["f1"] for r in rows)
        ties = [r for r in rows if r["directed"]["f1"] == best]
        pick = min(
            ties, key=lambda r: (r["N"], r["c"], -r["tau"])
        )  # smaller N, smaller c, larger tau
        assert (cell["tau"], cell["c"], cell["N"]) == (pick["tau"], pick["c"], pick["N"]) and cell[
            "val_directed_f1"
        ] == best
    assert (
        f["model_sha256"] == read_json(world["pretrain_results"])["model_sha256"]
        and f["corpus_id"] == world["corpus"].name
    )
    assert f["checkpoint"]["choice"] == "last" and f["checkpoint"]["trigger"]["fired"] in (
        True,
        False,
    )
    with pytest.raises(fz.FreezeExists):
        fz.main(
            [
                "--val-tables",
                str(world["root"] / "scoresweep" / VAL_TABLE_JSON),
                "--pretrain-results",
                str(world["pretrain_results"]),
                "--alt-val-tables",
                "",
                "--checkpoint-choice",
                "last",
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--seed",
                "0",
                "--freezes-dir",
                str(world["repo"] / "freezes"),
                "--output-folder",
                str(tmp_path / "again"),
            ]
        )


def test_freeze_checkpoint_trigger(world, tmp_path):
    """A pretrain record whose final val loss exceeds 1.01x its minimum refuses to freeze the last
    checkpoint without the argmin sweep."""
    pre = read_json(world["pretrain_results"])
    fired = dict(pre, val_final_over_min=1.05)
    write_json(tmp_path / "fired.json", fired)
    with pytest.raises(fz.FreezeRefusal, match="trigger fired"):
        fz.main(
            [
                "--val-tables",
                str(world["root"] / "scoresweep" / VAL_TABLE_JSON),
                "--pretrain-results",
                str(tmp_path / "fired.json"),
                "--alt-val-tables",
                "",
                "--checkpoint-choice",
                "last",
                "--rung",
                "s",
                "--variant",
                "latent",
                "--seed",
                "1",
                "--freezes-dir",
                str(tmp_path / "fr"),
                "--output-folder",
                str(tmp_path / "run"),
            ]
        )
    quiet = dict(pre, val_final_over_min=1.0)
    write_json(tmp_path / "quiet.json", quiet)
    with pytest.raises(fz.FreezeRefusal, match="did not fire"):
        fz.main(
            [
                "--val-tables",
                str(world["root"] / "scoresweep" / VAL_TABLE_JSON),
                "--pretrain-results",
                str(tmp_path / "quiet.json"),
                "--alt-val-tables",
                "",
                "--checkpoint-choice",
                "argmin",
                "--rung",
                "s",
                "--variant",
                "latent",
                "--seed",
                "1",
                "--freezes-dir",
                str(tmp_path / "fr"),
                "--output-folder",
                str(tmp_path / "run2"),
            ]
        )


# --- discover: freeze, staging and the test read -------------------------------------------------------------
def test_discover_test_requires_freeze(world, tmp_path):
    with pytest.raises(dc.FreezeRefusal, match="needs --freeze"):
        _discover(world, "core", "all", CORE_CELLS, "test", tmp_path / "nofreeze", freeze=None)


def test_discover_refuses_a_cell_outside_its_pass(world, tmp_path):
    with pytest.raises(ValueError, match="not a read-out of pass"):
        _discover(
            world, "core", "real", ("trace/core/shipped/frozen",), "val", tmp_path / "wrongpass"
        )


def test_freeze_commit_assert(world, tmp_path):
    w = world
    loose = w["repo"] / "freezes" / "loose.json"
    shutil.copy(w["freeze"], loose)
    with pytest.raises(dc.FreezeRefusal, match="commit it before the test read"):
        _discover(w, "core", "all", CORE_CELLS, "test", tmp_path / "loose", freeze=loose)
    loose.unlink()
    late = w["repo"] / "freezes" / "late.json"
    shutil.copy(w["freeze"], late)
    _git(["add", "-A"], w["repo"])
    _git(
        ["commit", "-q", "-m", "late"],
        w["repo"],
        env={
            "GIT_COMMITTER_DATE": "2035-01-01T00:00:00Z",
            "GIT_AUTHOR_DATE": "2035-01-01T00:00:00Z",
        },
    )
    with pytest.raises(dc.FreezeRefusal, match="started at"):
        _discover(w, "core", "all", CORE_CELLS, "test", tmp_path / "late", freeze=late)
    _git(["rm", "-q", "--cached", str(late)], w["repo"])
    late.unlink()
    _git(["commit", "-q", "-m", "rm"], w["repo"])
    outside = tmp_path / "outside.json"
    shutil.copy(w["freeze"], outside)
    with pytest.raises(dc.FreezeRefusal):
        _discover(w, "core", "all", CORE_CELLS, "test", tmp_path / "outside", freeze=outside)
    with pytest.raises(dc.FreezeRefusal, match="command line says"):
        _discover(
            w,
            "core",
            "all",
            CORE_CELLS,
            "test",
            tmp_path / "wrongc",
            freeze=w["freeze"],
            extra=["--context", "9", "--guidance", "3"],
        )


def test_discover_test_read_annotate_and_seqscore(world, tmp_path):
    w = world
    results = tmp_path / "results"
    reads = {}
    for probe, noise, cells in (
        ("core", "all", CORE_CELLS),
        ("cli-full", "all", CLI_CELLS[:2]),
        ("cli-full", "real", ("trace/cli/fixed/frozen",)),
    ):
        out = tmp_path / f"read-{probe}-{noise}"
        assert _discover(w, probe, noise, cells, "test", out, freeze=w["freeze"]) == 0
        r = read_json(out / RUN_DIR / RESULTS_JSON)
        assert (
            r["status"] == "ok" and r["freeze"]["sha"] and r["staging"]["form"] == "smallest-rung"
        )
        assert (out / f"scores-{probe}-{noise}-request.npz").exists() and (
            out / f"matrices-{probe}-{noise}-request.npz"
        ).exists()
        assert (out / "sequences-request.json").exists()
        m = np.load(out / f"matrices-{probe}-{noise}-request.npz")
        assert set(m.files) >= {"seq", "j", "q"} and (m["q"] > m["j"]).all()
        for cell in cells:
            key = f"{cell}/request"
            arm, path, cut = cell.rsplit("/", 2)
            slug = arm.replace("/", "-")
            assert (out / f"prediction-request-{slug}-{path}-{cut}.json").exists()
            assert r["cells"][key]["cut"] == cut
            if cut == "shipped":
                assert (
                    out / f"shippedcut-request-{slug}-{path}.json"
                ).exists() and "tau_by_lag" in r["cells"][key]
                assert r["cells"][key]["threshold_finite"] in (True, False)
            else:
                assert (out / f"ranking-request-{slug}-{path}-lag3.json").exists() and r["cells"][
                    key
                ]["per_lag_files"] == 3
                assert "n_cell_collapsed" in r["corrupted_cells"][path] if path != "none" else True
            res = results / "xs" / "latent" / "seed=0" / arm / path / cut / "request"
            assert (
                an.main(
                    [
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
                )
                == 0
            )
            a = read_json(res / "annotate.json")
            assert (
                a["structural_limitation"]["assumption"] == "causal_sufficiency"
                and a["structural_limitation"]["truth_bidirected_edges"] == 2
            )
            assert a["score"]["bidirected"]["recall"] == 0.0
            direct = score_corpus(
                w["corpus"],
                read_json(out / f"prediction-request-{slug}-{path}-{cut}.json"),
                grain="request",
            )
            assert json.dumps(read_json(res / "score.json"), sort_keys=True) == json.dumps(
                direct, sort_keys=True
            )  # scenario 28
            assert (
                a["causal_validity"]["value"] is None and "ADMG" in a["causal_validity"]["reason"]
            )  # scenario 5 on latent
            assert (
                a["unreachable_tokens"]["unreachable"] == 2
                and a["coverage"]["reachable_recall_ceiling"] <= 1
            )
            assert a["oracle"]["in_regime"] in (True, False) and a["memory"]["cap_gb"] == 2.0
            assert (
                "directed_predictions_on_bidirected_truth_pairs" in a["confounded_pairs"]
            )  # scenario 38
            assert (a["empty_prediction"] is None) == (
                len(direct) > 0 and a["score"]["directed"]["tp"] + a["score"]["directed"]["fp"] > 0
            )
            if cut == "shipped":
                assert a["shipped_cut"]["rule"] == CLI_SHIPPED_RULE and a["per_lag_recall"] == {}
            else:
                assert set(a["per_lag_recall"]) == {"1", "2", "3"}
            # the per-sequence axis (scenarios 35, 36)
            assert (
                sq.main(
                    [
                        "--corpus",
                        str(w["corpus"]),
                        "--run-dir",
                        str(out),
                        "--cell",
                        key,
                        "--max-lag",
                        "6",
                        "--output-folder",
                        str(res / "seq"),
                    ]
                )
                == 0
            )
            s = read_json(res / "seq" / "seqscore.json")
            assert s["axis"] == "per-sequence" and s["grain"] == "request"
            assert s["n_sequences"] + s["n_skipped_short_by_read"] == 24 == s["n_sequences_split"]
            assert (
                0 <= s["scoreable_fraction"] <= 1
                and "predict_all" in s
                and s["predict_all"]["pooled"] is not None
            )
            assert "ancestor_directed" in s["pooled"] and s["n_links_lost"] >= 0
            shutil.copy(res / "seq" / "seqscore.json", res / "seqscore.json")
        reads[(probe, noise)] = out
    world["results"] = results


# --- report ------------------------------------------------------------------------------------------------
def _five_seeds(results, model_override=None):
    """Fabricate seeds 1–4 from seed 0 (the report needs five; the fixture has one)."""
    base = results / "xs" / "latent" / "seed=0"
    for k in range(1, 5):
        dst = results / "xs" / "latent" / f"seed={k}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(base, dst)
        for ann in dst.rglob("annotate.json"):
            a = read_json(ann)
            a["seed"] = k
            a["score"]["directed"]["f1"] = (
                float(a["score"]["directed"]["f1"]) + 0.001 * k
            )  # a little seed dispersion for the t-test
            if model_override and model_override in str(ann):
                a["model_sha256"] = "f" * 64
            write_json(ann, a)
            sc = ann.parent / "score.json"
            s = read_json(sc)
            s["directed"]["f1"] = a["score"]["directed"]["f1"]
            write_json(sc, s)


def _reasons(tmp_path, results):
    present = {
        str(p.relative_to(results / "xs" / "latent" / "seed=0").parent)
        for p in (results / "xs" / "latent" / "seed=0").rglob("annotate.json")
    }
    absent = {}
    for arm in ("trace/core", "trace/cli", "trace/cli-atomic"):
        for grain in GRAINS:
            key = f"{arm}/shipped/frozen/{grain}"
            if key not in present:
                absent[key] = "not read in this test"
    for key in present:
        for grain in GRAINS:
            other = key.rsplit("/", 1)[0] + f"/{grain}"
            if other not in present:
                absent[other] = "not read in this test"
    p = tmp_path / "reasons.json"
    write_json(p, absent)
    return p


def test_report_five_seed_refusal_and_pairs(world, tmp_path):
    results = world["results"]
    reasons = _reasons(tmp_path, results)
    pairs = [
        "trace/core/shipped/frozen:trace/cli/shipped/frozen",
        "trace/cli/shipped/shipped:trace/cli/shipped/frozen",
        "trace/core/fixed-kl/frozen:trace/core/shipped/frozen",
    ]
    for k in range(1, 5):
        shutil.rmtree(results / "xs" / "latent" / f"seed={k}", ignore_errors=True)
    with pytest.raises(rp.ReportRefusal, match="at least 5"):
        rp.main(
            [
                "--results-dir",
                str(results),
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--pairs",
                *pairs,
                "--reasons",
                str(reasons),
                "--output-folder",
                str(tmp_path / "r1"),
            ]
        )
    _five_seeds(results)
    with pytest.raises(
        rp.ReportRefusal, match="absent on every seed"
    ):  # scenario 7: no reason given
        rp.main(
            [
                "--results-dir",
                str(results),
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--pairs",
                *pairs,
                "--reasons",
                "",
                "--output-folder",
                str(tmp_path / "r2"),
            ]
        )
    with pytest.raises(rp.ReportRefusal, match="scenario 39"):  # core vs a shipped cut
        rp.main(
            [
                "--results-dir",
                str(results),
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--pairs",
                "trace/core/shipped/frozen:trace/cli/shipped/shipped",
                "--reasons",
                str(reasons),
                "--output-folder",
                str(tmp_path / "r2b"),
            ]
        )
    assert (
        rp.main(
            [
                "--results-dir",
                str(results),
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--pairs",
                *pairs,
                "--reasons",
                str(reasons),
                "--output-folder",
                str(tmp_path / "r3"),
            ]
        )
        == 0
    )
    rep = read_json(tmp_path / "r3" / "xs-latent.json")
    assert (
        "trace/core/shipped/frozen/request" in rep["cells"]
        and "trace/cli/shipped/shipped/request" in rep["cells"]
    )
    assert rep["n_absent"] > 0
    cell = rep["cells"]["trace/core/shipped/frozen/request"]
    assert (
        cell["n_seeds"] == 5
        and cell["metrics"]["directed.f1"]["n_seeds"] == 5
        and "reason" in cell["metrics"]["causal_validity.sid"]
    )
    assert (
        cell["structural_limitation"]["truth_bidirected_edges"] == [2] * 5
        and "corrupted_cells.total" in cell["metrics"]
    )
    md = (tmp_path / "r3" / "xs-latent.md").read_text()
    assert (
        "structural: causal_sufficiency" in md
        and "paired differences" in md
        and "type-level axis" in md
    )
    pair = rep["pairs"]["trace/core/shipped/frozen:trace/cli/shipped/frozen/request"]
    assert pair["seeds"] == [0, 1, 2, 3, 4] and pair["metrics"]["directed.f1"]["n_seeds"] == 5
    t = pair["tests"]["directed.f1"]
    assert t["test"] == "paired t-test" and t["df"] == 4 and t["n"] == 5 and "mean_diff" in t
    assert all(tr[0] == world["corpus"].name for tr in pair["triples"])
    cutrow = rep["pairs"]["trace/cli/shipped/shipped:trace/cli/shipped/frozen/request"]
    assert cutrow["seeds"] == [0, 1, 2, 3, 4]
    # per-sequence tables stand alone (scenario 35)
    ps = read_json(tmp_path / "r3" / "xs-latent-perseq.json")
    assert ps["axis"] == "per-sequence" and "trace/core/shipped/frozen/request" in ps["cells"]
    assert "scoreable_fraction" in ps["cells"]["trace/core/shipped/frozen/request"]["metrics"]
    for cell in ps["cells"].values():  # no type-level metric key inside the per-sequence document
        assert not (set(rp.METRICS) & set(cell["metrics"]))
    psmd = (tmp_path / "r3" / "xs-latent-perseq.md").read_text()
    assert "never merged" in psmd


def test_report_paired_cells_model_hash(world, tmp_path):
    results = world["results"]
    reasons = _reasons(tmp_path, results)
    _five_seeds(results, model_override="trace/core/shipped")
    with pytest.raises(rp.ReportRefusal, match="model_sha256"):
        rp.main(
            [
                "--results-dir",
                str(results),
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--pairs",
                "trace/core/shipped/frozen:trace/cli/shipped/frozen",
                "--reasons",
                str(reasons),
                "--output-folder",
                str(tmp_path / "r"),
            ]
        )
    _five_seeds(results)
