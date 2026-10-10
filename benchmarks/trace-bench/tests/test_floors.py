"""The floor arm class (plans/floors.md, pre-registered 2026-10-09; RUN.md D-SB-15 / D-SB-16): both
model-free floors run on the fixture end to end — validation read → `scoresweep` on the floor grid
→ a floor freeze in a git repository → the test read under it → `annotate` → `report` with the
model-free pair exemption — with the known answer of the topology arm, the bigram score by hand,
and the refusals around them."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fixture_corpus import ALPHABET, call_edges, fixture_corpus
from tracebench.constants import OUTCOME_NAMES

from seq2causebench import annotate as an
from seq2causebench import floorprior as fp
from seq2causebench import floors as fl
from seq2causebench import freeze as fz
from seq2causebench import report as rp
from seq2causebench import scoresweep as ss
from seq2causebench.constants import GRAINS, RESULTS_JSON, RUN_DIR, VAL_TABLE_JSON
from seq2causebench.freezecheck import FreezeRefusal
from seq2causebench.record import read_json, write_json

FLOOR_TAUS = ["0", "0.1", "0.3", "0.5", "0.9"]
ARMS = ["floor/topology", "floor/bigram"]
TOPO, BIGRAM = ARMS
ERR, OK, SLOW, X4 = 3, 0, 4, 1


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


def _floors(
    corpus, split, grain, out, ledger, arms=ARMS, cells=("none",), freeze="", ordering="end"
):
    return fl.main(
        [
            "--corpus",
            str(corpus),
            "--arms",
            *arms,
            "--ordering",
            ordering,
            "--grain",
            grain,
            "--split",
            split,
            "--rung",
            "xs",
            "--num-sequences",
            "0",
            "--sequence-sample",
            "head",
            "--max-len",
            "24",
            "--seed",
            "0",
            "--cells",
            *cells,
            "--freeze",
            str(freeze),
            "--prior-rung-record",
            "smallest-rung",
            "--staging-ledger",
            str(ledger),
            "--replica",
            "local-test",
            "--aws-profile-name",
            "none",
            "--output-folder",
            str(out),
        ]
    )


def _scoresweep(corpus, sweep_dir, grain, out, floor_taus=FLOOR_TAUS):
    return ss.main(
        [
            "--corpus",
            str(corpus),
            "--sweep-dir",
            str(sweep_dir),
            "--grains",
            grain,
            "--taus",
            "--granger-taus",
            "--quantiles",
            "--floor-taus",
            *floor_taus,
            "--output-folder",
            str(out),
        ]
    )


@pytest.fixture(scope="module")
def fworld(tmp_path_factory):
    """Both floors through the registered protocol on the fixture, both grains, one seed."""
    root = tmp_path_factory.mktemp("floors")
    corpus = fixture_corpus("latent")
    ledger = root / "staging-ledger.json"
    write_json(ledger, {"schema": "seq2causebench/staging-ledger@1", "entries": []})
    for grain in GRAINS:
        assert _floors(corpus, "val", grain, root / f"fval-{grain}", ledger) == 0
        assert _scoresweep(corpus, root / f"fval-{grain}", grain, root / f"fss-{grain}") == 0
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    assert (
        fz.main(
            [
                "--val-tables",
                *[str(root / f"fss-{g}" / VAL_TABLE_JSON) for g in GRAINS],
                "--pretrain-results",
                "none",
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
    freeze = next((repo / "freezes").glob("*-xs-latent-s0-floors.json"))
    _git(["add", "-A"], repo)
    _git(
        ["commit", "-q", "-m", "floor freeze"],
        repo,
        env={
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        },
    )
    doc = read_json(freeze)
    results = root / "results"
    for grain in GRAINS:
        keys = [f"{a}/none/frozen/{grain}" for a in ARMS]
        specs = [f"{k}={doc['cells'][k]['tau']}" for k in keys]
        out = root / f"ftest-{grain}"
        assert _floors(corpus, "test", grain, out, ledger, cells=specs, freeze=freeze) == 0
        for k in keys:
            assert (
                an.main(
                    [
                        "--corpus",
                        str(corpus),
                        "--run-dir",
                        str(out),
                        "--cell",
                        k,
                        "--pretrain-results",
                        "none",
                        "--per-lag-files",
                        "0",
                        "--output-folder",
                        str(results / "xs" / "latent" / "seed=0" / k),
                    ]
                )
                == 0
            )
    return {
        "root": root,
        "corpus": corpus,
        "ledger": ledger,
        "repo": repo,
        "freeze": freeze,
        "doc": doc,
        "results": results,
    }


# --- the validation read, the sweep and the freeze ------------------------------------------------
def test_val_tables_and_floor_freeze(fworld):
    for grain in GRAINS:
        r = read_json(fworld["root"] / f"fval-{grain}" / RUN_DIR / RESULTS_JSON)
        assert r["status"] == "ok" and r["stage"] == "floors" and r["arms"] == ARMS
        assert r["model_sha256"] == "none" and r["probe"] is None and r["noise"] == "none"
        assert r["c"] == r["N"] == r["g"] == 0 and r["cells"] == {} and r["freeze"] is None
        assert r["staging"] == {
            "rung": "xs",
            "previous_rung": None,
            "form": "smallest-rung",
            "stage_matched": "sweep",
        }
        assert r["inputs"][TOPO] == ["topology/prior.json", "graphs/alphabet.json"]
        assert r["inputs"][BIGRAM] == [f"views/end-{grain}/sequences/split=val"]
        assert r["n_probed"] == r["facts"][BIGRAM]["n_sequences"] > 0
        for probe in ("topology", "bigram"):
            z = np.load(fworld["root"] / f"fval-{grain}" / f"scores-{probe}-none-{grain}.npz")
            assert str(z["model_sha256"]) == "none" and str(z["ordering"]) == "end"
            assert int(z["c"]) == int(z["N"]) == int(z["g"]) == 0 and z["tokens"].dtype.kind == "U"
        t = read_json(fworld["root"] / f"fss-{grain}" / VAL_TABLE_JSON)
        assert t["model_sha256"] == "none" and t["ordering"] == "end"
        assert t["floor_taus"] == [float(x) for x in FLOOR_TAUS] and t["taus"] == []
        rows = {(row["arm"], row["tau"]) for row in t["cells"]}
        assert rows == {(a, float(x)) for a in ARMS for x in FLOOR_TAUS}
        assert all(
            row["noise"] == "none" and row["c"] == row["N"] == row["g"] == 0 for row in t["cells"]
        )
    doc = fworld["doc"]
    assert fworld["freeze"].name.endswith("-xs-latent-s0-floors.json")
    assert doc["schema"] == "seq2causebench/freeze@3" and doc["family"] == "floor"
    assert doc["ordering"] == "end" and doc["model_sha256"] == "none"
    assert doc["floor_taus"] == [float(x) for x in FLOOR_TAUS]
    assert set(doc["cells"]) == {f"{a}/none/frozen/{g}" for a in ARMS for g in GRAINS}
    assert all(
        c["grid"] == "floor_taus" and c["c"] == c["N"] == c["g"] == 0 for c in doc["cells"].values()
    )
    assert doc["diagnostics"]["model_choice"] is None and doc["diagnostics"]["oracle"] is None
    # the topology arm's full ranking is its τ = 0 cut: the reference line equals that row's F1
    t = read_json(fworld["root"] / "fss-request" / VAL_TABLE_JSON)
    row0 = next(r for r in t["cells"] if r["arm"] == TOPO and r["tau"] == 0.0)
    cell = doc["cells"][f"{TOPO}/none/frozen/request"]
    assert cell["every_scored_pair_f1"] == pytest.approx(row0["directed"]["f1"])


# --- the topology arm: the known answer -----------------------------------------------------------
def _expected_topology():
    toks = {(op, OUTCOME_NAMES[o]) for op, o in ALPHABET}
    by_op = {}
    for op, name in toks:
        by_op.setdefault(op, []).append(f"{op}:{name}")
    want = set()
    for caller, callee, p in call_edges():  # the prior's edge is callee -> caller
        for a in by_op[callee]:
            for b in by_op[caller]:
                want.add((a, b, round(p, 4)))
    return want


def test_topology_read_is_the_prior_expanded_to_token_pairs(fworld):
    table = fp.topology_read(fworld["corpus"])
    tok = [str(t) for t in table["tokens"]]
    got = {
        (tok[s], tok[d], round(float(p), 4))
        for s, d, p in zip(table["src"], table["dst"], table["score"], strict=True)
    }
    assert got == _expected_topology() and len(got) == 20
    assert table["facts"]["n_op_pairs"] == 5 and table["facts"]["n_prior_edges"] == 5
    # the evidence's four-decimal p_call equals the instantiation's numeric value
    inst = read_json(fworld["corpus"] / "instantiation.json")["topology"]["edges"]
    prior = read_json(fworld["corpus"] / "topology" / "prior.json")
    for e, i in zip(prior["edges"], inst, strict=True):
        assert fp.p_call_of(e) == pytest.approx(i["p_call"], abs=1e-4)
    # the hop1-up construction of the 2026-09-28 probe: caller -> callee flipped, max p_call per pair
    down = {}
    for e in inst:
        down[(e["caller"], e["callee"])] = max(
            down.get((e["caller"], e["callee"]), 0.0), e["p_call"]
        )
    up = {(v, u): w for (u, v), w in down.items()}
    alphabet = read_json(fworld["corpus"] / "graphs" / "alphabet.json")["tokens"]
    by_op = {}
    for t in alphabet:
        by_op.setdefault(t["op_id"], []).append(t["token"])
    hop1 = {(a, b, round(w, 4)) for (u, v), w in up.items() for a in by_op[u] for b in by_op[v]}
    assert got == hop1
    with pytest.raises(fp.PriorRefusal):
        fp.p_call_of({"from": "x", "to": "y", "evidence": "fixture call 0->2"})
    with pytest.raises(fp.PriorRefusal):
        fp.op_pairs_from_prior(
            {"edges": [{"from": "nope:_x", "to": "nope:_y", "evidence": "p_call=1"}]}, {}
        )


def test_topology_test_read_scores_the_known_answer(fworld):
    r = read_json(fworld["root"] / "ftest-request" / RUN_DIR / RESULTS_JSON)
    key = f"{TOPO}/none/frozen/request"
    assert r["cells"][key]["n_pairs"] == 20 and r["cells"][key]["per_lag_files"] == 0
    assert (
        r["freeze"]["path"] == str(fworld["freeze"]) and r["staging"]["stage_matched"] == "discover"
    )
    cell = fworld["results"] / "xs" / "latent" / "seed=0" / key
    score = read_json(cell / "score.json")
    # every p_call (0.7 … 0.9) clears the frozen τ, so the cut is the whole prior: 5 of the 6
    # truth edges at the floor lie on direct callee -> caller op pairs, 15 predictions are false
    assert r["cells"][key]["n_edges"] == 20
    assert score["directed"]["tp"] == 5 and score["directed"]["fp"] == 15
    assert score["universe"]["predictions_outside_universe"] == 0
    assert score["directed"]["precision"] == pytest.approx(0.25)
    assert score["directed"]["recall"] == pytest.approx(5 / 6)
    ann = read_json(cell / "annotate.json")
    assert (
        ann["arm_class"] == "floor" and ann["model_sha256"] == "none" and ann["probe"] == "topology"
    )
    assert ann["oracle"]["in_regime"] is None and ann["budget"] is None
    assert ann["frozen"]["c"] == 0 and ann["per_lag_recall"] == {} and ann["corrupted_cells"] == {}
    assert ann["coverage"]["n_sequences_probed"] == 0  # sample-free (plans/floors.md §5)
    assert ann["coverage"]["pairs_scored"] == 20 and ann["coverage"]["n_pairs_in_table"] == 20


# --- the bigram arm -------------------------------------------------------------------------------
class _Fake:
    view_rel = "views/end-request"

    def __init__(self, seqs):
        self._seqs = seqs

    def sequences(self, split):
        for i, (ops, outs) in enumerate(self._seqs):
            yield (f"t{i}", list(ops), list(outs), len(ops))


def test_bigram_score_by_hand():
    seqs = [
        ([4, 2, 0], [ERR, ERR, X4]),  # 4:err -> 2:err -> 0:4xx
        ([4, 2, 2, 0], [ERR, ERR, OK, OK]),  # 4:err -> 2:err -> 2:ok (within-op) -> 0:ok
        ([2, 0], [OK, OK]),  # 2:ok -> 0:ok
        ([5], [ERR]),  # too short for a pair
    ]
    t = fl.bigram_read(_Fake(seqs), "val", 0, "head", 0, 10)
    tok = [str(x) for x in t["tokens"]]
    got = {
        (tok[s], tok[d]): (float(p), int(c))
        for s, d, p, c in zip(t["src"], t["dst"], t["score"], t["count"], strict=True)
    }
    assert got == {
        ("4:err", "2:err"): (1.0, 2),  # 2 sequences with the pair of the 2 holding 4:err
        ("2:err", "0:4xx"): (0.5, 1),  # 1 of the 2 sequences holding 2:err
        ("2:ok", "0:ok"): (1.0, 2),  # the within-op pair 2:err -> 2:ok is dropped
    }
    assert t["facts"] == {
        "n_sequences": 4,
        "n_skipped_short": 1,
        "n_tokens": 6,
        "inputs": ["views/end-request/sequences/split=val"],
    }
    # truncation to the arms' max_len, and the head sample
    t2 = fl.bigram_read(_Fake(seqs), "val", 0, "head", 0, 2)
    tok2 = [str(x) for x in t2["tokens"]]
    got2 = {
        (tok2[s], tok2[d]): float(p)
        for s, d, p in zip(t2["src"], t2["dst"], t2["score"], strict=True)
    }
    assert got2 == {("4:err", "2:err"): 1.0, ("2:ok", "0:ok"): 1.0}
    t3 = fl.bigram_read(_Fake(seqs), "val", 2, "head", 0, 10)  # the first two sequences only
    assert t3["facts"]["n_sequences"] == 2 and len(t3["src"]) == 3


def test_bigram_arm_opens_views_only(fworld, tmp_path, monkeypatch):
    import seq2causebench.corpus as cm

    seen = []
    real = cm.open_for_method

    def spy(corpus_dir, rel, mode="rb"):
        seen.append(str(rel))
        return real(corpus_dir, rel, mode)

    monkeypatch.setattr(cm, "open_for_method", spy)
    assert (
        _floors(fworld["corpus"], "val", "session", tmp_path / "b", fworld["ledger"], arms=[BIGRAM])
        == 0
    )
    assert seen and all(p.startswith("views/end-session/") for p in seen)
    r = read_json(tmp_path / "b" / RUN_DIR / RESULTS_JSON)
    assert r["arms"] == [BIGRAM] and TOPO not in r["inputs"]
    assert (tmp_path / "b" / "scores-bigram-none-session.npz").exists()
    assert not (tmp_path / "b" / "scores-topology-none-session.npz").exists()


def test_topology_arm_never_constructs_corpus(fworld, tmp_path, monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("the topology floor opened a view")

    monkeypatch.setattr(fl, "Corpus", Boom)
    assert (
        _floors(fworld["corpus"], "val", "request", tmp_path / "t", fworld["ledger"], arms=[TOPO])
        == 0
    )
    r = read_json(tmp_path / "t" / RUN_DIR / RESULTS_JSON)
    assert r["arms"] == [TOPO] and r["n_probed"] == 0 and list(r["inputs"]) == [TOPO]
    prior_src = Path(fp.__file__).read_text(encoding="utf-8")
    assert "from .corpus" not in prior_src and "Corpus(" not in prior_src  # no door to a view


# --- refusals -------------------------------------------------------------------------------------
def test_freeze_refusals(tmp_path):
    with pytest.raises(fz.FreezeRefusal, match="model-free"):
        fz.pretrain_facts("none", {"model_sha256": "a" * 64})
    rec = tmp_path / "pre.json"
    write_json(rec, {"model_sha256": "a" * 64})
    with pytest.raises(fz.FreezeRefusal, match="none"):
        fz.pretrain_facts(str(rec), {"model_sha256": "none"})
    assert fz.pretrain_facts("none", {"model_sha256": "none"})["oracle"] is None
    with pytest.raises(fz.FreezeRefusal, match="D-SB-16"):
        fz.family_of({f"{TOPO}/none/frozen/request": {}, "trace/core/shipped/frozen/request": {}})
    assert fz.family_of({"trace/core/shipped/frozen/request": {}}) == "model"
    assert fz.family_of({f"{BIGRAM}/none/frozen/session": {}}) == "floor"
    assert fz.freeze_name("2026-01-01", "xs", "latent", 0, "floor", "start") == (
        "2026-01-01-xs-latent-s0-floors-start.json"
    )
    assert (
        fz.freeze_name("2026-01-01", "xs", "latent", 0, "model", "end")
        == "2026-01-01-xs-latent-s0.json"
    )
    assert (
        fz.freeze_name("2026-01-01", "xs", "latent", 0, "model", None)
        == "2026-01-01-xs-latent-s0.json"
    )
    assert fz.grid_of(BIGRAM, "grid") == "floor_taus" and fz.grid_of("trace/core", "grid") == "taus"


def test_scoresweep_refuses_an_empty_floor_grid(fworld, tmp_path):
    with pytest.raises(ValueError, match="floor-taus"):
        _scoresweep(
            fworld["corpus"],
            fworld["root"] / "fval-request",
            "request",
            tmp_path / "s",
            floor_taus=[],
        )


def test_test_read_refusals(fworld, tmp_path):
    doc, corpus, ledger = fworld["doc"], fworld["corpus"], fworld["ledger"]
    keys = [f"{a}/none/frozen/request" for a in ARMS]
    good = [f"{k}={doc['cells'][k]['tau']}" for k in keys]
    with pytest.raises(FreezeRefusal, match="scenario 13"):  # no freeze on a test read
        _floors(corpus, "test", "request", tmp_path / "a", ledger, cells=good, freeze="")
    loose = tmp_path / "loose" / fworld["freeze"].name  # the same document outside any repository
    loose.parent.mkdir()
    shutil.copy(fworld["freeze"], loose)
    with pytest.raises(FreezeRefusal, match="git repository"):
        _floors(corpus, "test", "request", tmp_path / "b", ledger, cells=good, freeze=loose)
    bad_tau = [f"{keys[0]}={doc['cells'][keys[0]]['tau'] + 0.01}", good[1]]
    with pytest.raises(FreezeRefusal, match="command line says"):
        _floors(
            corpus,
            "test",
            "request",
            tmp_path / "c",
            ledger,
            cells=bad_tau,
            freeze=fworld["freeze"],
        )
    with pytest.raises(FreezeRefusal, match="view"):  # the freeze was swept on the end view
        _floors(
            corpus,
            "test",
            "request",
            tmp_path / "d",
            ledger,
            cells=good,
            freeze=fworld["freeze"],
            ordering="start",
        )
    model_doc = {**doc, "family": "model"}
    p = fworld["repo"] / "freezes" / "2026-01-02-xs-latent-s0.json"
    write_json(p, model_doc)
    _git(["add", "-A"], fworld["repo"])
    _git(
        ["commit", "-q", "-m", "a model freeze"],
        fworld["repo"],
        env={
            "GIT_COMMITTER_DATE": "2026-01-02T00:00:00Z",
            "GIT_AUTHOR_DATE": "2026-01-02T00:00:00Z",
        },
    )
    with pytest.raises(FreezeRefusal, match="not a floor freeze"):
        _floors(corpus, "test", "request", tmp_path / "e", ledger, cells=good, freeze=p)
    with pytest.raises(ValueError, match="cut"):  # the registry has no shipped cut for a floor
        fl.parse_floor_cells([f"{TOPO}/none/shipped/request=0.1"], "request", ARMS)
    with pytest.raises(ValueError, match="grain"):
        fl.parse_floor_cells([f"{TOPO}/none/frozen/session=0.1"], "request", ARMS)
    with pytest.raises(ValueError, match="not one of this read's arms"):
        fl.parse_floor_cells([f"{TOPO}/none/frozen/request=0.1"], "request", [BIGRAM])
    assert fl.parse_floor_cells(["none"], "request", ARMS) == {}


# --- report: the model-free pair exemption ----------------------------------------------------------
def _fabricate(results, n_seeds=5):
    """Seeds 1–4 from seed 0 and a model-bound `trace/core/shipped/frozen` cell beside the floors."""
    base = results / "xs" / "latent" / "seed=0"
    for grain in GRAINS:
        src = base / "floor" / "topology" / "none" / "frozen" / grain
        dst = base / "trace" / "core" / "shipped" / "frozen" / grain
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        a = read_json(dst / "annotate.json")
        a.update(
            cell=f"trace/core/shipped/frozen/{grain}",
            arm="trace/core",
            path="shipped",
            arm_class="reference",
            model_sha256="a" * 64,
            probe="core",
        )
        a["oracle"]["in_regime"] = True
        write_json(dst / "annotate.json", a)
    for k in range(1, n_seeds):
        dst = results / "xs" / "latent" / f"seed={k}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(base, dst)
        for ann in dst.rglob("annotate.json"):
            a = read_json(ann)
            a["seed"] = k
            a["score"]["directed"]["f1"] = float(a["score"]["directed"]["f1"]) + 0.001 * k
            write_json(ann, a)
            sc = ann.parent / "score.json"
            s = read_json(sc)
            s["directed"]["f1"] = a["score"]["directed"]["f1"]
            write_json(sc, s)


def _report(results, out, pairs, reasons):
    return rp.main(
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
            str(out),
        ]
    )


def test_report_floor_rows_and_model_free_pairs(fworld, tmp_path):
    results = tmp_path / "results"
    shutil.copytree(fworld["results"], results)
    _fabricate(results)
    reasons = tmp_path / "reasons.json"
    write_json(
        reasons,
        {
            f"{arm}/shipped/frozen/{g}": "not read in this test"
            for arm in ("trace/cli", "trace/cli-atomic")
            for g in GRAINS
        },
    )
    pairs = [
        f"{TOPO}/none/frozen:{BIGRAM}/none/frozen",
        f"{TOPO}/none/frozen:trace/core/shipped/frozen",
    ]
    assert _report(results, tmp_path / "r", pairs, reasons) == 0
    rep = read_json(tmp_path / "r" / "xs-latent.json")
    for arm in ARMS:
        for g in GRAINS:
            cell = rep["cells"][f"{arm}/none/frozen/{g}"]
            assert cell["n_seeds"] == 5 and cell["in_regime"] == [None] * 5
            assert cell["model_sha256"] == ["none"] * 5
    core = rep["cells"]["trace/core/shipped/frozen/request"]
    assert core["in_regime"] == [True] * 5
    for key, t in rep["pairs"].items():
        assert t["model_free"] is True, key
        assert t["seeds"] == [0, 1, 2, 3, 4] and t["tests"]["directed.f1"]["n"] == 5
    assert f"{TOPO}/none/frozen:trace/core/shipped/frozen/request" in rep["pairs"]
    md = (tmp_path / "r" / "xs-latent.md").read_text()
    assert "D-SB-15" in md and f"{BIGRAM}/none/frozen/session" in md
    ps = read_json(tmp_path / "r" / "xs-latent-perseq.json")
    assert ps["cells"] == {}  # a floor has no per-sequence axis; nothing is fabricated for it
    # a model-free pair still shares corpus and seed: another corpus on one side is refused
    ann = (
        results
        / "xs"
        / "latent"
        / "seed=2"
        / "floor"
        / "bigram"
        / "none"
        / "frozen"
        / "request"
        / "annotate.json"
    )
    a = read_json(ann)
    a["corpus_id"] = "xs/latent/seed=9"
    write_json(ann, a)
    with pytest.raises(rp.ReportRefusal, match="D-SB-15"):
        _report(results, tmp_path / "r2", pairs[:1], reasons)
    # and the floor cells are required: a results tree without them needs a reason
    bare = tmp_path / "bare"
    shutil.copytree(results, bare)
    for g in GRAINS:
        shutil.rmtree(
            bare / "xs" / "latent" / "seed=0" / "floor" / "bigram" / "none" / "frozen" / g
        )
    for k in range(1, 5):
        shutil.rmtree(bare / "xs" / "latent" / f"seed={k}" / "floor" / "bigram")
    with pytest.raises(rp.ReportRefusal, match="absent on every seed"):
        _report(bare, tmp_path / "r3", pairs[1:], reasons)
    with_reason = tmp_path / "reasons2.json"
    write_json(
        with_reason,
        {**read_json(reasons), **{f"{BIGRAM}/none/frozen/{g}": "not run" for g in GRAINS}},
    )
    assert _report(bare, tmp_path / "r4", pairs[1:], with_reason) == 0
    assert json.loads((tmp_path / "r4" / "xs-latent.json").read_text())["n_absent"] == 6
