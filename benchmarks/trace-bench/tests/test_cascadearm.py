"""The external arm family (plans/cascade.md, pre-registered 2026-11-03; RUN.md D-SB-27 / D-SB-28):
both arms run on the fixture end to end through a stub CASCADE — validation read (three settings)
→ `scoresweep` on the floor grid → an external freeze in a git repository → the test read under it
(one arm, the frozen setting) → `annotate` — with the events contract checked by hand, the mask
asserted, and the refusals around them. The stub stands in for the method's own environment: it
takes the runner's arguments and returns a graph that depends on the mask and the setting."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fixture_corpus import fixture_corpus

from seq2causebench import annotate as an
from seq2causebench import cascadearm as ca
from seq2causebench import floorprior as fp
from seq2causebench import freeze as fz
from seq2causebench import scoresweep as ss
from seq2causebench.constants import GRAINS, RESULTS_JSON, RUN_DIR, VAL_TABLE_JSON
from seq2causebench.freezecheck import FreezeRefusal
from seq2causebench.record import read_json, write_json

FLOOR_TAUS = ["0", "0.1", "0.5", "0.9"]
ARMS = ["cascade/topology", "cascade/none"]
TOPO, NONE = ARMS
SETTINGS = ["1000", "10000", "60000"]

STUB = '''
import numpy as np

def cascade(events, number_of_events=None, allow_instant=True, return_alignment=False, precision=2,
            max_delay=300, allowed_pairs=None):
    """A stand-in for CASCADE: every allowed pair under a mask; without one, each type to the next
    type and, from a 10 s window on, to the type after that."""
    n = number_of_events
    g = np.zeros((n, n), dtype=int)
    if allowed_pairs is not None:
        for a, b in allowed_pairs:
            g[a, b] = 1
        if max_delay >= 100000 and len(allowed_pairs):  # a window off the grid adds one edge outside the mask
            a, b = sorted(allowed_pairs)[0]
            g[b, a] = 1
        return g
    for i in range(n - 1):
        g[i, i + 1] = 1
        if max_delay >= 10000 and i + 2 < n:
            g[i, i + 2] = 1
    return g
'''


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


def _stub_src(root):
    src = root / "stub" / "src"
    src.mkdir(parents=True)
    (src / "main.py").write_text(STUB, encoding="utf-8")
    return root / "stub"


def _cascade(
    corpus,
    split,
    grain,
    out,
    ledger,
    stub,
    arms=ARMS,
    settings=SETTINGS,
    cells=("none",),
    freeze="",
    ordering="end",
):
    return ca.main(
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
            "--max-delays-ms",
            *settings,
            "--precision",
            "2",
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
            "--cascade-python",
            sys.executable,
            "--cascade-src",
            str(stub),
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
            "--arms",
            *ARMS,
            "--output-folder",
            str(out),
        ]
    )


@pytest.fixture(scope="module")
def cworld(tmp_path_factory):
    root = tmp_path_factory.mktemp("cascade")
    corpus = fixture_corpus("latent")
    stub = _stub_src(root)
    ledger = root / "staging-ledger.json"
    write_json(ledger, {"schema": "seq2causebench/staging-ledger@1", "entries": []})
    for grain in GRAINS:
        assert _cascade(corpus, "val", grain, root / f"cval-{grain}", ledger, stub) == 0
        assert _scoresweep(corpus, root / f"cval-{grain}", grain, root / f"css-{grain}") == 0
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    assert (
        fz.main(
            [
                "--val-tables",
                *[str(root / f"css-{g}" / VAL_TABLE_JSON) for g in GRAINS],
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
    freeze = next((repo / "freezes").glob("*-xs-latent-s0-external.json"))
    _git(["add", "-A"], repo)
    _git(
        ["commit", "-q", "-m", "external freeze"],
        repo,
        env={
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        },
    )
    doc = read_json(freeze)
    results = root / "results"
    for grain in GRAINS:
        for arm in ARMS:
            key = f"{arm}/none/frozen/{grain}"
            cell = doc["cells"][key]
            out = root / f"ctest-{arm.replace('/', '-')}-{grain}"
            assert (
                _cascade(
                    corpus,
                    "test",
                    grain,
                    out,
                    ledger,
                    stub,
                    arms=[arm],
                    settings=[str(cell["N"])],
                    cells=[f"{key}={cell['tau']}"],
                    freeze=freeze,
                )
                == 0
            )
            assert (
                an.main(
                    [
                        "--corpus",
                        str(corpus),
                        "--run-dir",
                        str(out),
                        "--cell",
                        key,
                        "--pretrain-results",
                        "none",
                        "--per-lag-files",
                        "0",
                        "--output-folder",
                        str(results / "xs" / "latent" / "seed=0" / key),
                    ]
                )
                == 0
            )
    return {
        "root": root,
        "corpus": corpus,
        "stub": stub,
        "ledger": ledger,
        "repo": repo,
        "freeze": freeze,
        "doc": doc,
        "results": results,
    }


# --- the events contract ---------------------------------------------------------------------
class _Fake:
    view_rel = "views/end-request"

    def __init__(self, rows):
        self._rows = rows

    def sequences(self, split, columns=None):
        for i, (ops, outs, t0, off, dur) in enumerate(self._rows):
            yield (f"t{i}", list(ops), list(outs), len(ops), t0, list(off), list(dur))


def test_events_are_span_ends_in_sorted_milliseconds_with_dense_types():
    ms = 1_000_000
    rows = [
        # ops, outcomes, start_ns, offsets (ns), durations (ns): ends at 5, 3, 9 ms
        ([4, 2, 0], [3, 3, 1], 100 * ms, [0, 1 * ms, 4 * ms], [5 * ms, 2 * ms, 5 * ms]),
        ([2, 0], [0, 0], 50 * ms, [0, 0], [1 * ms, 1 * ms]),  # both end at 51 ms
        ([5], [3], 10 * ms, [0], [0]),  # one token: counted short, still an event
    ]
    ev = ca.events_read(_Fake(rows), "val", 0, "head", 0, 10)
    tok = [str(t) for t in ev["tokens"]]
    assert tok == ["0:ok", "0:4xx", "2:ok", "2:err", "4:err", "5:err"]  # sorted by op * 8 + outcome
    seq = [tok[i] for i in ev["event"]]
    assert seq == ["5:err", "2:ok", "0:ok", "2:err", "4:err", "0:4xx"]
    assert ev["timestamp"].tolist() == [0, 41, 41, 93, 95, 99]  # relative to the earliest end
    assert ev["op_of_token"].tolist() == [0, 0, 2, 2, 4, 5]
    assert ev["facts"]["n_sequences"] == 3 and ev["facts"]["n_skipped_short"] == 1
    assert ev["facts"]["n_events"] == 6 and ev["facts"]["n_tokens"] == 6
    # the arms' truncation: max_len 2 drops the third token of the first sequence
    ev2 = ca.events_read(_Fake(rows), "val", 0, "head", 0, 2)
    assert ev2["facts"]["n_events"] == 5 and "0:4xx" not in [str(t) for t in ev2["tokens"]]


def test_mask_is_the_topology_floors_pairs_over_the_sample(cworld):
    r = read_json(cworld["root"] / "cval-request" / RUN_DIR / RESULTS_JSON)
    z = np.load(
        cworld["root"] / "cval-request" / "scores-cascade-topology-none-c0-N1000-request.npz"
    )
    tok = [str(t) for t in z["tokens"]]
    got = {(tok[s], tok[d]) for s, d in zip(z["src"], z["dst"], strict=True)}
    table = fp.topology_read(cworld["corpus"])
    ptok = [str(t) for t in table["tokens"]]
    prior = {(ptok[s], ptok[d]) for s, d in zip(table["src"], table["dst"], strict=True)}
    present = set(tok)
    want = {(a, b) for a, b in prior if a in present and b in present}
    assert got == want and len(got) == r["facts"][TOPO]["n_allowed_pairs"]
    assert r["facts"][TOPO]["n_prior_pairs"] == 20
    assert np.all(z["max_cascade-topology__none"] == 1.0) and int(z["N"]) == 1000
    assert int(z["c"]) == int(z["g"]) == 0 and str(z["model_sha256"]) == "none"
    assert len(str(z["method_sha256"])) == 64 and int(z["max_delay_ms"]) == 1000


# --- the validation read, the sweep and the freeze -------------------------------------------
def test_val_read_sweep_and_external_freeze(cworld):
    for grain in GRAINS:
        r = read_json(cworld["root"] / f"cval-{grain}" / RUN_DIR / RESULTS_JSON)
        assert r["status"] == "ok" and r["stage"] == "cascade" and r["arms"] == ARMS
        assert r["model_sha256"] == "none" and r["probe"] is None and r["noise"] == "none"
        assert r["method_patch"] == "0001-allowed-pairs" and len(r["method_sha256"]) == 64
        assert r["max_delays_ms"] == [1000, 10000, 60000] and r["N"] is None
        assert r["c"] == r["g"] == 0 and r["cells"] == {} and r["freeze"] is None
        assert r["staging"]["form"] == "smallest-rung" and r["staging"]["stage_matched"] == "sweep"
        assert r["inputs"][NONE] == [f"views/end-{grain}/sequences/split=val"]
        assert r["inputs"][TOPO][1].startswith("topology prior through floorprior")
        for arm in ARMS:
            fits = r["facts"][arm]["fits"]
            assert sorted(fits) == sorted(SETTINGS)
            assert all(f["wall_seconds"] >= 0 and f["n_types"] > 0 for f in fits.values())
            assert r["files"][arm] == [f["file"] for f in fits.values()]
        for probe in ("cascade-topology", "cascade-none"):
            for d in SETTINGS:
                z = np.load(
                    cworld["root"] / f"cval-{grain}" / f"scores-{probe}-none-c0-N{d}-{grain}.npz"
                )
                assert int(z["N"]) == int(d) and str(z["ordering"]) == "end"
        t = read_json(cworld["root"] / f"css-{grain}" / VAL_TABLE_JSON)
        assert t["model_sha256"] == "none" and t["arms"] == ARMS
        rows = {(row["arm"], row["N"], row["tau"]) for row in t["cells"]}
        assert rows == {(a, int(d), float(x)) for a in ARMS for d in SETTINGS for x in FLOOR_TAUS}
        # a hard output: every τ of one setting scores the same
        for a in ARMS:
            for d in SETTINGS:
                f1s = {
                    row["directed"]["f1"]
                    for row in t["cells"]
                    if row["arm"] == a and row["N"] == int(d)
                }
                assert len(f1s) == 1
    doc = cworld["doc"]
    assert cworld["freeze"].name.endswith("-xs-latent-s0-external.json")
    assert (
        doc["family"] == "external" and doc["model_sha256"] == "none" and doc["ordering"] == "end"
    )
    assert set(doc["cells"]) == {f"{a}/none/frozen/{g}" for a in ARMS for g in GRAINS}
    for c in doc["cells"].values():
        assert (
            c["grid"] == "floor_taus" and c["c"] == c["g"] == 0 and c["N"] in (1000, 10000, 60000)
        )
        assert c["tau"] == 0.9  # ties → larger τ; the cut keeps the whole DAG either way
        assert c["every_scored_pair_f1"] == pytest.approx(c["val_directed_f1"])
    # the stub's no-mask graph gains edges at 10 s; the freeze picks the setting by F1, ties → smaller N
    assert doc["cells"][f"{NONE}/none/frozen/request"]["N"] in (1000, 10000)


# --- the test read and annotate --------------------------------------------------------------
def test_test_read_under_the_freeze_and_annotate(cworld):
    doc = cworld["doc"]
    for grain in GRAINS:
        for arm in ARMS:
            key = f"{arm}/none/frozen/{grain}"
            out = cworld["root"] / f"ctest-{arm.replace('/', '-')}-{grain}"
            r = read_json(out / RUN_DIR / RESULTS_JSON)
            assert r["arms"] == [arm] and r["N"] == doc["cells"][key]["N"]
            assert (
                r["freeze"]["path"] == str(cworld["freeze"])
                and r["staging"]["stage_matched"] == "discover"
            )
            assert (
                r["cells"][key]["tau"] == 0.9
                and r["cells"][key]["n_edges"] == r["cells"][key]["n_pairs"]
            )
            probe = "cascade-topology" if arm == TOPO else "cascade-none"
            assert (out / f"scores-{probe}-none-{grain}.npz").exists()
            cell = cworld["results"] / "xs" / "latent" / "seed=0" / key
            ann = read_json(cell / "annotate.json")
            assert ann["arm_class"] == "external" and ann["model_sha256"] == "none"
            assert ann["probe"] == probe and ann["frozen"]["N"] == doc["cells"][key]["N"]
            assert ann["oracle"]["in_regime"] is None and ann["budget"] is None
            assert ann["coverage"]["n_sequences_probed"] == r["n_probed"] > 0
            score = read_json(cell / "score.json")
            assert score["universe"]["predictions_outside_universe"] >= 0
    # the masked arm's test edges lie inside the prior's pairs
    out = cworld["root"] / "ctest-cascade-topology-request"
    pred = read_json(next(out.glob("prediction-request-cascade-topology-*.json")))
    table = fp.topology_read(cworld["corpus"])
    ptok = [str(t) for t in table["tokens"]]
    prior = {(ptok[s], ptok[d]) for s, d in zip(table["src"], table["dst"], strict=True)}
    assert pred["directed"] and all((e["src"], e["dst"]) in prior for e in pred["directed"])


# --- refusals ---------------------------------------------------------------------------------
def test_refusals(cworld, tmp_path):
    corpus, ledger, stub, doc = cworld["corpus"], cworld["ledger"], cworld["stub"], cworld["doc"]
    key = f"{TOPO}/none/frozen/request"
    good = [f"{key}={doc['cells'][key]['tau']}"]
    frozen_n = str(doc["cells"][key]["N"])
    with pytest.raises(FreezeRefusal, match="scenario 13"):  # no freeze on a test read
        _cascade(
            corpus,
            "test",
            "request",
            tmp_path / "a",
            ledger,
            stub,
            arms=[TOPO],
            settings=[frozen_n],
            cells=good,
        )
    with pytest.raises(FreezeRefusal, match="one arm"):  # two arms on a test read
        _cascade(
            corpus,
            "test",
            "request",
            tmp_path / "b",
            ledger,
            stub,
            settings=[frozen_n],
            cells=good,
            freeze=cworld["freeze"],
        )
    other = "60000" if frozen_n != "60000" else "1000"
    with pytest.raises(FreezeRefusal, match="command line says"):  # the wrong setting
        _cascade(
            corpus,
            "test",
            "request",
            tmp_path / "c",
            ledger,
            stub,
            arms=[TOPO],
            settings=[other],
            cells=good,
            freeze=cworld["freeze"],
        )
    # the stub returns one edge outside the mask at a 100 s setting (off the grid): refused
    with pytest.raises(ca.CascadeRefusal, match="outside the mask"):
        _cascade(
            corpus, "val", "request", tmp_path / "d", ledger, stub, arms=[TOPO], settings=["100000"]
        )
    with pytest.raises(ValueError, match="cut"):
        ca.parse_cells([f"{TOPO}/none/shipped/request=0.1"], "request", ARMS)
    with pytest.raises(ValueError, match="not one of this read's arms"):
        ca.parse_cells([f"{TOPO}/none/frozen/request=0.1"], "request", [NONE])
    assert ca.parse_cells(["none"], "request", ARMS) == {}
    # the external family is frozen apart from every other family
    with pytest.raises(fz.FreezeRefusal, match="D-SB-27"):
        fz.family_of({key: {}, "trace/core/shipped/frozen/request": {}})
    with pytest.raises(fz.FreezeRefusal, match="D-SB-16"):  # the floor rule fires first
        fz.family_of({key: {}, "floor/topology/none/frozen/request": {}})
    assert fz.family_of({key: {}}) == "external"
    assert fz.freeze_name("2026-01-01", "xs", "latent", 0, "external", "end") == (
        "2026-01-01-xs-latent-s0-external.json"
    )
    assert fz.grid_of(TOPO, "grid") == "floor_taus"
    # the arm module names no non-method-readable path; the mask comes through floorprior
    src = Path(ca.__file__).read_text(encoding="utf-8")
    assert "topology/" not in src and "alphabet.json" not in src and "parent_pos" not in src
