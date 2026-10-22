"""The floors on the per-sequence axis (`plans/floors.md` / `plans/per-sequence-rules.md`, addenda
2026-10-21; D-SB-17): the re-derived read equals `discover`'s, the broadcast and adjacent-only
matrices are the floor's score at every candidate pair, the counts equal the arm's seqscore, every
refusal, and the five-seed table."""

import shutil

import numpy as np
import pytest
from test_floors import fworld  # noqa: F401  (module fixture: both floors through the protocol)
from test_m6_pipeline import _discover, world  # noqa: F401  (module fixture: model + freeze)

from seq2causebench import floorseq as fs
from seq2causebench import floorseqtable as ft
from seq2causebench import seqscore as sq
from seq2causebench.constants import RESULTS_JSON, RUN_DIR, SEQSCORE_JSON
from seq2causebench.corpus import Corpus
from seq2causebench.data import SequenceStore
from seq2causebench.record import read_json, write_json
from seq2causebench.report import ReportRefusal
from seq2causebench.vocab import Vocab

REF = "trace/core/shipped/frozen/request"
TOPO = "floor/topology/none/frozen/request"
BIGRAM = "floor/bigram/none/frozen/request"


@pytest.fixture(scope="module")
def read(world, fworld, tmp_path_factory):  # noqa: F811
    """One arm test read + its seqscore, and the floors' test read, on the same fixture corpus."""
    assert world["corpus"] == fworld["corpus"]
    root = tmp_path_factory.mktemp("floorseq")
    out = root / "read-core-all"
    assert (
        _discover(
            world,
            "core",
            "all",
            ("trace/core/shipped/frozen",),
            "test",
            out,
            freeze=world["freeze"],
        )
        == 0
    )
    ref = root / "ref-seq"
    assert (
        sq.main(
            [
                "--corpus",
                str(world["corpus"]),
                "--run-dir",
                str(out),
                "--cell",
                REF,
                "--max-lag",
                "6",
                "--output-folder",
                str(ref),
            ]
        )
        == 0
    )
    return {
        "root": root,
        "corpus": world["corpus"],
        "read": out,
        "ref": ref / SEQSCORE_JSON,
        "floor_read": fworld["root"] / "ftest-request",
    }


def _run(r, out, floor_cell=TOPO, row="broadcast", **over):
    argv = {
        "--corpus": str(r["corpus"]),
        "--read-dir": str(r["read"]),
        "--reference-cell": REF,
        "--reference-seqscore": str(r["ref"]),
        "--floor-read": str(r["floor_read"]),
        "--floor-cell": floor_cell,
        "--row": row,
        "--max-lag": "6",
        "--output-folder": str(out),
    }
    argv.update(over)
    return fs.main([x for kv in argv.items() for x in kv])


def test_stored_sequences_rederive_the_reads(read):
    disc = read_json(read["read"] / RUN_DIR / RESULTS_JSON)
    corpus = Corpus(read["corpus"], "end", "request")
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    store = SequenceStore.from_corpus(corpus, "test", vocab, 24, n=0, mode="head", seed=0)
    seqs, n_skipped = fs.stored_sequences(store, int(disc["c"]), 0)
    want = read_json(read["read"] / "sequences-request.json")["sequences"]
    assert seqs == [s["seq"] for s in want] and n_skipped == disc["n_skipped_short"]
    first, _ = fs.stored_sequences(store, int(disc["c"]), 3)
    assert first == seqs[:3]


def test_floor_matrix_by_hand():
    tok = {0: "1:ok", 1: "2:ok", 2: "1:5xx", 3: "3:ok"}
    scores = {("1:ok", "2:ok"): 0.4, ("1:ok", "3:ok"): 0.2, ("1:5xx", "3:ok"): 0.9}
    cands = [(0, 1), (0, 3), (1, 2), (2, 3)]
    b = fs.floor_matrix(4, cands, tok, scores, "broadcast")
    assert b[0, 1] == 0.4 and b[0, 3] == 0.2 and b[1, 2] == 0.0 and b[2, 3] == 0.9
    a = fs.floor_matrix(4, cands, tok, scores, "adjacent")
    assert a[0, 1] == 0.4 and a[0, 3] == 0.0 and a[2, 3] == 0.9  # lag 3 dropped, lag 1 kept
    assert np.count_nonzero(b) == 3 and np.count_nonzero(a) == 2


def test_rows_equal_the_reference_read_and_broadcast_the_floor(read, tmp_path):
    ref = read_json(read["ref"])
    out = {}
    for cell, row in ((TOPO, "broadcast"), (BIGRAM, "broadcast"), (BIGRAM, "adjacent")):
        d = tmp_path / f"{cell.split('/')[1]}-{row}"
        assert _run(read, d, cell, row) == 0
        s = read_json(d / SEQSCORE_JSON)
        for k in fs.IDENTITY_KEYS:
            assert fs._get(s, k) == fs._get(ref, k), k
        assert s["model_sha256"] == "none" and s["row"] == row and s["reference_cell"] == REF
        assert (
            s["floor_read_freeze"]["sha"]
            and s["tau"]
            == read_json(read["floor_read"] / RUN_DIR / RESULTS_JSON)["cells"][cell]["tau"]
        )
        out[(cell, row)] = s
    adj, bro = out[(BIGRAM, "adjacent")], out[(BIGRAM, "broadcast")]
    # adjacent-only: no truth pair beyond lag 1 is ever recovered; it never predicts more
    assert all(v == 0.0 for lag, v in adj["per_lag_recall"].items() if int(lag) > 1)
    n_adj = adj["pooled"]["directed"]
    assert n_adj["recall"] <= bro["pooled"]["directed"]["recall"]


def test_broadcast_scores_every_candidate_by_its_token_pair(read):
    """The pooled AUROC of the broadcast row equals one recomputed from the floor table directly."""
    from tracebench.score import auroc

    disc = read_json(read["read"] / RUN_DIR / RESULTS_JSON)
    c = int(disc["c"])
    corpus = Corpus(read["corpus"], "end", "request")
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    store = SequenceStore.from_corpus(corpus, "test", vocab, 24, n=0, mode="head", seed=0)
    seqs, _ = fs.stored_sequences(store, c, 0)
    seqrec = read_json(read["read"] / "sequences-request.json")
    from seq2causebench.seqtruth import truth_for_read

    ids = [store.get(s) for s in seqs]
    truth, _, _ = truth_for_read(read["corpus"], "end", "request", "test", seqrec, ids, c, 6)
    fargs = read_json(read["floor_read"] / RUN_DIR / "arguments.json")["arguments"]
    table = fs.floor_table(
        "floor/topology", read["corpus"], "end", "request", {**fargs, "split": "test"}
    )
    y, sc = [], []
    for s, i in zip(seqs, ids, strict=True):
        tok = fs.window_tokens(i, c, vocab)
        for j, q in truth[s]["candidates"]:
            y.append(int((j, q) in truth[s]["directed"]))
            sc.append(table.get((tok[j], tok[q]), 0.0))
    got = read_json(_rerun(read, TOPO, "broadcast"))
    assert got["pooled"]["auroc"] == pytest.approx(auroc(np.array(y), np.array(sc)), abs=1e-12)


def _rerun(read, cell, row):
    d = read["root"] / f"rerun-{cell.split('/')[1]}-{row}"
    if not (d / SEQSCORE_JSON).exists():
        assert _run(read, d, cell, row) == 0
    return d / SEQSCORE_JSON


def _copy_run(src, dst):
    shutil.copytree(src / RUN_DIR, dst / RUN_DIR)
    return dst


def test_refusals(read, tmp_path):
    with pytest.raises(fs.FloorSeqRefusal, match="not registered"):
        _run(read, tmp_path / "a", TOPO, "adjacent")
    with pytest.raises(fs.FloorSeqRefusal, match="max-lag"):
        _run(read, tmp_path / "b", **{"--max-lag": "5"})
    with pytest.raises(fs.FloorSeqRefusal, match="not a floor's frozen cell"):
        _run(read, tmp_path / "c", floor_cell=REF)
    with pytest.raises(fs.FloorSeqRefusal, match="not on grain"):
        _run(read, tmp_path / "d", floor_cell="floor/topology/none/frozen/session")
    # a floor validation read carries no freeze binding
    val = read["floor_read"].parent / "fval-request"
    with pytest.raises(fs.FloorSeqRefusal, match="freeze binding|holds no cell"):
        _run(read, tmp_path / "e", **{"--floor-read": str(val)})
    # a floor read whose recorded table size is not the recomputed one
    bad_floor = _copy_run(read["floor_read"], tmp_path / "bad-floor")
    r = read_json(bad_floor / RUN_DIR / RESULTS_JSON)
    r["cells"][TOPO]["n_pairs"] += 1
    write_json(bad_floor / RUN_DIR / RESULTS_JSON, r)
    with pytest.raises(fs.FloorSeqRefusal, match="recomputed floor table"):
        _run(read, tmp_path / "f", **{"--floor-read": str(bad_floor)})
    # an arm read whose recorded selection is not the re-derived one
    bad_read = _copy_run(read["read"], tmp_path / "bad-read")
    r = read_json(bad_read / RUN_DIR / RESULTS_JSON)
    r["n_stored"] += 1
    write_json(bad_read / RUN_DIR / RESULTS_JSON, r)
    with pytest.raises(fs.FloorSeqRefusal, match="re-derived n_stored"):
        _run(read, tmp_path / "g", **{"--read-dir": str(bad_read)})
    # the sequences record, where present, must be the re-derived list
    bad_seq = _copy_run(read["read"], tmp_path / "bad-seq")
    seqs = read_json(read["read"] / "sequences-request.json")
    seqs["sequences"] = seqs["sequences"][::-1]
    write_json(bad_seq / "sequences-request.json", seqs)
    with pytest.raises(fs.FloorSeqRefusal, match="sequences record"):
        _run(read, tmp_path / "h", **{"--read-dir": str(bad_seq)})
    # a reference seqscore of another read (identity counts)
    ref = read_json(read["ref"])
    ref["pooled"]["n_candidates"] += 1
    p = tmp_path / "ref.json"
    write_json(p, ref)
    with pytest.raises(fs.FloorSeqRefusal, match="not the reference cell's read"):
        _run(read, tmp_path / "i", **{"--reference-seqscore": str(p)})
    ref = read_json(read["ref"])
    ref["cell"] = "trace/cli/shipped/frozen/request"
    write_json(p, ref)
    with pytest.raises(fs.FloorSeqRefusal, match="not the reference cell's"):
        _run(read, tmp_path / "j", **{"--reference-seqscore": str(p)})


def test_five_seed_table_and_refusal(read, tmp_path):
    results = tmp_path / "results"
    seed0 = results / "xs" / "latent" / "seed=0"
    ref_dir = seed0 / REF / "seqscore"
    ref_dir.mkdir(parents=True)
    shutil.copy(read["ref"], ref_dir / SEQSCORE_JSON)
    for cell, row in ((TOPO, "broadcast"), (BIGRAM, "broadcast"), (BIGRAM, "adjacent")):
        d = seed0 / cell / fs.landing_dir_name(row, REF)
        d.mkdir(parents=True)
        shutil.copy(_rerun(read, cell, row), d / SEQSCORE_JSON)
    args = [
        "--results-dir",
        str(results),
        "--rung",
        "xs",
        "--variant",
        "latent",
        "--reference-cells",
        "trace/core/shipped/frozen",
        "--output-folder",
    ]
    with pytest.raises(ReportRefusal, match="needs 5"):
        ft.main([*args, str(tmp_path / "t1")])
    for k in range(1, 5):
        shutil.copytree(seed0, results / "xs" / "latent" / f"seed={k}")
    assert ft.main([*args, str(tmp_path / "t2")]) == 0
    doc = read_json(tmp_path / "t2" / "xs-latent-perseq-floors.json")
    assert {(t["floor_cell"], t["row"]) for t in doc["tables"]} == {
        (TOPO, "broadcast"),
        (BIGRAM, "broadcast"),
        (BIGRAM, "adjacent"),
    }
    for t in doc["tables"]:
        f1 = read_json(_rerun(read, t["floor_cell"], t["row"]))["pooled"]["directed"]["f1"]
        want = f1 - read_json(read["ref"])["pooled"]["directed"]["f1"]
        assert t["paired"]["pooled.directed.f1"]["values"] == pytest.approx([want] * 5)
        assert t["floor"]["predict_all.pooled"] == t["reference"]["predict_all.pooled"]
    assert "per-sequence axis" in (tmp_path / "t2" / "xs-latent-perseq-floors.md").read_text()
    # a floor row whose reference seqscore is of another read is refused
    bad = read_json(ref_dir / SEQSCORE_JSON)
    bad["n_scoreable"] += 1
    write_json(results / "xs" / "latent" / "seed=3" / REF / "seqscore" / SEQSCORE_JSON, bad)
    with pytest.raises(ReportRefusal, match="differ in"):
        ft.main([*args, str(tmp_path / "t3")])
