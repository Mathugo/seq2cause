# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""D-SB-10: the union at a threshold, the corpus-level max / mean / count / per-lag
accumulator over position-pair occurrences with named columns, strict selection at τ,
and a prediction the benchmark's own scorer accepts and scores."""

import numpy as np
import pytest
from fixture_corpus import REQUEST_DIRECTED, fixture_corpus
from tracebench.score import score_corpus

from seq2causebench.constants import EOS
from seq2causebench.corpus import Corpus
from seq2causebench.prediction import write_prediction
from seq2causebench.project import (
    PairAccumulator,
    columns_of,
    read_scores_npz,
    sequence_cells,
    sequence_type_edges,
    write_scores_npz,
)
from seq2causebench.record import read_json
from seq2causebench.select import per_lag_ranking, ranking, select_edges
from seq2causebench.vocab import Vocab

COLS = ("trace-core__shipped", "trace-core__fixed-kl", "baseline-granger__shipped")


def _vocab():
    return Vocab.from_model_vocab(Corpus(fixture_corpus("latent"), "end", "request").vocab_json())


def _stats(S):
    return {k: S for k in COLS}


def test_projection_union_max_mean_count():
    v = _vocab()
    # window tokens: base ids of ops 4, 2, 4, 0 then EOS; ops differ except positions 0 and 2
    w = np.array([v.base_id(4), v.base_id(2), v.base_id(4), v.base_id(0), EOS])
    S = np.full((5, 5), np.nan)
    S[0, 1] = 0.5
    S[0, 2] = 9.0
    S[0, 3] = 0.2
    S[1, 2] = 0.3
    S[1, 3] = 0.7
    S[2, 3] = 0.1
    S[0, 4] = 5.0
    S[3, 4] = 5.0
    j, q, lag, u, v_, vals = sequence_cells(_stats(S), w, v, max_lag=3)
    pairs = set(zip(u.tolist(), v_.tolist(), strict=True))
    assert (v.base_id(4), v.base_id(4)) not in pairs  # within-op pair dropped
    assert all(EOS not in p for p in pairs)  # EOS never a type
    assert (v.base_id(4), v.base_id(2)) in pairs and (v.base_id(2), v.base_id(0)) in pairs
    assert sequence_type_edges(S, w, v, tau=0.4, max_lag=3) == {
        (v.base_id(4), v.base_id(2)),
        (v.base_id(2), v.base_id(0)),
    }
    assert sequence_type_edges(S, w, v, tau=0.5, max_lag=3) == {
        (v.base_id(2), v.base_id(0))
    }  # strict
    acc = PairAccumulator(v.size, max_lag=3, columns=COLS, flush_every=1)
    acc.add(u, v_, lag, vals)
    acc.add(u, v_, lag, {k: vals[k] * 2 for k in COLS})  # a second sequence with doubled scores
    r = acc.result()
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(r["src"], r["dst"], strict=True))}
    i = key[
        (v.base_id(4), v.base_id(0))
    ]  # occurrences: (0,3) lag 3 = 0.2 and (2,3) lag 1 = 0.1, twice
    assert r["count"][i] == 4 and r["max_trace-core__shipped"][i] == 0.4
    assert abs(r["mean_trace-core__shipped"][i] - (0.2 + 0.1 + 0.4 + 0.2) / 4) < 1e-12
    lm = r["lag_max_trace-core__shipped"][i]
    assert (
        lm[0] == 0.2 and np.isneginf(lm[1]) and lm[2] == 0.4
    )  # per-lag max: lag 1 -> 0.2, lag 2 absent, lag 3 -> 0.4
    assert acc.n_sequences == 2 and acc.n_cells == 2 * len(u)
    assert columns_of(r) == tuple(sorted(COLS))
    with pytest.raises(ValueError):
        PairAccumulator(v.size, 3, columns=())


def test_accumulator_flush_merge_equals_single_pass():
    rng = np.random.default_rng(0)
    V, M = 30, 4
    cells = [
        (rng.integers(4, V, 40), rng.integers(4, V, 40), rng.integers(1, M + 1, 40), rng.random(40))
        for _ in range(9)
    ]
    outs = []
    for flush in (1, 4, 100):
        acc = PairAccumulator(V, M, columns=COLS, flush_every=flush)
        for u, v, lag, s in cells:
            keep = u != v
            acc.add(u[keep], v[keep], lag[keep], {k: s[keep] * (i + 1) for i, k in enumerate(COLS)})
        outs.append(acc.result())
    for o in outs[1:]:
        for k in outs[0]:
            if outs[0][k].dtype.kind == "f":
                assert np.allclose(o[k], outs[0][k], equal_nan=True, rtol=1e-12, atol=0), (
                    k
                )  # summation order only
            else:
                assert np.array_equal(o[k], outs[0][k]), k


def test_select_strict_and_rankings(tmp_path):
    scores = {"src": np.array([5, 6, 7]), "dst": np.array([8, 9, 10]), "count": np.array([1, 2, 3])}
    for k in COLS:
        scores[f"max_{k}"] = np.array([0.5, 1.0, 2.0])
        scores[f"mean_{k}"] = np.array([0.25, 0.5, 1.0])
        scores[f"lag_max_{k}"] = np.array([[0.5, -np.inf], [-np.inf, 1.0], [2.0, 2.0]])
    src, dst, s = select_edges(scores, COLS[0], "max", tau=1.0)
    assert src.tolist() == [7] and s.tolist() == [2.0]  # 1.0 is not > 1.0
    assert len(ranking(scores, COLS[1], "max")[0]) == 3
    src, dst, s = per_lag_ranking(scores, COLS[0], lag=2)
    assert src.tolist() == [6, 7] and s.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        select_edges(
            scores, COLS[0], "mean", 0.1
        )  # mean is a recorded column, not an aggregation sub-arm
    with pytest.raises(ValueError):
        select_edges(scores, "nope__shipped", "max", 0.1)
    write_scores_npz(tmp_path / "s.npz", scores, n_sequences=3, max_lag=2)
    back = read_scores_npz(tmp_path / "s.npz")
    assert back["n_sequences"] == 3 and np.array_equal(
        back[f"max_{COLS[0]}"], scores[f"max_{COLS[0]}"]
    )


def test_prediction_scored_by_tracebench(tmp_path):
    """A prediction built from a scores table carrying the truth edges scores perfectly through the benchmark."""
    corpus = fixture_corpus("latent")
    v = _vocab()
    tok = {v.token_string(t): t for t in v.real_ids}
    truth = [(s, d, w) for s, d, w in REQUEST_DIRECTED if w >= 0.05]
    # the fixture's truth names two tokens the correlator never minted (0:4xx, 5:err): unreachable by construction (D-SB-6)
    reachable = [(s, d, w) for s, d, w in truth if s in tok and d in tok]
    assert len(truth) == 6 and len(reachable) == 2
    src = np.array([tok[s] for s, _, _ in reachable])
    dst = np.array([tok[d] for _, d, _ in reachable])
    score = np.array([w for _, _, w in reachable])
    src = np.append(src, tok["0:ok"])
    dst = np.append(dst, tok["2:ok"])
    score = np.append(score, 0.001)
    hit = score > 0.01
    n = write_prediction(
        tmp_path / "prediction-request-trace-core-shipped-frozen.json",
        src[hit],
        dst[hit],
        score[hit],
        v,
        arm="trace/core",
        path="shipped",
        cut="frozen",
    )
    assert n == len(reachable)
    doc = read_json(tmp_path / "prediction-request-trace-core-shipped-frozen.json")
    assert (
        doc["bidirected"] == []
        and doc["structural_limitation"]["assumption"] == "causal_sufficiency"
    )
    assert (
        doc["structural_limitation"]["truth_bidirected_edges"] is None
        and doc["meta"]["arm"] == "trace/core"
    )
    res = score_corpus(corpus, doc, grain="request")
    assert res["directed"]["precision"] == 1.0 and abs(res["directed"]["recall"] - 2 / 6) < 1e-12
    assert res["directed"]["shd"] == 4 and res["universe"]["predictions_outside_universe"] == 0
    write_prediction(tmp_path / "ranking-request-trace-core-shipped.json", src, dst, score, v)
    rk = score_corpus(
        corpus, read_json(tmp_path / "ranking-request-trace-core-shipped.json"), grain="request"
    )
    assert rk["auroc"]["directed"] > 0.6 and rk["average_precision"]["directed"] > 0.3
    assert rk["directed"]["precision"] < 1.0  # every listed edge counts as present: hence two files
