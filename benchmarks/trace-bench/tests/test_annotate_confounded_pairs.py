"""The confounded-pair diagnostic (scenario 38) counts directed predictions on bidirected truth
pairs in either orientation, at the floor, and stays linear in the prediction: the truth pair set
is built once per call. At the `l` rung the set holds 50k–180k pairs and a frozen prediction over
100k edges; rebuilding it per predicted edge cost hours per cell (RUN.md, 2026-10-03)."""

import json
import time

from seq2causebench.annotate import GRAPHS_DIR, SCORING_TARGET_JSON, confounded_pair_count


def _corpus(tmp_path, bidirected):
    gdir = tmp_path / GRAPHS_DIR
    gdir.mkdir(parents=True)
    target = {
        "directed": [],
        "bidirected": [{"a": a, "b": b, "strength": s} for a, b, s in bidirected],
    }
    (gdir / SCORING_TARGET_JSON).write_text(json.dumps(target), encoding="utf-8")
    return tmp_path


def test_counts_either_orientation_at_the_floor(tmp_path):
    corpus = _corpus(
        tmp_path, [("1:ok", "2:err", 0.9), ("3:ok", "4:err", 0.9), ("5:ok", "6:ok", 0.01)]
    )
    prediction = {
        "directed": [
            {"src": "1:ok", "dst": "2:err"},  # truth order
            {"src": "4:err", "dst": "3:ok"},  # reversed
            {"src": "5:ok", "dst": "6:ok"},  # below the floor
            {"src": "1:ok", "dst": "3:ok"},  # not a truth pair
        ]
    }
    out = confounded_pair_count(corpus, "request", prediction, 0.05)
    assert out == {"directed_predictions_on_bidirected_truth_pairs": 2, "truth_bidirected_pairs": 2}


def test_linear_in_the_prediction(tmp_path):
    n = 20000
    corpus = _corpus(tmp_path, [(f"{i}:ok", f"{i}:err", 0.9) for i in range(n)])
    prediction = {"directed": [{"src": f"{i}:err", "dst": f"{i}:ok"} for i in range(n)]}
    start = time.perf_counter()
    out = confounded_pair_count(corpus, "request", prediction, 0.05)
    assert out["directed_predictions_on_bidirected_truth_pairs"] == n
    assert time.perf_counter() - start < 5.0  # the per-edge rebuild takes minutes at this size
