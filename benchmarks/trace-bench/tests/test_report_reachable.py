"""The reachable-only columns of `report` (plans/per-sequence-rules.md, addendum 2026-10-04):
derived from a cell's landed score and coverage, the benchmark's true and false positives kept,
the truth cut to the edges the read scored."""

import pytest

from seq2causebench import report as rp


def test_reachable_only_columns():
    directed = {"tp": 30, "fp": 70, "fn": 970, "precision": 0.3, "recall": 0.03}
    coverage = {"reachable_recall_ceiling": 0.05, "truth_directed": 1000, "pairs_in_universe": 400}
    out = rp.reachable_only(directed, coverage)
    assert out["reachable.precision"] == pytest.approx(0.3)  # unchanged by construction
    assert out["reachable.recall"] == pytest.approx(30 / 50)  # benchmark recall over the ceiling
    assert out["reachable.recall"] == pytest.approx(directed["recall"] / 0.05)
    assert out["reachable.f1"] == pytest.approx(2 * 0.3 * 0.6 / 0.9)
    base = 50 / 400  # predict every scored pair of the universe
    assert out["reachable.predict_all_f1"] == pytest.approx(2 * base / (1 + base))


def test_reachable_only_edges():
    coverage = {"reachable_recall_ceiling": 0.5, "truth_directed": 10, "pairs_in_universe": 20}
    empty = rp.reachable_only({"tp": 0, "fp": 0}, coverage)
    assert empty["reachable.precision"] == 0.0 and empty["reachable.f1"] == 0.0
    full = rp.reachable_only({"tp": 5, "fp": 0}, coverage)
    assert full["reachable.recall"] == 1.0 and full["reachable.f1"] == 1.0
    none = rp.reachable_only(
        {"tp": 0, "fp": 3},
        {"reachable_recall_ceiling": 0.0, "truth_directed": 10, "pairs_in_universe": 20},
    )
    assert all(v is None for v in none.values())


def test_reachable_columns_are_reported():
    for m in (
        "reachable.precision",
        "reachable.recall",
        "reachable.f1",
        "reachable.predict_all_f1",
    ):
        assert m in rp.METRICS and m in rp.SHOW
