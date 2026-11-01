"""A grain may be swept into several validation tables when every swept row and coverage key
lives in exactly one of them (the context family of plans/reference-arms.md, addendum
2026-10-31, scores one sweep directory per table — two per grain; D-SB-24). A row or a coverage
key present in two tables is refused, as is any disagreement on corpus, model or grid."""

import pytest

from seq2causebench import freeze as fz


def _row(arm, tau, grain="request", c=1, N=8):
    return {
        "arm": arm,
        "path": "fixed-kl",
        "cut": "frozen",
        "grain": grain,
        "probe": "core",
        "noise": "all",
        "c": c,
        "N": N,
        "g": 1,
        "tau": tau,
        "tau_source": "grid",
        "floor": 0.05,
        "n_edges": 10,
        "directed": {"f1": 0.3, "precision": 0.3, "recall": 0.3},
    }


def _table(rows, coverage, grain="request", model="m"):
    return {
        "corpus_id": "xs/latent/seed=0",
        "model_sha256": model,
        "grains": [grain],
        "cells": rows,
        "coverage": coverage,
        "taus": [1e-7, 1e-6],
        "granger_taus": [1e-5],
        "quantiles": [0.5],
        "floor_taus": [],
    }


def test_disjoint_tables_of_one_grain_merge():
    a = _table(
        [_row("trace/core-ig-overwrite", 1e-7)],
        {"trace/core-ig-overwrite/fixed-kl/frozen/request/c1/N8": 1},
    )
    b = _table(
        [_row("trace/core-ig-keep", 1e-7)], {"trace/core-ig-keep/fixed-kl/frozen/request/c1/N8": 2}
    )
    s = _table([_row("trace/core-ig-keep", 1e-7, grain="session")], {}, grain="session")
    merged = fz.merge_tables([a, b, s])
    assert merged["grains"] == ["request", "session"]
    assert [r["arm"] for r in merged["cells"]] == [
        "trace/core-ig-overwrite",
        "trace/core-ig-keep",
        "trace/core-ig-keep",
    ]
    assert set(merged["coverage"]) == {
        "trace/core-ig-overwrite/fixed-kl/frozen/request/c1/N8",
        "trace/core-ig-keep/fixed-kl/frozen/request/c1/N8",
    }


def test_one_table_per_grain_is_unchanged():
    a = _table([_row("trace/core", 1e-7)], {"k": 1})
    s = _table([_row("trace/core", 1e-7, grain="session")], {"j": 2}, grain="session")
    merged = fz.merge_tables([a, s])
    assert merged["grains"] == ["request", "session"] and len(merged["cells"]) == 2


def test_a_row_in_two_tables_is_refused():
    a = _table([_row("trace/core-ig-keep", 1e-7)], {})
    with pytest.raises(fz.FreezeRefusal, match="D-SB-24"):
        fz.merge_tables([a, dict(a)])


def test_a_coverage_key_in_two_tables_is_refused():
    a = _table([_row("trace/core-ig-overwrite", 1e-7)], {"same": 1})
    b = _table([_row("trace/core-ig-keep", 1e-7)], {"same": 2})
    with pytest.raises(fz.FreezeRefusal, match="coverage"):
        fz.merge_tables([a, b])


def test_disagreeing_models_are_refused():
    a = _table([_row("trace/core-ig-overwrite", 1e-7)], {})
    b = _table([_row("trace/core-ig-keep", 1e-7)], {}, model="other")
    with pytest.raises(fz.FreezeRefusal, match="model_sha256"):
        fz.merge_tables([a, b])
