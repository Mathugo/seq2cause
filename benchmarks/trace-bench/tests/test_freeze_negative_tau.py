"""A swept row at a negative τ is not a freeze candidate (plans/reference-arms.md, addendum
2026-10-05): the quantile family of a signed score (Shapley) puts its lowest member below zero at
xl, and that row is the cut "every scored pair", not a threshold on the arm's evidence."""

import pytest

from seq2causebench import freeze as fz


def _row(arm, tau, f1, source="grid", c=1, N=0):
    return {
        "arm": arm,
        "path": "none",
        "cut": "frozen",
        "grain": "request",
        "probe": "shapley",
        "noise": "all",
        "c": c,
        "N": N,
        "g": 1,
        "tau": tau,
        "tau_source": source,
        "floor": 0.05,
        "n_edges": 10,
        "directed": {"f1": f1, "precision": f1, "recall": f1},
    }


def _table(rows):
    return {
        "cells": rows,
        "coverage": {},
        "taus": [0.0, 1e-7],
        "granger_taus": [0.0, 1e-5],
        "quantiles": [0.0, 0.1, 0.5],
    }


def test_negative_tau_row_is_set_aside_and_counted():
    rows = [
        _row("baseline/shapley", -0.08, 0.0237, "quantile p0"),  # would win on F1
        _row("baseline/shapley", 3e-5, 0.0227, "quantile p10"),
        _row("baseline/shapley", 2.5e-3, 0.0171, "quantile p50"),
    ]
    cell = fz.select_cells(_table(rows))["baseline/shapley/none/frozen/request"]
    assert cell["tau"] == 3e-5 and cell["tau_source"] == "quantile p10"
    assert cell["n_rows_negative_tau"] == 1 and cell["n_rows_considered"] == 2
    assert cell["grid"] == "quantiles" and cell["at_grid_edge"] is False  # p10 is not an end


def test_zero_tau_stays_a_candidate():
    rows = [
        _row("baseline/shapley", 0.0, 0.02, "quantile p0"),
        _row("baseline/shapley", 0.5, 0.01, "quantile p50"),
    ]
    cell = fz.select_cells(_table(rows))["baseline/shapley/none/frozen/request"]
    assert cell["tau"] == 0.0 and cell["n_rows_negative_tau"] == 0 and cell["at_grid_edge"] is True


def test_all_rows_negative_is_refused():
    with pytest.raises(fz.FreezeRefusal):
        fz.select_cells(_table([_row("baseline/shapley", -1.0, 0.1, "quantile p0")]))
