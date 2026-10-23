"""The sensitivity table (`plans/reference-arms.md`, addendum 2026-10-22) on synthetic tables: the new
(c, N) cells at their own best tau beside the landed frozen value and the landed curve, the probed
count at c = 16, the expectation band, and the refusals (a landed table the freeze did not hash,
a sensitivity table on another model)."""

import pytest

from seq2causebench import sensitivitytable as st
from seq2causebench.record import read_json, sha256_file, write_json


def _row(arm, path, grain, c, n, tau, f1):
    return {
        "arm": arm,
        "path": path,
        "cut": "frozen",
        "grain": grain,
        "c": c,
        "N": n,
        "tau": tau,
        "directed": {"f1": f1},
    }


def _setup(tmp_path, model="m" * 64):
    landed = {
        "instance": "xs",
        "grains": ["session"],
        "model_sha256": model,
        "corpus_id": "xs/latent/seed=0",
        "ordering": "end",
        "coverage": {},
        "cells": [
            _row("trace/core", "fixed", "session", c, n, tau, f1)
            for c, n, tau, f1 in [
                (2, 2, 0.1, 0.30),
                (2, 32, 0.1, 0.34),
                (2, 32, 0.3, 0.34),
                (8, 8, 0.1, 0.31),
            ]
        ],
    }
    write_json(tmp_path / "landed.json", landed)
    freeze = {
        "seed": 0,
        "rung": "xs",
        "ordering": "end",
        "model_sha256": model,
        "corpus_id": "xs/latent/seed=0",
        "val_tables": [{"name": "val-table.json", "sha256": sha256_file(tmp_path / "landed.json")}],
        "cells": {
            "trace/core/fixed/frozen/session": {
                "c": 2,
                "N": 32,
                "tau": 0.3,
                "val_directed_f1": 0.34,
                "probe": "core",
                "noise": "real",
            },
            "trace/core/fixed/shipped/session": {"c": 2, "N": 32, "tau": None},
        },
    }
    write_json(tmp_path / "freeze.json", freeze)
    sens = {
        "instance": "xs",
        "grains": ["session"],
        "model_sha256": model,
        "corpus_id": "xs/latent/seed=0",
        "ordering": "end",
        "coverage": {
            "trace/core/fixed/frozen/session/c16/N2": {
                "coverage": {"reachable_recall_ceiling": 0.42}
            }
        },
        "cells": [
            _row("trace/core", "fixed", "session", 2, 128, 0.1, 0.345),
            _row("trace/core", "fixed", "session", 2, 128, 1.0, 0.345),
            _row("trace/core", "fixed", "session", 16, 2, 0.1, 0.36),
        ],
    }
    write_json(tmp_path / "sens.json", sens)
    write_json(
        tmp_path / "sweep/run/results.json",
        {
            "rung": "xs",
            "cells": [
                {
                    "grain": "session",
                    "probe": "core",
                    "noise": "real",
                    "c": 16,
                    "N": 2,
                    "n_probed": 125,
                    "n_sequences": 437,
                }
            ],
        },
    )
    return tmp_path


def _main(d, out):
    return st.main(
        [
            "--freezes",
            str(d / "freeze.json"),
            "--landed-val-tables",
            str(d / "landed.json"),
            "--sensitivity-val-tables",
            str(d / "sens.json"),
            "--sensitivity-sweep-records",
            str(d / "sweep"),
            "--output-folder",
            str(out),
        ]
    )


def test_rows_curve_probed_count_and_band(tmp_path):
    d = _setup(tmp_path)
    assert _main(d, d / "tables" / "sensitivity-latent") == 0
    doc = read_json(d / "tables" / "sensitivity-latent" / "sensitivity-latent.json")
    (row,) = doc["rows"]
    assert row["landed"] == {"c": 2, "N": 32, "tau": 0.3, "val_f1": 0.34}
    assert [(p["c"], p["N"], p["val_f1"], p["tau"]) for p in row["landed_curve"]] == [
        (2, 2, 0.30, 0.1),
        (2, 32, 0.34, 0.3),
        (8, 8, 0.31, 0.1),
    ]
    n128, c16 = row["new"]
    assert (n128["c"], n128["N"], n128["tau"]) == (2, 128, 1.0) and n128[
        "diff_from_landed"
    ] == pytest.approx(0.005)
    assert (c16["n_probed"], c16["n_sequences"], c16["reachable_recall_ceiling"]) == (
        125,
        437,
        0.42,
    )
    assert (
        row["expectation"]["c16_gain_over_best_landed"] == pytest.approx(0.02)
        and not row["expectation"]["within_band"]
    )
    md = (d / "tables" / "sensitivity-latent" / "sensitivity-latent.md").read_text()
    assert "[125/437, 0.420]" in md and "**no**" in md and doc["n_outside_band"] == 1


def test_refusals(tmp_path):
    d = _setup(tmp_path)
    write_json(d / "landed.json", {**read_json(d / "landed.json"), "cells": []})
    with pytest.raises(ValueError, match="hashed"):
        _main(d, d / "t1")
    d2 = _setup(tmp_path / "b")
    write_json(d2 / "sens.json", {**read_json(d2 / "sens.json"), "model_sha256": "x" * 64})
    with pytest.raises(ValueError, match="model"):
        _main(d2, d2 / "t2")
