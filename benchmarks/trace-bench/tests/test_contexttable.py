"""`contexttable` on fabricated five-seed cells (the pipeline test's `_five_seeds` shape): the family
and reference cells, the registered pairs, the per-sequence pair, the τ drift from freezes and
test-side tables, and the mechanical verdicts with their fail clauses (addendum 2026-10-31 §8)."""

import math

import pytest

from seq2causebench import contexttable as ct
from seq2causebench.arms import cell_key
from seq2causebench.record import read_json, write_json

SEEDS = (0, 1, 2, 3, 4)


def _cell(f1_by_seed, seq_f1=None, empty=None, model="m" * 64, corpus="xs/latent/seed"):
    out = {}
    for s in SEEDS:
        f1 = f1_by_seed[s]
        out[s] = {
            "flat": {
                "directed.f1": f1,
                "directed.precision": f1,
                "directed.recall": f1,
                "directed.shd": 1.0,
                "skeleton.f1": f1,
                "skeleton.shd": 1.0,
                "orientation.accuracy": 0.5,
                "orientation.accuracy_compelled": 0.5,
                "bidirected.recall": 0.0,
                "bidirected.precision": 0.0,
                "shd_mixed": 1.0,
                "auroc.directed": 0.5,
                "reachable.f1": f1,
            },
            "annotate": {
                "corpus_id": f"{corpus}={s}",
                "seed": s,
                "model_sha256": model,
                "tool_version": "0.3.0",
                "oracle": {"in_regime": True},
                "empty_prediction": empty,
            },
            "seqscore": None
            if seq_f1 is None
            else {
                "pooled": {"directed": {"f1": seq_f1[s]}},
                "n_scoreable": 5,
                "n_sequences": 5,
                "rules_version": "r",
            },
        }
    return out


def _world(
    tmp_path, cmi_gain=0.0, keep_gain=0.0, atomic_gain=0.0, shipped_empty=False, with_tests=True
):
    base = {s: 0.30 + 0.001 * s for s in SEEDS}
    plus = lambda g: {s: v + g for s, v in base.items()}  # noqa: E731
    cells = {}
    for grain in ("request", "session"):
        cells[cell_key(*ct.IG_OVER, grain)] = _cell(base, seq_f1=base)
        cells[cell_key(*ct.CMI_OVER, grain)] = _cell(plus(cmi_gain), seq_f1=base)
        cells[cell_key(*ct.IG_KEEP, grain)] = _cell(plus(keep_gain), seq_f1=plus(keep_gain))
        cells[cell_key(*ct.CMI_KEEP, grain)] = _cell(
            plus(cmi_gain + keep_gain), seq_f1=plus(keep_gain)
        )
        cells[cell_key(*ct.ATOMIC, grain)] = _cell(plus(atomic_gain))
        cells[cell_key(*ct.ATOMIC_REAL, grain)] = _cell(plus(atomic_gain))
        cells[cell_key(*ct.ATOMIC_SHIPPED, grain)] = _cell(
            plus(atomic_gain), empty={"kind": "genuine"} if shipped_empty else None
        )
        cells[cell_key(*ct.ATOMIC_REAL_SHIPPED, grain)] = _cell(plus(atomic_gain))
        cells[cell_key(*ct.LANDED_CORE, grain)] = _cell(base)
        cells[cell_key(*ct.LANDED_ATOMIC, grain)] = _cell(base)
        cells[cell_key(*ct.LANDED_ATOMIC_SHIPPED, grain)] = _cell(base)
    ctx_freezes, landed_freezes, test_tables = {}, {}, {}
    for s in SEEDS:
        fcells = {}
        for grain in ("request", "session"):
            for t in ct.FAMILY_CELLS:
                tau = None if t[2] == "shipped" else (1e-3 if "cmi" in t[0] else 1e-2)
                fcells[cell_key(*t, grain)] = {
                    "c": 1,
                    "N": 8,
                    "g": 1,
                    "tau": tau,
                    "val_directed_f1": 0.28,
                }
        ctx_freezes[s] = {
            "freeze": {
                "rung": "xs",
                "variant": "latent",
                "family": "context",
                "seed": s,
                "cells": fcells,
            },
            "path": f"ctx-{s}",
            "sha256": "a",
        }
        lcells = {}
        for grain in ("request", "session"):
            for t in ct.REFERENCE_CELLS:
                tau = None if t[2] == "shipped" else 1e-2
                lcells[cell_key(*t, grain)] = {
                    "c": 1,
                    "N": 8,
                    "g": 1,
                    "tau": tau,
                    "val_directed_f1": 0.29,
                }
        landed_freezes[s] = {
            "freeze": {
                "rung": "xs",
                "variant": "latent",
                "family": "model",
                "seed": s,
                "cells": lcells,
            },
            "path": f"landed-{s}",
            "sha256": "b",
        }
        if with_tests:
            for grain in ("request", "session"):
                for t in ct.FAMILY_CELLS + ct.REFERENCE_CELLS:
                    if t[2] == "shipped":
                        continue
                    # the CMI arms' test argmax sits one half-decade above the frozen τ, the others on it
                    best = 3e-3 if "cmi" in t[0] else 1e-2
                    test_tables[(s, t[0], t[1], grain, 1, 8)] = [
                        (1e-3, 0.2),
                        (3e-3, 0.25 if best == 3e-3 else 0.2),
                        (1e-2, 0.3 if best == 1e-2 else 0.2),
                    ]
    return cells, ctx_freezes, landed_freezes, test_tables


def test_pairs_drift_and_verdicts_pass_on_a_neutral_world(tmp_path):
    cells, cf, lf, tt = _world(tmp_path)
    doc = ct.build(cells, cf, lf, tt, "xs", "latent")
    assert set(doc["grains"]) == {"request", "session"} and doc["n_test_tables"] == 5
    g = doc["grains"]["session"]
    assert len(g["cells"]) == 11 and all(t["n_seeds"] == 5 for t in g["cells"].values())
    assert len(g["pairs"]) == sum(len(v) for v in ct.PAIRS.values())
    ident = g["pairs"][f"identity-i:{ct._triple(ct.IG_OVER)}:{ct._triple(ct.LANDED_CORE)}"]
    assert ident["tests"]["directed.f1"]["mean_diff"] == pytest.approx(0.0)
    # τ drift: the CMI arms' argmax sits half a decade above the frozen τ, the IG arms' on it
    d = g["tau_drift"]
    assert d[cell_key(*ct.CMI_OVER, "session")]["mean_abs_tau_drift_log10"] == pytest.approx(
        math.log10(3)
    )
    assert d[cell_key(*ct.IG_OVER, "session")]["mean_abs_tau_drift_log10"] == pytest.approx(0.0)
    assert d[cell_key(*ct.IG_OVER, "session")]["mean_f1_drift"] == pytest.approx(0.302 - 0.28)
    assert cell_key(*ct.ATOMIC_SHIPPED, "session") not in d  # a shipped cut has no τ
    v = g["verdicts"]
    assert v["H-ctx-cmi-f1"]["status"] == "pass" and v["H-ctx-keep-perseq"]["status"] == "pass"
    assert (
        v["H-atomic-cmi-f1"]["status"] == "pass" and v["H-atomic-cmi-shipped"]["status"] == "pass"
    )
    assert v["H-ctx-cmi-tau"]["overwrite"]["status"] == "pass"  # log10(3) < one half-decade step
    assert v["H-ctx-cmi-tau"]["overwrite"]["basis"] == "tau"
    assert v["H-atomic-cmi-tau"]["status"] == "pass"
    assert doc["grains"]["request"]["verdicts"]["H-ctx-keep-perseq"]["status"].startswith(
        "not applicable"
    )
    md = ct.render_md(doc)
    assert (
        "## session" in md
        and "H-ctx-cmi-f1" in md
        and "`trace/core-cmi-keep/fixed-kl/frozen/session`" in md
    )


def test_fail_clauses_and_not_computed(tmp_path):
    # CMI below IG by 0.05 on every seed; keep below overwrite per sequence; atomic below the landed arm;
    # an empty shipped cut on one seed
    cells, cf, lf, tt = _world(
        tmp_path, cmi_gain=-0.05, keep_gain=-0.05, atomic_gain=-0.05, shipped_empty=True
    )
    v = ct.build(cells, cf, lf, tt, "xs", "latent")["grains"]["session"]["verdicts"]
    assert v["H-ctx-cmi-f1"]["status"] == "fail" and all(
        r["fail"] for r in v["H-ctx-cmi-f1"]["rows"]
    )
    assert v["H-ctx-keep-perseq"]["status"] == "fail"
    assert v["H-atomic-cmi-f1"]["status"] == "fail"
    assert v["H-atomic-cmi-shipped"]["status"] == "fail"
    # a gain never fails the one-sided clauses, and a difference inside the band never fails
    cells, cf, lf, tt = _world(tmp_path, cmi_gain=0.005, keep_gain=0.2, atomic_gain=0.2)
    v = ct.build(cells, cf, lf, tt, "xs", "latent")["grains"]["session"]["verdicts"]
    assert all(
        v[h]["status"] == "pass" for h in ("H-ctx-cmi-f1", "H-ctx-keep-perseq", "H-atomic-cmi-f1")
    )
    # without test-side tables the τ hypotheses fall back to the F1 drift (l / xl)
    cells, cf, lf, tt = _world(tmp_path, with_tests=False)
    doc = ct.build(cells, cf, lf, tt, "xs", "latent")
    v = doc["grains"]["request"]["verdicts"]
    assert v["H-ctx-cmi-tau"]["keep"]["basis"] == "f1" and v["H-atomic-cmi-tau"]["basis"] == "f1"
    assert doc["n_test_tables"] == 0
    # fewer than five seeds is refused, as every headline is
    cells, cf, lf, tt = _world(tmp_path)
    for key in cells:
        cells[key].pop(4)
    with pytest.raises(ct.ReportRefusal, match="needs at least 5"):
        ct.build(cells, cf, lf, tt, "xs", "latent")


def test_loaders_refuse_the_wrong_family_rung_or_duplicate_seed(tmp_path):
    good = {"rung": "xs", "variant": "latent", "family": "context", "seed": 0, "cells": {}}
    write_json(tmp_path / "a.json", good)
    write_json(tmp_path / "b.json", {**good, "seed": 1})
    fz = ct.load_freezes([tmp_path / "a.json", tmp_path / "b.json"], "context", "xs", "latent")
    assert set(fz) == {0, 1} and fz[0]["sha256"]
    with pytest.raises(ct.ReportRefusal, match="family"):
        ct.load_freezes([tmp_path / "a.json"], "model", "xs", "latent")
    with pytest.raises(ct.ReportRefusal, match="not s/latent"):
        ct.load_freezes([tmp_path / "a.json"], "context", "s", "latent")
    write_json(tmp_path / "c.json", good)
    with pytest.raises(ct.ReportRefusal, match="two context freezes"):
        ct.load_freezes([tmp_path / "a.json", tmp_path / "c.json"], "context", "xs", "latent")
    write_json(
        tmp_path / "t.json",
        {
            "seed": 2,
            "cells": [
                {
                    "cut": "frozen",
                    "arm": "trace/core-ig-keep",
                    "path": "fixed-kl",
                    "grain": "request",
                    "c": 1,
                    "N": 8,
                    "tau": 1e-2,
                    "directed": {"f1": 0.1},
                },
                {
                    "cut": "shipped",
                    "arm": "x",
                    "path": "y",
                    "grain": "request",
                    "c": 1,
                    "N": 8,
                    "tau": None,
                    "directed": {"f1": 0.1},
                },
            ],
        },
    )
    tt = ct.load_test_tables([tmp_path / "t.json"])
    assert tt == {(2, "trace/core-ig-keep", "fixed-kl", "request", 1, 8): [(1e-2, 0.1)]}
    assert (
        ct.test_argmax_tau([(1e-3, 0.2), (1e-2, 0.2), (1e-1, 0.1)]) == 1e-2
    )  # ties to the larger τ
    assert ct.test_argmax_tau([]) is None
    write_json(tmp_path / "noseed.json", {"cells": []})
    with pytest.raises(ct.ReportRefusal, match="seed"):
        ct.load_test_tables([tmp_path / "noseed.json"])
    assert read_json(tmp_path / "a.json")["seed"] == 0
