"""Tests for `seq2cause.kl` and the `kl_mode` plumbing: the v0.1.9 clamp
algebra is pinned bit for bit against a stored fixture, the log-space
algebra is finite where the clamp corrupts, the two agree wherever the clamp
is finite, and the corrupted-cell counts are reported in either mode."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from seq2cause.causal_strength import calc_lag_info_gain
from seq2cause.core import SampleLevelCausalDiscovery
from seq2cause.diagnostics import compute_cmi_matrix, compute_cmi_matrix_sparse
from seq2cause.kl import (
    KL_MODES,
    bernoulli_kl_clamp,
    bernoulli_kl_from_log,
    corrupted_cell_counts,
    log1mexp,
    logmeanexp,
)
from seq2cause.scm import create_scm

FIXTURE = Path(__file__).parent / "fixtures" / "lag_info_gain_v019.json"


def _tensor(nested):
    """JSON -> float32 tensor; the strings 'nan'/'inf'/'-inf' become non-finite floats."""

    def conv(x):
        if isinstance(x, list):
            return [conv(y) for y in x]
        if x == "nan":
            return math.nan
        if x == "inf":
            return math.inf
        if x == "-inf":
            return -math.inf
        return float(x)

    return torch.tensor(conv(nested), dtype=torch.float32)


def _equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        torch.nan_to_num(a, nan=-1.0), torch.nan_to_num(b, nan=-1.0)
    ) and torch.equal(torch.isnan(a), torch.isnan(b))


# ---------------------------------------------------------------- primitives


def test_log1mexp_matches_naive_formula_away_from_the_ends():
    x = torch.tensor([-0.01, -0.5, -2.0, -20.0], dtype=torch.float64)
    naive = torch.log(1 - torch.exp(x))
    assert torch.allclose(log1mexp(x), naive, atol=1e-12)


def test_log1mexp_is_finite_at_zero_and_very_negative():
    x = torch.tensor([0.0, -1e-12, -1e-300, -800.0], dtype=torch.float64)
    assert torch.isfinite(log1mexp(x)).all()


def test_logmeanexp_is_log_of_mean_of_exp():
    torch.manual_seed(0)
    x = torch.randn(5, 3, dtype=torch.float64) - 3
    assert torch.allclose(logmeanexp(x, dim=0), torch.log(torch.exp(x).mean(dim=0)))


def test_bernoulli_kl_from_log_matches_clamp_where_finite():
    torch.manual_seed(0)
    q = torch.rand(1000) * 0.98 + 0.01
    p = torch.rand(1000) * 0.98 + 0.01
    clamp = bernoulli_kl_clamp(q, p)
    logspace = bernoulli_kl_from_log(torch.log(q), torch.log(p))
    assert torch.isfinite(clamp).all()
    assert torch.allclose(logspace.float(), clamp, rtol=1e-5, atol=1e-6)


def test_bernoulli_kl_from_log_is_finite_where_clamp_is_not():
    one = torch.tensor([1.0], dtype=torch.float32)  # a float32 probability saturated to 1.0
    half = torch.tensor([0.5], dtype=torch.float32)
    assert torch.isnan(bernoulli_kl_clamp(one, half))  # observed branch saturated -> NaN
    assert torch.isinf(bernoulli_kl_clamp(half, one))  # baseline branch saturated -> +inf
    # log_softmax keeps a finite log-probability where softmax rounds to 1.0
    logits = torch.tensor([60.0, 0.0, 0.0, 0.0])
    assert torch.softmax(logits, -1)[0].item() == 1.0
    log_one = torch.log_softmax(logits, -1)[0:1]
    assert torch.isfinite(bernoulli_kl_from_log(log_one, torch.log(half))).all()
    assert torch.isfinite(bernoulli_kl_from_log(torch.log(half), log_one)).all()


def test_corrupted_cell_counts_definition():
    q = torch.tensor([[1.0, 0.5, 0.5], [0.5, 0.5, 0.2]])
    p = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.5, 0.2]])
    counts = corrupted_cell_counts(q, p)
    assert counts == {"n_cell_collapsed": 1, "n_cell_saturated": 2, "n_cells": 6}
    band = torch.tensor([[True, True, False], [False, True, True]])
    counts = corrupted_cell_counts(q, p, band=band)
    assert counts == {"n_cell_collapsed": 1, "n_cell_saturated": 1, "n_cells": 4}


def test_corrupted_cell_counts_band_broadcasts_when_particles_and_rows_differ():
    """bs=1, N=2 particles, 3 row pairs, Lc=4: the band [3, 4] must count cells, not particles."""
    torch.manual_seed(0)
    q = torch.rand(1, 2, 3, 4)
    p = torch.rand(1, 2, 3, 4)
    q[0, 1, 2, 3] = 1.0  # one collapsed particle in cell (row 2, effect 3)
    band = torch.ones(3, 4, dtype=torch.bool)
    band[2, 3] = False  # ... which sits outside the band
    counts = corrupted_cell_counts(q, p, band=band, particle_dim=1)
    assert (
        counts["n_cells"] == 11
        and counts["n_cell_collapsed"] == 0
        and counts["n_particle_collapsed"] == 0
    )
    band[2, 3] = True
    counts = corrupted_cell_counts(q, p, band=band, particle_dim=1)
    assert (
        counts["n_cells"] == 12
        and counts["n_cell_collapsed"] == 1
        and counts["n_particle_collapsed"] == 1
    )


def test_corrupted_cell_counts_over_particles_any_particle_spoils_the_cell():
    # particles on dim 0: cell 0 has one collapsed particle, cell 1 one saturated, cell 2 clean
    q = torch.tensor([[1.0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    p = torch.tensor([[0.5, 0.5, 0.5], [0.5, 1.0, 0.5]])
    counts = corrupted_cell_counts(q, p, particle_dim=0)
    assert counts["n_cell_collapsed"] == 1 and counts["n_cell_saturated"] == 1
    assert counts["n_particle_collapsed"] == 1 and counts["n_particle_saturated"] == 1
    assert counts["n_cells"] == 3


# ---------------------------------------------------------------- calc_lag_info_gain


def _fixture():
    return json.loads(FIXTURE.read_text())


def test_calc_lag_info_gain_clamp_mode_reproduces_v019_bit_for_bit():
    fx = _fixture()["lag_info_gain"]
    params = {"sampling": {"context": fx["context"]}}
    ids = torch.tensor(fx["input_ids"])
    for key_logits, key_expected in (
        ("logits", "expected"),
        ("logits_saturated_baseline", "expected_saturated_baseline"),
        ("logits_saturated_observed", "expected_saturated_observed"),
    ):
        probs = torch.softmax(_tensor(fx[key_logits]), dim=-1)
        out = calc_lag_info_gain(probs, {"input_ids": ids}, params, kl_mode="clamp")
        assert _equal_with_nan(out, _tensor(fx[key_expected])), key_logits


def test_calc_lag_info_gain_v019_fixture_shows_both_failure_directions():
    fx = _fixture()["lag_info_gain"]
    sat = _tensor(fx["expected_saturated_baseline"])
    col = _tensor(fx["expected_saturated_observed"])
    clean = _tensor(fx["expected"])
    assert sat[0, 1, 2].item() == torch.finfo(torch.float32).max  # +inf coerced to float32 max
    assert col[0, 1, 2].item() == 0.0  # NaN coerced to zero
    assert 0 < clean[0, 1, 2].item() < 1


def test_calc_lag_info_gain_logspace_is_finite_and_counts_where_clamp_corrupts():
    fx = _fixture()["lag_info_gain"]
    params = {"sampling": {"context": fx["context"]}}
    ids = torch.tensor(fx["input_ids"])
    for key, direction in (
        ("logits_saturated_baseline", "n_cell_saturated"),
        ("logits_saturated_observed", "n_cell_collapsed"),
    ):
        logits = _tensor(fx[key])
        stats: dict[str, int] = {}
        out = calc_lag_info_gain(
            torch.log_softmax(logits, dim=-1), {"input_ids": ids}, params, stats=stats
        )
        assert torch.isfinite(out).all()
        assert out[0, 1, 2].item() < 100.0  # a large but finite divergence, not 3.4e38
        assert stats[direction] == 1, (key, stats)
        assert stats["n_cell_collapsed"] + stats["n_cell_saturated"] == 1
        # clamp mode on the same inputs reports the same count from the probabilities
        stats_clamp: dict[str, int] = {}
        calc_lag_info_gain(
            torch.softmax(logits, dim=-1),
            {"input_ids": ids},
            params,
            kl_mode="clamp",
            stats=stats_clamp,
        )
        assert stats_clamp[direction] == 1


def test_calc_lag_info_gain_modes_agree_where_clamp_is_finite():
    fx = _fixture()["lag_info_gain"]
    params = {"sampling": {"context": fx["context"]}}
    ids = torch.tensor(fx["input_ids"])
    logits = _tensor(fx["logits"])
    stats: dict[str, int] = {}
    clamp = calc_lag_info_gain(
        torch.softmax(logits, dim=-1), {"input_ids": ids}, params, kl_mode="clamp", stats=stats
    )
    logspace = calc_lag_info_gain(torch.log_softmax(logits, dim=-1), {"input_ids": ids}, params)
    assert torch.allclose(logspace, clamp, rtol=1e-5, atol=1e-7)
    assert stats["n_cell_collapsed"] == 0 and stats["n_cell_saturated"] == 0
    # Lc = 3 with 3 staircase rows tests causes 1 and 2 (row 0 is only ever a baseline), and
    # only effects after the cause are read: the single cell (1, 2).
    assert stats["n_cells"] == 1


def test_calc_lag_info_gain_reads_kl_mode_and_stats_from_params():
    fx = _fixture()["lag_info_gain"]
    ids = torch.tensor(fx["input_ids"])
    logits = _tensor(fx["logits_saturated_observed"])
    stats: dict[str, int] = {}
    params = {"sampling": {"context": fx["context"]}, "kl_mode": "clamp", "kl_stats": stats}
    calc_lag_info_gain(torch.softmax(logits, dim=-1), {"input_ids": ids}, params)
    assert stats["n_cell_collapsed"] == 1


def test_invalid_kl_mode_is_rejected():
    fx = _fixture()["lag_info_gain"]
    ids = torch.tensor(fx["input_ids"])
    probs = torch.softmax(_tensor(fx["logits"]), dim=-1)
    with pytest.raises(ValueError):
        calc_lag_info_gain(probs, {"input_ids": ids}, {"sampling": {"context": 2}}, kl_mode="bogus")
    assert set(KL_MODES) == {"logspace", "clamp"}


# ---------------------------------------------------------------- compute_cmi_matrix


def _cli_fixture_model_and_sequence():
    fx = _fixture()["cli_path"]
    scm, _ = create_scm(**fx["scm"])
    return fx, scm, torch.tensor(fx["sequence"])


def test_compute_cmi_matrix_clamp_mode_reproduces_v019_bit_for_bit():
    fx, scm, seq = _cli_fixture_model_and_sequence()
    for strategy, key in (("full", "expected_full"), ("atomic", "expected_atomic")):
        torch.manual_seed(fx["noise_seed"])
        out = compute_cmi_matrix(
            scm,
            seq,
            context_len=fx["context_len"],
            n_particles=fx["n_particles"],
            strategy=strategy,
            kl_mode="clamp",
        )
        assert _equal_with_nan(out, _tensor(fx[key])), strategy


def test_compute_cmi_matrix_logspace_agrees_with_clamp_and_counts():
    fx, scm, seq = _cli_fixture_model_and_sequence()
    for strategy in ("full", "atomic"):
        stats: dict[str, int] = {}
        torch.manual_seed(fx["noise_seed"])
        logspace = compute_cmi_matrix(
            scm,
            seq,
            context_len=fx["context_len"],
            n_particles=fx["n_particles"],
            strategy=strategy,
            stats=stats,
        )
        torch.manual_seed(fx["noise_seed"])
        clamp = compute_cmi_matrix(
            scm,
            seq,
            context_len=fx["context_len"],
            n_particles=fx["n_particles"],
            strategy=strategy,
            kl_mode="clamp",
        )
        assert torch.allclose(logspace, clamp, rtol=1e-5, atol=1e-7), strategy
        lc = seq.numel() - fx["context_len"]
        # full: the staircase never tests cause 0 -> (lc-1)(lc-2)/2 read cells;
        # atomic: every strict-upper-triangle cell -> lc(lc-1)/2.
        expected = (lc - 1) * (lc - 2) // 2 if strategy == "full" else lc * (lc - 1) // 2
        assert stats["n_cells"] == expected, strategy
        assert stats["n_cell_collapsed"] == 0 and stats["n_cell_saturated"] == 0


def test_compute_cmi_matrix_sparse_forwards_kl_mode_and_stats():
    scm, seqs = create_scm(vocab_size=12, memory=2, length=18, seed=1)
    stats: dict[str, int] = {}
    torch.manual_seed(0)
    out = compute_cmi_matrix_sparse(
        scm, seqs[0], context_len=5, memory=2, n_particles=4, kl_mode="logspace", stats=stats
    )
    assert torch.isfinite(out).all()
    assert stats["n_cells"] > 0


# ---------------------------------------------------------------- SampleLevelCausalDiscovery


def _tiny_model_and_dataset(vocab_size=6, seq_len=6):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=seq_len + 4,
    )
    model = LlamaForCausalLM(config).eval()
    ds_test = [{"input_ids": [1, 2, 3, 4, 5, 0], "attention_mask": [1, 1, 1, 1, 1, 1]}]
    return model, ds_test


def _params(kl_mode, stats, context=2, n_particles=3):
    return {
        "BS": 1,
        "full": True,
        "fp16": False,
        "cls_token_id": None,
        "kl_mode": kl_mode,
        "kl_stats": stats,
        "sampling": {"context": context, "guidance": context, "value": n_particles},
    }


def test_core_run_kl_modes_agree_and_record_counts():
    torch.manual_seed(0)
    model, ds_test = _tiny_model_and_dataset()
    outs = {}
    for mode in KL_MODES:
        stats: dict[str, int] = {}
        torch.manual_seed(1)
        algo = SampleLevelCausalDiscovery(model, _params(mode, stats), ds_test)
        algo.prepare()
        _, adj = algo.run()
        outs[mode] = adj
        assert stats["n_cells"] == 3  # Lc = 4: causes 1..3, effects after the cause -> 2 + 1 + 0
        assert "n_particle_collapsed" in stats
    assert torch.allclose(outs["logspace"], outs["clamp"], rtol=1e-5, atol=1e-7)
    assert torch.isfinite(outs["logspace"]).all()


def test_core_rejects_invalid_kl_mode():
    model, ds_test = _tiny_model_and_dataset()
    with pytest.raises(ValueError):
        SampleLevelCausalDiscovery(model, _params("bogus", {}), ds_test)
