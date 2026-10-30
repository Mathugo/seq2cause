import math

import pytest
import torch

from seq2cause.causal_strength import (
    calc_granger_score,
    calc_lag_cmi,
    calc_lag_info_gain,
    calc_neural_saliency,
    calc_neural_shapley,
)


def _synthetic_prob_x_inter(bs=1, n=2, num_rows=3, seq_len=5, vocab=4, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(bs, n, num_rows, seq_len, vocab)
    return torch.softmax(logits, dim=-1)


def _synthetic_batch(bs=1, seq_len=5, vocab=4, seed=1):
    torch.manual_seed(seed)
    return {"input_ids": torch.randint(0, vocab, (bs, seq_len))}


def test_calc_lag_info_gain_shape_and_zero_padding():
    context = 2
    prob_x_inter = _synthetic_prob_x_inter()
    batch = _synthetic_batch()
    params = {"sampling": {"context": context}}

    full_cmi = calc_lag_info_gain(prob_x_inter, batch, params)

    bs, n, num_rows, seq_len, vocab = prob_x_inter.shape
    lc = seq_len - context
    assert full_cmi.shape == (bs, lc, lc)
    # Only the last (num_rows - 1) rows are ever populated; earlier rows stay 0.
    start_row = lc - (num_rows - 1)
    assert torch.all(full_cmi[:, :start_row, :] == 0)
    assert not torch.isnan(full_cmi).any()


def test_calc_lag_info_gain_is_non_negative_kl():
    # KL divergence terms are always >= 0.
    prob_x_inter = _synthetic_prob_x_inter()
    batch = _synthetic_batch()
    params = {"sampling": {"context": 2}}
    full_cmi = calc_lag_info_gain(prob_x_inter, batch, params)
    assert torch.all(full_cmi >= 0)


def _hand_built_staircase(pi=0.01, q1=0.99, q0=0.01):
    """Probabilities of the observed token (id 0) for context c=1, L=4, three staircase
    rows. The pair (row 0, row 1) tests the cause at suffix position 1 on the effect at
    suffix position 2: pi at the cause position, q1 / q0 at the effect position."""
    c, seq_len = 1, 4
    p_event = torch.full((3, seq_len - c), 0.5, dtype=torch.float64)  # [rows, suffix pos]
    p_event[0, 1], p_event[1, 1] = pi, pi  # same trajectory before the cause
    p_event[0, 2], p_event[1, 2] = q0, q1
    probs = torch.zeros(1, 1, 3, seq_len, 2, dtype=torch.float64)
    probs[0, 0, :, c - 1 : seq_len - 1, 0] = p_event
    probs[0, 0, :, c - 1 : seq_len - 1, 1] = 1 - p_event
    probs[..., seq_len - 1, :] = 0.5
    batch = {"input_ids": torch.zeros(1, seq_len, dtype=torch.long)}
    params = {"sampling": {"context": c}}
    return probs, batch, params


def _bern_kl(q, p):
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def test_calc_lag_cmi_matches_closed_form():
    pi, q1, q0 = 0.01, 0.99, 0.01
    probs, batch, params = _hand_built_staircase(pi, q1, q0)
    m = pi * q1 + (1 - pi) * q0
    expected = pi * _bern_kl(q1, m) + (1 - pi) * _bern_kl(q0, m)

    cmi = calc_lag_cmi(probs.log(), batch, params, kl_mode="logspace")

    assert cmi[0, 1, 2].item() == pytest.approx(expected, rel=1e-9)
    assert cmi[0, 1, 2].item() == pytest.approx(0.0413, abs=1e-4)


def test_calc_lag_cmi_is_bounded_while_info_gain_is_not():
    # A rare cause with a strong effect: large information gain of the observed
    # cause, small CMI, since the CMI weights each value of the cause by its probability.
    probs, batch, params = _hand_built_staircase(pi=0.01, q1=0.99, q0=0.01)

    info_gain = calc_lag_info_gain(probs.log(), batch, params, kl_mode="logspace")
    cmi = calc_lag_cmi(probs.log(), batch, params, kl_mode="logspace")

    assert info_gain[0, 1, 2].item() > math.log(2)
    assert 0 < cmi[0, 1, 2].item() < math.log(2)


def test_calc_lag_cmi_detects_inhibitory_cause():
    probs, batch, params = _hand_built_staircase(pi=0.5, q1=0.05, q0=0.9)
    cmi = calc_lag_cmi(probs.log(), batch, params, kl_mode="logspace")
    assert cmi[0, 1, 2].item() > 0.1


def test_calc_lag_cmi_zero_without_effect():
    probs, batch, params = _hand_built_staircase(pi=0.3, q1=0.4, q0=0.4)
    cmi = calc_lag_cmi(probs.log(), batch, params, kl_mode="logspace")
    assert cmi[0, 1, 2].item() == pytest.approx(0.0, abs=1e-12)


def test_calc_lag_cmi_random_inputs_bounded_and_modes_agree():
    prob_x_inter = _synthetic_prob_x_inter(bs=2, n=4, num_rows=4, seq_len=6, vocab=5)
    batch = _synthetic_batch(bs=2, seq_len=6, vocab=5)
    params = {"sampling": {"context": 2}}

    cmi_log = calc_lag_cmi(prob_x_inter.log(), batch, params, kl_mode="logspace")
    cmi_clamp = calc_lag_cmi(prob_x_inter, batch, params, kl_mode="clamp")

    assert cmi_log.shape == (2, 4, 4)
    assert torch.all(cmi_log >= 0) and torch.all(cmi_log <= math.log(2) + 1e-6)
    # effects at or before the cause are never scored
    assert torch.all(torch.tril(cmi_log[0]) == 0)
    assert torch.allclose(cmi_log, cmi_clamp, atol=1e-5)


def test_calc_granger_score_diff_mode_shape():
    context = 2
    prob_x_inter = _synthetic_prob_x_inter()
    batch = _synthetic_batch()
    params = {"sampling": {"context": context}}

    score = calc_granger_score(prob_x_inter, batch, params, mode="diff")

    bs, n, num_rows, seq_len, vocab = prob_x_inter.shape
    lc = seq_len - context
    assert score.shape == (bs, lc, lc)
    assert not torch.isnan(score).any()
    # A probability difference must stay within [-1, 1].
    assert torch.all(score >= -1) and torch.all(score <= 1)


def test_calc_granger_score_log_ratio_mode_differs_from_diff():
    prob_x_inter = _synthetic_prob_x_inter()
    batch = _synthetic_batch()
    params = {"sampling": {"context": 2}}

    diff_score = calc_granger_score(prob_x_inter, batch, params, mode="diff")
    log_ratio_score = calc_granger_score(prob_x_inter, batch, params, mode="log_ratio")

    assert diff_score.shape == log_ratio_score.shape
    assert not torch.allclose(diff_score, log_ratio_score)


def test_calc_granger_score_rejects_invalid_mode():
    prob_x_inter = _synthetic_prob_x_inter()
    batch = _synthetic_batch()
    params = {"sampling": {"context": 2}}
    with pytest.raises(ValueError):
        calc_granger_score(prob_x_inter, batch, params, mode="bogus")


def _tiny_llama(vocab_size=6, seq_len=6):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=seq_len + 4,
    )
    return LlamaForCausalLM(config).eval()


def test_calc_neural_saliency_shape():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model = _tiny_llama(vocab_size, seq_len)
    batch = {"input_ids": torch.randint(0, vocab_size, (1, seq_len))}
    params = {"sampling": {"context": context}}

    _, adj = calc_neural_saliency(model, batch, params)

    assert adj.shape == (1, seq_len - context, seq_len - context)
    assert not torch.isnan(adj).any()


def test_calc_neural_shapley_shape():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 5, 2
    model = _tiny_llama(vocab_size, seq_len)
    batch = {"input_ids": torch.randint(0, vocab_size, (1, seq_len))}
    params = {"sampling": {"context": context}}

    _, adj = calc_neural_shapley(model, batch, params)

    assert adj.shape == (1, seq_len - context, seq_len - context)
    assert not torch.isnan(adj).any()
