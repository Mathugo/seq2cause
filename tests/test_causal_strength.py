import pytest
import torch

from seq2cause.causal_strength import (
    calc_granger_score,
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
