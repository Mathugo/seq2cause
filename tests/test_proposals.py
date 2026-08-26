import pytest
import torch

from seq2cause.sampling import uniform_sample, unigram_sample


def test_unigram_sample_requires_freqs():
    prob_x = torch.zeros(2, 4, 10)
    with pytest.raises(ValueError, match="unigram_freqs"):
        unigram_sample(prob_x, n_samples=3)


def test_unigram_sample_rejects_invalid_freqs():
    prob_x = torch.zeros(2, 4, 10)
    with pytest.raises(ValueError):
        unigram_sample(prob_x, n_samples=3, unigram_freqs=torch.zeros(10))
    with pytest.raises(ValueError):
        unigram_sample(prob_x, n_samples=3, unigram_freqs=-torch.ones(10))


def test_unigram_sample_only_draws_high_frequency_tokens():
    torch.manual_seed(0)
    vocab = 10
    freqs = torch.zeros(vocab)
    freqs[3] = 1.0  # all mass on token 3
    prob_x = torch.zeros(2, 5, vocab)
    sampled = unigram_sample(prob_x, n_samples=4, unigram_freqs=freqs)
    assert sampled.shape == (2, 4, 5)
    assert torch.all(sampled == 3)


def test_unigram_sample_cls_token_override():
    torch.manual_seed(0)
    vocab = 10
    freqs = torch.ones(vocab)
    prob_x = torch.zeros(2, 5, vocab)
    sampled = unigram_sample(prob_x, n_samples=4, unigram_freqs=freqs, cls_token_id=7)
    assert torch.all(sampled[:, :, 0] == 7)


def test_unigram_sample_2d_input():
    torch.manual_seed(0)
    vocab = 10
    freqs = torch.ones(vocab)
    prob_x = torch.zeros(3, vocab)
    sampled = unigram_sample(prob_x, unigram_freqs=freqs)
    assert sampled.shape == (3,)


def test_unigram_sample_matches_frequency_distribution_statistically():
    torch.manual_seed(0)
    vocab = 4
    freqs = torch.tensor([1.0, 3.0, 0.0, 0.0])  # token 1 should dominate ~75%
    prob_x = torch.zeros(1, 2000, vocab)
    sampled = unigram_sample(prob_x, n_samples=1, unigram_freqs=freqs)
    counts = torch.bincount(sampled.flatten(), minlength=vocab).float()
    proportions = counts / counts.sum()
    assert proportions[2] == 0.0
    assert proportions[3] == 0.0
    assert proportions[1] > proportions[0]


def test_uniform_sample_still_uniform_baseline_for_comparison():
    torch.manual_seed(0)
    vocab = 5
    prob_x = torch.zeros(1, 2000, vocab)
    sampled = uniform_sample(prob_x, n_samples=1)
    counts = torch.bincount(sampled.flatten(), minlength=vocab).float()
    proportions = counts / counts.sum()
    assert torch.all(proportions > 0.05)  # roughly uniform, not collapsed
