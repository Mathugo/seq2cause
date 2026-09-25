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


def test_uniform_sample_min_id_keeps_reserved_ids_out_of_the_draw():
    torch.manual_seed(0)
    vocab, n_specials = 9, 4
    for prob_x in (torch.zeros(3, vocab), torch.zeros(2, 5, vocab)):
        sampled = uniform_sample(prob_x, n_samples=64, min_id=n_specials)
        assert int(sampled.min()) >= n_specials
        assert int(sampled.max()) < vocab
    # every real id is still reachable
    sampled = uniform_sample(torch.zeros(1, 2000, vocab), n_samples=1, min_id=n_specials)
    counts = torch.bincount(sampled.flatten(), minlength=vocab)
    assert torch.all(counts[:n_specials] == 0) and torch.all(counts[n_specials:] > 0)


def test_uniform_sample_min_id_zero_is_the_previous_behaviour():
    torch.manual_seed(3)
    before = uniform_sample(torch.zeros(2, 7, 6), n_samples=5)
    torch.manual_seed(3)
    after = uniform_sample(torch.zeros(2, 7, 6), n_samples=5, min_id=0)
    assert torch.equal(before, after)


def test_uniform_sample_rejects_min_id_outside_the_vocabulary():
    for bad in (-1, 6, 7):
        with pytest.raises(ValueError):
            uniform_sample(torch.zeros(1, 6), n_samples=2, min_id=bad)
