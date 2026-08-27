import pytest
import torch

from seq2cause.diagnostics import (
    compare_intervention_strategies,
    compute_cmi_matrix,
    compute_cmi_matrix_sparse,
    estimate_oracle_score,
    ground_truth_adjacency,
    recall_by_lag,
    recall_by_lag_with_tau_by_lag,
    snr_by_lag,
    summary_graph,
)
from seq2cause.scm import NonlinearSCM, create_scm


def test_create_scm_returns_valid_sequence():
    scm, seq = create_scm(vocab_size=30, memory=3, length=15, seed=0, batch_size=2)
    assert isinstance(scm, NonlinearSCM)
    assert seq.shape == (2, 15)
    assert seq.min() >= 0
    assert seq.max() < 30


def test_decay_favors_recency_over_distant_lag():
    """Eq. 21's linear term decays as exp(-(k-1)), where k=1 is the MOST
    RECENT lag (strongest weight, decay=1) and k=memory is the most distant
    (weakest). Regression test for a bug where this was inverted: `context`
    is ordered oldest-first, so a naive `exp(-arange(memory))` (without
    reversing) instead puts the strongest weight on the OLDEST token."""
    scm = NonlinearSCM(vocab_size=15, memory=4, seed=0)
    assert scm._decay[-1].item() == pytest.approx(1.0)  # lag=1 (most recent token)
    assert scm._decay[0].item() == pytest.approx(torch.exp(torch.tensor(-3.0)).item())  # lag=4
    assert torch.all(scm._decay[1:] >= scm._decay[:-1])  # monotonically increasing with recency


def test_linear_term_weights_recent_lag_more_than_distant_lag():
    """End-to-end version of the above: swapping the MOST RECENT token in a
    context should perturb the (linear-only) logits more than swapping the
    OLDEST token in the same window."""
    scm = NonlinearSCM(vocab_size=15, memory=4, embed_dim=4, hidden_dim=8, seed=0)
    scm.W1.zero_()  # isolates the linear branch: ReLU(x @ 0) @ W2 == 0

    base_context = torch.zeros(scm.memory, dtype=torch.long)
    base_logits = scm.conditional_logits(base_context)

    oldest_swapped = base_context.clone()
    oldest_swapped[0] = 1  # lag = memory
    delta_oldest = (scm.conditional_logits(oldest_swapped) - base_logits).norm()

    most_recent_swapped = base_context.clone()
    most_recent_swapped[-1] = 1  # lag = 1
    delta_recent = (scm.conditional_logits(most_recent_swapped) - base_logits).norm()

    assert delta_recent > delta_oldest


def test_scm_forward_produces_valid_distributions():
    scm, seq = create_scm(vocab_size=20, memory=2, length=10, seed=1)
    out = scm.forward(input_ids=seq)
    probs = torch.softmax(out["logits"], dim=-1)
    assert probs.shape == (1, 10, 20)
    # softmax always sums to 1 regardless of the underlying logits, so this
    # alone doesn't prove the logits are meaningful -- also check that
    # positions with a valid context window (>= memory-1, see `forward`'s
    # docstring on its indexing convention) are non-uniform (i.e. genuinely
    # computed), while positions with insufficient context are exactly
    # uniform (all-zero logits).
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    uniform = torch.full((20,), 1 / 20)
    assert not torch.allclose(probs[0, scm.memory - 1], uniform, atol=1e-6)
    assert torch.allclose(probs[0, scm.memory - 2], uniform, atol=1e-6)


def test_scm_forward_matches_standard_causal_lm_shift_convention():
    """`logits[..., i, :]` must predict the token AT POSITION i+1 (using the
    memory tokens immediately before it), matching real transformers and
    every other piece of this codebase that consumes `forward()`'s output
    -- regression test for a caught off-by-one bug (an earlier version
    stored the prediction for token i at index i instead of i+1, which
    `estimate_oracle_score` exposed by reading epsilon_hat >> 0 for the
    oracle model evaluated against itself)."""
    scm = NonlinearSCM(vocab_size=15, memory=3, seed=0)
    seq = torch.randint(0, 15, (1, 8))
    logits = scm.forward(input_ids=seq)["logits"]
    for i in range(scm.memory - 1, seq.shape[-1] - 1):
        expected = scm.conditional_logits(seq[0, i - scm.memory + 1 : i + 1])
        assert torch.allclose(logits[0, i], expected, atol=1e-5)


def test_scm_forward_rejects_too_short_sequence():
    scm = NonlinearSCM(vocab_size=10, memory=4)
    import pytest

    with pytest.raises(ValueError):
        scm.forward(input_ids=torch.zeros(1, 2, dtype=torch.long))


def test_ground_truth_adjacency_only_has_edges_within_memory_window():
    torch.manual_seed(0)
    scm, seq = create_scm(vocab_size=25, memory=3, length=12, seed=3)
    adj = ground_truth_adjacency(scm, seq[0], threshold=0.02, n_counterfactuals=4)
    L = seq.shape[-1]
    for j in range(L):
        for q in range(L):
            if adj[j, q]:
                assert 1 <= (q - j) <= scm.memory


def test_snr_by_lag_and_recall_by_lag_shapes():
    lc = 6
    cmi = torch.rand(lc, lc)
    adjacency = torch.zeros(lc, lc, dtype=torch.bool)
    adjacency[1, 2] = True  # lag 1
    adjacency[0, 3] = True  # lag 3

    stats = snr_by_lag(cmi, adjacency, max_lag=3)
    assert set(stats.keys()) == {1, 2, 3}
    assert stats[1]["n_true"] == 1
    assert stats[3]["n_true"] == 1
    assert stats[2]["n_true"] == 0

    recall = recall_by_lag(cmi, adjacency, tau=2.0, max_lag=3)  # tau above any CMI value
    assert recall[1] == 0.0
    assert recall[3] == 0.0
    assert recall[2] != recall[2] or recall[2] == 0.0  # nan or 0, no true edges at lag 2


def test_recall_by_lag_with_tau_by_lag_tailors_threshold_per_lag():
    lc = 6
    cmi = torch.zeros(lc, lc)
    cmi[1, 2] = 0.5  # lag 1, large CMI scale
    cmi[0, 3] = 0.01  # lag 3, small CMI scale, but still a real true edge
    adjacency = torch.zeros(lc, lc, dtype=torch.bool)
    adjacency[1, 2] = True
    adjacency[0, 3] = True

    # A single shared tau tuned for lag 1's scale misses lag 3 entirely.
    shared = recall_by_lag(cmi, adjacency, tau=0.1, max_lag=3)
    assert shared[1] == 1.0
    assert shared[3] == 0.0

    # Per-lag tau, tailored to each lag's own causal-strength scale, recovers it.
    tailored = recall_by_lag_with_tau_by_lag(
        cmi, adjacency, tau_by_lag={1: 0.1, 3: 0.005}, max_lag=3
    )
    assert tailored[1] == 1.0
    assert tailored[3] == 1.0


def test_recall_by_lag_with_tau_by_lag_missing_lag_is_nan():
    lc = 4
    cmi = torch.zeros(lc, lc)
    adjacency = torch.zeros(lc, lc, dtype=torch.bool)
    result = recall_by_lag_with_tau_by_lag(cmi, adjacency, tau_by_lag={1: 0.1}, max_lag=2)
    assert result[1] != result[1] or result[1] == 0.0  # no true edges either way -> nan/0
    assert result[2] != result[2]  # lag 2 has no tau provided -> nan


def test_estimate_entropy_between_zero_and_log_vocab():
    scm = NonlinearSCM(vocab_size=25, memory=2, seed=0)
    h_p = scm.estimate_entropy(n_sequences=50, length=12)
    h_max = torch.log(torch.tensor(25.0)).item()
    assert 0.0 <= h_p <= h_max + 1e-4


def test_estimate_oracle_score_is_near_zero_for_the_oracle_itself():
    """Using the SCM as its own "model" should read epsilon_hat ~ 0, since
    its cross-entropy on fresh data IS (in expectation) the process's own
    irreducible entropy H(P)."""
    scm = NonlinearSCM(vocab_size=20, memory=2, seed=0)
    result = estimate_oracle_score(scm, scm, n_sequences=100, length=12)
    assert set(result.keys()) == {"loss", "h_p", "h_max", "epsilon_hat"}
    assert result["epsilon_hat"] == pytest.approx(0.0, abs=0.05)


def test_estimate_oracle_score_is_large_for_a_uniform_random_model():
    """A model that always predicts uniformly should read epsilon_hat ~ 1
    (loss ~ H_max, the worst case in the oracle-score normalization)."""

    class _UniformModel:
        def forward(self, input_ids, **kwargs):
            *lead, seq_len = input_ids.shape
            vocab_size = 20
            return {"logits": torch.zeros(*lead, seq_len, vocab_size)}

    scm = NonlinearSCM(vocab_size=20, memory=2, seed=0)
    result = estimate_oracle_score(_UniformModel(), scm, n_sequences=100, length=12)
    assert result["epsilon_hat"] == pytest.approx(1.0, abs=0.05)



def test_compare_intervention_strategies_runs_all_strategies_end_to_end():
    torch.manual_seed(0)
    vocab, memory, length, context = 25, 2, 12, 2
    scm, seq_batch = create_scm(vocab_size=vocab, memory=memory, length=length, seed=4)
    seq = seq_batch[0]
    adj = ground_truth_adjacency(scm, seq, threshold=0.02, n_counterfactuals=3)
    unigram_freqs = torch.bincount(seq, minlength=vocab).float() + 1.0

    results = compare_intervention_strategies(
        scm,
        seq,
        context_len=context,
        adjacency=adj,
        n_particles=4,
        window_k=1,
        tau=0.01,
        max_lag=memory,
        unigram_freqs=unigram_freqs,
    )

    expected_strategies = {"full", "atomic", "windowed", "independent_mediator", "in_distribution_noise"}
    assert set(results.keys()) == expected_strategies
    lc = length - context
    for res in results.values():
        assert res.cmi_matrix.shape == (lc, lc)
        assert set(res.snr.keys()) == set(range(1, memory + 1))
        assert res.recall is not None
        assert set(res.recall.keys()) == set(range(1, memory + 1))


def test_compare_intervention_strategies_without_unigram_freqs_skips_that_strategy():
    torch.manual_seed(0)
    vocab, memory, length, context = 20, 2, 10, 2
    scm, seq_batch = create_scm(vocab_size=vocab, memory=memory, length=length, seed=5)
    seq = seq_batch[0]
    adj = ground_truth_adjacency(scm, seq, threshold=0.02, n_counterfactuals=2)

    results = compare_intervention_strategies(
        scm, seq, context_len=context, adjacency=adj, n_particles=4, max_lag=memory
    )
    assert "in_distribution_noise" not in results
    assert "full" in results


# ---------------------------------------------------------------------------
# compute_cmi_matrix_sparse: bounded-memory ("sparse") construction.
# ---------------------------------------------------------------------------


def test_compute_cmi_matrix_sparse_matches_shape_of_full():
    memory, context_len = 3, 4
    scm, seq = create_scm(vocab_size=15, memory=memory, length=20, seed=0)
    sequence = seq[0]

    full_cmi = compute_cmi_matrix(
        scm, sequence, context_len=context_len, n_particles=16, strategy="full"
    )
    sparse_cmi = compute_cmi_matrix_sparse(
        scm, sequence, context_len=context_len, memory=memory, n_particles=16
    )
    assert sparse_cmi.shape == full_cmi.shape


def test_compute_cmi_matrix_sparse_never_writes_beyond_the_memory_bound():
    memory, context_len = 2, 5
    scm, seq = create_scm(vocab_size=12, memory=memory, length=18, seed=1)
    sequence = seq[0]

    sparse_cmi = compute_cmi_matrix_sparse(
        scm, sequence, context_len=context_len, memory=memory, n_particles=16
    )
    lc = sparse_cmi.shape[-1]
    for j in range(lc):
        for q in range(lc):
            lag = q - j
            if lag < 1 or lag > memory:
                assert sparse_cmi[j, q] == 0.0


def test_compute_cmi_matrix_sparse_requires_context_greater_than_memory():
    scm, seq = create_scm(vocab_size=10, memory=3, length=10, seed=0)
    with pytest.raises(ValueError):
        compute_cmi_matrix_sparse(scm, seq[0], context_len=3, memory=3, n_particles=8)
    with pytest.raises(ValueError):
        compute_cmi_matrix_sparse(scm, seq[0], context_len=2, memory=3, n_particles=8)


def test_compute_cmi_matrix_sparse_matches_full_within_the_bounded_lag_region():
    """Core correctness requirement: on a memory-bounded, decayed DGP, the
    sparse (bounded-memory) construction and the unbounded "full"
    construction should recover essentially the SAME per-cell CMI values
    within `lag <= memory` (small differences are only due to independently
    drawn do-intervention particles, not a systematic discrepancy -- see the
    correlation check below, which is robust to that noise)."""
    torch.manual_seed(0)
    memory, context_len = 3, 4
    scm, seq = create_scm(
        vocab_size=15, memory=memory, length=30, seed=0, sparsity=0.6, decay_rate=0.4
    )
    sequence = seq[0]

    full_cmi = compute_cmi_matrix(
        scm, sequence, context_len=context_len, n_particles=128, strategy="full"
    )
    sparse_cmi = compute_cmi_matrix_sparse(
        scm, sequence, context_len=context_len, memory=memory, n_particles=128
    )

    lc = full_cmi.shape[-1]
    lag = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
    mask = (lag >= 1) & (lag <= memory)

    f, s = full_cmi[mask], sparse_cmi[mask]
    assert torch.corrcoef(torch.stack([f, s]))[0, 1] > 0.9
    assert (f - s).abs().mean() < 0.02


def test_compute_cmi_matrix_sparse_f1_matches_full_f1():
    """The user-facing requirement: F1 (via a validation-selected threshold)
    computed from the sparse CMI matrix should match F1 from the full CMI
    matrix, pooled across several sequences from the same memory-bounded,
    decayed DGP."""
    from seq2cause.threshold import select_threshold_by_validation

    torch.manual_seed(0)
    memory, context_len = 3, 4
    scm, sequences = create_scm(
        vocab_size=15, memory=memory, length=30, seed=0, sparsity=0.6, decay_rate=0.4,
        batch_size=6,
    )

    def _pooled(strategy_fn):
        scores, labels = [], []
        for i in range(sequences.shape[0]):
            sequence = sequences[i]
            cmi = strategy_fn(sequence)
            adjacency = ground_truth_adjacency(scm, sequence, threshold=0.05, n_counterfactuals=12)
            adj_suffix = adjacency[context_len:, context_len:]
            scores.append(cmi.flatten())
            labels.append(adj_suffix.flatten())
        return torch.cat(scores), torch.cat(labels)

    full_scores, full_labels = _pooled(
        lambda sequence: compute_cmi_matrix(
            scm, sequence, context_len=context_len, n_particles=64, strategy="full"
        )
    )
    sparse_scores, sparse_labels = _pooled(
        lambda sequence: compute_cmi_matrix_sparse(
            scm, sequence, context_len=context_len, memory=memory, n_particles=64
        )
    )

    full_result = select_threshold_by_validation(full_scores, full_labels, emit_warnings=False)
    sparse_result = select_threshold_by_validation(
        sparse_scores, sparse_labels, emit_warnings=False
    )
    assert abs(full_result.f1 - sparse_result.f1) < 0.15


def test_summary_graph_projects_position_edges_to_event_types_with_union_aggregation():
    # context_len=2 -> suffix tokens are [3, 7, 3, 7, 5] at positions [0..4].
    sequence = torch.tensor([9, 9, 3, 7, 3, 7, 5])
    causal_graph = torch.zeros((5, 5), dtype=torch.bool)
    causal_graph[0, 1] = True  # pos0(3) -> pos1(7)
    causal_graph[2, 3] = True  # pos2(3) -> pos3(7): same type pair, union'd away
    causal_graph[1, 4] = True  # pos1(7) -> pos4(5)

    active_tokens, adj = summary_graph(sequence, causal_graph, context_len=2)

    assert torch.equal(active_tokens, torch.tensor([3, 5, 7]))
    assert adj.dtype == torch.bool
    assert adj.shape == (3, 3)
    # 3 -> 7
    assert adj[0, 2].item() is True
    # 7 -> 5
    assert adj[2, 1].item() is True
    assert int(adj.sum()) == 2


def test_summary_graph_excludes_self_loops_by_default():
    # suffix = [4, 4]: a position-level edge pos0 -> pos1 connects two
    # occurrences of the SAME event type.
    sequence = torch.tensor([0, 4, 4])
    causal_graph = torch.tensor([[False, True], [False, False]])

    active_tokens, adj = summary_graph(sequence, causal_graph, context_len=1)
    assert torch.equal(active_tokens, torch.tensor([4]))
    assert not bool(adj.any())

    active_tokens, adj = summary_graph(sequence, causal_graph, context_len=1, self_loops=True)
    assert bool(adj[0, 0])


def test_summary_graph_scales_with_sequence_not_vocab_size():
    """The summary graph must stay bounded by the number of DISTINCT event
    types actually observed, not by (a possibly huge) vocab_size."""
    vocab_size = 50_257  # e.g. GPT-2's real vocabulary
    sequence = torch.tensor([1, 2, 3, 2, 1])
    causal_graph = torch.zeros((4, 4), dtype=torch.bool)
    causal_graph[0, 1] = True

    active_tokens, adj = summary_graph(sequence, causal_graph, context_len=1)

    assert active_tokens.numel() < vocab_size
    assert adj.shape[0] < vocab_size
