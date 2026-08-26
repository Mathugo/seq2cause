"""Runs the exact Quick Start example shown in README.md, so the documented
usage never silently rots out of sync with the actual API."""

import torch

from seq2cause.diagnostics import compare_intervention_strategies, ground_truth_adjacency
from seq2cause.scm import create_scm
from seq2cause.threshold import select_threshold_by_validation


def test_readme_quick_start_example_recovers_a_sensible_causal_graph():
    torch.manual_seed(0)

    # 1. Stand-in "autoregressive model": a synthetic oracle generator (swap
    #    for your own trained GPT/LLaMA/RNN -- it just needs a `.vocab_size`
    #    attribute and a HF-style `forward(input_ids=...)` returning a dict
    #    with a "logits" key).
    scm, sequences = create_scm(vocab_size=15, memory=3, length=20, seed=0, sparsity=0.5)
    sequence = sequences[0]

    # 2. Ground-truth edges, only available/needed here because we're using a
    #    known synthetic generator to validate the method -- skip this step
    #    entirely when working with your own real event sequences.
    adjacency = ground_truth_adjacency(scm, sequence, threshold=0.05, n_counterfactuals=16)

    # 3. Per-(cause, effect) Conditional Mutual Information, using the
    #    recommended "atomic" intervention strategy (see README).
    results = compare_intervention_strategies(
        scm, sequence, context_len=3, adjacency=adjacency, n_particles=32, max_lag=3,
    )
    cmi_matrix = results["atomic"].cmi_matrix  # [L-c, L-c]

    # 4. Turn CMI scores into a binary causal graph by selecting a threshold
    #    on a held-out validation split (never hardcode a constant).
    scores = cmi_matrix.flatten()
    labels = adjacency[3:, 3:].flatten()
    result = select_threshold_by_validation(scores, labels, delta=0.05, emit_warnings=False)
    causal_graph = cmi_matrix >= result.tau

    assert cmi_matrix.shape == (17, 17)
    assert causal_graph.shape == (17, 17)
    assert result.f1 >= 0.5  # sanity check: this is a real, working recovery, not noise
