"""Runs the exact Quick Start / Evaluation examples shown in README.md, so
the documented usage never silently rots out of sync with the actual API."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from seq2cause.adapters import HFModelAdapter
from seq2cause.diagnostics import compare_intervention_strategies, ground_truth_adjacency
from seq2cause.scm import create_scm
from seq2cause.threshold import otsu_threshold, select_threshold_by_validation


def test_readme_quick_start_example_runs_without_ground_truth():
    """Quick Start: a real (here, randomly-initialized standing in for
    pretrained/fine-tuned) HF model on a real event sequence, no known
    generator, no labeled ground truth -- an unsupervised threshold cutoff."""
    torch.manual_seed(0)

    vocab_size = 20
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    )).eval()
    adapter = HFModelAdapter(model, vocab_size=vocab_size)
    sequence = torch.randint(0, vocab_size, (20,))

    placeholder_adjacency = torch.zeros(len(sequence), len(sequence), dtype=torch.bool)
    results = compare_intervention_strategies(
        adapter, sequence, context_len=3, adjacency=placeholder_adjacency, n_particles=32, max_lag=3,
    )
    cmi_matrix = results["atomic"].cmi_matrix

    causal_graph = cmi_matrix >= otsu_threshold(cmi_matrix.flatten())

    assert cmi_matrix.shape == (17, 17)
    assert causal_graph.shape == (17, 17)
    assert causal_graph.dtype == torch.bool


def test_readme_evaluation_example_recovers_a_sensible_causal_graph():
    """Evaluation: a known synthetic generator with ground-truth edges,
    validating the method end to end via a validation-selected threshold."""
    torch.manual_seed(0)

    scm, sequences = create_scm(vocab_size=15, memory=3, length=20, seed=0, sparsity=0.5)
    sequence = sequences[0]

    adjacency = ground_truth_adjacency(scm, sequence, threshold=0.05, n_counterfactuals=16)

    results = compare_intervention_strategies(
        scm, sequence, context_len=3, adjacency=adjacency, n_particles=32, max_lag=3,
    )
    cmi_matrix = results["atomic"].cmi_matrix

    scores, labels = cmi_matrix.flatten(), adjacency[3:, 3:].flatten()
    result = select_threshold_by_validation(scores, labels, delta=0.05, emit_warnings=False)
    causal_graph = cmi_matrix >= result.tau

    assert cmi_matrix.shape == (17, 17)
    assert causal_graph.shape == (17, 17)
    assert result.f1 >= 0.5  # sanity check: this is a real, working recovery, not noise

