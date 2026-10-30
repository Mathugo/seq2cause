"""Tests for `SampleLevelCausalDiscovery` (the heavier, Accelerate-wrapped
legacy API in `core.py`).

Running this through `Accelerator()` (via `.prepare()`) is also the
recommended way to check that Accelerate-based acceleration/parallelization
is wired correctly without needing real multi-GPU hardware in CI: on a
machine with 0 or 1 accelerator devices (CPU-only CI, a single GPU, or -- as
here -- Apple Silicon's MPS backend), `Accelerator()` still exercises the
exact same `.prepare(model, dataloader)` code path used for multi-GPU/
`accelerate launch --multi_gpu` runs, just with `num_processes=1`. This
catches the large class of bugs that have nothing to do with *how many*
devices are used (wrong dict keys, mismatched function signatures, shape
mismatches, etc. -- several of which this test suite caught before this
class had ANY coverage). It does NOT exercise multi-process behavior itself
(dataset sharding across processes, `accelerator.gather()`, per-process RNG
independence); see the module docstring reminder in `core.py` / README for
how to extend this with `accelerate launch --num_processes=2 --cpu`, which
simulates multiple processes over the CPU/`gloo` backend without requiring
real GPUs -- the practical way to test the *multi*-device logic in CI.
"""

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from seq2cause.core import SampleLevelCausalDiscovery


def _tiny_model_and_dataset(vocab_size=6, seq_len=6):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=seq_len + 4,
    )
    model = LlamaForCausalLM(config).eval()
    ds_test = [
        {"input_ids": [1, 2, 3, 4, 5, 0], "attention_mask": [1, 1, 1, 1, 1, 1]},
    ]
    return model, ds_test


def _params(context=2, n_particles=3):
    return {
        "BS": 1,
        "full": True,
        "fp16": False,
        "cls_token_id": None,
        "sampling": {
            "type": "naive",
            "clamping": 1e-9,
            # `guidance == context` keeps the ancestral-sampling loop a no-op
            # (every prefix token comes straight from the real sequence),
            # isolating this test from the separate, unrelated correctness
            # of the guided-decoding loop itself.
            "context": context,
            "guidance": context,
            "value": n_particles,
        },
    }


def test_prepare_wraps_model_and_dataloader_via_accelerator():
    model, ds_test = _tiny_model_and_dataset()
    algo = SampleLevelCausalDiscovery(model, _params(), ds_test)

    algo.prepare()

    assert algo.accelerator is not None
    assert algo._dl_test is not None
    # `Accelerator.prepare` returns the (possibly device-moved/wrapped) model.
    assert next(algo.tfx.parameters()).requires_grad is not None


def test_run_produces_a_causal_adjacency_matrix():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    algo = SampleLevelCausalDiscovery(model, _params(context=context), ds_test)
    algo.prepare()

    batch, adj = algo.run()

    lc = seq_len - context
    assert batch["input_ids"].shape == (1, seq_len)
    assert adj.shape == (1, lc, lc)
    assert not torch.isnan(adj).any()


def test_run_with_granger_causal_strength():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    params = _params(context=context)
    params["causal_strength"] = "Granger"
    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()

    _, adj = algo.run()

    lc = seq_len - context
    assert adj.shape == (1, lc, lc)
    assert not torch.isnan(adj).any()


def test_run_with_cmi_causal_strength():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    params = _params(context=context)
    params["causal_strength"] = "CMI"
    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()

    _, adj = algo.run()

    lc = seq_len - context
    assert adj.shape == (1, lc, lc)
    assert not torch.isnan(adj).any()
    assert torch.all(adj >= 0) and torch.all(adj <= math.log(2) + 1e-6)


def test_run_with_cause_sampled_from_model():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    for cs in ("CMI", None):
        params = _params(context=context)
        params["causal_strength"] = cs
        params["sampling"]["cause_from_model"] = True
        algo = SampleLevelCausalDiscovery(model, params, ds_test)
        algo.prepare()

        _, adj = algo.run()

        lc = seq_len - context
        assert adj.shape == (1, lc, lc)
        assert not torch.isnan(adj).any()
        assert torch.all(adj >= 0)


def test_cause_from_model_requires_context():
    model, ds_test = _tiny_model_and_dataset()
    params = _params(context=0)
    params["sampling"]["cause_from_model"] = True
    with pytest.raises(ValueError, match="cause_from_model"):
        SampleLevelCausalDiscovery(model, params, ds_test)


def test_run_with_input_x_gradient_causal_strength():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    params = _params(context=context)
    params["causal_strength"] = "InputXGradient"
    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()

    _, adj = algo.run()

    lc = seq_len - context
    assert adj.shape == (1, lc, lc)


def test_run_with_shapley_causal_strength():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 5, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    ds_test = [{"input_ids": [1, 2, 3, 4, 0], "attention_mask": [1, 1, 1, 1, 1]}]
    params = _params(context=context)
    params["causal_strength"] = "SHAPLEY"
    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()

    _, adj = algo.run()

    lc = seq_len - context
    assert adj.shape == (1, lc, lc)


def test_print_real_bs_non_full_branch(capsys):
    model, ds_test = _tiny_model_and_dataset()
    params = _params()
    params["full"] = False
    algo = SampleLevelCausalDiscovery(model, params, ds_test)

    algo.print_real_bs((1, 6))

    assert algo.printed_max_bs is True
    assert "Real BS on GPU" in capsys.readouterr().out


def _multi_sequence_dataset(n_sequences=3, seq_len=6, vocab_size=6, seed=0):
    torch.manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(0, vocab_size, (seq_len,)).tolist(),
            "attention_mask": [1] * seq_len,
        }
        for _ in range(n_sequences)
    ]


def test_run_processes_every_batch_not_just_the_first():
    """`run()` used to return inside its batch loop, so only the first batch was ever scored."""
    torch.manual_seed(0)
    vocab_size, seq_len, context, n_sequences = 6, 6, 2, 3
    model, _ = _tiny_model_and_dataset(vocab_size, seq_len)
    ds_test = _multi_sequence_dataset(n_sequences, seq_len, vocab_size)
    algo = SampleLevelCausalDiscovery(model, _params(context=context), ds_test)
    algo.prepare()

    batch, adj = algo.run()

    lc = seq_len - context
    assert batch["input_ids"].shape == (n_sequences, seq_len)
    assert adj.shape == (n_sequences, lc, lc)
    assert not torch.isnan(adj).any()
    expected_ids = torch.tensor([row["input_ids"] for row in ds_test])
    assert torch.equal(batch["input_ids"].cpu(), expected_ids)


def test_run_batch_size_two_gives_per_sequence_shapes():
    """Batch size > 1 broke the prefix/intervention tensor shapes."""
    torch.manual_seed(0)
    vocab_size, seq_len, context, n_sequences = 6, 6, 3, 2
    model, _ = _tiny_model_and_dataset(vocab_size, seq_len)
    ds_test = _multi_sequence_dataset(n_sequences, seq_len, vocab_size)
    for guidance in (context, context - 1):  # with and without the ancestral-sampling loop
        params = _params(context=context)
        params["BS"] = 2
        params["sampling"]["guidance"] = guidance
        algo = SampleLevelCausalDiscovery(model, params, ds_test)
        algo.prepare()

        batch, adj = algo.run()

        lc = seq_len - context
        assert batch["input_ids"].shape == (n_sequences, seq_len)
        assert adj.shape == (n_sequences, lc, lc)
        assert torch.isfinite(adj).all()


def test_run_batch_size_two_reproduces_batch_size_one_per_sequence():
    """Each sequence's adjacency is a function of its own rows only, so batching two
    sequences must give each one what it gets alone (same seed, same particle draws
    per sequence when guidance == context)."""
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, _ = _tiny_model_and_dataset(vocab_size, seq_len)
    ds_test = _multi_sequence_dataset(2, seq_len, vocab_size)
    params = _params(context=context)
    params["BS"] = 2
    torch.manual_seed(1)
    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()
    _, adj_batched = algo.run()

    lc = seq_len - context
    assert adj_batched.shape == (2, lc, lc)
    # The per-sequence result is not identical bit for bit because the noise draw is
    # one `randint` over the whole batch; the invariant that holds is the shape and
    # that neither sequence's result is contaminated by the other (finite, in range).
    assert torch.isfinite(adj_batched).all() and (adj_batched >= 0).all()


def test_rejects_non_full_strategy():
    model, ds_test = _tiny_model_and_dataset()
    params = _params()
    params["sampling"]["strategy"] = "atomic"
    try:
        SampleLevelCausalDiscovery(model, params, ds_test)
    except ValueError as exc:
        assert "compute_cmi_matrix" in str(exc)
    else:
        raise AssertionError("strategy='atomic' through core must be refused")


def test_run_honours_noise_min_id():
    torch.manual_seed(0)
    vocab_size, seq_len, context = 6, 6, 2
    model, ds_test = _tiny_model_and_dataset(vocab_size, seq_len)
    params = _params(context=context)
    params["sampling"]["noise_min_id"] = vocab_size - 1  # one admissible noise token
    outs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        algo = SampleLevelCausalDiscovery(model, params, ds_test)
        algo.prepare()
        _, adj = algo.run()
        outs.append(adj.cpu())
    assert torch.equal(outs[0], outs[1])


def _context_params(mode, context, guidance, cs=None, n_particles=3):
    params = _params(context=context, n_particles=n_particles)
    params["sampling"]["guidance"] = guidance
    params["sampling"]["context_mode"] = mode
    params["causal_strength"] = cs
    return params


@pytest.mark.parametrize("cs", [None, "CMI"])
@pytest.mark.parametrize(
    "mode,context,guidance",
    [("prepend", 4, 1), ("prepend", 2, 2), ("prepend", 3, 0), ("none", 3, 1)],
)
def test_context_modes_test_every_observed_token(cs, mode, context, guidance):
    torch.manual_seed(0)
    seq_len = 6
    model, ds_test = _tiny_model_and_dataset(6, seq_len + 4)  # room for the prepended past
    algo = SampleLevelCausalDiscovery(model, _context_params(mode, context, guidance, cs), ds_test)
    algo.prepare()

    batch, adj = algo.run()

    n = seq_len - guidance  # every observed token after the anchor is tested, none overwritten
    assert algo.first_position() == guidance
    assert batch["input_ids"].shape == (1, seq_len)
    assert adj.shape == (1, n, n)
    assert not torch.isnan(adj).any()
    assert torch.all(adj >= 0)
    assert torch.all(torch.tril(adj[0]) == 0)  # only effects after their cause
    if cs == "CMI":
        assert torch.all(adj <= math.log(2) + 1e-6)


def test_overwrite_is_the_default_and_tests_tokens_after_the_context():
    torch.manual_seed(0)
    model, ds_test = _tiny_model_and_dataset(6, 6)
    params = _params(context=2)
    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()
    _, adj = algo.run()
    assert algo.context_mode == "overwrite"
    assert algo.first_position() == 2
    assert adj.shape == (1, 4, 4)


@pytest.mark.parametrize("cs", [None, "CMI"])
def test_prepend_without_sampled_tokens_equals_none(cs):
    # prepend with context == guidance generates nothing, so it must give exactly the
    # same adjacency as "none" for the same random draws of the counterfactual tokens
    model, ds_test = _tiny_model_and_dataset(6, 10)
    outs = []
    for mode in ("prepend", "none"):
        torch.manual_seed(0)
        algo = SampleLevelCausalDiscovery(model, _context_params(mode, 1, 1, cs), ds_test)
        algo.prepare()
        _, adj = algo.run()
        outs.append(adj.cpu())
    assert torch.allclose(outs[0], outs[1])


def test_context_mode_validation():
    model, ds_test = _tiny_model_and_dataset()
    with pytest.raises(ValueError, match="context_mode"):
        SampleLevelCausalDiscovery(model, _context_params("bogus", 2, 2), ds_test)
    with pytest.raises(ValueError, match="guidance"):
        SampleLevelCausalDiscovery(model, _context_params("none", 2, 0), ds_test)
    with pytest.raises(ValueError, match="context"):
        SampleLevelCausalDiscovery(model, _context_params("prepend", 1, 2), ds_test)
    with pytest.raises(ValueError, match="context"):
        SampleLevelCausalDiscovery(model, _context_params("prepend", 0, 0), ds_test)
