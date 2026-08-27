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
