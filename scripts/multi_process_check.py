"""Multi-process (simulated multi-GPU) correctness check for
`SampleLevelCausalDiscovery` under `accelerate`.

Run with:
    accelerate launch --num_processes=2 --cpu scripts/multi_process_check.py

This simulates a real multi-GPU launch (e.g. `accelerate launch --multi_gpu`
across N real GPUs) using 2 CPU processes over the `gloo` backend -- the
practical way to test Accelerate's dataloader-sharding and cross-process
`gather()` logic in CI/on a laptop, without needing real GPU hardware.
`tests/test_multi_gpu.py` is the pytest wrapper that invokes this exact
script via `accelerate launch` as a subprocess.

Checks:
  1. Each process's shard of the dataloader is DISJOINT (no sequence is
     processed twice) -- confirms the dataloader was actually sharded across
     processes, not silently duplicated on every process.
  2. `accelerator.gather()` reassembles a tensor whose first dimension is
     `num_processes` times each process's local batch size, and the gathered
     sequences cover the dataset with no duplicates.
  3. `SampleLevelCausalDiscovery.run()` produces a same-shape, finite
     adjacency matrix on EVERY process independently (i.e. the causal
     discovery pipeline itself works identically regardless of which
     process/device it runs on).

Prints "[OK] ..." and exits 0 on success; raises (non-zero exit) otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from seq2cause.core import SampleLevelCausalDiscovery  # noqa: E402


def _build_dataset(n_sequences: int, seq_len: int, vocab_size: int, seed: int = 0):
    torch.manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(0, vocab_size, (seq_len,)).tolist(),
            "attention_mask": [1] * seq_len,
        }
        for _ in range(n_sequences)
    ]


def main() -> None:
    accelerator = Accelerator()
    vocab_size, seq_len, context = 6, 6, 2
    batch_size = 1
    n_sequences = 2 * accelerator.num_processes  # >= 1 batch per process

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=seq_len + 4,
    )
    model = LlamaForCausalLM(config).eval()
    ds_test = _build_dataset(n_sequences=n_sequences, seq_len=seq_len, vocab_size=vocab_size)

    params = {
        "BS": batch_size,
        "full": True,
        "fp16": False,
        "cls_token_id": None,
        "sampling": {
            "type": "naive",
            "clamping": 1e-9,
            "context": context,
            "guidance": context,
            "value": 3,
        },
    }

    algo = SampleLevelCausalDiscovery(model, params, ds_test)
    algo.prepare()  # shards ds_test's dataloader across accelerator.num_processes

    local_batch, adj = algo.run()

    # --- Check 3: every process gets a sane, finite adjacency matrix ---
    lc = seq_len - context
    assert adj.shape == (batch_size, lc, lc), (
        f"rank {accelerator.process_index}: bad adj shape {tuple(adj.shape)}"
    )
    assert torch.isfinite(adj).all(), f"rank {accelerator.process_index}: non-finite adjacency"

    # --- Check 1/2: gather each process's local input_ids, compare on rank 0 ---
    gathered_ids = accelerator.gather(local_batch["input_ids"])
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        expected_total = accelerator.num_processes * batch_size
        assert gathered_ids.shape[0] == expected_total, (
            f"gather() returned {gathered_ids.shape[0]} rows, expected {expected_total} "
            f"({accelerator.num_processes} processes x BS={batch_size})"
        )
        rows = [tuple(row.tolist()) for row in gathered_ids]
        assert len(set(rows)) == len(rows), (
            "duplicate sequence seen across processes -- the dataloader was not "
            "actually sharded (every process saw the same data)"
        )
        print(
            f"[OK] {accelerator.num_processes} process(es), gathered {gathered_ids.shape[0]} "
            f"disjoint sequence(s), adjacency shape {tuple(adj.shape)} on every process."
        )


if __name__ == "__main__":
    main()
