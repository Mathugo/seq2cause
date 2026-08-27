"""Evaluates the "sparse" (bounded-memory) CMI construction against the
unbounded "full" construction, on a synthetic, memory-bounded, DECAYED
`NonlinearSCM` (an oracle model, epsilon=0 -- isolates the CONSTRUCTION from
model-approximation error, same rationale as `snr_diagnostic.py`).

`compute_cmi_matrix_sparse` (see `seq2cause.diagnostics`) slides a short
local window across the sequence instead of running the "full" staircase
once on the whole sequence -- O(L) total rows, each of length O(memory)
instead of O(L). Whenever `memory` truly bounds the generator's memory
(exactly the assumption a `NonlinearSCM(memory=...)` DGP satisfies), this
should recover the SAME causal graph (same pooled F1, within noise) as the
unbounded computation, for a fraction of the wall-clock time on long
sequences.

Usage:
    python scripts/evaluate_sparse_vs_full.py
    python scripts/evaluate_sparse_vs_full.py --seq-len 200 --memory 3 --context 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from seq2cause.diagnostics import (  # noqa: E402
    compute_cmi_matrix,
    compute_cmi_matrix_sparse,
    ground_truth_adjacency,
)
from seq2cause.scm import create_scm  # noqa: E402
from seq2cause.threshold import select_threshold_by_validation  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-size", type=int, default=20)
    p.add_argument("--memory", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=120)
    p.add_argument(
        "--context", type=int, default=4,
        help="Fixed prefix length; must be > --memory (see compute_cmi_matrix_sparse).",
    )
    p.add_argument("--sparsity", type=float, default=0.6)
    p.add_argument("--decay-rate", type=float, default=0.4)
    p.add_argument("--n-eval-sequences", type=int, default=10)
    p.add_argument("--n-particles", type=int, default=64)
    p.add_argument("--n-counterfactuals", type=int, default=12)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.context <= args.memory:
        raise ValueError(f"--context ({args.context}) must be > --memory ({args.memory})")

    torch.manual_seed(args.seed)
    scm, sequences = create_scm(
        vocab_size=args.vocab_size, memory=args.memory, length=args.seq_len, seed=args.seed,
        sparsity=args.sparsity, decay_rate=args.decay_rate, batch_size=args.n_eval_sequences,
    )
    print(
        f"=== Sparse vs. Full, decayed memory-bounded DGP (vocab={args.vocab_size}, "
        f"memory={args.memory}, seq_len={args.seq_len}, decay_rate={args.decay_rate}, "
        f"n_eval_sequences={args.n_eval_sequences}, n_particles={args.n_particles}) ==="
    )

    full_scores, full_labels, sparse_scores, sparse_labels = [], [], [], []
    full_time, sparse_time = 0.0, 0.0
    n_true_edges = 0

    for i in range(args.n_eval_sequences):
        sequence = sequences[i]
        adjacency = ground_truth_adjacency(
            scm, sequence, threshold=args.delta, n_counterfactuals=args.n_counterfactuals,
        )
        adj_suffix = adjacency[args.context :, args.context :]
        n_true_edges += int(adj_suffix.sum())

        t0 = time.perf_counter()
        full_cmi = compute_cmi_matrix(
            scm, sequence, context_len=args.context, n_particles=args.n_particles, strategy="full",
        )
        full_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        sparse_cmi = compute_cmi_matrix_sparse(
            scm, sequence, context_len=args.context, memory=args.memory,
            n_particles=args.n_particles,
        )
        sparse_time += time.perf_counter() - t0

        full_scores.append(full_cmi.flatten())
        full_labels.append(adj_suffix.flatten())
        sparse_scores.append(sparse_cmi.flatten())
        sparse_labels.append(adj_suffix.flatten())

    full_scores_t, full_labels_t = torch.cat(full_scores), torch.cat(full_labels)
    sparse_scores_t, sparse_labels_t = torch.cat(sparse_scores), torch.cat(sparse_labels)

    full_result = select_threshold_by_validation(full_scores_t, full_labels_t, delta=args.delta, emit_warnings=False)
    sparse_result = select_threshold_by_validation(sparse_scores_t, sparse_labels_t, delta=args.delta, emit_warnings=False)

    print(f"\nTotal ground-truth edges across {args.n_eval_sequences} sequences: {n_true_edges}")
    print(f"\n{'':<10}{'F1':>8}{'Precision':>12}{'Recall':>10}{'tau':>12}{'time (s)':>12}")
    print(
        f"{'full':<10}{full_result.f1:>8.3f}{full_result.precision:>12.3f}"
        f"{full_result.recall:>10.3f}{full_result.tau:>12.4g}{full_time:>12.3f}"
    )
    print(
        f"{'sparse':<10}{sparse_result.f1:>8.3f}{sparse_result.precision:>12.3f}"
        f"{sparse_result.recall:>10.3f}{sparse_result.tau:>12.4g}{sparse_time:>12.3f}"
    )
    speedup = full_time / sparse_time if sparse_time > 0 else float("inf")
    print(f"\nSpeedup (full time / sparse time): {speedup:.2f}x")
    print(f"|F1 difference|: {abs(full_result.f1 - sparse_result.f1):.3f}")


if __name__ == "__main__":
    main()
