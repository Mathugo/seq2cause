"""Trains a SEPARATE tiny LlamaForCausalLM (real, imperfect, gradient-trained
model -- not the SCM's own exact-conditional "oracle") on each of several
distinct `NonlinearSCM` configurations, then re-runs the same per-sequence
vs. pooled-across-sequences comparison of the 4 unsupervised
`AdaptiveThreshold` base methods (`otsu`, `mad`, `percentile`, `gmm`) as
`scripts/threshold_benchmark.py`, but against the TRAINED model's own CMI
estimates instead of the oracle's.

This directly checks whether the oracle-based benchmark's conclusion
("no unsupervised method is universally best; `percentile` had the best
overall track record") survives once real model-approximation error
(imperfect training, reported via each model's own oracle score
epsilon_hat = (loss - H(P)) / (H_max - H(P)), Eq. 22) is layered on top,
instead of the SCM's exact conditional distribution.

Sweeps ONE factor at a time around a baseline configuration (vocab_size=300,
memory=3, length=60, sparsity=0.7, decay_rate=0.5), covering vocab_size from
100 to 1000, plus a second seed at the baseline -- a full factorial grid
across vocab_size x memory x length x sparsity x decay_rate x seed would
require training hundreds of separate models, which is not practical on a
laptop CPU; this is a deliberate, documented trade-off (see module docstring
of `threshold_benchmark.py` for the analogous oracle-only full-factorial
sweep, which IS cheap enough to run exhaustively since it needs no training).

Runs roughly 15-40s of training per configuration (13 configurations total,
a few minutes overall) on a laptop CPU with the defaults below.

Usage:
    python scripts/threshold_benchmark_trained.py
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from train_and_test_lagged_effects import TrainedModelAdapter, train_tiny_llama  # noqa: E402

from seq2cause.diagnostics import (  # noqa: E402
    compute_cmi_matrix,
    estimate_oracle_score,
    ground_truth_adjacency,
)
from seq2cause.scm import create_scm  # noqa: E402
from seq2cause.threshold import AdaptiveThreshold  # noqa: E402

METHODS = ["otsu", "mad", "percentile", "gmm"]
N_SEQ = 6  # first 4 pooled-fit the threshold, last 2 held out for evaluation

BASELINE = {"vocab_size": 300, "memory": 3, "length": 60, "sparsity": 0.7, "decay_rate": 0.5}

# One-factor-at-a-time grid around BASELINE (see module docstring for why not
# a full factorial). Each entry overrides exactly one field of BASELINE.
CONFIGS = [
    {"name": "baseline", "seed": 0, **BASELINE},
    {"name": "baseline (seed=1)", "seed": 1, **BASELINE},
    {"name": "vocab=100", "seed": 0, **{**BASELINE, "vocab_size": 100}},
    {"name": "vocab=600", "seed": 0, **{**BASELINE, "vocab_size": 600}},
    {"name": "vocab=1000", "seed": 0, **{**BASELINE, "vocab_size": 1000}},
    {"name": "memory=1", "seed": 0, **{**BASELINE, "memory": 1}},
    {"name": "memory=6", "seed": 0, **{**BASELINE, "memory": 6}},
    {"name": "length=20", "seed": 0, **{**BASELINE, "length": 20}},
    {"name": "length=120", "seed": 0, **{**BASELINE, "length": 120}},
    {"name": "sparsity=0.5", "seed": 0, **{**BASELINE, "sparsity": 0.5}},
    {"name": "sparsity=0.9", "seed": 0, **{**BASELINE, "sparsity": 0.9}},
    {"name": "decay_rate=0.2", "seed": 0, **{**BASELINE, "decay_rate": 0.2}},
    {"name": "decay_rate=1.0", "seed": 0, **{**BASELINE, "decay_rate": 1.0}},
]


def _f1(pred: torch.Tensor, gt: torch.Tensor) -> float:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _run_config(cfg: dict) -> dict:
    vocab_size, memory, length = cfg["vocab_size"], cfg["memory"], cfg["length"]
    sparsity, decay_rate, seed = cfg["sparsity"], cfg["decay_rate"], cfg["seed"]
    context_len = memory + 2

    torch.manual_seed(seed)
    scm, _ = create_scm(
        vocab_size=vocab_size, memory=memory, length=length, seed=seed,
        sparsity=sparsity, decay_rate=decay_rate, batch_size=1,
    )

    train_args = argparse.Namespace(
        vocab_size=vocab_size, hidden_size=48, num_layers=2, num_heads=2,
        seq_len=length, steps=500, batch_size=48, lr=6e-3,
    )
    t0 = time.perf_counter()
    model = train_tiny_llama(scm, train_args)
    train_time = time.perf_counter() - t0
    adapter = TrainedModelAdapter(model, vocab_size=vocab_size)

    oracle = estimate_oracle_score(adapter, scm, n_sequences=100, length=length)

    # Held-out sequences from the SAME scm instance (same causal structure),
    # via an independently-seeded generator -- NOT a freshly re-created scm,
    # which would have entirely different causal structure.
    eval_generator = torch.Generator(device=scm.device).manual_seed(seed + 1000)
    eval_sequences = scm.sample_sequence(length=length, batch_size=N_SEQ, generator=eval_generator)
    cmis, gts = [], []
    for i in range(N_SEQ):
        sequence = eval_sequences[i]
        cmi = compute_cmi_matrix(
            adapter, sequence, context_len=context_len, n_particles=48, strategy="atomic"
        )
        adjacency = ground_truth_adjacency(scm, sequence, threshold=0.05, n_counterfactuals=10)
        cmis.append(cmi)
        gts.append(adjacency[context_len:, context_len:])

    lc = cmis[0].shape[-1]
    lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
    valid = lag_matrix > 0
    max_lag = int(lag_matrix.max())

    method_f1 = {}
    for method in METHODS:
        threshold = AdaptiveThreshold(method=method)
        per_seq_f1s = [_f1(threshold.causal_graph(cmi), gt) for cmi, gt in zip(cmis, gts)]

        pooled_scores = torch.cat([cmis[i][valid] for i in range(4)])
        pooled_lags = torch.cat([lag_matrix[valid] for _ in range(4)])
        tau_by_lag = threshold.tau_by_lag(pooled_scores, pooled_lags, max_lag=max_lag)
        pooled_f1s = [
            _f1(threshold.apply_tau_by_lag(cmis[i], tau_by_lag), gts[i]) for i in (4, 5)
        ]
        method_f1[method] = {
            "per_seq": sum(per_seq_f1s) / len(per_seq_f1s),
            "pooled": sum(pooled_f1s) / len(pooled_f1s),
        }

    return {
        "name": cfg["name"], "vocab": vocab_size, "memory": memory, "length": length,
        "sparsity": sparsity, "decay_rate": decay_rate, "seed": seed,
        "epsilon_hat": oracle["epsilon_hat"], "loss": oracle["loss"], "h_p": oracle["h_p"],
        "train_time": train_time, "method_f1": method_f1,
    }


def main() -> None:
    results = []
    t_start = time.perf_counter()
    for cfg in CONFIGS:
        print(f"=== {cfg['name']}: vocab={cfg['vocab_size']} memory={cfg['memory']} "
              f"length={cfg['length']} sparsity={cfg['sparsity']} decay_rate={cfg['decay_rate']} "
              f"seed={cfg['seed']} ===")
        result = _run_config(cfg)
        results.append(result)
        print(
            f"  trained in {result['train_time']:.1f}s, epsilon_hat={result['epsilon_hat']:.3f} "
            f"(loss={result['loss']:.3f}, H(P)={result['h_p']:.3f})"
        )
        for method in METHODS:
            f1s = result["method_f1"][method]
            print(f"  {method:12s} per-seq F1={f1s['per_seq']:.3f}  pooled F1={f1s['pooled']:.3f}")
        print()

    elapsed = time.perf_counter() - t_start
    print(f"\nAll {len(results)} configs done in {elapsed:.1f}s\n")

    print("=== OVERALL (mean +/- std across all trained-model configs) ===")
    for method in METHODS:
        per = [r["method_f1"][method]["per_seq"] for r in results]
        pooled = [r["method_f1"][method]["pooled"] for r in results]
        print(
            f"{method:12s} per-seq: {st.mean(per):.3f} +/- {st.pstdev(per):.3f}   "
            f"pooled: {st.mean(pooled):.3f} +/- {st.pstdev(pooled):.3f}"
        )

    wins = dict.fromkeys(METHODS, 0)
    for r in results:
        scores = {m: r["method_f1"][m]["pooled"] for m in METHODS}
        best = max(scores.values())
        for m, v in scores.items():
            if v >= best - 1e-9:
                wins[m] += 1
    print(f"\n=== win-or-tie rate (best pooled F1) across {len(results)} trained-model configs ===")
    for method in METHODS:
        print(f"{method:12s} best-or-tied in {wins[method]}/{len(results)} configs")

    print("\n=== per-config detail (pooled F1, oracle score) ===")
    header = "config".ljust(20) + "eps_hat".rjust(9) + "".join(m.rjust(12) for m in METHODS)
    print(header)
    for r in results:
        line = r["name"].ljust(20) + f"{r['epsilon_hat']:9.3f}"
        for m in METHODS:
            line += f"{r['method_f1'][m]['pooled']:12.3f}"
        print(line)


if __name__ == "__main__":
    main()
