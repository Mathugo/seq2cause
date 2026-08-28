"""Benchmarks the four unsupervised base methods available to
`AdaptiveThreshold` (`otsu`, `mad`, `percentile`, `gmm`) across a grid of
vocab_size / memory / sequence length, against `NonlinearSCM`'s known
ground truth -- both fitting a fresh threshold per sequence, and fitting
ONE threshold on scores pooled across several sequences (see README
"Threshold Selection").

This exists because a single worked example is NOT representative: one
narrow (vocab=12, memory=2, length=40) example made `otsu` look uniformly
bad (it isolates a single dominant outlier instead of the true/false-edge
break) and `mad`/`gmm` look uniformly good. Sweeping vocab_size/memory/
length reveals the opposite trend in other regimes -- e.g. at
(vocab=200, memory=6, length=120), true edges are extremely sparse
(~0.05% of pairs) and `mad`/`gmm`'s implicit "a few percent of pairs are
edges" assumption floods the result with false positives (100+ predicted
edges for 3 true ones), while `otsu`'s single-outlier-isolation behavior
is actually well matched to that sparsity.

Conclusion: there is no universally best *unsupervised* method -- it
depends on how sparse the true causal graph is relative to what each
method implicitly assumes, which you cannot know without labels. Use
`select_threshold_by_validation` whenever ANY labels are available, even
a small held-out set; treat this sweep as a guide for the label-free
fallback case only.

Run: `python scripts/threshold_benchmark.py`
"""

from __future__ import annotations

import itertools
import statistics as st
import time

import torch

from seq2cause.diagnostics import compute_cmi_matrix, ground_truth_adjacency
from seq2cause.scm import create_scm
from seq2cause.threshold import AdaptiveThreshold

METHODS = ["otsu", "mad", "percentile", "gmm"]
VOCABS = [10, 50, 200]
MEMORIES = [1, 3, 6]
LENGTHS = [20, 60, 120]
SEEDS = [0, 1, 2]
N_SEQ = 6  # first 4 used for pooled-fit calibration, last 2 held out for eval


def _f1(pred: torch.Tensor, gt: torch.Tensor) -> float:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _run_config(vocab_size: int, memory: int, length: int, seed: int) -> list[dict]:
    context_len = memory + 2
    torch.manual_seed(seed)
    scm, sequences = create_scm(
        vocab_size=vocab_size, memory=memory, length=length, seed=seed, batch_size=N_SEQ
    )

    cmis, gts = [], []
    for i in range(N_SEQ):
        sequence = sequences[i]
        cmi = compute_cmi_matrix(
            scm, sequence, context_len=context_len, n_particles=64, strategy="atomic"
        )
        adjacency = ground_truth_adjacency(scm, sequence, threshold=0.05, n_counterfactuals=10)
        cmis.append(cmi)
        gts.append(adjacency[context_len:, context_len:])

    lc = cmis[0].shape[-1]
    lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
    valid = lag_matrix > 0
    max_lag = int(lag_matrix.max())

    rows = []
    for method in METHODS:
        threshold = AdaptiveThreshold(method=method)

        per_seq_f1s = [
            _f1(threshold.causal_graph(cmi), gt) for cmi, gt in zip(cmis, gts)
        ]

        pooled_scores = torch.cat([cmis[i][valid] for i in range(4)])
        pooled_lags = torch.cat([lag_matrix[valid] for _ in range(4)])
        tau_by_lag = threshold.tau_by_lag(pooled_scores, pooled_lags, max_lag=max_lag)
        pooled_f1s = [
            _f1(threshold.apply_tau_by_lag(cmis[i], tau_by_lag), gts[i]) for i in (4, 5)
        ]

        rows.append(
            {
                "vocab": vocab_size,
                "memory": memory,
                "length": length,
                "seed": seed,
                "method": method,
                "per_seq_f1_mean": sum(per_seq_f1s) / len(per_seq_f1s),
                "pooled_f1_mean": sum(pooled_f1s) / len(pooled_f1s),
            }
        )
    return rows


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    for vocab_size, memory, length, seed in itertools.product(VOCABS, MEMORIES, LENGTHS, SEEDS):
        if length - (memory + 2) < 12:
            continue
        rows.extend(_run_config(vocab_size, memory, length, seed))
    elapsed = time.perf_counter() - t0
    print(f"Grid done in {elapsed:.1f}s, {len(rows)} rows ({len(SEEDS)} seeds/config)\n")

    print("=== OVERALL (mean +/- std, pooled across all configs and seeds) ===")
    for m in METHODS:
        per = [r["per_seq_f1_mean"] for r in rows if r["method"] == m]
        pooled = [r["pooled_f1_mean"] for r in rows if r["method"] == m]
        print(
            f"{m:12s} per-seq: {st.mean(per):.3f} +/- {st.pstdev(per):.3f}   "
            f"pooled: {st.mean(pooled):.3f} +/- {st.pstdev(pooled):.3f}"
        )

    configs = sorted({(r["vocab"], r["memory"], r["length"]) for r in rows})
    wins = dict.fromkeys(METHODS, 0)
    for cfg in configs:
        scores = {
            m: st.mean(
                r["pooled_f1_mean"]
                for r in rows
                if r["method"] == m and (r["vocab"], r["memory"], r["length"]) == cfg
            )
            for m in METHODS
        }
        best = max(scores.values())
        for m, v in scores.items():
            if v >= best - 1e-9:
                wins[m] += 1
    print(f"\n=== win-or-tie rate (best pooled F1) across {len(configs)} configs ===")
    for m in METHODS:
        print(f"{m:12s} best-or-tied in {wins[m]}/{len(configs)} configs")

    for factor in ("vocab", "memory", "length"):
        print(f"\n=== BY {factor.upper()} (pooled-fit F1 mean over seeds) ===")
        values = sorted({r[factor] for r in rows})
        print("method".ljust(12) + "".join(str(v).rjust(10) for v in values))
        for m in METHODS:
            line = m.ljust(12)
            for v in values:
                f1s = [r["pooled_f1_mean"] for r in rows if r["method"] == m and r[factor] == v]
                line += f"{st.mean(f1s):10.3f}" if f1s else " " * 10
            print(line)


if __name__ == "__main__":
    main()
