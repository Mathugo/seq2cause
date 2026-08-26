"""SNR-by-lag diagnostic and intervention-strategy comparison for TRACE.

Usage:
    python scripts/snr_diagnostic.py [--vocab-size 1000] [--memory 6] [--length 64]
        [--context 6] [--n-sequences 8] [--n-particles 64] [--window-k 2]
        [--delta 0.05] [--seed 0]

Context
-------
Chadyuk, Zhang, and Kucukates ("Replicating TRACE: A Practitioner's Guide to
Its Threshold and Particle Budget", LotusFlare Inc., Aug 2026) report that,
under the "full" staircase intervention construction, recall collapses to
near 0 for cause-effect lags >= 2 regardless of the CMI threshold, while
lag-1 recall stays at 0.97-0.99. This script:

  1. Reproduces their per-lag SNR measurement ("Signal to noise ratio",
     `snr_stats` keyed by lag, extending the diagnostic already sketched in
     `experiment.ipynb`) on this codebase's own generator (a self-contained
     nonlinear SCM oracle, `seq2cause.scm.NonlinearSCM`, so there is zero
     model-approximation error -- any collapse observed is attributable to
     the intervention *construction*, not to an imperfectly-trained model).
  2. Compares the default "full" staircase against three alternative
     constructions ("atomic", "windowed", "independent_mediator") and an
     alternative do-intervention *noise proposal* ("in-distribution-noise",
     i.e. `unigram_sample` instead of `uniform_sample`), reporting per-lag
     recall and per-pair CMI magnitude for each so the relative contribution
     of "context loss" vs. "OOD noise" vs. "non-independent draws" can be
     read off directly.

This is a diagnostic, not a benchmark: sequence lengths/vocab sizes are kept
small by default so it runs in seconds on a laptop CPU. Increase
`--n-sequences`/`--vocab-size` for a more statistically robust reading.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Allow running as `python scripts/snr_diagnostic.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from seq2cause.diagnostics import (  # noqa: E402
    compare_intervention_strategies,
    ground_truth_adjacency,
)
from seq2cause.scm import create_scm  # noqa: E402
from seq2cause.threshold import select_threshold_by_validation  # noqa: E402

STRATEGY_ORDER = ("full", "atomic", "windowed", "independent_mediator", "in_distribution_noise")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-size", type=int, default=200)
    p.add_argument("--memory", type=int, default=4)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--n-sequences", type=int, default=8)
    p.add_argument("--n-particles", type=int, default=64)
    p.add_argument("--n-counterfactuals", type=int, default=8)
    p.add_argument("--window-k", type=int, default=2)
    p.add_argument("--delta", type=float, default=0.05, help="truth-margin KL threshold")
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "   n/a"
    return f"{x:6.3f}"


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    scm, sequences = create_scm(
        vocab_size=args.vocab_size,
        memory=args.memory,
        length=args.length,
        sparsity=args.sparsity,
        seed=args.seed,
        batch_size=args.n_sequences,
    )

    # Aggregate CMI scores/labels across sequences to also demonstrate the
    # validation-sweep threshold-selection workflow from Part 1 on the
    # "full" strategy's own CMI matrix.
    all_scores_by_strategy: dict[str, list[float]] = {name: [] for name in STRATEGY_ORDER}
    all_labels_by_strategy: dict[str, list[bool]] = {name: [] for name in STRATEGY_ORDER}
    snr_by_strategy_lag: dict[str, dict[int, dict[str, list[float]]]] = {
        name: {lag: {"true": [], "false": []} for lag in range(1, args.memory + 1)}
        for name in STRATEGY_ORDER
    }

    for i in range(args.n_sequences):
        seq = sequences[i]
        adjacency = ground_truth_adjacency(
            scm, seq, threshold=args.delta, n_counterfactuals=args.n_counterfactuals
        )
        unigram_freqs = torch.bincount(seq, minlength=args.vocab_size).float() + 1.0

        results = compare_intervention_strategies(
            scm,
            seq,
            context_len=args.context,
            adjacency=adjacency,
            n_particles=args.n_particles,
            window_k=args.window_k,
            tau=None,  # recall computed below via a swept-per-strategy tau
            max_lag=args.memory,
            unigram_freqs=unigram_freqs,
        )

        adjacency_suffix = adjacency[args.context :, args.context :]
        for name, res in results.items():
            scores = res.cmi_matrix.flatten().tolist()
            labels = adjacency_suffix.flatten().tolist()
            all_scores_by_strategy[name].extend(scores)
            all_labels_by_strategy[name].extend(labels)
            for lag, stats in res.snr.items():
                snr_by_strategy_lag[name][lag]["true"].append(stats["median_true"])
                snr_by_strategy_lag[name][lag]["false"].append(stats["median_false"])

    print(f"=== SNR-by-lag diagnostic (vocab={args.vocab_size}, memory={args.memory}, "
          f"L={args.length}, context={args.context}, N={args.n_particles}, "
          f"delta={args.delta}) ===\n")

    for name in STRATEGY_ORDER:
        scores = all_scores_by_strategy[name]
        labels = all_labels_by_strategy[name]
        if not scores:
            continue
        thr = select_threshold_by_validation(torch.tensor(scores), torch.tensor(labels), delta=args.delta)

        print(f"--- strategy = {name} ---")
        print(thr.summary())
        print(f"{'lag':>4}  {'median CMI (true)':>18}  {'median CMI (non-edge)':>22}  {'recall@tau':>10}")
        for lag in range(1, args.memory + 1):
            true_vals = [v for v in snr_by_strategy_lag[name][lag]["true"] if v == v]
            false_vals = [v for v in snr_by_strategy_lag[name][lag]["false"] if v == v]
            med_true = statistics.median(true_vals) if true_vals else float("nan")
            med_false = statistics.median(false_vals) if false_vals else float("nan")
            print(f"{lag:>4}  {_fmt(med_true):>18}  {_fmt(med_false):>22}  {_fmt(thr.recall):>10}")
        print()

    print(
        "Interpretation: if 'full' shows median CMI (true) collapsing toward "
        "median CMI (non-edge) for lag >= 2 while 'atomic'/'windowed' (larger "
        "window_k) recover a true/false gap, that supports the context-loss "
        "explanation. If 'in_distribution_noise' alone recovers a gap that "
        "'full' does not, OOD noise is also contributing. If "
        "'independent_mediator' differs materially from 'full', non-independent "
        "cause/mediator draws are contributing too."
    )


if __name__ == "__main__":
    main()
