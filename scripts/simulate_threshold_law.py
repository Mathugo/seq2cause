"""Multi-DGP simulation to fit and cross-validate a single, coherent
per-lag-AND-vocab-size threshold law, instead of a bespoke curve per
data-generating process.

Motivation: `train_and_test_lagged_effects.py`'s `fit_exponential_lag_threshold`/
`fit_power_law_lag_threshold` each calibrate tau(lag) for ONE trained model on
ONE synthetic DGP (one vocab size, one lag-decay rate). This script instead
trains a *separate* tiny Llama on `N_DGPS` distinct DGPs (varying vocabulary
size and lag-decay rate -- the two axes discussed with Chadyuk et al., 2026:
their own note found tau* decays sub-linearly with |X|, and we separately
found the per-lag decay is construction-dependent, sometimes sub-linear too),
pools every (CMI score, edge label, lag, vocab_size) tuple across ALL of
them, and fits ONE global law:

    tau(lag, |X|) = signal_scale * |X|^-vocab_exponent_signal * lag^-lag_exponent
                    + floor_scale * |X|^-vocab_exponent_floor

(`seq2cause.threshold.fit_joint_lag_vocab_threshold` -- see its docstring for
why the two vocab-size exponents are kept separate). Crucially, this script
also does LEAVE-ONE-DGP-OUT cross-validation: fit the law on 9 DGPs, evaluate
pooled F1 on the 10th (unseen) DGP, repeat for every DGP. That is the honest
measure of "does one formula transfer to a new setup", as opposed to the
in-sample fit (which only shows the law CAN approximate the pooled data it
was fit on).

Runs ~10 independent from-scratch training runs at a reduced budget (a few
minutes each on CPU); expect a total runtime of tens of minutes.

Usage:
    python scripts/simulate_threshold_law.py [--results-json PATH]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from train_and_test_lagged_effects import TrainedModelAdapter, train_tiny_llama  # noqa: E402

from seq2cause.diagnostics import (  # noqa: E402
    compare_intervention_strategies,
    ground_truth_adjacency,
)
from seq2cause.scm import NonlinearSCM  # noqa: E402
from seq2cause.threshold import (  # noqa: E402
    fit_exponential_lag_threshold,
    fit_joint_lag_vocab_threshold,
    fit_power_law_lag_threshold,
    make_log_grid,
    select_thresholds_by_group,
)

STRATEGIES_OF_INTEREST = ("full", "atomic")

# 10 DGPs: sweep vocabulary size (|X|) and lag-decay rate together, the two
# axes discussed with Chadyuk et al. (2026) -- their own |X|-scaling note,
# and our earlier finding that the per-lag decay itself is
# construction-dependent (sometimes sub-linear).
DGP_GRID = [
    {"vocab_size": 50, "decay_rate": 0.0},
    {"vocab_size": 50, "decay_rate": 0.3},
    {"vocab_size": 50, "decay_rate": 0.6},
    {"vocab_size": 100, "decay_rate": 0.0},
    {"vocab_size": 100, "decay_rate": 0.3},
    {"vocab_size": 100, "decay_rate": 0.6},
    {"vocab_size": 200, "decay_rate": 0.0},
    {"vocab_size": 200, "decay_rate": 0.3},
    {"vocab_size": 200, "decay_rate": 0.6},
    {"vocab_size": 400, "decay_rate": 0.3},
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--memory", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--hidden-size", type=int, default=48)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-eval-sequences", type=int, default=20)
    p.add_argument("--n-particles", type=int, default=24)
    p.add_argument("--n-counterfactuals", type=int, default=6)
    p.add_argument("--window-k", type=int, default=2)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--results-json", type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "results_threshold_law_sweep.json"),
    )
    return p.parse_args(argv)


def run_one_dgp(dgp_id: int, vocab_size: int, decay_rate: float, args: argparse.Namespace) -> dict:
    """Trains one tiny Llama on one DGP and returns pooled (score, label,
    lag) tuples per strategy of interest for this DGP alone."""
    seed = args.seed + 1000 * dgp_id
    torch.manual_seed(seed)
    print(f"\n--- DGP {dgp_id + 1}/{len(DGP_GRID)}: vocab_size={vocab_size}, "
          f"decay_rate={decay_rate}, seed={seed} ---")

    scm = NonlinearSCM(
        vocab_size=vocab_size, memory=args.memory, sparsity=args.sparsity,
        decay_rate=decay_rate, seed=seed,
    )
    # Reuse the exact same training routine as the single-DGP script so
    # results are comparable and nothing is silently reimplemented.
    dgp_args = argparse.Namespace(
        hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, seq_len=args.seq_len,
    )
    model = train_tiny_llama(scm, dgp_args)
    adapter = TrainedModelAdapter(model, vocab_size)

    gt_gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    eval_sequences = scm.sample_sequence(length=args.seq_len, batch_size=args.n_eval_sequences, generator=gt_gen)
    adjacencies = [
        ground_truth_adjacency(
            scm, eval_sequences[i], threshold=args.delta, n_counterfactuals=args.n_counterfactuals,
            generator=gt_gen,
        )
        for i in range(args.n_eval_sequences)
    ]
    n_true_edges = sum(int(adj.sum()) for adj in adjacencies)
    print(f"    true edges: {n_true_edges} across {args.n_eval_sequences} sequences")

    pooled: dict[str, dict[str, list]] = {
        name: {"scores": [], "labels": [], "lags": []} for name in STRATEGIES_OF_INTEREST
    }
    for i in range(args.n_eval_sequences):
        seq = eval_sequences[i]
        adjacency = adjacencies[i]
        unigram_freqs = torch.bincount(seq, minlength=vocab_size).float() + 1.0
        results = compare_intervention_strategies(
            adapter, seq, context_len=args.context, adjacency=adjacency,
            n_particles=args.n_particles, window_k=args.window_k, tau=None,
            max_lag=args.memory, unigram_freqs=unigram_freqs,
        )
        adjacency_suffix = adjacency[args.context:, args.context:]
        lc = adjacency_suffix.shape[-1]
        lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
        for name in STRATEGIES_OF_INTEREST:
            cmi_matrix = results[name].cmi_matrix
            valid = (lag_matrix > 0) & (lag_matrix <= args.memory)
            pooled[name]["scores"].extend(cmi_matrix[valid].flatten().tolist())
            pooled[name]["labels"].extend(adjacency_suffix[valid].flatten().tolist())
            pooled[name]["lags"].extend(lag_matrix[valid].flatten().tolist())

    # For each strategy, also anchor directly to the CMI scale itself (per
    # Chadyuk et al.'s follow-up: tau* tracks the median TRUE-EDGE CMI far
    # more tightly than it tracks any abstract covariate like |X| alone).
    # Compute, per lag: the median true-edge CMI, and the genuinely free
    # (unconstrained) per-lag F1-optimal tau* via `select_thresholds_by_group`
    # -- the same quantity Chadyuk et al.'s own protocol selects.
    for name in STRATEGIES_OF_INTEREST:
        scores_t = torch.tensor(pooled[name]["scores"])
        labels_t = torch.tensor(pooled[name]["labels"])
        lags_t = torch.tensor(pooled[name]["lags"])

        median_true_by_lag: dict[int, float] = {}
        for lag in range(1, args.memory + 1):
            true_vals = scores_t[(lags_t == lag) & labels_t]
            median_true_by_lag[lag] = float(true_vals.median()) if true_vals.numel() > 0 else float("nan")
        dgp_wide_true = scores_t[labels_t]
        dgp_wide_median_true = float(dgp_wide_true.median()) if dgp_wide_true.numel() > 0 else 1e-6

        by_group = select_thresholds_by_group(
            scores_t, labels_t, lags_t, min_group_size=4, fallback="global", emit_warnings=False,
        )
        tau_star_by_lag = {lag: res.tau for lag, res in by_group.items()}

        # Per-point lookup of "this point's own (lag, DGP) median true-edge
        # CMI", falling back to the DGP-wide median true CMI when that exact
        # lag had too few/no true edges to get a reliable median (or, in
        # principle, an out-of-range lag).
        median_true_lookup = [
            m if (m := median_true_by_lag.get(lag)) is not None and m == m else dgp_wide_median_true
            for lag in pooled[name]["lags"]
        ]
        pooled[name]["median_true_lookup"] = median_true_lookup
        pooled[name]["median_true_by_lag"] = median_true_by_lag
        pooled[name]["tau_star_by_lag"] = tau_star_by_lag

    return {"vocab_size": vocab_size, "decay_rate": decay_rate, "n_true_edges": n_true_edges, "pooled": pooled}


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"=== Simulating {len(DGP_GRID)} DGPs to fit a joint lag x vocab-size threshold law ===")

    dgp_results = [
        run_one_dgp(i, cfg["vocab_size"], cfg["decay_rate"], args) for i, cfg in enumerate(DGP_GRID)
    ]

    report: dict = {"dgp_grid": DGP_GRID, "strategies": {}}

    for name in STRATEGIES_OF_INTEREST:
        print(f"\n\n########## strategy=\"{name}\" ##########")
        all_scores, all_labels, all_lags, all_vocab = [], [], [], []
        per_dgp_data = []
        for res in dgp_results:
            p = res["pooled"][name]
            per_dgp_data.append(p)
            all_scores.extend(p["scores"])
            all_labels.extend(p["labels"])
            all_lags.extend(p["lags"])
            all_vocab.extend([res["vocab_size"]] * len(p["scores"]))

        all_scores_t = torch.tensor(all_scores)
        all_labels_t = torch.tensor(all_labels)
        all_lags_t = torch.tensor(all_lags)
        all_vocab_t = torch.tensor(all_vocab, dtype=torch.float32)

        # (a) Per-DGP bespoke fits: the "if you got to recalibrate for every
        # new setup" upper bound, for reference.
        per_dgp_best_f1 = []
        for i, res in enumerate(dgp_results):
            p = per_dgp_data[i]
            if len(set(p["labels"])) < 2:
                continue
            scores_t, labels_t, lags_t = torch.tensor(p["scores"]), torch.tensor(p["labels"]), torch.tensor(p["lags"])
            exp_fit = fit_exponential_lag_threshold(scores_t, labels_t, lags_t, max_lag=args.memory)
            pow_fit = fit_power_law_lag_threshold(scores_t, labels_t, lags_t, max_lag=args.memory)
            best_f1 = max(exp_fit.f1, pow_fit.f1)
            per_dgp_best_f1.append(best_f1)
            print(f"  DGP {i+1} (|X|={res['vocab_size']}, decay={res['decay_rate']}): "
                  f"bespoke best F1={best_f1:.3f} (exp={exp_fit.f1:.3f}, pow={pow_fit.f1:.3f})")

        # (b) In-sample joint fit: the law's own best possible fit, pooled
        # across every DGP at once (optimistic upper bound for the law).
        joint_fit = fit_joint_lag_vocab_threshold(all_scores_t, all_labels_t, all_lags_t, all_vocab_t)
        print(f"\n  In-sample joint law fit: {joint_fit.summary()}")

        # (c) Leave-one-DGP-out cross-validation: the honest transfer test.
        cv_f1s = []
        reduced_grids = {
            "signal_scale_grid": None,  # keep default (log grid) -- cheap axis
            "vocab_exponent_signal_grid": [0.0, 0.3, 0.5, 0.7, 1.0],
            "lag_exponent_grid": [0.0, 0.5, 1.0, 1.5],
            "floor_scale_grid": None,
            "vocab_exponent_floor_grid": [0.5, 1.0, 1.85, 2.5],
        }
        for held_out_idx in range(len(dgp_results)):
            train_scores, train_labels, train_lags, train_vocab = [], [], [], []
            for i, p in enumerate(per_dgp_data):
                if i == held_out_idx:
                    continue
                train_scores.extend(p["scores"])
                train_labels.extend(p["labels"])
                train_lags.extend(p["lags"])
                train_vocab.extend([dgp_results[i]["vocab_size"]] * len(p["scores"]))
            if len(set(train_labels)) < 2:
                continue
            cv_fit = fit_joint_lag_vocab_threshold(
                torch.tensor(train_scores), torch.tensor(train_labels),
                torch.tensor(train_lags), torch.tensor(train_vocab, dtype=torch.float32),
                **reduced_grids,
            )
            held_out = per_dgp_data[held_out_idx]
            if len(set(held_out["labels"])) < 2:
                continue
            held_scores_t = torch.tensor(held_out["scores"])
            held_labels_t = torch.tensor(held_out["labels"])
            held_lags_t = torch.tensor(held_out["lags"])
            held_vocab_t = torch.full_like(held_lags_t, dgp_results[held_out_idx]["vocab_size"], dtype=torch.float32)
            tau_per_point = cv_fit.tau(held_lags_t.float(), held_vocab_t)
            preds = held_scores_t >= tau_per_point
            tp = int((preds & held_labels_t).sum().item())
            fp = int((preds & ~held_labels_t).sum().item())
            fn = int((~preds & held_labels_t).sum().item())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            cv_f1s.append(f1)
            print(f"  held-out DGP {held_out_idx+1} (|X|={dgp_results[held_out_idx]['vocab_size']}, "
                  f"decay={dgp_results[held_out_idx]['decay_rate']}): transfer F1={f1:.3f}")

        # (d) Direct test of Chadyuk et al.'s specific claim: does the
        # F1-optimal tau* stay a roughly-CONSTANT multiple of the median
        # true-edge CMI, across every DGP and lag? (their reported band:
        # ratio in ~[0.15, 0.5]). Uses the genuinely free per-(DGP,lag)
        # tau* from `select_thresholds_by_group`, not the constrained
        # exponential/power-law/joint curves above.
        ratios = []
        for res in dgp_results:
            p = res["pooled"][name]
            for lag, tau_star in p["tau_star_by_lag"].items():
                m = p["median_true_by_lag"].get(lag)
                if m is None or m != m or m <= 0:  # NaN or degenerate
                    continue
                ratios.append(tau_star / m)
        if ratios:
            mean_ratio = statistics.mean(ratios)
            std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
            in_band = sum(1 for r in ratios if 0.15 <= r <= 0.5)
            print(f"\n  tau* / median(true-edge CMI) ratio across all (DGP, lag): "
                  f"mean={mean_ratio:.3f} +/- {std_ratio:.3f}, "
                  f"min={min(ratios):.3f}, max={max(ratios):.3f}, "
                  f"n={len(ratios)}, {in_band}/{len(ratios)} fall in Chadyuk et al.'s [0.15, 0.5] band")

        # (e) A 1-parameter, CMI-ANCHORED law: tau(lag, DGP) = k * median
        # true-edge CMI at that (lag, DGP) -- directly instantiates "the
        # threshold is primarily a function of the CMI scale itself", rather
        # than of abstract covariates like |X|. Fit k in-sample (pooled
        # across all DGPs), then leave-one-DGP-out cross-validate exactly
        # like the (lag, |X|) law above, for an apples-to-apples comparison.
        all_median_true = []
        for res in dgp_results:
            all_median_true.extend(res["pooled"][name]["median_true_lookup"])
        all_median_true_t = torch.tensor(all_median_true)

        def _f1_for_k(scores_t, labels_t, median_true_t, k):
            tau = k * median_true_t
            preds = scores_t >= tau
            tp = int((preds & labels_t).sum().item())
            fp = int((preds & ~labels_t).sum().item())
            fn = int((~preds & labels_t).sum().item())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        k_grid = make_log_grid(low=0.01, high=3.0, num=25)
        best_k, best_k_f1 = None, -1.0
        for k in k_grid:
            f1 = _f1_for_k(all_scores_t, all_labels_t, all_median_true_t, k)
            if f1 > best_k_f1:
                best_k, best_k_f1 = k, f1
        print(f"\n  CMI-anchored law (in-sample): tau = {best_k:.4g} * median(true-edge CMI | lag, DGP)"
              f"  (pooled F1={best_k_f1:.3f})")

        cmi_cv_f1s = []
        for held_out_idx in range(len(dgp_results)):
            train_scores2, train_labels2, train_median2 = [], [], []
            for i, res in enumerate(dgp_results):
                if i == held_out_idx:
                    continue
                train_scores2.extend(res["pooled"][name]["scores"])
                train_labels2.extend(res["pooled"][name]["labels"])
                train_median2.extend(res["pooled"][name]["median_true_lookup"])
            if len(set(train_labels2)) < 2:
                continue
            train_scores2_t, train_labels2_t, train_median2_t = (
                torch.tensor(train_scores2), torch.tensor(train_labels2), torch.tensor(train_median2)
            )
            fold_best_k, fold_best_f1 = None, -1.0
            for k in k_grid:
                f1 = _f1_for_k(train_scores2_t, train_labels2_t, train_median2_t, k)
                if f1 > fold_best_f1:
                    fold_best_k, fold_best_f1 = k, f1
            held_out_res = dgp_results[held_out_idx]["pooled"][name]
            if len(set(held_out_res["labels"])) < 2:
                continue
            held_scores2_t = torch.tensor(held_out_res["scores"])
            held_labels2_t = torch.tensor(held_out_res["labels"])
            held_median2_t = torch.tensor(held_out_res["median_true_lookup"])
            cmi_cv_f1s.append(_f1_for_k(held_scores2_t, held_labels2_t, held_median2_t, fold_best_k))

        mean_cmi_cv_f1 = statistics.mean(cmi_cv_f1s) if cmi_cv_f1s else float("nan")
        std_cmi_cv_f1 = statistics.stdev(cmi_cv_f1s) if len(cmi_cv_f1s) > 1 else 0.0
        print(f"  CMI-anchored law (leave-one-out CV): F1={mean_cmi_cv_f1:.3f} +/- {std_cmi_cv_f1:.3f}  "
              f"(n_folds={len(cmi_cv_f1s)})")

        mean_cv_f1 = statistics.mean(cv_f1s) if cv_f1s else float("nan")
        std_cv_f1 = statistics.stdev(cv_f1s) if len(cv_f1s) > 1 else 0.0
        mean_bespoke_f1 = statistics.mean(per_dgp_best_f1) if per_dgp_best_f1 else float("nan")
        print(f"\n  SUMMARY for \"{name}\":")
        print(f"    global (lag, |X|) law (in-sample):        F1={joint_fit.f1:.3f}   {joint_fit.summary()}")
        print(f"    global (lag, |X|) law (leave-one-out CV): F1={mean_cv_f1:.3f} +/- {std_cv_f1:.3f}  "
              f"(n_folds={len(cv_f1s)})")
        print(f"    CMI-anchored law (in-sample):             F1={best_k_f1:.3f}   tau = {best_k:.4g} * "
              f"median(true-edge CMI)")
        print(f"    CMI-anchored law (leave-one-out CV):      F1={mean_cmi_cv_f1:.3f} +/- {std_cmi_cv_f1:.3f}  "
              f"(n_folds={len(cmi_cv_f1s)})")
        print(f"    per-DGP bespoke fit (upper bound if refit every time): F1={mean_bespoke_f1:.3f}")
        print(f"    transfer cost, (lag,|X|) law (bespoke - CV): {mean_bespoke_f1 - mean_cv_f1:.3f}")
        print(f"    transfer cost, CMI-anchored law (bespoke - CV): {mean_bespoke_f1 - mean_cmi_cv_f1:.3f}")

        report["strategies"][name] = {
            "joint_fit_in_sample": {
                "signal_scale": joint_fit.signal_scale,
                "vocab_exponent_signal": joint_fit.vocab_exponent_signal,
                "lag_exponent": joint_fit.lag_exponent,
                "floor_scale": joint_fit.floor_scale,
                "vocab_exponent_floor": joint_fit.vocab_exponent_floor,
                "f1": joint_fit.f1,
            },
            "leave_one_out_cv_f1_mean": mean_cv_f1,
            "leave_one_out_cv_f1_std": std_cv_f1,
            "leave_one_out_cv_f1_per_fold": cv_f1s,
            "cmi_anchored_k_in_sample": best_k,
            "cmi_anchored_f1_in_sample": best_k_f1,
            "cmi_anchored_cv_f1_mean": mean_cmi_cv_f1,
            "cmi_anchored_cv_f1_std": std_cmi_cv_f1,
            "cmi_anchored_cv_f1_per_fold": cmi_cv_f1s,
            "tau_star_over_median_true_cmi_ratio": {
                "mean": statistics.mean(ratios) if ratios else float("nan"),
                "std": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
                "min": min(ratios) if ratios else float("nan"),
                "max": max(ratios) if ratios else float("nan"),
                "values": ratios,
            },
            "per_dgp_bespoke_f1_mean": mean_bespoke_f1,
            "per_dgp_bespoke_f1": per_dgp_best_f1,
        }

    if args.results_json:
        import json

        out_path = Path(args.results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
