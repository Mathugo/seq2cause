"""Figures demonstrating the exponential-decay-to-a-floor threshold
(`threshold.fit_exponential_lag_threshold`) against real per-lag recall data,
read from a JSON produced by:

    python scripts/train_and_test_lagged_effects.py ... --results-json <path>

Usage:
    python scripts/plot_exponential_threshold_demo.py \
        [--results-json reports/results_vocab100_memory4_with_exponential.json] \
        [--strategies full atomic]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SCHEME_STYLE = {
    "recall_shared_tau": {"color": "#D62728", "marker": "s", "label": "shared tau"},
    "recall_tailored_tau": {"color": "#2CA02C", "marker": "^", "label": "tailored tau (per-lag independent)"},
    "recall_exponential_tau": {"color": "#1F77B4", "marker": "o", "label": "exponential-decay-fit tau"},
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-json",
        type=str,
        default=str(REPORTS_DIR / "results_vocab100_memory4_with_exponential.json"),
    )
    p.add_argument("--strategies", nargs="+", default=["full", "atomic"])
    return p.parse_args(argv)


def _lags(d: dict) -> list[int]:
    return sorted(int(k) for k in d.keys())


def plot_recall_comparison(data: dict, strategies: list[str], out_path: Path) -> None:
    """One panel per strategy: recall vs. lag for the three thresholding
    schemes (shared / tailored / exponential-decay-fit)."""
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        entry = data["strategies"][strategy]
        for scheme_key, style in SCHEME_STYLE.items():
            recall_dict = entry[scheme_key]
            lags = _lags(recall_dict)
            means = [recall_dict[str(lag)][0] for lag in lags]
            stds = [recall_dict[str(lag)][1] for lag in lags]
            ax.errorbar(
                lags, means, yerr=stds, marker=style["marker"], capsize=3, linewidth=2,
                color=style["color"], label=style["label"],
            )
        ax.set_xticks(_lags(entry["recall_shared_tau"]))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Lag (cause-effect distance)")
        f1_shared = entry["pooled_f1_shared"][0]
        f1_tailored = entry["pooled_f1_tailored"][0]
        f1_exponential = entry["exponential_fit_params"]["f1"][0]
        ax.set_title(
            f'strategy="{strategy}"\npooled F1: shared={f1_shared:.3f}  '
            f'exponential={f1_exponential:.3f}  tailored={f1_tailored:.3f}',
            fontsize=10, fontweight="bold",
        )
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Recall")
    axes[0].legend(loc="lower left", fontsize=9)
    n_true = data["n_true_edges_total"]
    settings = data["settings"]
    fig.suptitle(
        "Per-lag recall: shared vs. tailored vs. exponential-decay-fit threshold\n"
        f"(vocab={settings['vocab_size']}, memory={settings['memory']}, decay_rate={settings['decay_rate']}, "
        f"N={settings['n_particles']} particles, {settings['n_eval_sequences']} sequences, "
        f"{n_true} true edges total, {settings['n_stability_repeats']} stability repeats)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tau_curve(data: dict, strategy: str, out_path: Path) -> None:
    """Overlays the shared tau (flat line), the independently-fit per-lag
    tau (tailored, scatter points), and the fitted exponential-decay-to-a-
    floor curve for one strategy -- the direct visual case for the
    exponential fit's usability: a 3-parameter curve tracing the same shape
    as the noisier independent per-lag fits, while also extrapolating
    smoothly to lags with too little data to fit independently.
    """
    entry = data["strategies"][strategy]
    lags = _lags(entry["recall_shared_tau"])
    shared_tau = entry["selected_tau"][0]
    tailored_tau = [entry["tau_by_lag_tailored"][str(lag)][0] for lag in lags]
    exponential_tau = [entry["tau_by_lag_exponential"][str(lag)][0] for lag in lags]
    fit_params = entry["exponential_fit_params"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(shared_tau, color="#D62728", linestyle="--", label="shared tau (flat)")
    ax.scatter(lags, tailored_tau, color="#2CA02C", marker="^", s=70,
               label="tailored tau (independent per-lag fit)", zorder=3)
    ax.plot(lags, exponential_tau, color="#1F77B4", marker="o",
            label="exponential-decay-fit tau (3 shared parameters)", zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(lags)
    ax.set_xlabel("Lag (cause-effect distance)")
    ax.set_ylabel("Selected threshold tau (nats, log scale)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(
        f'strategy="{strategy}": tau(lag) = {fit_params["noise_floor"][0]:.3g} + '
        f'({fit_params["tau_at_lag1"][0]:.3g} - {fit_params["noise_floor"][0]:.3g}) '
        f'* exp(-{fit_params["decay_rate"][0]:.2g} * (lag-1))\n'
        f'pooled F1 = {fit_params["f1"][0]:.3f} (mean over stability repeats)',
        fontsize=10,
    )
    tau1 = fit_params["tau_at_lag1"][0]
    floor = fit_params["noise_floor"][0]
    ax.text(
        0.02, 0.03,
        f"noise_floor is {tau1 / floor:.0f}x smaller than tau_at_lag1: over these {max(lags)} "
        "lags the curve is still in its\npure-exponential regime, so log(tau) is ~linear in "
        "lag by construction (log-y axis) -- this is\nthe EXPECTED shape for a good "
        "exponential-decay fit, not evidence the fit collapsed to linear. Curvature/\n"
        "flattening would only appear once tau(lag) approaches noise_floor, i.e. at much "
        "larger lags.",
        transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = json.loads(Path(args.results_json).read_text())
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_recall_comparison(
        data, args.strategies, FIGURES_DIR / "fig4_recall_shared_vs_tailored_vs_exponential.png"
    )
    for strategy in args.strategies:
        plot_tau_curve(
            data, strategy, FIGURES_DIR / f"fig5_tau_curve_{strategy}.png"
        )
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
