"""Generates the figures for the reply to Chadyuk et al. (2026) from the
results captured in `reports/results_vocab100_memory4.json` (produced by
`scripts/train_and_test_lagged_effects.py`; see that file's `common_settings`
for the exact experimental configuration).

Usage:
    python scripts/plot_lagged_effects_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_PATH = REPORTS_DIR / "results_vocab100_memory4.json"

STRATEGIES = ["full", "atomic", "windowed", "independent_mediator", "in_distribution_noise"]
STRATEGY_LABELS = {
    "full": "full (staircase)",
    "atomic": "atomic",
    "windowed": "windowed (k=2)",
    "independent_mediator": "independent_mediator",
    "in_distribution_noise": "in_distribution_noise",
}
STRATEGY_COLORS = {
    "full": "#D62728",
    "atomic": "#2CA02C",
    "windowed": "#1F77B4",
    "independent_mediator": "#9467BD",
    "in_distribution_noise": "#FF7F0E",
}
# `full` and `in_distribution_noise`/`independent_mediator` frequently land on
# nearly-identical recall values (see the decayed-DGP/shared-tau panel), so a
# plain same-marker/same-zorder line plot can silently hide `full`'s line
# underneath a later-drawn, visually-identical one. Give `full` a distinct
# marker, extra line width, and the highest zorder so it always stays visible
# on top even when curves coincide exactly.
STRATEGY_MARKERS = {
    "full": "*",
    "atomic": "o",
    "windowed": "s",
    "independent_mediator": "^",
    "in_distribution_noise": "D",
}
STRATEGY_ZORDER = {
    "full": 5,
    "atomic": 4,
    "windowed": 3,
    "independent_mediator": 2,
    "in_distribution_noise": 1,
}
REGIME_LABELS = {
    "flat_decay": "Flat DGP (decay_rate=0.0):\nequal strength at every lag",
    "decayed": "Decayed DGP (decay_rate=0.3):\npaper-style exponential lag decay",
}


def _lags(recall_dict: dict) -> list[int]:
    return sorted(int(k) for k in recall_dict.keys())


def plot_recall_grid(data: dict, out_path: Path) -> None:
    """2x2 grid: rows = DGP regime (flat / decayed), cols = thresholding
    scheme (shared / tailored). Each panel: recall vs. lag, one line per
    intervention strategy, error bars = std across 3 independent particle-
    draw repeats."""
    regimes = ["flat_decay", "decayed"]
    schemes = [("recall_shared_tau", "Single SHARED tau per strategy"),
               ("recall_tailored_tau", "Tau TAILORED per lag")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, sharex=True)

    for row, regime in enumerate(regimes):
        regime_data = data["regimes"][regime]
        for col, (scheme_key, scheme_title) in enumerate(schemes):
            ax = axes[row, col]
            for strategy in STRATEGIES:
                per_lag = regime_data[scheme_key][strategy]
                lags = _lags(per_lag)
                means = [per_lag[str(lag)][0] for lag in lags]
                stds = [per_lag[str(lag)][1] for lag in lags]
                is_full = strategy == "full"
                ax.errorbar(
                    lags, means, yerr=stds, marker=STRATEGY_MARKERS[strategy], capsize=3,
                    linewidth=3 if is_full else 2,
                    markersize=11 if is_full else 7,
                    zorder=STRATEGY_ZORDER[strategy],
                    color=STRATEGY_COLORS[strategy],
                    label=STRATEGY_LABELS[strategy] if (row == 0 and col == 0) else None,
                )
            ax.set_xticks(lags)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(scheme_title, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{REGIME_LABELS[regime]}\n\nRecall", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("Lag (cause-effect distance)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(
        "Per-lag recall: shared vs. tailored threshold, flat vs. decayed DGP\n"
        "(vocab=100, memory=4, N=64 particles, 50 held-out sequences, 3 stability repeats)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cmi_separation(data: dict, out_path: Path) -> None:
    """Median CMI (log scale), true edges vs. non-edges, one panel per DGP
    regime, grouped bars per strategy."""
    regimes = ["flat_decay", "decayed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(STRATEGIES))
    width = 0.35

    for ax, regime in zip(axes, regimes):
        regime_data = data["regimes"][regime]["median_cmi"]
        true_means = [regime_data[s]["true"][0] for s in STRATEGIES]
        false_means = [regime_data[s]["false"][0] for s in STRATEGIES]

        ax.bar(x - width / 2, true_means, width, label="true edges", color="#2CA02C")
        ax.bar(x + width / 2, false_means, width, label="non-edges", color="#D62728")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [STRATEGY_LABELS[s] for s in STRATEGIES], fontsize=8, rotation=25, ha="right"
        )
        ax.set_title(REGIME_LABELS[regime].replace("\n", " "), fontsize=10)
        ax.set_ylabel("Median CMI (nats, log scale)")
        ax.grid(alpha=0.3, axis="y")

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Median CMI, true edges vs. non-edges (mean over 3 repeats)\n"
        "Separation of ~2-3 orders of magnitude in every condition -- the intervention is detecting real signal.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tailored_tau_by_lag(data: dict, out_path: Path) -> None:
    """Shows how the tailored tau shrinks with lag for the 'full' strategy in
    the decayed regime -- the concrete reason a single shared tau under-calls
    deeper lags."""
    regime_data = data["regimes"]["decayed"]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()

    lags = _lags(regime_data["recall_tailored_tau"]["full"])
    tailored_tau = [regime_data["recall_tailored_tau"]["full"][str(lag)][2] for lag in lags]
    shared_tau = [regime_data["selected_tau"]["full"][0]] * len(lags)
    recall_shared = [regime_data["recall_shared_tau"]["full"][str(lag)][0] for lag in lags]
    recall_tailored = [regime_data["recall_tailored_tau"]["full"][str(lag)][0] for lag in lags]

    ax1.plot(lags, shared_tau, "k--", marker="s", label="shared tau (same at every lag)")
    ax1.plot(lags, tailored_tau, color="#1F77B4", marker="o", label="tailored tau (per lag)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Lag (cause-effect distance)")
    ax1.set_ylabel("Selected threshold tau (nats, log scale)")
    ax1.set_xticks(lags)

    ax2.plot(lags, recall_shared, color="#D62728", marker="x", linestyle=":", label="recall @ shared tau")
    ax2.plot(lags, recall_tailored, color="#2CA02C", marker="^", linestyle=":", label="recall @ tailored tau")
    ax2.set_ylabel("Recall (strategy=full)")
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)
    ax1.set_title("strategy=\"full\", decayed DGP: a shared tau under-calls deep lags\nnot because signal is absent, but because it sits at a smaller scale")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_PATH.read_text())

    plot_recall_grid(data, FIGURES_DIR / "fig1_recall_by_lag_shared_vs_tailored.png")
    plot_cmi_separation(data, FIGURES_DIR / "fig2_cmi_true_vs_false_separation.png")
    plot_tailored_tau_by_lag(data, FIGURES_DIR / "fig3_tau_and_recall_by_lag_full_decayed.png")

    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
