"""Figures comparing ALL per-lag thresholding schemes (shared / tailored /
exponential-decay-fit / sub-linear power-law-fit / MAD / percentile / Otsu /
GMM), read from a JSON produced by:

    python scripts/train_and_test_lagged_effects.py ... --results-json <path>

Usage:
    python scripts/plot_all_threshold_schemes.py \
        [--results-json reports/results_vocab100_memory4_all_schemes.json] \
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
    "shared": {"color": "#D62728", "marker": "s", "label": "shared", "supervised": True},
    "tailored": {"color": "#2CA02C", "marker": "^", "label": "tailored", "supervised": True},
    "exponential": {"color": "#1F77B4", "marker": "o", "label": "exponential-fit", "supervised": True},
    "power_law": {"color": "#9467BD", "marker": "P", "label": "power-law-fit (sub-linear)", "supervised": True},
    "mad": {"color": "#FF7F0E", "marker": "x", "label": "MAD (unsupervised)", "supervised": False},
    "percentile": {"color": "#8C564B", "marker": "v", "label": "percentile-95 (unsupervised)", "supervised": False},
    "otsu": {"color": "#E377C2", "marker": "D", "label": "Otsu (unsupervised)", "supervised": False},
    "gmm": {"color": "#7F7F7F", "marker": "*", "label": "GMM (unsupervised)", "supervised": False},
}
SCHEME_ORDER = list(SCHEME_STYLE.keys())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-json",
        type=str,
        default=str(REPORTS_DIR / "results_vocab100_memory4_all_schemes.json"),
    )
    p.add_argument("--strategies", nargs="+", default=["full", "atomic"])
    return p.parse_args(argv)


def _lags(d: dict) -> list[int]:
    return sorted(int(k) for k in d.keys())


def plot_f1_bars(data: dict, strategies: list[str], out_path: Path) -> None:
    """Grouped bar chart: pooled F1 per scheme, one group per strategy."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_schemes = len(SCHEME_ORDER)
    width = 0.8 / len(strategies)
    x = range(n_schemes)

    for i, strategy in enumerate(strategies):
        entry = data["strategies"][strategy]
        means = [entry["schemes"][s]["pooled_f1"][0] for s in SCHEME_ORDER]
        stds = [entry["schemes"][s]["pooled_f1"][1] for s in SCHEME_ORDER]
        offset = (i - (len(strategies) - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x], means, width, yerr=stds, capsize=3,
            label=f'strategy="{strategy}"', alpha=0.85,
        )
        for bar, scheme in zip(bars, SCHEME_ORDER):
            if not SCHEME_STYLE[scheme]["supervised"]:
                bar.set_hatch("///")

    ax.set_xticks(list(x))
    ax.set_xticklabels([SCHEME_STYLE[s]["label"] for s in SCHEME_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Pooled F1 (all lags combined)")
    ax.set_ylim(0, 1.0)
    ax.axvline(3.5, color="black", linestyle=":", linewidth=1)
    ax.text(3.5, 0.95, "supervised | unsupervised (label-free) ->", fontsize=8, ha="center")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        "Pooled F1 by thresholding scheme: validation-supervised schemes (solid) "
        "vs. anomaly-detection-style unsupervised cutoffs (hatched)\n"
        "(vocab=100, memory=4, decay_rate=0.3, 3 stability repeats)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recall_by_lag_all_schemes(data: dict, strategy: str, out_path: Path) -> None:
    """Line plot: recall vs. lag, one line per scheme, for a single strategy."""
    entry = data["strategies"][strategy]
    lags = _lags(entry["schemes"]["shared"]["recall_by_lag"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for scheme in SCHEME_ORDER:
        style = SCHEME_STYLE[scheme]
        recall_dict = entry["schemes"][scheme]["recall_by_lag"]
        means = [recall_dict[str(lag)][0] for lag in lags]
        stds = [recall_dict[str(lag)][1] for lag in lags]
        linestyle = "-" if style["supervised"] else "--"
        ax.errorbar(
            lags, means, yerr=stds, marker=style["marker"], capsize=3, linewidth=2,
            linestyle=linestyle, color=style["color"], label=style["label"],
        )
    ax.set_xticks(lags)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Lag (cause-effect distance)")
    ax.set_ylabel("Recall")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f'Per-lag recall, all thresholding schemes, strategy="{strategy}"', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decay_exponents(data: dict, strategies: list[str], out_path: Path) -> None:
    """Bar chart comparing the fitted exponential decay_rate against the
    fitted power-law exponent, per strategy -- the direct answer to "is the
    per-lag threshold decay sub-linear (power law) or exponential?"."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(strategies))
    width = 0.35
    exp_rates = [data["strategies"][s]["schemes"]["exponential"]["fit_params"]["decay_rate"][0] for s in strategies]
    pow_exponents = [data["strategies"][s]["schemes"]["power_law"]["fit_params"]["exponent"][0] for s in strategies]
    ax.bar([xi - width / 2 for xi in x], exp_rates, width, label="exponential decay_rate", color="#1F77B4")
    ax.bar([xi + width / 2 for xi in x], pow_exponents, width, label="power-law exponent b", color="#9467BD")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.text(len(strategies) - 0.5, 1.02, "b=1 (harmonic, ~1/lag)", fontsize=7, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("Fitted decay parameter")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        "Fitted lag-decay steepness by strategy\n"
        "(power-law exponent b<1 = sub-linear decay, matching Chadyuk et al.'s |X|-scaling finding)",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = json.loads(Path(args.results_json).read_text())
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_f1_bars(data, args.strategies, FIGURES_DIR / "fig6_pooled_f1_all_schemes.png")
    for strategy in args.strategies:
        plot_recall_by_lag_all_schemes(
            data, strategy, FIGURES_DIR / f"fig7_recall_all_schemes_{strategy}.png"
        )
    plot_decay_exponents(data, list(data["strategies"].keys()), FIGURES_DIR / "fig8_decay_exponents.png")
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
