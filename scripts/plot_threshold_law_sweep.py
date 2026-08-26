"""Figure showing, per held-out DGP, the leave-one-out cross-validated F1 of
the joint lag x vocab-size threshold law against that DGP's own bespoke
(per-DGP-refit) upper bound -- visualizes exactly where/how much
"transfer cost" the single global formula pays, read from the JSON produced
by `scripts/simulate_threshold_law.py`.

Usage:
    python scripts/plot_threshold_law_sweep.py \
        [--results-json reports/results_threshold_law_sweep.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-json", type=str, default=str(REPORTS_DIR / "results_threshold_law_sweep.json"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = json.loads(Path(args.results_json).read_text())
    dgp_grid = data["dgp_grid"]
    strategies = list(data["strategies"].keys())

    fig, axes = plt.subplots(1, len(strategies), figsize=(7 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        entry = data["strategies"][strategy]
        bespoke = entry["per_dgp_bespoke_f1"]
        cv = entry["leave_one_out_cv_f1_per_fold"]
        labels = [f'|X|={d["vocab_size"]}\nrate={d["decay_rate"]}' for d in dgp_grid[: len(bespoke)]]
        x = range(len(labels))
        width = 0.35
        ax.bar([xi - width / 2 for xi in x], bespoke, width, label="bespoke (per-DGP refit)", color="#2CA02C")
        ax.bar([xi + width / 2 for xi in x], cv, width, label="leave-one-out CV (global law)", color="#1F77B4")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        mean_bespoke = entry["per_dgp_bespoke_f1_mean"]
        mean_cv = entry["leave_one_out_cv_f1_mean"]
        std_cv = entry["leave_one_out_cv_f1_std"]
        ax.set_title(
            f'strategy="{strategy}"\nmean bespoke={mean_bespoke:.3f}  '
            f"mean CV={mean_cv:.3f}+/-{std_cv:.3f}  (transfer cost={mean_bespoke - mean_cv:.3f})",
            fontsize=9,
        )

    fig.suptitle(
        "Per-DGP F1: bespoke (per-setup refit) vs. leave-one-DGP-out cross-validated global law",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig9_threshold_law_transfer_cost.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {out_path}")


if __name__ == "__main__":
    main()
