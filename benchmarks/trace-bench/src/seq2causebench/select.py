# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Edge selection at the frozen cut: an edge is present iff its aggregated score exceeds τ strictly.

`scores` is the table `project.PairAccumulator.result()` returns (or
`read_scores_npz`): one row per `(src, dst)` token-id pair with `max_<col>`,
`mean_<col>`, `count` and `lag_max_<col>` per column (an arm on a path). The
threshold-free axes read the full ranking, so every scored pair is also
written with its score.
"""

from __future__ import annotations

import numpy as np

from .constants import AGGREGATIONS


def column(scores, col, agg):
    if agg not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {agg!r}")
    key = f"{agg}_{col}"
    if key not in scores:
        raise ValueError(f"the scores table has no column {col!r} (looked for {key!r})")
    return np.asarray(scores[key], dtype=np.float64)


def select_edges(scores, col, agg, tau):
    """`(src, dst, score)` arrays of the pairs with `score > tau`."""
    s = column(scores, col, agg)
    hit = s > tau
    return scores["src"][hit], scores["dst"][hit], s[hit]


def ranking(scores, col, agg):
    """Every scored pair with its score (for AUROC / AP)."""
    s = column(scores, col, agg)
    return scores["src"], scores["dst"], s


def per_lag_ranking(scores, col, lag):
    """Pairs whose per-lag max at `lag` (1-based) exists, with that value."""
    key = f"lag_max_{col}"
    if key not in scores:
        raise ValueError(f"the scores table has no per-lag column for {col!r}")
    lm = np.asarray(scores[key], dtype=np.float64)
    col_vals = lm[:, lag - 1]
    ok = np.isfinite(col_vals)
    return scores["src"][ok], scores["dst"][ok], col_vals[ok]


__all__ = ["column", "select_edges", "ranking", "per_lag_ranking"]
