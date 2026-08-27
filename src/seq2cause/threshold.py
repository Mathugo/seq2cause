"""Threshold selection utilities for TRACE-style CMI thresholding.

Practice credited to:
    Chadyuk, A., Zhang, A., and Kucukates, R. "Replicating TRACE: A
    Practitioner's Guide to Its Threshold and Particle Budget." LotusFlare
    Inc., August 2026.

Their independent replication found that a single hardcoded threshold (e.g.
the tau=3e-5 printed alongside our Table 2 configuration) does *not* transfer
across generator/backbone setups, but that selecting tau by maximizing F1 on
a held-out validation split *does* reproduce the reported performance
robustly across vocabulary sizes (100 to 2000+, and out-of-sample at 5000).
Their reading of *why* it transfers: the blind validation optimum is pinned
to the truth margin delta used to define ground-truth edges (roughly
delta/2 to delta once the CMI estimator is level-calibrated), not to any
single portable constant. This module operationalizes "select tau on a
held-out validation split" as the recommended default workflow, and reports
the selected tau relative to delta (when known) rather than as a bare
number, plus a warning when the selection looks like it may be an artifact
of a too-small/unrepresentative validation split.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "make_log_grid",
    "f1_precision_recall",
    "pooled_f1_for_tau_by_lag",
    "ThresholdSelectionResult",
    "select_threshold_by_validation",
    "select_thresholds_by_group",
    "ExponentialLagThresholdFit",
    "exponential_lag_threshold",
    "fit_exponential_lag_threshold",
    "mad_threshold",
    "percentile_threshold",
    "otsu_threshold",
    "GMM1DFit",
    "gmm_threshold",
    "AdaptiveThreshold",
    "PowerLawLagThresholdFit",
    "power_law_lag_threshold",
    "fit_power_law_lag_threshold",
    "JointLagVocabThresholdFit",
    "fit_joint_lag_vocab_threshold",
    "resolve_threshold",
]

# Sensible default sweep range: wide enough to contain both the constants
# historically used in our example configs (~1e-5 to 1e-4) and the much
# larger validation optima reported by Chadyuk et al. (2026) (~1e-2).
DEFAULT_GRID_MIN = 1e-6
DEFAULT_GRID_MAX = 1e-1
DEFAULT_GRID_POINTS = 23

# Chadyuk et al. (2026) report the blind validation optimum typically lands
# within roughly [delta/2, delta] for a level-calibrated estimator. We flag
# selections that fall well outside a slightly widened version of that band.
_DELTA_RATIO_LOW = 0.3
_DELTA_RATIO_HIGH = 1.2


def make_log_grid(
    low: float = DEFAULT_GRID_MIN,
    high: float = DEFAULT_GRID_MAX,
    num: int = DEFAULT_GRID_POINTS,
) -> list[float]:
    """Builds a log-spaced grid of candidate thresholds between `low` and `high`."""
    if low <= 0 or high <= 0:
        raise ValueError(f"grid bounds must be strictly positive, got low={low}, high={high}")
    if high < low:
        raise ValueError(f"`high` ({high}) must be >= `low` ({low})")
    if num < 1:
        raise ValueError(f"`num` must be >= 1, got {num}")
    return torch.logspace(math.log10(low), math.log10(high), steps=num).tolist()


def _binary_confusion(scores: Tensor, labels: Tensor, tau: float) -> tuple[int, int, int, int]:
    preds = scores >= tau
    labels = labels.bool()
    tp = int((preds & labels).sum().item())
    fp = int((preds & ~labels).sum().item())
    fn = int((~preds & labels).sum().item())
    tn = int((~preds & ~labels).sum().item())
    return tp, fp, fn, tn


def f1_precision_recall(
    scores: Tensor, labels: Tensor, tau: float, eps: float = 1e-12
) -> tuple[float, float, float]:
    """Computes (F1, precision, recall) for calling an edge whenever `scores >= tau`."""
    tp, fp, fn, _tn = _binary_confusion(scores, labels, tau)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + eps) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


@dataclass
class ThresholdSelectionResult:
    """Result of a validation-split threshold sweep.

    `tau` is the selected threshold. Read it together with `delta` (the truth
    margin, if known) via `tau_over_delta` rather than as a bare constant --
    see module docstring.
    """

    tau: float
    f1: float
    precision: float
    recall: float
    grid: list[float]
    f1_by_tau: dict[float, float]
    delta: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def tau_over_delta(self) -> float | None:
        if self.delta is None or self.delta <= 0:
            return None
        return self.tau / self.delta

    def summary(self) -> str:
        lines = [
            f"selected tau = {self.tau:.4g}  "
            f"(F1={self.f1:.3f}, precision={self.precision:.3f}, recall={self.recall:.3f})"
        ]
        ratio = self.tau_over_delta
        if ratio is not None:
            lines.append(
                f"tau / delta = {ratio:.3f}  (delta={self.delta:.4g}); Chadyuk et al. (2026) "
                "report the blind validation optimum typically falls within roughly "
                "[delta/2, delta] for a level-calibrated estimator"
            )
        else:
            lines.append(
                "delta (truth margin) was not provided -- report tau relative to delta when "
                "it is known, rather than as an absolute constant (Chadyuk et al., 2026)."
            )
        for w in self.warnings:
            lines.append(f"[warning] {w}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


def select_threshold_by_validation(
    cmi_scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[bool],
    grid: Iterable[float] | None = None,
    delta: float | None = None,
    hardcoded_defaults: Mapping[str, float] | Sequence[float] | None = None,
    warn_order_of_magnitude: float = 1.0,
    emit_warnings: bool = True,
) -> ThresholdSelectionResult:
    """Selects tau by maximizing F1 on a held-out validation split.

    This is the practice credited to Chadyuk et al. (2026): rather than
    trusting a fixed constant across setups, sweep a log-spaced grid of
    candidate thresholds and pick the one that maximizes F1 against
    validation-split labels (edge / non-edge, from your own ground-truth
    procedure), then apply that *selected* tau (never re-tuned) to the test
    split.

    Args:
        cmi_scores: per-pair (or per-position) CMI estimates on the
            validation split, one scalar per candidate (cause, effect) pair.
        labels: boolean (or 0/1) ground-truth edge labels, aligned with
            `cmi_scores`.
        grid: candidate thresholds to sweep. Defaults to a 23-point
            log-spaced grid from 1e-6 to 1e-1 (see `make_log_grid`).
        delta: the truth margin used to define a ground-truth edge (e.g. the
            KL threshold used by your intervention-based oracle), if known.
            Used only for reporting/warnings -- never to bias the selection.
        hardcoded_defaults: name -> value mapping (or a bare sequence) of any
            fixed thresholds hardcoded elsewhere in your configs/pipeline
            (e.g. the tau printed in a paper table). Triggers a warning if
            the selected tau lands within `warn_order_of_magnitude` decades
            of one of them, which Chadyuk et al. (2026) note can indicate the
            validation split is too small/unrepresentative to move the
            optimum away from a stale constant.
        warn_order_of_magnitude: width, in decades (log10 units), of the
            "too close to a hardcoded default" warning band. Default 1.0
            (i.e. within one order of magnitude).
        emit_warnings: if True (default), also raise Python `UserWarning`s
            for anything appended to `result.warnings`.

    Returns:
        A `ThresholdSelectionResult` with the selected tau and diagnostics.
    """
    if grid is None:
        grid = make_log_grid()
    grid = list(grid)
    if not grid:
        raise ValueError("`grid` must contain at least one candidate threshold.")

    scores_t = torch.as_tensor(cmi_scores, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.bool)
    if scores_t.numel() == 0:
        raise ValueError("cmi_scores is empty; cannot select a threshold.")
    if scores_t.shape != labels_t.shape:
        raise ValueError(
            f"cmi_scores and labels must have the same shape, got {tuple(scores_t.shape)} "
            f"vs {tuple(labels_t.shape)}"
        )

    f1_by_tau: dict[float, float] = {}
    best: tuple[float, float, float, float] | None = None
    for tau in grid:
        f1, precision, recall = f1_precision_recall(scores_t, labels_t, tau)
        f1_by_tau[tau] = f1
        if best is None or f1 > best[1]:
            best = (tau, f1, precision, recall)

    assert best is not None
    tau, f1, precision, recall = best

    collected_warnings: list[str] = []

    if delta is not None and delta > 0:
        ratio = tau / delta
        if not (_DELTA_RATIO_LOW <= ratio <= _DELTA_RATIO_HIGH):
            collected_warnings.append(
                f"selected tau ({tau:.4g}) is {ratio:.2f}x the truth margin delta ({delta:.4g}); "
                "Chadyuk et al. (2026) observed the blind validation optimum typically sits "
                f"within roughly [{_DELTA_RATIO_LOW}, {_DELTA_RATIO_HIGH}]x delta once the "
                "estimator is level-calibrated. A large deviation can indicate an "
                "uncalibrated estimator, too few validation sequences, or a delta that does "
                "not match this estimator's regime."
            )

    if hardcoded_defaults:
        if isinstance(hardcoded_defaults, Mapping):
            named_defaults = list(hardcoded_defaults.items())
        else:
            named_defaults = [(f"default[{i}]", v) for i, v in enumerate(hardcoded_defaults)]
        for name, default_val in named_defaults:
            if default_val is None or default_val <= 0:
                continue
            log_gap = abs(math.log10(tau) - math.log10(default_val))
            if log_gap < warn_order_of_magnitude:
                collected_warnings.append(
                    f"selected tau ({tau:.4g}) is within {warn_order_of_magnitude:.1f} decade(s) "
                    f"of the hardcoded default '{name}'={default_val:.4g}. Per Chadyuk et al. "
                    "(2026), this pattern can be a sign that the validation split is too "
                    "small/unrepresentative to move the optimum away from a stale constant -- "
                    "consider a larger or more diverse validation split before trusting this "
                    "selection."
                )

    if emit_warnings:
        for w in collected_warnings:
            warnings.warn(w, stacklevel=2)

    return ThresholdSelectionResult(
        tau=tau,
        f1=f1,
        precision=precision,
        recall=recall,
        grid=grid,
        f1_by_tau=f1_by_tau,
        delta=delta,
        warnings=collected_warnings,
    )


def select_thresholds_by_group(
    cmi_scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[bool],
    groups: Tensor | Sequence[int],
    grid: Iterable[float] | None = None,
    delta: float | None = None,
    min_group_size: int = 1,
    min_reliable_true_edges: int = 10,
    fallback: str = "global",
    warn_on_unreliable_groups: bool = True,
    **kwargs,
) -> dict[int, ThresholdSelectionResult]:
    """Selects a SEPARATE tau per group (e.g. per lag) instead of one shared
    global tau.

    Motivation: CMI magnitude naturally decays with lag/delay -- whether
    because a do-intervention construction loses local context for distant
    causes (Chadyuk et al., 2026), or simply because the data-generating
    process's own causal strength decays with lag, or both (see
    `seq2cause.scm.NonlinearSCM`'s `decay_rate`). A single global tau, fit
    mostly by whichever group has the most (usually near-lag) pairs, then
    systematically under-calls groups whose CMI is reliably separated from
    noise but sits at a smaller absolute scale. Selecting tau per group
    tailors the threshold to that group's own causal-strength scale.

    Reliability caveat (found via `scripts/simulate_threshold_law.py`'s
    10-DGP cross-validated sweep): a per-group tau is only as reliable as
    the group's own true-edge sample count. In that sweep, the ratio of a
    group's F1-optimal tau to its own median true-edge CMI was tightly
    banded (roughly [0.3, 1.0], median ~0.7-0.8) for groups with more than
    ~10 true-edge samples, but swung over TWO ORDERS OF MAGNITUDE (up to
    ~300x) for thinner groups -- not because CMI-anchoring is wrong in
    principle, but because a median (or an F1-argmax) computed from a
    handful of points is itself unreliable. See `min_reliable_true_edges`.

    Args:
        cmi_scores: per-pair CMI estimates, one scalar per candidate pair.
        labels: aligned boolean ground-truth edge labels.
        groups: aligned integer group id per pair (e.g. the lag `q - j`).
        grid: candidate thresholds to sweep; forwarded to
            `select_threshold_by_validation` for every group.
        delta: the truth margin, for reporting; forwarded per group.
        min_group_size: groups with fewer samples than this are skipped (or
            handled per `fallback`) rather than fit on too little data.
        min_reliable_true_edges: groups that DO get their own per-group fit
            (i.e. pass `min_group_size` and have both classes) but have
            fewer true-edge samples than this are still fit and returned,
            but flagged as unreliable (see `warn_on_unreliable_groups`) --
            prefer `fit_exponential_lag_threshold`/`fit_power_law_lag_threshold`
            (which pool statistical strength across every group at once) for
            such groups instead of trusting their own tau in isolation.
        fallback: what to do for a group skipped by `min_group_size` (or with
            no true edges at all, which would make F1 degenerate):
            "global" (default) reuses the tau fit on the pooled, all-group
            data; "drop" omits the group from the returned dict entirely.
        warn_on_unreliable_groups: if True (default), also raises a Python
            `UserWarning` (in addition to appending to
            `result.warnings`) for any group with fewer than
            `min_reliable_true_edges` true-edge samples.
        **kwargs: forwarded to `select_threshold_by_validation` (e.g.
            `hardcoded_defaults`, `warn_order_of_magnitude`, `emit_warnings`).

    Returns:
        `{group_id: ThresholdSelectionResult}`.
    """
    if fallback not in ("global", "drop"):
        raise ValueError(f"fallback must be 'global' or 'drop', got {fallback!r}")

    scores_t = torch.as_tensor(cmi_scores, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.bool)
    groups_t = torch.as_tensor(groups)
    if not (scores_t.shape == labels_t.shape == groups_t.shape):
        raise ValueError(
            "cmi_scores, labels, and groups must all have the same shape, got "
            f"{tuple(scores_t.shape)}, {tuple(labels_t.shape)}, {tuple(groups_t.shape)}"
        )

    global_result = None
    if fallback == "global":
        global_result = select_threshold_by_validation(
            scores_t, labels_t, grid=grid, delta=delta, **kwargs
        )

    results: dict[int, ThresholdSelectionResult] = {}
    for group_id in sorted(set(groups_t.tolist())):
        mask = groups_t == group_id
        group_scores = scores_t[mask]
        group_labels = labels_t[mask]
        has_enough_samples = int(mask.sum()) >= min_group_size
        has_both_classes = bool(group_labels.any()) and bool((~group_labels).any())
        if has_enough_samples and has_both_classes:
            result = select_threshold_by_validation(
                group_scores, group_labels, grid=grid, delta=delta, **kwargs
            )
            n_true_in_group = int(group_labels.sum())
            if n_true_in_group < min_reliable_true_edges:
                msg = (
                    f"group {group_id!r}'s tau was selected from only {n_true_in_group} "
                    f"true-edge sample(s) (< min_reliable_true_edges={min_reliable_true_edges}). "
                    "A per-group threshold fit on this few true edges can be off by 2+ orders of "
                    "magnitude from what a larger sample would select (see "
                    "scripts/simulate_threshold_law.py's cross-validated sweep) -- prefer "
                    "fit_exponential_lag_threshold/fit_power_law_lag_threshold (which pool "
                    "statistical strength across every group at once) for this group instead of "
                    "trusting this tau in isolation, or collect more validation sequences."
                )
                result.warnings.append(msg)
                if warn_on_unreliable_groups:
                    warnings.warn(msg, stacklevel=2)
            results[int(group_id)] = result
        elif fallback == "global":
            results[int(group_id)] = global_result
        # else "drop": simply omit this group_id
    return results


@dataclass
class ExponentialLagThresholdFit:
    """Result of `fit_exponential_lag_threshold`: a 3-parameter tau(lag)
    curve, plus the pooled F1 it achieves and the resulting per-lag tau dict
    (ready to pass straight into `diagnostics.recall_by_lag_with_tau_by_lag`).
    """

    tau_at_lag1: float
    decay_rate: float
    noise_floor: float
    f1: float
    tau_by_lag: dict[int, float]

    def summary(self) -> str:
        return (
            f"tau(lag) = {self.noise_floor:.4g} + "
            f"({self.tau_at_lag1:.4g} - {self.noise_floor:.4g}) * exp(-{self.decay_rate:.3g} * (lag-1))"
            f"  (pooled F1={self.f1:.3f})"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


def exponential_lag_threshold(
    tau_at_lag1: float, decay_rate: float, noise_floor: float, max_lag: int
) -> dict[int, float]:
    """Builds a per-lag threshold that decays exponentially from
    `tau_at_lag1` (at lag=1) down toward a `noise_floor` "base intensity" as
    lag grows:

        tau(lag) = noise_floor + (tau_at_lag1 - noise_floor) * exp(-decay_rate * (lag - 1))

    Motivation: the validation-selected tau tends to shrink with lag toward
    (but not below) a floor set by the non-edge noise level, tracking the
    CMI's own decay -- rather than fitting each lag's threshold
    independently (as `select_thresholds_by_group` does, which can overfit
    lags with very few true edges), this bakes in that shape as a
    3-parameter curve that can be fit jointly across all lags at once (see
    `fit_exponential_lag_threshold`).

    Args:
        tau_at_lag1: threshold value at lag=1 (the curve's starting point).
        decay_rate: exponential decay rate; 0 recovers a flat threshold
            (`tau_at_lag1` at every lag), larger values decay faster.
        noise_floor: asymptotic threshold as lag -> infinity -- the "base
            intensity" the tau curve decays toward but never crosses.
        max_lag: largest lag to generate a threshold for.

    Returns:
        `{lag: tau}` for `lag` in `1..max_lag`.
    """
    if noise_floor < 0:
        raise ValueError(f"noise_floor must be >= 0, got {noise_floor}")
    if tau_at_lag1 < noise_floor:
        raise ValueError(
            f"tau_at_lag1 ({tau_at_lag1}) must be >= noise_floor ({noise_floor}) -- the curve "
            "decays from tau_at_lag1 down toward noise_floor as lag grows, so it would never "
            "reach tau_at_lag1 to begin with."
        )
    if decay_rate < 0:
        raise ValueError(f"decay_rate must be >= 0, got {decay_rate}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    return {
        lag: noise_floor + (tau_at_lag1 - noise_floor) * math.exp(-decay_rate * (lag - 1))
        for lag in range(1, max_lag + 1)
    }


def pooled_f1_for_tau_by_lag(
    scores: Tensor, labels: Tensor, lags: Tensor, tau_by_lag: dict[int, float]
) -> tuple[float, float, float]:
    """Computes pooled (F1, precision, recall) across ALL lags at once, for
    calling an edge at pair `p` whenever `scores[p] >= tau_by_lag[lags[p]]`.

    Useful for putting a shared-tau, a per-lag-tailored-tau, and a fitted
    exponential-decay-tau on the exact same footing: apply each scheme's own
    `tau_by_lag` dict (a shared tau is just `{lag: same_value for every lag}`)
    and compare the resulting single pooled F1 number directly.
    """
    preds = torch.zeros_like(labels, dtype=torch.bool)
    for lag, tau in tau_by_lag.items():
        mask = lags == lag
        preds = preds | (mask & (scores >= tau))
    tp = int((preds & labels).sum().item())
    fp = int((preds & ~labels).sum().item())
    fn = int((~preds & labels).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def fit_exponential_lag_threshold(
    cmi_scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[bool],
    lags: Tensor | Sequence[int],
    max_lag: int | None = None,
    tau_at_lag1_grid: Iterable[float] | None = None,
    noise_floor_grid: Iterable[float] | None = None,
    decay_rate_grid: Iterable[float] | None = None,
) -> ExponentialLagThresholdFit:
    """Grid-searches the 3 parameters of `exponential_lag_threshold` (the
    threshold's value at lag=1, its decay rate, and its noise-floor
    asymptote) to maximize pooled F1 across ALL lags at once.

    Because the curve has only 3 free parameters shared across every lag
    (rather than one independent threshold per lag), it borrows statistical
    strength across lags -- useful when some lags have too few true edges to
    reliably fit their own threshold via `select_thresholds_by_group`.

    Args:
        cmi_scores: per-pair CMI estimates, one scalar per candidate pair.
        labels: aligned boolean ground-truth edge labels.
        lags: aligned integer lag (`effect - cause`) per pair.
        max_lag: largest lag to fit a threshold for; defaults to `max(lags)`.
        tau_at_lag1_grid: candidate values for `tau_at_lag1`. Defaults to
            `make_log_grid()`.
        noise_floor_grid: candidate values for `noise_floor`. Defaults to a
            log-spaced grid an order of magnitude below `tau_at_lag1_grid`.
        decay_rate_grid: candidate decay rates. Defaults to
            `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]` (0 = flat).

    Returns:
        The best-F1 `ExponentialLagThresholdFit`.
    """
    scores_t = torch.as_tensor(cmi_scores, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.bool)
    lags_t = torch.as_tensor(lags, dtype=torch.long)
    if not (scores_t.shape == labels_t.shape == lags_t.shape):
        raise ValueError(
            "cmi_scores, labels, and lags must all have the same shape, got "
            f"{tuple(scores_t.shape)}, {tuple(labels_t.shape)}, {tuple(lags_t.shape)}"
        )
    if scores_t.numel() == 0:
        raise ValueError("cmi_scores is empty; cannot fit a threshold.")

    max_lag = max_lag if max_lag is not None else int(lags_t.max().item())
    tau_at_lag1_grid = list(tau_at_lag1_grid) if tau_at_lag1_grid is not None else make_log_grid()
    noise_floor_grid = (
        list(noise_floor_grid)
        if noise_floor_grid is not None
        else make_log_grid(low=DEFAULT_GRID_MIN * 1e-1, high=DEFAULT_GRID_MAX * 1e-1, num=10)
    )
    decay_rate_grid = (
        list(decay_rate_grid)
        if decay_rate_grid is not None
        else [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    )

    best: tuple[float, float, float, float, dict[int, float]] | None = None
    for tau1 in tau_at_lag1_grid:
        for floor in noise_floor_grid:
            if floor >= tau1:
                continue
            for rate in decay_rate_grid:
                tau_by_lag = exponential_lag_threshold(tau1, rate, floor, max_lag)
                f1, _precision, _recall = pooled_f1_for_tau_by_lag(scores_t, labels_t, lags_t, tau_by_lag)
                if best is None or f1 > best[0]:
                    best = (f1, tau1, rate, floor, tau_by_lag)

    if best is None:
        raise ValueError("no valid (tau_at_lag1, noise_floor) combination found in the given grids")
    f1, tau1, rate, floor, tau_by_lag = best
    return ExponentialLagThresholdFit(
        tau_at_lag1=tau1, decay_rate=rate, noise_floor=floor, f1=f1, tau_by_lag=tau_by_lag
    )


# ---------------------------------------------------------------------------
# Unsupervised, anomaly-detection-style thresholds.
#
# Everything above selects tau by sweeping a grid against KNOWN edge/non-edge
# labels (a held-out validation split). The functions below instead treat
# "is this pair an edge" as an unsupervised outlier-detection problem on the
# CMI score distribution itself -- no labels required at selection time (only
# for the post-hoc F1 evaluation callers may still want to do). Useful when a
# validation split with reliable labels isn't available, and as a sanity
# check on whether the validation-swept tau is doing something more
# sophisticated than a generic anomaly-detection cutoff would.
# ---------------------------------------------------------------------------


def mad_threshold(scores: Tensor | Sequence[float], k: float = 3.5) -> float:
    """Modified z-score cutoff: `median(scores) + k * 1.4826 * MAD(scores)`.

    Classic robust outlier-detection threshold (Iglewicz & Hoya, 1993): unlike
    a mean+k*std cutoff, the median/MAD are not themselves dragged upward by
    the (rare, large) true-edge scores we're trying to detect, so this stays
    well-calibrated even though true edges are a small minority of pairs.
    `k=3.5` is the commonly-recommended default for flagging outliers.
    """
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    if scores_t.numel() == 0:
        raise ValueError("scores is empty; cannot compute a MAD threshold.")
    median = scores_t.median()
    mad = (scores_t - median).abs().median()
    return float(median + k * 1.4826 * mad)


def percentile_threshold(scores: Tensor | Sequence[float], quantile: float = 0.95) -> float:
    """Threshold at a fixed quantile of the (pooled, label-free) score
    distribution -- the simplest anomaly-detection baseline, equivalent to
    assuming a fixed "contamination rate" of `1 - quantile` edges among all
    tested pairs, regardless of the true rate.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    if scores_t.numel() == 0:
        raise ValueError("scores is empty; cannot compute a percentile threshold.")
    return float(torch.quantile(scores_t, quantile))


def otsu_threshold(scores: Tensor | Sequence[float]) -> float:
    """Otsu's method (Otsu, 1979): the cut point that maximizes between-class
    variance, treating the score distribution as an (unlabeled) mixture of
    two populations. Computed exactly (no histogram binning) by sweeping
    every sorted-unique candidate split via cumulative sums.

    A standard unsupervised bimodal-histogram threshold, most at home in
    image binarization/anomaly detection -- included here as a check on
    whether the true/non-edge CMI populations are separable by shape alone,
    without ever looking at labels.
    """
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    n = scores_t.numel()
    if n < 2:
        raise ValueError("otsu_threshold needs at least 2 scores.")
    sorted_scores, _ = torch.sort(scores_t)
    cumsum = torch.cumsum(sorted_scores, dim=0)
    total_sum = cumsum[-1]
    counts = torch.arange(1, n, dtype=torch.float32)  # candidate split after index k (1-indexed)
    sum_below = cumsum[:-1]
    w0 = counts / n
    w1 = 1.0 - w0
    mu0 = sum_below / counts
    mu1 = (total_sum - sum_below) / (n - counts)
    between_class_var = w0 * w1 * (mu0 - mu1) ** 2
    best_k = int(torch.argmax(between_class_var).item())
    # Threshold sits between the last "below" sample and the first "above" one.
    return float((sorted_scores[best_k] + sorted_scores[best_k + 1]) / 2.0)


@dataclass
class GMM1DFit:
    """Result of fitting a 2-component 1D Gaussian mixture via EM."""

    weights: tuple[float, float]
    means: tuple[float, float]
    stds: tuple[float, float]
    tau: float

    def summary(self) -> str:
        return (
            f"component 0: weight={self.weights[0]:.3f}, mean={self.means[0]:.4g}, "
            f"std={self.stds[0]:.4g}  |  component 1: weight={self.weights[1]:.3f}, "
            f"mean={self.means[1]:.4g}, std={self.stds[1]:.4g}  |  crossover tau={self.tau:.4g}"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


def gmm_threshold(
    scores: Tensor | Sequence[float], n_iter: int = 100, tol: float = 1e-6, seed: int = 0
) -> GMM1DFit:
    """Fits an unsupervised 2-component 1D Gaussian mixture to `scores` via
    EM, then returns the crossover point where the two components' weighted
    densities are equal (the natural decision boundary between them) as
    `.tau`.

    Initialized by splitting at the median (component 0 = below, component 1
    = above) so, given the heavily right-skewed CMI distributions we see in
    practice (few large true-edge scores, many near-zero non-edge scores),
    EM reliably converges with component 0 tracking the noise floor and
    component 1 tracking the true-edge population.
    """
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    n = scores_t.numel()
    if n < 4:
        raise ValueError("gmm_threshold needs at least 4 scores to fit 2 components.")

    torch.manual_seed(seed)
    median = scores_t.median()
    low_mask = scores_t <= median
    means = torch.stack(
        [
            scores_t[low_mask].mean() if low_mask.any() else scores_t.min(),
            scores_t[~low_mask].mean() if (~low_mask).any() else scores_t.max(),
        ]
    )
    stds = torch.stack(
        [
            scores_t[low_mask].std(unbiased=False).clamp_min(1e-6) if low_mask.any() else torch.tensor(1e-3),
            scores_t[~low_mask].std(unbiased=False).clamp_min(1e-6) if (~low_mask).any() else torch.tensor(1e-3),
        ]
    )
    weights = torch.tensor([0.5, 0.5])

    def _log_gaussian(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)

    prev_ll = None
    for _ in range(n_iter):
        # E-step: responsibilities (posterior component membership) per point.
        log_comp = torch.stack(
            [torch.log(weights[k]) + _log_gaussian(scores_t, means[k], stds[k]) for k in range(2)]
        )
        log_norm = torch.logsumexp(log_comp, dim=0)
        resp = torch.exp(log_comp - log_norm)  # [2, n]

        # M-step: update weights/means/stds from the responsibilities.
        n_k = resp.sum(dim=1).clamp_min(1e-8)
        weights = n_k / n
        means = (resp * scores_t.unsqueeze(0)).sum(dim=1) / n_k
        variances = (resp * (scores_t.unsqueeze(0) - means.unsqueeze(1)) ** 2).sum(dim=1) / n_k
        stds = variances.clamp_min(1e-12).sqrt()

        ll = log_norm.sum().item()
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    lo, hi = (0, 1) if means[0] <= means[1] else (1, 0)
    mu0, mu1 = means[lo].item(), means[hi].item()
    sigma0, sigma1 = stds[lo].item(), stds[hi].item()
    w0, w1 = weights[lo].item(), weights[hi].item()

    # Find the crossover point between the two weighted densities by a fine
    # grid search between the two means (robust to the quadratic having 0 or
    # 2 roots depending on the variances, unlike solving it in closed form).
    grid = torch.linspace(mu0, mu1, steps=2000)
    log_p0 = math.log(max(w0, 1e-12)) + _log_gaussian(grid, torch.tensor(mu0), torch.tensor(sigma0))
    log_p1 = math.log(max(w1, 1e-12)) + _log_gaussian(grid, torch.tensor(mu1), torch.tensor(sigma1))
    diff = log_p0 - log_p1
    sign_change = (diff[:-1] * diff[1:]) <= 0
    if sign_change.any():
        idx = int(torch.nonzero(sign_change, as_tuple=False)[0].item())
        tau = float((grid[idx] + grid[idx + 1]) / 2.0)
    else:
        # Components didn't separate (e.g. near-identical); fall back to the
        # midpoint between the two means.
        tau = (mu0 + mu1) / 2.0

    return GMM1DFit(weights=(w0, w1), means=(mu0, mu1), stds=(sigma0, sigma1), tau=tau)


# ---------------------------------------------------------------------------
# AdaptiveThreshold: a single configurable, label-free tau(lag) construction.
#
# `mad_threshold`/`percentile_threshold`/`otsu_threshold`/`gmm_threshold`
# above are all label-free *point* estimators (one scalar tau from one score
# population). The natural way to use one across multiple lags is either
# (a) refit it independently at EVERY lag ("per_lag=True" below), or (b) fit
# it once, globally, and reuse that single value everywhere. (a) tends to
# perform poorly in practice: deeper lags have far fewer (cause, effect)
# pairs pooled across a batch of sequences, so an independent per-lag fit is
# fit on a much noisier, smaller sample -- see
# `scripts/train_and_test_lagged_effects.py`'s "otsu" (per-lag) vs.
# "otsu_global" (single fit) comparison, where per-lag Otsu cost ~15-18 F1
# points relative to a single global fit.
#
# `AdaptiveThreshold` generalizes (b) with a middle ground informed by the
# same lag-decay shape `fit_exponential_lag_threshold`/
# `fit_power_law_lag_threshold` already use for LABELED validation-swept
# thresholds: anchor tau at lag=1 (usually the best-populated, highest
# signal-to-noise lag) using `method`, then decay that anchor toward a floor
# as lag increases, entirely without labels.
# ---------------------------------------------------------------------------

_UNSUPERVISED_METHOD_FNS: dict[str, Callable[[Tensor], float]] = {
    "mad": mad_threshold,
    "percentile": percentile_threshold,
    "otsu": otsu_threshold,
    "gmm": lambda scores: gmm_threshold(scores).tau,
}


@dataclass
class AdaptiveThreshold:
    """Configurable, label-free tau(lag) construction.

    Args:
        method: base unsupervised cutoff used to anchor tau -- one of
            "otsu" (default), "mad", "percentile", "gmm".
        per_lag: if True, independently refits `method` at EVERY lag
            (noisy for deep lags with few pooled pairs -- see module note
            above). If False (default), a single lag=1 anchor is used,
            optionally decayed via `decay`/`decay_type`.
        decay: if True (default), decays the lag=1 anchor toward `floor` as
            lag increases, instead of reusing one tau at every lag. Ignored
            if `per_lag=True`.
        decay_type: "exponential" (default, `tau(lag) = floor + (tau1 -
            floor) * exp(-decay_rate * (lag - 1))`), "power_law" (`tau(lag)
            = floor + (tau1 - floor) * lag ** -exponent`), or "none" (reuse
            tau1 unchanged at every lag -- equivalent to `decay=False`).
        decay_rate: shape parameter for `decay_type="exponential"`. Not fit
            from data (no labels available here) -- a tunable knob; default
            0.3 matches the lag-decay rate used throughout this project's
            own synthetic DGPs (`NonlinearSCM(decay_rate=...)`).
        exponent: shape parameter for `decay_type="power_law"`.
        floor: absolute floor value the decay approaches. If None (default),
            estimated unsupervised as `method` applied to the deepest
            available lag's own score population (falls back to `10%` of
            the lag=1 anchor if that lag has too few samples).
        min_group_size: minimum pooled samples a lag needs before fitting
            `method` on it directly; below this, falls back to the whole
            population's fit (mirrors `select_thresholds_by_group`'s
            fallback semantics).
    """

    method: str = "otsu"
    per_lag: bool = False
    decay: bool = True
    decay_type: str = "exponential"
    decay_rate: float = 0.3
    exponent: float = 0.5
    floor: float | None = None
    min_group_size: int = 8

    def __post_init__(self) -> None:
        if self.method not in _UNSUPERVISED_METHOD_FNS:
            raise ValueError(
                f"method must be one of {sorted(_UNSUPERVISED_METHOD_FNS)}, got {self.method!r}"
            )
        if self.decay_type not in ("exponential", "power_law", "none"):
            raise ValueError(
                f"decay_type must be 'exponential', 'power_law', or 'none', got {self.decay_type!r}"
            )

    def tau_by_lag(
        self, scores: Tensor | Sequence[float], lags: Tensor | Sequence[int], max_lag: int
    ) -> dict[int, float]:
        """Returns `{lag: tau}` for `lag` in `1..max_lag`, from pooled,
        label-free `scores` and their matching per-pair `lags`."""
        fn = _UNSUPERVISED_METHOD_FNS[self.method]
        scores_t = torch.as_tensor(scores, dtype=torch.float32)
        lags_t = torch.as_tensor(lags)
        global_tau = fn(scores_t)

        if self.per_lag:
            tau_by_lag = {}
            for lag in range(1, max_lag + 1):
                group = scores_t[lags_t == lag]
                tau_by_lag[lag] = (
                    fn(group) if group.numel() >= self.min_group_size else global_tau
                )
            return tau_by_lag

        lag1_group = scores_t[lags_t == 1]
        tau1 = fn(lag1_group) if lag1_group.numel() >= self.min_group_size else global_tau

        if not self.decay or self.decay_type == "none":
            return {lag: tau1 for lag in range(1, max_lag + 1)}

        floor = self.floor
        if floor is None:
            deep_group = scores_t[lags_t == max_lag]
            floor = fn(deep_group) if deep_group.numel() >= self.min_group_size else 0.1 * tau1

        tau_by_lag = {}
        for lag in range(1, max_lag + 1):
            if self.decay_type == "exponential":
                tau_by_lag[lag] = floor + (tau1 - floor) * math.exp(-self.decay_rate * (lag - 1))
            else:  # "power_law"
                tau_by_lag[lag] = floor + (tau1 - floor) * (lag ** -self.exponent)
        return tau_by_lag


@dataclass
class PowerLawLagThresholdFit:
    """Result of `fit_power_law_lag_threshold`: a 3-parameter, SUB-LINEAR
    `tau(lag) = noise_floor + (tau_at_lag1 - noise_floor) * lag ** (-exponent)`
    curve (as opposed to `ExponentialLagThresholdFit`'s exponential decay),
    motivated by Chadyuk et al.'s (2026) follow-up finding that the analogous
    scale (their tau* as a function of vocabulary size |X|) decays
    SUB-LINEARLY (a fitted power law with exponent b ~ -0.5 to -0.56 across
    seeds) rather than as the originally-printed C/|X| (exponent -1) or a
    fixed constant (exponent 0).
    """

    tau_at_lag1: float
    exponent: float
    noise_floor: float
    f1: float
    tau_by_lag: dict[int, float]

    def summary(self) -> str:
        return (
            f"tau(lag) = {self.noise_floor:.4g} + "
            f"({self.tau_at_lag1:.4g} - {self.noise_floor:.4g}) * lag^(-{self.exponent:.3g})"
            f"  (pooled F1={self.f1:.3f})"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


def power_law_lag_threshold(
    tau_at_lag1: float, exponent: float, noise_floor: float, max_lag: int
) -> dict[int, float]:
    """Builds a per-lag threshold that decays as a SUB-LINEAR power law from
    `tau_at_lag1` (at lag=1, where `lag ** (-exponent) == 1` for any
    exponent) down toward `noise_floor`:

        tau(lag) = noise_floor + (tau_at_lag1 - noise_floor) * lag ** (-exponent)

    `exponent=1` recovers a `~1/lag` (harmonic) decay; `exponent < 1` decays
    SLOWER than that (the "sub-linear" regime Chadyuk et al., 2026 found for
    the analogous vocabulary-size scaling); `exponent=0` recovers a flat
    threshold. See `exponential_lag_threshold` for the alternative
    exponential-decay parametrization.
    """
    if noise_floor < 0:
        raise ValueError(f"noise_floor must be >= 0, got {noise_floor}")
    if tau_at_lag1 < noise_floor:
        raise ValueError(
            f"tau_at_lag1 ({tau_at_lag1}) must be >= noise_floor ({noise_floor})."
        )
    if exponent < 0:
        raise ValueError(f"exponent must be >= 0, got {exponent}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    return {
        lag: noise_floor + (tau_at_lag1 - noise_floor) * (lag ** (-exponent))
        for lag in range(1, max_lag + 1)
    }


def fit_power_law_lag_threshold(
    cmi_scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[bool],
    lags: Tensor | Sequence[int],
    max_lag: int | None = None,
    tau_at_lag1_grid: Iterable[float] | None = None,
    noise_floor_grid: Iterable[float] | None = None,
    exponent_grid: Iterable[float] | None = None,
) -> PowerLawLagThresholdFit:
    """Grid-searches the 3 parameters of `power_law_lag_threshold` to
    maximize pooled F1 across ALL lags at once -- the SUB-LINEAR-decay
    counterpart of `fit_exponential_lag_threshold`. See that function's
    docstring for the general approach; the only difference is the
    functional form and the swept parameter (`exponent` instead of
    `decay_rate`).

    Args:
        exponent_grid: candidate power-law exponents. Defaults to
            `[0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0]`,
            deliberately dense around the `~0.5` sub-linear regime Chadyuk
            et al. (2026) found for their analogous |X|-scaling law.
    """
    scores_t = torch.as_tensor(cmi_scores, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.bool)
    lags_t = torch.as_tensor(lags, dtype=torch.long)
    if not (scores_t.shape == labels_t.shape == lags_t.shape):
        raise ValueError(
            "cmi_scores, labels, and lags must all have the same shape, got "
            f"{tuple(scores_t.shape)}, {tuple(labels_t.shape)}, {tuple(lags_t.shape)}"
        )
    if scores_t.numel() == 0:
        raise ValueError("cmi_scores is empty; cannot fit a threshold.")

    max_lag = max_lag if max_lag is not None else int(lags_t.max().item())
    tau_at_lag1_grid = list(tau_at_lag1_grid) if tau_at_lag1_grid is not None else make_log_grid()
    noise_floor_grid = (
        list(noise_floor_grid)
        if noise_floor_grid is not None
        else make_log_grid(low=DEFAULT_GRID_MIN * 1e-1, high=DEFAULT_GRID_MAX * 1e-1, num=10)
    )
    exponent_grid = (
        list(exponent_grid)
        if exponent_grid is not None
        else [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0]
    )

    best: tuple[float, float, float, float, dict[int, float]] | None = None
    for tau1 in tau_at_lag1_grid:
        for floor in noise_floor_grid:
            if floor >= tau1:
                continue
            for exponent in exponent_grid:
                tau_by_lag = power_law_lag_threshold(tau1, exponent, floor, max_lag)
                f1, _precision, _recall = pooled_f1_for_tau_by_lag(scores_t, labels_t, lags_t, tau_by_lag)
                if best is None or f1 > best[0]:
                    best = (f1, tau1, exponent, floor, tau_by_lag)

    if best is None:
        raise ValueError("no valid (tau_at_lag1, noise_floor) combination found in the given grids")
    f1, tau1, exponent, floor, tau_by_lag = best
    return PowerLawLagThresholdFit(
        tau_at_lag1=tau1, exponent=exponent, noise_floor=floor, f1=f1, tau_by_lag=tau_by_lag
    )


# ---------------------------------------------------------------------------
# Joint lag x vocabulary-size threshold law.
#
# `fit_exponential_lag_threshold`/`fit_power_law_lag_threshold` above each
# calibrate tau(lag) for ONE dataset-generating process (one vocabulary size,
# one lag-decay rate, etc). Chadyuk et al.'s (2026) follow-up note found an
# analogous, SEPARATE decay of tau* as a function of vocabulary size |X|
# (sub-linear, exponent ~0.5-0.56, not the originally-printed C/|X|), and
# reported that the true-edge CMI signal itself decays with |X| at a
# shallower rate (~|X|^-0.3) than the non-edge/noise floor (~|X|^-1.85).
# `fit_joint_lag_vocab_threshold` combines both axes into one anchored,
# multiplicative-plus-additive law fit jointly across MANY (DGP, lag) pairs
# at once, rather than one bespoke curve per DGP:
#
#   tau(lag, |X|) = signal_scale * |X|^-vocab_exponent_signal * lag^-lag_exponent
#                   + floor_scale * |X|^-vocab_exponent_floor
#
# The two vocabulary-size exponents are deliberately kept SEPARATE (rather
# than reusing one exponent for both terms) because Chadyuk et al. found the
# signal and the noise floor decay with |X| at markedly different rates.
# ---------------------------------------------------------------------------


@dataclass
class JointLagVocabThresholdFit:
    """Result of `fit_joint_lag_vocab_threshold`: a 5-parameter threshold law
    jointly anchored across BOTH the lag axis and the vocabulary-size axis,
    fit across pooled data from many data-generating processes at once (see
    module docstring section above for the functional form and motivation).
    """

    signal_scale: float
    vocab_exponent_signal: float
    lag_exponent: float
    floor_scale: float
    vocab_exponent_floor: float
    f1: float

    def tau(self, lag: int | Tensor, vocab_size: int | Tensor) -> float | Tensor:
        """Evaluates the fitted law at a given (lag, vocab_size)."""
        return (
            self.signal_scale * (vocab_size ** (-self.vocab_exponent_signal))
            * (lag ** (-self.lag_exponent))
            + self.floor_scale * (vocab_size ** (-self.vocab_exponent_floor))
        )

    def summary(self) -> str:
        return (
            f"tau(lag, |X|) = {self.signal_scale:.4g} * |X|^-{self.vocab_exponent_signal:.3g} "
            f"* lag^-{self.lag_exponent:.3g} + {self.floor_scale:.4g} * "
            f"|X|^-{self.vocab_exponent_floor:.3g}  (pooled F1={self.f1:.3f})"
        )

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()


def _f1_for_joint_formula(
    scores: Tensor, labels: Tensor, lags: Tensor, vocab_sizes: Tensor,
    signal_scale: float, vocab_exponent_signal: float, lag_exponent: float,
    floor_scale: float, vocab_exponent_floor: float,
) -> float:
    tau = (
        signal_scale * (vocab_sizes.float() ** (-vocab_exponent_signal))
        * (lags.float() ** (-lag_exponent))
        + floor_scale * (vocab_sizes.float() ** (-vocab_exponent_floor))
    )
    preds = scores >= tau
    tp = int((preds & labels).sum().item())
    fp = int((preds & ~labels).sum().item())
    fn = int((~preds & labels).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def fit_joint_lag_vocab_threshold(
    cmi_scores: Tensor | Sequence[float],
    labels: Tensor | Sequence[bool],
    lags: Tensor | Sequence[int],
    vocab_sizes: Tensor | Sequence[int],
    signal_scale_grid: Iterable[float] | None = None,
    vocab_exponent_signal_grid: Iterable[float] | None = None,
    lag_exponent_grid: Iterable[float] | None = None,
    floor_scale_grid: Iterable[float] | None = None,
    vocab_exponent_floor_grid: Iterable[float] | None = None,
) -> JointLagVocabThresholdFit:
    """Grid-searches the 5 parameters of the joint lag x vocab-size law to
    maximize pooled F1 across every (DGP, lag) pair supplied at once --
    pool data from as many data-generating processes (vocab sizes, lag-decay
    rates, etc.) as you have available; each pair's own `vocab_sizes[i]`
    lets a single call calibrate one law that spans all of them, instead of
    one bespoke curve per DGP that may not transfer to an unseen setup.

    Args:
        cmi_scores, labels, lags: as in `fit_exponential_lag_threshold`.
        vocab_sizes: aligned per-pair vocabulary size `|X|` of the DGP that
            pair came from.
        signal_scale_grid: candidate `signal_scale` values. Defaults to
            `make_log_grid()`.
        vocab_exponent_signal_grid: candidate exponents for the signal term's
            vocab-size decay. Defaults to `[0.0, 0.15, 0.3, 0.5, 0.7, 1.0]`,
            centered on Chadyuk et al.'s (2026) reported ~0.3 for the
            true-edge CMI's own |X|-decay.
        lag_exponent_grid: candidate lag-decay exponents. Defaults to
            `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`.
        floor_scale_grid: candidate `floor_scale` values. Defaults to a
            log-spaced grid an order of magnitude below `signal_scale_grid`.
        vocab_exponent_floor_grid: candidate exponents for the floor term's
            (steeper) vocab-size decay. Defaults to
            `[0.5, 1.0, 1.5, 1.85, 2.0, 2.5]`, centered on Chadyuk et al.'s
            (2026) reported ~1.85 for the non-edge/noise floor's |X|-decay.

    Returns:
        The best-pooled-F1 `JointLagVocabThresholdFit`.
    """
    scores_t = torch.as_tensor(cmi_scores, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.bool)
    lags_t = torch.as_tensor(lags, dtype=torch.float32)
    vocab_t = torch.as_tensor(vocab_sizes, dtype=torch.float32)
    if not (scores_t.shape == labels_t.shape == lags_t.shape == vocab_t.shape):
        raise ValueError(
            "cmi_scores, labels, lags, and vocab_sizes must all have the same shape, got "
            f"{tuple(scores_t.shape)}, {tuple(labels_t.shape)}, {tuple(lags_t.shape)}, "
            f"{tuple(vocab_t.shape)}"
        )
    if scores_t.numel() == 0:
        raise ValueError("cmi_scores is empty; cannot fit a threshold.")

    signal_scale_grid = list(signal_scale_grid) if signal_scale_grid is not None else make_log_grid()
    vocab_exponent_signal_grid = (
        list(vocab_exponent_signal_grid)
        if vocab_exponent_signal_grid is not None
        else [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
    )
    lag_exponent_grid = (
        list(lag_exponent_grid) if lag_exponent_grid is not None else [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    )
    floor_scale_grid = (
        list(floor_scale_grid)
        if floor_scale_grid is not None
        else make_log_grid(low=DEFAULT_GRID_MIN * 1e-1, high=DEFAULT_GRID_MAX * 1e-1, num=8)
    )
    vocab_exponent_floor_grid = (
        list(vocab_exponent_floor_grid)
        if vocab_exponent_floor_grid is not None
        else [0.5, 1.0, 1.5, 1.85, 2.0, 2.5]
    )

    best: tuple[float, float, float, float, float, float] | None = None
    for signal_scale in signal_scale_grid:
        for p in vocab_exponent_signal_grid:
            for b in lag_exponent_grid:
                for floor_scale in floor_scale_grid:
                    if floor_scale >= signal_scale:
                        continue
                    for q in vocab_exponent_floor_grid:
                        f1 = _f1_for_joint_formula(
                            scores_t, labels_t, lags_t, vocab_t, signal_scale, p, b, floor_scale, q
                        )
                        if best is None or f1 > best[0]:
                            best = (f1, signal_scale, p, b, floor_scale, q)

    if best is None:
        raise ValueError("no valid parameter combination found in the given grids")
    f1, signal_scale, p, b, floor_scale, q = best
    return JointLagVocabThresholdFit(
        signal_scale=signal_scale, vocab_exponent_signal=p, lag_exponent=b,
        floor_scale=floor_scale, vocab_exponent_floor=q, f1=f1,
    )


def resolve_threshold(
    threshold_config: Mapping[str, object],
    cmi_scores: Tensor | Sequence[float] | None = None,
    labels: Tensor | Sequence[bool] | None = None,
    delta: float | None = None,
) -> float | ThresholdSelectionResult:
    """Resolves a `params["threshold"]`-style config into a concrete tau.

    Supported `threshold_config["type"]` values:
        - "validation_sweep" (default/recommended): calls
          `select_threshold_by_validation` using `cmi_scores`/`labels` from a
          held-out split. `threshold_config` may additionally provide
          "grid" and "hardcoded_defaults" to forward to that call. Returns
          the full `ThresholdSelectionResult` (access `.tau` for the scalar).
        - "static": returns `threshold_config["value"]` unchanged, but emits
          a warning, since Chadyuk et al. (2026) found fixed constants do not
          transfer across generator/backbone setups. Kept only as an
          explicit opt-out/override for reproducing a specific prior run.

    This makes "select on validation data" the default without removing the
    ability to pin a specific fixed value when that is genuinely desired
    (e.g. to exactly reproduce a previously-published number).
    """
    ttype = threshold_config.get("type", "validation_sweep")

    if ttype == "static":
        warnings.warn(
            "threshold config uses type='static' (a hardcoded tau). Chadyuk et al. (2026) "
            "found fixed thresholds do not transfer across generator/backbone setups -- "
            "prefer type='validation_sweep' (the default) unless you are intentionally "
            "reproducing a specific prior configuration.",
            stacklevel=2,
        )
        return threshold_config["value"]  # type: ignore[return-value]

    if ttype == "validation_sweep":
        if cmi_scores is None or labels is None:
            raise ValueError(
                "threshold type='validation_sweep' requires `cmi_scores` and `labels` from a "
                "held-out validation split."
            )
        grid = threshold_config.get("grid")
        return select_threshold_by_validation(
            cmi_scores,
            labels,
            grid=grid,  # type: ignore[arg-type]
            delta=threshold_config.get("delta", delta),  # type: ignore[arg-type]
            hardcoded_defaults=threshold_config.get("hardcoded_defaults"),  # type: ignore[arg-type]
        )

    raise ValueError(f"Unknown threshold config type: {ttype!r}")
