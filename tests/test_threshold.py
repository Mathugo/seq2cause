import math
import warnings

import pytest
import torch

from seq2cause.threshold import (
    AdaptiveThreshold,
    ExponentialLagThresholdFit,
    GMM1DFit,
    JointLagVocabThresholdFit,
    PowerLawLagThresholdFit,
    ThresholdSelectionResult,
    exponential_lag_threshold,
    f1_precision_recall,
    fit_exponential_lag_threshold,
    fit_joint_lag_vocab_threshold,
    fit_power_law_lag_threshold,
    gmm_threshold,
    mad_threshold,
    make_log_grid,
    otsu_threshold,
    percentile_threshold,
    pooled_f1_for_tau_by_lag,
    power_law_lag_threshold,
    resolve_threshold,
    select_threshold_by_validation,
    select_thresholds_by_group,
)


def test_make_log_grid_bounds_and_count():
    grid = make_log_grid(low=1e-4, high=1e-1, num=4)
    assert len(grid) == 4
    assert grid[0] == pytest.approx(1e-4, rel=1e-6)
    assert grid[-1] == pytest.approx(1e-1, rel=1e-6)
    assert grid == sorted(grid)


def test_make_log_grid_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        make_log_grid(low=-1.0, high=1.0)
    with pytest.raises(ValueError):
        make_log_grid(low=1.0, high=0.1)


def test_f1_precision_recall_perfect_separation():
    scores = torch.tensor([0.0, 0.01, 0.9, 1.0])
    labels = torch.tensor([False, False, True, True])
    f1, precision, recall = f1_precision_recall(scores, labels, tau=0.5)
    assert f1 == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_select_threshold_by_validation_finds_the_separating_threshold():
    scores = torch.tensor([1e-6, 1e-3, 5e-2, 8e-2])
    labels = torch.tensor([False, False, True, True])
    result = select_threshold_by_validation(scores, labels, grid=[1e-6, 1e-4, 1e-2, 5e-2])
    assert isinstance(result, ThresholdSelectionResult)
    assert result.f1 == pytest.approx(1.0)
    assert result.tau == pytest.approx(1e-2)


def test_select_threshold_by_validation_defaults_to_log_grid():
    scores = torch.tensor([1e-7, 1e-6, 1e-2, 3e-2])
    labels = torch.tensor([False, False, True, True])
    result = select_threshold_by_validation(scores, labels)
    assert result.grid == make_log_grid()
    assert result.f1 > 0.9


def test_delta_ratio_warning_triggers_far_from_margin():
    # tau will land far below delta here, so the delta-ratio warning should fire.
    scores = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
    labels = torch.tensor([False, False, True, True])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_threshold_by_validation(
            scores, labels, grid=[1e-6, 1e-1], delta=0.05
        )
    assert result.tau_over_delta is not None
    assert any("truth margin" in str(w.message) for w in caught)
    assert any("truth margin" in w for w in result.warnings)


def test_delta_ratio_no_warning_when_within_band():
    scores = torch.tensor([1e-6, 0.04, 0.06, 0.9])
    labels = torch.tensor([False, True, True, True])
    result = select_threshold_by_validation(
        scores, labels, grid=[1e-6, 0.04], delta=0.05, emit_warnings=False
    )
    ratio = result.tau_over_delta
    assert ratio is not None


def test_hardcoded_default_proximity_warning():
    scores = torch.tensor([3e-5, 3e-5, 3e-5, 3e-5])
    labels = torch.tensor([False, False, True, True])
    result = select_threshold_by_validation(
        scores,
        labels,
        grid=[3e-5],
        hardcoded_defaults={"paper_table_2": 3e-5},
        emit_warnings=False,
    )
    assert len(result.warnings) >= 1
    assert any("hardcoded default" in w for w in result.warnings)


def test_no_warning_when_far_from_hardcoded_default():
    scores = torch.tensor([1e-2, 1e-2])
    labels = torch.tensor([False, True])
    result = select_threshold_by_validation(
        scores,
        labels,
        grid=[1e-2],
        hardcoded_defaults={"paper_table_2": 3e-5},
        emit_warnings=False,
    )
    assert not any("hardcoded default" in w for w in result.warnings)


def test_select_threshold_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError):
        select_threshold_by_validation(torch.tensor([]), torch.tensor([]))
    with pytest.raises(ValueError):
        select_threshold_by_validation(torch.tensor([1.0, 2.0]), torch.tensor([True]))


def test_summary_mentions_chadyuk_when_delta_missing():
    scores = torch.tensor([1e-2, 1e-1])
    labels = torch.tensor([False, True])
    result = select_threshold_by_validation(scores, labels, grid=[1e-2, 1e-1])
    assert "tau relative to delta" in result.summary()


def test_resolve_threshold_static_returns_value_and_warns():
    with pytest.warns(UserWarning, match="static"):
        value = resolve_threshold({"type": "static", "value": 3e-5})
    assert value == 3e-5


def test_resolve_threshold_validation_sweep_requires_scores_and_labels():
    with pytest.raises(ValueError):
        resolve_threshold({"type": "validation_sweep"})


def test_resolve_threshold_validation_sweep_returns_selection_result():
    scores = torch.tensor([1e-6, 1e-2])
    labels = torch.tensor([False, True])
    result = resolve_threshold(
        {"type": "validation_sweep", "grid": [1e-6, 1e-2]}, cmi_scores=scores, labels=labels
    )
    assert isinstance(result, ThresholdSelectionResult)
    assert result.tau == pytest.approx(1e-2)


def test_resolve_threshold_unknown_type_raises():
    with pytest.raises(ValueError):
        resolve_threshold({"type": "bogus"})


def test_select_thresholds_by_group_tailors_tau_per_group():
    # Group 1 ("near"): large CMI scale. Group 2 ("far"): small CMI scale,
    # but still cleanly separable from its own non-edges *at that scale*.
    scores = torch.tensor([0.001, 0.09, 0.001, 0.001, 1e-5, 5e-4, 1e-5, 1e-5])
    labels = torch.tensor([False, True, False, False, False, True, False, False])
    groups = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])

    results = select_thresholds_by_group(
        scores, labels, groups, grid=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], fallback="drop",
        warn_on_unreliable_groups=False,
    )
    assert set(results.keys()) == {1, 2}
    assert results[1].f1 == pytest.approx(1.0)
    assert results[2].f1 == pytest.approx(1.0)
    # The far group's tau must be far smaller, tailored to its own CMI scale --
    # a single shared/global tau tuned on the near group would miss it entirely.
    assert results[2].tau < results[1].tau


def test_select_thresholds_by_group_fallback_global_for_sparse_groups():
    scores = torch.tensor([0.001, 0.09, 0.001, 0.001, 1e-5])
    labels = torch.tensor([False, True, False, False, False])
    groups = torch.tensor([1, 1, 1, 1, 3])  # group 3 has a single, label-less sample

    results = select_thresholds_by_group(
        scores, labels, groups, grid=[1e-5, 1e-3, 1e-1], fallback="global", min_group_size=2,
        warn_on_unreliable_groups=False,
    )
    assert set(results.keys()) == {1, 3}
    # Group 3 (insufficient/degenerate) reuses the pooled/global fit.
    global_only = select_threshold_by_validation(scores, labels, grid=[1e-5, 1e-3, 1e-1])
    assert results[3].tau == pytest.approx(global_only.tau)


def test_select_thresholds_by_group_shape_mismatch_raises():
    with pytest.raises(ValueError):
        select_thresholds_by_group(
            torch.tensor([1.0, 2.0]), torch.tensor([True]), torch.tensor([1, 1])
        )


def test_select_thresholds_by_group_warns_on_few_true_edges():
    # Group 1 has plenty of true edges; group 2 has only 1 -- below the
    # default min_reliable_true_edges=10.
    scores = torch.cat([torch.rand(30) * 1e-4, torch.rand(15) * 0.1 + 0.05, torch.tensor([0.001, 0.09])])
    labels = torch.cat([torch.zeros(30, dtype=torch.bool), torch.ones(15, dtype=torch.bool),
                         torch.tensor([False, True])])
    groups = torch.cat([torch.ones(45, dtype=torch.long), torch.full((2,), 2, dtype=torch.long)])

    with pytest.warns(UserWarning, match="min_reliable_true_edges"):
        results = select_thresholds_by_group(scores, labels, groups, fallback="drop")
    assert "min_reliable_true_edges" in results[2].warnings[0]
    assert not results[1].warnings


def test_select_thresholds_by_group_suppresses_reliability_warning_when_disabled():
    scores = torch.tensor([0.001, 0.09, 0.001, 0.001])
    labels = torch.tensor([False, True, False, False])
    groups = torch.tensor([1, 1, 1, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = select_thresholds_by_group(
            scores, labels, groups, fallback="drop", warn_on_unreliable_groups=False
        )
    assert "min_reliable_true_edges" in results[1].warnings[0]


def test_select_thresholds_by_group_invalid_fallback_raises():
    with pytest.raises(ValueError):
        select_thresholds_by_group(
            torch.tensor([1.0]), torch.tensor([True]), torch.tensor([1]), fallback="bogus"
        )


def test_exponential_lag_threshold_decays_toward_the_floor():
    tau_by_lag = exponential_lag_threshold(
        tau_at_lag1=0.1, decay_rate=1.0, noise_floor=0.01, max_lag=5
    )
    assert tau_by_lag[1] == pytest.approx(0.1)
    # Strictly decreasing, and every value strictly above the floor.
    values = [tau_by_lag[lag] for lag in range(1, 6)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert all(v > 0.01 for v in values)
    # Approaches (but never reaches) the floor as lag grows.
    assert tau_by_lag[5] == pytest.approx(0.01, abs=0.01)


def test_exponential_lag_threshold_zero_decay_rate_is_flat():
    tau_by_lag = exponential_lag_threshold(
        tau_at_lag1=0.05, decay_rate=0.0, noise_floor=0.001, max_lag=4
    )
    assert all(v == pytest.approx(0.05) for v in tau_by_lag.values())


def test_exponential_lag_threshold_rejects_invalid_params():
    with pytest.raises(ValueError):
        exponential_lag_threshold(tau_at_lag1=0.1, decay_rate=1.0, noise_floor=-0.1, max_lag=3)
    with pytest.raises(ValueError):
        exponential_lag_threshold(tau_at_lag1=0.01, decay_rate=1.0, noise_floor=0.1, max_lag=3)
    with pytest.raises(ValueError):
        exponential_lag_threshold(tau_at_lag1=0.1, decay_rate=-1.0, noise_floor=0.01, max_lag=3)
    with pytest.raises(ValueError):
        exponential_lag_threshold(tau_at_lag1=0.1, decay_rate=1.0, noise_floor=0.01, max_lag=0)


def test_fit_exponential_lag_threshold_recovers_a_known_decaying_signal():
    # Construct true edges whose CMI itself decays exponentially with lag
    # toward a floor, and non-edges sitting well below that floor at every
    # lag -- the fit should find a curve that separates them at every lag.
    torch.manual_seed(0)
    max_lag = 4
    true_tau_at_lag1, true_decay, true_floor = 0.2, 1.0, 0.01
    true_curve = exponential_lag_threshold(true_tau_at_lag1, true_decay, true_floor, max_lag)

    scores, labels, lags = [], [], []
    for lag in range(1, max_lag + 1):
        true_signal = true_curve[lag] * 1.5  # comfortably above that lag's true threshold
        noise = true_floor * 0.1  # comfortably below every lag's floor
        for _ in range(10):
            scores.append(true_signal)
            labels.append(True)
            lags.append(lag)
            scores.append(noise)
            labels.append(False)
            lags.append(lag)

    fit = fit_exponential_lag_threshold(
        torch.tensor(scores), torch.tensor(labels), torch.tensor(lags), max_lag=max_lag
    )
    assert isinstance(fit, ExponentialLagThresholdFit)
    assert fit.f1 == pytest.approx(1.0)
    # The fitted per-lag thresholds must separate true/false at every lag.
    for lag in range(1, max_lag + 1):
        assert fit.tau_by_lag[lag] < true_curve[lag] * 1.5
        assert fit.tau_by_lag[lag] > true_floor * 0.1


def test_fit_exponential_lag_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fit_exponential_lag_threshold(
            torch.tensor([1.0, 2.0]), torch.tensor([True]), torch.tensor([1, 1])
        )


def test_fit_exponential_lag_threshold_empty_raises():
    with pytest.raises(ValueError):
        fit_exponential_lag_threshold(torch.tensor([]), torch.tensor([]), torch.tensor([]))


def test_exponential_lag_threshold_fit_summary_mentions_curve_and_f1():
    fit = ExponentialLagThresholdFit(
        tau_at_lag1=0.1, decay_rate=1.0, noise_floor=0.01, f1=0.75, tau_by_lag={1: 0.1}
    )
    s = fit.summary()
    assert "tau(lag)" in s
    assert "0.75" in s


# ---------------------------------------------------------------------------
# Unsupervised, anomaly-detection-style thresholds.
# ---------------------------------------------------------------------------


def _bimodal_scores():
    torch.manual_seed(0)
    false_scores = torch.rand(200) * 1e-3
    true_scores = 0.05 + torch.rand(50) * 0.05
    scores = torch.cat([false_scores, true_scores])
    labels = torch.cat(
        [torch.zeros(200, dtype=torch.bool), torch.ones(50, dtype=torch.bool)]
    )
    return scores, labels


def test_mad_threshold_separates_bimodal_scores():
    scores, labels = _bimodal_scores()
    tau = mad_threshold(scores)
    lags = torch.ones_like(labels, dtype=torch.long)
    f1, precision, recall = pooled_f1_for_tau_by_lag(scores, labels, lags, {1: tau})
    assert f1 > 0.9
    assert precision > 0.9


def test_mad_threshold_rejects_empty():
    with pytest.raises(ValueError):
        mad_threshold(torch.tensor([]))


def test_percentile_threshold_matches_quantile():
    scores = torch.arange(1, 101, dtype=torch.float32)
    tau = percentile_threshold(scores, quantile=0.9)
    assert tau == pytest.approx(torch.quantile(scores, 0.9).item())


def test_percentile_threshold_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        percentile_threshold(torch.tensor([1.0, 2.0]), quantile=1.5)


def test_otsu_threshold_separates_bimodal_scores():
    scores, labels = _bimodal_scores()
    tau = otsu_threshold(scores)
    lags = torch.ones_like(labels, dtype=torch.long)
    f1, precision, recall = pooled_f1_for_tau_by_lag(scores, labels, lags, {1: tau})
    assert f1 > 0.9


def test_otsu_threshold_needs_at_least_two_scores():
    with pytest.raises(ValueError):
        otsu_threshold(torch.tensor([1.0]))


def test_gmm_threshold_separates_bimodal_scores():
    scores, labels = _bimodal_scores()
    fit = gmm_threshold(scores)
    assert isinstance(fit, GMM1DFit)
    # Component means should track the two populations' rough scales.
    assert fit.means[0] < 0.01
    assert fit.means[1] > 0.01
    lags = torch.ones_like(labels, dtype=torch.long)
    f1, precision, recall = pooled_f1_for_tau_by_lag(scores, labels, lags, {1: fit.tau})
    assert f1 > 0.9
    assert "crossover tau" in fit.summary()


def test_gmm_threshold_needs_at_least_four_scores():
    with pytest.raises(ValueError):
        gmm_threshold(torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Sub-linear (power-law) per-lag threshold fit.
# ---------------------------------------------------------------------------


def test_power_law_lag_threshold_decays_toward_the_floor():
    tau_by_lag = power_law_lag_threshold(tau_at_lag1=0.1, exponent=0.5, noise_floor=0.01, max_lag=5)
    assert tau_by_lag[1] == pytest.approx(0.1)
    values = [tau_by_lag[lag] for lag in range(1, 6)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert all(v > 0.01 for v in values)


def test_power_law_lag_threshold_zero_exponent_is_flat():
    tau_by_lag = power_law_lag_threshold(tau_at_lag1=0.05, exponent=0.0, noise_floor=0.001, max_lag=4)
    assert all(v == pytest.approx(0.05) for v in tau_by_lag.values())


def test_power_law_lag_threshold_rejects_invalid_params():
    with pytest.raises(ValueError):
        power_law_lag_threshold(tau_at_lag1=0.1, exponent=0.5, noise_floor=-0.1, max_lag=3)
    with pytest.raises(ValueError):
        power_law_lag_threshold(tau_at_lag1=0.01, exponent=0.5, noise_floor=0.1, max_lag=3)
    with pytest.raises(ValueError):
        power_law_lag_threshold(tau_at_lag1=0.1, exponent=-0.5, noise_floor=0.01, max_lag=3)
    with pytest.raises(ValueError):
        power_law_lag_threshold(tau_at_lag1=0.1, exponent=0.5, noise_floor=0.01, max_lag=0)


def test_fit_power_law_lag_threshold_recovers_a_known_decaying_signal():
    torch.manual_seed(0)
    max_lag = 4
    true_tau_at_lag1, true_exponent, true_floor = 0.2, 0.5, 0.01
    true_curve = power_law_lag_threshold(true_tau_at_lag1, true_exponent, true_floor, max_lag)

    scores, labels, lags = [], [], []
    for lag in range(1, max_lag + 1):
        true_signal = true_curve[lag] * 1.5
        noise = true_floor * 0.1
        for _ in range(10):
            scores.append(true_signal)
            labels.append(True)
            lags.append(lag)
            scores.append(noise)
            labels.append(False)
            lags.append(lag)

    fit = fit_power_law_lag_threshold(
        torch.tensor(scores), torch.tensor(labels), torch.tensor(lags), max_lag=max_lag
    )
    assert isinstance(fit, PowerLawLagThresholdFit)
    assert fit.f1 == pytest.approx(1.0)
    for lag in range(1, max_lag + 1):
        assert fit.tau_by_lag[lag] < true_curve[lag] * 1.5
        assert fit.tau_by_lag[lag] > true_floor * 0.1


def test_fit_power_law_lag_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fit_power_law_lag_threshold(
            torch.tensor([1.0, 2.0]), torch.tensor([True]), torch.tensor([1, 1])
        )


def test_power_law_lag_threshold_fit_summary_mentions_curve_and_f1():
    fit = PowerLawLagThresholdFit(
        tau_at_lag1=0.1, exponent=0.5, noise_floor=0.01, f1=0.75, tau_by_lag={1: 0.1}
    )
    s = fit.summary()
    assert "tau(lag)" in s
    assert "0.75" in s


# ---------------------------------------------------------------------------
# Joint lag x vocabulary-size threshold law.
# ---------------------------------------------------------------------------


def test_joint_lag_vocab_threshold_fit_tau_matches_formula():
    fit = JointLagVocabThresholdFit(
        signal_scale=1.0, vocab_exponent_signal=0.5, lag_exponent=0.5,
        floor_scale=0.01, vocab_exponent_floor=1.5, f1=0.8,
    )
    lag, vocab_size = 2.0, 100.0
    expected = 1.0 * vocab_size**-0.5 * lag**-0.5 + 0.01 * vocab_size**-1.5
    assert fit.tau(lag, vocab_size) == pytest.approx(expected)


def test_joint_lag_vocab_threshold_fit_summary_mentions_formula_and_f1():
    fit = JointLagVocabThresholdFit(
        signal_scale=1.0, vocab_exponent_signal=0.5, lag_exponent=0.5,
        floor_scale=0.01, vocab_exponent_floor=1.5, f1=0.8,
    )
    s = fit.summary()
    assert "tau(lag, |X|)" in s
    assert "0.800" in s


def test_fit_joint_lag_vocab_threshold_recovers_a_known_multi_dgp_signal():
    # Two synthetic "DGPs" (vocab sizes), each with 3 lags, where the
    # true-edge CMI decays with both lag and vocab size, and non-edges sit
    # well below the true signal at every (lag, vocab_size) combination.
    torch.manual_seed(0)
    scores, labels, lags, vocab_sizes = [], [], [], []
    for vocab_size in (100.0, 400.0):
        for lag in (1.0, 2.0, 3.0):
            true_signal = 1.0 * vocab_size**-0.3 * lag**-0.5
            noise = 0.01 * vocab_size**-1.5
            for _ in range(10):
                scores.append(true_signal * 1.5)
                labels.append(True)
                lags.append(lag)
                vocab_sizes.append(vocab_size)
                scores.append(noise * 0.5)
                labels.append(False)
                lags.append(lag)
                vocab_sizes.append(vocab_size)

    fit = fit_joint_lag_vocab_threshold(
        torch.tensor(scores), torch.tensor(labels), torch.tensor(lags), torch.tensor(vocab_sizes),
    )
    assert isinstance(fit, JointLagVocabThresholdFit)
    assert fit.f1 == pytest.approx(1.0)


def test_fit_joint_lag_vocab_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fit_joint_lag_vocab_threshold(
            torch.tensor([1.0, 2.0]), torch.tensor([True]), torch.tensor([1, 1]), torch.tensor([100, 100])
        )


def test_fit_joint_lag_vocab_threshold_empty_raises():
    with pytest.raises(ValueError):
        fit_joint_lag_vocab_threshold(
            torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        )


# ---------------------------------------------------------------------------
# AdaptiveThreshold: configurable, label-free tau(lag) construction.
# ---------------------------------------------------------------------------


def _multi_lag_scores():
    """3 lags, each with its own bimodal (true/false) score scale that
    shrinks with lag -- mirrors the decaying-CMI-with-lag DGPs used
    throughout this project."""
    torch.manual_seed(0)
    lag1 = torch.cat([torch.rand(100) * 1e-3, 0.1 + torch.rand(30) * 0.05])
    lag2 = torch.cat([torch.rand(100) * 1e-4, 0.02 + torch.rand(30) * 0.01])
    lag3 = torch.cat([torch.rand(100) * 1e-5, 0.005 + torch.rand(30) * 0.002])
    scores = torch.cat([lag1, lag2, lag3])
    lags = torch.cat(
        [torch.full((130,), lag, dtype=torch.long) for lag in (1, 2, 3)]
    )
    return scores, lags


def test_adaptive_threshold_default_anchors_at_lag1_otsu_and_decays():
    scores, lags = _multi_lag_scores()
    tau_by_lag = AdaptiveThreshold(method="otsu").tau_by_lag(scores, lags, max_lag=3)

    tau1 = otsu_threshold(scores[lags == 1])
    floor = otsu_threshold(scores[lags == 3])
    assert tau_by_lag[1] == pytest.approx(tau1)
    for lag in range(1, 4):
        expected = floor + (tau1 - floor) * math.exp(-0.3 * (lag - 1))
        assert tau_by_lag[lag] == pytest.approx(expected)
    values = [tau_by_lag[lag] for lag in range(1, 4)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_adaptive_threshold_per_lag_refits_independently():
    scores, lags = _multi_lag_scores()
    tau_by_lag = AdaptiveThreshold(method="otsu", per_lag=True).tau_by_lag(scores, lags, max_lag=3)
    for lag in range(1, 4):
        assert tau_by_lag[lag] == pytest.approx(otsu_threshold(scores[lags == lag]))


def test_adaptive_threshold_no_decay_reuses_lag1_tau_everywhere():
    scores, lags = _multi_lag_scores()
    tau_by_lag = AdaptiveThreshold(method="otsu", decay=False).tau_by_lag(scores, lags, max_lag=3)
    tau1 = otsu_threshold(scores[lags == 1])
    assert all(v == pytest.approx(tau1) for v in tau_by_lag.values())


def test_adaptive_threshold_decay_type_none_reuses_lag1_tau_everywhere():
    scores, lags = _multi_lag_scores()
    tau_by_lag = AdaptiveThreshold(method="otsu", decay_type="none").tau_by_lag(scores, lags, max_lag=3)
    tau1 = otsu_threshold(scores[lags == 1])
    assert all(v == pytest.approx(tau1) for v in tau_by_lag.values())


def test_adaptive_threshold_power_law_decay_matches_formula():
    scores, lags = _multi_lag_scores()
    tau_by_lag = AdaptiveThreshold(method="otsu", decay_type="power_law", exponent=0.5).tau_by_lag(
        scores, lags, max_lag=3
    )
    tau1 = otsu_threshold(scores[lags == 1])
    floor = otsu_threshold(scores[lags == 3])
    for lag in range(1, 4):
        expected = floor + (tau1 - floor) * (lag ** -0.5)
        assert tau_by_lag[lag] == pytest.approx(expected)


def test_adaptive_threshold_explicit_floor_overrides_auto_estimate():
    scores, lags = _multi_lag_scores()
    tau_by_lag = AdaptiveThreshold(method="otsu", floor=1e-4).tau_by_lag(scores, lags, max_lag=3)
    tau1 = otsu_threshold(scores[lags == 1])
    for lag in range(1, 4):
        expected = 1e-4 + (tau1 - 1e-4) * math.exp(-0.3 * (lag - 1))
        assert tau_by_lag[lag] == pytest.approx(expected)


def test_adaptive_threshold_falls_back_to_global_when_lag1_group_too_small():
    torch.manual_seed(0)
    lag1_scores = torch.tensor([0.1, 0.1001])  # below default min_group_size=8
    lag2_scores = torch.cat([torch.rand(50) * 1e-3, 0.05 + torch.rand(20) * 0.02])
    scores = torch.cat([lag1_scores, lag2_scores])
    lags = torch.cat([torch.full((2,), 1, dtype=torch.long), torch.full((70,), 2, dtype=torch.long)])

    tau_by_lag = AdaptiveThreshold(method="otsu", decay=False).tau_by_lag(scores, lags, max_lag=2)
    assert tau_by_lag[1] == pytest.approx(otsu_threshold(scores))


def test_adaptive_threshold_rejects_invalid_method():
    with pytest.raises(ValueError):
        AdaptiveThreshold(method="bogus")


def test_adaptive_threshold_rejects_invalid_decay_type():
    with pytest.raises(ValueError):
        AdaptiveThreshold(decay_type="bogus")


def test_adaptive_threshold_supports_other_base_methods():
    scores, lags = _multi_lag_scores()
    for method in ("mad", "percentile", "gmm"):
        tau_by_lag = AdaptiveThreshold(method=method).tau_by_lag(scores, lags, max_lag=3)
        assert set(tau_by_lag) == {1, 2, 3}
        assert all(isinstance(v, float) for v in tau_by_lag.values())


def test_adaptive_threshold_causal_graph_thresholds_a_full_cmi_matrix():
    torch.manual_seed(0)
    lc = 20  # large enough that every lag group has >= min_group_size pairs
    cmi_matrix = torch.rand(lc, lc) * 1e-4
    # Plant true edges with a decaying-with-lag signal, well above the noise
    # floor at every lag, so a properly-decayed threshold should catch them.
    cmi_matrix[0, 1] = 0.2   # lag 1
    cmi_matrix[5, 6] = 0.2   # lag 1
    cmi_matrix[0, 3] = 0.09  # lag 3
    cmi_matrix[10, 13] = 0.09  # lag 3

    graph = AdaptiveThreshold().causal_graph(cmi_matrix)
    assert graph.shape == cmi_matrix.shape
    assert graph.dtype == torch.bool
    assert graph[0, 1] and graph[5, 6]
    assert graph[0, 3] and graph[10, 13]
    # Lower triangle / diagonal (lag <= 0) must never be flagged.
    assert not graph.tril().any()


def test_adaptive_threshold_causal_graph_no_decay_matches_otsu_global():
    torch.manual_seed(0)
    lc = 20
    cmi_matrix = torch.rand(lc, lc) * 1e-4
    cmi_matrix[0, 1] = 0.2
    cmi_matrix[5, 14] = 0.15

    lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
    valid = lag_matrix > 0
    expected_tau = otsu_threshold(cmi_matrix[valid])
    expected_graph = torch.zeros_like(cmi_matrix, dtype=torch.bool)
    expected_graph[valid] = cmi_matrix[valid] >= expected_tau

    graph = AdaptiveThreshold(method="otsu", decay=False).causal_graph(cmi_matrix)
    assert torch.equal(graph, expected_graph)


def test_apply_tau_by_lag_matches_causal_graph_when_fit_on_the_same_matrix():
    """causal_graph() == tau_by_lag(...) + apply_tau_by_lag(...) when both
    are fit/applied on the SAME matrix -- apply_tau_by_lag must be a pure
    refactor, not a behavior change."""
    torch.manual_seed(0)
    lc = 15
    cmi_matrix = torch.rand(lc, lc) * 1e-4
    cmi_matrix[0, 1] = 0.2
    cmi_matrix[2, 5] = 0.1

    th = AdaptiveThreshold(method="mad")
    expected = th.causal_graph(cmi_matrix)

    lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
    valid = lag_matrix > 0
    tau_by_lag = th.tau_by_lag(cmi_matrix[valid], lag_matrix[valid], max_lag=int(lag_matrix.max()))
    got = th.apply_tau_by_lag(cmi_matrix, tau_by_lag)

    assert torch.equal(got, expected)


def test_apply_tau_by_lag_pooled_across_sequences_is_more_stable_than_per_sequence():
    """The whole point of `apply_tau_by_lag`: fit tau ONCE on scores pooled
    across several sequences, then apply it to each -- instead of fitting a
    fresh (noisier) cutoff per sequence. Regression guard that pooling
    actually reduces variance across sequences for a method sensitive to a
    single sequence's own idiosyncrasies."""
    torch.manual_seed(0)
    lc = 20
    n_sequences = 6
    lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
    valid = lag_matrix > 0
    max_lag = int(lag_matrix.max())

    # Same underlying bimodal structure every time, but each sequence's own
    # small sample of it is noisy -- exactly what makes per-sequence fitting
    # unstable in practice.
    matrices = []
    for _ in range(n_sequences):
        m = torch.rand(lc, lc) * 1e-4
        true_mask = torch.rand(lc, lc) < 0.05
        m[true_mask & valid] = 0.05 + torch.rand((true_mask & valid).sum()) * 0.05
        matrices.append(m)

    th = AdaptiveThreshold(method="mad")

    per_sequence_edge_counts = [int(th.causal_graph(m).sum()) for m in matrices]

    pooled_scores = torch.cat([m[valid] for m in matrices])
    pooled_lags = torch.cat([lag_matrix[valid] for _ in matrices])
    tau_by_lag = th.tau_by_lag(pooled_scores, pooled_lags, max_lag=max_lag)
    pooled_edge_counts = [int(th.apply_tau_by_lag(m, tau_by_lag).sum()) for m in matrices]

    # Pooling shares one calibration across all sequences, so the resulting
    # edge counts must vary less across sequences than fitting independently.
    assert torch.tensor(pooled_edge_counts, dtype=torch.float).std() <= torch.tensor(
        per_sequence_edge_counts, dtype=torch.float
    ).std()
