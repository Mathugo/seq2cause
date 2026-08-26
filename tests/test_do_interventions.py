import pytest
import torch

from seq2cause.sampling import (
    INTERVENTION_STRATEGIES,
    do_interventions,
    windowed_pair_index,
)


def _make_inputs(l_minus_c=5, c=2, n=1, noise_value=-1):
    """bs=1 fixture with distinguishable "real" vs "noise" sentinel values so
    masking behavior can be checked by direct value inspection."""
    rest_untouched = torch.arange(100, 100 + l_minus_c).unsqueeze(0).float()  # [1, L_minus_c]
    prefix = torch.arange(0, c).unsqueeze(0).float()  # [1, c]
    rest_upsampled = torch.full((1, n, l_minus_c), float(noise_value))
    return rest_upsampled, rest_untouched, prefix


class TestFullStrategyBackwardCompatibility:
    def test_default_strategy_is_full(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs()
        default = do_interventions(rest_upsampled, rest_untouched, prefix)
        explicit = do_interventions(rest_upsampled, rest_untouched, prefix, strategy="full")
        assert torch.equal(default, explicit)

    def test_shape_with_and_without_context(self):
        L_minus_c, c, N = 5, 2, 3
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c, N)
        with_ctx = do_interventions(rest_upsampled, rest_untouched, prefix, prepend_context_back=True)
        without_ctx = do_interventions(rest_upsampled, rest_untouched, prefix, prepend_context_back=False)
        assert with_ctx.shape == (1, N, L_minus_c, c + L_minus_c)
        assert without_ctx.shape == (1, N, L_minus_c, L_minus_c)

    def test_row_j_reveals_real_up_to_j_only(self):
        """Row j (0-indexed cause-under-test) must be real at columns <= j and
        noise at columns > j -- the core staircase invariant."""
        L_minus_c, c = 6, 0
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c)
        out = do_interventions(rest_upsampled, rest_untouched, prefix, prepend_context_back=False)
        out = out[0, 0]  # [num_rows, L_minus_c]
        for j in range(L_minus_c):
            for col in range(L_minus_c):
                expected = rest_untouched[0, col] if col <= j else -1.0
                assert out[j, col].item() == pytest.approx(expected)

    def test_lag1_context_is_preserved_across_the_row_shift(self):
        """For effect q and cause j = q-1 (lag 1), column q-1 (== j, the cause
        itself) is real in the treatment row (row j) and noise in the baseline
        row (row j-1) -- i.e. lag-1 pairs get a genuine contrast."""
        L_minus_c = 6
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out = do_interventions(rest_upsampled, rest_untouched, prefix, prepend_context_back=False)[0, 0]
        for q in range(1, L_minus_c):
            j = q - 1
            baseline_row = out[j - 1] if j - 1 >= 0 else None
            treatment_row = out[j]
            assert treatment_row[j].item() == rest_untouched[0, j].item()
            if baseline_row is not None:
                assert baseline_row[j].item() == -1.0

    def test_lag_ge2_context_is_lost_on_both_sides_of_the_contrast(self):
        """Characterizes the bug reported by Chadyuk et al. (2026): for an
        effect q and a candidate cause j < q-1 (lag >= 2), the immediately
        preceding position q-1 is noise in BOTH the baseline row (j-1) and the
        treatment row (j) whenever q-1 > j, i.e. the local context the model
        depends on most is destroyed on both sides of the CMI contrast."""
        L_minus_c = 6
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out = do_interventions(rest_upsampled, rest_untouched, prefix, prepend_context_back=False)[0, 0]

        q = 5
        j = 2  # lag = 3, so j < q - 1
        assert j < q - 1
        baseline_row = out[j - 1]
        treatment_row = out[j]
        assert baseline_row[q - 1].item() == -1.0
        assert treatment_row[q - 1].item() == -1.0

    def test_sparse_memory_bound_generates_only_last_m_rows(self):
        L_minus_c, m = 6, 2
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out = do_interventions(rest_upsampled, rest_untouched, prefix, m=m, prepend_context_back=False)
        assert out.shape[-2] == m
        # Row 0 of the sparse output corresponds to cause index L_minus_c - m.
        out = out[0, 0]
        first_row_cause = L_minus_c - m
        for col in range(L_minus_c):
            expected = rest_untouched[0, col] if col <= first_row_cause else -1.0
            assert out[0, col].item() == pytest.approx(expected)

    def test_boundary_single_position_sequence(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs(l_minus_c=1, c=1)
        out = do_interventions(rest_upsampled, rest_untouched, prefix, prepend_context_back=True)
        assert out.shape == (1, 1, 1, 2)
        # The only row/cause (index 0) reveals the only column as real.
        assert out[0, 0, 0, 1].item() == rest_untouched[0, 0].item()

    def test_unknown_strategy_raises(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs()
        with pytest.raises(ValueError):
            do_interventions(rest_upsampled, rest_untouched, prefix, strategy="not-a-strategy")


class TestAtomicStrategy:
    def test_only_the_diagonal_column_is_noised_per_row(self):
        L_minus_c = 5
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out = do_interventions(
            rest_upsampled, rest_untouched, prefix, strategy="atomic", prepend_context_back=False
        )[0, 0]
        assert out.shape == (L_minus_c, L_minus_c)
        for row in range(L_minus_c):
            for col in range(L_minus_c):
                if col == row:
                    assert out[row, col].item() == -1.0
                else:
                    assert out[row, col].item() == rest_untouched[0, col].item()

    def test_mediators_between_cause_and_lag_ge2_effect_stay_real(self):
        """Directly contrasts with the "full" strategy's lag>=2 bug: here,
        position q-1 stays real (observed) when testing a distant cause j."""
        L_minus_c = 6
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out = do_interventions(
            rest_upsampled, rest_untouched, prefix, strategy="atomic", prepend_context_back=False
        )[0, 0]
        j, q = 1, 5
        assert out[j, q - 1].item() == rest_untouched[0, q - 1].item()

    def test_max_pairs_guard(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs(l_minus_c=10, c=0)
        with pytest.raises(ValueError, match="max_pairs"):
            do_interventions(
                rest_upsampled, rest_untouched, prefix, strategy="atomic", max_pairs=1
            )
        # Should not raise when the guard is disabled or generous.
        do_interventions(rest_upsampled, rest_untouched, prefix, strategy="atomic", max_pairs=None)


class TestWindowedStrategy:
    def test_pair_index_count_and_order(self):
        pairs = windowed_pair_index(4)
        assert pairs == [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]

    def test_pair_index_excludes_near_lag_pairs_within_the_window(self):
        """Pairs with lag <= window_k are excluded: for those, the cause
        position would already be inside the always-preserved window in
        every row of that effect's group, making the row-shift comparison a
        no-op (identical sequences either side) regardless of true causal
        strength -- see `windowed_pair_index`'s docstring."""
        pairs = windowed_pair_index(6, window_k=2)
        for j, q in pairs:
            assert q - j > 2
        # (j=q-1, lag=1) and (j=q-2, lag=2) pairs must be gone.
        assert (4, 5) not in pairs
        assert (3, 5) not in pairs
        assert (2, 5) in pairs  # lag=3 > window_k=2, still tested

    def test_window_k_zero_matches_plain_tril_masking(self):
        L_minus_c = 5
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out, pairs = do_interventions(
            rest_upsampled,
            rest_untouched,
            prefix,
            strategy="windowed",
            window_k=0,
            prepend_context_back=False,
        )
        for row_idx, (j, q) in enumerate(pairs):
            row = out[0, 0, row_idx]
            for col in range(L_minus_c):
                expected = rest_untouched[0, col] if col <= j else -1.0
                assert row[col].item() == pytest.approx(expected), (row_idx, j, q, col)

    def test_large_window_k_excludes_every_pair(self):
        """With window_k >= L_minus_c, every possible lag falls inside the
        always-preserved window, so no (cause, effect) pair can be tested by
        this construction -- all of them are left to strategy="full" instead
        (see `compare_intervention_strategies`, which fills these cells in
        from `full`'s own CMI matrix)."""
        L_minus_c = 6
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out, pairs = do_interventions(
            rest_upsampled,
            rest_untouched,
            prefix,
            strategy="windowed",
            window_k=L_minus_c,
            prepend_context_back=False,
        )
        assert pairs == []
        assert out.shape == (1, 1, 0, L_minus_c)

    def test_near_lag_causes_no_longer_produce_degenerate_identical_rows(self):
        """Regression test for a bug found while empirically testing lagged
        effects on a trained model: previously, `windowed_pair_index` still
        generated near-lag pairs (lag <= window_k), whose row-shift baseline
        and treatment rows turned out *identical* (the cause column was
        already forced real by the always-preserved window in both rows),
        silently reading CMI=0 regardless of true causal strength. Those
        pairs must no longer be generated at all -- while the boundary pair
        at lag == window_k + 1 (the last meaningful transition) must still
        be tested (row j-1 has a genuine noise gap that row j fills in)."""
        L_minus_c = 8
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        out, pairs = do_interventions(
            rest_upsampled,
            rest_untouched,
            prefix,
            strategy="windowed",
            window_k=2,
            prepend_context_back=False,
        )
        # lag <= window_k (2, 1): degenerate, excluded.
        assert (5, 7) not in pairs
        assert (6, 7) not in pairs
        # lag == window_k + 1 (3): the last valid, non-degenerate transition.
        assert (4, 7) in pairs
        row_idx = pairs.index((4, 7))
        row = out[0, 0, row_idx]
        assert row[4].item() == rest_untouched[0, 4].item()  # cause j=4 is real here
        # Its baseline (boundary j=3) must leave a genuine noise gap at column 4,
        # i.e. the two rows actually differ -- otherwise the CMI would still
        # collapse to 0 despite the pair being "included".
        baseline_row_idx = pairs.index((3, 7))
        baseline_row = out[0, 0, baseline_row_idx]
        assert baseline_row[4].item() != row[4].item()

    def test_max_pairs_guard(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs(l_minus_c=10, c=0)
        with pytest.raises(ValueError, match="max_pairs"):
            do_interventions(
                rest_upsampled, rest_untouched, prefix, strategy="windowed", max_pairs=1
            )

    def test_boundary_no_possible_pairs(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs(l_minus_c=1, c=1)
        out, pairs = do_interventions(
            rest_upsampled, rest_untouched, prefix, strategy="windowed", prepend_context_back=True
        )
        assert pairs == []
        assert out.shape == (1, 1, 0, 2)

    def test_negative_window_k_raises(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs()
        with pytest.raises(ValueError):
            do_interventions(rest_upsampled, rest_untouched, prefix, strategy="windowed", window_k=-1)


class TestIndependentMediatorStrategy:
    def test_requires_mediator_tensor(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs()
        with pytest.raises(ValueError, match="rest_upsampled_mediator"):
            do_interventions(rest_upsampled, rest_untouched, prefix, strategy="independent_mediator")

    def test_shape_mismatch_raises(self):
        rest_upsampled, rest_untouched, prefix = _make_inputs(l_minus_c=5)
        bad_mediator = torch.zeros(1, 1, 3)
        with pytest.raises(ValueError, match="same shape"):
            do_interventions(
                rest_upsampled,
                rest_untouched,
                prefix,
                strategy="independent_mediator",
                rest_upsampled_mediator=bad_mediator,
            )

    def test_cause_and_mediator_noise_are_used_in_the_right_roles(self):
        """Column j's noise value should come from the "cause" tensor only in
        the single row where j is the tested cause-under-transition, and from
        the "mediator" tensor in every other (earlier) row where it is noise."""
        L_minus_c, c = 5, 0
        rest_untouched = torch.arange(100, 100 + L_minus_c).unsqueeze(0).float()
        prefix = torch.zeros(1, c)
        cause_noise = torch.full((1, 1, L_minus_c), -1.0)
        mediator_noise = torch.full((1, 1, L_minus_c), -2.0)

        out = do_interventions(
            cause_noise,
            rest_untouched,
            prefix,
            strategy="independent_mediator",
            rest_upsampled_mediator=mediator_noise,
            prepend_context_back=False,
        )[0, 0]

        # Row 0 (cause=0 boundary) has columns 1..L_minus_c-1 as noise. Column 1
        # is "the cause under test in the next transition" (row0 -> row1) so it
        # must come from the cause tensor; columns 2.. are pure mediators.
        assert out[0, 1].item() == -1.0
        for col in range(2, L_minus_c):
            assert out[0, col].item() == -2.0

    def test_matches_full_strategy_masking_pattern(self):
        L_minus_c = 5
        rest_upsampled, rest_untouched, prefix = _make_inputs(L_minus_c, c=0)
        mediator = torch.full_like(rest_upsampled, -1.0)
        full = do_interventions(rest_upsampled, rest_untouched, prefix, strategy="full", prepend_context_back=False)
        indep = do_interventions(
            rest_upsampled,
            rest_untouched,
            prefix,
            strategy="independent_mediator",
            rest_upsampled_mediator=mediator,
            prepend_context_back=False,
        )
        assert torch.equal(full, indep)


def test_all_declared_strategies_are_dispatchable():
    rest_upsampled, rest_untouched, prefix = _make_inputs(l_minus_c=4, c=1)
    mediator = torch.full_like(rest_upsampled, -2.0)
    for strategy in INTERVENTION_STRATEGIES:
        kwargs = {}
        if strategy == "independent_mediator":
            kwargs["rest_upsampled_mediator"] = mediator
        result = do_interventions(rest_upsampled, rest_untouched, prefix, strategy=strategy, **kwargs)
        if strategy == "windowed":
            tensor, pairs = result
            assert isinstance(pairs, list)
        else:
            tensor = result
        assert tensor.dim() == 4
