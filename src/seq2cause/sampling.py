from __future__ import annotations

import time

import torch
from jaxtyping import Float, Int
from torch import Tensor

# Strategies supported by `do_interventions`. "full" is the original
# staircase construction and remains the default for backward compatibility.
# See the `do_interventions` docstring for what each strategy changes and why.
INTERVENTION_STRATEGIES = ("full", "atomic", "windowed", "independent_mediator")


def do_interventions(
    rest_upsampled: Float[Tensor, "bs N L_minus_c"],
    rest_untouched: Float[Tensor, "bs L_minus_c"],
    prefix: Float[Tensor, "bs c"],
    m: int = None,
    prepend_context_back: bool = True,
    strategy: str = "full",
    window_k: int = 1,
    max_pairs: int | None = 20000,
    rest_upsampled_mediator: Float[Tensor, "bs N L_minus_c"] | None = None,
    **kwargs,
) -> Float[Tensor, "bs N num_rows L"] | tuple[Float[Tensor, "bs N num_rows L"], list]:
    """Builds the intervention tensor used to detect (potentially delayed) causal effects.

    This dispatches to one of several pluggable construction strategies. The
    default, "full", is the original staircase construction from Math &
    Lienhart (2026): it generates all L-c rows corresponding to testing
    causes X_0 ... X_{L-c-1} in a single O(L) sweep (or, if `m` is provided,
    only the last `m` rows -- the sparse/bounded-memory variant).

    An independent replication (Chadyuk, Zhang, and Kucukates, "Replicating
    TRACE: A Practitioner's Guide to Its Threshold and Particle Budget",
    LotusFlare Inc., Aug 2026) found that the "full" staircase collapses
    recall for cause-effect lags >= 2 almost to zero regardless of the
    threshold: for a candidate cause at row j and an effect at position
    q > j + 1, the position q-1 (immediately preceding the effect, which an
    autoregressive model relies on most) is randomized in *both* rows being
    differenced (row j-1, the baseline, and row j, the do-branch), because
    both rows noise everything strictly after their own diagonal. This
    collapses the model's predicted probability of the true token at q
    toward its marginal frequency on both sides of the CMI contrast,
    regardless of the true causal strength between j and q. The alternative
    strategies below exist to isolate and test fixes for this:

    Strategies
    ----------
    "full" (default): see above. O(L) rows. Loses local context for lag >= 2.
    "atomic": literal per-cause construction. Only the single candidate-cause
        position is replaced with noise; every *other* position -- crucially
        including the mediators strictly between the cause and any
        downstream effect -- is kept at its real observed value. Still O(L)
        rows to *construct* (one row per candidate cause is enough, since
        the mask no longer depends on which effect is being read off), but
        scoring the O(L^2) (cause, effect) pairs this construction is
        typically used to test mirrors the pre-"staircase" complexity
        discussed in the paper (Sec. 5.1); gated by `max_pairs`.
    "windowed": generalizes "full" by preserving a trailing local-context
        radius `window_k` immediately before *each* candidate effect
        position, instead of noising everything after the cause. This is
        genuinely O(L^2) rows (one per far (cause, effect) pair) because the
        preserved window is a function of the *effect* position, which a
        single shared staircase row cannot express for more than one effect
        at a time; gated by `max_pairs`. `window_k=0` degenerates back to
        "full"'s masking (restricted to one effect at a time). Only pairs
        with lag `q - j > window_k` are tested here: for a nearer cause, the
        always-preserved window already forces it real in every row of its
        effect's group, making the row-to-row transition a no-op (CMI = 0 by
        construction, not by causal strength) -- see `windowed_pair_index`.
        Callers combining "windowed" with "full" (as
        `diagnostics.compare_intervention_strategies` does) should fill
        those excluded near-lag cells in from "full"'s own CMI matrix.
    "independent_mediator": identical masking to "full", but draws the
        "cause" particle (the token used at the position that transitions
        noise -> real between two adjacent staircase rows) and the
        "mediator" particles (every other noised position) from two
        statistically independent noise tensors (`rest_upsampled` and
        `rest_upsampled_mediator`), instead of reusing one shared tensor via
        `.repeat()` as "full" does.

    Args:
        rest_upsampled: proposal/noise samples used to fill in intervened
            positions (the "cause" tensor for "independent_mediator").
        rest_untouched: the real, observed suffix tokens (ground truth).
        prefix: the (possibly upsampled) fixed context to prepend.
        m: if set and < L_minus_c, only build the last `m` rows (sparse
            variant). Ignored by "atomic"/"windowed" (whose row structure is
            already determined by the strategy).
        prepend_context_back: whether to concatenate `prefix` back in front.
        strategy: one of `INTERVENTION_STRATEGIES`.
        window_k: local-context radius for strategy="windowed".
        max_pairs: safety guard on the number of (cause, effect) pairs
            implied by "atomic"/"windowed" -- raises `ValueError` if exceeded
            (pass `None` to disable). See Chadyuk et al. (2026) on the O(L^2)
            cost of literal per-pair CI-tests.
        rest_upsampled_mediator: required for strategy="independent_mediator";
            a second noise tensor with the same shape as `rest_upsampled`,
            drawn independently (e.g. via a second, separate call to a
            proposal function such as `uniform_sample`).

    Returns:
        For "full", "atomic", and "independent_mediator": a tensor
        `[bs, N, num_rows, L]` (or `[bs, N, num_rows, L_minus_c]` if
        `prepend_context_back=False`).
        For "windowed": a tuple `(tensor, pair_index)` where `pair_index` is
        a list of `(cause_j, effect_q)` tuples describing what each row
        along the "num_rows" dimension corresponds to -- rows are no longer
        a simple monotonic staircase, since the preserved window couples
        `window_k` different effects together.
    """
    strategy = strategy or "full"
    if strategy not in INTERVENTION_STRATEGIES:
        raise ValueError(
            f"Unknown do_interventions strategy {strategy!r}, expected one of "
            f"{INTERVENTION_STRATEGIES}"
        )

    if strategy == "full":
        return _do_interventions_full(
            rest_upsampled, rest_untouched, prefix, m=m, prepend_context_back=prepend_context_back
        )
    if strategy == "atomic":
        return _do_interventions_atomic(
            rest_upsampled,
            rest_untouched,
            prefix,
            prepend_context_back=prepend_context_back,
            max_pairs=max_pairs,
        )
    if strategy == "windowed":
        return _do_interventions_windowed(
            rest_upsampled,
            rest_untouched,
            prefix,
            window_k=window_k,
            prepend_context_back=prepend_context_back,
            max_pairs=max_pairs,
        )
    # strategy == "independent_mediator"
    if rest_upsampled_mediator is None:
        raise ValueError(
            "strategy='independent_mediator' requires `rest_upsampled_mediator`, a second, "
            "independently-drawn noise tensor with the same shape as `rest_upsampled` (e.g. "
            "obtained via a second, separate call to a proposal function such as "
            "`uniform_sample`)."
        )
    return _do_interventions_independent_mediator(
        rest_upsampled,
        rest_upsampled_mediator,
        rest_untouched,
        prefix,
        m=m,
        prepend_context_back=prepend_context_back,
    )


def _prepend_context(
    rest_final: Tensor, prefix: Tensor, n: int, num_rows: int, prepend_context_back: bool
) -> Tensor:
    if not prepend_context_back:
        return rest_final
    device = rest_final.device
    prefix_expanded = prefix.unsqueeze(1).unsqueeze(2).repeat(1, n, num_rows, 1).to(device)
    return torch.cat([prefix_expanded, rest_final], dim=-1)


def _do_interventions_full(
    rest_upsampled: Tensor,
    rest_untouched: Tensor,
    prefix: Tensor,
    m: int = None,
    prepend_context_back: bool = True,
) -> Tensor:
    """The original staircase intervention construction (default strategy="full")."""

    device = rest_upsampled.device
    bs, N, L_minus_c = rest_upsampled.size()

    # 1. Determine number of rows
    # If m is set, we only generate the LAST m experiments.
    # This corresponds to testing causes: X_{L-m} ... X_{L-1}
    if m is not None and m < L_minus_c:
        num_rows = m
        print(f"[!] Applying memory bounding: Generating last {m} rows only.")
    else:
        num_rows = L_minus_c

    # 2. Expand vanilla tokens (Ground Truth)
    # Shape: [bs, N, num_rows, L-c]
    rest_untouched_exp = rest_untouched.unsqueeze(1).unsqueeze(1).repeat(1, N, num_rows, 1)

    # 3. Expand proposal samples (Noise)
    rest_intervened_exp = rest_upsampled.unsqueeze(2).repeat(1, 1, num_rows, 1)

    # 4. Build the Staircase mask
    # We start with the full Lower Triangular mask for the whole sequence
    full_mask = torch.tril(torch.ones((L_minus_c, L_minus_c), device=device, dtype=torch.bool))

    # Take the LAST num_rows
    # Row 0 of this slice corresponds to the experiment for cause index (L_minus_c - num_rows)
    staircase_mask = full_mask[-num_rows:, :]

    # Expand for batch and particles
    # [bs, N, num_rows, L-c]
    staircase_mask = staircase_mask.unsqueeze(0).unsqueeze(0).repeat(bs, N, 1, 1)

    # 5. Apply interventions
    rest_final = torch.where(
        staircase_mask,
        rest_untouched_exp,  # Lower Triangle: Ground Truth
        rest_intervened_exp,  # Upper Triangle: Noise
    )

    # 6. Add back the untouched context
    return _prepend_context(rest_final, prefix, N, num_rows, prepend_context_back)


def _do_interventions_atomic(
    rest_upsampled: Tensor,
    rest_untouched: Tensor,
    prefix: Tensor,
    prepend_context_back: bool = True,
    max_pairs: int | None = 20000,
) -> Tensor:
    """Literal per-cause construction: only the tested cause position is noised.

    Every other position (including mediators between the cause and any
    downstream effect) stays at its real, observed value. One row per
    candidate cause suffices to read off every downstream effect at once,
    but the number of (cause, effect) pairs this is meant to test is
    `L_minus_c * (L_minus_c - 1) / 2`, guarded by `max_pairs`.
    """
    device = rest_upsampled.device
    bs, N, L_minus_c = rest_upsampled.size()
    num_rows = L_minus_c

    implied_pairs = num_rows * (num_rows - 1) // 2
    if max_pairs is not None and implied_pairs > max_pairs:
        raise ValueError(
            f"strategy='atomic' implies scoring {implied_pairs} (cause, effect) pairs "
            f"(L_minus_c={L_minus_c}), exceeding max_pairs={max_pairs}. Pass a larger "
            "`max_pairs` (or None to disable this guard) if you intend to run at this "
            "scale -- see Chadyuk et al. (2026) on the O(L^2) cost of literal per-pair "
            "CI-tests, matching the paper's own pre-'staircase' complexity discussion "
            "(Sec. 5.1: L(L+1)/2 CI-tests)."
        )

    rest_untouched_exp = rest_untouched.unsqueeze(1).unsqueeze(1).repeat(1, N, num_rows, 1)
    rest_intervened_exp = rest_upsampled.unsqueeze(2).repeat(1, 1, num_rows, 1)

    # Real everywhere except the single tested-cause column (row == column).
    keep_real_mask = ~torch.eye(num_rows, L_minus_c, device=device, dtype=torch.bool)
    keep_real_mask = keep_real_mask.unsqueeze(0).unsqueeze(0).repeat(bs, N, 1, 1)

    rest_final = torch.where(keep_real_mask, rest_untouched_exp, rest_intervened_exp)
    return _prepend_context(rest_final, prefix, N, num_rows, prepend_context_back)


def windowed_pair_index(l_minus_c: int, window_k: int = 0) -> list[tuple[int, int]]:
    """Row order used by `strategy="windowed"`.

    Rows are grouped by effect `q` (ascending), and within each `q`, ordered
    by cause `j` ascending -- mirroring the staircase's row-shift convention
    (row `j` vs row `j-1`) but restricted to a single effect `q` at a time,
    since the preserved local-context window depends on `q`.

    Only pairs with `j < q - window_k` (i.e. lag `q - j > window_k`) are
    included: for a cause `j` *inside* the always-preserved trailing window
    `[q - window_k, q - 1]`, position `j` would already be real in every row
    of `q`'s group regardless of the staircase boundary, making the row `j`
    vs row `j-1` transition a no-op (identical sequences either side, hence
    CMI = 0 by construction, not by causal strength). Those near-lag pairs
    (`lag <= window_k`) are exactly what strategy="full" already tests
    correctly (see its lag-1 regression test), so they are deliberately left
    out here rather than silently reported as zero; combine with "full"'s
    CMI matrix for those cells (see `diagnostics.compare_intervention_strategies`).
    """
    if window_k < 0:
        raise ValueError(f"window_k must be >= 0, got {window_k}")
    pairs: list[tuple[int, int]] = []
    for q in range(1, l_minus_c):
        for j in range(0, max(0, q - window_k)):
            pairs.append((j, q))
    return pairs


def _do_interventions_windowed(
    rest_upsampled: Tensor,
    rest_untouched: Tensor,
    prefix: Tensor,
    window_k: int = 1,
    prepend_context_back: bool = True,
    max_pairs: int | None = 20000,
) -> tuple[Tensor, list[tuple[int, int]]]:
    """Staircase generalization that preserves a trailing local-context window.

    For each *far* (cause `j`, effect `q`) pair with `q - j > window_k`,
    positions `<= j` (the cause and everything before it) and positions in
    `[q - window_k, q - 1]` (the preserved trailing window immediately
    before the effect) are real; everything strictly between (the "far"
    mediators being tested) is noise. Near-lag pairs (`q - j <= window_k`)
    are excluded entirely (see `windowed_pair_index`); `window_k=0` recovers
    "full"'s per-effect masking exactly (no pair is ever near-lag).
    """
    device = rest_upsampled.device
    bs, N, L_minus_c = rest_upsampled.size()
    if window_k < 0:
        raise ValueError(f"window_k must be >= 0, got {window_k}")

    pairs = windowed_pair_index(L_minus_c, window_k=window_k)
    num_rows = len(pairs)

    if max_pairs is not None and num_rows > max_pairs:
        raise ValueError(
            f"strategy='windowed' implies {num_rows} (cause, effect) rows "
            f"(L_minus_c={L_minus_c}), exceeding max_pairs={max_pairs}. Pass a larger "
            "`max_pairs` (or None to disable this guard) if you intend to run at this "
            "scale -- see Chadyuk et al. (2026) on the O(L^2) cost of literal per-pair "
            "CI-tests."
        )

    if num_rows == 0:
        empty = rest_untouched.new_zeros((bs, N, 0, L_minus_c))
        return _prepend_context(empty, prefix, N, 0, prepend_context_back), pairs

    mask = torch.zeros((num_rows, L_minus_c), dtype=torch.bool, device=device)
    for r, (j, q) in enumerate(pairs):
        mask[r, : j + 1] = True  # cause (and everything before it) real
        window_start = q - window_k  # > j+1 by construction (pairs are lag > window_k)
        if window_start < q:
            mask[r, window_start:q] = True  # preserved local context before the effect

    rest_untouched_exp = rest_untouched.unsqueeze(1).unsqueeze(1).repeat(1, N, num_rows, 1)
    rest_intervened_exp = rest_upsampled.unsqueeze(2).repeat(1, 1, num_rows, 1)
    mask_exp = mask.unsqueeze(0).unsqueeze(0).repeat(bs, N, 1, 1)

    rest_final = torch.where(mask_exp, rest_untouched_exp, rest_intervened_exp)
    return _prepend_context(rest_final, prefix, N, num_rows, prepend_context_back), pairs


def _do_interventions_independent_mediator(
    rest_upsampled_cause: Tensor,
    rest_upsampled_mediator: Tensor,
    rest_untouched: Tensor,
    prefix: Tensor,
    m: int = None,
    prepend_context_back: bool = True,
) -> Tensor:
    """Same masking as "full", but cause vs. mediator noise are drawn independently.

    In "full", the single `rest_upsampled` tensor is generated once and
    `.repeat()`-ed across every staircase row, so the noise value used for a
    given column when it plays the role of "the candidate cause being
    tested" (the row-to-row transition) is *not* independent of the noise
    value used for that same column when it plays the role of "a far,
    untested mediator" for earlier rows. This variant separates the two
    roles: `rest_upsampled_cause[..., j]` fills column `j` specifically in
    the one row where it is the tested cause (row `j - row_offset - 1`,
    i.e. the last row where it is still noise, right before the row where it
    flips to real); `rest_upsampled_mediator[..., j]` fills column `j` in
    every other row where it is noise.
    """
    device = rest_upsampled_cause.device
    bs, N, L_minus_c = rest_upsampled_cause.size()
    if rest_upsampled_mediator.shape != rest_upsampled_cause.shape:
        raise ValueError(
            "rest_upsampled_mediator must have the same shape as rest_upsampled (cause), got "
            f"{tuple(rest_upsampled_mediator.shape)} vs {tuple(rest_upsampled_cause.shape)}"
        )

    if m is not None and m < L_minus_c:
        num_rows = m
        print(f"[!] Applying memory bounding: Generating last {m} rows only.")
    else:
        num_rows = L_minus_c
    row_offset = L_minus_c - num_rows

    rest_untouched_exp = rest_untouched.unsqueeze(1).unsqueeze(1).repeat(1, N, num_rows, 1)
    cause_noise_exp = rest_upsampled_cause.unsqueeze(2).repeat(1, 1, num_rows, 1)
    mediator_noise_exp = rest_upsampled_mediator.unsqueeze(2).repeat(1, 1, num_rows, 1)

    full_mask = torch.tril(torch.ones((L_minus_c, L_minus_c), device=device, dtype=torch.bool))
    staircase_mask = full_mask[-num_rows:, :]  # True == real

    # For row index i (0-indexed within the selected num_rows), the column
    # that transitions noise -> real when moving from row i to row i+1 is
    # (row_offset + i + 1); that is the "cause under test" column for row i.
    row_idx = torch.arange(num_rows, device=device).unsqueeze(1)
    col_idx = torch.arange(L_minus_c, device=device).unsqueeze(0)
    is_cause_role = row_idx == (col_idx - row_offset - 1)  # [num_rows, L_minus_c]

    noise_source = torch.where(
        is_cause_role.unsqueeze(0).unsqueeze(0).repeat(bs, N, 1, 1),
        cause_noise_exp,
        mediator_noise_exp,
    )

    staircase_mask_exp = staircase_mask.unsqueeze(0).unsqueeze(0).repeat(bs, N, 1, 1)
    rest_final = torch.where(staircase_mask_exp, rest_untouched_exp, noise_source)

    return _prepend_context(rest_final, prefix, N, num_rows, prepend_context_back)


def uniform_sample(
    prob_x: Float[Tensor, "bs vocab"] | Float[Tensor, "bs L vocab"],
    n_samples: int = 128,
    cls_token_id: int | None = None,
    device: torch.device | None = None,
) -> Int[Tensor, "bs n_samples"] | Int[Tensor, "bs n_samples L"]:
    """Uniform sampling over the vocabulary for virtual do-interventions.

    Supports both single-step distributions and full trajectory distributions.
    Internal logic automatically detects dimensionality to return consistent shapes.

    Args:
        prob_x: Next token probabilities over the vocabulary.
            Can be [batch_size, vocab] or [batch_size, seq_len, vocab].
        n_samples: Number of samples (particles) to generate per batch element.
        cls_token_id: If provided, forces the first token of every sample to this ID.
        device: Target device for sampled tensors.

    Returns:
        sampled_tokens: The discrete samples. [bs, n_samples] for 2D input
        or [bs, n_samples, L] for 3D.
    """

    device = device or prob_x.device

    if prob_x.dim() == 2:
        # ---- Single-step intervention ----
        bs, vocab = prob_x.shape

        sampled_tokens = torch.randint(low=0, high=vocab, size=(bs,), device=device)

        if cls_token_id is not None:
            sampled_tokens[:] = cls_token_id

        return sampled_tokens

    elif prob_x.dim() == 3:
        # ---- Trajectory intervention ----
        bs, L, vocab = prob_x.shape
        n = n_samples

        sampled = torch.randint(low=0, high=vocab, size=(bs, n, L), device=device)

        # Force CLS token if needed
        if cls_token_id is not None:
            sampled[:, :, 0] = cls_token_id

        return sampled
    else:
        raise ValueError("prob_x must be 2D or 3D tensor")


def unigram_sample(
    prob_x: Float[Tensor, "bs vocab"] | Float[Tensor, "bs L vocab"],
    n_samples: int = 128,
    cls_token_id: int | None = None,
    device: torch.device | None = None,
    unigram_freqs: Tensor | None = None,
    **kwargs,
) -> Int[Tensor, "bs n_samples"] | Int[Tensor, "bs n_samples L"]:
    """In-distribution-noise proposal: samples do-intervention tokens from a fixed
    unigram frequency distribution over the vocabulary, instead of `uniform_sample`'s
    uniform distribution.

    Motivation (Chadyuk et al., 2026, "Replicating TRACE"): a uniform proposal is
    maximally out-of-distribution for the autoregressive model, which could itself
    contribute to collapsing the predicted probability of the true token toward its
    marginal rarity -- on top of, and confoundable with, the loss-of-local-context
    mechanism identified for cause-effect lags >= 2. Sampling the do-intervention
    tokens from the corpus's own unigram frequency keeps the noise "in-distribution"
    for the model while remaining causally uninformative (a marginal, context-free
    draw), which lets a diagnostic separate "noise destroys local context" from
    "uniform noise is additionally OOD for the model."

    Args:
        prob_x: Next token probabilities, used only to infer batch/particle/seq
            shape. Can be [batch_size, vocab] or [batch_size, seq_len, vocab].
        n_samples: Number of samples (particles) to generate per batch element.
        cls_token_id: If provided, forces the first token of every sample to this ID.
        device: Target device for sampled tensors.
        unigram_freqs: 1D tensor of per-token frequencies (or probabilities; need
            not be pre-normalized) over the vocabulary. Required.

    Returns:
        sampled_tokens: The discrete samples. [bs, n_samples] for 2D input
        or [bs, n_samples, L] for 3D.
    """
    if unigram_freqs is None:
        raise ValueError(
            "unigram_sample requires `unigram_freqs`, a 1D frequency/probability tensor "
            "over the vocabulary (e.g. token counts from the training corpus)."
        )

    device = device or prob_x.device
    freqs = torch.as_tensor(unigram_freqs, dtype=torch.float32, device=device).flatten()
    if torch.any(freqs < 0):
        raise ValueError("unigram_freqs must be non-negative.")
    total = freqs.sum()
    if total <= 0:
        raise ValueError("unigram_freqs must sum to a strictly positive value.")
    freqs = freqs / total

    if prob_x.dim() == 2:
        bs, _vocab = prob_x.shape
        sampled_tokens = torch.multinomial(freqs, num_samples=bs, replacement=True).to(device)

        if cls_token_id is not None:
            sampled_tokens[:] = cls_token_id

        return sampled_tokens

    elif prob_x.dim() == 3:
        bs, L, _vocab = prob_x.shape
        n = n_samples

        flat = torch.multinomial(freqs, num_samples=bs * n * L, replacement=True)
        sampled = flat.view(bs, n, L).to(device)

        if cls_token_id is not None:
            sampled[:, :, 0] = cls_token_id

        return sampled
    else:
        raise ValueError("prob_x must be 2D or 3D tensor")


def multinomial_sample(
    prob_x: Float[Tensor, "bs vocab"] | Float[Tensor, "bs L vocab"],
    n_samples: int = 128,
    cls_token_id: int | None = None,
    **kwargs,
) -> Int[Tensor, "bs n_samples"] | Int[Tensor, "bs n_samples L"]:
    """
    Multinomial sampling from prob_x.

    Args:
        prob_x: Next token probabilities over the vocabulary.
            Can be [batch_size, vocab] or [batch_size, seq_len, vocab].
        n_samples: Number of samples (particles) to generate per batch element.
        cls_token_id: If provided, forces the first token of every sample to this ID.
        device: Target device for sampled tensors.

    Returns:
        sampled_tokens: The discrete samples. [bs, n_samples]
        for 2D input or [bs, n_samples, L] for 3D.
    """

    if prob_x.dim() == 2:
        # ---- Single-step sampling ----
        # prob_x: [bs, vocab]
        sampled_tokens = torch.multinomial(prob_x, 1)
        return sampled_tokens

    elif prob_x.dim() == 3:
        # ---- Trajectory sampling ----
        bs, L, vocab = prob_x.shape
        n = n_samples

        # Expand for n samples
        probs = prob_x.unsqueeze(1).expand(bs, n, L, vocab)
        probs = probs.reshape(-1, vocab)  # [(bs*n*L), vocab]

        # Sample
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        sampled = sampled.view(bs, n, L)

        # Force CLS token if needed
        if cls_token_id is not None:
            sampled[:, :, 0] = cls_token_id

        return sampled

    else:
        raise ValueError("prob_x must be 2D or 3D tensor")


def ancestral_sampling(
    model: any,
    encoded_input: dict[str, Float[Tensor, "bs L"]],
    value: int = 64,
    guidance: int = 2,
    context: int = 10,
    proposal=multinomial_sample,
    **kwargs,
):
    """
    Standard Ancestral Sampling using a proposal function.
    This sampling is sequential. Given an autoregressive model,
    it generates sequences of length `context` conditioned on the first `guidance` tokens
    for a batch of input sequences. The proposal function is used to sample the next token at each step.

    Args:
        model: The autoregressive model to sample from.
        encoded_input: A dictionary containing 'input_ids' and 'attention_mask' tensors.
        value: Number of samples (particles) to generate per batch element.
        guidance: Number of initial tokens to use as conditioning context to guide the sampling.
        context: Total length of the generated sequence (including guidance).
        proposal: The sampling function to use for generating tokens (e.g., multinomial_sample).
    Returns:
        sampled_tokens: The generated token sequences. Shape [bs*value, context].
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    N = value

    with torch.no_grad():
        # ---- Step 1: initialize ----
        start_tokens = encoded_input["input_ids"][:, :guidance].to(model.device).clone()
        attn_mask = encoded_input["attention_mask"].to(model.device)

        # upsample (repeat N times)
        start_tokens = start_tokens.unsqueeze(1).repeat(1, N, 1).reshape(-1, guidance)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, N, 1).reshape(-1, attn_mask.size(-1))

        for i in range(0, context - guidance):
            output = model(
                input_ids=start_tokens,
                attention_mask=attn_mask[:, : guidance + i].to(model.device),
            )
            # we take the last digit. Be careful if padded
            prob_x = torch.nn.functional.softmax(output["logits"][:, -1, :], dim=-1)
            random_token = proposal(prob_x)
            start_tokens = torch.cat([start_tokens, random_token], dim=-1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print("Ancestral Sampling - Elapsed time: ", elapsed)
    return start_tokens
