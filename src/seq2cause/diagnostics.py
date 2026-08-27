"""Diagnostics for the TRACE intervention construction (`do_interventions`).

Motivated by Chadyuk, Zhang, and Kucukates, "Replicating TRACE: A
Practitioner's Guide to Its Threshold and Particle Budget" (LotusFlare Inc.,
Aug 2026): before assuming their finding -- that recall collapses to near 0
for cause-effect lag >= 2, regardless of tau, under the "full" staircase
construction -- transfers to this codebase's own generator, this module lets
you measure it directly on our own generator, and lets you compare
alternative intervention-construction strategies ("atomic", "windowed",
"independent_mediator", and the "in-distribution-noise" proposal) on the
same held-out sequences.

To isolate the *construction* mechanism from ordinary model-approximation
error, these diagnostics use `seq2cause.scm.NonlinearSCM` as an *oracle*
autoregressive "model": its own exact conditional distribution is used both
to generate the ground-truth sequences and to answer every do-intervention
query (epsilon = 0 by construction, per Assumption 3.6 of Math & Lienhart,
2026). Any recall collapse or CMI magnitude gap measured here is therefore
attributable to `do_interventions` itself, not to imperfect model training.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from seq2cause.sampling import do_interventions, uniform_sample, unigram_sample
from seq2cause.scm import NonlinearSCM

__all__ = [
    "ground_truth_adjacency",
    "snr_by_lag",
    "recall_by_lag",
    "recall_by_lag_with_tau_by_lag",
    "StrategyResult",
    "compare_intervention_strategies",
    "compute_cmi_matrix",
    "compute_cmi_matrix_sparse",
    "estimate_oracle_score",
]


def _bernoulli_kl(q: Tensor, p: Tensor, eps: float = 1e-9) -> Tensor:
    """KL(Bernoulli(q) || Bernoulli(p)), matching `calc_lag_info_gain`'s formula.

    Convention (matches `causal_strength.calc_lag_info_gain`): `q` is the
    "treatment"/do-observed branch, `p` is the "baseline"/do-noised branch.
    """
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))


@torch.no_grad()
def ground_truth_adjacency(
    scm: NonlinearSCM,
    sequence: Tensor,
    threshold: float = 0.05,
    n_counterfactuals: int = 10,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Exact interventional ground truth, reproducing the paper's own
    evaluation protocol (Appendix E.1): for every ordered pair (cause
    position `j`, effect position `q > j`), atomically randomizes *only*
    position `j` (every other position, including mediators, stays at its
    real observed value) and measures the mean KL divergence between the
    SCM's exact conditional at `q` with vs. without the intervention, over
    `n_counterfactuals` draws. An edge exists iff the mean KL exceeds
    `threshold`.

    Because this uses the SCM's *exact* conditional (no do_interventions
    machinery, no Monte-Carlo model approximation), it is independent of
    everything under test below and can be trusted as ground truth.

    Args:
        scm: the oracle generator (also acts as ground truth).
        sequence: 1D integer tensor `[L]`, a single realized sequence.
        threshold: KL threshold defining a true edge (the paper's `delta`,
            confusingly also `tau` for the discovery threshold in other
            places -- here it is the *evaluation* margin, "delta").
        n_counterfactuals: number of random replacement draws averaged per
            pair.
        generator: optional `torch.Generator` for the counterfactual draws.

    Returns:
        Boolean tensor `[L, L]`; `adj[j, q]` is True iff position j is a
        cause of position q. Only entries with `q - scm.memory <= j < q`
        can ever be True (outside that window the SCM has no dependence on
        `j` by construction).
    """
    device = sequence.device
    m = scm.memory
    L = sequence.shape[-1]
    adj = torch.zeros((L, L), dtype=torch.bool, device=device)

    for q in range(m, L):
        context_real = sequence[q - m : q]
        p_true_orig = scm.conditional_probs(context_real)[sequence[q]]

        for j in range(max(0, q - m), q):
            offset = j - (q - m)
            divergences = []
            for _ in range(n_counterfactuals):
                noise_val = int(
                    torch.randint(0, scm.vocab_size, (1,), device=device, generator=generator)
                )
                if noise_val == int(sequence[j]):
                    continue
                cf_context = context_real.clone()
                cf_context[offset] = noise_val
                p_cf = scm.conditional_probs(cf_context)[sequence[q]]
                divergences.append(
                    _bernoulli_kl(p_cf.unsqueeze(0), p_true_orig.unsqueeze(0)).item()
                )
            if divergences and (sum(divergences) / len(divergences)) > threshold:
                adj[j, q] = True
    return adj


def snr_by_lag(
    cmi_matrix: Tensor, adjacency: Tensor, max_lag: int
) -> dict[int, dict[str, float]]:
    """Median CMI on true-edge vs. non-edge (cause, effect) pairs, grouped by
    lag = effect_index - cause_index.

    Extends the "Signal to noise ratio" diagnostic already present in
    `experiment.ipynb` (there, `snr_stats` keyed by lag) into a reusable,
    testable function: this is what should be run *before* assuming Chadyuk
    et al.'s (2026) lag>=2 collapse transfers to a new generator/backbone.

    Args:
        cmi_matrix: `[L_minus_c, L_minus_c]` estimated CMI, `cmi_matrix[j, q]`
            for candidate cause `j` and effect `q`.
        adjacency: `[L_minus_c, L_minus_c]` boolean ground-truth edges, same
            indexing as `cmi_matrix`.
        max_lag: largest lag to report.

    Returns:
        `{lag: {"median_true": ..., "median_false": ..., "n_true": ..., "n_false": ...}}`.
    """
    L = cmi_matrix.shape[-1]
    stats: dict[int, dict[str, float]] = {}
    for lag in range(1, max_lag + 1):
        true_vals: list[float] = []
        false_vals: list[float] = []
        for q in range(lag, L):
            j = q - lag
            val = float(cmi_matrix[j, q])
            (true_vals if adjacency[j, q] else false_vals).append(val)
        stats[lag] = {
            "median_true": float(torch.tensor(true_vals).median()) if true_vals else float("nan"),
            "median_false": float(torch.tensor(false_vals).median())
            if false_vals
            else float("nan"),
            "n_true": len(true_vals),
            "n_false": len(false_vals),
        }
    return stats


def recall_by_lag(
    cmi_matrix: Tensor, adjacency: Tensor, tau: float, max_lag: int
) -> dict[int, float]:
    """Recall of true edges at threshold `tau`, grouped by lag."""
    L = adjacency.shape[-1]
    out: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        tp = 0
        total = 0
        for q in range(lag, L):
            j = q - lag
            if adjacency[j, q]:
                total += 1
                if cmi_matrix[j, q] >= tau:
                    tp += 1
        out[lag] = (tp / total) if total > 0 else float("nan")
    return out


def recall_by_lag_with_tau_by_lag(
    cmi_matrix: Tensor, adjacency: Tensor, tau_by_lag: dict[int, float], max_lag: int
) -> dict[int, float]:
    """Like `recall_by_lag`, but thresholds each lag at its OWN tau instead of
    one shared value.

    CMI magnitude naturally decays with lag -- whether because the do-
    intervention construction loses local context (Chadyuk et al., 2026) or
    simply because the data-generating process's own causal strength decays
    with lag (e.g. `NonlinearSCM`'s `decay_rate`), or both. A single global
    tau, tuned mostly by the (usually much larger) lag-1 population, can
    then systematically under-call deeper lags even when their CMI reliably
    separates true from false edges *at their own scale*. Calibrating tau
    per lag (e.g. via `threshold.select_thresholds_by_group` with
    `groups=lag`) targets the causal strength at that specific delay instead.
    """
    L = adjacency.shape[-1]
    out: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        tau = tau_by_lag.get(lag)
        if tau is None:
            out[lag] = float("nan")
            continue
        tp = 0
        total = 0
        for q in range(lag, L):
            j = q - lag
            if adjacency[j, q]:
                total += 1
                if cmi_matrix[j, q] >= tau:
                    tp += 1
        out[lag] = (tp / total) if total > 0 else float("nan")
    return out


@torch.no_grad()
def estimate_oracle_score(
    adapter,
    scm: NonlinearSCM,
    n_sequences: int = 200,
    length: int = 32,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Estimates the model's oracle score epsilon-hat (paper's Eq. 22),
    epsilon_hat = (L_AR - H(P)) / (H_max - H(P)), on FRESH sequences from
    `scm` (independent of whatever the model was trained on).

    Unlike the common practice of approximating Ĥ by a model's minimum
    observed validation loss (an upper bound on the true H(P) -- see
    Chadyuk et al. 2026's bookkeeping note on exactly this point), `scm`'s
    own exact conditional distribution gives H(P) directly via
    `NonlinearSCM.estimate_entropy`, so this is on the *same* scale as the
    paper's Table 2 (which also used the generator's exact H(P)).

    Returns:
        `{"loss": ..., "h_p": ..., "h_max": ..., "epsilon_hat": ...}`.
    """
    seqs = scm.sample_sequence(length=length, batch_size=n_sequences, generator=generator)
    logits = adapter.forward(input_ids=seqs)["logits"]
    m = scm.memory
    log_probs = torch.log_softmax(logits[:, m - 1 : -1, :], dim=-1)
    targets = seqs[:, m:]
    token_ll = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -token_ll.mean().item()

    h_p = scm.estimate_entropy(n_sequences=n_sequences, length=length, generator=generator)
    h_max = torch.log(torch.tensor(float(scm.vocab_size))).item()
    denom = h_max - h_p
    epsilon_hat = (loss - h_p) / denom if denom > 0 else float("nan")
    return {"loss": loss, "h_p": h_p, "h_max": h_max, "epsilon_hat": epsilon_hat}


def _predicted_true_token_probs(
    scm: NonlinearSCM, rows: Tensor, true_tokens: Tensor, context_len: int
) -> Tensor:
    """For each row (a full `[..., L]` sequence, context + candidate suffix),
    returns the model's predicted probability of the *real* observed suffix
    token at every suffix position, `[..., L_minus_c]`.

    Mirrors `causal_strength.calc_lag_info_gain`'s "Step 1: Align Suffix
    Probs" (prediction for suffix token `i` comes from logits at position
    `context_len + i - 1`), but implemented directly against the oracle SCM
    to avoid pulling in `captum`/`accelerate` for a lightweight diagnostic.
    """
    out = scm.forward(input_ids=rows)
    probs = torch.softmax(out["logits"], dim=-1)  # [..., L, vocab]
    seq_len = rows.shape[-1]
    pred = probs[..., context_len - 1 : seq_len - 1, :]  # [..., Lc, vocab]
    lc = pred.shape[-2]
    idx = true_tokens.view(*([1] * (pred.dim() - 2)), lc, 1).expand(*pred.shape[:-1], 1)
    return torch.gather(pred, dim=-1, index=idx).squeeze(-1)


def _cmi_matrix_from_staircase(p_event_mean: Tensor) -> Tensor:
    """`p_event_mean`: `[num_rows, Lc]` (mean over particles). Row-shift
    comparison (row `r` vs row `r-1`), matching `calc_lag_info_gain`. Row 0
    is used only as a baseline (cause 0 is never assigned a CMI value, by
    the same convention as the original "full" implementation)."""
    num_rows, lc = p_event_mean.shape
    p = p_event_mean[:-1, :]
    q = p_event_mean[1:, :]
    kl = _bernoulli_kl(q, p)  # [num_rows-1, Lc]
    full = torch.zeros((lc, lc))
    start_row = lc - (num_rows - 1)
    full[start_row:, :] = kl
    return full


def _cmi_matrix_from_atomic(p_event_mean: Tensor, baseline_probs: Tensor) -> Tensor:
    """`p_event_mean`: `[Lc(=cause rows), Lc(=effect cols)]`. `baseline_probs`:
    `[Lc]`, the fully-real (no intervention) predicted true-token
    probability. Only the upper triangle (`j < q`) is meaningful."""
    lc = p_event_mean.shape[-1]
    baseline_exp = baseline_probs.unsqueeze(0).expand(lc, lc)
    kl = _bernoulli_kl(baseline_exp, p_event_mean)
    mask = torch.triu(torch.ones(lc, lc, dtype=torch.bool), diagonal=1)
    return torch.where(mask, kl, torch.zeros_like(kl))


@torch.no_grad()
def compute_cmi_matrix(
    model,
    sequence: Tensor,
    context_len: int,
    n_particles: int = 32,
    strategy: str = "atomic",
    max_pairs: int | None = 20000,
) -> Tensor:
    """Computes the per-(cause, effect) Conditional Mutual Information matrix
    for a single sequence, using a single do-intervention `strategy`.

    This is the lightweight, no-ground-truth-required entry point for actual
    use (see README Quick Start): unlike `compare_intervention_strategies`
    (which computes all 5 strategies at once purely to compare them against
    known ground truth for research/diagnostic purposes), this only runs the
    one strategy you ask for and never requires an `adjacency` argument.

    Defaults to `strategy="atomic"`: only the candidate-cause position is
    randomized (every other position, including mediators, stays real),
    which does not collapse recall at cause-effect lag >= 2 the way the
    paper's original `strategy="full"` staircase does (Chadyuk et al., 2026;
    see README "Alternative Intervention Constructions").

    Args:
        model: anything satisfying `.vocab_size` + a HuggingFace-style
            `forward(input_ids=...)` returning a dict with a `"logits"` key
            (e.g. a real model wrapped in `seq2cause.adapters.HFModelAdapter`,
            or `seq2cause.scm.NonlinearSCM` for validation against known
            ground truth).
        sequence: `[L]` single sequence (context + candidate-effect suffix).
        context_len: length of the always-real fixed prefix/context.
        n_particles: number of do-intervention noise particles.
        strategy: `"atomic"` (default, recommended) or `"full"` (the
            paper's original staircase). Use `compare_intervention_strategies`
            for `"windowed"`/`"independent_mediator"`/`"in_distribution_noise"`.
        max_pairs: forwarded to `do_interventions` as an O(L^2) guard.

    Returns:
        `[L - context_len, L - context_len]` CMI matrix; `[j, q]` is the
        estimated causal strength of candidate cause `j` on candidate
        effect `q`.
    """
    if strategy not in ("full", "atomic"):
        raise ValueError(
            f"strategy must be 'full' or 'atomic', got {strategy!r} -- use "
            "compare_intervention_strategies for 'windowed'/'independent_mediator'/"
            "'in_distribution_noise'."
        )
    device = sequence.device
    seq_len = sequence.shape[-1]
    lc = seq_len - context_len
    prefix = sequence[:context_len].unsqueeze(0)  # [1, c]
    rest = sequence[context_len:].unsqueeze(0)  # [1, Lc]

    dummy = torch.zeros(1, lc, model.vocab_size, device=device)
    noise = uniform_sample(dummy, n_samples=n_particles, device=device)

    if strategy == "full":
        rows = do_interventions(noise, rest, prefix, strategy="full").squeeze(0)
        p = _predicted_true_token_probs(model, rows, rest.squeeze(0), context_len)
        return _cmi_matrix_from_staircase(p.mean(dim=0))

    rows = do_interventions(noise, rest, prefix, strategy="atomic", max_pairs=max_pairs).squeeze(0)
    p_atomic = _predicted_true_token_probs(model, rows, rest.squeeze(0), context_len)
    baseline_rows = sequence.unsqueeze(0)  # [1, L], fully real, no intervention
    p_baseline = _predicted_true_token_probs(model, baseline_rows, rest.squeeze(0), context_len)
    return _cmi_matrix_from_atomic(p_atomic.mean(dim=0), p_baseline.squeeze(0))


@torch.no_grad()
def compute_cmi_matrix_sparse(
    model,
    sequence: Tensor,
    context_len: int,
    memory: int,
    n_particles: int = 32,
) -> Tensor:
    """Bounded-memory ("sparse") variant of `compute_cmi_matrix`.

    Assumes the true causal structure has no edges beyond lag `memory`
    (matching e.g. a `NonlinearSCM(memory=memory)` generator, or any
    autoregressive process with a known/assumed finite receptive field).
    Instead of running the "full" staircase once on the WHOLE sequence
    (`compute_cmi_matrix(strategy="full")`: O(L) rows, each of length O(L)),
    this slides a LOCAL window across the sequence and runs the SAME "full"
    staircase construction independently on each short local slice -- O(L)
    total rows, but each of length O(memory) instead of O(L). Since a
    transformer's causal attention means logits at position `t` never
    depend on anything after `t` anyway, and a memory-bounded generator's
    conditional at `t` never depends on anything before `t - memory`, this
    is exact (not an approximation): whenever `memory` truly bounds the
    generator's memory, this recovers the SAME causal graph as the
    unbounded "full" computation (see `tests/test_scm_and_diagnostics.py`
    for an empirical full-vs-sparse comparison on a decayed, memory-bounded
    `NonlinearSCM`), for a fraction of the compute on long sequences.

    Construction, per non-overlapping chunk of `memory` new effect
    positions `[chunk_start, chunk_end)` (Lc-relative):
      - local, testable suffix = `[chunk_start - memory, chunk_end)`
        (Lc-relative; the `memory` positions immediately before this chunk
        are included so cross-chunk-boundary causes can still be tested,
        not just intra-chunk ones).
      - local, FIXED (never-intervened) context = the `memory` real tokens
        immediately before that local suffix (or fewer, near the start of
        `sequence` -- see `context_len` requirement below).
      - `compute_cmi_matrix(..., strategy="full")` is run on this short
        local slice; only the cells attributing an effect that's actually
        NEW to this chunk (and whose cause isn't inside the original,
        always-fixed `context_len` prefix) are written into the full
        `[Lc, Lc]` result -- every valid (lag <= memory) cell is written by
        exactly one chunk, so there is no double-counting.

    Args:
        model: same interface as `compute_cmi_matrix`.
        sequence: `[L]` (context + suffix).
        context_len: length of the always-real fixed prefix/context. Must
            be `> memory` (strictly) so every local computation has a
            non-empty fixed context of its own.
        memory: assumed maximum true causal lag; only cells with
            `1 <= lag <= memory` are ever computed -- everything else stays
            0 (never tested, not "found to be zero").
        n_particles: number of do-intervention noise particles per chunk.

    Returns:
        `[L - context_len, L - context_len]` CMI matrix, same convention as
        `compute_cmi_matrix(strategy="full")`.
    """
    if context_len <= memory:
        raise ValueError(
            f"context_len ({context_len}) must be > memory ({memory}) -- every local "
            "chunk needs its own non-empty fixed context."
        )
    seq_len = sequence.shape[-1]
    lc = seq_len - context_len
    full_cmi = torch.zeros((lc, lc), device=sequence.device)

    chunk_start = 0
    while chunk_start < lc:
        chunk_end = min(chunk_start + memory, lc)

        local_suffix_start_lc = chunk_start - memory  # may be negative
        local_suffix_start_abs = context_len + local_suffix_start_lc
        local_context_start_abs = max(0, local_suffix_start_abs - memory)
        local_context_len = local_suffix_start_abs - local_context_start_abs
        local_suffix_end_abs = context_len + chunk_end

        local_sequence = sequence[local_context_start_abs:local_suffix_end_abs]
        local_cmi = compute_cmi_matrix(
            model, local_sequence, context_len=local_context_len,
            n_particles=n_particles, strategy="full",
        )
        local_suffix_len = local_suffix_end_abs - local_suffix_start_abs

        for local_j in range(local_suffix_len):
            global_j_lc = local_suffix_start_lc + local_j
            if global_j_lc < 0:
                continue  # cause falls inside the original fixed context -- never tested
            for local_q in range(local_j + 1, local_suffix_len):
                if local_q - local_j > memory:
                    continue
                global_q_lc = local_suffix_start_lc + local_q
                if not (chunk_start <= global_q_lc < chunk_end):
                    continue  # this effect belongs to a different chunk (already written)
                full_cmi[global_j_lc, global_q_lc] = local_cmi[local_j, local_q]

        chunk_start += memory

    return full_cmi


def _cmi_matrix_from_windowed(
    p_event_mean: Tensor, pairs: list[tuple[int, int]], lc: int
) -> Tensor:
    """`p_event_mean`: `[num_pairs, Lc]`, one row per *far* (lag > window_k)
    `(cause, effect)` pair (see `sampling.windowed_pair_index`). Groups rows
    by effect `q`, and within each group performs the same row-shift
    comparison as `_cmi_matrix_from_staircase`, restricted to that single
    effect column.

    Cells for near-lag pairs (`lag <= window_k`), which `windowed_pair_index`
    deliberately excludes, are left at 0 here -- `compare_intervention_strategies`
    fills them in from the "full" strategy's own CMI matrix instead, since
    those pairs are indistinguishable from strategy="full" by construction
    (the always-preserved window already makes the cause real in every row).
    """
    full = torch.zeros((lc, lc))
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_idx, (j, q) in enumerate(pairs):
        groups[q].append((j, row_idx))

    for q, items in groups.items():
        items.sort(key=lambda pair: pair[0])
        row_indices = [row_idx for _j, row_idx in items]
        p_q = p_event_mean[row_indices, q]  # [num_causes_for_q], ordered by j=0..q-1
        if p_q.numel() < 2:
            continue
        p = p_q[:-1]
        qv = p_q[1:]
        kl = _bernoulli_kl(qv, p)  # causes j=1..q-1
        for idx in range(kl.numel()):
            full[idx + 1, q] = kl[idx]
    return full


@dataclass
class StrategyResult:
    """Per-strategy diagnostic output for `compare_intervention_strategies`."""

    strategy: str
    cmi_matrix: Tensor
    snr: dict[int, dict[str, float]]
    recall: dict[int, float] | None = None
    extra: dict[str, object] = field(default_factory=dict)


@torch.no_grad()
def compare_intervention_strategies(
    scm: NonlinearSCM,
    sequence: Tensor,
    context_len: int,
    adjacency: Tensor,
    n_particles: int = 32,
    window_k: int = 1,
    tau: float | None = None,
    max_lag: int | None = None,
    unigram_freqs: Tensor | None = None,
    max_pairs: int | None = 20000,
    generator: torch.Generator | None = None,
) -> dict[str, StrategyResult]:
    """Runs "full", "atomic", "windowed", "independent_mediator", and
    "in-distribution-noise" (a "full" staircase using `unigram_sample`
    instead of `uniform_sample`) on the *same* held-out sequence, and reports
    per-lag CMI magnitude (`.snr`) and, if `tau` is given, per-lag recall
    (`.recall`) for each -- so the four candidate explanations for the
    lag>=2 collapse (context loss, OOD noise, non-independent draws, or a
    combination) can be compared directly.

    Args:
        scm: the oracle model (also the ground-truth generator).
        sequence: `[L]` single sequence (context + suffix).
        context_len: length of the always-real fixed prefix/context.
        adjacency: `[L, L]` ground-truth edges (see `ground_truth_adjacency`).
        n_particles: number of do-intervention noise particles.
        window_k: local-context radius for the "windowed" strategy.
        tau: if given, also computes per-lag recall at this threshold.
        max_lag: largest lag to report; defaults to `scm.memory`.
        unigram_freqs: required for the "in-distribution-noise" strategy; a
            1D frequency/probability tensor over the vocabulary.
        max_pairs: forwarded to `do_interventions` as a guard for the
            O(L^2) strategies ("atomic", "windowed").
        generator: optional `torch.Generator` for reproducible noise draws.

    Returns:
        `{strategy_name: StrategyResult}`.
    """
    device = sequence.device
    seq_len = sequence.shape[-1]
    lc = seq_len - context_len
    max_lag = max_lag if max_lag is not None else scm.memory
    prefix = sequence[:context_len].unsqueeze(0)  # [1, c]
    rest = sequence[context_len:].unsqueeze(0)  # [1, Lc]
    adjacency_suffix = adjacency[context_len:, context_len:]

    def _noise(vocab_size, n, length, kind="uniform"):
        # `uniform_sample`/`unigram_sample` only use `prob_x` to infer shape,
        # so a zero-filled placeholder of the right shape suffices here --
        # this exercises the real, pluggable proposal functions rather than
        # reimplementing sampling logic inline.
        dummy = torch.zeros(1, length, vocab_size, device=device)
        if kind == "uniform":
            return uniform_sample(dummy, n_samples=n, device=device)
        if kind == "unigram":
            if unigram_freqs is None:
                raise ValueError("unigram_freqs is required for the in-distribution-noise strategy")
            return unigram_sample(dummy, n_samples=n, device=device, unigram_freqs=unigram_freqs)
        raise ValueError(kind)

    results: dict[str, StrategyResult] = {}

    def _finalize(name, cmi_matrix, extra=None):
        snr = snr_by_lag(cmi_matrix, adjacency_suffix, max_lag)
        recall = recall_by_lag(cmi_matrix, adjacency_suffix, tau, max_lag) if tau is not None else None
        results[name] = StrategyResult(
            strategy=name, cmi_matrix=cmi_matrix, snr=snr, recall=recall, extra=extra or {}
        )

    # --- "full" (default staircase) ---
    noise_full = _noise(scm.vocab_size, n_particles, lc, "uniform")
    rows_full = do_interventions(noise_full, rest, prefix, strategy="full")
    rows_full = rows_full.squeeze(0)  # [N, num_rows, L]
    p_full = _predicted_true_token_probs(scm, rows_full, rest.squeeze(0), context_len)
    full_cmi = _cmi_matrix_from_staircase(p_full.mean(dim=0))
    _finalize("full", full_cmi)

    # --- "atomic" ---
    noise_atomic = _noise(scm.vocab_size, n_particles, lc, "uniform")
    rows_atomic = do_interventions(
        noise_atomic, rest, prefix, strategy="atomic", max_pairs=max_pairs
    )
    rows_atomic = rows_atomic.squeeze(0)  # [N, Lc(rows), L]
    p_atomic = _predicted_true_token_probs(scm, rows_atomic, rest.squeeze(0), context_len)
    baseline_rows = sequence.unsqueeze(0)  # [1, L], fully real, no intervention
    p_baseline = _predicted_true_token_probs(scm, baseline_rows, rest.squeeze(0), context_len)
    _finalize("atomic", _cmi_matrix_from_atomic(p_atomic.mean(dim=0), p_baseline.squeeze(0)))

    # --- "windowed" ---
    # Near-lag pairs (lag <= window_k) are excluded from `pairs` by
    # `windowed_pair_index` (they degenerate to CMI=0 by construction, not by
    # causal strength -- see its docstring); those cells are filled in from
    # `full_cmi` instead, so the reported matrix combines "full" for near
    # lags with the context-preserving construction for far lags.
    noise_windowed = _noise(scm.vocab_size, n_particles, lc, "uniform")
    rows_windowed, pairs = do_interventions(
        noise_windowed,
        rest,
        prefix,
        strategy="windowed",
        window_k=window_k,
        max_pairs=max_pairs,
    )
    rows_windowed = rows_windowed.squeeze(0)  # [N, num_pairs, L]
    cmi_windowed = full_cmi.clone()
    if rows_windowed.shape[1] > 0:
        p_windowed = _predicted_true_token_probs(scm, rows_windowed, rest.squeeze(0), context_len)
        far_cmi = _cmi_matrix_from_windowed(p_windowed.mean(dim=0), pairs, lc)
        far_mask = torch.zeros((lc, lc), dtype=torch.bool)
        for j, q in pairs:
            far_mask[j, q] = True
        cmi_windowed[far_mask] = far_cmi[far_mask]
    _finalize("windowed", cmi_windowed, extra={"window_k": window_k})

    # --- "independent_mediator" ---
    noise_cause = _noise(scm.vocab_size, n_particles, lc, "uniform")
    noise_mediator = _noise(scm.vocab_size, n_particles, lc, "uniform")
    rows_indep = do_interventions(
        noise_cause,
        rest,
        prefix,
        strategy="independent_mediator",
        rest_upsampled_mediator=noise_mediator,
    )
    rows_indep = rows_indep.squeeze(0)
    p_indep = _predicted_true_token_probs(scm, rows_indep, rest.squeeze(0), context_len)
    _finalize("independent_mediator", _cmi_matrix_from_staircase(p_indep.mean(dim=0)))

    # --- "in-distribution-noise" (unigram proposal on top of the "full" mask) ---
    if unigram_freqs is not None:
        noise_unigram = _noise(scm.vocab_size, n_particles, lc, "unigram")
        rows_unigram = do_interventions(noise_unigram, rest, prefix, strategy="full")
        rows_unigram = rows_unigram.squeeze(0)
        p_unigram = _predicted_true_token_probs(scm, rows_unigram, rest.squeeze(0), context_len)
        _finalize("in_distribution_noise", _cmi_matrix_from_staircase(p_unigram.mean(dim=0)))

    return results
