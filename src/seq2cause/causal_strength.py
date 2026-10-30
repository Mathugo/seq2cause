from __future__ import annotations

import torch
from captum.attr import InputXGradient, ShapleyValueSampling
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from seq2cause.kl import (
    DEFAULT_KL_MODE,
    LOG_CLAMP,
    accumulate_counts,
    bernoulli_kl_from_log,
    check_kl_mode,
    corrupted_cell_counts,
    log1mexp,
    wide_dtype,
)
from seq2cause.utils import estimate_tensor_bytes, format_bytes


def _band_mask(num_pairs: int, lc: int, start_row: int, device) -> Tensor:
    """`[num_pairs, lc]` boolean mask of the cells that are read from the
    row-pair window: row `r` tests cause `start_row + r`, and only effects
    strictly after the cause (`t > start_row + r`) are meaningful."""
    causes = torch.arange(num_pairs, device=device) + start_row
    effects = torch.arange(lc, device=device)
    return effects.unsqueeze(0) > causes.unsqueeze(1)


def calc_lag_info_gain(
    prob_x_inter: Float[Tensor, "bs N num_rows L V"],
    batch: dict[str, Float[Tensor, "bs L"]],
    params: dict,
    eps: float = 1e-9,
    kl_mode: str | None = None,
    stats: dict | None = None,
) -> Float[Tensor, "bs L_minus_c L_minus_c"]:
    """Calculates Lagged Information Gain (KL Divergence) between the intervention and baseline distributions.
    This method quantifies how much information about the true token is gained by observing the cause (intervention)
    compared to not observing it (baseline). It is the information gain of the OBSERVED cause
    (`e_t = 1`), averaged over the particles: it does not average over the two values of the
    cause, so it is not the conditional mutual information and is not bounded by `log 2`
    (see `calc_lag_cmi` for the CMI).
    Args:
        prob_x_inter: The next-token distributions obtained from the intervention sampling. In
            `"logspace"` mode (the default) these are LOG-probabilities (`log_softmax` of the
            logits); in `"clamp"` mode they are probabilities (`softmax`), as in v0.1.9.
        batch: The original input batch containing 'input_ids' and 'attention_mask'.
        params: A dictionary of parameters, including sampling context and clamping epsilon.
        eps: A small constant to prevent log(0) (`"clamp"` mode only).
        kl_mode: `"logspace"` evaluates the Bernoulli KL from log-probabilities in float64, so a
            next-token probability that saturates to `1.0` in float32 stays finite; `"clamp"`
            is the v0.1.9 algebra, whose `1 - eps` clamp is a no-op at the upper end in
            float32 (see `seq2cause.kl`). Defaults to `params.get("kl_mode", "logspace")`.
        stats: An optional dict into which the corrupted-cell counts (what `"clamp"` mode
            turns into NaN or +inf on these inputs) are accumulated, in either mode. Defaults
            to `params.get("kl_stats")`; `None` skips the count.

    Returns:
        A tensor of shape [bs, L_minus_c, L_minus_c] representing the lagged information gain for each potential cause-effect pair.
    """

    if kl_mode is None:
        kl_mode = params.get("kl_mode", DEFAULT_KL_MODE)
    check_kl_mode(kl_mode)
    if stats is None:
        stats = params.get("kl_stats")

    device = prob_x_inter.device
    c = params["sampling"]["context"]
    bs, N, num_rows, L, V = prob_x_inter.shape
    input_ids = batch["input_ids"]
    Lc = input_ids.shape[1] - c
    suffix_tokens = input_ids[:, c:]  # [bs, Lc]

    # --- Step 1: Align Suffix Probs ---
    # Prediction for token t comes from logit t-1
    prob_suffix = prob_x_inter[:, :, :, c - 1 : L - 1, :]  # [bs, N, num_rows, Lc, V]

    gather_ids = (
        suffix_tokens.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, N, num_rows, -1, 1)
    )
    p_event_full = torch.gather(prob_suffix, dim=-1, index=gather_ids).squeeze(
        -1
    )  # [bs, N, num_rows, Lc]

    # --- Step 2: Handle the Baseline for the First Token ---
    # To test the first cause in our window, we need a baseline where it's NOT there.
    # We can't just use zeros. We use a tiny constant (ε) or the previous row.
    # If using sparse, we assume any cause BEFORE the window has 0 gain.

    p = p_event_full[:, :, :-1, :]  # Baseline (X_j is noise)
    q = p_event_full[:, :, 1:, :]  # Interv (X_j is observed)

    # The causes we just tested are the LAST (num_rows-1) causes.
    # If m=6 and Lc=12, we tested causes 7, 8, 9, 10, 11 (indices 6-11).
    start_row = Lc - (num_rows - 1)

    if stats is not None:
        band = _band_mask(num_rows - 1, Lc, start_row, device)
        q32, p32 = (q, p) if kl_mode == "clamp" else (q.exp(), p.exp())
        accumulate_counts(stats, corrupted_cell_counts(q32, p32, band=band, particle_dim=1))

    if kl_mode == "clamp":
        # v0.1.9 algebra, verbatim.
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)

        kl = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
        cmi_window = torch.mean(kl, dim=1)  # [bs, num_rows-1, Lc]
    else:
        kl = bernoulli_kl_from_log(q, p)  # float64, finite for every finite input
        cmi_window = torch.mean(kl, dim=1).to(prob_x_inter.dtype)  # [bs, num_rows-1, Lc]

    # --- Step 3: Sparse-to-Full Matrix Mapping ---
    # We need to map these (num_rows-1) causes to their actual positions in the (Lc x Lc) matrix
    full_cmi = torch.zeros((bs, Lc, Lc), device=device)
    full_cmi[:, start_row:, :] = cmi_window

    return torch.nan_to_num(full_cmi, nan=0.0)


def calc_lag_cmi(
    prob_x_inter: Float[Tensor, "bs N num_rows L V"],
    batch: dict[str, Float[Tensor, "bs L"]],
    params: dict,
    eps: float = 1e-9,
    kl_mode: str | None = None,
) -> Float[Tensor, "bs L_minus_c L_minus_c"]:
    """Conditional mutual information `I(E_t', E_t | X_<t)` between a cause and an effect.

    `calc_lag_info_gain` returns the information gain of the observed cause only. The CMI
    averages the information gain over both values of the binary cause indicator `E_t`:

        pi  = P(E_t = 1 | x_<t)              model probability of the observed cause token
        q1  = P(E_t' = 1 | E_t = 1, x_<t)    effect probability, cause observed
        q0  = P(E_t' = 1 | E_t = 0, x_<t)    effect probability, cause replaced by noise
        m   = pi * q1 + (1 - pi) * q0        effect probability without knowing E_t
        CMI = pi * KL(Bern(q1) || Bern(m)) + (1 - pi) * KL(Bern(q0) || Bern(m))

    per particle, then averaged over the N particles. This is the mutual information between
    a Bernoulli(pi) variable and a binary effect, so `0 <= CMI <= log 2`. Both an excitatory
    (q1 > q0) and an inhibitory (q1 < q0) cause give a positive value.

    The rows of `prob_x_inter` follow the "full" staircase: row r keeps the suffix positions
    0..r observed and noises the later ones, so rows r and r-1 differ only at the cause r.
    `E_t = 0` is realised by the noise token of the proposal (uniform by default), which
    differs from the observed token with probability 1 - 1/V.

    Args:
        prob_x_inter: Next-token LOG-probabilities in `"logspace"` mode (the default), or
            probabilities in `"clamp"` mode, as for `calc_lag_info_gain`.
        batch: The original input batch containing 'input_ids'.
        params: Parameters, including `params["sampling"]["context"]`.
        eps: Clamping constant for probabilities in `"clamp"` mode.
        kl_mode: `"logspace"` or `"clamp"`; defaults to `params.get("kl_mode", "logspace")`.

    Returns:
        A tensor of shape [bs, L_minus_c, L_minus_c]; entry [b, cause, effect] is the CMI.
    """
    if kl_mode is None:
        kl_mode = params.get("kl_mode", DEFAULT_KL_MODE)
    check_kl_mode(kl_mode)

    device = prob_x_inter.device
    c = params["sampling"]["context"]
    bs, N, num_rows, L, V = prob_x_inter.shape
    input_ids = batch["input_ids"]
    Lc = input_ids.shape[1] - c
    suffix_tokens = input_ids[:, c:]  # [bs, Lc]

    # Probability of the observed token at every suffix position, per particle and row.
    prob_suffix = prob_x_inter[:, :, :, c - 1 : L - 1, :]  # [bs, N, num_rows, Lc, V]
    gather_ids = (
        suffix_tokens.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, N, num_rows, -1, 1)
    )
    p_event = torch.gather(prob_suffix, dim=-1, index=gather_ids).squeeze(-1)

    dtype = wide_dtype(p_event)
    if kl_mode == "clamp":
        log_p_event = torch.log(torch.clamp(p_event.to(dtype), eps, 1 - eps))
    else:
        log_p_event = p_event.to(dtype)

    log_q0 = log_p_event[:, :, :-1, :]  # cause noised  [bs, N, R, Lc]
    log_q1 = log_p_event[:, :, 1:, :]  # cause observed [bs, N, R, Lc]

    # pi: probability of the observed cause token given its trajectory, read in the
    # cause-observed row at the cause position (the tokens before it are the same in both rows).
    start_row = Lc - (num_rows - 1)
    cause_pos = torch.arange(num_rows - 1, device=device) + start_row  # [R]
    idx = cause_pos.view(1, 1, -1, 1).expand(bs, N, -1, 1)
    log_pi = torch.gather(log_q1, dim=-1, index=idx)  # [bs, N, R, 1]
    log_pi = torch.clamp(log_pi, max=LOG_CLAMP)
    log_1m_pi = log1mexp(log_pi)

    log_m = torch.logaddexp(log_pi + log_q1, log_1m_pi + log_q0)
    kl1 = bernoulli_kl_from_log(log_q1, log_m)
    kl0 = bernoulli_kl_from_log(log_q0, log_m)
    cmi = torch.exp(log_pi) * kl1 + torch.exp(log_1m_pi) * kl0  # [bs, N, R, Lc]
    cmi_window = torch.mean(cmi, dim=1).to(prob_x_inter.dtype)  # [bs, R, Lc]

    # Only effects strictly after the cause are meaningful.
    band = _band_mask(num_rows - 1, Lc, start_row, device)
    cmi_window = cmi_window * band

    full_cmi = torch.zeros((bs, Lc, Lc), device=device, dtype=prob_x_inter.dtype)
    full_cmi[:, start_row:, :] = cmi_window
    return torch.nan_to_num(full_cmi, nan=0.0)


def calc_neural_shapley(
    tfx: any, batch: dict[str, Float[Tensor, "bs L"]], params: dict
) -> Float[Tensor, "bs L_minus_c L_minus_c"]:
    """Calculates Shapley Values using Captum's ShapleyValueSampling.
    This method estimates the contribution of each token in the history to the prediction of each token in the suffix.
    It does this by treating the autoregressive model as a black box and sampling different subsets of the input sequence
    to see how the prediction changes.
    """

    device = tfx.device
    tfx.eval()
    tfx.zero_grad()

    input_ids = batch["input_ids"]
    c = params["sampling"]["context"]
    bs, L = input_ids.shape

    # Use Pad token or 0 as baseline
    pad_token_id = tfx.config.pad_token_id if tfx.config.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, pad_token_id)

    # --- 1. Define Forward Function ---
    def forward_func(inputs, target_pos):
        # inputs: [bs, L] (These are the PERTURBED inputs from Captum)
        # target_pos: Integer (The time step 't' we passed via additional_forward_args)

        # We predict token at 'target_pos', so we need logit at 'target_pos - 1'
        pred_idx = target_pos - 1

        # We must use 'inputs' (the perturbed sequence), NOT global input_ids
        outputs = tfx(input_ids=inputs)
        logits = outputs.logits  # [bs, L, V]

        # Extract logits for the relevant step
        step_logits = logits[:, pred_idx, :]  # [bs, V]
        probs = torch.softmax(step_logits, dim=-1)

        # We want the probability of the TRUE ORIGINAL token
        # We get this from the original batch (global scope), not the perturbed inputs
        true_token_id = input_ids[:, target_pos]  # [bs]

        # Gather the prob for the true class
        target_probs = torch.gather(probs, 1, true_token_id.unsqueeze(1)).squeeze(1)

        return target_probs  # Returns [bs] scalar per sample

    # --- 2. Initialize Sampler ---
    shapley = ShapleyValueSampling(forward_func)

    final_adj = torch.zeros((bs, L, L), device=device)

    # Rough VRAM/RAM estimate: `perturbations_per_eval` forward passes of
    # shape [perturbations_per_eval, L, hidden_size] per suffix position --
    # see utils.estimate_tensor_bytes's caveats (lower bound, not exact).
    perturbations_per_eval = bs * 4
    hidden_size = getattr(tfx.config, "hidden_size", None)
    est_bytes = (
        estimate_tensor_bytes(perturbations_per_eval, L, hidden_size)
        if hidden_size is not None
        else 0
    )
    print(
        f"Computing Shapley for suffix (L={L}, bs={bs}), "
        f"est. VRAM/RAM per position: {format_bytes(est_bytes) if est_bytes else 'n/a'}"
    )

    # --- 3. Iterate ---
    for t in tqdm(range(c, L), desc="CI-tests (batch x context), Shapley", unit="pos"):
        attributions = shapley.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(t,),  # <--- Passes 't' to 'target_pos'
            n_samples=10,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=False,
        )

        final_adj[:, :, t] = attributions.detach()

    # 4. Cleanup
    final_adj = final_adj[:, c:, c:]

    return batch, final_adj


def calc_neural_saliency(
    tfx: any, batch: dict[str, Float[Tensor, "bs L"]], params: dict[str, any]
) -> Float[Tensor, "bs L_minus_c L_minus_c"]:
    """
    Computes Saliency using Captum (Input x Gradient).
    """
    device = tfx.device
    tfx.eval()
    tfx.zero_grad()

    input_ids = batch["input_ids"]
    c = params["sampling"]["context"]
    bs, L = input_ids.shape

    # 1. Define a Wrapper Function
    # Captum needs a function that takes embeddings and outputs a scalar (the target logit/prob)
    def forward_func(inputs_embeds, target_index):
        # inputs_embeds: [bs, L, hidden]
        outputs = tfx(inputs_embeds=inputs_embeds)
        logits = outputs.logits  # [bs, L, V]

        # We want the logit for the specific target token at position 'target_index'
        # The prediction for token at 't' comes from 't-1'
        pred_idx = target_index - 1

        # Get logits at the prediction step
        step_logits = logits[:, pred_idx, :]  # [bs, V]
        return step_logits  # Captum will select the target class from this

    # 2. Initialize Attribute Method
    # InputXGradient is usually superior to vanilla Saliency for NLP
    saliency = InputXGradient(forward_func)

    # 3. Get Input Embeddings
    input_embeds = tfx.get_input_embeddings()(input_ids).detach()
    input_embeds.requires_grad = True

    final_adj = torch.zeros((bs, L - c, L - c), device=device)

    # Rough VRAM/RAM estimate for the tracked embedding tensor (the biggest
    # tensor Captum keeps gradients for here) -- see
    # utils.estimate_tensor_bytes's caveats (lower bound, not exact).
    hidden_size = getattr(tfx.config, "hidden_size", None)
    est_bytes = estimate_tensor_bytes(bs, L, hidden_size) if hidden_size is not None else 0
    print(
        f"Computing Saliency for suffix (L={L}, bs={bs}), "
        f"est. VRAM/RAM: {format_bytes(est_bytes) if est_bytes else 'n/a'}"
    )

    # 4. Iterate over Suffix Targets
    # We explain why token 't' happened, based on history
    for i, t in enumerate(
        tqdm(range(c, L), desc="CI-tests (batch x context), Saliency", unit="pos")
    ):
        target_class = input_ids[:, t]  # [bs] (The token indices we want to explain)

        # Compute attribution
        # inputs: embedding tensor
        # target: the index of the class we want to explain (the true token)
        # additional_forward_args: passed to forward_func (the time index)
        attributions = saliency.attribute(
            inputs=input_embeds, target=target_class, additional_forward_args=(t,)
        )

        # Attributions shape: [bs, L, hidden]
        # Summarize hidden dim (L2 norm) to get scalar score per token
        score_per_token = torch.norm(attributions, p=2, dim=-1)  # [bs, L]

        # We want the influence of 'history' on 'target t'
        # The target 't' corresponds to column 'i' in our adj matrix
        # We only care about history (indices < t)

        # Slice out the suffix part [c:L]
        # This gives us the influence of suffix tokens on the current target
        # (plus we ignore influence of tokens > t, which should be 0 anyway)
        valid_influence = score_per_token[:, c:]  # [bs, L-c]

        final_adj[:, :, i] = valid_influence
    return batch, final_adj


def calc_granger_score(
    prob_x_inter: Float[Tensor, "bs N num_rows L V"],
    batch: dict[str, Float[Tensor, "bs L"]],
    params: dict[str, any],
    mode: str = "diff",
) -> Float[Tensor, "bs L_minus_c L_minus_c"]:
    """
    Computes Granger Causality as the 'Probability Gain' of the true token.
    Compares P(Effect | Cause + History) vs P(Effect | History).
    """
    device = prob_x_inter.device
    c = params["sampling"]["context"]
    bs, N, num_rows, L, V = prob_x_inter.shape
    input_ids = batch["input_ids"]

    # 1. Identify Target Tokens (The "Effect")
    # We only care about the probability assigned to the TRUE observed token
    Lc = input_ids.shape[1] - c
    suffix_tokens = input_ids[:, c:]  # [bs, Lc]

    # 2. Gather Probabilities
    # We want predictions for the suffix starting at c.
    # Prediction for token at t comes from logit at t-1.
    prob_suffix = prob_x_inter[:, :, :, c - 1 : L - 1, :]  # [bs, N, num_rows, Lc, V]

    # Gather only the probability of the ACTUAL token that happened
    # suffix_tokens: [bs, Lc] -> [bs, N, num_rows, Lc, 1]
    gather_ids = (
        suffix_tokens.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(-1, N, num_rows, -1, 1)
    )

    # p_event_full[b, n, row, t] = Probability of True Token t given Staircase Row "row"
    p_event_full = torch.gather(prob_suffix, dim=-1, index=gather_ids).squeeze(-1)

    # 3. Handle First Token Baseline
    # Row j: Fixes history up to j.
    # We compare Row j (Cause Observed) vs Row j-1 (Cause Noised).

    # Baseline (P): Cause is Noise (Row j-1)
    p_base = p_event_full[:, :, :-1, :]

    # Intervention (Q): Cause is Observed (Row j)
    p_inter = p_event_full[:, :, 1:, :]

    # 4. Compute Granger Score
    if mode == "diff":
        # Simple Probability Difference: How much did the probability mass increase?
        # Range: [-1, 1]. Positive = Causal. Negative = Inhibitory/Noise.
        score_raw = p_inter - p_base

    elif mode == "log_ratio":
        # Log-Likelihood Ratio (Standard in Information Theory)
        # Identical to Pointwise Mutual Information (PMI)
        eps = 1e-9
        score_raw = torch.log(p_inter + eps) - torch.log(p_base + eps)

    else:
        raise ValueError(f"mode must be 'diff' or 'log_ratio', got {mode!r}")

    # Average over particles
    score_window = torch.mean(score_raw, dim=1)  # [bs, num_rows-1, Lc]

    # 5. Map back to Full Matrix (Padding if sparse)
    full_score = torch.zeros((bs, Lc, Lc), device=device)
    start_row = Lc - (num_rows - 1)
    full_score[:, start_row:, :] = score_window

    return torch.nan_to_num(full_score, nan=0.0)
