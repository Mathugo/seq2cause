import torch
from captum.attr import InputXGradient, ShapleyValueSampling
from jaxtyping import Float
from torch import Tensor


def calc_lag_info_gain(
    prob_x_inter: Float[Tensor, "bs N num_rows L V"],
    batch: dict[str, Float[Tensor, "bs L"]],
    params: dict,
    eps: float = 1e-9,
) -> Float[Tensor, "bs L_minus_c L_minus_c"]:
    """Calculates Lagged Information Gain (KL Divergence) between the intervention and baseline distributions.
    This method quantifies how much information about the true token is gained by observing the cause (intervention)
    compared to not observing it (baseline). By averaging over the particles, this is equivalent to the conditional
    mutual information I(X_t; X'_t | History) for each potential cause X_t' and effect X_t.
    Args:
        prob_x_inter: The probability distributions obtained from the intervention sampling.
        batch: The original input batch containing 'input_ids' and 'attention_mask'.
        params: A dictionary of parameters, including sampling context and clamping epsilon.
        eps: A small constant to prevent log(0).

    Returns:
        A tensor of shape [bs, L_minus_c, L_minus_c] representing the lagged information gain for each potential cause-effect pair.
    """

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

    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)

    kl = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    cmi_window = torch.mean(kl, dim=1)  # [bs, num_rows-1, Lc]

    # --- Step 3: Sparse-to-Full Matrix Mapping ---
    # We need to map these (num_rows-1) causes to their actual positions in the (Lc x Lc) matrix
    full_cmi = torch.zeros((bs, Lc, Lc), device=device)

    # The causes we just tested are the LAST (num_rows-1) causes.
    # If m=6 and Lc=12, we tested causes 7, 8, 9, 10, 11 (indices 6-11).
    start_row = Lc - (num_rows - 1)
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

    print(f"Computing Shapley for suffix (L={L})...")

    # --- 3. Iterate ---
    for t in range(c, L):
        attributions = shapley.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(t,),  # <--- Passes 't' to 'target_pos'
            n_samples=10,
            perturbations_per_eval=bs * 4,
            show_progress=True,
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

    # 4. Iterate over Suffix Targets
    # We explain why token 't' happened, based on history
    for i, t in enumerate(range(c, L)):
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

    # Average over particles
    score_window = torch.mean(score_raw, dim=1)  # [bs, num_rows-1, Lc]

    # 5. Map back to Full Matrix (Padding if sparse)
    full_score = torch.zeros((bs, Lc, Lc), device=device)
    start_row = Lc - (num_rows - 1)
    full_score[:, start_row:, :] = score_window

    return torch.nan_to_num(full_score, nan=0.0)
