import time

import torch
from jaxtyping import Float, Int
from torch import Tensor


def do_interventions(
    rest_upsampled: Float[Tensor, "bs N, L_minus_c"],
    rest_untouched: Float[Tensor, "bs L_minus_c"],
    prefix: Float[Tensor, "bs c"],
    m: int = None,
    prepend_context_back: bool = True,
    **kwargs,
):
    """
    Builds the staircase intervention tensor to detect delayed causal effects.
    If using the full version, it generates all L-c rows corresponding to testing causes X_0 ... X_{L-c-1}.
    Elif m is provided, it builds ONLY the last m rows (testing the last m causes) (sparse variant).
    """

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
    if prepend_context_back:
        # Context is always fixed/observed
        prefix_expanded = prefix.unsqueeze(1).unsqueeze(2).repeat(1, N, num_rows, 1).to(device)
        rest_final = torch.cat([prefix_expanded, rest_final], dim=-1)

    return rest_final


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


def multinomial_sample(
    prob_x: Float[Tensor, "bs vocab"] | Float[Tensor, "bs L vocab"],
    n_samples: int = 128,
    cls_token_id: int | None = None,
    **kwargs,
):
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

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print("Ancestral Sampling - Elapsed time: ", elapsed)
    return start_tokens
