"""Minimal, self-contained nonlinear Structural Causal Model (SCM) oracle.

This reimplements the synthetic benchmark generator described in Math &
Lienhart (2026), Appendix E.1, Eq. (21):

    P(X_t | X_{t-h:t-1}) = softmax(
        b + sum_{k=1}^{h} exp(-(k-1)) * W[x_{t-k}]
          + ReLU([E_{x_{t-h}}, ..., E_{x_{t-1}}] W1) W2
    )

where `W` is a sparse, mixed-sign interaction matrix, `E` are fixed random
per-token embeddings, and `W1`/`W2` form a one-hidden-layer MLP capturing
nonlinear interactions.

Why this exists: the original research repository's `create_scm`/`extract_causes`
helpers (referenced from `experiment.ipynb`) are not part of this published
package. `NonlinearSCM` provides a small, dependency-free, CPU-friendly
stand-in that is directly usable both as the *ground-truth generator* and as
an *oracle model* (its own exact conditional distribution, i.e. epsilon = 0
approximation error) wherever this codebase expects an autoregressive model
with a HuggingFace-style `forward(input_ids=..., attention_mask=...) ->
{"logits": ...}` interface (see `seq2cause.sampling.do_interventions` and
`seq2cause.diagnostics`). Using an oracle model for diagnostics isolates the
`do_interventions` *construction* from ordinary model-approximation error:
any effect measured this way is attributable to the intervention
construction itself.
"""

from __future__ import annotations

import torch
from torch import Tensor


class _SimpleConfig:
    def __init__(self, pad_token_id: int = 0, bos_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id


class NonlinearSCM:
    """Nonlinear, `memory`-order-Markov synthetic event-sequence generator."""

    def __init__(
        self,
        vocab_size: int,
        memory: int = 6,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        sparsity: float = 0.9,
        decay_rate: float = 1.0,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if memory < 1:
            raise ValueError("memory must be >= 1")
        if decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")

        self.vocab_size = vocab_size
        self.memory = memory
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(seed)

        w = torch.randn(vocab_size, vocab_size, generator=g)
        keep_mask = torch.rand(vocab_size, vocab_size, generator=g) > sparsity
        self.W = (w * keep_mask).to(self.device)

        self.E = torch.randn(vocab_size, embed_dim, generator=g).to(self.device)
        self.W1 = (
            torch.randn(memory * embed_dim, hidden_dim, generator=g) / (memory * embed_dim) ** 0.5
        ).to(self.device)
        self.W2 = (torch.randn(hidden_dim, vocab_size, generator=g) / hidden_dim**0.5).to(
            self.device
        )
        self.b = torch.zeros(vocab_size, device=self.device)
        # Eq. 21's decay is exp(-rate*(k-1)) where k=1 is the MOST RECENT lag
        # (strongest weight) and k=memory is the most distant (weakest); the
        # paper's own default is rate=1.0. `context` (see `conditional_logits`)
        # is ordered oldest-first, i.e. context[..., -1] is the most recent
        # token (lag=1), so the decay applied elementwise against it must be
        # *reversed* relative to a naive `exp(-rate*arange(memory))` (which
        # would instead put the strongest weight on the OLDEST/most-distant
        # token) -- this was a real bug caught empirically: `ground_truth_
        # adjacency` was finding true edges almost exclusively at
        # lag == memory and none at lag < memory, the opposite of the
        # intended recency-decayed design.
        #
        # Note rate=1.0 decays fast enough (lag=2 is already only ~37% of
        # lag=1's linear strength, lag=4 ~5%) that with a fixed evaluation
        # margin `delta`, true edges concentrate almost entirely at lag=1 in
        # practice -- pass a smaller `decay_rate` (e.g. 0.3) to get
        # comparable signal strength across lags for lag-recall comparisons.
        self._decay = torch.exp(-decay_rate * torch.arange(memory, device=self.device).float()).flip(0)
        self._config = _SimpleConfig()

    @property
    def config(self):
        return self._config

    def conditional_logits(self, context: Tensor) -> Tensor:
        """Args:
            context: integer tensor `[..., memory]` of the `memory` most
                recent tokens (oldest first, i.e. `context[..., -1]` is
                `x_{t-1}`).

        Returns:
            logits over the vocabulary, `[..., vocab_size]`.
        """
        h = context.shape[-1]
        if h != self.memory:
            raise ValueError(f"expected context length {self.memory}, got {h}")

        # Linear (lag-decayed interaction) term.
        w_rows = self.W[context]  # [..., h, vocab]
        decay_shape = (1,) * (context.dim() - 1) + (h, 1)
        linear = (w_rows * self._decay.view(*decay_shape)).sum(dim=-2)

        # Nonlinear (one-hidden-layer MLP over concatenated embeddings) term.
        emb = self.E[context]  # [..., h, embed_dim]
        emb_flat = emb.reshape(*emb.shape[:-2], h * self.embed_dim)
        hidden = torch.relu(emb_flat @ self.W1)
        nonlinear = hidden @ self.W2

        return self.b + linear + nonlinear

    def conditional_probs(self, context: Tensor) -> Tensor:
        return torch.softmax(self.conditional_logits(context), dim=-1)

    @torch.no_grad()
    def estimate_entropy(
        self, n_sequences: int = 200, length: int = 32, generator: torch.Generator | None = None
    ) -> float:
        """Monte Carlo estimate of the process's irreducible entropy H(P)
        (paper's Eq. 22 Ĥ, used as the oracle-score denominator/baseline).

        Because `conditional_probs` is the SCM's *exact* conditional
        distribution (not an empirical approximation), averaging its exact
        Shannon entropy over many sampled contexts gives an unbiased,
        low-variance estimate of H(P) -- unlike the usual practice of
        approximating Ĥ by a model's minimum validation loss (which is only
        an upper bound on the true H(P), see Chadyuk et al. 2026's
        bookkeeping note on this exact discrepancy).
        """
        seqs = self.sample_sequence(length=length, batch_size=n_sequences, generator=generator)
        entropies = []
        for t in range(self.memory, length):
            probs = self.conditional_probs(seqs[:, t - self.memory : t])  # [n_sequences, vocab]
            probs = torch.clamp(probs, min=1e-12)
            entropies.append(-(probs * probs.log()).sum(dim=-1))  # [n_sequences]
        return torch.cat(entropies).mean().item()

    @torch.no_grad()
    def sample_sequence(
        self, length: int, batch_size: int = 1, generator: torch.Generator | None = None
    ) -> Tensor:
        """Ancestrally samples `batch_size` sequences of `length` tokens.

        The first `memory` tokens (the "burn-in"/prefix) are drawn uniformly
        at random and never causally explained (matching the paper's use of
        a fixed context window that is always observed).
        """
        if length <= self.memory:
            raise ValueError(f"length ({length}) must be > memory ({self.memory})")
        seq = torch.randint(
            0, self.vocab_size, (batch_size, self.memory), generator=generator, device=self.device
        )
        for _ in range(length - self.memory):
            probs = self.conditional_probs(seq[:, -self.memory :])
            nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], dim=-1)
        return seq

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs
    ) -> dict[str, Tensor]:
        """HuggingFace-style forward pass: next-token logits at every position.

        Follows the SAME convention as a real causal LM (and as every other
        piece of this codebase that consumes `forward()`'s output, e.g.
        `diagnostics._predicted_true_token_probs`/`estimate_oracle_score`):
        `logits[..., i, :]` predicts the token AT POSITION `i + 1`, using the
        `memory` tokens immediately before it, `input_ids[..., i-memory+1:i+1]`
        -- NOT the token at position `i` itself. (An earlier version of this
        method stored the prediction for token `i` at index `i`, which is
        off-by-one relative to that convention; caught via
        `estimate_oracle_score` reading epsilon_hat >> 0 for the oracle model
        evaluated against itself, which should be ~0 by construction.)

        Positions with too little preceding context (`i < memory - 1`) or no
        valid target (`i == seq_len - 1`, the last position) are filled with
        zero logits and should not be relied upon by callers -- this
        codebase always keeps at least `memory` tokens as a fixed, always-real
        prefix/context, so those positions are never queried in practice.
        """
        *batch_dims, seq_len = input_ids.shape
        m = self.memory
        if seq_len < m:
            raise ValueError(f"NonlinearSCM.forward requires sequence length >= memory ({m})")

        logits = input_ids.new_zeros((*batch_dims, seq_len, self.vocab_size), dtype=torch.float32)
        for i in range(m - 1, seq_len - 1):
            logits[..., i, :] = self.conditional_logits(input_ids[..., i - m + 1 : i + 1])
        return {"logits": logits}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):  # pragma: no cover - HF-model-compatibility no-op
        return self

    def to(self, device):  # pragma: no cover - HF-model-compatibility no-op
        return self


def create_scm(
    vocab_size: int,
    memory: int = 6,
    length: int = 64,
    sparsity: float = 0.9,
    background_noise: float = 0.0,
    seed: int = 0,
    batch_size: int = 1,
    embed_dim: int = 16,
    hidden_dim: int = 64,
    decay_rate: float = 1.0,
    device: str | torch.device = "cpu",
):
    """Minimal factory mirroring the `create_scm(...)` helper referenced by the
    original (unpublished) research notebooks: `scm, sequence = create_scm(...)`.

    `background_noise` is accepted for signature compatibility but currently
    unused by this minimal reimplementation (the published package does not
    include the original's background-noise injection mechanism).

    Returns:
        `(scm, sequence)` where `scm` is a `NonlinearSCM` and `sequence` is a
        `[batch_size, length]` ancestrally-sampled batch of sequences from it.
    """
    del background_noise  # unused, kept for signature compatibility
    scm = NonlinearSCM(
        vocab_size=vocab_size,
        memory=memory,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        sparsity=sparsity,
        decay_rate=decay_rate,
        seed=seed,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sequence = scm.sample_sequence(length=length, batch_size=batch_size, generator=generator)
    return scm, sequence
