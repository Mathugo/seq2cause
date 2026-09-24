"""Bernoulli KL divergence for the do-intervention CI-test, in two modes.

The lagged information gain (`causal_strength.calc_lag_info_gain`) and the
CLI's CMI matrices (`diagnostics.compute_cmi_matrix`) both reduce to
`KL(Bernoulli(q) || Bernoulli(p))` between the model's probability of the
observed next token with the candidate cause observed (`q`) and with it
replaced by noise (`p`).

`"clamp"` is the v0.1.9 algebra, kept verbatim: probabilities clamped to
`[eps, 1 - eps]`, then `q*log(q/p) + (1-q)*log((1-q)/(1-p))`. In float32 the
literal `1 - 1e-9` rounds to `1.0`, so the upper clamp is a no-op: a
probability that has saturated to exactly `1.0` (a logit gap of roughly 17
nats is enough) yields `NaN` when it is the observed branch and `+inf` when
it is the baseline branch. `calc_lag_info_gain`'s `nan_to_num` then maps
those to `0` and to the float32 maximum (`3.4e38`); `compute_cmi_matrix`
leaves them in place, and one such cell makes the CLI's pooled percentile
threshold `NaN`, i.e. zero edges for every sequence.

`"logspace"` (the default) evaluates the same divergence from
log-probabilities (`log_softmax`, never `softmax`) in float64, with
`log(1 - p) = log1mexp(log p)`, so a saturated softmax never reaches the
formula. Wherever clamp mode is finite the two agree to rounding.

`corrupted_cell_counts` reports, for the inputs of either mode, how many
cells clamp mode turns into `NaN` ("collapsed") or `+inf` ("saturated"), so
a run records what the published algebra would have produced.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "KL_MODES",
    "DEFAULT_KL_MODE",
    "LOG_CLAMP",
    "check_kl_mode",
    "log1mexp",
    "logmeanexp",
    "bernoulli_kl_clamp",
    "bernoulli_kl_from_log",
    "wide_dtype",
    "corrupted_cell_counts",
    "accumulate_counts",
]

KL_MODES = ("logspace", "clamp")
DEFAULT_KL_MODE = "logspace"
# Log-probabilities are capped strictly below 0 before `log1mexp`, so an
# exact `log p == 0` (only possible if a caller passes `log(1.0)`) stays finite.
LOG_CLAMP = -1e-12


def check_kl_mode(kl_mode: str) -> str:
    if kl_mode not in KL_MODES:
        raise ValueError(f"kl_mode must be one of {KL_MODES}, got {kl_mode!r}")
    return kl_mode


def log1mexp(x: Tensor) -> Tensor:
    """`log(1 - exp(x))` for `x <= 0`, numerically stable on both ends
    (Maechler, 2012): `log(-expm1(x))` near 0, `log1p(-exp(x))` otherwise."""
    x = torch.clamp(x, max=LOG_CLAMP)
    return torch.where(x > -math.log(2.0), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def logmeanexp(x: Tensor, dim: int) -> Tensor:
    """`log(mean(exp(x), dim))`: the particle mean of probabilities, taken in
    log space."""
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def bernoulli_kl_clamp(q: Tensor, p: Tensor, eps: float = 1e-9) -> Tensor:
    """`KL(Bernoulli(q) || Bernoulli(p))` exactly as v0.1.9 computes it, from
    probabilities. Not robust to `q == 1.0` or `p == 1.0` in float32 (see the
    module docstring); kept for reproducibility of published numbers."""
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))


def wide_dtype(t: Tensor) -> torch.dtype:
    """float64 where the device supports it; Apple's MPS backend has no float64,
    so the log-space KL runs in float32 there (still finite for every finite
    input -- `log_softmax` never saturates -- with float32 rounding)."""
    return torch.float32 if t.device.type == "mps" else torch.float64


def bernoulli_kl_from_log(log_q: Tensor, log_p: Tensor) -> Tensor:
    """`KL(Bernoulli(q) || Bernoulli(p))` from log-probabilities, in float64
    (float32 on MPS, see `wide_dtype`).

    `q * (log q - log p) + (1 - q) * (log(1 - q) - log(1 - p))` with
    `log(1 - .)` via `log1mexp`. Finite for every finite input; agrees with
    `bernoulli_kl_clamp` to rounding wherever the latter is finite. Callers
    cast back if they need the input dtype.
    """
    dtype = wide_dtype(log_q)
    lq = torch.clamp(log_q.to(dtype), max=LOG_CLAMP)
    lp = torch.clamp(log_p.to(dtype), max=LOG_CLAMP)
    q = torch.exp(lq)
    return q * (lq - lp) + (1.0 - q) * (log1mexp(lq) - log1mexp(lp))


def corrupted_cell_counts(
    q32: Tensor,
    p32: Tensor,
    band: Tensor | None = None,
    particle_dim: int | None = None,
) -> dict[str, int]:
    """Count the cells clamp mode corrupts on these inputs.

    `q32`/`p32` are the observed-branch and baseline-branch probabilities as
    float32 (clamp mode's own inputs; in logspace mode pass `exp(log_prob)`,
    which reproduces the float32 saturation to rounding). A cell is
    *collapsed* when `q == 1.0` (clamp mode gives `NaN`, coerced to `0` by
    `calc_lag_info_gain`) and *saturated* when `p == 1.0` with `q < 1.0`
    (clamp mode gives `+inf`, coerced to the float32 maximum). With
    `particle_dim` set, a cell is the mean over that dimension: it collapses
    if any particle collapses, otherwise it saturates if any particle
    saturates; per-particle counts are reported too. `band` (broadcastable
    to a cell) restricts the count to the cells that are read, i.e. the
    strict upper triangle cause < effect.
    """
    q32 = q32.detach().to(torch.float32)
    p32 = p32.detach().to(torch.float32)
    collapsed = q32 >= 1.0
    saturated = (p32 >= 1.0) & ~collapsed
    out: dict[str, int] = {}
    if particle_dim is not None:
        band_p = band.unsqueeze(particle_dim) if band is not None else None
        if band_p is not None:
            band_p = torch.broadcast_to(band_p, collapsed.shape)
            out["n_particle_collapsed"] = int((collapsed & band_p).sum())
            out["n_particle_saturated"] = int((saturated & band_p).sum())
        else:
            out["n_particle_collapsed"] = int(collapsed.sum())
            out["n_particle_saturated"] = int(saturated.sum())
        collapsed_cell = collapsed.any(dim=particle_dim)
        saturated_cell = saturated.any(dim=particle_dim) & ~collapsed_cell
    else:
        collapsed_cell, saturated_cell = collapsed, saturated
    if band is not None:
        band_c = torch.broadcast_to(band, collapsed_cell.shape)
        collapsed_cell = collapsed_cell & band_c
        saturated_cell = saturated_cell & band_c
        n_cells = int(band_c.sum())
    else:
        n_cells = int(collapsed_cell.numel())
    out["n_cell_collapsed"] = int(collapsed_cell.sum())
    out["n_cell_saturated"] = int(saturated_cell.sum())
    out["n_cells"] = n_cells
    return out


def accumulate_counts(stats: dict, counts: dict[str, int]) -> dict:
    """Add `counts` into `stats` in place (missing keys start at 0)."""
    for key, value in counts.items():
        stats[key] = stats.get(key, 0) + value
    return stats
