"""Information gain of the observed cause vs. CMI, and how the counterfactual
tokens are drawn, on a small synthetic SCM.

For every held-out sequence, every candidate cause position j and every later
effect position q, two branches are evaluated per particle:
  e_t = 1: the observed cause x_j,
  e_t = 0: x_j replaced by a counterfactual token.
From the model probability of the observed effect token in both branches
(q1, q0) and the model probability of the observed cause (pi), two scores are
computed and averaged over the N particles:
  info gain (observed cause):  KL(Bern(q1) || Bern(q0))           (TRACE, v0.1.9)
  CMI:  pi KL(Bern(q1) || Bern(m)) + (1 - pi) KL(Bern(q0) || Bern(m)),
        m = pi q1 + (1 - pi) q0                                    (calc_lag_cmi)

Design choices compared (one row per combination):
  context    observed  : the first c tokens are the observed ones
             ancestral : first `guidance` tokens observed, the rest of the
                         context sampled autoregressively from the model
                         (as SampleLevelCausalDiscovery does)
  cause      uniform   : uniform over the vocabulary (may equal the observed token)
             uniform!= : uniform, never the observed token
             model!=   : model distribution given the trajectory, observed token
                         excluded (`cause_from_model=True`)
  mediators  uniform   : uniform noise, shared by both branches (staircase "full")
             observed  : observed tokens (atomic intervention)
             model     : drawn from the model given the observed trajectory,
                         shared by both branches (non-autoregressive)
             ar        : generated autoregressively after each branch's cause,
                         separately per branch (total effect, not direct effect)

Two ground truths, since each favours its own estimand:
  GT-IG  : `diagnostics.ground_truth_adjacency` (exact SCM, atomic intervention,
           uniform replacement, information gain of the observed cause > delta);
  GT-CMI : exact CMI of the binary cause indicator from the SCM (mediators
           observed), top-k pairs with k = number of GT-IG edges.
Metrics over all pairs c <= j < q: average precision (AP), best F1 over
thresholds, and the median relative standard error of the particle mean on
GT-IG edges (lower = more stable).

Two models: the SCM itself as an exact oracle (isolates the estimator design
from model error) and a tiny Llama trained on SCM samples.

Usage:
    python scripts/compare_cmi_and_proposals.py [--vocab-size 50] [--n-eval 30]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from seq2cause.diagnostics import ground_truth_adjacency  # noqa: E402
from seq2cause.scm import NonlinearSCM  # noqa: E402
from seq2cause.threshold import (  # noqa: E402
    gmm_threshold,
    mad_threshold,
    otsu_threshold,
    percentile_threshold,
)

CONTEXTS = ("observed", "ancestral")
CAUSES = ("uniform", "uniform!=", "model!=")
MEDIATORS = ("uniform", "observed", "model", "ar")
EPS = 1e-9


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vocab-size", type=int, default=50)
    p.add_argument("--memory", type=int, default=3)
    p.add_argument("--sparsity", type=float, default=0.7)
    p.add_argument("--decay-rate", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--guidance", type=int, default=3, help="observed context tokens (>= memory)")
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--n-particles", type=int, default=32)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument(
        "--target-eps", type=float, default=0.05, help="train until eps_hat + 2 SE <= this"
    )
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", type=str, default="oracle,trained")
    p.add_argument(
        "--mode",
        choices=(
            "full",
            "multiseed",
            "particles",
            "train",
            "context",
            "length",
            "sampler",
            "nullcal",
            "nosample",
        ),
        default="full",
        help="full: all variants on one SCM + threshold analysis; multiseed: reduced variant set "
        "over --n-seeds SCM/model seeds (mean +- std); particles: stability vs. number of particles",
    )
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument(
        "--memories", type=str, default="2,4,8", help="context/length ablations: SCM memories"
    )
    p.add_argument(
        "--context-offsets",
        type=str,
        default="0,2,4,8,16",
        help="context ablation: c = memory + offset",
    )
    p.add_argument("--ablation-len", type=int, default=40, help="context ablation: sequence length")
    p.add_argument(
        "--lengths", type=str, default="16,32,64,96", help="length ablation: sequence lengths"
    )
    p.add_argument(
        "--ref-len", type=int, default=32, help="length ablation: length where tau is fitted"
    )
    p.add_argument(
        "--mediators", type=str, default="uniform,observed", help="ablations: mediator modes"
    )
    p.add_argument(
        "--simulators",
        type=str,
        default="naive,weighted",
        help="ablations: Monte Carlo simulators for sampled pasts (naive: equal weights; weighted: "
        "weights proportional to the likelihood of the observed history after the sampled past)",
    )
    p.add_argument(
        "--context-samplers",
        type=str,
        default="multinomial,topk:5,topk:20,nucleus:0.9,nucleus:0.7,temp:0.7,temp:1.5,greedy",
        help="sampler mode: decoding rules for the generated past",
    )
    p.add_argument(
        "--sampler-offset", type=int, default=8, help="sampler mode: context c = memory + offset"
    )
    p.add_argument(
        "--null-min-lag",
        type=int,
        default=16,
        help="nullcal mode: pairs with lag >= this are assumed to have no effect (bounded lagged effects) "
        "and form the empirical null",
    )
    p.add_argument(
        "--null-alpha",
        type=float,
        default=0.01,
        help="nullcal mode: per-pair level of the null quantile",
    )
    p.add_argument(
        "--fdr",
        type=float,
        default=0.1,
        help="nullcal mode: Benjamini-Hochberg false discovery rate",
    )
    p.add_argument(
        "--particle-grid", type=str, default="8,32,128", help="nosample mode: particle counts"
    )
    p.add_argument(
        "--exact-past-particles",
        type=int,
        default=8,
        help="nosample mode: sampled pasts for exact+prepend",
    )
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- model helpers


class FastSCM:
    """Vectorised `NonlinearSCM.forward` (same logits, same convention: logits[..., i] predicts
    token i + 1 from tokens i-m+1..i; positions i < m-1 are zero)."""

    def __init__(self, scm: NonlinearSCM):
        self.scm = scm

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> dict:
        m = self.scm.memory
        *b, L = input_ids.shape
        windows = input_ids.unfold(-1, m, 1)  # [..., L-m+1, m]: window k = tokens k..k+m-1
        logits = input_ids.new_zeros((*b, L, self.scm.vocab_size), dtype=torch.float32)
        logits[..., m - 1 : L - 1, :] = self.scm.conditional_logits(windows[..., : L - m, :])
        return {"logits": logits}


@torch.no_grad()
def next_token_probs(model, tokens: torch.Tensor) -> torch.Tensor:
    """[B, L] -> [B, L, V] float64 probabilities; [:, i] is the distribution of token i + 1."""
    logits = model(input_ids=tokens)["logits"]
    return torch.softmax(logits.double(), dim=-1)


def prob_of(probs: torch.Tensor, tokens: torch.Tensor, positions: range) -> torch.Tensor:
    """Probability of tokens[:, p] at each p in positions -> [B, len(positions)]."""
    idx = torch.tensor(list(positions))
    return probs[:, idx - 1, :].gather(-1, tokens[:, idx].unsqueeze(-1)).squeeze(-1)


def sample_excluding(probs: torch.Tensor, exclude: torch.Tensor | None) -> torch.Tensor:
    """Row-wise multinomial draw from [B, V] probabilities, optionally excluding one id per row."""
    p = probs.clone()
    if exclude is not None:
        p.scatter_(-1, exclude.view(-1, 1), 0.0)
    p = p + 1e-12
    if exclude is not None:
        p.scatter_(-1, exclude.view(-1, 1), 0.0)
    return torch.multinomial(p / p.sum(-1, keepdim=True), 1).squeeze(-1)


def apply_sampler(probs: torch.Tensor, sampler: str) -> torch.Tensor:
    """Draw one token per row from [B, V] probabilities with a decoding rule:
    multinomial (full distribution), topk:K, nucleus:P (smallest set with mass >= P),
    temp:T (probabilities ** (1/T), renormalised), greedy (argmax)."""
    if sampler == "greedy":
        return probs.argmax(-1)
    p = probs
    if sampler.startswith("topk:"):
        k = int(sampler.split(":")[1])
        kth = torch.topk(p, k, dim=-1).values[:, -1:]
        p = torch.where(p >= kth, p, torch.zeros_like(p))
    elif sampler.startswith("nucleus:"):
        top_p = float(sampler.split(":")[1])
        sp, idx = torch.sort(p, dim=-1, descending=True)
        keep = torch.cumsum(sp, dim=-1) - sp < top_p  # tokens whose preceding mass is < top_p
        p = torch.zeros_like(p).scatter(-1, idx, sp * keep)
    elif sampler.startswith("temp:"):
        t = float(sampler.split(":")[1])
        p = p.clamp_min(1e-300) ** (1.0 / t)
    elif sampler != "multinomial":
        raise ValueError(f"unknown sampler {sampler!r}")
    return torch.multinomial(p / p.sum(-1, keepdim=True), 1).squeeze(-1)


@torch.no_grad()
def autoregressive_fill(
    model, tokens: torch.Tensor, start: int, stop: int, sampler: str = "multinomial"
) -> torch.Tensor:
    """Regenerate positions start..stop-1 of every row from the model, one step at a time.

    The prefix passed to the model includes position i itself (its current token cannot
    influence the logits at i - 1 of a causal model): `NonlinearSCM.forward` returns zero
    logits at the last position, so reading the last position of `tokens[:, :i]` would
    sample uniformly instead of from the SCM.
    """
    tokens = tokens.clone()
    for i in range(start, stop):
        probs = next_token_probs(model, tokens[:, : i + 1])[:, i - 1]
        tokens[:, i] = apply_sampler(probs, sampler)
    return tokens


def bern_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    q = q.clamp(EPS, 1 - EPS)
    p = p.clamp(EPS, 1 - EPS)
    return q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))


# ----------------------------------------------------------------------------- scoring


@torch.no_grad()
def score_sequence(
    model,
    seq: torch.Tensor,
    ctx_mode,
    cause_mode,
    med_mode,
    args,
    vocab,
    simulator="naive",
    sampler="multinomial",
):
    """Returns ig, cmi, ig_rse, cmi_rse, each [L, L] indexed [cause j, effect q] of the observed
    sequence (pairs that a context mode cannot test stay 0), and ess, the mean effective sample
    size fraction of the particle weights.

    Context modes:
      observed  : the first c observed tokens are the context; causes j >= c are tested.
      ancestral : the first `guidance` observed tokens are kept and positions guidance..c-1 are
                  overwritten with tokens generated from the model; causes j >= c are tested.
      prepend   : c tokens are generated per particle (the first `guidance` uniformly, as the SCM
                  burn-in, the rest from the model) and placed BEFORE the full observed sequence;
                  every observed token is kept and causes j >= 0 are tested.

    Simulators (how the particles are averaged):
      naive    : equal weights 1/N (the sampled pasts follow the model prior p(past)).
      weighted : self-normalised importance weights proportional to the model likelihood of the
                 observed tokens between the sampled past and the cause, so that the average
                 targets p(past | observed history) instead of p(past). Weighting by p(past)
                 itself would count it twice, since the pasts are already drawn from it. With
                 an observed context there is no sampled past and the weights are uniform.
    """
    L, c, N = seq.shape[0], args.context, args.n_particles
    prepend = ctx_mode == "prepend"
    obs = seq.unsqueeze(0).repeat(N, 1)
    if prepend:
        o = c  # offset of observed position 0 in the model input
        g = min(args.guidance, c)
        past = torch.randint(0, vocab, (N, c))
        base = autoregressive_fill(model, torch.cat([past, obs], dim=1), g, c, sampler)
        first = 0
    else:
        o = 0
        base = obs
        if ctx_mode == "ancestral":
            base = autoregressive_fill(model, base, args.guidance, c, sampler)
        first = c
    T = base.shape[1]
    weighted = simulator == "weighted" and ctx_mode != "observed"

    out = {k: torch.zeros(L, L, dtype=torch.float64) for k in ("ig", "cmi", "ig_rse", "cmi_rse")}
    base_probs = next_token_probs(model, base)
    if weighted:
        # log P(x_i | sampled past, x_s0..x_{i-1}) for the observed positions i >= s0 that follow
        # the sampled past (all of them when prepending, those after the context otherwise)
        s0 = 0 if prepend else c
        pos = torch.arange(s0, L)
        ll = torch.log(
            base_probs[:, o + pos - 1, :]
            .gather(-1, seq[pos].view(1, -1, 1).expand(N, -1, 1))
            .squeeze(-1)
            .clamp_min(1e-300)
        )
        cum_ll = torch.cat([torch.zeros(N, 1, dtype=ll.dtype), torch.cumsum(ll, dim=1)], dim=1)
    ess = []

    for j in range(first, L - 1):
        J = o + j  # model position of the cause
        effects = range(J + 1, T)
        obs_j = seq[j].expand(N)
        # distribution of the cause given its past; with no past at all (position 0), the uniform
        # prior, which is also the SCM's burn-in distribution
        prior_j = (
            base_probs[:, J - 1, :]
            if J > 0
            else torch.full((N, vocab), 1.0 / vocab, dtype=torch.float64)
        )
        pi = prior_j.gather(-1, obs_j.view(-1, 1)).squeeze(-1)  # [N]

        # counterfactual token for the cause
        if cause_mode == "uniform":
            repl = torch.randint(0, vocab, (N,))
        elif cause_mode == "uniform!=":
            repl = sample_excluding(torch.ones(N, vocab, dtype=torch.float64), obs_j)
        else:  # model!=
            repl = sample_excluding(prior_j, obs_j)

        x1 = base.clone()
        x0 = base.clone()
        x0[:, J] = repl

        # mediators (positions after the cause)
        if med_mode == "uniform":
            noise = torch.randint(0, vocab, (N, T - J - 1))
            x1[:, J + 1 :] = noise
            x0[:, J + 1 :] = noise
        elif med_mode == "model":
            draw = torch.stack(
                [
                    torch.multinomial(base_probs[:, p - 1, :], 1).squeeze(-1)
                    for p in range(J + 1, T)
                ],
                dim=1,
            )
            x1[:, J + 1 :] = draw
            x0[:, J + 1 :] = draw
        elif med_mode == "ar":
            both = autoregressive_fill(model, torch.cat([x1, x0]), J + 1, T - 1)
            x1, x0 = both[:N], both[N:]
        # "observed": keep the observed tokens

        probs = next_token_probs(model, torch.cat([x1, x0]))
        target = base.repeat(2, 1)  # positions >= o hold the OBSERVED tokens
        pq = prob_of(probs, target, effects)
        q1, q0 = pq[:N], pq[N:]

        ig = bern_kl(q1, q0)  # [N, Q]
        m = pi.unsqueeze(-1) * q1 + (1 - pi.unsqueeze(-1)) * q0
        cmi = pi.unsqueeze(-1) * bern_kl(q1, m) + (1 - pi.unsqueeze(-1)) * bern_kl(q0, m)

        if weighted:
            w = torch.softmax(cum_ll[:, j - s0], dim=0).unsqueeze(-1)  # likelihood of x_s0..x_{j-1}
        else:
            w = torch.full((N, 1), 1.0 / N, dtype=torch.float64)
        ess.append(float(1.0 / (w**2).sum()) / N)
        for name, v in (("ig", ig), ("cmi", cmi)):
            mean = (w * v).sum(0)
            se = torch.sqrt((w**2 * (v - mean) ** 2).sum(0))
            out[name][j, j + 1 :] = mean
            out[name + "_rse"][j, j + 1 :] = se / mean.clamp_min(1e-12)
    out["ess"] = torch.tensor(sum(ess) / max(1, len(ess)), dtype=torch.float64)
    return out


# ----------------------------------------------------------------------------- metrics


@torch.no_grad()
def exact_cmi_matrix(scm: NonlinearSCM, seq: torch.Tensor, c: int) -> torch.Tensor:
    """Exact CMI of the binary cause indicator from the SCM (mediators observed):
    pi = P(x_j | past), q1 = P(x_q | x_j), q0 = P(x_q | X_j != x_j) averaged with the
    SCM's own P(X_j | past), m = sum_v P(v | past) P(x_q | v)."""
    L, mem, V = seq.shape[0], scm.memory, scm.vocab_size
    M = torch.zeros(L, L, dtype=torch.float64)
    for j in range(c, L - 1):
        if j < mem:  # burn-in position: the SCM draws it uniformly
            pj = torch.full((V,), 1.0 / V, dtype=torch.float64)
        else:
            pj = scm.conditional_probs(seq[j - mem : j]).double()
        xj = int(seq[j])
        for q in range(max(j + 1, mem), min(j + mem, L - 1) + 1):
            ctx = seq[q - mem : q].repeat(V, 1)
            ctx[:, j - (q - mem)] = torch.arange(V)
            pq = scm.conditional_probs(ctx)[:, seq[q]].double()
            pi, q1 = pj[xj], pq[xj]
            mm = (pj * pq).sum()
            q0 = (mm - pi * q1) / (1 - pi)
            M[j, q] = pi * bern_kl(q1, mm) + (1 - pi) * bern_kl(q0, mm)
    return M


def average_precision_and_best_f1(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    order = torch.argsort(scores, descending=True)
    lab = labels[order].double()
    tp = torch.cumsum(lab, 0)
    k = torch.arange(1, len(lab) + 1, dtype=torch.float64)
    precision = tp / k
    recall = tp / lab.sum()
    ap = float((precision * lab).sum() / lab.sum())
    f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).max()
    return ap, float(f1)


# ----------------------------------------------------------------------------- thresholds

THRESH_VARIANTS = (("ancestral", "uniform", "uniform"), ("ancestral", "uniform", "observed"))
SWEEP = (
    (50, 0.5),
    (50, 0.2),
    (50, 1.0),
    (25, 0.5),
    (100, 0.5),
)  # (|X|, decay_rate); first = reference


def best_tau(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Threshold (predict edge iff score >= tau) maximizing F1, and that F1."""
    order = torch.argsort(scores, descending=True)
    s, lab = scores[order], labels[order].double()
    tp = torch.cumsum(lab, 0)
    precision = tp / torch.arange(1, len(lab) + 1, dtype=torch.float64)
    recall = tp / lab.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    i = int(torch.argmax(f1))
    return float(s[i]), float(f1[i])


def f1_at(scores: torch.Tensor, labels: torch.Tensor, tau: float) -> float:
    pred = scores >= tau
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def threshold_stats(per_seq: list, per_seq_labels: list, lags: torch.Tensor, memory: int) -> dict:
    S, Y = torch.cat(per_seq), torch.cat(per_seq_labels)
    lag_all = lags.repeat(len(per_seq))
    tau, f1 = best_tau(S, Y)

    half = len(per_seq) // 2
    tau_tr, _ = best_tau(torch.cat(per_seq[:half]), torch.cat(per_seq_labels[:half]))
    s_te, y_te = torch.cat(per_seq[half:]), torch.cat(per_seq_labels[half:])
    f1_transfer = f1_at(s_te, y_te, tau_tr)
    _, f1_te_best = best_tau(s_te, y_te)

    taus_seq = [best_tau(s, y)[0] for s, y in zip(per_seq, per_seq_labels) if int(y.sum()) >= 2]
    log_t = torch.log10(torch.tensor(taus_seq, dtype=torch.float64).clamp_min(1e-12))
    iqr = (
        float(torch.quantile(log_t, 0.75) - torch.quantile(log_t, 0.25))
        if len(taus_seq) > 3
        else float("nan")
    )

    taus_lag = [best_tau(S[lag_all == k], Y[lag_all == k])[0] for k in range(1, memory + 1)]
    lag_spread = math.log10(max(taus_lag) / max(min(taus_lag), 1e-12))

    return {
        "tau": tau,
        "f1": f1,
        "f1_transfer": f1_transfer,
        "f1_test_best": f1_te_best,
        "iqr": iqr,
        "taus_lag": taus_lag,
        "lag_spread": lag_spread,
        "otsu": f1_at(S, Y, otsu_threshold(S)),
        "p95": f1_at(S, Y, percentile_threshold(S, 0.95)),
        "mad": f1_at(S, Y, mad_threshold(S)),
        "gmm": f1_at(S, Y, gmm_threshold(S).tau),
    }


def threshold_row(per_seq: list, per_seq_labels: list, lags: torch.Tensor, memory: int) -> str:
    t = threshold_stats(per_seq, per_seq_labels, lags, memory)
    lag_txt = "/".join(f"{v:.2g}" for v in t["taus_lag"])
    return (
        f"{t['tau']:9.3g}{t['f1']:6.3f} | {t['f1_transfer']:6.3f}/{t['f1_test_best']:5.3f} | {t['iqr']:6.2f} | "
        f"{lag_txt:<22}{t['lag_spread']:5.2f} | "
        + " ".join(f"{t[k]:5.3f}" for k in ("otsu", "p95", "mad", "gmm"))
    )


THRESH_HEADER = (
    f"{'variant':<20}{'score':<5}{'GT':<5}|{'tau*':>9}{'F1*':>6} | {'half->half':>11} | "
    f"{'IQR':>6} | {'tau* by lag 1/2/3':<22}{'spr':>5} | otsu  p95   mad   gmm"
)
THRESH_LEGEND = (
    "tau*, F1*: best threshold on all pairs and its F1.  half->half: F1 on the second half of the\n"
    "sequences with tau* fitted on the first half / best F1 achievable on the second half.  IQR:\n"
    "interquartile range of per-sequence best thresholds, in decades (log10).  spr: decades between\n"
    "the smallest and largest per-lag tau*.  otsu/p95/mad/gmm: F1 of the unsupervised thresholds of\n"
    "seq2cause.threshold on the pooled scores."
)


def build_setting(vocab, decay, args):
    """SCM, held-out sequences, per-sequence GT-IG and GT-CMI labels over the scored pairs."""
    scm = NonlinearSCM(
        vocab_size=vocab,
        memory=args.memory,
        sparsity=args.sparsity,
        decay_rate=decay,
        seed=args.seed,
    )
    gen = torch.Generator().manual_seed(args.seed + 1)
    seqs = scm.sample_sequence(args.seq_len, batch_size=args.n_eval, generator=gen)
    c, L = args.context, args.seq_len
    pair_mask = torch.zeros(L, L, dtype=torch.bool)
    for j in range(c, L - 1):
        pair_mask[j, j + 1 :] = True
    lab_ig = [
        ground_truth_adjacency(scm, s, threshold=args.delta, generator=gen)[pair_mask] for s in seqs
    ]
    cmi = torch.cat([exact_cmi_matrix(scm, s, c)[pair_mask] for s in seqs])
    k = int(torch.cat(lab_ig).sum())
    flat = torch.zeros_like(cmi, dtype=torch.bool)
    flat[torch.argsort(cmi, descending=True)[:k]] = True
    lab_cmi = list(flat.split(int(pair_mask.sum())))
    jj, qq = torch.nonzero(pair_mask, as_tuple=True)
    return scm, seqs, pair_mask, lab_ig, lab_cmi, qq - jj


def sweep_report(args):
    """Does the best threshold transfer across SCM settings? (exact SCM, default variant)"""
    print(
        f"\n=== Threshold transfer across SCM settings (exact SCM, ancestral/uniform/uniform) ===\n"
        f"tau_ref = best threshold at |X|={SWEEP[0][0]}, decay={SWEEP[0][1]}, applied unchanged elsewhere.\n"
        f"{'setting':<18}{'score':<5}{'GT':<5}|{'tau*':>9}{'F1*':>7}{'F1(tau_ref)':>13}"
    )
    ref = {}
    for vocab, decay in SWEEP:
        scm, seqs, pair_mask, lab_ig, lab_cmi, _ = build_setting(vocab, decay, args)
        torch.manual_seed(args.seed + 7)
        res = [
            score_sequence(FastSCM(scm), s, "ancestral", "uniform", "uniform", args, vocab)
            for s in seqs
        ]
        for name in ("ig", "cmi"):
            S = torch.cat([r[name][pair_mask] for r in res])
            for gt, labs in (("IG", lab_ig), ("CMI", lab_cmi)):
                Y = torch.cat(labs)
                tau, f1 = best_tau(S, Y)
                key = (name, gt)
                if key not in ref:
                    ref[key] = tau
                print(
                    f"|X|={vocab:<4}decay={decay:<4} {name:<5}{gt:<5}|{tau:9.3g}{f1:7.3f}"
                    f"{f1_at(S, Y, ref[key]):13.3f}"
                )


# ----------------------------------------------------------------------------- main


def train_model(scm, args):
    """Tiny Llama trained on fresh SCM samples until the exact oracle score reaches
    `args.target_eps` (or `args.max_steps`); returns (model, oracle-score dict)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=scm.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 2,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,
        max_position_embeddings=args.seq_len + 8,
    )
    model = LlamaForCausalLM(config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_steps, eta_min=args.lr * 0.05
    )
    step, oracle = 0, None
    while step < args.max_steps:
        model.train()
        for _ in range(args.eval_every):
            batch = scm.sample_sequence(length=args.seq_len, batch_size=args.batch_size)
            logits = model(input_ids=batch[:, :-1]).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, scm.vocab_size), batch[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        model.eval()
        oracle = exact_oracle_score(model, scm, length=args.seq_len)
        print(
            f"  step {step:5d}  train loss={loss.item():.3f}  KL={oracle['kl']:.4f}  "
            f"eps_hat={oracle['epsilon_hat']:.4f} +- {oracle['epsilon_se']:.4f}",
            flush=True,
        )
        if oracle["epsilon_hat"] + 2 * oracle["epsilon_se"] <= args.target_eps:
            break
    oracle["steps"] = step
    return model, oracle


@torch.no_grad()
def exact_oracle_score(
    model, scm, n_sequences: int = 1000, length: int = 24, seed: int = 12345
) -> dict:
    """Oracle score eps_hat = (L_AR - H(P)) / (H_max - H(P)) from the SCM's exact conditionals.

    L_AR - H(P) equals the expected KL(P(.|x_<t) || P_theta(.|x_<t)), computed exactly per token
    (summed over the vocabulary) instead of as a difference of two Monte Carlo estimates; H(P) is
    the exact entropy of each true conditional, averaged over fresh sequences. Positions
    0..memory-1 (uniform burn-in) are excluded, as in `estimate_oracle_score`.
    """
    gen = torch.Generator().manual_seed(seed)
    seqs = scm.sample_sequence(length=length, batch_size=n_sequences, generator=gen)
    m = scm.memory
    true = torch.stack(
        [scm.conditional_probs(seqs[:, t - m : t]) for t in range(m, length)], dim=1
    ).double()
    log_true = torch.log(true.clamp_min(1e-300))
    log_model = torch.log_softmax(
        model(input_ids=seqs)["logits"][:, m - 1 : -1, :].double(), dim=-1
    )
    h_tok = -(true * log_true).sum(-1)  # [n, L-m]
    kl_tok = (true * (log_true - log_model)).sum(-1)
    h_p = float(h_tok.mean())
    h_max = math.log(scm.vocab_size)
    kl_seq = kl_tok.mean(-1)  # per-sequence mean KL
    kl = float(kl_seq.mean())
    denom = h_max - h_p
    return {
        "h_p": h_p,
        "h_max": h_max,
        "kl": kl,
        "loss": h_p + kl,
        "epsilon_hat": kl / denom,
        "epsilon_se": float(kl_seq.std() / math.sqrt(n_sequences)) / denom,
    }


MS_VARIANTS = tuple(("ancestral", cz, md) for cz in ("uniform", "model!=") for md in MEDIATORS)


def _pm(values: list) -> str:
    t = torch.tensor(values, dtype=torch.float64)
    return f"{float(t.mean()):.3f}+-{float(t.std()) if len(values) > 1 else 0.0:.3f}"


def multiseed_report(args):
    """Reduced variant set over several SCM/model seeds: mean +- std."""
    from collections import defaultdict

    acc = defaultdict(list)
    eps = []
    for seed in range(args.n_seeds):
        sargs = argparse.Namespace(**{**vars(args), "seed": seed})
        torch.manual_seed(seed)
        scm, seqs, pair_mask, lab_ig, lab_cmi, lags = build_setting(
            args.vocab_size, args.decay_rate, sargs
        )
        models = {}
        if "oracle" in args.models:
            models["oracle"] = FastSCM(scm)
        if "trained" in args.models:
            torch.manual_seed(seed)
            model, oracle = train_model(scm, sargs)
            eps.append(oracle["epsilon_hat"])
            print(
                f"[seed {seed}] trained: H(P)={oracle['h_p']:.4f}, H_max={oracle['h_max']:.4f}, "
                f"KL={oracle['kl']:.4f}, eps_hat={oracle['epsilon_hat']:.4f} +- {oracle['epsilon_se']:.4f} "
                f"after {oracle['steps']} steps",
                flush=True,
            )
            models["trained"] = model
        print(f"[seed {seed}] {int(torch.cat(lab_ig).sum())} GT-IG edges", flush=True)
        for mname, model in models.items():
            for variant in MS_VARIANTS:
                torch.manual_seed(seed + 7)
                res = [score_sequence(model, s, *variant, sargs, args.vocab_size) for s in seqs]
                for name in ("ig", "cmi"):
                    per_seq = [r[name][pair_mask] for r in res]
                    S = torch.cat(per_seq)
                    key = (mname, variant[1], variant[2], name)
                    ap1, f1a = average_precision_and_best_f1(S, torch.cat(lab_ig))
                    ap2, f1b = average_precision_and_best_f1(S, torch.cat(lab_cmi))
                    acc[key + ("ap_ig",)].append(ap1)
                    acc[key + ("ap_cmi",)].append(ap2)
                    acc[key + ("f1_cmi",)].append(f1b)
                    if variant[1] == "uniform" and variant[2] in ("uniform", "observed"):
                        for gt, labs in (("IG", lab_ig), ("CMI", lab_cmi)):
                            t = threshold_stats(per_seq, labs, lags, args.memory)
                            acc[key + (gt, "transfer")].append(
                                t["f1_transfer"] / max(t["f1_test_best"], 1e-12)
                            )
                            acc[key + (gt, "iqr")].append(t["iqr"])
                            acc[key + (gt, "lag_spread")].append(t["lag_spread"])
                            acc[key + (gt, "p95")].append(t["p95"] / max(t["f1"], 1e-12))

    print(
        f"\n=== {args.n_seeds} seeds (SCM + model + particles), |X|={args.vocab_size}, "
        f"{args.n_eval} sequences, N={args.n_particles}, ancestral context ==="
    )
    if eps:
        print(f"trained tiny Llama oracle score eps_hat: {_pm(eps)}")
    print(
        f"{'model':<9}{'cause':<9}{'mediators':<10}{'score':<6}{'AP(GT-IG)':>14}{'AP(GT-CMI)':>14}{'F1(GT-CMI)':>14}"
    )
    for mname in ("oracle", "trained"):
        for _, cz, md in MS_VARIANTS:
            for name in ("ig", "cmi"):
                key = (mname, cz, md, name)
                if key + ("ap_ig",) not in acc:
                    continue
                print(
                    f"{mname:<9}{cz:<9}{md:<10}{name:<6}{_pm(acc[key + ('ap_ig',)]):>14}"
                    f"{_pm(acc[key + ('ap_cmi',)]):>14}{_pm(acc[key + ('f1_cmi',)]):>14}"
                )

    print(
        "\nThresholds (uniform cause). transfer: F1 with tau* fitted on half of the sequences / best F1\n"
        "on the other half.  IQR: per-sequence tau* spread (decades).  lag spr: per-lag tau* spread\n"
        "(decades).  p95/F1*: F1 of the label-free 95th-percentile threshold relative to the best F1."
    )
    print(
        f"{'model':<9}{'mediators':<10}{'score':<6}{'GT':<5}{'transfer':>14}{'IQR':>14}{'lag spr':>14}{'p95/F1*':>14}"
    )
    for mname in ("oracle", "trained"):
        for md in ("uniform", "observed"):
            for name in ("ig", "cmi"):
                for gt in ("IG", "CMI"):
                    key = (mname, "uniform", md, name, gt)
                    if key + ("iqr",) not in acc:
                        continue
                    print(
                        f"{mname:<9}{md:<10}{name:<6}{gt:<5}{_pm(acc[key + ('transfer',)]):>14}"
                        f"{_pm(acc[key + ('iqr',)]):>14}{_pm(acc[key + ('lag_spread',)]):>14}"
                        f"{_pm(acc[key + ('p95',)]):>14}"
                    )


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def _ablation_data(memory: int, length: int, args, seed: int):
    """SCM with this memory, held-out sequences of this length, full [n, L, L] ground truths
    (GT-IG boolean, exact CMI) and the models to score with."""
    scm = NonlinearSCM(
        vocab_size=args.vocab_size,
        memory=memory,
        sparsity=args.sparsity,
        decay_rate=args.decay_rate,
        seed=seed,
    )
    gen = torch.Generator().manual_seed(seed + 1)
    seqs = scm.sample_sequence(length, batch_size=args.n_eval, generator=gen)
    gt_ig = torch.stack(
        [ground_truth_adjacency(scm, s, threshold=args.delta, generator=gen) for s in seqs]
    )
    cmi = torch.stack([exact_cmi_matrix(scm, s, 0) for s in seqs])
    return scm, seqs, gt_ig, cmi


def _models_for(scm, length: int, args, seed: int) -> dict:
    models = {}
    if "oracle" in args.models:
        models["oracle"] = FastSCM(scm)
    if "trained" in args.models:
        targs = argparse.Namespace(**{**vars(args), "seq_len": length, "seed": seed})
        torch.manual_seed(seed)
        model, oracle = train_model(scm, targs)
        print(
            f"    trained (memory={scm.memory}, L={length}, seed={seed}): H(P)={oracle['h_p']:.3f}, "
            f"eps_hat={oracle['epsilon_hat']:.4f} after {oracle['steps']} steps",
            flush=True,
        )
        models["trained"] = model
    return models


def _labels(
    gt_ig: torch.Tensor, cmi: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """GT-IG labels and top-k exact-CMI labels (k = number of GT-IG edges) over the masked pairs,
    as flat [n * pairs] tensors."""
    y_ig = gt_ig[:, mask].reshape(-1)
    c = cmi[:, mask].reshape(-1)
    y_cmi = torch.zeros_like(y_ig)
    y_cmi[torch.argsort(c, descending=True)[: int(y_ig.sum())]] = True
    return y_ig, y_cmi


def _score_all(
    model, seqs, ctx_mode, med_mode, sargs, simulator="naive", sampler="multinomial"
) -> dict:
    res = [
        score_sequence(
            model,
            s,
            ctx_mode,
            "uniform",
            med_mode,
            sargs,
            sargs.vocab_size,
            simulator=simulator,
            sampler=sampler,
        )
        for s in seqs
    ]
    out = {name: torch.stack([r[name] for r in res]) for name in ("ig", "cmi", "ig_rse", "cmi_rse")}
    out["ess"] = float(torch.stack([r["ess"] for r in res]).mean())
    return out


def _methods(args) -> list[tuple[str, str]]:
    """(context mode, simulator) pairs; an observed context has no sampled past to weight."""
    sims = args.simulators.split(",")
    return [("observed", "naive")] + [
        (ctx, sim) for ctx in ("ancestral", "prepend") for sim in sims
    ]


def _pair_mask(L: int, first_cause: int, last_cause: int | None = None) -> torch.Tensor:  # noqa: N803
    mask = torch.zeros(L, L, dtype=torch.bool)
    for j in range(first_cause, L - 1 if last_cause is None else last_cause):
        mask[j, j + 1 :] = True
    return mask


def context_report(args):
    """Context ablation: for each SCM memory m, context c = m + offset, built as
      observed  : first c observed tokens (causes j >= c scored),
      ancestral : first m observed tokens kept, positions m..c-1 generated (causes j >= c),
      prepend   : c generated tokens placed before the full observed sequence (causes j >= 0),
    each with the naive or the likelihood-weighted simulator (see `score_sequence`).
    Every method is evaluated on ALL pairs of the sequence: pairs it cannot test count as
    score 0 (never detected)."""
    L = args.ablation_len
    meds = args.mediators.split(",")
    methods = _methods(args)
    print(
        f"=== Context ablation: |X|={args.vocab_size}, L={L}, {args.n_eval} sequences, N={args.n_particles}, "
        f"{args.n_seeds} seed(s), uniform cause ===\n"
        "AP vs. the GT of the same estimand (ig: GT-IG, cmi: GT-CMI), on ALL pairs of the sequence\n"
        "(untested pairs score 0); common: causes at positions >= max c; early: causes at positions\n"
        "< max c. RSE: median relative SE of the particle mean on true common pairs. ESS: effective\n"
        "sample size of the particle weights / N."
    )
    for memory in _ints(args.memories):
        cs = [memory + o for o in _ints(args.context_offsets) if memory + o <= L - 4]
        masks = {
            "all": _pair_mask(L, 0),
            "common": _pair_mask(L, max(cs)),
            "early": _pair_mask(L, 0, max(cs)),
        }
        acc: dict = {}
        for seed in range(args.n_seeds):
            scm, seqs, gt_ig, cmi = _ablation_data(memory, L, args, seed)
            labs = {k: _labels(gt_ig, cmi, m) for k, m in masks.items()}
            for mname, model in _models_for(scm, L + max(cs), args, seed).items():
                for c in cs:
                    for ctx_mode, sim in methods:
                        if ctx_mode == "ancestral" and c == memory:
                            continue  # nothing to generate
                        sargs = argparse.Namespace(
                            **{
                                **vars(args),
                                "seq_len": L,
                                "context": c,
                                "guidance": memory,
                                "memory": memory,
                            }
                        )
                        for med_mode in meds:
                            torch.manual_seed(seed + 7)
                            scores = _score_all(
                                model, seqs, ctx_mode, med_mode, sargs, simulator=sim
                            )
                            for k, name in enumerate(("ig", "cmi")):
                                s = scores[name]
                                ap_a, f1_a = average_precision_and_best_f1(
                                    s[:, masks["all"]].reshape(-1), labs["all"][k]
                                )
                                ap_c, _ = average_precision_and_best_f1(
                                    s[:, masks["common"]].reshape(-1), labs["common"][k]
                                )
                                ap_e, _ = average_precision_and_best_f1(
                                    s[:, masks["early"]].reshape(-1), labs["early"][k]
                                )
                                rse = scores[name + "_rse"][:, masks["common"]].reshape(-1)[
                                    labs["common"][k]
                                ]
                                key = (mname, med_mode, name, c, ctx_mode, sim)
                                for metric, v in (
                                    ("ap_all", ap_a),
                                    ("f1_all", f1_a),
                                    ("ap_common", ap_c),
                                    ("ap_early", ap_e),
                                    ("rse", float(rse.median())),
                                    ("ess", scores["ess"]),
                                ):
                                    acc.setdefault(key + (metric,), []).append(v)
        print(
            f"\nmemory={memory}  (common: causes at positions >= {max(cs)}; early: < {max(cs)})\n"
            f"{'model':<9}{'mediators':<10}{'score':<6}{'c':>4} {'context':<10}{'simulator':<10}"
            f"{'AP all':>14}{'F1 all':>14}{'AP common':>14}{'AP early':>14}{'RSE':>14}{'ESS':>7}"
        )
        for mname in ("oracle", "trained"):
            for med_mode in meds:
                for name in ("ig", "cmi"):
                    for c in cs:
                        for ctx_mode, sim in methods:
                            key = (mname, med_mode, name, c, ctx_mode, sim)
                            if key + ("ap_all",) not in acc:
                                continue
                            print(
                                f"{mname:<9}{med_mode:<10}{name:<6}{c:>4} {ctx_mode:<10}{sim:<10}"
                                + "".join(
                                    f"{_pm(acc[key + (m,)]):>14}"
                                    for m in ("ap_all", "f1_all", "ap_common", "ap_early", "rse")
                                )
                                + f"{float(torch.tensor(acc[key + ('ess',)]).mean()):7.2f}"
                            )


def length_report(args):
    """Sequence-length ablation: one SCM per memory, sequences of increasing length, context
    c = memory + 2, every (context mode, simulator). Evaluated on ALL pairs of the sequence
    (untested pairs score 0). The threshold is fitted at --ref-len and applied unchanged at every
    length, to see whether fewer edges are detected in longer sequences; recall is also split by
    the position of the cause (first / middle / last third of the sequence)."""
    lengths = sorted(set(_ints(args.lengths)) | {args.ref_len})
    meds = args.mediators.split(",")
    methods = _methods(args)
    print(
        f"=== Length ablation: |X|={args.vocab_size}, lengths {lengths}, tau fitted at L={args.ref_len}, "
        f"{args.n_eval} sequences, N={args.n_particles}, {args.n_seeds} seed(s) ===\n"
        "AP, F1* vs. the GT of the same estimand, on ALL pairs (untested pairs score 0). At tau_ref:\n"
        "recall, precision, detected/true edges, and recall for causes in the first / middle / last\n"
        "third of the sequence."
    )
    for memory in _ints(args.memories):
        c = memory + 2
        acc: dict = {}
        for seed in range(args.n_seeds):
            models = None
            tau_ref = {}
            for L in [args.ref_len] + [x for x in lengths if x != args.ref_len]:
                scm, seqs, gt_ig, cmi = _ablation_data(memory, L, args, seed)
                if models is None:  # one model per seed, trained on the longest sequences
                    models = _models_for(scm, max(lengths) + c, args, seed)
                allp = _pair_mask(L, 0)
                labs = _labels(gt_ig, cmi, allp)
                cause_pos = torch.nonzero(allp, as_tuple=True)[0].repeat(args.n_eval)
                thirds = torch.bucketize(
                    cause_pos.double(),
                    torch.tensor([(L - 1) / 3, 2 * (L - 1) / 3], dtype=torch.float64),
                )
                sargs = argparse.Namespace(
                    **{
                        **vars(args),
                        "seq_len": L,
                        "context": c,
                        "guidance": memory,
                        "memory": memory,
                    }
                )
                for mname, model in models.items():
                    for ctx_mode, sim in methods:
                        for med_mode in meds:
                            torch.manual_seed(seed + 7)
                            scores = _score_all(
                                model, seqs, ctx_mode, med_mode, sargs, simulator=sim
                            )
                            for k, name in enumerate(("ig", "cmi")):
                                S, Y = scores[name][:, allp].reshape(-1), labs[k]
                                key = (mname, med_mode, name, ctx_mode, sim)
                                if L == args.ref_len:
                                    tau_ref[key] = best_tau(S, Y)[0]
                                ap, f1 = average_precision_and_best_f1(S, Y)
                                pred = S >= tau_ref[key]
                                tp = int((pred & Y).sum())
                                vals = {
                                    "ap": ap,
                                    "f1": f1,
                                    "recall": tp / max(1, int(Y.sum())),
                                    "precision": tp / max(1, int(pred.sum())),
                                    "det_ratio": int(pred.sum()) / max(1, int(Y.sum())),
                                }
                                for b in range(3):
                                    yb = Y & (thirds == b)
                                    vals[f"rec{b}"] = int((pred & yb).sum()) / max(1, int(yb.sum()))
                                for metric, v in vals.items():
                                    acc.setdefault(key + (L, metric), []).append(v)
        print(
            f"\nmemory={memory}, context c={c}\n"
            f"{'model':<9}{'mediators':<10}{'score':<6}{'context':<10}{'simulator':<10}{'L':>4}"
            f"{'AP':>14}{'F1*':>14}{'recall':>14}{'precision':>14}{'det/true':>14}{'rec 1st/2nd/3rd third':>24}"
        )
        for mname in ("oracle", "trained"):
            for med_mode in meds:
                for name in ("ig", "cmi"):
                    for ctx_mode, sim in methods:
                        for L in lengths:
                            key = (mname, med_mode, name, ctx_mode, sim, L)
                            if key + ("ap",) not in acc:
                                continue
                            mean = {
                                m: float(torch.tensor(acc[key + (m,)]).mean())
                                for m in ("rec0", "rec1", "rec2")
                            }
                            print(
                                f"{mname:<9}{med_mode:<10}{name:<6}{ctx_mode:<10}{sim:<10}{L:>4}"
                                + "".join(
                                    f"{_pm(acc[key + (m,)]):>14}"
                                    for m in ("ap", "f1", "recall", "precision", "det_ratio")
                                )
                                + f"{mean['rec0']:>10.3f}/{mean['rec1']:.3f}/{mean['rec2']:.3f}"
                            )


def label_free_thresholds(
    S: torch.Tensor,  # noqa: N803
    lags: torch.Tensor,
    args,
    margin: float | None = None,
) -> dict:
    """Thresholds computed from the scores alone (no labels).

    Null-calibrated: under the bounded-lagged-effects assumption, pairs with lag >= --null-min-lag
    have no effect, so their scores are an empirical null of the estimator (Monte Carlo noise and
    model error). null-q: the (1 - alpha) quantile of the null. null-BH: empirical p-values
    p(s) = (1 + #{null >= s}) / (1 + n_null) for the candidate pairs (lag < --null-min-lag), then
    Benjamini-Hochberg at level --fdr. Generic: percentiles, Otsu, MAD, 2-component GMM,
    mean + 3 sd (the mu + k sigma rule of OSCAR with k = 3).

    A null test only separates "some effect" from "no effect". When edges are defined by a
    minimum effect size (the relevance margin delta of the ground truth), the margin is a domain
    choice that a label-free threshold also needs: delta, and null-q + delta (noise floor plus
    margin), are added when `margin` is given (information gain, whose GT is IG > delta).
    """
    cand = lags < args.null_min_lag
    null = torch.sort(S[~cand]).values
    out = {
        "p90": percentile_threshold(S, 0.90),
        "p95": percentile_threshold(S, 0.95),
        "p99": percentile_threshold(S, 0.99),
        "otsu": otsu_threshold(S),
        "mad": mad_threshold(S),
        "gmm": gmm_threshold(S).tau,
        "mean+3sd": float(S.mean() + 3 * S.std()),
    }
    tiny = 1e-12  # an exactly-zero null must not make every pair an edge, only every positive one
    out["null-q"] = max(float(torch.quantile(null, 1 - args.null_alpha)), tiny)
    s_c = S[cand]
    n_null = null.numel()
    ge = n_null - torch.searchsorted(null, s_c.contiguous(), right=False)  # #null >= s
    pvals = (1.0 + ge.double()) / (1.0 + n_null)
    order = torch.argsort(pvals)
    m = pvals.numel()
    below = pvals[order] <= torch.arange(1, m + 1, dtype=torch.float64) / m * args.fdr
    if bool(below.any()):
        k = int(torch.nonzero(below)[-1]) + 1
        out["null-BH"] = max(float(s_c[order[:k]].min()), tiny)
    else:
        out["null-BH"] = float("inf")
    if margin is not None:
        out["delta"] = margin
        out["null-q+delta"] = out["null-q"] + margin
    return out


@torch.no_grad()
def score_sequence_exact(
    model, seq: torch.Tensor, past_mode: str, weights: str, args, vocab: int, n_past: int = 1
):
    """Deterministic estimator: mediators observed, and the cause-absent branch computed exactly
    by enumerating every alternative token v != x_j at the cause position (one batched forward
    pass of V sequences per cause) instead of sampling replacement tokens.

      past_mode none    : no past before x_0 (position 0 uses the uniform prior), no sampling at all;
                prepend : `n_past` pasts of length c sampled as in `score_sequence` and averaged.
      weights   uniform : alternatives weighted uniformly (the do-intervention of GT-IG);
                model   : weighted by the model's P(v | past) / (1 - pi), i.e. X_t ~ P(X_t | x_<t, X_t != x_t).

    ig  = sum_v w_v KL(Bern(q1) || Bern(q_v))      (expected information gain of the observed cause)
    cmi = pi KL(q1 || m) + (1 - pi) KL(q0 || m),  q0 = sum_v w_v q_v,  m = pi q1 + (1 - pi) q0
    """
    L, V = seq.shape[0], vocab
    if past_mode == "prepend" and args.context > 0:
        c, P = args.context, n_past
        past = torch.randint(0, V, (P, c))
        base = autoregressive_fill(
            model, torch.cat([past, seq.unsqueeze(0).repeat(P, 1)], dim=1), min(args.guidance, c), c
        )
        o = c
    else:
        P, o = 1, 0
        base = seq.unsqueeze(0)
    T = base.shape[1]
    base_probs = next_token_probs(model, base)  # [P, T, V]
    out = {k: torch.zeros(L, L, dtype=torch.float64) for k in ("ig", "cmi")}
    allv = torch.arange(V)
    for j in range(0, L - 1):
        J = o + j
        xj = int(seq[j])
        prior = (
            base_probs[:, J - 1, :] if J > 0 else torch.full((P, V), 1.0 / V, dtype=torch.float64)
        )
        X = base.repeat_interleave(V, dim=0)  # [P*V, T]
        X[:, J] = allv.repeat(P)
        probs = next_token_probs(model, X)
        qv = prob_of(probs, X, range(J + 1, T)).view(P, V, -1)  # P(observed effect | cause = v)
        q1 = qv[:, xj, :]  # [P, Q]
        if weights == "uniform":
            w = torch.full((P, V), 1.0 / (V - 1), dtype=torch.float64)
        else:
            w = prior.clone()
        w[:, xj] = 0.0
        w = w / w.sum(-1, keepdim=True)
        q0 = (w.unsqueeze(-1) * qv).sum(1)  # [P, Q]
        ig = (w.unsqueeze(-1) * bern_kl(q1.unsqueeze(1), qv)).sum(1)
        pi = prior[:, xj].unsqueeze(-1)
        m = pi * q1 + (1 - pi) * q0
        cmi = pi * bern_kl(q1, m) + (1 - pi) * bern_kl(q0, m)
        out["ig"][j, j + 1 :] = ig.mean(0)
        out["cmi"][j, j + 1 :] = cmi.mean(0)
    return out


def nosample_report(args):
    """Do we need particles at all? Mediators observed (atomic), all pairs of the sequence.
    Particle estimators (N in --particle-grid) with or without a sampled past (prepend c=m+2 or no
    past), replacement tokens uniform-except-observed or from the model, vs. the deterministic
    exact estimator (no particles), with or without sampled pasts."""
    L = args.ablation_len
    grid = _ints(args.particle_grid)
    methods = [
        ("particles", past, cz, n)
        for past in ("none", "prepend")
        for cz in ("uniform!=", "model!=")
        for n in grid
    ]
    methods += [
        ("exact", past, wz, 0) for past in ("none", "prepend") for wz in ("uniform", "model")
    ]
    print(
        f"=== Particles vs. no sampling: |X|={args.vocab_size}, L={L}, {args.n_eval} sequences, "
        f"{args.n_seeds} seed(s), mediators observed, all pairs ===\n"
        "past: none = no tokens before x_0; prepend = c = memory + 2 sampled tokens before x_0\n"
        f"(exact+prepend averages {args.exact_past_particles} sampled pasts). cause-absent branch: particles\n"
        "draw one replacement token per particle (uniform!= or model!=); exact enumerates all of them\n"
        "(weights uniform or model). AP and best F1 vs. GT-IG and GT-CMI. Note: on the exact SCM, exact\n"
        "+ model weights is the GT-CMI estimand itself and exact + uniform is (up to the KL direction)\n"
        "the GT-IG estimand; the trained model is the informative case. s/seq: seconds per sequence."
    )
    for memory in _ints(args.memories):
        c = memory + 2
        allp = _pair_mask(L, 0)
        acc: dict = {}
        for seed in range(args.n_seeds):
            scm, seqs, gt_ig, cmi = _ablation_data(memory, L, args, seed)
            labs = _labels(gt_ig, cmi, allp)
            for mname, model in _models_for(scm, L + c, args, seed).items():
                for kind, past, opt, n in methods:
                    sargs = argparse.Namespace(
                        **{
                            **vars(args),
                            "seq_len": L,
                            "context": c if past == "prepend" else 0,
                            "guidance": memory,
                            "memory": memory,
                            "n_particles": max(n, 1),
                        }
                    )
                    torch.manual_seed(seed + 7)
                    t0 = time.time()
                    if kind == "particles":
                        res = [
                            score_sequence(
                                model, s, "prepend", opt, "observed", sargs, args.vocab_size
                            )
                            for s in seqs
                        ]
                    else:
                        res = [
                            score_sequence_exact(
                                model,
                                s,
                                past,
                                opt,
                                sargs,
                                args.vocab_size,
                                n_past=args.exact_past_particles if past == "prepend" else 1,
                            )
                            for s in seqs
                        ]
                    dt = (time.time() - t0) / len(seqs)
                    for name in ("ig", "cmi"):
                        S = torch.stack([r[name] for r in res])[:, allp].reshape(-1)
                        ap1, f1a = average_precision_and_best_f1(S, labs[0])
                        ap2, f1b = average_precision_and_best_f1(S, labs[1])
                        key = (mname, name, kind, past, opt, n)
                        for metric, v in (
                            ("ap_ig", ap1),
                            ("f1_ig", f1a),
                            ("ap_cmi", ap2),
                            ("f1_cmi", f1b),
                            ("sec", dt),
                        ):
                            acc.setdefault(key + (metric,), []).append(v)
        print(
            f"\nmemory={memory}\n"
            f"{'model':<9}{'score':<6}{'estimator':<10}{'past':<9}{'cause-absent':<13}{'N':>4}"
            f"{'AP(GT-IG)':>14}{'F1(GT-IG)':>14}{'AP(GT-CMI)':>14}{'F1(GT-CMI)':>14}{'s/seq':>7}"
        )
        for mname in ("oracle", "trained"):
            for name in ("ig", "cmi"):
                for kind, past, opt, n in methods:
                    key = (mname, name, kind, past, opt, n)
                    if key + ("ap_ig",) not in acc:
                        continue
                    print(
                        f"{mname:<9}{name:<6}{kind:<10}{past:<9}{opt:<13}{n:>4}"
                        + "".join(
                            f"{_pm(acc[key + (m,)]):>14}"
                            for m in ("ap_ig", "f1_ig", "ap_cmi", "f1_cmi")
                        )
                        + f"{float(torch.tensor(acc[key + ('sec',)]).mean()):7.2f}"
                    )


def nullcal_report(args):
    """Label-free thresholds vs. the oracle (best-F1) threshold. Prepended context, naive
    simulator, uniform cause, all pairs of the sequence, sequence length --ablation-len."""
    L = args.ablation_len
    meds = args.mediators.split(",")
    names = (
        "oracle tau*",
        "null-BH",
        "null-q",
        "delta",
        "null-q+delta",
        "p90",
        "p95",
        "p99",
        "otsu",
        "mad",
        "gmm",
        "mean+3sd",
    )
    print(
        f"=== Label-free thresholds: |X|={args.vocab_size}, L={L}, prepended context c=memory+2, "
        f"{args.n_eval} sequences, N={args.n_particles}, {args.n_seeds} seed(s) ===\n"
        f"null: pairs with lag >= {args.null_min_lag} (bounded lagged effects); null-q: {1 - args.null_alpha:.2f} "
        f"quantile of the null; null-BH: Benjamini-Hochberg at FDR {args.fdr} with empirical p-values.\n"
        "F1 / precision / recall vs. the GT of the same estimand (ig: GT-IG, cmi: GT-CMI), all pairs;\n"
        "rel: F1 / best F1 (oracle tau*). The oracle row uses the labels; every other row does not."
    )
    for memory in _ints(args.memories):
        c = memory + 2
        allp = _pair_mask(L, 0)
        jj, qq = torch.nonzero(allp, as_tuple=True)
        acc: dict = {}
        for seed in range(args.n_seeds):
            scm, seqs, gt_ig, cmi = _ablation_data(memory, L, args, seed)
            labs = _labels(gt_ig, cmi, allp)
            lags = (qq - jj).repeat(args.n_eval)
            sargs = argparse.Namespace(
                **{**vars(args), "seq_len": L, "context": c, "guidance": memory, "memory": memory}
            )
            for mname, model in _models_for(scm, L + c, args, seed).items():
                for med_mode in meds:
                    torch.manual_seed(seed + 7)
                    scores = _score_all(model, seqs, "prepend", med_mode, sargs)
                    for k, name in enumerate(("ig", "cmi")):
                        S, Y = scores[name][:, allp].reshape(-1), labs[k]
                        taus = label_free_thresholds(
                            S, lags, args, margin=args.delta if name == "ig" else None
                        )
                        tau_star, f1_star = best_tau(S, Y)
                        taus = {"oracle tau*": tau_star, **taus}
                        for tname, tau in taus.items():
                            pred = S >= tau
                            tp = int((pred & Y).sum())
                            f1 = f1_at(S, Y, tau)
                            vals = {
                                "f1": f1,
                                "precision": tp / max(1, int(pred.sum())),
                                "recall": tp / max(1, int(Y.sum())),
                                "rel": f1 / max(f1_star, 1e-12),
                            }
                            for metric, v in vals.items():
                                acc.setdefault((mname, med_mode, name, tname, metric), []).append(v)
        print(
            f"\nmemory={memory}, c={c}\n"
            f"{'model':<9}{'mediators':<10}{'score':<6}{'threshold':<13}"
            f"{'F1':>14}{'precision':>14}{'recall':>14}{'rel':>14}"
        )
        for mname in ("oracle", "trained"):
            for med_mode in meds:
                for name in ("ig", "cmi"):
                    for tname in names:
                        key = (mname, med_mode, name, tname)
                        if key + ("f1",) not in acc:
                            continue
                        print(
                            f"{mname:<9}{med_mode:<10}{name:<6}{tname:<13}"
                            + "".join(
                                f"{_pm(acc[key + (m,)]):>14}"
                                for m in ("f1", "precision", "recall", "rel")
                            )
                        )


def sampler_report(args):
    """How the past is generated: ancestral (overwrite positions m..c-1) vs prepend (c tokens
    before the full sequence), each with several decoding rules for the generated tokens,
    against the observed-context baseline. Context c = memory + --sampler-offset, naive
    simulator, evaluated on ALL pairs (untested pairs score 0)."""
    L = args.ablation_len
    meds = args.mediators.split(",")
    samplers = args.context_samplers.split(",")
    methods = [("observed", "-")] + [(ctx, s) for ctx in ("ancestral", "prepend") for s in samplers]
    print(
        f"=== Context sampler comparison: |X|={args.vocab_size}, L={L}, c=memory+{args.sampler_offset}, "
        f"{args.n_eval} sequences, N={args.n_particles}, {args.n_seeds} seed(s), uniform cause, naive simulator ===\n"
        "AP vs. the GT of the same estimand on ALL pairs (untested pairs score 0); common: causes at\n"
        "positions >= c (pairs every method tests). RSE: median relative SE of the particle mean on\n"
        "true common pairs. s/seq: seconds per sequence."
    )
    for memory in _ints(args.memories):
        c = memory + args.sampler_offset
        masks = {"all": _pair_mask(L, 0), "common": _pair_mask(L, c)}
        acc: dict = {}
        for seed in range(args.n_seeds):
            scm, seqs, gt_ig, cmi = _ablation_data(memory, L, args, seed)
            labs = {k: _labels(gt_ig, cmi, m) for k, m in masks.items()}
            sargs = argparse.Namespace(
                **{**vars(args), "seq_len": L, "context": c, "guidance": memory, "memory": memory}
            )
            for mname, model in _models_for(scm, L + c, args, seed).items():
                for ctx_mode, sampler in methods:
                    for med_mode in meds:
                        torch.manual_seed(seed + 7)
                        t0 = time.time()
                        scores = _score_all(
                            model,
                            seqs,
                            ctx_mode,
                            med_mode,
                            sargs,
                            sampler=sampler if sampler != "-" else "multinomial",
                        )
                        dt = (time.time() - t0) / len(seqs)
                        for k, name in enumerate(("ig", "cmi")):
                            s = scores[name]
                            ap_a, f1_a = average_precision_and_best_f1(
                                s[:, masks["all"]].reshape(-1), labs["all"][k]
                            )
                            ap_c, _ = average_precision_and_best_f1(
                                s[:, masks["common"]].reshape(-1), labs["common"][k]
                            )
                            rse = scores[name + "_rse"][:, masks["common"]].reshape(-1)[
                                labs["common"][k]
                            ]
                            key = (mname, med_mode, name, ctx_mode, sampler)
                            for metric, v in (
                                ("ap_all", ap_a),
                                ("f1_all", f1_a),
                                ("ap_common", ap_c),
                                ("rse", float(rse.median())),
                                ("sec", dt),
                            ):
                                acc.setdefault(key + (metric,), []).append(v)
        print(
            f"\nmemory={memory}, c={c}\n"
            f"{'model':<9}{'mediators':<10}{'score':<6}{'context':<10}{'sampler':<13}"
            f"{'AP all':>14}{'F1 all':>14}{'AP common':>14}{'RSE':>14}{'s/seq':>7}"
        )
        for mname in ("oracle", "trained"):
            for med_mode in meds:
                for name in ("ig", "cmi"):
                    for ctx_mode, sampler in methods:
                        key = (mname, med_mode, name, ctx_mode, sampler)
                        if key + ("ap_all",) not in acc:
                            continue
                        print(
                            f"{mname:<9}{med_mode:<10}{name:<6}{ctx_mode:<10}{sampler:<13}"
                            + "".join(
                                f"{_pm(acc[key + (m,)]):>14}"
                                for m in ("ap_all", "f1_all", "ap_common", "rse")
                            )
                            + f"{float(torch.tensor(acc[key + ('sec',)]).mean()):7.2f}"
                        )


def particles_report(args):
    """Stability vs. number of particles N (exact SCM): two independent particle draws per N."""
    scm, seqs, pair_mask, lab_ig, lab_cmi, lags = build_setting(
        args.vocab_size, args.decay_rate, args
    )
    print(
        f"=== Particles (exact SCM, |X|={args.vocab_size}, {args.n_eval} sequences, ancestral context, "
        f"uniform cause) ===\n"
        "AP vs. the GT of the same estimand (ig: GT-IG, cmi: GT-CMI), median relative SE of the particle\n"
        "mean on true edges, Spearman rank correlation of the scores between two independent draws (pairs\n"
        "with lag <= memory), and |log10(tau*_1 / tau*_2)| between the two draws."
    )
    print(
        f"{'N':>5} {'mediators':<10}{'score':<6}{'AP':>7}{'RSE':>7}{'rank corr':>11}{'tau drift':>11}"
    )
    for N in (8, 32, 128):
        nargs = argparse.Namespace(**{**vars(args), "n_particles": N})
        for md in ("uniform", "observed"):
            draws = []
            for rep in range(2):
                torch.manual_seed(args.seed + 7 + 100 * rep)
                draws.append(
                    [
                        score_sequence(
                            FastSCM(scm), s, "ancestral", "uniform", md, nargs, args.vocab_size
                        )
                        for s in seqs
                    ]
                )
            for name, labs in (("ig", lab_ig), ("cmi", lab_cmi)):
                Y = torch.cat(labs)
                S1 = torch.cat([r[name][pair_mask] for r in draws[0]])
                S2 = torch.cat([r[name][pair_mask] for r in draws[1]])
                rse = torch.cat([r[name + "_rse"][pair_mask] for r in draws[0]])[Y]
                ap, _ = average_precision_and_best_f1(S1, Y)
                near = (
                    lags.repeat(len(seqs)) <= args.memory
                )  # beyond the memory the exact SCM gives ties at 0
                r1 = torch.argsort(torch.argsort(S1[near])).double()
                r2 = torch.argsort(torch.argsort(S2[near])).double()
                rho = float(torch.corrcoef(torch.stack([r1, r2]))[0, 1])
                t1, _ = best_tau(S1, Y)
                t2, _ = best_tau(S2, Y)
                drift = abs(math.log10(max(t1, 1e-12) / max(t2, 1e-12)))
                print(
                    f"{N:>5} {md:<10}{name:<6}{ap:7.3f}{float(rse.median()):7.3f}{rho:11.3f}{drift:11.3f}"
                )


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "nosample":
        nosample_report(args)
        return
    if args.mode == "nullcal":
        nullcal_report(args)
        return
    if args.mode == "sampler":
        sampler_report(args)
        return
    if args.mode == "context":
        context_report(args)
        return
    if args.mode == "length":
        length_report(args)
        return
    if args.mode == "multiseed":
        multiseed_report(args)
        return
    if args.mode == "particles":
        particles_report(args)
        return
    if args.mode == "train":
        scm = NonlinearSCM(
            vocab_size=args.vocab_size,
            memory=args.memory,
            sparsity=args.sparsity,
            decay_rate=args.decay_rate,
            seed=args.seed,
        )
        torch.manual_seed(args.seed)
        t0 = time.time()
        _, oracle = train_model(scm, args)
        print(f"done in {time.time() - t0:.0f}s: {oracle}")
        return
    torch.manual_seed(args.seed)
    scm, eval_seqs, pair_mask, lab_ig, lab_cmi, lags = build_setting(
        args.vocab_size, args.decay_rate, args
    )
    c, L = args.context, args.seq_len
    labels = torch.cat(lab_ig)
    labels_cmi = torch.cat(lab_cmi)
    k = int(labels.sum())
    overlap = int((labels & labels_cmi).sum())
    print(
        f"SCM: |X|={args.vocab_size}, memory={args.memory}, L={L}, c={c}, "
        f"{args.n_eval} eval sequences, {k} true edges / {labels.numel()} pairs, "
        f"N={args.n_particles}\n"
        f"GT-IG: atomic info gain of the observed cause > {args.delta} (ground_truth_adjacency); "
        f"GT-CMI: top-{k} exact CMI; the two share {overlap} edges"
    )

    models = {}
    if "oracle" in args.models:
        models["oracle (exact SCM)"] = FastSCM(scm)
    if "trained" in args.models:
        t0 = time.time()
        model, oracle = train_model(scm, args)
        print(
            f"trained tiny Llama in {time.time() - t0:.0f}s: loss={oracle['loss']:.3f}, "
            f"H(P)={oracle['h_p']:.3f}, oracle score eps_hat={oracle['epsilon_hat']:.3f}"
        )
        models["trained tiny Llama"] = model

    stored = {}
    for mname, model in models.items():
        print(f"\n=== {mname} ===")
        print(
            f"{'context':<10}{'cause':<11}{'mediators':<10}| {'score':<6}"
            f"{'AP(GT-IG)':>10}{'F1':>7}{'AP(GT-CMI)':>11}{'F1':>7}{'RSE':>7}"
        )
        for ctx_mode, cause_mode, med_mode in itertools.product(CONTEXTS, CAUSES, MEDIATORS):
            torch.manual_seed(args.seed + 7)
            t0 = time.time()
            res = [
                score_sequence(model, s, ctx_mode, cause_mode, med_mode, args, args.vocab_size)
                for s in eval_seqs
            ]
            for name in ("ig", "cmi"):
                if (ctx_mode, cause_mode, med_mode) in THRESH_VARIANTS:
                    stored[(mname, med_mode, name)] = [r[name][pair_mask] for r in res]
                scores = torch.cat([r[name][pair_mask] for r in res])
                rse = torch.cat([r[name + "_rse"][pair_mask] for r in res])[labels]
                ap1, f1a = average_precision_and_best_f1(scores, labels)
                ap2, f1b = average_precision_and_best_f1(scores, labels_cmi)
                tag = (
                    "*"
                    if (ctx_mode, cause_mode, med_mode, name)
                    == ("ancestral", "uniform", "uniform", "ig")
                    else " "
                )
                print(
                    f"{ctx_mode:<10}{cause_mode:<11}{med_mode:<10}| {name + tag:<6}"
                    f"{ap1:10.3f}{f1a:7.3f}{ap2:11.3f}{f1b:7.3f}{float(rse.median()):7.3f}"
                    + (f"   ({time.time() - t0:.0f}s)" if name == "cmi" else "")
                )
    print("\n* = current TRACE default (ancestral context, uniform cause and mediators, info gain)")

    print("\n=== Thresholds: information gain of the observed cause vs. CMI ===")
    print(THRESH_LEGEND)
    for mname in models:
        print(f"\n{mname}")
        print(THRESH_HEADER)
        for med_mode in ("uniform", "observed"):
            variant = "staircase (uniform)" if med_mode == "uniform" else "atomic (observed)"
            for name in ("ig", "cmi"):
                for gt, labs in (("IG", lab_ig), ("CMI", lab_cmi)):
                    row = threshold_row(stored[(mname, med_mode, name)], labs, lags, args.memory)
                    print(f"{variant:<20}{name:<5}{gt:<5}|{row}")

    sweep_report(args)


if __name__ == "__main__":
    main()
