"""Run LMT once on an events file (plans/lmt.md §2; D-SB-29, D-LM-1 … D-LM-4). Executed by the
method's own interpreter (the `lmt` conda environment: python 3.11, torch, numpy), never imported
by the harness:

    <lmt python> -I lmt_runner.py --method-src <clone> --events <events.npz> \\
        --allowed <pairs.npy | none> --setting <prior_weight_milli> \\
        --method-args epochs=200 lr=0.05 beta_dag=0.1 lambda_e=0.1 init_std=0.35 a_init=-2 seed=0 \\
                      device=cpu q_in=0.5 q_out=0.01 dtype=float64 param_dtype=float64 \\
                      chunk_events=0 --out <scores.npy>

LMT (Xiao, Chen, Sun, Shi, Yue 2026; `github.com/xx987/LMT`, no licence file — imported from
the pinned clone under `--method-src`, never vendored) fits a multivariate exponential-kernel
Hawkes process over the event types by MAP: the objective of the shipped trainer
`chemical_1000_event_repro/temporal_hawkes_pure_map.py` (`PureTemporalMultivariateHawkes`),

    J = L_time(A, α, β, μ) + w · log p(A | q) − β_dag · h(A) − λ_e · Σ_{i≠j} A_ij,   loss = −J,

with `A = σ(A_raw) ⊙ (1 − I)`, `α = softplus(·) ⊙ (1 − I) + 1e-8`, `β = softplus(·) ⊙ (1 − I) +
1e-4`, `μ = softplus(·) + 1e-5`, Adam, `log μ` initialised from `N / (K · T)`, `A_raw` from the
offset `a_init` (shipped: 0) and (`init_std` > 0) Gaussian noise on every raw parameter in the
shipped order, and the prior / penalty terms imported
from the clone's `text_causal.graph_prior` (`semantic_log_bernoulli`, `dag_penalty_h`,
`sparsity_sum`). The weight `w` is the harness's setting, `prior_weight_milli / 1000` (the paper's
λ; the shipped 1000-event trainer has no knob, i.e. `w = 1`). The fixed arguments are registered
in `plans/lmt.md` (its addendum of 2026-11-03, D-LM-5: `epochs=200 beta_dag=0.1 a_init=-2`
selected on the xs seed-0 validation split; the shipped `epochs=50 beta_dag=1.0 a_init=0` start,
σ(0) = 0.5 on every pair, puts h(A) at exp(0.25 (K − 1)) and leaves the ranking at chance at K = 77).

Here the clusters are the event types (K = n_types; no `node`), and the q-matrix is built from the
mask: `q_in` on every allowed (cause, effect) pair, `q_out` on every other off-diagonal pair, 0.5 on
the diagonal as the shipped loader does; with `--allowed none`, `q_default` everywhere (D-LM-3).
Timestamps arrive as integer milliseconds and are converted to seconds (D-LM-2).

**D-LM-1 — the likelihood is reimplemented, vectorised.** The shipped `forward` is a pure-Python
double loop over events (O(N²) scalar torch ops) that cannot run on the harness's samples; the
owner approved a lab reimplementation of the same objective (charter policy for missing / broken
code). For every event n and cause type a, with the pair's own decay β_{a, z_n},

    S_{a, z_n}(t_n) = Σ_{m : z_m = a, t_m < t_n} exp(−β_{a, z_n} (t_n − t_m))
                    = exp( logcumsumexp_{j < k_n^a}(β_{a, ·} t^a_j)[z_n] − β_{a, z_n} t_n ),

computed per cause type a over its sorted times `t^a` (one column per effect type), read at
`k_n^a` = the number of type-a events strictly before `t_n` (`searchsorted`, side left — ties
contribute nothing, as the shipped `dt <= 0: continue`); λ_{z_n}(t_n) = μ_{z_n} + Σ_a A α β S;
the compensator is the shipped closed form `T Σ_b μ_b + Σ_b Σ_m A α (1 − exp(−β_{z_m, b}(T −
t_m)))` with `T = t_N`. Memory is O(N · K); `chunk_events` > 0 evaluates the per-event terms in
chunks of that many events with one backward per chunk (same gradient; the prefix sums are
recomputed per chunk). The kernel sums run in `dtype` (default float64: `logcumsumexp` subtracts
two numbers of size β · T, which float32 resolves to ~1e-3 relative on a day-long timeline); the
raw parameters, the prior and the penalty run in `param_dtype` (default float64 — the shipped
model in double precision; `float32` is the shipped precision, under which h(A)'s gradient
overflows Adam's second moment from K ≈ 200 at the shipped initialisation and the adjacency never
moves, D-LM-4). The
equality of this forward with the shipped loop on the clone's 50-event sample is recorded in
`algos/lmt-xiao-2026/RUN.md` (D-LM-1).

Output: the final `A = σ(A_raw) ⊙ (1 − I)` as a float32 `(n_types, n_types)` matrix in [0, 1]
(row = cause; every off-diagonal entry is positive, so the τ sweep is the selection) and the
sidecar `<out>.json` (`n_types`, `n_events`, `n_allowed_pairs`, `n_nonzero`, `wall_seconds`,
`setting`, `method_args`, `method_version` = the clone's commit, `method_env` = this
interpreter's `pip freeze`, `final_loglike`, `final_h`, `epochs`, `n_above_half`,
`n_above_half_outside_mask`, timing).

**Test-only path.** `fit=stub` returns a graded matrix without torch — `0.3 + 0.7 · rank
fraction` on every allowed pair (in sorted pair order), 0.9 on one pair outside the mask (the
reverse of the first allowed pair), 0 elsewhere; without a mask, `0.3 + 0.7 · i / (K − 2)` on
`(i, i + 1)` — so the harness
test suite exercises the family's plumbing (a soft output, an edge outside the mask) without a
fit. It is recorded in the sidecar's `method_args`; a read whose record carries `fit=stub` is not
a result. Standard library and numpy only on the harness side; torch only inside the fit.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time

DEFAULTS = {
    # the registered fixed args (plans/lmt.md addendum 2026-11-03, D-LM-5: selected on the xs seed-0
    # validation split from a pre-stated grid); the clone's `chemical_1000_event_repro/run_k5.sh`
    # values stay selectable: epochs=50 beta_dag=1.0 a_init=0
    "epochs": 200,
    "lr": 0.05,
    "beta_dag": 0.1,
    "lambda_e": 0.1,
    "init_std": 0.35,
    "a_init": -2.0,  # offset of A_raw at init: σ(−2) ≈ 0.12 per pair (the shipped start is 0, σ(0) = 0.5)
    "seed": 0,
    "device": "cpu",
    # the q-matrix rule (D-LM-3)
    "q_in": 0.5,
    "q_out": 0.01,
    "q_default": 0.5,
    # implementation knobs of the vectorised likelihood (D-LM-1); numerically neutral up to
    # floating-point summation order
    "dtype": "float64",
    "param_dtype": "float64",
    "chunk_events": 0,
    # the test-only path
    "fit": "torch",
}
TYPES = {
    "epochs": int,
    "lr": float,
    "beta_dag": float,
    "lambda_e": float,
    "init_std": float,
    "a_init": float,
    "seed": int,
    "device": str,
    "q_in": float,
    "q_out": float,
    "q_default": float,
    "dtype": str,
    "param_dtype": str,
    "chunk_events": int,
    "fit": str,
}
LAM_FLOOR = 1e-12  # the shipped `lam.clamp(min=1e-12)`


def pip_freeze():
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        return [ln for ln in out.splitlines() if ln.strip()] or ["<empty pip freeze>"]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [f"<pip freeze unavailable: {type(e).__name__}>"]


def method_version(method_src):
    """The clone's HEAD commit, else 'unknown' (a stub directory under test)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=method_src,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_method_args(items):
    """`k=v` tokens over DEFAULTS; an unknown key or a bad value is refused."""
    cfg = dict(DEFAULTS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"lmt_runner: --method-args entries are KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"lmt_runner: unknown method-arg {k!r}; known: {sorted(cfg)}")
        try:
            cfg[k] = TYPES[k](v)
        except ValueError as e:
            raise SystemExit(f"lmt_runner: bad value for {k}: {v!r} ({e})") from e
    for k in ("q_in", "q_out", "q_default"):
        if not 0.0 < cfg[k] < 1.0:
            raise SystemExit(f"lmt_runner: {k} must lie strictly inside (0, 1), got {cfg[k]}")
    if cfg["epochs"] < 1 or cfg["lr"] <= 0 or cfg["init_std"] < 0 or cfg["chunk_events"] < 0:
        raise SystemExit("lmt_runner: epochs >= 1, lr > 0, init_std >= 0, chunk_events >= 0")
    for k in ("dtype", "param_dtype"):
        if cfg[k] not in ("float32", "float64"):
            raise SystemExit(f"lmt_runner: {k} must be float32 or float64, got {cfg[k]!r}")
    if cfg["fit"] not in ("torch", "stub"):
        raise SystemExit(f"lmt_runner: fit must be torch or stub, got {cfg['fit']!r}")
    return cfg


def q_matrix(n_types, allowed, q_in, q_out, q_default):
    """The Bernoulli prior over edges (D-LM-3): `q_in` on the allowed (cause, effect) pairs,
    `q_out` on every other off-diagonal pair; `q_default` everywhere without a mask; 0.5 on the
    diagonal as the clone's `load_q_prior_matrix`."""
    import numpy as np

    if allowed is None:
        q = np.full((n_types, n_types), q_default, dtype=np.float64)
    else:
        q = np.full((n_types, n_types), q_out, dtype=np.float64)
        if len(allowed):
            q[allowed[:, 0], allowed[:, 1]] = q_in
    np.fill_diagonal(q, 0.5)
    return q


def stub_scores(n_types, allowed):
    """The test-only graded matrix (module docstring)."""
    import numpy as np

    m = np.zeros((n_types, n_types), dtype=np.float32)
    if allowed is None:
        k = max(n_types - 2, 1)
        for i in range(n_types - 1):
            m[i, i + 1] = 0.3 + 0.7 * i / k
        return m
    pairs = sorted({(int(a), int(b)) for a, b in allowed if int(a) != int(b)})
    k = max(len(pairs) - 1, 1)
    for i, (a, b) in enumerate(pairs):
        m[a, b] = 0.3 + 0.7 * i / k
    inside = set(pairs)
    # the one edge outside the mask: the reverse of the first allowed pair (a topology mask
    # names no reverse pair, and an allowed pair never lies within one op), else the first
    # off-diagonal pair outside
    candidates = [(b, a) for a, b in pairs[:1]] + [
        (a, b) for a in range(n_types) for b in range(n_types) if a != b
    ]
    for a, b in candidates:
        if (a, b) not in inside:
            m[a, b] = 0.9
            break
    return m


# --- the vectorised likelihood (D-LM-1) ------------------------------------------------------
def hawkes_chunk(t, c, A, Alpha, Beta, mu, n_types, n0, n1, with_mu_term):
    """The per-event log-intensities of events `n0 … n1 − 1` and their share of the compensator,
    vectorised (module docstring). `t` (N,) sorted times in the compute dtype, `c` (N,) int64
    types, `A`/`Alpha`/`Beta` (K, K) and `mu` (K,) in the compute dtype. `T Σ_b μ_b` is added
    when `with_mu_term`. Returns `(log_lam, comp)`: `log_lam` the chunk's `log λ_{z_n}(t_n)`
    (clamped as shipped), `comp` a scalar."""
    import torch

    T = t[-1]
    W = A * Alpha * Beta  # the intensity weight of every (cause, effect) pair
    AA = A * Alpha  # the compensator weight
    tc = t[n0:n1]
    cc = c[n0:n1]
    lam = mu[cc].clone()
    for a in range(n_types):
        ta = t[c == a]  # the type-a times, sorted (t is)
        if ta.shape[0] == 0:
            continue
        k = torch.searchsorted(ta, tc, right=False)  # type-a events strictly before t_n
        kmax = int(k.max().item())
        if kmax == 0:
            continue
        # logcumsumexp over the prefix of type-a times that this chunk can see
        L = torch.logcumsumexp(Beta[a][None, :] * ta[:kmax, None], dim=0)  # (kmax, K)
        has = k > 0
        row = (k - 1).clamp(min=0)
        g = L[row, cc]  # (n1 - n0,)
        # the exponent is log Σ_{m<n} exp(−β (t_n − t_m)) ≤ log k on rows that have an
        # earlier type-a event; rows without one are zeroed before the exp so that no
        # overflow (and no 0 · inf in the backward) can enter
        expo = g - Beta[a, cc] * tc
        expo = torch.where(has, expo, torch.zeros_like(expo))
        s = torch.exp(expo) * has.to(expo.dtype)
        lam = lam + W[a, cc] * s
    log_lam = torch.log(lam.clamp(min=LAM_FLOOR))
    comp = (AA[cc] * (1.0 - torch.exp(-Beta[cc] * (T - tc)[:, None]))).sum()
    if with_mu_term:
        comp = comp + mu.sum() * T
    return log_lam, comp


def chunk_ranges(n_events, chunk_events):
    size = n_events if chunk_events <= 0 else int(chunk_events)
    return [(n0, min(n_events, n0 + size)) for n0 in range(0, n_events, size)]


def hawkes_terms(t, c, A, Alpha, Beta, mu, n_types, chunk_events=0):
    """`hawkes_chunk` over the chunks of the timeline (no backward in between): yields one
    `(log_lam, comp)` per chunk; summed, they equal the shipped `(log_lam.sum(), comp)`."""
    for i, (n0, n1) in enumerate(chunk_ranges(int(t.shape[0]), chunk_events)):
        yield hawkes_chunk(t, c, A, Alpha, Beta, mu, n_types, n0, n1, i == 0)


def fit_torch(t_sec, event, n_types, q_np, prior_weight, cfg):
    """The shipped trainer's loop on the vectorised forward; returns the final A (float32 numpy)
    and the fit's facts."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from text_causal import graph_prior  # from the clone, under --method-src

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg["device"] == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("lmt_runner: device=mps requested but MPS is unavailable")
        if cfg["dtype"] != "float32" or cfg["param_dtype"] != "float32":
            raise SystemExit("lmt_runner: device=mps supports float32 only")
    device = torch.device(cfg["device"])
    cdtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32
    pdtype = torch.float64 if cfg["param_dtype"] == "float64" else torch.float32
    K = n_types
    N = int(len(event))
    t = torch.tensor(np.asarray(t_sec, dtype=np.float64), dtype=cdtype, device=device)
    c = torch.tensor(np.asarray(event, dtype=np.int64), dtype=torch.long, device=device)
    q = torch.tensor(q_np, dtype=pdtype, device=device)
    eye = torch.eye(K, device=device, dtype=pdtype)
    off = 1.0 - eye

    # the shipped parametrisation (a_gate_temperature = 1, the default); `param_dtype` is the
    # dtype of the raw parameters, the prior and the penalty (float32 as shipped)
    log_mu = torch.zeros(K, device=device, dtype=pdtype, requires_grad=True)
    A_raw = torch.zeros(K, K, device=device, dtype=pdtype, requires_grad=True)
    log_alpha = torch.zeros(K, K, device=device, dtype=pdtype, requires_grad=True)
    log_beta = torch.zeros(K, K, device=device, dtype=pdtype, requires_grad=True)

    def A_matrix():
        return torch.sigmoid(A_raw) * off

    def alpha_matrix():
        return F.softplus(log_alpha) * off + 1e-8

    def beta_matrix():
        return F.softplus(log_beta) * off + 1e-4

    def mu_vec():
        return F.softplus(log_mu) + 1e-5

    # the shipped init: log μ from N / (K T), then seed-dependent noise in the shipped order
    T_obs = float(t_sec[-1])
    mu_target = N / max(1.0, K * T_obs)
    softplus_target = max(mu_target - 1e-5, 1e-8)
    init_log_mu = float(np.log(np.expm1(softplus_target)))
    with torch.no_grad():
        log_mu.fill_(init_log_mu)
        A_raw.fill_(cfg["a_init"])  # the shipped start is 0 (σ(0) = 0.5 on every pair)
        if cfg["init_std"] > 0:
            A_raw.add_(torch.randn_like(A_raw) * cfg["init_std"])
            log_alpha.add_(torch.randn_like(log_alpha) * cfg["init_std"])
            log_beta.add_(torch.randn_like(log_beta) * cfg["init_std"])
            log_mu.add_(torch.randn_like(log_mu) * cfg["init_std"])
    params = [log_mu, A_raw, log_alpha, log_beta]
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    def likelihood(grad):
        """Σ log λ − compensator over the chunks of the timeline; with `grad` each chunk's −value
        is backpropagated (the K × K transforms are rebuilt per chunk, so every backward walks
        its own graph)."""
        total = 0.0
        for i, (n0, n1) in enumerate(chunk_ranges(N, cfg["chunk_events"])):
            A, Al, Be, mu = A_matrix(), alpha_matrix(), beta_matrix(), mu_vec()
            log_lam, comp = hawkes_chunk(
                t, c, A.to(cdtype), Al.to(cdtype), Be.to(cdtype), mu.to(cdtype), K, n0, n1, i == 0
            )
            ll = log_lam.sum() - comp
            if grad:
                (-ll).backward()
            total += float(ll.item())
        return total

    losses = []
    epoch_seconds = []
    for ep in range(cfg["epochs"]):
        t0 = time.monotonic()
        opt.zero_grad()
        loglike = likelihood(grad=True)
        A = A_matrix()
        sem = graph_prior.semantic_log_bernoulli(A, q)
        h = graph_prior.dag_penalty_h(A)
        ssum = graph_prior.sparsity_sum(A)
        reg = prior_weight * sem - cfg["beta_dag"] * h - cfg["lambda_e"] * ssum
        (-reg).backward()
        opt.step()
        epoch_seconds.append(time.monotonic() - t0)
        losses.append(-(loglike + float(reg.item())))
        if ep % 10 == 0 or ep == cfg["epochs"] - 1:
            print(
                f"epoch {ep:4d} loss {losses[-1]:.6g} loglike {loglike:.6g} h {float(h.item()):.6g} "
                f"{epoch_seconds[-1]:.2f}s",
                flush=True,
            )
    with torch.no_grad():
        final_loglike = likelihood(grad=False)
        A = A_matrix()
        final_h = float(graph_prior.dag_penalty_h(A).item())
        final_sem = float(graph_prior.semantic_log_bernoulli(A, q).item())
        final_ssum = float(graph_prior.sparsity_sum(A).item())
        A_final = A.detach().cpu().numpy().astype(np.float32)
    return A_final, {
        "epochs": int(cfg["epochs"]),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "final_loglike": final_loglike,
        "final_h": final_h,
        "final_log_prior": final_sem,
        "final_sparsity_sum": final_ssum,
        "epoch_seconds_mean": float(sum(epoch_seconds) / len(epoch_seconds)),
        "epoch_seconds_max": float(max(epoch_seconds)),
        "span_seconds": T_obs,
        "init_log_mu": init_log_mu,
        "a_init": cfg["a_init"],
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": cfg["dtype"],
        "param_dtype": cfg["param_dtype"],
        "finite": bool(np.all(np.isfinite(A_final))) and math.isfinite(final_loglike) and math.isfinite(final_h),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method-src", required=True, help="the pinned LMT clone")
    p.add_argument("--events", required=True)
    p.add_argument("--allowed", required=True, help="an (k, 2) int64 .npy of allowed pairs, or 'none'")
    p.add_argument("--setting", required=True, type=int, help="prior_weight_milli (the N slot)")
    p.add_argument("--method-args", required=True, nargs="*", help="k=v over the runner's DEFAULTS")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import numpy as np

    cfg = parse_method_args(args.method_args)
    sys.path.insert(0, args.method_src + "/chemical_1000_event_repro")
    from text_causal import graph_prior  # noqa: F401  (the import is the contract; stub under test)

    with np.load(args.events) as z:
        event = z["event"].astype(np.int64)
        timestamp = z["timestamp"].astype(np.int64)
    if len(event) < 2:
        raise SystemExit("lmt_runner: need at least two events")
    n_types = int(event.max()) + 1
    if n_types < 2:
        raise SystemExit("lmt_runner: need at least two event types")
    if np.any(np.diff(timestamp) < 0):
        raise SystemExit("lmt_runner: timestamps must be sorted")
    allowed = None
    if args.allowed != "none":
        allowed = np.load(args.allowed).astype(np.int64).reshape(-1, 2)
    t_sec = timestamp.astype(np.float64) / 1000.0  # D-LM-2: milliseconds → seconds
    prior_weight = float(args.setting) / 1000.0
    q = q_matrix(n_types, allowed, cfg["q_in"], cfg["q_out"], cfg["q_default"])

    t0 = time.monotonic()
    if cfg["fit"] == "stub":
        scores, facts = stub_scores(n_types, allowed), {"epochs": 0}
    else:
        scores, facts = fit_torch(t_sec, event, n_types, q, prior_weight, cfg)
    wall = time.monotonic() - t0
    scores = np.asarray(scores, dtype=np.float32)
    np.fill_diagonal(scores, 0.0)
    if scores.shape != (n_types, n_types) or not np.all(np.isfinite(scores)):
        raise SystemExit("lmt_runner: the score matrix is malformed")
    scores = np.clip(scores, 0.0, 1.0)
    np.save(args.out, scores)
    offdiag = ~np.eye(n_types, dtype=bool)
    above = (scores > 0.5) & offdiag
    outside = np.ones_like(offdiag) & offdiag
    if allowed is not None and len(allowed):
        outside[allowed[:, 0], allowed[:, 1]] = False
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_types": n_types,
                "n_events": int(len(event)),
                "n_allowed_pairs": None if allowed is None else int(len(allowed)),
                "n_nonzero": int(np.count_nonzero(scores[offdiag] > 0)),
                "n_above_half": int(above.sum()),
                "n_above_half_outside_mask": None if allowed is None else int((above & outside).sum()),
                "wall_seconds": wall,
                "setting": int(args.setting),
                "prior_weight": prior_weight,
                "method_args": {k: cfg[k] for k in sorted(cfg)},
                "method_version": method_version(args.method_src),
                "method_env": pip_freeze(),
                "time_unit": "seconds (integer milliseconds / 1000; D-LM-2)",
                **facts,
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
