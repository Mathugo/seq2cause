"""The engine adapter: the shipped seq2cause functions, called in the shipped order
from one forward pass per (probe, noise) — D-SB-2.

This is the only harness module that imports `seq2cause`. It drives the two
shipped estimators and the three shipped read-outs exactly as the package's
own entry points do (`core.SampleLevelCausalDiscovery.run` and
`diagnostics.compute_cmi_matrix`), but keeps the forward pass's logits so
that every read-out of one pass — the shipped KL, the fixed KL and the
Granger score for the core probe; the shipped and fixed KL for the two cli
probes — comes from the same tensor and the same noise draws. Bit-equal
parity against both entry points is asserted on the fixture
(`tests/test_engine_parity.py`).

Probes (forward-sharing units):
- `core` — `ancestral_sampling` → `uniform_sample` → `do_interventions(full)` →
  one forward over `N · rows` rows → `calc_lag_info_gain` / `calc_granger_score`.
- `cli-full` / `cli-atomic` — `uniform_sample` → `do_interventions` → one
  forward (plus the fully-real baseline row for `atomic`) →
  `_predicted_true_token_(log_)probs`'s gather → `_cmi_matrix_from_*`.
- `saliency` / `shapley` — `calc_neural_saliency` / `calc_neural_shapley`
  on the raw library model (no particles).

Noise: `all` draws over `[0, V)` (v0.1.9), `real` over `[N_SPECIALS, V)`
(fix PR #3). KL: `clamp` (v0.1.9) or `logspace` (fix PR #1), both read from
the same probabilities / log-probabilities of one forward. The corrupted-cell
counts are taken on every path (PRD scenarios 10, 41).

The shipped samplers use the global torch RNG; `seed_sequence` seeds it per
sequence from the run seed, the split and the sequence index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from seq2cause.adapters import HFModelAdapter
from seq2cause.causal_strength import (
    calc_granger_score,
    calc_lag_info_gain,
    calc_neural_saliency,
    calc_neural_shapley,
)
from seq2cause.diagnostics import (
    _cmi_matrix_from_atomic,
    _cmi_matrix_from_atomic_log,
    _cmi_matrix_from_staircase,
    _cmi_matrix_from_staircase_log,
    _count_atomic,
    _count_staircase,
    summary_graph,
)
from seq2cause.kl import accumulate_counts, corrupted_cell_counts, logmeanexp
from seq2cause.sampling import ancestral_sampling, do_interventions, uniform_sample
from seq2cause.threshold import AdaptiveThreshold
from seq2cause.utils import estimate_tensor_bytes

from .arms import column_of
from .backbone import load_model, model_sha256, n_params
from .constants import (
    ARMS,
    KL_FIXED,
    KL_SHIPPED,
    N_SPECIALS,
    NOISE_ALL,
    NOISES,
    PARTICLE_PROBES,
    PATH_SPEC,
    PROBE_CLI_ATOMIC,
    PROBE_CLI_FULL,
    PROBE_CORE,
    PROBE_SALIENCY,
    PROBE_SHAPLEY,
    PROBES,
)

SPLIT_STREAM = {"train": 0, "val": 1, "test": 2}
SHIPPED_MAX_PAIRS = 20000  # compute_cmi_matrix's default O(L^2) guard; never binds at L <= 64


class MemoryRefusal(RuntimeError):
    pass


# --- what one pass must compute ------------------------------------------------------------------
def probe_columns(probe, noise):
    """`{arm: {path: column}}` for every (arm, path) whose forward is this (probe, noise) pass."""
    if probe not in PROBES:
        raise ValueError(f"probe must be one of {PROBES}, got {probe!r}")
    if noise not in NOISES:
        raise ValueError(f"noise must be one of {NOISES}, got {noise!r}")
    out = {}
    for arm, spec in ARMS.items():
        if spec["probe"] != probe:
            continue
        for path in spec["paths"]:
            want = PATH_SPEC[path]["noise"]
            if want == noise or (want is None and noise == NOISE_ALL):
                out.setdefault(arm, {})[path] = column_of(arm, path)
    return out


def noise_min_id(noise):
    return 0 if noise == NOISE_ALL else N_SPECIALS


def seed_sequence(seed, split, s):
    """Seed the global torch RNG (the shipped samplers draw from it) per sequence."""
    if split not in SPLIT_STREAM:
        raise ValueError(f"split must be one of {sorted(SPLIT_STREAM)}, got {split!r}")
    torch.manual_seed((int(seed) * 1000003 + SPLIT_STREAM[split]) * 1000003 + int(s))


# --- the backbone as the shipped code sees it ----------------------------------------------------
def load_backbone(model_dir, device):
    """`(hf_model, adapter, model_sha256)`: the raw library model (core probe, saliency,
    Shapley) and the shipped adapter around it (cli probes)."""
    hf_model, _ = load_model(model_dir, device)
    adapter = HFModelAdapter(hf_model, vocab_size=hf_model.config.vocab_size)
    return hf_model, adapter, model_sha256(model_dir)


def _autocast(device, amp):
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(amp == "bf16"))


# --- the core probe: SampleLevelCausalDiscovery.run()'s tensor build, verbatim order ---------------
@torch.inference_mode()
def core_forward(hf_model, ids, c, g, N, noise, amp="none"):
    """`(batch, logits [1, N, rows, L, V] float32)` for one sequence `ids [L]`."""
    device = ids.device
    L = int(ids.shape[0])
    batch = {
        "input_ids": ids.unsqueeze(0),
        "attention_mask": torch.ones(1, L, dtype=torch.long, device=device),
    }
    with _autocast(device.type, amp):
        o_b = hf_model(attention_mask=batch["attention_mask"], input_ids=batch["input_ids"])[
            "logits"
        ].float()
    prob_x = torch.nn.functional.softmax(o_b, dim=-1)
    expanded_attention_mask = batch["attention_mask"].unsqueeze(1).repeat(1, N, 1)
    with _autocast(device.type, amp):
        prefix_upsampled = ancestral_sampling(hf_model, batch, value=N, guidance=g, context=c)
    prefix_upsampled = prefix_upsampled.reshape(1, N, -1)
    rest = batch["input_ids"][:, c:]
    rest_upsampled_using_q = uniform_sample(
        prob_x[:, c:, :], n_samples=N, cls_token_id=None, min_id=noise_min_id(noise)
    )
    rest_expanded_intervened = do_interventions(
        rest_upsampled_using_q, rest, prefix_upsampled, strategy="full", prepend_context_back=False
    )
    interv_dim = rest_expanded_intervened.shape[-2]
    prefix_upsampled_expanded = prefix_upsampled.unsqueeze(-2).repeat(1, 1, interv_dim, 1)
    rows = torch.cat(
        [prefix_upsampled_expanded, rest_expanded_intervened], dim=-1
    )  # [1, N, rows, L]
    mask = expanded_attention_mask.unsqueeze(-2).repeat(1, 1, interv_dim, 1)
    with _autocast(device.type, amp):
        logits = hf_model(attention_mask=mask.reshape(-1, L), input_ids=rows.reshape(-1, L))[
            "logits"
        ]
    logits = logits.float().reshape(1, N, interv_dim, L, -1)
    return batch, logits


def core_readouts(logits, batch, c, columns):
    """`columns`: `{arm: {path: column}}` of the core probe. Returns `({column: [Lc, Lc] np.float32},
    {path: counts})`; the shipped KL, the fixed KL and the Granger score are read from the same
    logits, each through the shipped function."""
    mats, counts = {}, {}
    core = columns.get("trace/core", {})
    granger = columns.get("baseline/granger", {})
    need_probs = any(PATH_SPEC[p]["kl"] == KL_SHIPPED for p in core) or bool(granger)
    if need_probs:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        for path, col in core.items():
            if PATH_SPEC[path]["kl"] == KL_SHIPPED:
                stats = {}
                adj = calc_lag_info_gain(
                    probs,
                    batch,
                    {"sampling": {"context": c}, "kl_mode": KL_SHIPPED, "kl_stats": stats},
                )
                mats[col] = adj[0].cpu().numpy().astype(np.float32)
                counts[path] = stats
        for col in granger.values():
            adj = calc_granger_score(probs, batch, {"sampling": {"context": c}})
            mats[col] = adj[0].cpu().numpy().astype(np.float32)
        del probs
    if any(PATH_SPEC[p]["kl"] == KL_FIXED for p in core):
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        for path, col in core.items():
            if PATH_SPEC[path]["kl"] == KL_FIXED:
                stats = {}
                adj = calc_lag_info_gain(
                    logp,
                    batch,
                    {"sampling": {"context": c}, "kl_mode": KL_FIXED, "kl_stats": stats},
                )
                mats[col] = adj[0].cpu().numpy().astype(np.float32)
                counts[path] = stats
        del logp
    return mats, counts


# --- the cli probes: compute_cmi_matrix's construction, one forward for both KL modes ----------------
def _gather_true_token(x, true_tokens, context_len):
    """`_predicted_true_token_(log_)probs`'s gather on an already-normalised `[..., L, V]`."""
    seq_len = x.shape[-2]
    pred = x[..., context_len - 1 : seq_len - 1, :]
    lc = pred.shape[-2]
    idx = true_tokens.view(*([1] * (pred.dim() - 2)), lc, 1).expand(*pred.shape[:-1], 1)
    return torch.gather(pred, dim=-1, index=idx).squeeze(-1)


@torch.no_grad()
def cli_forward(adapter, sequence, c, N, strategy, noise, amp="none", max_pairs=SHIPPED_MAX_PAIRS):
    """`(rows_logits [N, Lc, L, V], baseline_logits [1, L, V] | None, rest [Lc])` for one sequence."""
    if strategy not in ("full", "atomic"):
        raise ValueError(f"strategy must be 'full' or 'atomic', got {strategy!r}")
    device = sequence.device
    seq_len = sequence.shape[-1]
    lc = seq_len - c
    prefix = sequence[:c].unsqueeze(0)
    rest = sequence[c:].unsqueeze(0)
    dummy = torch.zeros(1, lc, adapter.vocab_size, device=device)
    noise_t = uniform_sample(dummy, n_samples=N, device=device, min_id=noise_min_id(noise))
    if strategy == "full":
        rows = do_interventions(noise_t, rest, prefix, strategy="full").squeeze(0)
    else:
        rows = do_interventions(
            noise_t, rest, prefix, strategy="atomic", max_pairs=max_pairs
        ).squeeze(0)
    with _autocast(device.type, amp):
        rows_logits = adapter.forward(input_ids=rows)["logits"].float()
    baseline_logits = None
    if strategy == "atomic":
        with _autocast(device.type, amp):
            baseline_logits = adapter.forward(input_ids=sequence.unsqueeze(0))["logits"].float()
    return rows_logits, baseline_logits, rest.squeeze(0)


def cli_readouts(rows_logits, baseline_logits, rest, c, strategy, columns):
    """`({column: [Lc, Lc] np.float32}, {path: counts})` for a cli probe's arm."""
    arm = "trace/cli" if strategy == "full" else "trace/cli-atomic"
    paths = columns.get(arm, {})
    mats, counts = {}, {}
    lc = rest.shape[-1]
    for path, col in paths.items():
        stats = {}
        if PATH_SPEC[path]["kl"] == KL_SHIPPED:
            p = _gather_true_token(torch.softmax(rows_logits, dim=-1), rest, c)
            p_mean = p.mean(dim=0)
            if strategy == "full":
                _count_staircase(stats, p_mean[1:, :], p_mean[:-1, :])
                m = _cmi_matrix_from_staircase(p_mean)
            else:
                p_base = _gather_true_token(
                    torch.softmax(baseline_logits, dim=-1), rest, c
                ).squeeze(0)
                _count_atomic(stats, p_base.unsqueeze(0).expand(lc, lc), p_mean)
                m = _cmi_matrix_from_atomic(p_mean, p_base)
        else:
            logp = _gather_true_token(torch.log_softmax(rows_logits, dim=-1), rest, c)
            logp_mean = logmeanexp(logp, dim=0)
            if strategy == "full":
                _count_staircase(stats, logp_mean[1:, :].exp(), logp_mean[:-1, :].exp())
                m = _cmi_matrix_from_staircase_log(logp_mean)
            else:
                logp_base = _gather_true_token(
                    torch.log_softmax(baseline_logits, dim=-1), rest, c
                ).squeeze(0)
                _count_atomic(stats, logp_base.exp().unsqueeze(0).expand(lc, lc), logp_mean.exp())
                m = _cmi_matrix_from_atomic_log(logp_mean, logp_base)
        mats[col] = m.cpu().numpy().astype(np.float32)
        counts[path] = stats
    return mats, counts


# --- the read-out baselines (no particles) -----------------------------------------------------------
def saliency_matrix(hf_model, ids, c):
    batch = {"input_ids": ids.unsqueeze(0)}
    _, adj = calc_neural_saliency(hf_model, batch, {"sampling": {"context": c}})
    return adj[0].detach().cpu().numpy().astype(np.float32)


def shapley_matrix(hf_model, ids, c):
    batch = {"input_ids": ids.unsqueeze(0)}
    _, adj = calc_neural_shapley(hf_model, batch, {"sampling": {"context": c}})
    return adj[0].detach().cpu().numpy().astype(np.float32)


# --- one sequence, one pass -------------------------------------------------------------------------
@dataclass
class ProbeResult:
    matrices: dict  # column -> [Lc, Lc] float32, the full matrix as the shipped code returns it
    counts: dict  # path -> corrupted-cell counts (particle probes)
    tok_pos: int  # token positions forwarded


def probe_sequence(probe, noise, hf_model, adapter, ids, c, g, N, columns, amp="none"):
    """Run one (probe, noise) pass on one sequence `ids [L]` (long, on the model's device) and
    read out every column of `columns` (`probe_columns(probe, noise)`)."""
    L = int(ids.shape[0])
    lc = L - c
    if probe == PROBE_CORE:
        batch, logits = core_forward(hf_model, ids, c, g, N, noise, amp)
        mats, counts = core_readouts(logits, batch, c, columns)
        tok_pos = int(logits.shape[1] * logits.shape[2] * L) + L
        del logits
        return ProbeResult(mats, counts, tok_pos)
    if probe in (PROBE_CLI_FULL, PROBE_CLI_ATOMIC):
        strategy = "full" if probe == PROBE_CLI_FULL else "atomic"
        rows_logits, baseline_logits, rest = cli_forward(adapter, ids, c, N, strategy, noise, amp)
        mats, counts = cli_readouts(rows_logits, baseline_logits, rest, c, strategy, columns)
        tok_pos = int(rows_logits.shape[0] * rows_logits.shape[1] * L) + (
            L if baseline_logits is not None else 0
        )
        del rows_logits, baseline_logits
        return ProbeResult(mats, counts, tok_pos)
    if probe == PROBE_SALIENCY:
        col = columns["baseline/saliency"]["none"]
        return ProbeResult({col: saliency_matrix(hf_model, ids, c)}, {}, lc * L)
    if probe == PROBE_SHAPLEY:
        col = columns["baseline/shapley"]["none"]
        return ProbeResult({col: shapley_matrix(hf_model, ids, c)}, {}, 10 * L * lc)
    raise ValueError(f"unknown probe {probe!r}")


def merge_counts(total, counts):
    """Accumulate `{path: counts}` into `total` in place."""
    for path, stats in counts.items():
        accumulate_counts(total.setdefault(path, {}), stats)
    return total


# --- the strict-upper triangle stored per sequence ---------------------------------------------------
def triangle(matrix):
    """`(j, q, values)` of the strict upper triangle (every lag > 0 cell, row 0 and within-operation
    cells included — what `cli.py` pools for its threshold)."""
    lc = matrix.shape[-1]
    j, q = np.triu_indices(lc, k=1)
    return j.astype(np.int16), q.astype(np.int16), np.asarray(matrix)[j, q].astype(np.float32)


def from_triangle(lc, j, q, values):
    """The `[Lc, Lc]` matrix with the stored strict-upper cells and zeros elsewhere (the cells
    `apply_tau_by_lag` and the pooled percentile read are exactly the stored ones)."""
    m = np.zeros((lc, lc), dtype=np.float32)
    m[j.astype(np.int64), q.astype(np.int64)] = values
    return m


# --- the shipped cut: cli.py's pooled threshold and union, verbatim ----------------------------------
def shipped_cut(matrices, rule):
    """`(tau_by_lag, [bool graphs], threshold_finite)`: `AdaptiveThreshold(**rule).tau_by_lag` fitted
    once on every lag > 0 cell of every matrix, then `apply_tau_by_lag` per matrix — the CLI's
    second pass. A NaN or +inf cell makes the pooled percentile NaN and every graph empty."""
    threshold = AdaptiveThreshold(**rule)
    pooled_scores, pooled_lags = [], []
    mats = [torch.as_tensor(np.asarray(m), dtype=torch.float32) for m in matrices]
    for cmi_matrix in mats:
        lc = cmi_matrix.shape[-1]
        lag_matrix = torch.tensor([[q - j for q in range(lc)] for j in range(lc)])
        valid = lag_matrix > 0
        pooled_scores.append(cmi_matrix[valid])
        pooled_lags.append(lag_matrix[valid])
    pooled_scores = torch.cat(pooled_scores)
    pooled_lags = torch.cat(pooled_lags)
    tau_by_lag = threshold.tau_by_lag(pooled_scores, pooled_lags, max_lag=int(pooled_lags.max()))
    graphs = [threshold.apply_tau_by_lag(m, tau_by_lag) for m in mats]
    finite = all(math.isfinite(float(v)) for v in tau_by_lag.values())
    return {int(k): float(v) for k, v in tau_by_lag.items()}, graphs, finite


def shipped_cut_union(sequences, graphs, c, vocab):
    """The tool's summary graphs unioned over sequences, as type edges `(u, v)` over token ids.
    Edges touching a special id are dropped (they have no scorer node name) and counted; within-
    operation edges are kept and counted (the scorer counts them as outside its universe)."""
    edges = set()
    n_special = 0
    n_within_op = 0
    for ids, graph in zip(sequences, graphs, strict=True):
        seq_t = torch.as_tensor(np.asarray(ids), dtype=torch.long)
        active, adj = summary_graph(
            seq_t, torch.as_tensor(graph, dtype=torch.bool), context_len=c, self_loops=False
        )
        active = active.tolist()
        for a, b in torch.nonzero(adj).tolist():
            u, v = int(active[a]), int(active[b])
            if not (vocab.is_real(u) and vocab.is_real(v)):
                n_special += 1
                continue
            if vocab.op_of(u) == vocab.op_of(v):
                n_within_op += 1
            edges.add((u, v))
    return edges, {
        "n_edges": len(edges),
        "n_dropped_special": n_special,
        "n_within_op": n_within_op,
    }


# --- the memory estimate (D-SB-12, PRD scenario 18) ---------------------------------------------------
def estimate_peak_bytes(probe, N, L, c, V, n_layers, d_model, params):
    """`{shipped_estimate, harness_estimate}` in bytes for the largest sequence of a pass
    (`plans/caps.md`)."""
    lc = max(L - c, 1)
    if probe in PARTICLE_PROBES:
        shipped = estimate_tensor_bytes(N, lc, L, V)
        rows = N * lc
    else:
        rows = 4 if probe == PROBE_SHAPLEY else 1
        shipped = estimate_tensor_bytes(rows, L, V)
    activations = rows * L * d_model * n_layers * 4 * 4
    return {
        "shipped_estimate": int(shipped),
        "harness_estimate": int(2 * shipped + activations + params * 4),
    }


def check_memory(hf_model, probe, N, max_L, c, cap_gb):
    cfg = hf_model.config
    est = estimate_peak_bytes(
        probe,
        N,
        max_L,
        c,
        cfg.vocab_size,
        cfg.num_hidden_layers,
        cfg.hidden_size,
        n_params(hf_model),
    )
    cap = int(float(cap_gb) * 2**30)
    est.update(
        {
            "cap_bytes": cap,
            "cap_gb": float(cap_gb),
            "max_seq_len": int(max_L),
            "N": int(N),
            "c": int(c),
            "probe": probe,
        }
    )
    if est["harness_estimate"] > cap:
        raise MemoryRefusal(
            f"estimated peak {est['harness_estimate'] / 2**30:.2f} GiB for probe {probe!r} at N={N}, L={max_L}, c={c} "
            f"exceeds --memory-cap-gb {cap_gb} (declared in plans/caps.md); reduce N or --max-len, or raise the cap "
            "by a dated addendum (PRD scenario 18)"
        )
    return est


__all__ = [
    "SPLIT_STREAM",
    "MemoryRefusal",
    "ProbeResult",
    "probe_columns",
    "noise_min_id",
    "seed_sequence",
    "load_backbone",
    "core_forward",
    "core_readouts",
    "cli_forward",
    "cli_readouts",
    "saliency_matrix",
    "shapley_matrix",
    "probe_sequence",
    "merge_counts",
    "triangle",
    "from_triangle",
    "shipped_cut",
    "shipped_cut_union",
    "estimate_peak_bytes",
    "check_memory",
    "corrupted_cell_counts",
]
