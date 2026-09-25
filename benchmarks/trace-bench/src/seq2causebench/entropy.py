# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""An independent entropy floor for the oracle score (D-SB-13).

Eq. 22 divides the excess loss by `log|X| − H(P)`; its prose takes `H` as
the minimum validation loss, which makes the score vanish on any converging
run. Here `H` is the plug-in order-`k` conditional entropy of the training
split — `H(X_t | X_{t-k..t-1})` in nats over every predicted position
(events and EOS), contexts shorter than `k` at the start of a sequence
standing as their own contexts — an estimate that does not depend on the
model. On the gate's synthetic worlds the generator's exact conditional
entropy is used instead (see `generator.py`).

Vectorised: contexts are packed into int64 keys (`k <= 3`, vocab < 2^20).
"""

from __future__ import annotations

import math

import numpy as np

MAX_ORDER = 3


def order_k_floor(store, k, vocab_size):
    """`{entropy_floor, order, n_positions, n_contexts, log_vocab}` on a `SequenceStore`."""
    if not 0 <= k <= MAX_ORDER:
        raise ValueError(f"order must be in [0, {MAX_ORDER}], got {k}")
    base = int(vocab_size) + 1  # 0 = no token (short context filler)
    if base ** max(k, 1) * base >= 2**62:
        raise ValueError("vocabulary too large to pack the context into int64")
    T = store.tokens.astype(np.int64)
    offsets = store.offsets
    n_seq = len(store)
    if n_seq == 0 or len(T) == 0:
        return {
            "entropy_floor": None,
            "order": k,
            "n_positions": 0,
            "n_contexts": 0,
            "log_vocab": math.log(vocab_size),
        }
    lengths = np.diff(offsets)
    pos = np.arange(len(T)) - np.repeat(offsets[:-1], lengths)  # position within its sequence
    target = pos >= 1  # every predicted position
    ctx = np.zeros(len(T), dtype=np.int64)
    for j in range(1, k + 1):
        valid = pos >= j
        prev = np.zeros(len(T), dtype=np.int64)
        prev[j:] = T[:-j] + 1
        ctx = ctx * base + np.where(valid, prev, 0)
    ctx_t = ctx[target]
    nxt = T[target]
    pair = ctx_t * base + nxt
    u_pair, c_pair = np.unique(pair, return_counts=True)
    u_ctx, c_ctx = np.unique(ctx_t, return_counts=True)
    ctx_of_pair = u_pair // base
    ctx_count = c_ctx[np.searchsorted(u_ctx, ctx_of_pair)]
    n = float(c_pair.sum())
    h = float(-(c_pair / n * np.log(c_pair / ctx_count)).sum())
    return {
        "entropy_floor": h,
        "order": k,
        "n_positions": int(n),
        "n_contexts": int(len(u_ctx)),
        "log_vocab": math.log(vocab_size),
    }


def oracle_score(val_loss, entropy_floor, log_alphabet, in_regime_below):
    """Eq. 22 with an independent floor: `(L − H) / (log|X| − H)`."""
    denom = log_alphabet - entropy_floor
    eps = (val_loss - entropy_floor) / denom if denom > 0 else None
    return {
        "eps_hat": eps,
        "val_loss": val_loss,
        "entropy_floor": entropy_floor,
        "log_alphabet": log_alphabet,
        "in_regime": (eps is not None and eps < in_regime_below),
    }
