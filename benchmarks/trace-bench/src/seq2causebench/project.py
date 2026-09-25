# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Projection from positions to event types (D-SB-10).

Per sequence a probe leaves position matrices `S[Lc, Lc]` over the window,
one per column (arm on a path); the type of window position `t` is the
observed token `ids[c + t]`. Two projections:

- **the shipped union** (`diagnostics.summary_graph`, the CLI's per-sequence
  graph): a type edge `u → v` is present iff some position pair `(j, q)` with
  `S[j, q]` above the threshold carries it;
- **corpus level** (the benchmark's summary graph over the sampled
  sequences): for every ordered within-sequence pair occurrence with lag
  `<= M` accumulate per `(u, v)` the `max`, `sum`, `count` and per-lag `max`;
  the type-level score is the `max` (mean and count are recorded columns).

Self-loops and within-operation pairs are dropped (the scorer never scores
them), as are cells whose effect or cause is not a real event token (EOS,
PAD). The accumulator keeps a flat `(key, lag, score)` table per column and
reduces by sorting, so it never allocates `|X|²` arrays.
"""

from __future__ import annotations

import numpy as np

NEG = -np.inf


def sequence_cells(stats, window_tokens, vocab, max_lag):
    """`(j, q, lag, u, v, {column: value})` arrays for the real, cross-operation cells of one sequence."""
    lc = window_tokens.shape[0]
    j, q = np.triu_indices(lc, k=1)
    lag = q - j
    keep = lag <= max_lag
    j, q, lag = j[keep], q[keep], lag[keep]
    u, v = window_tokens[j], window_tokens[q]
    real = np.array([vocab.is_real(int(t)) for t in window_tokens], dtype=bool)
    ops = np.array(
        [vocab.op_of(int(t)) if vocab.is_real(int(t)) else -1 for t in window_tokens],
        dtype=np.int64,
    )
    keep = real[j] & real[q] & (ops[j] != ops[q])
    j, q, lag, u, v = j[keep], q[keep], lag[keep], u[keep], v[keep]
    values = {k: np.asarray(stats[k])[j, q] for k in stats}
    return j, q, lag, u, v, values


def sequence_type_edges(stat_matrix, window_tokens, vocab, tau, max_lag):
    """The union at threshold τ: the set of `(u, v)` type pairs (token ids) with any `S > τ`."""
    j, q, lag, u, v, values = sequence_cells({"s": stat_matrix}, window_tokens, vocab, max_lag)
    s = values["s"]
    hit = s > tau
    return set(zip(u[hit].tolist(), v[hit].tolist(), strict=True))


class PairAccumulator:
    """Corpus-level max / sum / count and per-lag max per `(u, v)` for every column."""

    def __init__(self, vocab_size, max_lag, columns, flush_every=2000):
        self.V = int(vocab_size)
        self.M = int(max_lag)
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("PairAccumulator needs at least one column")
        self.flush_every = flush_every
        self._keys, self._lags = [], []
        self._vals = {k: [] for k in self.columns}
        self._partial = None  # reduced table: dict of arrays
        self._pending = 0
        self.n_sequences = 0
        self.n_cells = 0

    def add(self, u, v, lag, values):
        key = u.astype(np.int64) * self.V + v.astype(np.int64)
        self._keys.append(key)
        self._lags.append(lag.astype(np.int64))
        for k in self.columns:
            self._vals[k].append(np.asarray(values[k], dtype=np.float64))
        self.n_sequences += 1
        self.n_cells += len(key)
        self._pending += 1
        if self._pending >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self._keys:
            return
        key = np.concatenate(self._keys)
        lag = np.concatenate(self._lags)
        vals = {k: np.concatenate(self._vals[k]) for k in self.columns}
        count = np.ones(len(key), dtype=np.int64)
        table = _reduce(
            key, lag, vals, dict(vals), count, dict.fromkeys(self.columns), self.M, self.columns
        )
        self._partial = (
            table if self._partial is None else _merge(self._partial, table, self.M, self.columns)
        )
        self._keys, self._lags = [], []
        self._vals = {k: [] for k in self.columns}
        self._pending = 0

    def result(self):
        """`{src, dst, count, max_<col>, mean_<col>, lag_max_<col> [n, M]}` sorted by key."""
        self._flush()
        t = self._partial
        if t is None:
            empty = np.zeros(0)
            out = {
                "src": np.zeros(0, dtype=np.int64),
                "dst": np.zeros(0, dtype=np.int64),
                "count": np.zeros(0, dtype=np.int64),
            }
            for k in self.columns:
                out[f"max_{k}"] = empty
                out[f"mean_{k}"] = empty
                out[f"lag_max_{k}"] = np.zeros((0, self.M))
            return out
        out = {"src": t["key"] // self.V, "dst": t["key"] % self.V, "count": t["count"]}
        for k in self.columns:
            out[f"max_{k}"] = t[f"max_{k}"]
            out[f"mean_{k}"] = t[f"sum_{k}"] / t["count"]
            out[f"lag_max_{k}"] = t[f"lag_max_{k}"]
        return out


def _reduce(key, lag, max_vals, sum_vals, count, lag_max_in, M, columns):
    """Group by key: max, sum, count per column and per-lag max (−inf where absent)."""
    order = np.argsort(key, kind="stable")
    key, lag, count = key[order], lag[order], count[order]
    ukey, start = np.unique(key, return_index=True)
    table = {"key": ukey, "count": np.add.reduceat(count, start)}
    for k in columns:
        mv = max_vals[k][order]
        sv = sum_vals[k][order]
        table[f"max_{k}"] = np.maximum.reduceat(mv, start)
        table[f"sum_{k}"] = np.add.reduceat(sv, start)
        if lag_max_in[k] is None:
            lm = np.full((len(ukey), M), NEG)
            klag = key * (M + 1) + lag
            o2 = np.argsort(klag, kind="stable")
            uk2, s2 = np.unique(klag[o2], return_index=True)
            m2 = np.maximum.reduceat(mv[o2], s2)
            rows = np.searchsorted(ukey, uk2 // (M + 1))
            cols = uk2 % (M + 1) - 1
            lm[rows, cols] = m2
        else:
            lmi = lag_max_in[k][order]
            lm = np.maximum.reduceat(lmi, start, axis=0)
        table[f"lag_max_{k}"] = lm
    return table


def _merge(a, b, M, columns):
    key = np.concatenate([a["key"], b["key"]])
    lag = np.zeros(len(key), dtype=np.int64)
    count = np.concatenate([a["count"], b["count"]])
    return _reduce(
        key,
        lag,
        {k: np.concatenate([a[f"max_{k}"], b[f"max_{k}"]]) for k in columns},
        {k: np.concatenate([a[f"sum_{k}"], b[f"sum_{k}"]]) for k in columns},
        count,
        {k: np.concatenate([a[f"lag_max_{k}"], b[f"lag_max_{k}"]]) for k in columns},
        M,
        columns,
    )


def write_scores_npz(path, scores, **meta):
    np.savez(path, **scores, **{k: np.asarray(v) for k, v in meta.items()})


def read_scores_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def columns_of(scores):
    """The column names a scores table carries (from its `max_<col>` keys)."""
    return tuple(sorted(k[len("max_") :] for k in scores if k.startswith("max_")))


__all__ = [
    "sequence_cells",
    "sequence_type_edges",
    "PairAccumulator",
    "write_scores_npz",
    "read_scores_npz",
    "columns_of",
    "NEG",
]
