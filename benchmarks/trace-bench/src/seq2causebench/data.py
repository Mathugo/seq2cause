# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Tokenisation, sequence storage, sampling and batching (D-SB-7, D-SB-14).

A sequence is `BOS + ids + EOS`, truncated to `--max-len` real tokens (the
tail is dropped and counted). A `SequenceStore` holds one split as a flat
int32 token array with offsets. Batches are length-bucketed and padded to the
batch maximum with PAD, which the model masks from attention and the loss.
`select_sequences` takes the first `n` sequences in file order (`head`) or a
seeded reservoir sample (`uniform`); `n = 0` means the whole split.
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import BOS, EOS, PAD, SEQUENCE_SAMPLES


def encode_sequence(vocab, ops, outcomes, max_len):
    """`(ids, truncated, folded)`: BOS + at most `max_len` real tokens + EOS."""
    ids, folded = vocab.encode(ops, outcomes)
    truncated = len(ids) > max_len
    if truncated:
        ids = ids[:max_len]
    return [BOS, *ids, EOS], truncated, folded


def select_sequences(seqs, n, mode, rng):
    """`head`: the first `n` in file order; `uniform`: reservoir sample of size `n`
    (Algorithm R, seeded); `n = 0`: everything. Returns a list."""
    if mode not in SEQUENCE_SAMPLES:
        raise ValueError(f"mode must be one of {SEQUENCE_SAMPLES}, got {mode!r}")
    if n < 0:
        raise ValueError("n must be >= 0 (0 = the whole split)")
    if n == 0:
        return list(seqs)
    if mode == "head":
        out = []
        for s in seqs:
            out.append(s)
            if len(out) >= n:
                break
        return out
    out = []
    for i, s in enumerate(seqs):
        if i < n:
            out.append(s)
        else:
            j = int(rng.integers(0, i + 1))
            if j < n:
                out[j] = s
    return out


class SequenceStore:
    """One split, encoded: `tokens` (int32, flat), `offsets` (int64, len n+1), `trace_ids`."""

    def __init__(self, tokens, offsets, trace_ids, n_truncated, n_folded, n_spans):
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.trace_ids = list(trace_ids)
        self.n_truncated = int(n_truncated)
        self.n_folded = int(n_folded)
        self.n_spans = np.asarray(n_spans, dtype=np.int32)

    @classmethod
    def from_sequences(cls, seqs, vocab, max_len):
        tokens, offsets, tids, spans = [], [0], [], []
        n_trunc = n_fold = 0
        for tid, ops, outcomes, n_spans in seqs:
            ids, truncated, folded = encode_sequence(vocab, ops, outcomes, max_len)
            tokens.extend(ids)
            offsets.append(len(tokens))
            tids.append(tid)
            spans.append(n_spans)
            n_trunc += truncated
            n_fold += folded
        return cls(tokens, offsets, tids, n_trunc, n_fold, spans)

    @classmethod
    def from_corpus(cls, corpus, split, vocab, max_len, n=0, mode="head", seed=0):
        rng = np.random.default_rng(seed)
        seqs = select_sequences(corpus.sequences(split), n, mode, rng)
        return cls.from_sequences(seqs, vocab, max_len)

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def lengths(self):
        return np.diff(self.offsets)

    @property
    def n_tokens(self):
        return int(self.offsets[-1])

    def get(self, i):
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]

    def batches(self, batch_size, rng=None, pool_batches=50):
        """Yield lists of sequence indices. With an rng: shuffled, then length-sorted
        within pools of `pool_batches` batches, batch order shuffled. Without: length-
        sorted over the whole store (deterministic, minimal padding)."""
        n = len(self)
        lengths = self.lengths
        if rng is None:
            order = np.argsort(lengths, kind="stable")
            for i in range(0, n, batch_size):
                yield order[i : i + batch_size].tolist()
            return
        perm = rng.permutation(n)
        pool = batch_size * pool_batches
        batches = []
        for i in range(0, n, pool):
            chunk = perm[i : i + pool]
            chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
            for j in range(0, len(chunk), batch_size):
                batches.append(chunk[j : j + batch_size].tolist())
        for k in rng.permutation(len(batches)):
            yield batches[k]

    def collate(self, indices, device="cpu"):
        return collate([self.get(i) for i in indices], device)


def collate(arrays, device="cpu"):
    """`(ids [B, L] int64, pad_mask [B, L] bool)`, padded with PAD to the batch maximum."""
    L = max(len(a) for a in arrays)
    ids = np.full((len(arrays), L), PAD, dtype=np.int64)
    for i, a in enumerate(arrays):
        ids[i, : len(a)] = a
    ids_t = torch.from_numpy(ids).to(device)
    return ids_t, ids_t.eq(PAD)


def endless_batches(store, batch_size, seed):
    """Training stream: epochs of shuffled, length-bucketed batches, forever."""
    epoch = 0
    while True:
        rng = np.random.default_rng([seed, epoch])
        yield from store.batches(batch_size, rng)
        epoch += 1


__all__ = [
    "encode_sequence",
    "select_sequences",
    "SequenceStore",
    "collate",
    "endless_batches",
    "EOS",
    "BOS",
    "PAD",
]
