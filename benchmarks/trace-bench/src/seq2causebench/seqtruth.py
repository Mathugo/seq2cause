"""The per-sequence truth, induced per grain (PRD Goal 6, Definitions "Per-sequence truth";
`plans/per-sequence-rules.md` §1–3). Score side only: this is the one module that reads the
views' parent column, after the method process has exited (scenario 34).

For a probed sequence (its `ops`, `outcomes` and `parent_pos` re-read by trace id), window
position `t` (0-based, matrices are over the window) is `ops` index `c + t − 1`; positions beyond
`max_len` were truncated by the read.

- request grain: `(j, q)` directed truth iff `parent_pos[o_j] == o_q` and `(tok_j, tok_q)` is a
  directed target edge at the floor;
- session grain: `root(x)` follows `parent_pos` to −1; within one request the request rule; across
  requests `(j < q)` iff `(tok_j, tok_q)` is a session target edge at the floor absent from the
  request target at the floor;
- bidirected target pairs → adjacencies over every occurrence pair, no call-link requirement;
- the ancestor-link column: the request rule with any ancestor in place of the direct parent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tracebench.constants import ALPHABET_JSON, GRAPHS_DIR, SCORING_TARGET_JSON
from tracebench.score import SCORING_TARGET_SESSION_JSON, build_universe, truth_sets

from .constants import BOS
from .corpus import Corpus
from .record import read_json

TRUTH_COLUMNS = ("trace_id", "ops", "outcomes", "parent_pos")
RULES_VERSION = "per-sequence-rules@2026-09-24"


def load_targets(corpus_dir, grain):
    """`(target, alphabet, universe ordered set, universe unordered set, directed truth set,
    bidirected truth set, request-directed truth set)` at the target's default floor."""
    gdir = Path(corpus_dir) / GRAPHS_DIR
    target = read_json(
        gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON)
    )
    alphabet = read_json(gdir / ALPHABET_JSON)
    floor = float(target["default_floor"])
    ordered, unordered = build_universe(alphabet, target)
    directed, bidirected = truth_sets(target, floor)
    if grain == "request":
        request_directed = directed
    else:
        req = read_json(gdir / SCORING_TARGET_JSON)
        request_directed, _ = truth_sets(req, float(req["default_floor"]))
    return {
        "target": target,
        "alphabet": alphabet,
        "floor": floor,
        "ordered": set(map(tuple, ordered)),
        "unordered": {frozenset(p) for p in unordered},
        "directed": set(map(tuple, directed)),
        "bidirected": {frozenset(p) for p in bidirected},
        "request_directed": set(map(tuple, request_directed)),
    }


def rows_by_trace_id(corpus, split, trace_ids):
    """`{trace_id: (ops, outcomes, parent_pos)}` for the wanted rows of one split."""
    wanted = set(trace_ids)
    out = {}
    for tid, ops, outcomes, parent_pos in corpus.sequences(split, columns=TRUTH_COLUMNS):
        if tid in wanted:
            out[tid] = (list(ops), list(outcomes), list(parent_pos))
            if len(out) == len(wanted):
                break
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"{len(missing)} probed trace id(s) not found in split {split!r}")
    return out


def roots(parent_pos):
    """Root index of every span (follow the parent chain to −1)."""
    n = len(parent_pos)
    root = [-1] * n
    for i in range(n):
        x = i
        seen = 0
        while parent_pos[x] >= 0 and seen <= n:
            x = parent_pos[x]
            seen += 1
        root[i] = x
    return root


def ancestors(parent_pos, i):
    out = set()
    x = parent_pos[i]
    guard = 0
    while x >= 0 and guard <= len(parent_pos):
        out.add(x)
        x = parent_pos[x]
        guard += 1
    return out


def induce(ids, ops, outcomes, parent_pos, c, vocab, grain, targets, max_lag):
    """Per-sequence truth over the window of one probed sequence.

    `ids` is the stored token sequence (`BOS + ids[:max_len] + EOS`); the window is positions
    `c..L-1`, i.e. window index `t` ↔ ops index `o = c + t − 1` (BOS at position 0). Returns
    `{candidates: [(j, q)], directed: set, adjacency: set, ancestor_directed: set, n_links_lost}`
    with `(j, q)` window indices, `j < q`.
    """
    L = len(ids)
    lc = L - c
    n_ops = len(ops)
    kept = min(n_ops, L - 2)  # real tokens kept by the read (after BOS, before EOS)
    assert ids[0] == BOS
    tok = {}  # window index -> scorer token string (real tokens only)
    op_index = {}  # window index -> ops index
    for t in range(lc):
        pos = c + t
        o = pos - 1
        if 0 <= o < kept and vocab.is_real(int(ids[pos])):
            tok[t] = vocab.token_string(int(ids[pos]))
            op_index[t] = o
    root = roots(parent_pos) if grain == "session" else None
    n_links_lost = sum(1 for o in range(kept) if parent_pos[o] >= kept)
    candidates, directed, adjacency, anc_directed = [], set(), set(), set()
    for j in range(lc):
        if j not in tok:
            continue
        for q in range(j + 1, min(lc, j + max_lag + 1)):
            if q not in tok:
                continue
            pair = (tok[j], tok[q])
            if pair not in targets["ordered"]:
                continue
            candidates.append((j, q))
            oj, oq = op_index[j], op_index[q]
            unordered = frozenset(pair)
            if unordered in targets["bidirected"]:
                adjacency.add((j, q))
            if pair in targets["directed"]:
                same_request = grain == "request" or root[oj] == root[oq]
                if same_request:
                    if parent_pos[oj] == oq:
                        directed.add((j, q))
                    if oq in ancestors(parent_pos, oj):
                        anc_directed.add((j, q))
                elif pair not in targets["request_directed"]:
                    directed.add((j, q))
                    anc_directed.add((j, q))
    return {
        "candidates": candidates,
        "directed": directed,
        "adjacency": adjacency | {p for p in directed},
        "ancestor_directed": anc_directed,
        "n_links_lost": n_links_lost,
    }


def truth_for_read(corpus_dir, ordering, grain, split, sequences_record, stored_ids, c, max_lag):
    """Per-sequence truth for every stored sequence of a read, keyed by its store index."""
    corpus = Corpus(corpus_dir, ordering, grain)
    from .vocab import Vocab

    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    targets = load_targets(corpus_dir, grain)
    rows = rows_by_trace_id(corpus, split, [s["trace_id"] for s in sequences_record["sequences"]])
    out = {}
    for entry, ids in zip(sequences_record["sequences"], stored_ids, strict=True):
        ops, outcomes, parent_pos = rows[entry["trace_id"]]
        out[int(entry["seq"])] = induce(
            np.asarray(ids), ops, outcomes, parent_pos, c, vocab, grain, targets, max_lag
        )
    return out, targets, vocab


__all__ = [
    "TRUTH_COLUMNS",
    "RULES_VERSION",
    "load_targets",
    "rows_by_trace_id",
    "roots",
    "ancestors",
    "induce",
    "truth_for_read",
]
