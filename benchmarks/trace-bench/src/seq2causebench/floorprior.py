"""The topology floor's reader (score side; `plans/floors.md` §1–§2, pre-registered 2026-10-09).

`floor/topology` reads two shipped files of a corpus and nothing else: `topology/prior.json`,
the deployment topology as a structural prior (edges `from` = callee column, `to` = caller
column, the per-edge call probability `p_call` inside the edge's `evidence` field, four
decimals), and `graphs/alphabet.json`, which names the corpus's `(op, outcome)` tokens and maps
the prior's `<service>:<endpoint>` columns onto op ids. It never opens a view and never
constructs `Corpus`; it is not a method-side module and may name these paths.

The prediction is every ordered token pair `(t_callee, t_caller)` of every direct call edge,
scored by that edge's `p_call` (a repeated op pair keeps the larger value); pairs outside the
scorer's universe are the scorer's to drop and count.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from tracebench.topology import endpoint_column

from .constants import ALPHABET_JSON, GRAPHS_DIR, PRIOR_JSON, TOPOLOGY_DIR
from .record import read_json

P_CALL_RE = re.compile(r"p_call=([0-9]*\.?[0-9]+)")


class PriorRefusal(ValueError):
    pass


def prior_rel():
    return f"{TOPOLOGY_DIR}/{PRIOR_JSON}"


def alphabet_rel():
    return f"{GRAPHS_DIR}/{ALPHABET_JSON}"


def column_to_op(alphabet):
    """`{"<service>:<endpoint column>": op_id}` from the alphabet's tokens (the prior's column
    convention, `tracebench.topology.column_name`); refuses a column that names two ops."""
    out = {}
    for t in alphabet["tokens"]:
        key = f"{t['service']}:{endpoint_column(t['name'])}"
        op = int(t["op_id"])
        if out.setdefault(key, op) != op:
            raise PriorRefusal(f"prior column {key!r} would name two ops ({out[key]}, {op})")
    return out


def p_call_of(edge):
    m = P_CALL_RE.search(str(edge.get("evidence", "")))
    if m is None:
        raise PriorRefusal(
            f"prior edge {edge.get('from')!r} -> {edge.get('to')!r} carries no p_call in its evidence"
        )
    return float(m.group(1))


def op_pairs_from_prior(prior, col_to_op):
    """`{(callee_op, caller_op): p_call}`; the prior's `from` is the callee and `to` the caller
    (outcomes propagate callee -> caller); a repeated op pair keeps the larger p_call."""
    pairs = {}
    for e in prior["edges"]:
        try:
            a, b = col_to_op[e["from"]], col_to_op[e["to"]]
        except KeyError as k:
            raise PriorRefusal(f"prior column {k} is not an op of the alphabet") from None
        if a == b:
            raise PriorRefusal(f"prior edge {e['from']!r} calls itself")
        p = p_call_of(e)
        pairs[(a, b)] = max(pairs.get((a, b), 0.0), p)
    return pairs


def topology_read(corpus_dir):
    """The floor's score table: `tokens` (the alphabet, in order), `src` / `dst` indices into it,
    `score` = p_call, `count` = 1, and the facts the run record carries."""
    corpus_dir = Path(corpus_dir)
    prior = read_json(corpus_dir / prior_rel())
    alphabet = read_json(corpus_dir / alphabet_rel())
    tokens = np.array([str(t["token"]) for t in alphabet["tokens"]])
    op_ids = np.array([int(t["op_id"]) for t in alphabet["tokens"]], dtype=np.int64)
    pairs = op_pairs_from_prior(prior, column_to_op(alphabet))
    by_op = {int(op): np.flatnonzero(op_ids == op) for op in np.unique(op_ids)}
    empty = np.zeros(0, dtype=np.int64)
    src, dst, score = [empty], [empty], [np.zeros(0)]
    for (a, b), p in sorted(pairs.items()):
        ia, ib = by_op.get(a, empty), by_op.get(b, empty)
        src.append(np.repeat(ia, len(ib)))
        dst.append(np.tile(ib, len(ia)))
        score.append(np.full(len(ia) * len(ib), float(p)))
    src, dst, score = np.concatenate(src), np.concatenate(dst), np.concatenate(score)
    order = np.lexsort((dst, src))
    return {
        "tokens": tokens,
        "src": src[order].astype(np.int64),
        "dst": dst[order].astype(np.int64),
        "score": score[order].astype(np.float64),
        "count": np.ones(len(order), dtype=np.int64),
        "facts": {
            "n_sequences": 0,  # sample-free: no view is read (plans/floors.md §5)
            "n_skipped_short": 0,
            "n_prior_edges": int(len(prior["edges"])),
            "n_prior_columns": int(len(prior.get("columns", []))),
            "n_op_pairs": int(len(pairs)),
            "n_alphabet_tokens": int(len(tokens)),
            "inputs": [prior_rel(), alphabet_rel()],
        },
    }


__all__ = [
    "PriorRefusal",
    "P_CALL_RE",
    "column_to_op",
    "p_call_of",
    "op_pairs_from_prior",
    "topology_read",
]
