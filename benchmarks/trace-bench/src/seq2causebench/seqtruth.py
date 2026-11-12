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

# the truth rules: the landed one (the views' first-caller parent link) and the forced re-run of
# each sampled request (`counterfactual.py`, `truthbuild.py`; plans/per-sequence-rules.md addendum)
RULE_FIRST_CALLER_PARENT = "first-caller-parent"
RULE_FORCED_OK = "forced-ok-counterfactual"
RULES = (RULE_FIRST_CALLER_PARENT, RULE_FORCED_OK)
TRUTH_FILE_SCHEMA = "seq2causebench/seqtruth-forced-ok-counterfactual@1"
TRUTH_FILE_FMT = "seqtruth-forced-ok-counterfactual-{grain}.json"
OUTCOME_OK_NAME = "ok"
DROP_OK_TARGET, DROP_UNMAPPED, DROP_CLIENT_RETRY = "ok_target", "unmapped", "client_retry"
DROP_REASONS = (DROP_OK_TARGET, DROP_UNMAPPED, DROP_CLIENT_RETRY)


class TruthFileRefusal(ValueError):
    pass


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


def is_ok_token(token):
    return token.rsplit(":", 1)[-1] == OUTCOME_OK_NAME


def same_step_other_attempt(req_a, req_b):
    """Two request ids (`(session << 8) | (step * 16 + client attempt)`) of one step of one
    session with different client attempts: a client retry and the attempt it retries."""
    if req_a == req_b or (req_a >> 8) != (req_b >> 8):
        return False
    return ((req_a & 0xFF) // 16) == ((req_b & 0xFF) // 16)


def reduce_candidates(candidates, tok, req):
    """The candidate set of `forced-ok-counterfactual`: `induce`'s candidates minus the pairs whose
    target is OK, the pairs with a position no request could be assigned to, and the pairs between
    two client attempts of one step. `tok` maps a window index to its token, `req` to its request
    id (None = unmapped). Returns `(kept, {reason: n dropped})`; the first matching reason counts."""
    kept, dropped = [], dict.fromkeys(DROP_REASONS, 0)
    for j, q in candidates:
        if is_ok_token(tok[q]):
            dropped[DROP_OK_TARGET] += 1
        elif req[j] is None or req[q] is None:
            dropped[DROP_UNMAPPED] += 1
        elif same_step_other_attempt(req[j], req[q]):
            dropped[DROP_CLIENT_RETRY] += 1
        else:
            kept.append((j, q))
    return kept, dropped


def forced_ok_truth(base, entry, tok, bidirected):
    """One sequence's truth under `forced-ok-counterfactual` from its truth-file entry and the
    landed `induce` result (`base`, whose candidates do not depend on the rule). Gates the file's
    window-index convention: every pair's stored tokens must be the tokens at those window
    indices, every pair must be a reduced candidate, and no pair may have an OK target."""
    req = entry["req"]
    if len(req) != base["n_window"]:
        raise TruthFileRefusal(
            f"sequence {entry['seq']}: the truth file lists {len(req)} window positions, "
            f"the read has {base['n_window']}"
        )
    kept, dropped = reduce_candidates(base["candidates"], tok, req)
    allowed = set(kept)
    directed, anc = set(), set()
    for j, q, column, _single, _contingent, _source, _critical, tok_j, tok_q in entry["pairs"]:
        pair = (int(j), int(q))
        if tok.get(pair[0]) != tok_j or tok.get(pair[1]) != tok_q:
            raise TruthFileRefusal(
                f"sequence {entry['seq']}: pair {pair} was built on tokens ({tok_j}, {tok_q}), the "
                f"read has ({tok.get(pair[0])}, {tok.get(pair[1])}) there; window index t must be "
                "view position c + t - 1 after truncation"
            )
        if pair not in allowed:
            raise TruthFileRefusal(f"sequence {entry['seq']}: pair {pair} is not a candidate")
        if is_ok_token(tok_q):
            raise TruthFileRefusal(f"sequence {entry['seq']}: pair {pair} has an OK target")
        anc.add(pair)
        if column == "directed":
            directed.add(pair)
    adjacency = {p for p in kept if frozenset((tok[p[0]], tok[p[1]])) in bidirected}
    return {
        "candidates": kept,
        "directed": directed,
        "adjacency": adjacency | directed,
        "ancestor_directed": anc,
        "n_links_lost": 0,
        "dropped": dropped,
    }


def load_truth_file(path, sequences_record, c, max_lag):
    """The truth file of a read, refused unless it was built for exactly this read."""
    doc = read_json(path)
    if doc.get("schema") != TRUTH_FILE_SCHEMA or doc.get("rule") != RULE_FORCED_OK:
        raise TruthFileRefusal(f"{path} is not a {RULE_FORCED_OK} truth file")
    want = {"c": int(c), "max_lag": int(max_lag), "grain": sequences_record["grain"],
            "split": sequences_record["split"]}  # fmt: skip
    got = {k: doc.get(k) for k in want}
    if got != want:
        raise TruthFileRefusal(f"{path} was built for {got}, the read is {want}")
    mine = [(int(s["seq"]), s["trace_id"], int(s["length"])) for s in sequences_record["sequences"]]
    theirs = [(int(s["seq"]), s["trace_id"], int(s["length"])) for s in doc["sequences"]]
    if mine != theirs:
        raise TruthFileRefusal(f"{path} lists other sequences than the read's record")
    return doc


def truth_for_read(
    corpus_dir,
    ordering,
    grain,
    split,
    sequences_record,
    stored_ids,
    c,
    max_lag,
    rule=RULE_FIRST_CALLER_PARENT,
    truth_file=None,
):
    """Per-sequence truth for every stored sequence of a read, keyed by its store index.
    `rule` picks the truth rule; `forced-ok-counterfactual` reads `truth_file`."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    corpus = Corpus(corpus_dir, ordering, grain)
    from .vocab import Vocab

    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    targets = load_targets(corpus_dir, grain)
    rows = rows_by_trace_id(corpus, split, [s["trace_id"] for s in sequences_record["sequences"]])
    entries = None
    if rule == RULE_FORCED_OK:
        doc = load_truth_file(truth_file, sequences_record, c, max_lag)
        entries = {int(s["seq"]): s for s in doc["sequences"]}
    out = {}
    for entry, ids in zip(sequences_record["sequences"], stored_ids, strict=True):
        ops, outcomes, parent_pos = rows[entry["trace_id"]]
        ids = np.asarray(ids)
        base = induce(ids, ops, outcomes, parent_pos, c, vocab, grain, targets, max_lag)
        if entries is not None:
            tok = window_tokens(ids, c, len(ops), vocab)
            base["n_window"] = len(ids) - c
            base = forced_ok_truth(base, entries[int(entry["seq"])], tok, targets["bidirected"])
        out[int(entry["seq"])] = base
    return out, targets, vocab


def window_tokens(ids, c, n_ops, vocab):
    """`{window index: token}` of the real tokens of one stored sequence, exactly as `induce`
    resolves them: window index `t` is view position `c + t - 1` of the row after truncation."""
    kept = min(n_ops, len(ids) - 2)
    return {
        t: vocab.token_string(int(ids[c + t]))
        for t in range(len(ids) - c)
        if 0 <= c + t - 1 < kept and vocab.is_real(int(ids[c + t]))
    }


__all__ = [
    "TRUTH_COLUMNS",
    "RULES_VERSION",
    "load_targets",
    "rows_by_trace_id",
    "roots",
    "ancestors",
    "induce",
    "truth_for_read",
    "RULE_FIRST_CALLER_PARENT",
    "RULE_FORCED_OK",
    "RULES",
    "TRUTH_FILE_SCHEMA",
    "TRUTH_FILE_FMT",
    "DROP_REASONS",
    "TruthFileRefusal",
    "is_ok_token",
    "same_step_other_attempt",
    "reduce_candidates",
    "forced_ok_truth",
    "load_truth_file",
    "window_tokens",
]
