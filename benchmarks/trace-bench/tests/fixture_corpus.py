# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""A tiny trace-bench-shaped corpus for the test suite, built once per process.

Six operations over three services, four minted outcome variants and two
observed-but-unminted ones (so folding is exercised), four views, a request
and a session scoring target whose every edge lies inside the scorer's
universe (so `tracebench.score.self_check` passes on both grains), a shipped
prior, non-method-readable graphs / labels / topology, a manifest at the
benchmark tool version this repository scores, and the completion marker.
Sequences are drawn from a small call tree whose outcomes propagate
callee -> caller, so a probe on the fixture has something to find; the
target is written by hand to be consistent with that tree, not derived
from the sequences (the fixture pins contracts, not truth).
"""

from __future__ import annotations

import functools
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tracebench.allowlist import is_method_readable
from tracebench.constants import (
    BOS,
    EOS,
    FLOOR_SWEEP,
    N_SPECIALS,
    OUTCOME_IDS,
    OUTCOME_NAMES,
    PAD,
    UNK,
)
from tracebench.correlate.views import ARROW_SCHEMA
from tracebench.record import sha256_file, write_json

_ROOT = Path(tempfile.mkdtemp(prefix="seq2causebench-tests-"))

OK, X4, ERR, SLOW = OUTCOME_IDS["ok"], OUTCOME_IDS["4xx"], OUTCOME_IDS["err"], OUTCOME_IDS["slow"]

# op id -> (service, endpoint name, kind)  kinds: 0 bff, 1 service
OPS = {
    0: ("tb-edge", "/v1/alpha/root", 0),
    1: ("tb-edge", "/v1/alpha/other", 0),
    2: ("tb-beta", "/v1/beta/lookup", 1),
    3: ("tb-beta", "/v1/beta/write", 1),
    4: ("tb-gamma", "/v1/gamma/store", 1),
    5: ("tb-gamma", "/v1/gamma/audit", 1),
}
# potential (op, outcome) tokens = the alphabet; the minted subset is the model vocabulary
ALPHABET = [
    (0, OK),
    (0, X4),
    (1, OK),
    (1, ERR),
    (2, OK),
    (2, ERR),
    (3, OK),
    (3, SLOW),
    (4, OK),
    (4, ERR),
    (5, OK),
    (5, ERR),
]
MINTED = [(1, ERR), (2, ERR), (3, SLOW), (4, ERR)]  # (5, err) and (0, 4xx) fold to their base op
CALLS = {0: (2, 3), 1: (3,), 2: (4,), 3: (5,), 4: (), 5: ()}  # caller -> callees
CALL_P = {(0, 2): 0.8, (0, 3): 0.8, (1, 3): 0.9, (2, 4): 0.7, (3, 5): 0.7}

REQUEST_DIRECTED = [
    ("4:err", "2:err", 0.70),
    ("2:err", "0:4xx", 0.60),
    ("5:err", "3:slow", 0.50),
    ("3:slow", "0:4xx", 0.30),
    ("3:slow", "1:err", 0.40),
    ("5:err", "1:err", 0.20),
    ("4:err", "0:4xx", 0.02),
]
REQUEST_BIDIRECTED = [("2:err", "3:slow", 0.40), ("4:err", "5:err", 0.10), ("0:4xx", "1:err", 0.01)]
REQUEST_SUPPORT = [[0, 2], [0, 3], [0, 4], [0, 5], [1, 3], [1, 5], [2, 3], [2, 4], [3, 5], [4, 5]]
SESSION_EXTRA_DIRECTED = [("0:ok", "1:ok", 0.30)]
SESSION_SUPPORT = REQUEST_SUPPORT + [[0, 1], [1, 2], [1, 4], [2, 5], [3, 4]]

N_REQUESTS = {"train": 160, "val": 24, "test": 24}
N_SESSIONS = {"train": 60, "val": 10, "test": 10}


def token(op, o):
    return f"{op}:{OUTCOME_NAMES[o]}"


def column(op):
    svc, name, _ = OPS[op]
    return f"{svc}:{name.strip('/').replace('/', '_')}"


# --- sequences ------------------------------------------------------------------------------
def _request(rng, root):
    """Spans as (op, outcome, parent_index_in_preorder); outcomes propagate callee -> caller."""
    spans = []  # preorder

    def visit(op, parent):
        idx = len(spans)
        spans.append([op, OK, parent])
        for callee in CALLS[op]:
            if rng.random() < CALL_P[(op, callee)]:
                visit(callee, idx)
        return idx

    visit(root, -1)
    # outcomes bottom-up
    for idx in range(len(spans) - 1, -1, -1):
        op = spans[idx][0]
        kids = [s for s in spans if s[2] == idx]
        kid_bad = any(s[1] != OK for s in kids)
        if op in (4, 5):
            spans[idx][1] = ERR if rng.random() < 0.15 else OK
        elif op == 2:
            spans[idx][1] = ERR if rng.random() < (0.7 if kid_bad else 0.03) else OK
        elif op == 3:
            spans[idx][1] = SLOW if rng.random() < (0.5 if kid_bad else 0.05) else OK
        elif op == 0:
            spans[idx][1] = X4 if rng.random() < (0.6 if kid_bad else 0.02) else OK
        elif op == 1:
            spans[idx][1] = ERR if rng.random() < (0.5 if kid_bad else 0.02) else OK
    return spans


def _order(spans, ordering):
    """Positions in `end` (post-order: callee precedes caller) or `start` (pre-order)."""
    n = len(spans)
    if ordering == "start":
        perm = list(range(n))
    else:
        perm = []

        def post(idx):
            for j, s in enumerate(spans):
                if s[2] == idx:
                    post(j)
            perm.append(idx)

        post(0)
    pos = {old: new for new, old in enumerate(perm)}
    ops = [spans[i][0] for i in perm]
    outcomes = [spans[i][1] for i in perm]
    parent_pos = [pos[spans[i][2]] if spans[i][2] >= 0 else -1 for i in perm]
    return ops, outcomes, parent_pos


def _row(trace_id, ops, outcomes, parent_pos, kind, t0):
    n = len(ops)
    return {
        "trace_id": trace_id,
        "ops": ops,
        "outcomes": outcomes,
        "offsets": [i * 1_000_000 for i in range(n)],
        "durations": [5_000_000] * n,
        "parent_pos": parent_pos,
        "healthy": all(o == OK for o in outcomes),
        "n_err_spans": sum(o != OK for o in outcomes),
        "n_spans": n,
        "truncated": False,
        "scenario_hash": hashlib.sha1(
            ";".join(
                sorted(
                    {
                        f"{ops[p] if p >= 0 else -1},{o}"
                        for o, p in zip(ops, parent_pos, strict=False)
                    }
                )
            ).encode()
        ).hexdigest(),
        "scenario_sid": -1,
        "start_ns": t0,
        "attribution_level": 0,
        "sequence_kind": kind,
    }


def _sequences(seed):
    """All sequences of the corpus: {(ordering, grain, split): [rows]}."""
    rng = np.random.default_rng(seed)
    out = {}
    t0 = 1_767_600_000_000_000_000
    for split, n_req in N_REQUESTS.items():
        reqs = []
        for i in range(n_req):
            spans = _request(rng, int(rng.integers(0, 2)))
            tid = hashlib.md5(f"{seed}:{split}:req:{i}".encode()).hexdigest()
            reqs.append((tid, spans))
        for ordering in ("end", "start"):
            out[(ordering, "request", split)] = [
                _row(tid, *_order(spans, ordering), "request", t0 + i * 10**9)
                for i, (tid, spans) in enumerate(reqs)
            ]
        # sessions: 1..3 consecutive requests, positions re-based, parents kept within each request
        sess_rows = {"end": [], "start": []}
        i = 0
        k = 0
        while i < len(reqs) and k < N_SESSIONS[split]:
            m = int(rng.integers(1, 4))
            group = reqs[i : i + m]
            i += m
            for ordering in ("end", "start"):
                ops, outs, pars = [], [], []
                for _tid, spans in group:
                    o, u, p = _order(spans, ordering)
                    base = len(ops)
                    ops += o
                    outs += u
                    pars += [x + base if x >= 0 else -1 for x in p]
                tid = hashlib.md5(f"{seed}:{split}:sess:{k}".encode()).hexdigest()
                sess_rows[ordering].append(_row(tid, ops, outs, pars, "session", t0 + k * 10**9))
            k += 1
        for ordering in ("end", "start"):
            out[(ordering, "session", split)] = sess_rows[ordering]
    return out


# --- records --------------------------------------------------------------------------------
def _model_vocab():
    base_ops = [
        {"id": op, "service": OPS[op][0], "name": OPS[op][1], "kind": OPS[op][2]}
        for op in sorted(OPS)
    ]
    variants = [
        {"op_id": op, "outcome": OUTCOME_NAMES[o], "outcome_id": o} for op, o in sorted(MINTED)
    ]
    return {
        "version": 1,
        "normalizer_version": "tracebench-1",
        "n_specials": N_SPECIALS,
        "specials": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "UNK": UNK},
        "vocab_size": N_SPECIALS + len(base_ops) + len(variants),
        "base_ops": base_ops,
        "variants": variants,
    }


def _alphabet():
    toks = [
        {
            "token": token(op, o),
            "op_id": op,
            "outcome": OUTCOME_NAMES[o],
            "service": OPS[op][0],
            "name": OPS[op][1],
            "kind": OPS[op][2],
        }
        for op, o in ALPHABET
    ]
    return {"description": "fixture alphabet", "n_tokens": len(toks), "tokens": toks}


def _target(variant, grain):
    directed = list(REQUEST_DIRECTED) + (SESSION_EXTRA_DIRECTED if grain == "session" else [])
    bidirected = list(REQUEST_BIDIRECTED) if variant == "latent" else []
    support = SESSION_SUPPORT if grain == "session" else REQUEST_SUPPORT
    d = [{"src": s, "dst": t, "strength": w, "via": []} for s, t, w in directed]
    b = [{"a": a, "b": c, "strength": w, "via": "fixture"} for a, c, w in bidirected]
    return {
        "description": "fixture scoring target",
        "variant": variant,
        "grain": grain,
        "default_floor": 0.05,
        "directed": d,
        "bidirected": b,
        "bidirected_groups": [],
        "n_directed": len(d),
        "n_directed_at_floor": sum(e["strength"] >= 0.05 for e in d),
        "n_bidirected": len(b),
        "n_bidirected_at_floor": sum(e["strength"] >= 0.05 for e in b),
        "directed_acyclic_at_floor": True,
        "support_op_pairs": support,
        "support_pairs": len(support),
        "retry_pairs": [],
    }


def _prior():
    columns = [column(op) for op in sorted(OPS)]
    edges = [
        {
            "from": column(callee),
            "to": column(caller),
            "prob": 1.0,
            "why": "deployment call edge; outcomes propagate callee -> caller",
            "evidence": f"fixture call {caller}->{callee}",
        }
        for caller, callees in sorted(CALLS.items())
        for callee in callees
    ]
    return {
        "description": "fixture deployment topology as a structural prior (callee -> caller)",
        "dataset": "trace-bench",
        "columns": columns,
        "default_prob": 0.01,
        "reverse_prob": 0.005,
        "edges": edges,
    }


def write_fixture_corpus(out, variant="latent", seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = _sequences(seed)
    vocab = _model_vocab()
    rows_stat = {}
    for ordering in ("end", "start"):
        for grain in ("request", "session"):
            view = out / "views" / f"{ordering}-{grain}"
            for split in ("train", "val", "test"):
                rows = seqs[(ordering, grain, split)]
                rows_stat[f"{ordering}-{grain}-{split}"] = len(rows)
                part = (
                    view / "sequences" / f"split={split}" / "date=2026-01-01" / "part-0000.parquet"
                )
                part.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows, schema=ARROW_SCHEMA), part)
            (view / "model-vocab.json").write_text(json.dumps(vocab, sort_keys=True))
            write_json(
                view / "export-stats.json",
                {
                    "rows": rows_stat,
                    "vocab_size": vocab["vocab_size"],
                    "k_ops": len(OPS),
                    "n_variants": len(MINTED),
                    "alphabet_size_realized_train": len(ALPHABET),
                    "ordering": ordering,
                    "grain": grain,
                },
            )
            write_json(view / "scenario-prevalence.json", {})
            write_json(view / "slow-thresholds.json", {})
    g = out / "graphs"
    write_json(g / "alphabet.json", _alphabet())
    write_json(g / "scoring-target.json", _target(variant, "request"))
    write_json(g / "scoring-target-session.json", _target(variant, "session"))
    write_json(
        g / "floor-sensitivity.json",
        {"default_floor": 0.05, "sweep": [{"floor": f} for f in FLOOR_SWEEP]},
    )
    write_json(g / "mechanism-graph.json", {"fixture": True})
    write_json(out / "topology" / "prior.json", _prior())
    write_json(out / "topology" / "callgraph.json", {"fixture": True})
    write_json(out / "labels" / "cases.json", {"fixture": True})
    write_json(out / "instantiation.json", {"fixture": True, "seed": seed})
    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out).as_posix()
            files.append(
                {
                    "path": rel,
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                    "method_readable": is_method_readable(rel),
                }
            )
    write_json(
        out / "manifest.json",
        {
            "schema": "tracebench/manifest@1",
            "tool": "trace-bench",
            "tool_version": "0.3.0",
            "config_hash": f"fixture-{variant}-{seed}",
            "instance": "fixture",
            "variant": variant,
            "seed": seed,
            "license": "CC-BY-4.0",
            "code_license": "MIT",
            "default_floor": 0.05,
            "alphabet_size_potential": len(ALPHABET),
            "alphabet_size_realized_train": len(ALPHABET),
            "alphabet_size_vocab": vocab["vocab_size"],
            "files": files,
        },
    )
    (out / "COMPLETE").write_text("")
    return out


@functools.cache
def fixture_corpus(variant="latent", seed=0):
    """Path of the fixture corpus for (variant, seed), built once per process."""
    return write_fixture_corpus(_ROOT / f"fixture-{variant}-{seed}", variant, seed)
