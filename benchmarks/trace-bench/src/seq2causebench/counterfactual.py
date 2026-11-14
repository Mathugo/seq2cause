"""Forced re-runs of released requests (score side; the `forced-ok-counterfactual` truth rule).

trace-bench's engine draws every random number from a counter keyed by the request, the operation
or edge and the attempt, so re-evaluating a request draws the same numbers whatever was forced.
This module rebuilds the engine of a released corpus, re-runs its shards, and re-evaluates chosen
requests with one outcome forced to OK through the engine's own hooks:

- `force_final[op] = OK`      — the operation's final outcome (the mechanism's `F:<op>` node);
- `force_attempt[(op, 0)] = OK` — its first attempt, which also removes the retry it triggered;
- `force_bff[(scenario, step, client attempt)] = y` — the edge operation's outcome, in a whole-shard
  re-run, read only for the client outcome of that same client attempt.

A pair (source position, target position) of one request is a truth edge when forcing the source
changes the target's outcome, alone (`single`) or once the target operation's other non-OK callees
are forced OK too (`contingent`). Baseline and forced runs are joined on identity only —
(request, operation, attempt) for an operation attempt, (session, step, client attempt) for a
client record — never on row order or the engine's row counters, which shift in a re-run.

It reads the oracle tables and the instantiation of a corpus, so it is score-side only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tracebench.engine import Engine, SpillExceeded, Table
from tracebench.generate import (
    SPILL_RETRY_MARGIN,
    SPILL_RETRY_MAX,
    _n_ticks,
    _shard_bounds,
    spill_allowance,
)
from tracebench.instantiate import load_instantiation
from tracebench.latents import Slots, compile_fault_forcings, shard_checkpoints, simulate_latents

OK, ABSENT = 0, 5
SRC_FINAL, SRC_ATTEMPT0, SRC_EDGE = "final", "attempt0", "edge"
COL_DIRECTED, COL_ANCESTOR = "directed", "ancestor"
KIND_ERROR, KIND_TXN = "sentry.error", "sentry.transaction"
CLIENT_ERR, CLIENT_TXN = "err", "txn"


class CounterfactualRefusal(RuntimeError):
    pass


def step_of(req_gid):
    return (int(req_gid) & 0xFF) // 16


def client_attempt_of(req_gid):
    return (int(req_gid) & 0xFF) % 16


def session_of(req_gid):
    return int(req_gid) >> 8


def shard_of(req_gid):
    return int(req_gid) >> 40


class ShardRun:
    """One shard's unforced re-run, indexed by identity."""

    def __init__(self, runs, shard):
        self.runs = runs
        self.shard = int(shard)
        self.res = runs.run_shard(shard, {})
        r, h, c = self.res.requests.cols, self.res.hops.cols, self.res.clients.cols
        self.row_of = {int(g): i for i, g in enumerate(r["req_gid"])}
        self.hops = defaultdict(dict)  # req_gid -> {(op, attempt): outcome}
        for g, op, k, o in zip(h["req_gid"], h["op"], h["attempt"], h["outcome"], strict=True):
            self.hops[int(g)][(int(op), int(k))] = int(o)
        # (session, step, client attempt) -> (client outcome 0/1, sampled transaction, true ts ms)
        self.clients = {
            (int(g), int(j), int(k)): (int(o), bool(t), int(ts))
            for g, j, k, o, t, ts in zip(
                c["gid"],
                c["step"],
                c["attempt"],
                c["outcome"],
                c["sampled_txn"],
                c["ts_ms"],
                strict=True,
            )
        }
        self.requests_of = defaultdict(list)  # session -> its requests in start order
        for i in np.argsort(r["start_ms"], kind="stable"):
            self.requests_of[int(r["gid"][i])].append(int(r["req_gid"][i]))
        self.engine = Engine(runs.inst, runs.seed, runs.slots, runs.forcings)
        self._client_runs = {}

    def bff_of(self, req_gid):
        return int(self.res.requests.cols["bff_op"][self.row_of[req_gid]])

    def evaluate(self, req_gids, forced, narrow, detail):
        """Re-evaluate the given requests with `forced` = a set of `(SRC_FINAL | SRC_ATTEMPT0, op)`
        held at OK; every request keeps its baseline start time, tick and latent state.
        `{req_gid: {(op, attempt): outcome}}`, or the full hop columns per identity when `detail`."""
        r = self.res.requests.cols
        eng = self.engine
        eng.force_final = {op: OK for kind, op in forced if kind == SRC_FINAL}
        eng.force_attempt = {(op, 0): OK for kind, op in forced if kind == SRC_ATTEMPT0}
        groups = defaultdict(list)
        for g in req_gids:
            i = self.row_of[g]
            groups[(int(r["bff_op"][i]), int(r["step"][i]), int(r["attempt"][i]))].append(i)
        out = {int(g): {} for g in req_gids}
        try:
            for (b, step, attempt), rows in sorted(groups.items()):
                rows = np.asarray(rows, dtype=np.int64)
                full = eng.reach[b]
                if narrow:
                    # the operations any of these requests invoked, plus their direct callees:
                    # the engine's timing pass looks up every callee of an operation that ran
                    used = {op for i in rows for op, _k in self.hops[int(r["req_gid"][i])]}
                    used |= {w for op in used for w in self.runs.callees[op]}
                    eng.reach[b] = [op for op in full if op in used]
                eng._req_counter = eng._hop_counter = 0
                requests, hops = Table(), Table()
                try:
                    eng._evaluate_tree(
                        b,
                        r["session_row"][rows],
                        self.res.sessions,
                        step,
                        attempt,
                        r["start_ms"][rows],
                        r["tick"][rows],
                        self.res.latents,
                        requests,
                        hops,
                    )
                finally:
                    eng.reach[b] = full
                h = hops.finalize().cols
                for n in range(len(h["req_gid"])):
                    key = (int(h["op"][n]), int(h["attempt"][n]))
                    if detail:
                        val = tuple(
                            int(h[name][n])
                            for name in ("outcome", "start_ms", "duration_us", "own_us", "tick")
                        )
                    else:
                        val = int(h["outcome"][n])
                    out[int(h["req_gid"][n])][key] = val
        finally:
            eng.force_final, eng.force_attempt = {}, {}
        return out

    def client_under(self, step, attempt, y):
        """`{session: client outcome}` of client attempt `(step, attempt)` in a whole-shard re-run
        with the edge outcome of that attempt forced to `y` for every scenario. Only this client
        attempt is read; nothing after it."""
        key = (int(step), int(attempt), int(y))
        if key not in self._client_runs:
            n_scen = len(self.runs.inst.sset.scenarios)
            res = self.runs.run_shard(
                self.shard, {(s, int(step), int(attempt)): int(y) for s in range(n_scen)}
            )
            c = res.clients.cols
            self._client_runs[key] = {
                int(g): int(o)
                for g, j, k, o in zip(c["gid"], c["step"], c["attempt"], c["outcome"], strict=True)
                if int(j) == key[0] and int(k) == key[1]
            }
        return self._client_runs[key]


class CorpusRuns:
    """The engine of a released corpus, rebuilt from its own files."""

    def __init__(self, corpus_dir):
        self.dir = Path(corpus_dir)
        self.inst = load_instantiation(self.dir)
        self.seed = int(json.loads((self.dir / "instantiation.json").read_text())["seed"])
        self.slots = Slots.build(self.inst.topo)
        self.forcings, _ = compile_fault_forcings(self.inst, self.slots)
        cfg = self.inst.cfg
        self.n_ticks = _n_ticks(cfg)
        self.bounds = {k: (t0, t1) for k, t0, t1 in _shard_bounds(cfg)}
        self.checkpoints = shard_checkpoints(
            self.inst, self.seed, self.slots, cfg.run.shard_ticks, self.n_ticks, self.forcings
        )
        self.spill = spill_allowance(self.inst)
        topo = self.inst.topo
        self.bff_ops = {int(o) for o in topo.bff_ops}
        self.client_ops = {int(o) for o in topo.client_ops}
        self.callers = defaultdict(dict)  # callee -> {caller: critical on any edge}
        self.callees = defaultdict(set)  # caller -> callees
        for e in topo.edges:
            prev = self.callers[int(e.callee)].get(int(e.caller), False)
            self.callers[int(e.callee)][int(e.caller)] = bool(e.critical) or prev
            self.callees[int(e.caller)].add(int(e.callee))
        self._shards = {}

    def run_shard(self, shard, force_bff):
        """One shard through `Engine.run_shard`, as the generator runs it (spill retry included)."""
        t0, t1 = self.bounds[int(shard)]
        spill = self.spill
        for attempt in range(SPILL_RETRY_MAX):
            latents, _ = simulate_latents(
                self.inst,
                self.seed,
                self.slots,
                self.checkpoints[int(shard)],
                t0,
                min(t1 + spill, self.n_ticks + spill),
                self.forcings,
            )
            engine = Engine(self.inst, self.seed, self.slots, self.forcings)
            engine.force_bff = dict(force_bff)
            try:
                return engine.run_shard(int(shard), t0, t1, latents)
            except SpillExceeded as e:
                if attempt == SPILL_RETRY_MAX - 1:
                    raise
                spill = e.needed_ticks - (t1 - t0) + SPILL_RETRY_MARGIN
        raise AssertionError("unreachable")

    def shard(self, shard):
        if shard not in self._shards:
            self._shards[shard] = ShardRun(self, shard)
        return self._shards[shard]

    # --- identity check 1: the re-run equals the corpus -----------------------------------------
    def check_against_oracle(self, shard):
        """Refuses unless the unforced re-run of `shard` equals its oracle span and session tables
        and reproduces the linkage table's client records."""
        run = self.shard(shard)
        odir = self.dir / "oracle" / f"shard={int(shard):04d}"
        h = run.res.hops.cols
        mine = {
            (int(h["req_gid"][i]), int(h["op"][i]), int(h["attempt"][i])): (
                int(h["outcome"][i]),
                int(h["start_ms"][i]),
                int(h["duration_us"][i]),
                int(h["own_us"][i]),
                int(h["tick"][i]),
                int(h["parent_op"][i]),
            )
            for i in range(len(h["req_gid"]))
        }
        spans = pq.read_table(odir / "spans.parquet").to_pylist()
        op_of = {s["span_id"]: s["op"] for s in spans}
        theirs = {
            (s["req_gid"], s["op"], s["attempt"]): (
                s["outcome"],
                s["start_ms"],
                s["dur_us"],
                s["own_us"],
                s["tick"],
                op_of[s["parent_span_id"]] if s["parent_span_id"] else -1,
            )
            for s in spans
        }
        if mine != theirs:
            bad = [k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)]
            raise CounterfactualRefusal(
                f"shard {shard}: the re-run differs from the oracle spans on {len(bad)} of "
                f"{len(theirs)} rows, e.g. {bad[:3]}"
            )
        s = run.res.sessions
        mine_s = {
            int(s["gid"][i]): (int(s["scenario"][i]), int(s["arrival_ms"][i]), int(s["net"][i]),
                               int(s["auth"][i]), int(s["steps_done"][i]), bool(s["final_ok"][i]))
            for i in range(len(s["gid"]))
        }  # fmt: skip
        theirs_s = {
            x["gid"]: (x["scenario"], x["arrival_ms"], x["net"], x["auth"], x["steps_done"], x["final_ok"])
            for x in pq.read_table(odir / "sessions.parquet").to_pylist()
        }  # fmt: skip
        if mine_s != theirs_s:
            raise CounterfactualRefusal(
                f"shard {shard}: the re-run differs from the oracle sessions"
            )
        link = pq.read_table(
            odir / "linkage.parquet", columns=["kind", "req_gid"], filters=[("kind", "in", [KIND_ERROR, KIND_TXN])]
        ).to_pylist()  # fmt: skip
        theirs_c = {(session_of(x["req_gid"]), step_of(x["req_gid"]), client_attempt_of(x["req_gid"]), x["kind"]) for x in link}  # fmt: skip
        mine_c = set()
        for (g, j, k), (outcome, sampled, _ts) in run.clients.items():
            if outcome == 1:
                mine_c.add((g, j, k, KIND_ERROR))
            if sampled:
                mine_c.add((g, j, k, KIND_TXN))
        if mine_c != theirs_c:
            raise CounterfactualRefusal(
                f"shard {shard}: the re-run's client records differ from the linkage table "
                f"({len(mine_c ^ theirs_c)} of {len(theirs_c)})"
            )
        return {"hops": len(theirs), "sessions": len(theirs_s), "client_records": len(theirs_c)}


def request_edges(run, runs, req_gids):
    """The truth pairs of each given request, in identity terms.

    `{req_gid: [pair]}`, a pair being `{"src": (op, attempt), "dst": (op, attempt) | CLIENT_ERR,
    "column", "single", "contingent", "source", "critical"}`. Sources are non-OK only; targets are
    non-OK attempts of another operation of the same request, or the request's own client error.
    Also returns the counters of the re-runs made."""
    base = {g: run.hops[g] for g in req_gids}
    plan = {}  # req_gid -> [(source kind, callee op, source attempt)]
    specs = defaultdict(set)  # frozenset of forced items -> requests needing that run
    for g, hops in base.items():
        attempts = defaultdict(dict)
        for (op, k), o in hops.items():
            attempts[op][k] = o
        final = {op: a[max(a)] for op, a in attempts.items()}
        b = run.bff_of(g)
        sources = []
        for op, a in sorted(attempts.items()):
            if op == b:
                continue
            if final[op] != OK:
                sources.append((SRC_FINAL, op, max(a)))
            if max(a) > 0 and a[0] != OK:
                sources.append((SRC_ATTEMPT0, op, 0))
        for kind, op, _k in sources:
            specs[frozenset({(kind, op)})].add(g)
            for caller in runs.callers[op]:
                if caller not in attempts:
                    continue
                others = frozenset(
                    (SRC_FINAL, w)
                    for w in runs.callees[caller]
                    if w != op and w in final and final[w] != OK
                )
                if others:
                    specs[others].add(g)
                    specs[others | {(kind, op)}].add(g)
        plan[g] = (sources, final, b)
    forced = {
        spec: run.evaluate(sorted(gs), spec, True, False) for spec, gs in sorted(specs.items(), key=lambda x: sorted(x[0]))
    }  # fmt: skip
    out = {}
    n_client_runs = set()
    for g, (sources, final, b) in plan.items():
        hops = base[g]
        key = (session_of(g), step_of(g), client_attempt_of(g))
        client_err = run.clients[key][0] == 1
        pairs = []
        for kind, op, k_src in sources:
            alone = forced[frozenset({(kind, op)})][g]
            for (tgt, k), o in sorted(hops.items()):
                if tgt == op or o == OK:
                    continue
                single = alone.get((tgt, k), ABSENT) != o
                direct = tgt in runs.callers[op]
                contingent = False
                if direct:
                    others = frozenset(
                        (SRC_FINAL, w)
                        for w in runs.callees[tgt]
                        if w != op and w in final and final[w] != OK
                    )
                    if others:
                        x = forced[others][g].get((tgt, k), ABSENT)
                        y = forced[others | {(kind, op)}][g].get((tgt, k), ABSENT)
                        contingent = x != y
                if direct and (single or contingent):
                    pairs.append({"src": (op, k_src), "dst": (tgt, k), "column": COL_DIRECTED,
                                  "single": single, "contingent": contingent, "source": kind,
                                  "critical": runs.callers[op][tgt]})  # fmt: skip
                elif single:
                    pairs.append({"src": (op, k_src), "dst": (tgt, k), "column": COL_ANCESTOR,
                                  "single": True, "contingent": False, "source": kind,
                                  "critical": None})  # fmt: skip
            # the request's own client error, through the edge outcome this forcing produces
            y = alone.get((b, 0), ABSENT)
            if client_err and y != hops[(b, 0)] and y != ABSENT:
                n_client_runs.add((key[1], key[2], y))
                if run.client_under(key[1], key[2], y)[key[0]] == 0:
                    pairs.append({"src": (op, k_src), "dst": CLIENT_ERR, "column": COL_ANCESTOR,
                                  "single": True, "contingent": False, "source": kind,
                                  "critical": None})  # fmt: skip
        if client_err and hops[(b, 0)] != OK:
            n_client_runs.add((key[1], key[2], OK))
            if run.client_under(key[1], key[2], OK)[key[0]] == 0:
                pairs.append({"src": (b, 0), "dst": CLIENT_ERR, "column": COL_DIRECTED,
                              "single": True, "contingent": False, "source": SRC_EDGE,
                              "critical": None})  # fmt: skip
        out[g] = pairs
    return out, {"n_tree_runs": len(specs), "n_shard_runs": len(n_client_runs)}


__all__ = [
    "OK",
    "ABSENT",
    "SRC_FINAL",
    "SRC_ATTEMPT0",
    "SRC_EDGE",
    "COL_DIRECTED",
    "COL_ANCESTOR",
    "CLIENT_ERR",
    "CLIENT_TXN",
    "KIND_ERROR",
    "KIND_TXN",
    "CounterfactualRefusal",
    "CorpusRuns",
    "ShardRun",
    "request_edges",
    "step_of",
    "client_attempt_of",
    "session_of",
    "shard_of",
]
