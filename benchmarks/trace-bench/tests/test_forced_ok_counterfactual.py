"""The `forced-ok-counterfactual` truth rule (plans/per-sequence-rules.md, addendum 2026-11-11):
which pairs a forced re-run makes truth, which pairs are scored at all, the truth file's
window-index gate, and the identity join of a forced shard re-run to its baseline."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from seq2causebench.counterfactual import (
    ABSENT,
    CLIENT_ERR,
    COL_ANCESTOR,
    COL_DIRECTED,
    OK,
    SRC_ATTEMPT0,
    SRC_EDGE,
    SRC_FINAL,
    CorpusRuns,
    client_attempt_of,
    request_edges,
    session_of,
    step_of,
)
from seq2causebench.seqscore import aggregate_read
from seq2causebench.seqtruth import (
    DROP_CLIENT_RETRY,
    DROP_OK_TARGET,
    DROP_UNMAPPED,
    TruthFileRefusal,
    forced_ok_truth,
    reduce_candidates,
    window_tokens,
)

ENGINE_FIXTURE = Path(__file__).parent / "fixtures" / "xs-latent-seed0-engine"
ERR, SLOW, E5 = 3, 4, 2


def req_gid(session, step, attempt):
    return (session << 8) | (step * 16 + attempt)


# --- a hand-built diamond: X calls A and B, both call C; B also calls D on a non-critical edge ----
X, A, B, C, D = 0, 1, 2, 3, 4


class ToyRuns:
    callers = {C: {A: True, B: True}, D: {B: False}, A: {X: True}, B: {X: True}, X: {}}
    callees = defaultdict(set, {A: {C}, B: {C, D}, X: {A, B}})


class ToyRun:
    """Two requests of one session. Request 1: C fails twice (retried), so A, B and X fail and the
    client errors. Request 2: D is slow, which makes B slow through the non-critical edge."""

    def __init__(self):
        self.r1, self.r2 = req_gid(7, 0, 0), req_gid(7, 1, 0)
        self.c_fails = {self.r1: True, self.r2: False}
        self.d_slow = {self.r1: False, self.r2: True}
        self.hops = {g: self._tree(g, frozenset()) for g in (self.r1, self.r2)}
        self.clients = {(7, 0, 0): (1, False, 0), (7, 1, 0): (0, False, 0)}
        self.shard_runs = []

    def _tree(self, g, forced):
        out = {}
        if self.c_fails[g]:
            out[(C, 0)] = OK if (SRC_ATTEMPT0, C) in forced else E5
            if out[(C, 0)] != OK:
                out[(C, 1)] = E5
        else:
            out[(C, 0)] = OK
        c_final = OK if (SRC_FINAL, C) in forced else out[max(k for k in out if k[0] == C)]
        d_final = OK if (SRC_FINAL, D) in forced else (SLOW if self.d_slow[g] else OK)
        out[(D, 0)] = SLOW if self.d_slow[g] else OK
        out[(A, 0)] = ERR if c_final != OK else OK
        out[(B, 0)] = ERR if c_final != OK else (SLOW if d_final == SLOW else OK)
        a_final = OK if (SRC_FINAL, A) in forced else out[(A, 0)]
        b_final = OK if (SRC_FINAL, B) in forced else out[(B, 0)]
        out[(X, 0)] = ERR if ERR in (a_final, b_final) else OK
        return out

    def bff_of(self, g):
        return X

    def evaluate(self, req_gids, forced, narrow, detail):
        return {g: self._tree(g, forced) for g in req_gids}

    def client_under(self, step, attempt, y):
        self.shard_runs.append((step, attempt, y))
        return {7: 0 if y == OK else 1}


def test_diamond_single_contingent_retry_and_time_path():
    run = ToyRun()
    edges, _ = request_edges(run, ToyRuns(), [run.r1, run.r2])
    got = {(p["src"], p["dst"]): p for p in edges[run.r1]}
    # the shared callee reaches both callers, from its final attempt and from the attempt it retried
    for src, source in (((C, 1), SRC_FINAL), ((C, 0), SRC_ATTEMPT0)):
        for caller in (A, B):
            p = got[(src, (caller, 0))]
            assert (p["column"], p["single"], p["source"]) == (COL_DIRECTED, True, source)
        # X is not a caller of C: an ancestor pair, and through the edge the client error too
        assert got[(src, (X, 0))]["column"] == COL_ANCESTOR
        assert got[(src, CLIENT_ERR)]["column"] == COL_ANCESTOR
    # A and B both reach X: forcing one alone leaves X failed, forcing it with the other forced does not
    for callee in (A, B):
        p = got[((callee, 0), (X, 0))]
        assert (p["column"], p["single"], p["contingent"]) == (COL_DIRECTED, False, True)
    # the edge and its own client error
    p = got[((X, 0), CLIENT_ERR)]
    assert (p["column"], p["source"]) == (COL_DIRECTED, SRC_EDGE)
    assert len(got) == 11
    # request 2: the time path on a non-critical edge; X stayed OK and is never a target
    assert [(p["src"], p["dst"], p["column"], p["critical"]) for p in edges[run.r2]] == [
        ((D, 0), (B, 0), COL_DIRECTED, False)
    ]
    # no pair has an OK target, and no source is OK
    for g in (run.r1, run.r2):
        for p in edges[g]:
            assert run.hops[g][p["src"]] != OK
            assert p["dst"] == CLIENT_ERR or run.hops[g][p["dst"]] != OK
    # shard re-runs were asked only for request 1's own client attempt
    assert {(s, a) for s, a, _y in run.shard_runs} == {(0, 0)}


# --- which pairs are scored ------------------------------------------------------------------------


def test_reduce_candidates_drops_ok_targets_unmapped_positions_and_client_retries():
    tok = {0: "3:5xx", 1: "2:err", 2: "1:ok", 3: "0:err", 4: "0:err", 5: "9:err"}
    first, retry, later = req_gid(7, 0, 0), req_gid(7, 0, 1), req_gid(7, 1, 0)
    req = [first, first, first, first, retry, None]
    cands = [(0, 1), (0, 2), (1, 3), (3, 4), (0, 5), (4, 5)]
    kept, dropped = reduce_candidates(cands, tok, req)
    assert kept == [(0, 1), (1, 3)]
    assert dropped == {DROP_OK_TARGET: 1, DROP_UNMAPPED: 2, DROP_CLIENT_RETRY: 1}
    # another step of the same session is not a client retry
    assert reduce_candidates([(0, 1)], {0: "0:err", 1: "0:err"}, [first, later])[0] == [(0, 1)]


class TinyVocab:
    names = {4: "3:5xx", 5: "2:err", 6: "1:ok", 7: "0:err"}

    def is_real(self, tid):
        return tid in self.names

    def token_string(self, tid):
        return self.names[tid]


def _sequence(c, max_len):
    """A row of six positions read with `max_len` (so truncated) and context `c`."""
    ops_tokens = [4, 5, 6, 7, 5, 7]
    ids = np.array([1, *ops_tokens[:max_len], 2])
    tok = window_tokens(ids, c, len(ops_tokens), TinyVocab())
    return ids, tok, len(ids) - c


def test_window_index_is_view_position_c_plus_t_minus_one_after_truncation():
    ids, tok, lc = _sequence(c=3, max_len=5)
    # window index t is view position c + t - 1: positions 2, 3, 4 of the row; position 5 was cut
    assert tok == {0: "1:ok", 1: "0:err", 2: "2:err"}
    assert lc == 4 and 3 not in tok  # the last window index is the end marker


def test_truth_file_gate_refuses_an_off_by_one_window_index():
    ids, tok, lc = _sequence(c=3, max_len=5)
    base = {"candidates": [(0, 1), (0, 2), (1, 2)], "n_window": lc}
    req = [req_gid(7, 0, 0)] * lc
    good = {"seq": 0, "req": req,
            "pairs": [[1, 2, COL_DIRECTED, True, False, SRC_FINAL, True, "0:err", "2:err"]]}  # fmt: skip
    truth = forced_ok_truth(base, good, tok, set())
    assert truth["directed"] == {(1, 2)} and truth["candidates"] == [(0, 1), (0, 2), (1, 2)]
    # the same pair written one window index early (view position c + t instead of c + t - 1)
    shifted = {
        **good,
        "pairs": [[0, 1, COL_DIRECTED, True, False, SRC_FINAL, True, "0:err", "2:err"]],
    }
    with pytest.raises(TruthFileRefusal, match="window index"):
        forced_ok_truth(base, shifted, tok, set())
    # a pair that is not a candidate, a pair on an OK target, a wrong number of window positions
    with pytest.raises(TruthFileRefusal, match="not a candidate"):
        forced_ok_truth({**base, "candidates": [(0, 1)]}, good, tok, set())
    ok_target = {
        **good,
        "pairs": [[1, 0, COL_DIRECTED, True, False, SRC_FINAL, True, "0:err", "1:ok"]],
    }
    with pytest.raises(TruthFileRefusal):
        forced_ok_truth(base, ok_target, tok, set())
    with pytest.raises(TruthFileRefusal, match="window positions"):
        forced_ok_truth(base, {**good, "req": req[:-1]}, tok, set())


def test_an_edge_marked_on_an_ok_target_changes_no_pooled_count():
    tok = {0: "3:5xx", 1: "1:ok", 2: "0:err"}
    base = {"candidates": [(0, 1), (0, 2), (1, 2)], "n_window": 4}
    entry = {"seq": 0, "req": [req_gid(7, 0, 0)] * 4,
             "pairs": [[0, 2, COL_DIRECTED, True, False, SRC_FINAL, True, "3:5xx", "0:err"]]}  # fmt: skip
    truth = {0: forced_ok_truth(base, entry, tok, set())}
    assert truth[0]["candidates"] == [(0, 2), (1, 2)] and truth[0]["dropped"][DROP_OK_TARGET] == 1

    def pooled(marked):
        matrix = np.zeros((4, 4))
        for j, q in marked:
            matrix[j, q] = 1.0
        agg = aggregate_read([0], [np.arange(6)], truth, lambda s, ids: matrix,
                             lambda m, cands: {(j, q) for j, q in cands if m[j, q] > 0.5})  # fmt: skip
        return (
            agg["pooled"]["directed"],
            agg["pooled"]["n_candidates"],
            agg["predict_all"]["pooled"],
        )

    assert pooled([(0, 2), (0, 1)]) == pooled([(0, 2)])
    assert pooled([(0, 2)])[1] == 2


# --- a forced shard re-run in which a client retry vanishes -----------------------------------------


def test_forced_session_rerun_is_joined_on_identity_when_a_client_retry_vanishes():
    runs = CorpusRuns(ENGINE_FIXTURE)
    found = None
    for k in sorted(runs.bounds):
        run = runs.shard(k)
        for gid, reqs in run.requests_of.items():
            have = {(step_of(g), client_attempt_of(g)) for g in reqs}
            first = req_gid(gid, 0, 0)
            if (
                {(0, 0), (0, 1), (1, 0)} <= have
                and run.clients[(gid, 0, 0)][0] == 1
                and run.hops[first][(run.bff_of(first), 0)] != OK
                and run.client_under(0, 0, OK)[gid] == 0
            ):
                found = (k, gid)
                break
        if found:
            break
    assert found, "the fixture holds no session whose failed first step is retried and goes on"
    k, gid = found
    run = runs.shard(k)
    first, retry, second = req_gid(gid, 0, 0), req_gid(gid, 0, 1), req_gid(gid, 1, 0)
    n_scen = len(runs.inst.sset.scenarios)
    forced = runs.run_shard(k, {(s, 0, 0): OK for s in range(n_scen)})
    f = forced.requests.cols
    forced_row = {int(g): i for i, g in enumerate(f["req_gid"])}
    # the retry vanished, the second step survived, and the engine's row numbers shifted under it
    assert retry not in forced_row and second in forced_row
    assert forced_row[second] != run.row_of[second]
    # only the forced client attempt is read from the re-run: one outcome per session that has it
    read = run.client_under(0, 0, OK)
    assert set(read) == {g for (g, j, a) in run.clients if (j, a) == (0, 0)}
    # the one change the edge forcing reports is the first attempt's own client error ...
    edges, _ = request_edges(run, runs, [first])
    assert set(edges) == {first}
    from_edge = [p for p in edges[first] if p["source"] == SRC_EDGE]
    assert [(p["src"], p["dst"], p["column"]) for p in from_edge] == [
        ((run.bff_of(first), 0), CLIENT_ERR, COL_DIRECTED)
    ]
    # ... every pair stays inside that request: targets are its own attempts or its own client error
    for p in edges[first]:
        assert p["dst"] == CLIENT_ERR or p["dst"] in run.hops[first]
        assert run.hops[first].get(p["dst"], ABSENT) != OK
    # and no pair between the two client attempts of the step is a candidate; the next step's are
    tok = {0: "0:err", 1: "0:err", 2: "0:err"}
    kept, dropped = reduce_candidates([(0, 1), (0, 2), (1, 2)], tok, [first, retry, second])
    assert kept == [(0, 2), (1, 2)] and dropped[DROP_CLIENT_RETRY] == 1
    assert session_of(first) == session_of(retry) == gid
