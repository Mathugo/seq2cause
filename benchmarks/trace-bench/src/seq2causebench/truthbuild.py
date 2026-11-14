"""Build the `forced-ok-counterfactual` truth file of one landed read (score side).

For the stored sequences of a discover read, re-derived from its recorded selection:

1. map every view row to its request (request grain: the trace id is a hash of the request id) or
   its session (session grain: the sequence id is a hash of the session id), and every position to
   an identity — (request, operation, attempt) for an operation position, (request, error |
   transaction) for a client record. Request membership comes from this mapping, never from the
   view's root links;
2. re-run every touched shard and refuse unless it equals the corpus's oracle tables;
3. force each non-OK source to OK (`counterfactual.request_edges`) and keep the pairs whose target
   changes;
4. write, per stored sequence, the request id of every window position and the truth pairs in
   window indices (window index `t` is view position `c + t - 1` after truncation).

The corpus must hold its oracle tables, `config.yaml`, `constants.json` and `instantiation.json`
besides the score tier. No method is run and no parent link decides a truth pair.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from tracebench.correlate.__main__ import _window_epoch_ms
from tracebench.rng import stable_hex, stable_id, stable_uuid
from tracebench.score import truth_sets

from .constants import RESULTS_JSON, RUN_DIR
from .corpus import Corpus
from .counterfactual import (
    CLIENT_ERR,
    CLIENT_TXN,
    COL_ANCESTOR,
    COL_DIRECTED,
    OK,
    SRC_ATTEMPT0,
    SRC_FINAL,
    CorpusRuns,
    CounterfactualRefusal,
    client_attempt_of,
    request_edges,
    session_of,
    shard_of,
    step_of,
)
from .data import SequenceStore
from .floorseq import stored_sequences
from .log import log
from .record import RunRecord, read_json, write_json
from .seqtruth import (
    DROP_REASONS,
    RULE_FORCED_OK,
    TRUTH_FILE_FMT,
    TRUTH_FILE_SCHEMA,
    induce,
    is_ok_token,
    load_targets,
    reduce_candidates,
    roots,
    same_step_other_attempt,
    window_tokens,
)
from .vocab import Vocab

ROW_COLUMNS = ("trace_id", "ops", "outcomes", "offsets", "durations", "parent_pos", "start_ns")
SHARDS_ALL, SHARDS_AUTO = "all", "auto"
SHARD_SLACK_TICKS = 120  # clock skew and the client record's lead, on either side
ATTR_SESSION = 0
# why a re-run pair is not scored: its source or target record is not in the row (a client record
# the correlator filed elsewhere), a position's view outcome differs from the engine's, the source
# is ordered after the target, a position is outside the window (context, truncation, a folded
# token), or the pair is not a candidate (token pair outside the grain's universe, lag)
LOST_REASONS = ("not_in_row", "outcome_mismatch", "reversed", "outside_window", "not_candidate")


def shards_of_sample(runs, corpus_dir, starts_ns):
    """The shards the sampled rows can come from. A session lives in the shard of its arrival
    tick and may run on for the spill allowance, so a row that starts at tick `t` belongs to a
    shard between `t - spill` and `t`."""
    cfg = runs.inst.cfg
    with open(Path(corpus_dir) / "config.yaml", encoding="utf-8") as f:
        epoch_ms = _window_epoch_ms(yaml.safe_load(f))
    tick_ms, per_shard = cfg.run.tick_s * 1000, cfg.run.shard_ticks
    need = set()
    for start in starts_ns:
        tick = (int(start) // 1_000_000 - epoch_ms) / tick_ms
        lo = max(0, int((tick - runs.spill - SHARD_SLACK_TICKS) // per_shard))
        need.update(range(lo, int((tick + SHARD_SLACK_TICKS) // per_shard) + 1))
    return sorted(k for k in need if k in runs.bounds)


def request_ids(corpus_dir, seed, shards):
    """`{trace id: request id}` and `{sequence id: session id}` of the given shards, from the oracle."""
    by_trace, by_sequence = {}, {}
    for k in shards:
        odir = Path(corpus_dir) / "oracle" / f"shard={int(k):04d}"
        for g in set(
            pq.read_table(odir / "spans.parquet", columns=["req_gid"])["req_gid"].to_pylist()
        ):
            by_trace[stable_hex(seed, "trace", int(g), length=32)] = int(g)
        for g in pq.read_table(odir / "sessions.parquet", columns=["gid"])["gid"].to_pylist():
            by_sequence[
                stable_id("seq", ATTR_SESSION, stable_uuid(seed, "session", int(g)), "")
            ] = int(g)
    return by_trace, by_sequence


def map_request(run, runs, req, row, positions, ident, mismatch):
    """Assign the operation positions of one request: the k-th position of an operation in start
    order is its attempt k. A position the engine has no attempt for stays unmapped."""
    ops, outcomes, offsets, durations = (
        row["ops"],
        row["outcomes"],
        row["offsets"],
        row["durations"],
    )
    hops = run.hops[req]
    by_op = defaultdict(list)
    for o in positions:
        by_op[ops[o]].append(o)
    for op, pos in by_op.items():
        pos.sort(key=lambda o: (offsets[o], offsets[o] + durations[o], o))
        for k, o in enumerate(pos):
            if (op, k) in hops:
                ident[o] = (req, (op, k))
                mismatch[o] = hops[(op, k)] != outcomes[o]


def client_kind(outcome):
    return CLIENT_TXN if outcome == OK else CLIENT_ERR


def client_records(run, reqs):
    """`{kind: {request: true timestamp ms}}` of the client records the engine emitted for these
    requests: an error record when the client attempt failed, a transaction record when sampled."""
    out = {CLIENT_ERR: {}, CLIENT_TXN: {}}
    for g in reqs:
        outcome, sampled, ts = run.clients[(session_of(g), step_of(g), client_attempt_of(g))]
        if outcome == 1:
            out[CLIENT_ERR][g] = ts
        if sampled:
            out[CLIENT_TXN][g] = ts
    return out


def map_clients(run, reqs, row, cli_pos, ident, stats):
    """Assign the client positions of a row. A row holds a subset of the engine's client records
    (the correlator files the ones it cannot tie to a session elsewhere), so counts are not
    compared. An attached record — the parent of an edge position — belongs to that edge's
    request. A standalone one is placed by time: its end minus the engine's true timestamp is one
    constant per session (the browser clock skew is one draw per session), read off the attached
    records of the same row; without an attached record, equal counts per kind are zipped in time
    order. Anything else stays unmapped."""
    ops, outcomes = row["ops"], row["outcomes"]
    end_ms = {o: (row["offsets"][o] + row["durations"][o]) // 1_000_000 for o in cli_pos}
    records = client_records(run, reqs)
    used, consts = set(), Counter()
    for e, p in enumerate(row["parent_pos"]):
        if p < 0 or ops[p] not in run.runs.client_ops or ident[e] is None or ident[p] is not None:
            continue
        g, kind = ident[e][0], client_kind(outcomes[p])
        if g in records[kind] and (g, kind) not in used:
            ident[p] = (g, kind)
            used.add((g, kind))
            consts[end_ms[p] - records[kind][g]] += 1
        else:
            stats["client_attached_without_record"] += 1
    left = [o for o in cli_pos if ident[o] is None]
    if consts:
        const = consts.most_common(1)[0][0]
        stats["rows_client_skew_not_constant"] += len(consts) > 1
        for o in left:
            kind = client_kind(outcomes[o])
            hit = [
                g
                for g, ts in records[kind].items()
                if ts == end_ms[o] - const and (g, kind) not in used
            ]
            if len(hit) == 1:
                ident[o] = (hit[0], kind)
                used.add((hit[0], kind))
    else:
        for kind in (CLIENT_ERR, CLIENT_TXN):
            pos = sorted(
                (o for o in left if client_kind(outcomes[o]) == kind), key=lambda o: (end_ms[o], o)
            )
            free = sorted((ts, g) for g, ts in records[kind].items() if (g, kind) not in used)
            if pos and len(pos) == len(free):
                for o, (_ts, g) in zip(pos, free, strict=True):
                    ident[o] = (g, kind)


def map_row(runs, grain, by_trace, by_sequence, row, stats):
    """`(identity per position | None, outcome-mismatch flag per position)` of one view row."""
    ops = row["ops"]
    n = len(ops)
    ident, mismatch = [None] * n, [False] * n
    hop_pos = [o for o in range(n) if ops[o] not in runs.client_ops]
    cli_pos = [o for o in range(n) if ops[o] in runs.client_ops]
    if grain == "request":
        req = by_trace.get(row["trace_id"])
        if req is None:
            stats["rows_unmapped"] += 1
            return ident, mismatch
        run = runs.shard(shard_of(req))
        map_request(run, runs, req, row, hop_pos, ident, mismatch)
        reqs = [req]
        trees = None
    else:
        gid = by_sequence.get(row["trace_id"])
        if gid is None:
            stats["rows_unmapped"] += 1
            return ident, mismatch
        run = runs.shard(gid >> 32)
        reqs = run.requests_of[gid]
        root = roots(row["parent_pos"])
        trees = defaultdict(list)
        for o in hop_pos:
            trees[root[o]].append(o)
        ordered = []
        for pos in trees.values():
            edges = [o for o in pos if ops[o] in runs.bff_ops]
            if len(edges) == 1:
                ordered.append((row["offsets"][edges[0]], edges[0], pos))
        ordered.sort()
        if len(ordered) == len(reqs) == len(trees) and all(
            ops[e] == run.bff_of(g) for (_, e, _), g in zip(ordered, reqs, strict=True)
        ):
            for (_, _, pos), g in zip(ordered, reqs, strict=True):
                map_request(run, runs, g, row, pos, ident, mismatch)
        else:
            stats["rows_request_trees_unmatched"] += 1
    map_clients(run, reqs, row, cli_pos, ident, stats)
    stats["positions"] += n
    stats["positions_unmapped"] += sum(1 for x in ident if x is None)
    stats["positions_outcome_mismatch"] += sum(mismatch)
    stats["client_positions"] += len(cli_pos)
    stats["client_positions_unmapped"] += sum(1 for o in cli_pos if ident[o] is None)
    return ident, mismatch


def run_truthbuild(args, rec):
    run_dir = Path(args.run_dir)
    disc = read_json(run_dir / RUN_DIR / RESULTS_JSON)
    args_rec = read_json(run_dir / RUN_DIR / "arguments.json")["arguments"]
    if disc.get("status") != "ok":
        raise ValueError(f"{run_dir} is not a completed discover run")
    grain, split, c = str(disc["grain"]), str(disc["split"]), int(disc["c"])
    ordering, max_len, max_lag = (
        str(args_rec["ordering"]),
        int(args_rec["max_len"]),
        int(args.max_lag),
    )
    corpus = Corpus(args.corpus, ordering, grain)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    store = SequenceStore.from_corpus(
        corpus,
        split,
        vocab,
        max_len,
        n=int(args_rec["num_sequences"]),
        mode=str(args_rec["sequence_sample"]),
        seed=int(args_rec["seed"]),
    )
    stored, n_skipped = stored_sequences(store, c, int(args_rec["cells_sequences"]))
    got = {"n_sequences": len(store), "n_skipped_short": n_skipped, "n_stored": len(stored)}
    for key, v in got.items():
        if disc.get(key) != v:
            raise ValueError(f"re-derived {key} = {v}, the read recorded {disc.get(key)}")
    wanted = {store.trace_ids[s] for s in stored}
    rows = {}
    for values in corpus.sequences(split, columns=ROW_COLUMNS):
        if values[0] in wanted:
            rows[values[0]] = dict(
                zip(
                    ROW_COLUMNS, (values[0], *map(list, values[1:-1]), int(values[-1])), strict=True
                )
            )
            if len(rows) == len(wanted):
                break

    runs = CorpusRuns(args.corpus)
    if args.shards == SHARDS_ALL:
        shards = sorted(runs.bounds)
    elif args.shards == SHARDS_AUTO:
        shards = shards_of_sample(runs, args.corpus, [r["start_ns"] for r in rows.values()])
    else:
        shards = [int(x) for x in args.shards.split(",")]
    checks = {int(k): runs.check_against_oracle(k) for k in shards}  # identity check 1
    by_trace, by_sequence = request_ids(args.corpus, runs.seed, shards)
    targets = load_targets(args.corpus, grain)
    population_any, _ = truth_sets(targets["target"], 0.0)
    population_any = set(map(tuple, population_any))

    stats = Counter()
    mapped = {
        s: map_row(runs, grain, by_trace, by_sequence, rows[store.trace_ids[s]], stats)
        for s in stored
    }
    if grain == "request" and stats["rows_unmapped"]:
        raise CounterfactualRefusal(
            f"{stats['rows_unmapped']} request rows have no request in shards {shards}; "
            f"pass --shards {SHARDS_ALL}"
        )
    needed = defaultdict(set)  # shard -> requests with a mapped position
    for ident, _ in mapped.values():
        for x in ident:
            if x is not None:
                needed[shard_of(x[0])].add(x[0])
    edges, n_runs = {}, Counter()
    for k, reqs in sorted(needed.items()):
        run = runs.shard(k)
        reqs = sorted(reqs)
        # identity check 2: the narrowed evaluation of these requests equals the shard re-run
        if run.evaluate(reqs, frozenset(), True, False) != {g: dict(run.hops[g]) for g in reqs}:
            raise CounterfactualRefusal(
                f"shard {k}: the narrowed evaluation differs from the baseline"
            )
        # identity check 3: forcing an operation a request did not invoke changes nothing in it
        ops = sorted({op for g in reqs for op, _k in run.hops[g]} - runs.bff_ops)
        step = max(1, len(ops) // max(1, int(args.null_forcing_ops)))
        for w in ops[::step][: int(args.null_forcing_ops)]:
            sub = [g for g in reqs if all(op != w for op, _k in run.hops[g])]
            null = run.evaluate(sub, frozenset({(SRC_FINAL, w), (SRC_ATTEMPT0, w)}), False, False)
            if null != {g: dict(run.hops[g]) for g in sub}:
                raise CounterfactualRefusal(
                    f"shard {k}: forcing operation {w} changed requests without it"
                )
            n_runs["n_null_forcing_runs"] += 1
        got_edges, counters = request_edges(run, runs, reqs)
        edges.update(got_edges)
        n_runs.update(counters)

    sequences = []
    lost, dropped, flags, compare = Counter(), Counter(), Counter(), Counter()
    totals = Counter()
    for s in stored:
        row = rows[store.trace_ids[s]]
        ids = np.asarray(store.get(s))
        ident, mismatch = mapped[s]
        lc, kept_ops = len(ids) - c, min(len(row["ops"]), len(ids) - 2)
        tok = window_tokens(ids, c, len(row["ops"]), vocab)
        base = induce(
            ids, row["ops"], row["outcomes"], row["parent_pos"], c, vocab, grain, targets, max_lag
        )
        req = [
            ident[c + t - 1][0]
            if 0 <= c + t - 1 < kept_ops and ident[c + t - 1] is not None
            else None
            for t in range(lc)
        ]
        kept, drop = reduce_candidates(base["candidates"], tok, req)
        allowed = set(kept)
        dropped.update(drop)
        where = {x: o for o, x in enumerate(ident) if x is not None}
        pairs, new_directed = [], set()
        for g in sorted({x[0] for x in ident if x is not None}):
            for p in edges[g]:
                o_src, o_dst = where.get((g, p["src"])), where.get((g, p["dst"]))
                if o_src is None or o_dst is None:
                    lost[(p["column"], "not_in_row")] += 1
                    continue
                if mismatch[o_src] or mismatch[o_dst]:
                    lost[(p["column"], "outcome_mismatch")] += 1
                    continue
                j, q = o_src - c + 1, o_dst - c + 1
                if j > q:
                    lost[(p["column"], "reversed")] += 1
                elif j not in tok or q not in tok:
                    lost[(p["column"], "outside_window")] += 1
                elif (j, q) not in allowed:
                    lost[(p["column"], "not_candidate")] += 1
                else:
                    pairs.append([j, q, p["column"], p["single"], p["contingent"], p["source"],
                                  p["critical"], tok[j], tok[q]])  # fmt: skip
                    if p["column"] == COL_DIRECTED:
                        new_directed.add((j, q))
                        flags["single" if p["single"] else "contingent_only"] += 1
                        flags[f"source_{p['source']}"] += 1
                        flags[f"critical_{p['critical']}"] += 1
                        token_pair = (tok[j], tok[q])
                        flags["population_edge_at_floor"] += token_pair in targets["directed"]
                        flags["population_edge_any_strength"] += token_pair in population_any
                        if (
                            grain == "session"
                            and roots(row["parent_pos"])[o_src] != roots(row["parent_pos"])[o_dst]
                        ):
                            flags["within_request_by_oracle_cross_request_by_roots"] += 1
        pairs.sort()
        # old against new (check 7)
        for pair in base["directed"]:
            if pair in new_directed:
                compare["landed_kept"] += 1
            elif pair in allowed:
                compare["landed_lost_not_a_rerun_edge"] += 1
            elif is_ok_token(tok[pair[1]]):
                compare["landed_lost_ok_target"] += 1
            elif req[pair[0]] is None or req[pair[1]] is None:
                compare["landed_lost_unmapped"] += 1
            elif same_step_other_attempt(req[pair[0]], req[pair[1]]):
                compare["landed_lost_client_retry"] += 1
            else:
                raise AssertionError("a landed pair left the candidates for no listed reason")
        compare["new_gained"] += len(new_directed - base["directed"])
        healthy = all(o == OK for o in row["outcomes"])
        totals["sequences"] += 1
        totals["sequences_with_candidates_landed"] += bool(base["candidates"])
        totals["sequences_with_candidates"] += bool(kept)
        totals["sequences_without_candidates_fully_healthy"] += (not kept) and healthy
        totals["sequences_without_candidates_other"] += (not kept) and not healthy
        totals["sequences_with_truth"] += bool(new_directed)
        totals["candidates_landed"] += len(base["candidates"])
        totals["candidates"] += len(kept)
        totals["truth_directed_landed"] += len(base["directed"])
        totals["truth_directed"] += len(new_directed)
        sequences.append({"seq": int(s), "trace_id": store.trace_ids[s], "length": int(store.lengths[s]),
                          "healthy": healthy, "req": req, "pairs": pairs})  # fmt: skip

    doc = {
        "schema": TRUTH_FILE_SCHEMA,
        "rule": RULE_FORCED_OK,
        "corpus_id": disc["corpus_id"],
        "grain": grain,
        "split": split,
        "ordering": ordering,
        "c": c,
        "max_len": max_len,
        "max_lag": max_lag,
        "read": run_dir.name,
        "pair_fields": [
            "j",
            "q",
            "column",
            "single",
            "contingent",
            "source",
            "critical",
            "token_j",
            "token_q",
        ],
        "counts": {k: int(v) for k, v in sorted(totals.items())},
        "dropped_candidates": {k: int(dropped[k]) for k in DROP_REASONS},
        "rerun_pairs_not_scored": {
            col: {k: int(lost[(col, k)]) for k in LOST_REASONS}
            for col in (COL_DIRECTED, COL_ANCESTOR)
        },
        "directed_pairs": {k: int(v) for k, v in sorted(flags.items())},
        "old_against_new": {k: int(v) for k, v in sorted(compare.items())},
        "mapping": {k: int(v) for k, v in sorted(stats.items())},
        "reruns": {"shards": shards, **{k: int(v) for k, v in sorted(n_runs.items())}},
        "oracle_check": checks,
        "sequences": sequences,
    }
    name = TRUTH_FILE_FMT.format(grain=grain)
    write_json(rec.out_dir / name, doc)
    summary = {
        k: v for k, v in doc.items() if k not in ("sequences", "oracle_check", "pair_fields")
    }
    log({"event": "truthbuild_done", **summary})
    return {"truth_file": name, **summary}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corpus", required=True, help="a corpus with its oracle tables, config and score tier"
    )
    p.add_argument("--run-dir", required=True, help="the landed discover read whose sample is used")
    p.add_argument("--max-lag", required=True, type=int, help="candidate pairs within this lag")
    p.add_argument(
        "--shards",
        required=True,
        help="'all', 'auto' (the shards the sampled rows' start times allow), or a comma-separated list",
    )
    p.add_argument(
        "--null-forcing-ops",
        required=True,
        type=int,
        help="operations per shard on which the null-forcing check is run (evenly spaced)",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "truthbuild", vars(args)) as rec:
        res = run_truthbuild(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
