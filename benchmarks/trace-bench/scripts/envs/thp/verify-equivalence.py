"""Equivalence check for `0001-decomposable-score.patch` (plans/thp.md §9; RUN.md D-TH-7): run
gCastle's TTPM from one `castle/` tree (the pinned, unpatched clone or the patched install) on
one input and record its trace — the BIC score logged at the start of every sweep, the move each
sweep accepted, the final causal matrix, the sweep count and the fit's wall clock — as JSON; then
compare two traces. The shipped search exposes no moves, so they are reconstructed from its `_em`
calls (in `product` order): the accepted matrix of a sweep is the first candidate whose score
equals the next sweep's logged score, and the move is its difference from the sweep's start.

    <thp python> -I verify-equivalence.py run --castle-src <dir> --events <npz> --allowed <npy|none> \\
        --setting <decay_ms> --max-hop 1 --max-iter <n> --out <trace.json>
    <thp python> -I verify-equivalence.py run --castle-src <dir> --synthetic <seed> ... --out <trace.json>
    <thp python> -I verify-equivalence.py compare <a.json> <b.json>

Standard library, numpy, pandas and the `castle/` under `--castle-src` only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def synthetic(seed, n_types=8, n_nodes=6, n_edges=10, T=800000):
    """A small THPSimulation sample: a ring topology over the nodes, a random DAG over the types
    (edges from lower to higher index), events over [0, T]; returns (df, topology)."""
    import numpy as np
    from castle.datasets import THPSimulation

    rng = np.random.default_rng(seed)
    causal = np.zeros((n_types, n_types), dtype=int)
    pairs = [(a, b) for a in range(n_types) for b in range(a + 1, n_types)]
    for k in rng.choice(len(pairs), size=n_edges, replace=False):
        causal[pairs[k]] = 1
    topo = np.zeros((n_nodes, n_nodes), dtype=int)
    for a in range(n_nodes):
        topo[a, (a + 1) % n_nodes] = topo[(a + 1) % n_nodes, a] = 1
    np.random.seed(seed)
    df = THPSimulation(causal, topo).simulate(T, max_hop=1, beta=10)
    df = df.sort_values(["timestamp", "node", "event"], kind="stable").reset_index(drop=True)
    df["event"] = df["event"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["node"] = df["node"].astype(np.int64)
    return df, topo, causal


def edges_of(mat):
    import numpy as np

    m = np.asarray(mat)
    off = m.copy()
    np.fill_diagonal(off, 0)
    return sorted((int(a), int(b)) for a, b in zip(*np.nonzero(off > 0), strict=True))


def run(args):
    import numpy as np
    import pandas as pd

    sys.path.insert(0, args.castle_src)
    from castle.algorithms import TTPM
    from castle.common.priori_knowledge import PrioriKnowledge

    if args.synthetic is not None:
        df, topo, truth = synthetic(args.synthetic)
        allowed = None
        source = {"synthetic_seed": args.synthetic, "truth_edges": edges_of(truth)}
    else:
        with np.load(args.events) as z:
            df = pd.DataFrame(
                {
                    "event": z["event"].astype(np.int64),
                    "timestamp": z["timestamp"].astype(np.int64),
                    "node": z["node"].astype(np.int64),
                }
            )
            topo = np.asarray(z["op_topology"]).astype(np.int64)
        allowed = None
        if args.allowed != "none":
            allowed = {(int(a), int(b)) for a, b in np.load(args.allowed).reshape(-1, 2)}
        source = {"events": args.events, "allowed": args.allowed}
    n_types = int(df["event"].max()) + 1
    priori = None
    if allowed is not None:
        priori = PrioriKnowledge(n_types)
        priori.add_forbidden_edges(
            [
                (a, b)
                for a in range(n_types)
                for b in range(n_types)
                if a != b and (a, b) not in allowed
            ]
        )
        priori.forbidden_edges = set(priori.forbidden_edges)
    ttpm = TTPM(
        topo,
        delta=1.0 / args.setting,
        epsilon=1,
        max_hop=args.max_hop,
        penalty="BIC",
        max_iter=args.max_iter,
        priori_knowledge=priori,
    )

    # the sweep trace from the method's own log line, the candidates from its `_em` calls
    sweeps = []  # (num_iter, score repr)
    calls = []  # (sweep index, score repr, edges)

    class Catch(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if msg.startswith("[iter "):
                k, val = msg[len("[iter ") :].split("]: likelihood_score = ")
                sweeps.append((int(k), val))

    logging.getLogger().addHandler(Catch())
    em = ttpm._em

    def em_traced(edge_mat):
        out = em(edge_mat)
        calls.append((len(sweeps) - 1, repr(float(out[0])), edges_of(edge_mat)))
        return out

    ttpm._em = em_traced
    t0 = time.monotonic()
    ttpm.learn(df)
    wall = time.monotonic() - t0
    final = ttpm.causal_matrix
    final_edges = edges_of(final)
    n_sweeps = len(sweeps)
    moves = getattr(ttpm, "moves_", None)
    if moves is None:  # the shipped search: reconstruct the accepted move of every sweep
        moves = []
        start = edges_of(np.eye(n_types) + (priori.matrix == 1 if priori is not None else 0))
        for k, _score in sweeps:
            nxt = sweeps[k + 1][1] if k + 1 < len(sweeps) else None
            if nxt is None:
                # the last sweep: accepted iff the final graph differs from the start
                accepted = final_edges if final_edges != start else None
            else:
                accepted = next((e for s, sc, e in calls if s == k and sc == nxt), None)
            if accepted is None:
                break
            added = sorted(set(accepted) - set(start))
            removed = sorted(set(start) - set(accepted))
            if added and removed:
                kind, (j, i) = "reverse", added[0]
            elif added:
                kind, (j, i) = "add", added[0]
            else:
                kind, (j, i) = "delete", removed[0]
            moves.append((k, j, i, kind))
            start = accepted
    trace = {
        "castle_src": args.castle_src,
        "source": source,
        "n_types": n_types,
        "n_events": int(len(df)),
        "setting": args.setting,
        "max_hop": args.max_hop,
        "max_iter": args.max_iter,
        "sweep_scores": [s for _, s in sweeps],
        "moves": [list(m) for m in moves],
        "final_edges": final_edges,
        "n_edges": len(final_edges),
        "n_sweeps": n_sweeps,
        "n_em_calls": len(calls),
        "converged": bool(getattr(ttpm, "converged_", n_sweeps < args.max_iter)),
        "wall_seconds": wall,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=1)
    print(
        f"{args.castle_src}: {n_sweeps} sweeps, {len(final_edges)} edges, {len(calls)} _em calls, "
        f"{wall:.1f} s, converged={trace['converged']}"
    )
    return 0


def compare(args):
    a = json.load(open(args.a, encoding="utf-8"))
    b = json.load(open(args.b, encoding="utf-8"))
    n = min(len(a["sweep_scores"]), len(b["sweep_scores"]))
    same_scores = a["sweep_scores"][:n] == b["sweep_scores"][:n]
    m = min(len(a["moves"]), len(b["moves"]))
    same_moves = a["moves"][:m] == b["moves"][:m]
    same_final = a["final_edges"] == b["final_edges"] and a["n_sweeps"] == b["n_sweeps"]
    print(f"sweep scores identical over {n} sweeps: {same_scores}")
    print(f"accepted moves identical over {m} moves: {same_moves}")
    print(
        f"final graph identical ({a['n_edges']} vs {b['n_edges']} edges, "
        f"{a['n_sweeps']} vs {b['n_sweeps']} sweeps): {same_final}"
    )
    print(
        f"wall: {a['wall_seconds']:.1f} s vs {b['wall_seconds']:.1f} s; _em calls {a['n_em_calls']} vs {b['n_em_calls']}"
    )
    return (
        0 if (same_scores and same_moves and (same_final or a["max_iter"] != b["max_iter"])) else 1
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--castle-src", required=True)
    r.add_argument("--events")
    r.add_argument("--allowed", default="none")
    r.add_argument("--synthetic", type=int)
    r.add_argument("--setting", type=int, required=True)
    r.add_argument("--max-hop", type=int, default=1)
    r.add_argument("--max-iter", type=int, required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args(argv)
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
