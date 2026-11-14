"""Count the scored root-cause traces where the call-tree reference misses a shared callee (score side).

The reference row `floor/trace-topology` ranks, for every failed span, the services of its
call-tree subtree (`parent_pos` followed downwards). An operation with several callers has one
recorded parent, so a failed caller that is not that parent does not reach the shared callee.
This script recomputes rule R and the scored set with the benchmark's own functions, ranks every
scored trace twice — on the call tree (the landed rule) and on the call tree plus every
call-graph edge between two operations present in the trace — and counts the traces whose
closure, reached set, first-ranked service and hit@1 differ. It runs no method and writes one
JSON record; the tree-side hit rates are the landed reference's, for checking.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tracebench.score_rca_trace import (
    case_windows_ns,
    descendants,
    fault_index_of,
    hit,
    load_labels,
    load_ops,
    nonok_positions,
    rank_by_closure,
    select_traces,
    services_sorted,
    took_nonok,
    topology_rank,
)

SELECT = "post-nonok"  # rule R, uncapped: the per-trace axis's selection
READINGS = {"hop1": 0, "hop0": 1}


def graph_descendants(parent_pos, ops_row, edges, q):
    """`{position: hop >= 1}` below `q` when a position's children are its call-tree children plus
    every position whose operation it calls in the call graph."""
    children = defaultdict(set)
    for j, p in enumerate(parent_pos):
        if int(p) >= 0:
            children[int(p)].add(j)
    by_op = defaultdict(list)
    for j, op in enumerate(ops_row):
        by_op[op].append(j)
    out, seen, frontier, hop = {}, {q}, [q], 0
    while frontier:
        hop += 1
        nxt = []
        for u in frontier:
            kids = set(children.get(u, ()))
            for callee in edges.get(ops_row[u], ()):
                kids.update(by_op.get(callee, ()))
            for v in sorted(kids):
                if v not in seen:
                    seen.add(v)
                    out[v] = hop
                    nxt.append(v)
        frontier = nxt
    return out


def graph_rank(row, ops, client_ops, hop0, services_order, edges):
    """`topology_rank` with the call-graph children added; also whether any closure grew."""
    ops_row = [int(x) for x in row["ops"]]
    closures, grew = [], False
    for q in nonok_positions(row, client_ops):
        tree = descendants(row["parent_pos"], q)
        desc = graph_descendants(row["parent_pos"], ops_row, edges, q)
        grew = grew or set(desc) != set(tree)
        if hop0:
            desc[q] = 0
        svc = {}
        for p, h in desc.items():
            if ops_row[p] in client_ops:
                continue
            s = ops[ops_row[p]][0]
            svc[s] = min(h, svc.get(s, h))
        closures.append(svc)
    services = rank_by_closure(closures, services_order)
    return {"services": services, "set": sorted(services)}, grew


def run(args):
    corpus = Path(args.corpus)
    edges = defaultdict(set)  # caller op -> callee ops
    for e in json.loads((corpus / "instantiation.json").read_text())["topology"]["edges"]:
        edges[int(e["caller"])].add(int(e["callee"]))
    ops, client_ops = load_ops(corpus)
    order = services_sorted(corpus)
    labels = load_labels(corpus)
    windows = case_windows_ns(corpus)
    selected = select_traces(corpus, windows, client_ops, SELECT, 0)
    cases = []
    for case_dir, case, _, _ in windows:
        rc = str(case["root_cause_component"])
        fi = fault_index_of(case)
        c = {"case_dir": case_dir, "n": 0, "n_closure_grew": 0}
        for name in READINGS:
            c[name] = {"tree_s@1": 0, "graph_s@1": 0, "n_set_differs": 0, "n_top1_differs": 0}
        for row in selected[case_dir]:
            if not took_nonok(row, labels[(str(row["trace_id"]), fi)]):
                continue
            c["n"] += 1
            for name, hop0 in READINGS.items():
                tree = topology_rank(row, ops, client_ops, hop0, order)
                graph, grew = graph_rank(row, ops, client_ops, hop0, order, edges)
                k = c[name]
                k["tree_s@1"] += hit(tree["services"], {rc}, 1)
                k["graph_s@1"] += hit(graph["services"], {rc}, 1)
                k["n_set_differs"] += tree["set"] != graph["set"]
                k["n_top1_differs"] += tree["services"][:1] != graph["services"][:1]
                if hop0:
                    c["n_closure_grew"] += grew
        cases.append(c)
    return {
        "schema": "tracebenchrca/shared-callee-rca-count@1",
        "corpus": "/".join(corpus.parts[-3:]),
        "select": SELECT,
        "n_scored_traces": sum(c["n"] for c in cases),
        "n_closure_grew": sum(c["n_closure_grew"] for c in cases),
        "cases": cases,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="a corpus with its rca and trace tiers pulled")
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("n_")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
