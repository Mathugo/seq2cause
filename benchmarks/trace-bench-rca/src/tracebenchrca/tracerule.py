"""TRACE's own root-cause rule on the sample graphs of a case's window traces (method side; no model code).

`plans/rca-arms.md`, addendum 2026-10-30 (D-TBR-13): the lab's reading of the author's rule, as sent to him on
2026-10-21. For every probed trace, the sample graph is the frozen cut of the trace's score matrix — a directed
edge `j -> q` for every window position pair `j < q` with `S[j, q] > tau` (the runner writes the cut edges; this
module never sees a score). A **non-OK position** is a window position whose token's outcome is one of
`TRACE_NONOK_OUTCOMES`. Its **parents** are the positions `j < q` with an edge `j -> q`; its **upstream set** is
the transitive closure of the parent relation, with the hop distance = the shortest parent chain. Positions map
to services through the runner's per-position `(op, outcome, service, is_client)`; the synthetic client's
positions are dropped and counted. Per case, a service's score is the number of probed traces in which it is
upstream (hop >= 1) of at least one non-OK position; ties by the mean, over those traces, of the trace's smallest
hop to the service (smaller first), then `services.json` order; services never upstream follow in
`services.json` order. A case with no probed trace, no non-OK position or no upstream service is a counted miss.
The hop-0 counts (a non-OK position's own service) are recorded as `self_counts`, never ranked.

`rank_trace` is the per-trace axis's reading of the same rule (plan of 2026-10-31): inside ONE probed trace, every
reached service (and every reached window position) is ranked by the number of non-OK positions whose upstream
closure holds it, ties by the smallest hop, then `services.json` order (positions: by index). The `hop0` reading
adds each non-OK position itself, and its service, to its closure at hop 0; the `hop1` reading is upstream only.
Client positions are dropped and counted under both readings. `rank_case` is untouched.
"""

from __future__ import annotations

from collections import deque

from .constants import (
    TRACE_MISS_NO_NONOK,
    TRACE_MISS_NO_TRACE,
    TRACE_MISS_NO_UPSTREAM,
    TRACE_NONOK_OUTCOMES,
)


def parents_of(edges, n_positions):
    """`[[j, ...] for q in range(n)]` from cut edges `(j, q, score)`; refuses a non-backward edge."""
    parents = [[] for _ in range(n_positions)]
    for j, q, *_ in edges:
        j, q = int(j), int(q)
        if not 0 <= j < q < n_positions:
            raise ValueError(f"edge {j} -> {q} is not a backward-in-time edge inside {n_positions} positions")
        if j not in parents[q]:
            parents[q].append(j)
    return parents


def upstream(parents, q):
    """`{position: hop}` of every position upstream of `q` (hop >= 1; the shortest parent chain)."""
    hops = {}
    queue = deque([(q, 0)])
    while queue:
        p, h = queue.popleft()
        for j in parents[p]:
            if j not in hops:
                hops[j] = h + 1
                queue.append((j, h + 1))
    return hops


def nonok_positions(positions, nonok=TRACE_NONOK_OUTCOMES):
    """Indices of the real window positions whose outcome is non-OK (a special token has no entry)."""
    return [t for t, pos in enumerate(positions) if pos is not None and pos["outcome"] in nonok]


def trace_upstream_services(trace, nonok=TRACE_NONOK_OUTCOMES):
    """One probed trace: `(reached {service: min hop}, self_services, n_nonok, n_client_dropped)`.
    `trace["positions"][t]` is `{"op", "outcome", "service", "is_client"}` or None for a special token;
    `trace["edges"]` the cut edges `[j, q, score]`."""
    positions = trace["positions"]
    parents = parents_of(trace["edges"], len(positions))
    reached, selfs, n_client = {}, set(), 0
    nonok_idx = nonok_positions(positions, nonok)
    for q in nonok_idx:
        own = positions[q]
        if not own["is_client"]:
            selfs.add(own["service"])
        for p, h in upstream(parents, q).items():
            pos = positions[p]
            if pos is None:  # the runner never cuts an edge at a special token; refuse rather than skip silently
                raise ValueError(f"edge into a special token at window position {p}")
            if pos["is_client"]:
                n_client += 1
                continue
            s = pos["service"]
            if s not in reached or h < reached[s]:
                reached[s] = h
    return reached, selfs, len(nonok_idx), n_client


def rank_trace(trace, services, nonok=TRACE_NONOK_OUTCOMES, hop0=False):
    """One probed trace's ranking of its reached services and window positions (the per-trace axis).
    `{"services": [reached, ranked], "positions": [reached window positions, ranked], "n_nonok", "n_client_dropped"}`;
    an item's statistic is `(number of non-OK positions whose closure holds it, min hop)`, ties by `services` order
    (positions by index). `hop0=True` puts each non-OK position itself (and its service) into its closure at hop 0.
    Services the view names outside `services` are refused."""
    positions = trace["positions"]
    parents = parents_of(trace["edges"], len(positions))
    order = {s: i for i, s in enumerate(services)}
    svc_count, svc_hop, pos_count, pos_hop = {}, {}, {}, {}
    n_client = 0
    nonok_idx = nonok_positions(positions, nonok)
    for q in nonok_idx:
        closure = upstream(parents, q)
        if hop0:
            closure[q] = 0
        held = {}  # service -> its smallest hop inside this closure
        for p, h in closure.items():
            pos = positions[p]
            if pos is None:  # the runner never cuts an edge at a special token; refuse rather than skip silently
                raise ValueError(f"edge into a special token at window position {p}")
            if pos["is_client"]:
                n_client += 1
                continue
            pos_count[p] = pos_count.get(p, 0) + 1
            pos_hop[p] = min(h, pos_hop.get(p, h))
            s = pos["service"]
            if s not in order:
                raise ValueError(f"the view names a service outside services.json: {s!r}")
            held[s] = min(h, held.get(s, h))
        for s, h in held.items():
            svc_count[s] = svc_count.get(s, 0) + 1
            svc_hop[s] = min(h, svc_hop.get(s, h))
    ranked_services = sorted(svc_count, key=lambda s: (-svc_count[s], svc_hop[s], order[s]))
    ranked_positions = sorted(pos_count, key=lambda p: (-pos_count[p], pos_hop[p], p))
    return {
        "services": ranked_services,
        "positions": ranked_positions,
        "n_nonok": len(nonok_idx),
        "n_client_dropped": n_client,
    }


def rank_case(traces, services, nonok=TRACE_NONOK_OUTCOMES):
    """The service ranking of one case from its probed traces (the runner's per-trace records) over the
    deployed services (`services.json` order). Returns the record fields of the case, with `services` None
    and a `miss` reason when the rule has no evidence."""
    order = {s: i for i, s in enumerate(services)}
    counts = dict.fromkeys(services, 0)
    hops = {s: [] for s in services}
    self_counts = dict.fromkeys(services, 0)
    n_nonok = n_client = n_with_nonok = n_with_upstream = n_edges = 0
    for tr in traces:
        n_edges += len(tr["edges"])
        reached, selfs, k, c = trace_upstream_services(tr, nonok)
        n_nonok += k
        n_client += c
        n_with_nonok += k > 0
        n_with_upstream += bool(reached)
        unknown = sorted((set(reached) | selfs) - set(services))
        if unknown:
            raise ValueError(f"the view names services outside services.json: {unknown}")
        for s, h in reached.items():
            counts[s] += 1
            hops[s].append(h)
        for s in selfs:
            self_counts[s] += 1
    facts = {
        "n_traces_probed": len(traces),
        "n_edges": n_edges,
        "n_nonok_positions": n_nonok,
        "n_traces_with_nonok": n_with_nonok,
        "n_traces_with_upstream": n_with_upstream,
        "n_client_positions_dropped": n_client,
        "n_services_reached": sum(1 for s in services if counts[s] > 0),
        "counts": {s: counts[s] for s in services if counts[s] > 0},
        "mean_hops": {s: sum(hops[s]) / len(hops[s]) for s in services if hops[s]},
        "self_counts": {s: self_counts[s] for s in services if self_counts[s] > 0},
    }
    if not traces:
        return {**facts, "services": None, "miss": TRACE_MISS_NO_TRACE}
    if n_nonok == 0:
        return {**facts, "services": None, "miss": TRACE_MISS_NO_NONOK}
    if facts["n_services_reached"] == 0:
        return {**facts, "services": None, "miss": TRACE_MISS_NO_UPSTREAM}
    reached = [s for s in services if counts[s] > 0]
    reached.sort(key=lambda s: (-counts[s], facts["mean_hops"][s], order[s]))
    rest = [s for s in services if counts[s] == 0]
    return {**facts, "services": reached + rest, "miss": None}


__all__ = ["parents_of", "upstream", "nonok_positions", "trace_upstream_services", "rank_trace", "rank_case"]
