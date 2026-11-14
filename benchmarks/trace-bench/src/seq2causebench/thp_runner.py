"""Run THP (gCastle's `TTPM`, Cai et al. 2024) once on an events file (plans/thp.md §2; D-SB-29).
Executed by the method's own interpreter (the `thp` environment), never imported by the harness:

    <thp python> -I thp_runner.py --method-src <site-packages> --events <events.npz> \\
        --allowed <allowed.npy | none> --setting <decay_ms> \\
        --method-args max_hop=1 penalty=BIC max_iter=5000 epsilon=1 --out <scores.npy>

`events.npz` holds `event` (dense type ids, int64), `timestamp` (integer milliseconds, sorted),
`node` (the dense op id of every event: node = op), `n_ops` and `op_topology` (the undirected
int8 op adjacency the allowed pairs span). The event type is TTPM's `event`, the op its `node`,
`op_topology` its `topology_matrix`; `delta = 1 / setting` (the exponential kernel's decay, per
millisecond; `setting` = `decay_ms`, the registered `N` slot). `allowed.npy` is an int64 (k, 2)
array of allowed (cause, effect) type ids: every other off-diagonal pair is handed to TTPM as a
forbidden edge of a `PrioriKnowledge`, so the hill-climb never adds it; the word `none` passes no
prior. Writes the DAG TTPM returns (row = cause; diagonal stripped) as a float32 0/1 score matrix
and a sidecar JSON with the counts, the wall clock, the setting, the method's version and commit
and `method_env` (the `pip freeze` lines of this interpreter, carried into the read's
`run_meta.json`), plus `n_sweeps_run`, `converged` and `max_iter` from the patched search
(`scripts/envs/thp/0001-decomposable-score.patch`, which `--method-src` must carry: the runner
refuses an unpatched TTPM). Standard library, numpy and pandas only (plus the method under
`--method-src`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

# the plan's fixed settings (plans/thp.md §3); a `--method-args` token overrides its key.
# `max_iter` is a safety cap: the hill-climb stops by itself when no move improves the score
# (one accepted move per sweep, so the cap is also a cap on the edge count; D-TH-8)
DEFAULT_METHOD_ARGS = {"max_hop": "1", "penalty": "BIC", "max_iter": "5000", "epsilon": "1"}
PENALTIES = ("BIC", "AIC")


def pip_freeze():
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        return [ln for ln in out.splitlines() if ln.strip()] or ["<empty pip freeze>"]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [f"<pip freeze unavailable: {type(e).__name__}>"]


def method_commit(method_src):
    """The pinned commit under `method_src`: a pip install from a git URL records it in
    `<dist>.dist-info/direct_url.json`; a clone answers `git rev-parse HEAD`; else 'unknown'."""
    for p in sorted(glob.glob(os.path.join(method_src, "*.dist-info", "direct_url.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                commit = json.load(f).get("vcs_info", {}).get("commit_id")
        except (OSError, ValueError):
            continue
        if commit:
            return str(commit)
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=method_src,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def parse_method_args(items):
    """`k=v` tokens over the registered keys, defaults filled in; typed values returned."""
    raw = dict(DEFAULT_METHOD_ARGS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--method-args entries are KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if k not in DEFAULT_METHOD_ARGS:
            raise ValueError(
                f"unknown method-arg {k!r}; the registered keys are {sorted(DEFAULT_METHOD_ARGS)}"
            )
        raw[k] = v
    max_hop, max_iter, epsilon = int(raw["max_hop"]), int(raw["max_iter"]), float(raw["epsilon"])
    if max_hop < 0 or max_iter < 1 or epsilon < 0:
        raise ValueError(f"bad method-args {raw}")
    if raw["penalty"] not in PENALTIES:
        raise ValueError(f"penalty must be one of {PENALTIES}, got {raw['penalty']!r}")
    if epsilon == int(epsilon):
        epsilon = int(epsilon)
    return {"max_hop": max_hop, "penalty": raw["penalty"], "max_iter": max_iter, "epsilon": epsilon}


def is_acyclic(adj):
    """Kahn's algorithm over a 0/1 adjacency (row = cause), the diagonal ignored."""
    import numpy as np

    a = (np.asarray(adj) > 0).astype(np.int64)
    np.fill_diagonal(a, 0)
    alive = np.ones(len(a), dtype=bool)
    indeg = a.sum(axis=0)
    while True:
        src = np.flatnonzero(alive & (indeg == 0))
        if not len(src):
            break
        alive[src] = False
        indeg = indeg - a[src].sum(axis=0)
    return not bool(alive.any())


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--method-src", required=True, help="where `castle/` imports from (the pinned install)"
    )
    p.add_argument("--events", required=True)
    p.add_argument(
        "--allowed", required=True, help="an (k, 2) int64 .npy of allowed pairs, or 'none'"
    )
    p.add_argument("--setting", required=True, type=int, help="decay_ms: 1/delta in milliseconds")
    p.add_argument(
        "--method-args",
        required=True,
        nargs="*",
        help="KEY=VALUE over max_hop, penalty, max_iter, epsilon",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.setting <= 0:
        raise ValueError(f"--setting (decay_ms) must be positive, got {args.setting}")
    margs = parse_method_args(args.method_args)

    import numpy as np
    import pandas as pd

    sys.path.insert(0, args.method_src)
    import castle  # noqa: E402
    from castle.algorithms import TTPM  # noqa: E402
    from castle.common.priori_knowledge import PrioriKnowledge  # noqa: E402

    with np.load(args.events) as z:
        for key in ("event", "timestamp", "node", "n_ops", "op_topology"):
            if key not in z.files:
                raise ValueError(f"{args.events} lacks {key!r} (THP needs the nodes)")
        event = z["event"].astype(np.int64)
        timestamp = z["timestamp"].astype(np.int64)
        node = z["node"].astype(np.int64)
        n_ops = int(z["n_ops"])
        topology = np.asarray(z["op_topology"]).astype(np.int64)
    if not len(event):
        raise ValueError("no events")
    n_types = int(event.max()) + 1
    present = np.unique(event)
    if len(present) != n_types or int(present[0]) != 0:
        # TTPM sizes its matrix over the event names it sees; a gap would shrink it
        raise ValueError("event ids must be dense 0 .. n_types-1 and every id present")
    if topology.shape != (n_ops, n_ops) or not np.array_equal(topology, topology.T):
        raise ValueError(f"op_topology must be a symmetric ({n_ops}, {n_ops}) matrix")
    if int(node.min()) < 0 or int(node.max()) >= n_ops:
        raise ValueError("every node must be an op id below n_ops")
    if np.any(np.diff(timestamp) < 0):
        raise ValueError("timestamps must be sorted")

    allowed = None
    priori = None
    n_forbidden = None
    if args.allowed != "none":
        pairs = np.load(args.allowed).astype(np.int64).reshape(-1, 2)
        allowed = {(int(a), int(b)) for a, b in pairs}
        if any(a == b or not (0 <= a < n_types and 0 <= b < n_types) for a, b in allowed):
            raise ValueError("allowed pairs must be off-diagonal type ids below n_types")
        priori = PrioriKnowledge(n_types)
        forbidden = [
            (int(a), int(b))
            for a in range(n_types)
            for b in range(n_types)
            if a != b and (a, b) not in allowed
        ]
        priori.add_forbidden_edges(forbidden)
        # TTPM asks `is_forbidden` (a membership test on this attribute) once per ordered pair and
        # hill-climb iteration; a set answers in O(1) where the list it keeps answers in O(k)
        priori.forbidden_edges = set(priori.forbidden_edges)
        n_forbidden = len(forbidden)

    # the event name is the dense id itself: TTPM sorts the names it sees (ints sort numerically)
    # and labels its matrix with their string forms, so position i is type i; checked below
    df = pd.DataFrame({"event": event, "timestamp": timestamp, "node": node})
    delta = 1.0 / float(args.setting)
    ttpm = TTPM(
        topology,
        delta=delta,
        epsilon=margs["epsilon"],
        max_hop=margs["max_hop"],
        penalty=margs["penalty"],
        max_iter=margs["max_iter"],
        priori_knowledge=priori,
    )
    t0 = time.monotonic()
    ttpm.learn(df)
    wall = time.monotonic() - t0
    if not hasattr(ttpm, "n_sweeps_") or not hasattr(ttpm, "converged_"):
        raise RuntimeError(
            "TTPM reports no n_sweeps_ / converged_: the 0001-decomposable-score patch is not "
            "applied under --method-src (plans/thp.md §2; scripts/envs/thp-env.sh)"
        )
    n_sweeps = int(ttpm.n_sweeps_)
    converged = bool(ttpm.converged_)
    cm = ttpm.causal_matrix
    rows = [int(str(x)) for x in list(cm.index)]
    cols = [int(str(x)) for x in list(cm.columns)]
    if rows != list(range(n_types)) or cols != list(range(n_types)):
        raise ValueError("the event names did not round-trip to the dense ids in order")
    g = np.array(np.asarray(cm), dtype=np.float64, copy=True)
    if g.shape != (n_types, n_types):
        raise ValueError(f"TTPM returned a {g.shape} matrix for {n_types} types")
    np.fill_diagonal(g, 0.0)
    if not np.all(np.isin(g, (0.0, 1.0))):
        raise ValueError("TTPM's causal matrix must be 0/1")
    if not is_acyclic(g):
        raise ValueError("TTPM returned a cyclic graph")
    src, dst = np.nonzero(g > 0)
    n_outside = (
        None
        if allowed is None
        else int(sum((int(a), int(b)) not in allowed for a, b in zip(src, dst, strict=True)))
    )
    scores = g.astype(np.float32)
    np.save(args.out, scores)
    n_rows = int(len(getattr(ttpm, "tensor", df)))
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_types": n_types,
                "n_events": int(len(event)),
                "n_ops": n_ops,
                "n_topology_edges": int(topology.sum() // 2),
                "n_rows": n_rows,  # TTPM's (event, timestamp, node) rows after its grouping
                "n_allowed_pairs": None if allowed is None else len(allowed),
                "n_forbidden_pairs": n_forbidden,
                "n_nonzero": int(len(src)),
                "n_edges": int(len(src)),
                "n_edges_outside_mask": n_outside,
                "acyclic": True,
                "n_sweeps_run": n_sweeps,
                "converged": converged,
                "max_iter": margs["max_iter"],
                "wall_seconds": wall,
                "setting": int(args.setting),
                "delta": delta,
                "method_args": margs,
                "method_version": f"gcastle {getattr(castle, '__version__', 'unknown')} @ {method_commit(args.method_src)}",
                "method_env": pip_freeze(),
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
