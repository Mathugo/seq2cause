"""Run CASCADE once on an events file (plans/cascade.md §2; D-SB-27). Executed by the method's
own interpreter (the `cascade` conda environment), never imported by the harness:

    <cascade python> -I cascade_runner.py --cascade-src <clone> --events <events.npz> \
        --allowed <pairs.npy | none> --max-delay <ms> --precision 2 --instant 1 --out <graph.npy>

`events.npz` holds `event` (dense type ids, int64) and `timestamp` (integer milliseconds, sorted);
`pairs.npy` is an int64 array of shape (k, 2) of allowed (cause, effect) type ids, or the word
`none` for no mask. Writes the 0/1 adjacency (row = cause) as int8 and a sidecar JSON with the
wall clock, the edge count and `method_env` (the `pip freeze` lines of this interpreter, carried
into the read's `run_meta.json`; addendum 2026-11-04). Standard library, numpy and pandas only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def pip_freeze():
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True, timeout=120
        ).stdout
        return [ln for ln in out.splitlines() if ln.strip()] or ["<empty pip freeze>"]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [f"<pip freeze unavailable: {type(e).__name__}>"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cascade-src", required=True, help="the pinned, patched CASCADE clone")
    p.add_argument("--events", required=True)
    p.add_argument(
        "--allowed", required=True, help="an (k, 2) int64 .npy of allowed pairs, or 'none'"
    )
    p.add_argument(
        "--max-delay", required=True, type=int, help="CASCADE's max_delay, in the events' unit"
    )
    p.add_argument("--precision", required=True, type=int)
    p.add_argument("--instant", required=True, type=int, choices=(0, 1), help="allow_instant")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import numpy as np
    import pandas as pd

    sys.path.insert(0, args.cascade_src)
    from src.main import cascade  # the shipped entry point, patched for allowed_pairs

    with np.load(args.events) as z:
        event = z["event"].astype(np.int64)
        timestamp = z["timestamp"].astype(np.int64)
    n_types = int(event.max()) + 1 if len(event) else 0
    allowed = None
    if args.allowed != "none":
        pairs = np.load(args.allowed).astype(np.int64).reshape(-1, 2)
        allowed = {(int(a), int(b)) for a, b in pairs}
    df = pd.DataFrame({"event": event, "timestamp": timestamp})
    t0 = time.monotonic()
    graph = cascade(
        df,
        number_of_events=n_types,
        allow_instant=bool(args.instant),
        precision=args.precision,
        max_delay=args.max_delay,
        allowed_pairs=allowed,
    )
    wall = time.monotonic() - t0
    graph = np.asarray(graph, dtype=np.int8)
    np.save(args.out, graph)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_types": n_types,
                "n_events": int(len(event)),
                "n_allowed_pairs": None if allowed is None else len(allowed),
                "max_delay": args.max_delay,
                "precision": args.precision,
                "allow_instant": bool(args.instant),
                "n_edges": int(graph.sum()),
                "wall_seconds": wall,
                "method_env": pip_freeze(),
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
