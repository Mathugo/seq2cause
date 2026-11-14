"""The external arm family's read (`plans/cascade.md`, pre-registered 2026-11-03; D-SB-27,
D-SB-28): CASCADE on the sampled sequences of one corpus, grain, split and view ordering, for
`cascade/topology` (constrained to the topology floor's token pairs) and / or `cascade/none`.

    python -m seq2causebench.cascadearm --corpus <dir> --arms cascade/topology cascade/none \\
        --ordering end --grain request --split val --rung xs --num-sequences 0 \\
        --sequence-sample head --max-len 64 --seed 0 --max-delays-ms 1000 10000 60000 \\
        --precision 2 --cells none --freeze '' --prior-rung-record smallest-rung \\
        --staging-ledger plans/staging-ledger.json --replica local --aws-profile-name none \\
        --cascade-python /opt/miniconda3/envs/cascade/bin/python \\
        --cascade-src <the pinned, patched clone> --output-folder out/<run>

The events are the arms' sample of the split (`--num-sequences`, `--sequence-sample`, D-SB-14),
each sequence's first `--max-len` tokens (D-SB-7), one event per token: type `<op>:<outcome>`
from the view rows (the floors' token space), time = the span's end in integer milliseconds
relative to the earliest sampled event. The method runs once per setting (`--max-delays-ms`) as
a subprocess under its own interpreter (`cascade_runner`), and every edge of the DAG it returns
is scored 1.0 in a sweep-shaped table whose `N` slot carries the setting (one file per setting,
`c = g = 0`, the model hash "none", the method's `method_sha256` beside it). The mask of
`cascade/topology` is the score-side topology floor's pair set (`floorprior`), handed over as
dense type ids; the view is read through `Corpus` only. On `--split test` the read asserts the
committed external freeze (family, the one cell, τ, the frozen setting, the corpus, the
ordering) and writes the prediction and ranking files for `annotate`; one arm per test read.

Since the family's generalisation (addendum 2026-11-03 (later), D-SB-29) this module is CASCADE's front
end on `externalarm`: its command line, record keys and work-directory file names are unchanged;
the read itself, the events, the mask and the score table are the generic module's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import externalarm as ext
from .constants import (
    EXTERNAL_METHODS,
    EXTERNAL_STAGE,
    MASK_REFUSE,
    METHOD_CASCADE,
)
from .externalarm import (  # noqa: F401  (re-exported: the tests and the plan name them here)
    EVENT_COLUMNS,
    INT32_MAX,
    NS_PER_MS,
    ExternalRefusal,
    events_read,
    parse_cells,
    topology_mask,
)
from .record import RunRecord

PATCH_NAME = EXTERNAL_METHODS[METHOD_CASCADE]["hash_extras"][0]
CascadeRefusal = ExternalRefusal


def method_sha256(cascade_src):
    """sha256 over the sorted `src/*.py` (name + bytes) of the pinned clone, its commit and the
    patch name (plans/cascade.md §2)."""
    root = Path(cascade_src)
    h = hashlib.sha256()
    for p in sorted((root / "src").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    h.update(commit.encode())
    h.update(PATCH_NAME.encode())
    return h.hexdigest(), commit


def run_cascade(python, runner, cascade_src, work, events, allowed, max_delay_ms, precision):
    """One fit as a subprocess; returns the adjacency and the runner's sidecar facts."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ev_path = work / "events.npz"
    np.savez(ev_path, event=events["event"], timestamp=events["timestamp"])
    if allowed is None:
        allowed_arg = "none"
    else:
        allowed_arg = str(work / "allowed.npy")
        np.save(allowed_arg, allowed)
    out = work / f"graph-d{max_delay_ms}.npy"
    cmd = [
        str(python),
        "-I",
        str(runner),
        "--cascade-src",
        str(cascade_src),
        "--events",
        str(ev_path),
        "--allowed",
        allowed_arg,
        "--max-delay",
        str(int(max_delay_ms)),
        "--precision",
        str(int(precision)),
        "--instant",
        "1",
        "--out",
        str(out),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    (work / f"runner-d{max_delay_ms}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise CascadeRefusal(
            f"cascade_runner exited {proc.returncode} for max_delay {max_delay_ms} ms: "
            f"{proc.stderr.strip()[-2000:]}"
        )
    graph = np.load(out)
    facts = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    facts["wall_seconds_subprocess"] = wall
    return graph, facts


def dag_table(graph, events, allowed):
    """The arm's score table: every edge of the DAG scored 1.0, within-op pairs dropped
    (D-SB-10), any edge outside the mask refused."""
    return ext.score_table(
        (np.asarray(graph) > 0).astype(np.float32), events, allowed, MASK_REFUSE
    )


def run_external(args, rec):
    runner = Path(__file__).with_name("cascade_runner.py")
    sha, commit = method_sha256(args.cascade_src)
    settings = [int(d) for d in args.max_delays_ms]

    def fit(arm, setting, events, allowed, work):
        graph, facts = run_cascade(
            args.cascade_python,
            runner,
            args.cascade_src,
            work,
            events,
            allowed,
            setting,
            args.precision,
        )
        return (np.asarray(graph) > 0).astype(np.float32), facts

    return ext.run_external(
        args,
        rec,
        METHOD_CASCADE,
        fit,
        settings,
        sha,
        commit,
        extra_results={
            "method_patch": PATCH_NAME,
            "cascade_python": str(args.cascade_python),
            "max_delays_ms": settings,
            "precision": int(args.precision),
        },
        npz_meta=lambda d: {"max_delay_ms": d},
    )


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", required=True, nargs="+", choices=EXTERNAL_METHODS[METHOD_CASCADE]["arms"])
    p.add_argument(
        "--max-delays-ms",
        required=True,
        nargs="+",
        type=int,
        help="the settings grid (plans/cascade.md §3) on a validation read; the one frozen value on a test read",
    )
    p.add_argument(
        "--precision", required=True, type=int, help="CASCADE's precision (published default 2)"
    )
    p.add_argument(
        "--cascade-python", required=True, help="the method's interpreter (the `cascade` env)"
    )
    p.add_argument("--cascade-src", required=True, help="the pinned, patched CASCADE clone")
    return ext.add_common_arguments(p)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.freeze = args.freeze or None
    with RunRecord(args.output_folder, EXTERNAL_STAGE, vars(args)) as rec:
        res = run_external(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
