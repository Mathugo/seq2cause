"""The external arm family's read (`plans/cascade.md`, pre-registered 2026-11-03; D-SB-27,
D-SB-28): CASCADE on the sampled sequences of one corpus, grain, split and view ordering, for
`cascade/topology` (constrained to the topology floor's token pairs) and / or `cascade/none`.

    python -m seq2causebench.cascadearm --corpus <dir> --arms cascade/topology cascade/none \
        --ordering end --grain request --split val --rung xs --num-sequences 0 \
        --sequence-sample head --max-len 64 --seed 0 --max-delays-ms 1000 10000 60000 \
        --precision 2 --cells none --freeze '' --prior-rung-record smallest-rung \
        --staging-ledger plans/staging-ledger.json --replica local --aws-profile-name none \
        --cascade-python /opt/miniconda3/envs/cascade/bin/python \
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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .arms import arm_slug, column_of, parse_cell_key, probe_of
from .constants import (
    ARM_CASCADE_TOPOLOGY,
    CUT_FROZEN,
    EXTERNAL_ARMS,
    EXTERNAL_STAGE,
    FAMILY_EXTERNAL,
    GRAINS,
    MODEL_NONE,
    NOISE_NONE,
    ORDERINGS,
    OUTCOME_NAMES,
    PATH_NONE,
    RUNGS,
    SCORES_NPZ_FMT,
    SEQUENCE_SAMPLES,
    SEQUENCES_DIR,
    SPLITS,
    SWEEP_SCORES_NPZ_FMT,
)
from .corpus import Corpus, corpus_id_of
from .data import select_sequences
from .floorprior import topology_read
from .floors import NO_CELLS, OUTCOME_BASE, STAGE_OF_SPLIT, floor_scores, write_floor_cells
from .freezecheck import FreezeRefusal, assert_freeze
from .log import log
from .project import write_scores_npz
from .record import RunRecord, read_json
from .staging import require_prior_rung

PATCH_NAME = "0001-allowed-pairs"
NS_PER_MS = 1_000_000
INT32_MAX = 2**31 - 1
EVENT_COLUMNS = ("trace_id", "ops", "outcomes", "n_spans", "start_ns", "offsets", "durations")


class CascadeRefusal(RuntimeError):
    pass


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


def events_read(corpus, split, n, mode, seed, max_len):
    """The sampled sequences as one timeline: dense type ids, millisecond end times (sorted,
    ties in row order), the token list and the facts the record carries."""
    rng = np.random.default_rng(seed)
    seqs = select_sequences(corpus.sequences(split, columns=EVENT_COLUMNS), n, mode, rng)
    keys, times = [], []
    n_short = 0
    for _tid, ops, outs, _n, start_ns, offsets, durations in seqs:
        ops = np.asarray(ops[:max_len], dtype=np.int64)
        outs = np.asarray(outs[:max_len], dtype=np.int64)
        off = np.asarray(offsets[:max_len], dtype=np.int64)
        dur = np.asarray(durations[:max_len], dtype=np.int64)
        if len(ops) < 2:
            n_short += 1
        keys.append(ops * OUTCOME_BASE + outs)
        times.append((int(start_ns) + off + dur) // NS_PER_MS)
    if not keys:
        raise CascadeRefusal("the sample holds no sequence")
    key = np.concatenate(keys)
    t = np.concatenate(times)
    t = t - int(t.min())
    if int(t.max()) > INT32_MAX:
        raise CascadeRefusal("the sampled timeline exceeds int32 milliseconds")
    order = np.argsort(t, kind="stable")
    key, t = key[order], t[order]
    toks, dense = np.unique(key, return_inverse=True)
    tokens = np.array(
        [f"{int(k) // OUTCOME_BASE}:{OUTCOME_NAMES[int(k) % OUTCOME_BASE]}" for k in toks]
    )
    return {
        "event": dense.astype(np.int64),
        "timestamp": t.astype(np.int64),
        "tokens": tokens,
        "op_of_token": (toks // OUTCOME_BASE).astype(np.int64),
        "facts": {
            "n_sequences": int(len(seqs)),
            "n_skipped_short": int(n_short),
            "n_events": int(len(key)),
            "n_tokens": int(len(toks)),
            "span_ms": int(t.max()),
            "inputs": [f"{corpus.view_rel}/{SEQUENCES_DIR}/split={split}"],
        },
    }


def topology_mask(corpus_dir, tokens):
    """The topology floor's ordered token pairs (score side, `floorprior.topology_read`) as dense
    (cause, effect) ids over the sample's token list; pairs over tokens absent from the sample
    are dropped and counted."""
    table = topology_read(corpus_dir)
    prior_tokens = np.asarray(table["tokens"])
    pos = {str(tok): i for i, tok in enumerate(tokens)}
    pairs, dropped = [], 0
    for s, d in zip(table["src"], table["dst"], strict=True):
        a, b = pos.get(str(prior_tokens[s])), pos.get(str(prior_tokens[d]))
        if a is None or b is None:
            dropped += 1
            continue
        pairs.append((a, b))
    return np.array(sorted(set(pairs)), dtype=np.int64).reshape(-1, 2), {
        "n_prior_pairs": int(len(table["src"])),
        "n_prior_pairs_outside_sample": int(dropped),
        "n_allowed_pairs": int(len(set(pairs))),
    }


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
    src, dst = np.nonzero(np.asarray(graph) > 0)
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    if allowed is not None and len(src):
        allowed_set = {(int(a), int(b)) for a, b in allowed}
        outside = [
            (int(a), int(b)) for a, b in zip(src, dst, strict=True) if (a, b) not in allowed_set
        ]
        if outside:
            raise CascadeRefusal(f"{len(outside)} edge(s) outside the mask, e.g. {outside[:3]}")
    op = events["op_of_token"]
    keep = op[src] != op[dst]
    n_within = int((~keep).sum())
    src, dst = src[keep], dst[keep]
    order = np.lexsort((dst, src))
    return {
        "tokens": events["tokens"],
        "src": src[order],
        "dst": dst[order],
        "score": np.ones(len(order), dtype=np.float64),
        "count": np.ones(len(order), dtype=np.int64),
        "facts": {"n_edges_returned": int(len(keep)), "n_within_op_dropped": n_within},
    }


def parse_cells(specs, grain, arms):
    specs = list(specs)
    if specs == [NO_CELLS]:
        return {}
    cells = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"--cells entries are KEY=TAU, or the single word {NO_CELLS!r}; got {spec!r}"
            )
        key, val = spec.rsplit("=", 1)
        arm, _path, cut, g = parse_cell_key(key)
        if arm not in arms:
            raise ValueError(f"cell {key} is not one of this read's arms {list(arms)}")
        if g != grain:
            raise ValueError(f"cell {key} is on grain {g!r} but this read is on {grain!r}")
        if cut != CUT_FROZEN:
            raise ValueError(f"cell {key}: an external arm has the frozen cut only")
        cells[key] = float(val)
    return cells


def run_external(args, rec):
    if args.split == "test" and not args.freeze:
        raise FreezeRefusal(
            "a test read needs --freeze <committed external freeze> (PRD scenario 13)"
        )
    stage = STAGE_OF_SPLIT.get(args.split, "sweep")
    staging = {
        **require_prior_rung(args.prior_rung_record, args.rung, stage, args.staging_ledger),
        "stage_matched": stage,
    }
    rec.note(replica=args.replica, aws_profile=args.aws_profile_name)
    corpus_id = corpus_id_of(args.corpus)
    if "/" in corpus_id and not corpus_id.startswith(args.rung + "/"):
        raise ValueError(f"corpus {corpus_id!r} is not at rung {args.rung!r}")
    arms = list(dict.fromkeys(args.arms))
    settings = list(dict.fromkeys(int(d) for d in args.max_delays_ms))
    cells = parse_cells(args.cells, args.grain, arms)
    if args.split == "test" and (len(arms) != 1 or len(settings) != 1 or len(cells) != 1):
        raise FreezeRefusal(
            "a test read fits one arm at its one frozen setting for one cell (plans/cascade.md §4)"
        )
    sha, commit = method_sha256(args.cascade_src)
    freeze_facts = None
    if args.freeze:
        freeze = read_json(args.freeze)
        if freeze.get("family") != FAMILY_EXTERNAL:
            raise FreezeRefusal(
                f"freeze {args.freeze} is not an external freeze (family {freeze.get('family')!r}; D-SB-27)"
            )
        freeze_facts = assert_freeze(
            freeze,
            args.freeze,
            cells,
            0,
            settings[0],
            0,
            MODEL_NONE,
            corpus_id,
            rec.started,
            args.ordering,
        )
        log({"event": "freeze_ok", **freeze_facts})
    out = rec.out_dir
    runner = Path(__file__).with_name("cascade_runner.py")
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    events = events_read(
        corpus, args.split, args.num_sequences, args.sequence_sample, args.seed, args.max_len
    )
    log({"event": "events_read", "grain": args.grain, **events["facts"]})
    tables, inputs, facts, files = {}, {}, {}, {}
    for arm in arms:
        probe = probe_of(arm)
        col = column_of(arm, PATH_NONE)
        allowed, mask_facts = (None, {})
        arm_inputs = list(events["facts"]["inputs"])
        if arm == ARM_CASCADE_TOPOLOGY:
            allowed, mask_facts = topology_mask(args.corpus, events["tokens"])
            arm_inputs += ["topology prior through floorprior.topology_read (score side)"]
        fits = {}
        for d in settings:
            graph, run_facts = run_cascade(
                args.cascade_python,
                runner,
                args.cascade_src,
                out / "cascade-work" / f"{arm_slug(arm)}-{args.grain}",
                events,
                allowed,
                d,
                args.precision,
            )
            table = dag_table(graph, events, allowed)
            scores = floor_scores(table, col)
            if args.split == "test":
                name = SCORES_NPZ_FMT.format(probe=probe, noise=NOISE_NONE, grain=args.grain)
            else:
                name = SWEEP_SCORES_NPZ_FMT.format(
                    probe=probe, noise=NOISE_NONE, c=0, n=d, grain=args.grain
                )
            write_scores_npz(
                out / name,
                scores,
                probe=probe,
                noise=NOISE_NONE,
                c=0,
                N=d,
                g=0,
                grain=args.grain,
                split=args.split,
                ordering=args.ordering,
                model_sha256=MODEL_NONE,
                method_sha256=sha,
                max_delay_ms=d,
                corpus_id=corpus_id,
                rung=args.rung,
                n_sequences=int(events["facts"]["n_sequences"]),
            )
            fits[str(d)] = {
                **run_facts,
                **table["facts"],
                "n_pairs": int(len(scores["src"])),
                "file": name,
            }
            tables[arm] = (scores, col)
            log(
                {
                    "event": "cascade_fit",
                    "arm": arm,
                    "grain": args.grain,
                    "max_delay_ms": d,
                    **fits[str(d)],
                }
            )
        inputs[arm] = arm_inputs
        facts[arm] = {
            "n_sequences": int(events["facts"]["n_sequences"]),
            "n_skipped_short": int(events["facts"]["n_skipped_short"]),
            "n_events": int(events["facts"]["n_events"]),
            "n_tokens": int(events["facts"]["n_tokens"]),
            "span_ms": int(events["facts"]["span_ms"]),
            **mask_facts,
            "fits": fits,
        }
        files[arm] = [f["file"] for f in fits.values()]
    summary = write_floor_cells(
        out,
        tables,
        cells,
        args.grain,
        split=args.split,
        model_sha256=MODEL_NONE,
        corpus_id=corpus_id,
    )
    return {
        "stage": EXTERNAL_STAGE,
        "arms": arms,
        "probe": probe_of(arms[0]) if len(arms) == 1 else None,
        "noise": NOISE_NONE,
        "grain": args.grain,
        "split": args.split,
        "ordering": args.ordering,
        "rung": args.rung,
        "corpus_id": corpus_id,
        "model_sha256": MODEL_NONE,
        "method_sha256": sha,
        "method_commit": commit,
        "method_patch": PATCH_NAME,
        "cascade_python": str(args.cascade_python),
        "max_delays_ms": settings,
        "precision": int(args.precision),
        "c": 0,
        "N": settings[0] if len(settings) == 1 else None,
        "g": 0,
        "probe_amp": None,
        "max_len": args.max_len,
        "n_sequences": int(events["facts"]["n_sequences"]),
        "n_probed": int(events["facts"]["n_sequences"]),
        "n_skipped_short": int(events["facts"]["n_skipped_short"]),
        "corrupted_cells": {},
        "memory": None,
        "cells": summary,
        "shipped_rule": None,
        "freeze": freeze_facts,
        "staging": staging,
        "inputs": inputs,
        "facts": facts,
        "files": files,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument("--arms", required=True, nargs="+", choices=EXTERNAL_ARMS)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--num-sequences", required=True, type=int, help="0 = the whole split (D-SB-14)")
    p.add_argument("--sequence-sample", required=True, choices=SEQUENCE_SAMPLES, help="D-SB-14")
    p.add_argument(
        "--max-len", required=True, type=int, help="cap on real tokens per sequence (D-SB-7)"
    )
    p.add_argument("--seed", required=True, type=int, help="the sample's seed (uniform mode)")
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
        "--cells", required=True, nargs="+", help=f"KEY=TAU on a test read, or {NO_CELLS!r}"
    )
    p.add_argument(
        "--freeze", required=True, help="the committed external freeze, or '' (val only)"
    )
    p.add_argument("--prior-rung-record", required=True, help="see staging.py (scenario 46)")
    p.add_argument("--staging-ledger", required=True, help="the tracked plans/staging-ledger.json")
    p.add_argument(
        "--replica", required=True, help="the dated provisioning replica; 'local' locally"
    )
    p.add_argument("--aws-profile-name", required=True, help="the AWS profile name; 'none' locally")
    p.add_argument(
        "--cascade-python", required=True, help="the method's interpreter (the `cascade` env)"
    )
    p.add_argument("--cascade-src", required=True, help="the pinned, patched CASCADE clone")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.freeze = args.freeze or None
    with RunRecord(args.output_folder, EXTERNAL_STAGE, vars(args)) as rec:
        res = run_external(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
