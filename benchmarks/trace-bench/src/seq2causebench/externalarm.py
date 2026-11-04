"""The external arm family's generic read (plans/cascade.md and its addendum of 2026-11-04;
plans/{thp,lmt,causalnet}.md; D-SB-27 … D-SB-29): a published discovery method outside seq2cause —
registered in `constants.EXTERNAL_METHODS` with its ledger stage, its own freeze suffix, its runner
module and the meaning of its `N` slot — fitted on the sampled sequences of one corpus, grain,
split and view ordering as a subprocess under the method's own interpreter.

    python -m seq2causebench.externalarm --method thp --arms thp/topology \\
        --ordering end --grain request --split val --rung xs --num-sequences 0 \\
        --sequence-sample head --max-len 64 --seed 0 --settings 100 1000 10000 \\
        --method-args max_hop=1 max_iter=20 --cells none --freeze '' \\
        --prior-rung-record smallest-rung --staging-ledger plans/staging-ledger.json \\
        --replica local --aws-profile-name none \\
        --method-python /opt/miniconda3/envs/thp/bin/python --method-src <dir> \\
        --output-folder out/<run>

The events are the arms' sample of the split (`--num-sequences`, `--sequence-sample`, D-SB-14),
each sequence's first `--max-len` tokens (D-SB-7), one event per token: type `<op>:<outcome>`
from the view rows (the floors' token space), time = the span's end in integer milliseconds
relative to the earliest sampled event; the node of an event is its op (node = op). The method
runs once per setting (`--settings`, the registered meaning of the `N` slot) as a subprocess under
its interpreter (`<method>_runner.py`); the score matrix it returns is read out in a sweep-shaped
table whose `N` slot carries the setting (one file per setting, `c = g = 0`, the model hash
"none", the method's `method_sha256` and pinned commit beside it). A `*/topology` arm's mask is the
score-side topology floor's pair set (`floorprior`), handed over as dense type ids together with
the undirected op-level topology it spans; the view is read through `Corpus` only. On `--split
test` the read asserts the committed external freeze of the method (family, method, the one cell,
τ, the frozen setting, the corpus, the ordering) and writes the prediction and ranking files for
`annotate`; one arm per test read. CASCADE keeps its own front end (`cascadearm`) on this module.

The runner contract (`<method>_runner.py`; standard library + numpy (+ pandas) only; never
imported by the harness; it puts `--method-src` at the head of `sys.path` and imports the method
from there):

    <python> -I <runner>.py --method-src <dir> --events <events.npz> --allowed <allowed.npy|none> \\
        --setting <int> --method-args k=v ... --out <scores.npy>

`events.npz`: `event` (dense type ids, int64), `timestamp` (integer milliseconds, sorted; ties in
row order), and when the method `needs_nodes`: `node` (the dense op id of every event), `n_ops`,
`op_topology` (undirected int8 `(n_ops, n_ops)` adjacency over the ops, spanned by the allowed
pairs; zeros without a mask). `allowed.npy`: int64 `(k, 2)` allowed (cause, effect) type ids. The
runner writes `<out>` as a float32 `(n_types, n_types)` score matrix in [0, 1] (row = cause; the
diagonal is ignored) and `<out>.json` with `n_types`, `n_events`, `n_allowed_pairs`, `n_nonzero`,
`wall_seconds`, `setting`, `method_args`, `method_version` and `method_env` (the `pip freeze`
lines of the runner's interpreter, carried into the read's `run_meta.json`).
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
    CUT_FROZEN,
    EXTERNAL_ARMS,
    EXTERNAL_METHOD_OF_ARM,
    EXTERNAL_METHODS,
    FAMILY_EXTERNAL,
    GRAINS,
    MASK_COUNT,
    MASK_MODES,
    MASK_REFUSE,
    METHOD_CASCADE,
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

NS_PER_MS = 1_000_000
INT32_MAX = 2**31 - 1
EVENT_COLUMNS = ("trace_id", "ops", "outcomes", "n_spans", "start_ns", "offsets", "durations")
GENERIC_METHODS = tuple(m for m in EXTERNAL_METHODS if m != METHOD_CASCADE)


class ExternalRefusal(RuntimeError):
    pass


# --- the method's identity -------------------------------------------------------------------
def method_pin(method_src):
    """The pinned commit of the method under `method_src`: the clone's HEAD, else the commit of a
    pip install from a git URL (`<dist>.dist-info/direct_url.json`), else "unknown"."""
    root = Path(method_src)
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    for p in sorted(root.glob("*.dist-info/direct_url.json")):
        try:
            commit = json.loads(p.read_text(encoding="utf-8")).get("vcs_info", {}).get("commit_id")
        except (OSError, ValueError):
            continue
        if commit:
            return str(commit)
    return "unknown"


def method_sha256(method, method_src, runner=None):
    """sha256 over the method's registered files under `method_src` (relative name + bytes, sorted),
    its pinned commit, the registered extras (patch names) and the runner's own bytes."""
    spec = EXTERNAL_METHODS[method]
    root = Path(method_src)
    files = sorted({p for pat in spec["hash_globs"] for p in root.glob(pat) if p.is_file()})
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    commit = method_pin(root)
    h.update(commit.encode())
    for extra in spec["hash_extras"]:
        h.update(str(extra).encode())
    if runner is not None:
        h.update(Path(runner).read_bytes())
    return h.hexdigest(), commit


# --- the events, the mask and the node topology ----------------------------------------------
def events_read(corpus, split, n, mode, seed, max_len):
    """The sampled sequences as one timeline: dense type ids, millisecond end times (sorted,
    ties in row order), the token list, the op of every token (raw and dense) and the facts the
    record carries."""
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
        raise ExternalRefusal("the sample holds no sequence")
    key = np.concatenate(keys)
    t = np.concatenate(times)
    t = t - int(t.min())
    if int(t.max()) > INT32_MAX:
        raise ExternalRefusal("the sampled timeline exceeds int32 milliseconds")
    order = np.argsort(t, kind="stable")
    key, t = key[order], t[order]
    toks, dense = np.unique(key, return_inverse=True)
    tokens = np.array(
        [f"{int(k) // OUTCOME_BASE}:{OUTCOME_NAMES[int(k) % OUTCOME_BASE]}" for k in toks]
    )
    op_of_token = (toks // OUTCOME_BASE).astype(np.int64)
    ops_present, op_id_of_token = np.unique(op_of_token, return_inverse=True)
    return {
        "event": dense.astype(np.int64),
        "timestamp": t.astype(np.int64),
        "tokens": tokens,
        "op_of_token": op_of_token,
        "op_id_of_token": op_id_of_token.astype(np.int64),
        "n_ops": int(len(ops_present)),
        "facts": {
            "n_sequences": int(len(seqs)),
            "n_skipped_short": int(n_short),
            "n_events": int(len(key)),
            "n_tokens": int(len(toks)),
            "n_ops": int(len(ops_present)),
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


def op_topology(allowed, op_id_of_token, n_ops):
    """The undirected op-level adjacency the allowed pairs span (node = op): `A[op(a), op(b)] =
    A[op(b), op(a)] = 1` for every allowed `(a, b)`; zeros without a mask."""
    A = np.zeros((n_ops, n_ops), dtype=np.int8)
    if allowed is not None and len(allowed):
        a = op_id_of_token[allowed[:, 0]]
        b = op_id_of_token[allowed[:, 1]]
        A[a, b] = 1
        A[b, a] = 1
    return A


# --- one fit as a subprocess ------------------------------------------------------------------
def run_method(python, runner, method_src, work, events, allowed, setting, method_args, needs_nodes):
    """One fit under the method's interpreter; returns the float32 score matrix and the runner's
    sidecar facts."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ev_path = work / "events.npz"
    arrays = {"event": events["event"], "timestamp": events["timestamp"]}
    if needs_nodes:
        arrays["node"] = events["op_id_of_token"][events["event"]]
        arrays["n_ops"] = np.asarray(events["n_ops"], dtype=np.int64)
        arrays["op_topology"] = op_topology(allowed, events["op_id_of_token"], events["n_ops"])
    np.savez(ev_path, **arrays)
    if allowed is None:
        allowed_arg = "none"
    else:
        allowed_arg = str(work / "allowed.npy")
        np.save(allowed_arg, allowed)
    out = work / f"scores-N{int(setting)}.npy"
    cmd = [
        str(python),
        "-I",
        str(runner),
        "--method-src",
        str(method_src),
        "--events",
        str(ev_path),
        "--allowed",
        allowed_arg,
        "--setting",
        str(int(setting)),
        "--method-args",
        *[str(a) for a in method_args],
        "--out",
        str(out),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    (work / f"runner-N{int(setting)}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise ExternalRefusal(
            f"{Path(runner).name} exited {proc.returncode} for setting {setting}: "
            f"{proc.stderr.strip()[-2000:]}"
        )
    scores = np.load(out)
    facts = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    facts["wall_seconds_subprocess"] = wall
    return scores, facts


def score_table(scores, events, allowed, mask_mode):
    """The arm's score table from the method's `(n, n)` score matrix (row = cause): the diagonal
    ignored, every positive entry an edge at its score, within-op pairs dropped (D-SB-10); an
    edge outside the mask is refused (`refuse`) or kept and counted (`count`)."""
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    m = np.array(scores, dtype=np.float64, copy=True)
    n = int(len(events["tokens"]))
    if m.shape != (n, n):
        raise ExternalRefusal(f"the method returned a {m.shape} matrix for {n} types")
    if not np.all(np.isfinite(m)) or float(m.min()) < 0 or float(m.max()) > 1:
        raise ExternalRefusal("the method's scores must be finite and within [0, 1]")
    np.fill_diagonal(m, 0.0)
    src, dst = np.nonzero(m > 0)
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    n_outside = 0
    if allowed is not None and len(src):
        allowed_set = {(int(a), int(b)) for a, b in allowed}
        outside = [
            (int(a), int(b)) for a, b in zip(src, dst, strict=True) if (a, b) not in allowed_set
        ]
        n_outside = len(outside)
        if outside and mask_mode == MASK_REFUSE:
            raise ExternalRefusal(f"{len(outside)} edge(s) outside the mask, e.g. {outside[:3]}")
    op = events["op_of_token"]
    keep = op[src] != op[dst]
    n_within = int((~keep).sum())
    src, dst = src[keep], dst[keep]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    return {
        "tokens": events["tokens"],
        "src": src,
        "dst": dst,
        "score": m[src, dst].astype(np.float64),
        "count": np.ones(len(src), dtype=np.int64),
        "facts": {
            "n_edges_returned": int(len(keep)),
            "n_within_op_dropped": n_within,
            "n_edges_outside_mask": int(n_outside),
            "mask_mode": mask_mode if allowed is not None else None,
        },
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


def parse_method_args(items):
    """`k=v` tokens → an ordered dict of strings (recorded verbatim; the runner parses values)."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--method-args entries are KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if not k or k in out:
            raise ValueError(f"--method-args: bad or repeated key in {item!r}")
        out[k] = v
    return out


# --- the read ------------------------------------------------------------------------------------
def run_external(args, rec, method, fit, settings, sha, commit, extra_results=None, npz_meta=None):
    """The family's read for one method: `fit(arm, setting, events, allowed, work_dir)` returns
    the method's score matrix and the runner's facts; `settings` are the `N` values; `sha` and
    `commit` the method's identity; `extra_results` keys join the results; `npz_meta(setting)`
    extra score-file metadata (CASCADE's `max_delay_ms`)."""
    spec = EXTERNAL_METHODS[method]
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
    for arm in arms:
        if EXTERNAL_METHOD_OF_ARM.get(arm) != method:
            raise ValueError(f"arm {arm!r} is not a {method!r} arm")
    settings = list(dict.fromkeys(int(d) for d in settings))
    cells = parse_cells(args.cells, args.grain, arms)
    if args.split == "test" and (len(arms) != 1 or len(settings) != 1 or len(cells) != 1):
        raise FreezeRefusal(
            "a test read fits one arm at its one frozen setting for one cell (plans/cascade.md §4)"
        )
    freeze_facts = None
    if args.freeze:
        freeze = read_json(args.freeze)
        if freeze.get("family") != FAMILY_EXTERNAL:
            raise FreezeRefusal(
                f"freeze {args.freeze} is not an external freeze (family {freeze.get('family')!r}; D-SB-27)"
            )
        if freeze.get("method") not in (None, method):
            raise FreezeRefusal(
                f"freeze {args.freeze} is the {freeze.get('method')!r} method's, not {method!r}'s (D-SB-29)"
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
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    events = events_read(
        corpus, args.split, args.num_sequences, args.sequence_sample, args.seed, args.max_len
    )
    log({"event": "events_read", "grain": args.grain, **events["facts"]})
    tables, inputs, facts, files = {}, {}, {}, {}
    method_env = None
    for arm in arms:
        probe = probe_of(arm)
        col = column_of(arm, PATH_NONE)
        allowed, mask_facts = (None, {})
        arm_inputs = list(events["facts"]["inputs"])
        if arm.endswith("/topology"):
            allowed, mask_facts = topology_mask(args.corpus, events["tokens"])
            arm_inputs += ["topology prior through floorprior.topology_read (score side)"]
        fits = {}
        for d in settings:
            matrix, run_facts = fit(
                arm, d, events, allowed, out / f"{method}-work" / f"{arm_slug(arm)}-{args.grain}"
            )
            if method_env is None and run_facts.get("method_env"):
                method_env = list(run_facts["method_env"])
            run_facts = {k: v for k, v in run_facts.items() if k != "method_env"}
            table = score_table(matrix, events, allowed, spec["mask_mode"])
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
                method_commit=commit,
                method=method,
                setting_name=spec["setting"],
                corpus_id=corpus_id,
                rung=args.rung,
                n_sequences=int(events["facts"]["n_sequences"]),
                **(npz_meta(d) if npz_meta else {}),
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
                    "event": "external_fit",
                    "method": method,
                    "arm": arm,
                    "grain": args.grain,
                    spec["setting"]: d,
                    **fits[str(d)],
                }
            )
        inputs[arm] = arm_inputs
        facts[arm] = {
            "n_sequences": int(events["facts"]["n_sequences"]),
            "n_skipped_short": int(events["facts"]["n_skipped_short"]),
            "n_events": int(events["facts"]["n_events"]),
            "n_tokens": int(events["facts"]["n_tokens"]),
            "n_ops": int(events["facts"]["n_ops"]),
            "span_ms": int(events["facts"]["span_ms"]),
            **mask_facts,
            "fits": fits,
        }
        files[arm] = [f["file"] for f in fits.values()]
    rec.note(method=method, method_commit=commit, method_env=method_env)
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
        "stage": spec["stage"],
        "method": method,
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
        "setting_name": spec["setting"],
        "settings": settings,
        "mask_mode": spec["mask_mode"],
        "hard_output": bool(spec["hard_output"]),
        **(extra_results or {}),
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


def add_common_arguments(p):
    """The flags every external read shares (all required, no defaults)."""
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
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
    p.add_argument("--output-folder", required=True)
    return p


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method", required=True, choices=GENERIC_METHODS)
    p.add_argument("--arms", required=True, nargs="+", choices=EXTERNAL_ARMS)
    p.add_argument(
        "--settings",
        required=True,
        nargs="+",
        type=int,
        help="the method's registered `N`-slot grid on a validation read; the one frozen value on a test read",
    )
    p.add_argument(
        "--method-args",
        required=True,
        nargs="*",
        help="KEY=VALUE tokens passed to the runner verbatim and recorded (the plan's fixed settings)",
    )
    p.add_argument("--method-python", required=True, help="the method's interpreter (its own env)")
    p.add_argument(
        "--method-src",
        required=True,
        help="the directory the runner imports the method from (a pinned clone, or the site-packages "
        "directory of a pinned pip install)",
    )
    return add_common_arguments(p)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.freeze = args.freeze or None
    method = args.method
    spec = EXTERNAL_METHODS[method]
    method_args = parse_method_args(args.method_args)
    runner = Path(__file__).with_name(spec["runner"] + ".py")
    sha, commit = method_sha256(method, args.method_src, runner)

    def fit(arm, setting, events, allowed, work):
        return run_method(
            args.method_python,
            runner,
            args.method_src,
            work,
            events,
            allowed,
            setting,
            [f"{k}={v}" for k, v in method_args.items()],
            spec["needs_nodes"],
        )

    with RunRecord(args.output_folder, spec["stage"], vars(args)) as rec:
        res = run_external(
            args,
            rec,
            method,
            fit,
            args.settings,
            sha,
            commit,
            extra_results={
                "method_args": method_args,
                "method_python": str(args.method_python),
                "method_src": str(args.method_src),
            },
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
