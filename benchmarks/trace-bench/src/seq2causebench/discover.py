# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Run one probe pass on one corpus with one frozen model and write every cell of that pass
(PRD Interface "discover"; scenarios 13, 14, 18, 22, 23, 29, 39, 46).

    python -m seq2causebench.discover --corpus <dir> --ordering end --grain request --split test \
        --rung xs --model <model dir> --probe core --noise all --context 2 --guidance 2 --particles 8 \
        --cells trace/core/shipped/frozen/request=3e-2 trace/core/fixed-kl/frozen/request=1e-2 \
               baseline/granger/shipped/frozen/request=1e-3 \
        --shipped-method percentile --shipped-per-lag false --shipped-decay true \
        --shipped-decay-type exponential --shipped-decay-rate 0.3 --shipped-exponent 0.5 \
        --shipped-floor none --shipped-min-group-size 8 \
        --num-sequences 0 --sequence-sample head --cells-sequences 0 --max-len 64 --max-lag 63 \
        --memory-cap-gb 20 --probe-amp none --per-lag-files 8 --freeze freezes/<date>-xs-latent-s0.json \
        --prior-rung-record smallest-rung --staging-ledger plans/staging-ledger.json \
        --replica <tfvars name> --aws-profile-name default --seed 0 --device cuda --output-folder out/<run>

A pass is one (probe, noise) forward per sequence; its cells are the (arm, path, cut) read-outs
that share that forward (`engine.probe_columns`). `--cells` names each cell with its frozen τ
(`KEY=TAU`), or `KEY=shipped` for a cli arm's shipped cut, which needs no τ: the tool's own rule
(`--shipped-*`, the v0.1.9 defaults passed explicitly and recorded) is fitted on the read's own
matrices. With `--split test` a committed freeze is mandatory (scenarios 13, 29): the commit that
added the freeze must be an ancestor of HEAD and predate the run, and every frozen `(τ, c, N, g)`
must equal the command line and the loaded model's hash. A rung above the smallest names a
completed record of the same stage at the previous rung (scenario 46). The stage refuses an
output folder holding a completed run and refuses above its declared memory cap (scenario 18).

Writes `matrices-<probe>-<noise>-<grain>.npz` (the full strict-upper triangle of every stored
sequence, one column per path — the per-sequence axis's input and the shipped cut's),
`sequences-<grain>.json` (the probed sequences), `scores-<probe>-<noise>-<grain>.npz` (the
type-level table), and per cell the prediction, ranking and per-lag files; `results.json`
carries the corrupted-cell counts per path, the shipped cut's `tau_by_lag` and `threshold_finite`,
the memory estimate, throughput and wall clock.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .arms import arm_slug, parse_cell_key
from .constants import (
    AMP_MODES,
    CLI_SHIPPED_RULE,
    CUT_FROZEN,
    CUT_SHIPPED,
    DEVICES,
    GRAINS,
    MATRICES_NPZ_FMT,
    NOISES,
    ORDERINGS,
    PARTICLE_PROBES,
    PREDICTION_JSON_FMT,
    PREDICTION_LAG_JSON_FMT,
    PROBES,
    RANKING_JSON_FMT,
    RANKING_LAG_JSON_FMT,
    RUNGS,
    SCORES_NPZ_FMT,
    SEQUENCE_SAMPLES,
    SEQUENCES_JSON_FMT,
    SHIPPED_CUT_READ_JSON_FMT,
    SPLITS,
)
from .corpus import Corpus
from .data import SequenceStore
from .engine import (
    check_memory,
    load_backbone,
    merge_counts,
    probe_columns,
    probe_sequence,
    seed_sequence,
    shipped_cut,
    shipped_cut_union,
    triangle,
)
from .log import log
from .prediction import write_prediction
from .project import PairAccumulator, sequence_cells, write_scores_npz
from .record import RunRecord, read_json, write_json
from .select import per_lag_ranking, ranking, select_edges
from .staging import require_prior_rung
from .vocab import Vocab


class FreezeRefusal(RuntimeError):
    pass


# --- the shipped rule's knobs (v0.1.9 defaults, passed explicitly) -----------------------------------
def add_shipped_rule_knobs(p):
    p.add_argument(
        "--shipped-method",
        required=True,
        choices=["otsu", "mad", "percentile", "gmm"],
        help="AdaptiveThreshold.method",
    )
    p.add_argument("--shipped-per-lag", required=True, choices=["true", "false"])
    p.add_argument("--shipped-decay", required=True, choices=["true", "false"])
    p.add_argument("--shipped-decay-type", required=True, choices=["exponential", "power"])
    p.add_argument("--shipped-decay-rate", required=True, type=float)
    p.add_argument("--shipped-exponent", required=True, type=float)
    p.add_argument("--shipped-floor", required=True, help="a float or 'none'")
    p.add_argument("--shipped-min-group-size", required=True, type=int)


def shipped_rule(args):
    rule = {
        "method": args.shipped_method,
        "per_lag": args.shipped_per_lag == "true",
        "decay": args.shipped_decay == "true",
        "decay_type": args.shipped_decay_type,
        "decay_rate": float(args.shipped_decay_rate),
        "exponent": float(args.shipped_exponent),
        "floor": None if args.shipped_floor == "none" else float(args.shipped_floor),
        "min_group_size": int(args.shipped_min_group_size),
    }
    return rule


def is_v019_rule(rule):
    return rule == CLI_SHIPPED_RULE


# --- the cells of a pass ------------------------------------------------------------------------------------
def parse_cells(specs, grain, probe, noise):
    """`{cell_key: tau | "shipped"}` from `KEY=TAU` specs; every key must be a cell of this pass."""
    columns = probe_columns(probe, noise)
    cells = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--cells entries are KEY=TAU or KEY=shipped, got {spec!r}")
        key, val = spec.rsplit("=", 1)
        arm, path, cut, g = parse_cell_key(key)
        if g != grain:
            raise ValueError(f"cell {key} is on grain {g!r} but this read is on {grain!r}")
        if arm not in columns or path not in columns[arm]:
            raise ValueError(
                f"cell {key} is not a read-out of pass ({probe}, {noise}); its cells are "
                f"{sorted(f'{a}/{p}' for a, ps in columns.items() for p in ps)}"
            )
        if cut == CUT_SHIPPED:
            if val != "shipped":
                raise ValueError(f"cell {key} is a shipped cut: write {key}=shipped")
            cells[key] = "shipped"
        else:
            cells[key] = float(val)
        if key in cells and list(cells).count(key) > 1:
            raise ValueError(f"cell {key} given twice")
    if not cells:
        raise ValueError("--cells is empty")
    return cells


# --- the probing core ------------------------------------------------------------------------------------------
def probe_store(
    hf_model,
    adapter,
    vocab,
    store,
    probe,
    noise,
    c,
    g,
    N,
    max_lag,
    memory_cap_gb,
    amp,
    seed,
    split,
    cells_sequences,
    progress_every=256,
):
    """One pass over one sequence store. Returns `(scores table, triangle table | None, per-sequence full
    matrices of the stored sequences per column, facts)`."""
    if probe in PARTICLE_PROBES and N < 1:
        raise ValueError(f"probe {probe!r} needs --particles >= 1")
    if probe not in PARTICLE_PROBES and N != 0:
        raise ValueError(f"probe {probe!r} has no particle axis; pass --particles 0")
    if probe == "core" and not 1 <= g <= c:
        raise ValueError(f"guidance g = {g} must satisfy 1 <= g <= c = {c}")
    if len(store) == 0:
        raise ValueError("the sequence store is empty")
    columns = probe_columns(probe, noise)
    cols = sorted(col for paths in columns.values() for col in paths.values())
    max_L = int(store.lengths.max())
    mem = check_memory(hf_model, probe, N, max_L, c, memory_cap_gb)
    log({"event": "memory", "split": split, "probe": probe, "noise": noise, "c": c, "N": N, **mem})
    device = next(hf_model.parameters()).device
    acc = PairAccumulator(vocab.size, max_lag, cols)
    tri = {"seq": [], "j": [], "q": [], **{col: [] for col in cols}}
    stored = {col: [] for col in cols}
    stored_seq = []
    counts = {}
    n_skipped = 0
    tok_pos = 0
    t0 = time.monotonic()
    keep_all = cells_sequences == 0
    for s in range(len(store)):
        ids_np = store.get(s)
        if len(ids_np) - c < 2:  # no position pair to test inside the window
            n_skipped += 1
            continue
        ids = torch.as_tensor(np.asarray(ids_np), dtype=torch.long, device=device)
        seed_sequence(seed, split, s)
        res = probe_sequence(probe, noise, hf_model, adapter, ids, c, g, N, columns, amp)
        tok_pos += res.tok_pos
        merge_counts(counts, res.counts)
        window = np.asarray(ids_np[c:])
        j, q, lag, u, v, values = sequence_cells(res.matrices, window, vocab, max_lag)
        acc.add(u, v, lag, values)
        if keep_all or len(stored_seq) < cells_sequences:
            stored_seq.append(s)
            tj, tq, _ = triangle(res.matrices[cols[0]])
            tri["seq"].append(np.full(len(tj), s, dtype=np.int32))
            tri["j"].append(tj)
            tri["q"].append(tq)
            for col in cols:
                _, _, vals = triangle(res.matrices[col])
                tri[col].append(vals)
                stored[col].append(res.matrices[col])
        if (s + 1) % progress_every == 0:
            log(
                {
                    "event": "probe_progress",
                    "split": split,
                    "probe": probe,
                    "noise": noise,
                    "c": c,
                    "N": N,
                    "done": s + 1,
                    "of": len(store),
                    "tok_pos_per_s": tok_pos / max(time.monotonic() - t0, 1e-9),
                }
            )
    wall = time.monotonic() - t0
    table = {
        k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32 if k in cols else np.int32))
        for k, v in tri.items()
    }
    facts = {
        "n_sequences": len(store),
        "n_skipped_short": n_skipped,
        "n_probed": len(store) - n_skipped,
        "n_stored": len(stored_seq),
        "stored_seq": stored_seq,
        "n_cells": int(acc.n_cells),
        "tok_pos": int(tok_pos),
        "wall_clock_s": round(wall, 3),
        "tok_pos_per_s": tok_pos / max(wall, 1e-9),
        "memory": mem,
        "counts": counts,
        "columns": cols,
    }
    log(
        {
            "event": "probe_done",
            "split": split,
            "probe": probe,
            "noise": noise,
            "c": c,
            "N": N,
            **{
                k: v
                for k, v in facts.items()
                if k not in ("memory", "counts", "stored_seq", "columns")
            },
        }
    )
    return acc.result(), table, stored, facts


def write_matrices(path, table):
    np.savez_compressed(path, **table)


def sequences_record(store, stored_seq, c, grain, split):
    return {
        "grain": grain,
        "split": split,
        "context": int(c),
        "sequences": [
            {
                "seq": int(s),
                "trace_id": store.trace_ids[s],
                "length": int(store.lengths[s]),
                "n_spans": int(store.n_spans[s]),
            }
            for s in stored_seq
        ],
    }


# --- the prediction files of the pass's cells --------------------------------------------------------------------
def write_cell_outputs(
    out_dir,
    scores,
    cells,
    grain,
    vocab,
    per_lag_files,
    max_lag,
    stored,
    stored_ids,
    context,
    rule,
    **meta,
):
    """Per cell: the frozen cut's prediction + ranking + per-lag files, or the shipped cut's union
    prediction + its record. Returns `{cell_key: summary}`."""
    out_dir = Path(out_dir)
    n_lags = min(int(per_lag_files), int(max_lag))
    summary = {}
    for key, tau in cells.items():
        arm, path, cut, _ = parse_cell_key(key)
        slug = arm_slug(arm)
        col = f"{slug}__{path}"
        common = {"arm": arm, "path": path, "cut": cut, "grain": grain, **meta}
        if cut == CUT_FROZEN:
            src, dst, s = select_edges(scores, col, "max", tau)
            n = write_prediction(
                out_dir / PREDICTION_JSON_FMT.format(grain=grain, arm=slug, path=path, cut=cut),
                src,
                dst,
                s,
                vocab,
                kind="prediction",
                tau=float(tau),
                **common,
            )
            rs, rd, rv = ranking(scores, col, "max")
            write_prediction(
                out_dir / RANKING_JSON_FMT.format(grain=grain, arm=slug, path=path),
                rs,
                rd,
                rv,
                vocab,
                kind="ranking",
                **common,
            )
            for lag in range(1, n_lags + 1):
                ls, ld, lv = per_lag_ranking(scores, col, lag)
                write_prediction(
                    out_dir
                    / RANKING_LAG_JSON_FMT.format(grain=grain, arm=slug, path=path, lag=lag),
                    ls,
                    ld,
                    lv,
                    vocab,
                    kind="ranking",
                    lag=lag,
                    **common,
                )
                hit = lv > tau
                write_prediction(
                    out_dir
                    / PREDICTION_LAG_JSON_FMT.format(grain=grain, arm=slug, path=path, lag=lag),
                    ls[hit],
                    ld[hit],
                    lv[hit],
                    vocab,
                    kind="prediction",
                    lag=lag,
                    tau=float(tau),
                    **common,
                )
            summary[key] = {
                "cut": cut,
                "tau": float(tau),
                "n_edges": int(n),
                "n_pairs": int(len(rs)),
                "per_lag_files": n_lags,
            }
        else:
            mats = stored[col]
            if not mats:
                raise ValueError(
                    f"cell {key}: no stored matrices for the shipped cut (--cells-sequences must keep them)"
                )
            tau_by_lag, graphs, finite = shipped_cut(mats, rule)
            edges, facts = shipped_cut_union(stored_ids, graphs, context, vocab)
            src = np.array([u for u, _ in sorted(edges)], dtype=np.int64)
            dst = np.array([v for _, v in sorted(edges)], dtype=np.int64)
            n = write_prediction(
                out_dir / PREDICTION_JSON_FMT.format(grain=grain, arm=slug, path=path, cut=cut),
                src,
                dst,
                np.ones(len(src)),
                vocab,
                kind="prediction",
                tau=None,
                rule=rule,
                **common,
            )
            record = {
                "cell": key,
                "rule": rule,
                "rule_is_v019": is_v019_rule(rule),
                "tau_by_lag": tau_by_lag,
                "threshold_finite": finite,
                "n_sequences_pooled": len(mats),
                **facts,
            }
            write_json(
                out_dir / SHIPPED_CUT_READ_JSON_FMT.format(grain=grain, arm=slug, path=path), record
            )
            summary[key] = {
                "cut": cut,
                "tau": None,
                "n_edges": int(n),
                "threshold_finite": finite,
                "tau_by_lag": tau_by_lag,
                "n_sequences_pooled": len(mats),
                **facts,
            }
    return summary


# --- the freeze assertions (scenarios 13, 29) ----------------------------------------------------------------------
def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def freeze_commit(freeze_path):
    """`(sha, committed_at)` of the commit that added the freeze file; refuses an uncommitted or
    modified freeze."""
    path = Path(freeze_path).resolve()
    if not path.exists():
        raise FreezeRefusal(f"freeze {path} does not exist")
    try:
        top = Path(_git(["rev-parse", "--show-toplevel"], cwd=path.parent))
    except subprocess.CalledProcessError as e:
        raise FreezeRefusal(f"freeze {path} is not inside a git repository") from e
    rel = path.relative_to(top).as_posix()
    if _git(["status", "--porcelain", "--", rel], cwd=top):
        raise FreezeRefusal(
            f"freeze {rel} has uncommitted changes; commit it before the test read (PRD scenario 13)"
        )
    lines = _git(["log", "--diff-filter=A", "--format=%H %cI", "--", rel], cwd=top).splitlines()
    if not lines:
        raise FreezeRefusal(
            f"freeze {rel} is not committed; commit it before the test read (PRD scenario 13)"
        )
    sha, committed_at = lines[-1].split()  # the commit that first added the file
    ok = (
        subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=top).returncode == 0
    )
    if not ok:
        raise FreezeRefusal(f"freeze commit {sha[:8]} is not an ancestor of HEAD (PRD scenario 13)")
    return sha, committed_at


def _iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def assert_freeze(freeze, freeze_path, cells, c, N, g, model_sha256, corpus_id, started):
    """The freeze names this run's values exactly and predates it (scenarios 13, 29)."""
    sha, committed_at = freeze_commit(freeze_path)
    if not _iso(committed_at) < _iso(started):
        raise FreezeRefusal(
            f"the run started at {started} but the freeze was committed at {committed_at} (PRD scenario 13)"
        )
    for key, tau in cells.items():
        cell = freeze.get("cells", {}).get(key)
        if cell is None:
            raise FreezeRefusal(f"freeze has no cell {key}")
        want = {"c": int(c), "N": int(N), "g": int(g)}
        if tau != "shipped":
            want["tau"] = float(tau)
        got = {k: cell.get(k) for k in want}
        if got != want:
            raise FreezeRefusal(
                f"freeze cell {key} = {got} but the command line says {want} (PRD scenario 13)"
            )
    if freeze.get("model_sha256") != model_sha256:
        raise FreezeRefusal(
            f"freeze binds model {freeze.get('model_sha256')} but the loaded model hashes to {model_sha256}"
        )
    if freeze.get("corpus_id") != corpus_id:
        raise FreezeRefusal(
            f"freeze is for corpus {freeze.get('corpus_id')!r}, this run reads {corpus_id!r}"
        )
    return {"sha": sha, "committed_at": committed_at, "path": str(freeze_path)}


# --- the command ------------------------------------------------------------------------------------------------------
def run_discover(args, rec):
    if args.split == "test" and not args.freeze:
        raise FreezeRefusal(
            "a test read needs --freeze <committed freeze record> (PRD scenario 13)"
        )
    staging = require_prior_rung(args.prior_rung_record, args.rung, "discover", args.staging_ledger)
    rec.note(replica=args.replica, aws_profile=args.aws_profile_name)
    hf_model, adapter, model_sha = load_backbone(args.model, args.device)
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    if "/" in corpus.corpus_id and not corpus.corpus_id.startswith(args.rung + "/"):
        raise ValueError(f"corpus {corpus.corpus_id!r} is not at rung {args.rung!r}")
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    if vocab.size != hf_model.config.vocab_size:
        raise ValueError(
            f"the view's vocabulary has {vocab.size} ids but the model was trained on {hf_model.config.vocab_size}"
        )
    cells = parse_cells(args.cells, args.grain, args.probe, args.noise)
    rule = shipped_rule(args)
    freeze_facts = None
    if args.freeze:
        freeze = read_json(args.freeze)
        freeze_facts = assert_freeze(
            freeze,
            args.freeze,
            cells,
            args.context,
            args.particles,
            args.guidance,
            model_sha,
            corpus.corpus_id,
            rec.started,
        )
        log({"event": "freeze_ok", **freeze_facts})
    store = SequenceStore.from_corpus(
        corpus,
        args.split,
        vocab,
        args.max_len,
        n=args.num_sequences,
        mode=args.sequence_sample,
        seed=args.seed,
    )
    scores, table, stored, facts = probe_store(
        hf_model,
        adapter,
        vocab,
        store,
        args.probe,
        args.noise,
        args.context,
        args.guidance,
        args.particles,
        args.max_lag,
        args.memory_cap_gb,
        args.probe_amp,
        args.seed,
        args.split,
        args.cells_sequences,
    )
    out = rec.out_dir
    write_matrices(
        out / MATRICES_NPZ_FMT.format(probe=args.probe, noise=args.noise, grain=args.grain), table
    )
    write_json(
        out / SEQUENCES_JSON_FMT.format(grain=args.grain),
        sequences_record(store, facts["stored_seq"], args.context, args.grain, args.split),
    )
    meta = {
        "probe": args.probe,
        "noise": args.noise,
        "c": args.context,
        "N": args.particles,
        "g": args.guidance,
        "grain": args.grain,
        "split": args.split,
        "ordering": args.ordering,
        "model_sha256": model_sha,
        "corpus_id": corpus.corpus_id,
        "rung": args.rung,
        "n_sequences": facts["n_probed"],
        "max_lag": args.max_lag,
        "probe_amp": args.probe_amp,
    }
    write_scores_npz(
        out / SCORES_NPZ_FMT.format(probe=args.probe, noise=args.noise, grain=args.grain),
        scores,
        **meta,
    )
    stored_ids = [store.get(s) for s in facts["stored_seq"]]
    summary = write_cell_outputs(
        out,
        scores,
        cells,
        args.grain,
        vocab,
        args.per_lag_files,
        args.max_lag,
        stored,
        stored_ids,
        args.context,
        rule,
        c=args.context,
        N=args.particles,
        g=args.guidance,
        split=args.split,
        model_sha256=model_sha,
        corpus_id=corpus.corpus_id,
        n_sequences=facts["n_probed"],
        probe=args.probe,
        noise=args.noise,
    )
    rec.note(tok_pos_per_s=facts["tok_pos_per_s"])
    return {
        "stage": "discover",
        "probe": args.probe,
        "noise": args.noise,
        "grain": args.grain,
        "split": args.split,
        "rung": args.rung,
        "corpus_id": corpus.corpus_id,
        "model_sha256": model_sha,
        "c": args.context,
        "N": args.particles,
        "g": args.guidance,
        "probe_amp": args.probe_amp,
        "columns": facts["columns"],
        "n_sequences": facts["n_sequences"],
        "n_probed": facts["n_probed"],
        "n_skipped_short": facts["n_skipped_short"],
        "n_stored": facts["n_stored"],
        "n_cells": facts["n_cells"],
        "n_pairs": int(len(scores["src"])),
        "corrupted_cells": facts["counts"],
        "memory": facts["memory"],
        "tok_pos": facts["tok_pos"],
        "tok_pos_per_s": facts["tok_pos_per_s"],
        "probe_wall_clock_s": facts["wall_clock_s"],
        "cells": summary,
        "shipped_rule": rule,
        "freeze": freeze_facts,
        "staging": staging,
        "files": {
            "matrices": MATRICES_NPZ_FMT.format(
                probe=args.probe, noise=args.noise, grain=args.grain
            ),
            "scores": SCORES_NPZ_FMT.format(probe=args.probe, noise=args.noise, grain=args.grain),
            "sequences": SEQUENCES_JSON_FMT.format(grain=args.grain),
        },
    }


def add_probe_knobs(p):
    p.add_argument(
        "--max-len", required=True, type=int, help="cap on real tokens per sequence (D-SB-7)"
    )
    p.add_argument(
        "--max-lag", required=True, type=int, help="lag bound of the type-level projection"
    )
    p.add_argument("--sequence-sample", required=True, choices=SEQUENCE_SAMPLES, help="D-SB-14")
    p.add_argument(
        "--memory-cap-gb",
        required=True,
        type=float,
        help="the declared cap of plans/caps.md (scenario 18)",
    )
    p.add_argument(
        "--probe-amp",
        required=True,
        choices=AMP_MODES,
        help="autocast of the probe's forwards; 'none' is the reference",
    )
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--prior-rung-record", required=True, help="see staging.py (scenario 46)")
    p.add_argument("--staging-ledger", required=True, help="the tracked plans/staging-ledger.json")
    p.add_argument(
        "--replica",
        required=True,
        help="the dated provisioning replica, by file name (scenario 25); 'local-test' locally",
    )
    p.add_argument(
        "--aws-profile-name",
        required=True,
        help="the AWS profile name (scenario 30); 'none' locally",
    )
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", required=True, choices=DEVICES)
    p.add_argument("--output-folder", required=True)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--model", required=True, help="the frozen backbone directory (D-SB-7)")
    p.add_argument("--probe", required=True, choices=PROBES)
    p.add_argument("--noise", required=True, choices=NOISES)
    p.add_argument("--context", required=True, type=int, help="c, BOS-counted")
    p.add_argument("--guidance", required=True, type=int, help="g, 1 <= g <= c (core probe)")
    p.add_argument(
        "--particles", required=True, type=int, help="N; 0 for the saliency and Shapley probes"
    )
    p.add_argument(
        "--cells", required=True, nargs="+", help="KEY=TAU or KEY=shipped per cell of this pass"
    )
    add_shipped_rule_knobs(p)
    p.add_argument(
        "--num-sequences", required=True, type=int, help="sequences probed; 0 = the whole split"
    )
    p.add_argument(
        "--cells-sequences",
        required=True,
        type=int,
        help="sequences whose matrices are stored; 0 = all",
    )
    p.add_argument(
        "--per-lag-files",
        required=True,
        type=int,
        help="per-lag ranking/prediction files for lags 1..K",
    )
    p.add_argument(
        "--freeze", required=True, help="the committed freeze record, or '' (allowed on val only)"
    )
    add_probe_knobs(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.freeze = args.freeze or None
    with RunRecord(args.output_folder, "discover", vars(args)) as rec:
        res = run_discover(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
