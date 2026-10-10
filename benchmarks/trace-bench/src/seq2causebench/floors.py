"""The two model-free floors of `plans/floors.md` (pre-registered 2026-10-09): one validation or
test read of one corpus, grain, split and view ordering for `floor/topology` and / or
`floor/bigram`.

    python -m seq2causebench.floors --corpus <dir> --arms floor/topology floor/bigram \
        --ordering end --grain request --split val --rung xs --num-sequences 0 \
        --sequence-sample head --max-len 64 --seed 0 --cells none --freeze '' \
        --prior-rung-record smallest-rung --staging-ledger plans/staging-ledger.json \
        --replica local --aws-profile-name none --output-folder out/<run>

`floor/bigram` is a method-side read: the views only, through `Corpus`, on the arms' sample
(`--num-sequences`, `--sequence-sample`, `--max-len`; D-SB-14, D-SB-7). Its score of `a -> b` is
the share of sampled sequences containing `a` in which `b` immediately follows `a`, within-op
pairs dropped (D-SB-10). `floor/topology` reads the shipped deployment-topology prior on the
score side (`floorprior`), never a view and never through `Corpus`. Each arm's table is written
in the sweep's shape (`scores-<probe>-none-<grain>.npz`: one column, `c = N = g = 0`, the model
hash "none") with its own token list, so `scoresweep` sweeps the floor grid and `freeze` fixes a
τ per cell (a separate `…-floors.json`). On `--split test` the read asserts the committed floor
freeze (family, cells, τ, ordering, corpus) and writes the prediction and ranking files of every
`--cells` entry for `annotate`. The staging rule names the arms' previous-rung ledger entry
(`sweep` on a validation read, `discover` on a test read), so rungs run independently.
"""

from __future__ import annotations

import argparse

import numpy as np

from .arms import arm_slug, column_of, parse_cell_key, probe_of
from .constants import (
    ARM_FLOOR_BIGRAM,
    ARM_FLOOR_TOPOLOGY,
    CUT_FROZEN,
    FAMILY_FLOOR,
    FLOOR_ARMS,
    FLOOR_STAGE,
    GRAINS,
    MODEL_NONE,
    NOISE_NONE,
    ORDERINGS,
    OUTCOME_NAMES,
    PATH_NONE,
    PREDICTION_JSON_FMT,
    RANKING_JSON_FMT,
    RUNGS,
    SCORES_NPZ_FMT,
    SEQUENCE_SAMPLES,
    SEQUENCES_DIR,
    SPLITS,
)
from .corpus import Corpus, corpus_id_of
from .data import select_sequences
from .floorprior import topology_read
from .freezecheck import FreezeRefusal, assert_freeze
from .log import log
from .prediction import TokenTable, write_prediction
from .project import write_scores_npz
from .record import RunRecord, read_json
from .select import ranking, select_edges
from .staging import require_prior_rung

OUTCOME_BASE = 8  # token key = op * 8 + outcome class (five classes, room to spare)
NO_CELLS = "none"  # --cells on a validation read
STAGE_OF_SPLIT = {"val": "sweep", "test": "discover"}  # the arms' ledger stage a floor read names


def bigram_read(corpus, split, n, mode, seed, max_len):
    """The order-count floor's table over the sampled sequences of one split (plans/floors.md
    §2): `n_pair(a, b)` sequences with `b` immediately after `a`, `n_tok(a)` sequences with `a`."""
    rng = np.random.default_rng(seed)
    seqs = select_sequences(corpus.sequences(split), n, mode, rng)
    tok_chunks, pair_chunks, n_short = [], [], 0
    for _tid, ops, outs, _n in seqs:
        t = np.asarray(ops[:max_len], dtype=np.int64) * OUTCOME_BASE + np.asarray(
            outs[:max_len], dtype=np.int64
        )
        tok_chunks.append(np.unique(t))
        if len(t) < 2:
            n_short += 1
            continue
        pair_chunks.append(np.unique(np.stack([t[:-1], t[1:]], axis=1), axis=0))
    empty = np.zeros(0, dtype=np.int64)
    if tok_chunks:
        toks, n_with = np.unique(np.concatenate(tok_chunks), return_counts=True)
    else:
        toks, n_with = empty, empty
    if pair_chunks:
        pairs, n_pair = np.unique(np.concatenate(pair_chunks, axis=0), axis=0, return_counts=True)
    else:
        pairs, n_pair = np.zeros((0, 2), dtype=np.int64), empty
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (a // OUTCOME_BASE) != (b // OUTCOME_BASE)  # within-op pairs are dropped (D-SB-10)
    a, b, n_pair = a[keep], b[keep], n_pair[keep]
    src, dst = np.searchsorted(toks, a), np.searchsorted(toks, b)
    score = n_pair / n_with[src] if len(src) else np.zeros(0)
    order = np.lexsort((dst, src))
    tokens = np.array(
        [f"{int(k) // OUTCOME_BASE}:{OUTCOME_NAMES[int(k) % OUTCOME_BASE]}" for k in toks]
    )
    return {
        "tokens": tokens,
        "src": src[order].astype(np.int64),
        "dst": dst[order].astype(np.int64),
        "score": np.asarray(score, dtype=np.float64)[order],
        "count": n_pair[order].astype(np.int64),
        "facts": {
            "n_sequences": int(len(seqs)),
            "n_skipped_short": int(n_short),
            "n_tokens": int(len(toks)),
            "inputs": [f"{corpus.view_rel}/{SEQUENCES_DIR}/split={split}"],
        },
    }


def floor_scores(table, col):
    """The sweep-shaped score table of one floor arm: one column whose max and mean are the pair's
    score, a one-lag per-lag column, and the table's own token list."""
    s = np.asarray(table["score"], dtype=np.float64)
    return {
        "src": np.asarray(table["src"], dtype=np.int64),
        "dst": np.asarray(table["dst"], dtype=np.int64),
        "count": np.asarray(table["count"], dtype=np.int64),
        f"max_{col}": s,
        f"mean_{col}": s.copy(),
        f"lag_max_{col}": s.reshape(-1, 1).copy(),
        "tokens": np.asarray(table["tokens"]),
    }


def parse_floor_cells(specs, grain, arms):
    """`{cell_key: tau}` from `KEY=TAU` specs over this read's arms; `none` alone on a validation read."""
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
            raise ValueError(f"cell {key}: a floor has the frozen cut only")
        if key in cells:
            raise ValueError(f"cell {key} given twice")
        cells[key] = float(val)
    return cells


def write_floor_cells(out_dir, tables, cells, grain, **meta):
    """The prediction and ranking files of every cell (the scorer's shape, D-SB-10)."""
    summary = {}
    for key, tau in cells.items():
        arm, path, cut, _ = parse_cell_key(key)
        scores, col = tables[arm]
        mapper = TokenTable(scores["tokens"])
        slug = arm_slug(arm)
        common = {"arm": arm, "path": path, "cut": cut, "grain": grain, **meta}
        src, dst, s = select_edges(scores, col, "max", tau)
        n = write_prediction(
            out_dir / PREDICTION_JSON_FMT.format(grain=grain, arm=slug, path=path, cut=cut),
            src,
            dst,
            s,
            mapper,
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
            mapper,
            kind="ranking",
            **common,
        )
        summary[key] = {
            "cut": cut,
            "tau": float(tau),
            "n_edges": int(n),
            "n_pairs": int(len(rs)),
            "per_lag_files": 0,
        }
    return summary


def run_floors(args, rec):
    if args.split == "test" and not args.freeze:
        raise FreezeRefusal("a test read needs --freeze <committed floor freeze> (PRD scenario 13)")
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
    cells = parse_floor_cells(args.cells, args.grain, arms)
    freeze_facts = None
    if args.freeze:
        freeze = read_json(args.freeze)
        if freeze.get("family") != FAMILY_FLOOR:
            raise FreezeRefusal(
                f"freeze {args.freeze} is not a floor freeze (family {freeze.get('family')!r}; D-SB-16)"
            )
        freeze_facts = assert_freeze(
            freeze,
            args.freeze,
            cells,
            0,
            0,
            0,
            MODEL_NONE,
            corpus_id,
            rec.started,
            args.ordering,
        )
        log({"event": "freeze_ok", **freeze_facts})
    out = rec.out_dir
    tables, inputs, facts, files = {}, {}, {}, {}
    for arm in arms:
        probe = probe_of(arm)
        if arm == ARM_FLOOR_TOPOLOGY:
            table = topology_read(args.corpus)
        elif arm == ARM_FLOOR_BIGRAM:
            corpus = Corpus(args.corpus, args.ordering, args.grain)
            table = bigram_read(
                corpus,
                args.split,
                args.num_sequences,
                args.sequence_sample,
                args.seed,
                args.max_len,
            )
        else:
            raise ValueError(f"arm {arm!r} is not a floor arm {FLOOR_ARMS}")
        col = column_of(arm, PATH_NONE)
        scores = floor_scores(table, col)
        name = SCORES_NPZ_FMT.format(probe=probe, noise=NOISE_NONE, grain=args.grain)
        write_scores_npz(
            out / name,
            scores,
            probe=probe,
            noise=NOISE_NONE,
            c=0,
            N=0,
            g=0,
            grain=args.grain,
            split=args.split,
            ordering=args.ordering,
            model_sha256=MODEL_NONE,
            corpus_id=corpus_id,
            rung=args.rung,
            n_sequences=int(table["facts"].get("n_sequences", 0)),
        )
        tables[arm] = (scores, col)
        inputs[arm] = list(table["facts"]["inputs"])
        facts[arm] = {k: v for k, v in table["facts"].items() if k != "inputs"}
        facts[arm]["n_pairs"] = int(len(scores["src"]))
        files[arm] = name
        log({"event": "floor_read", "arm": arm, "grain": args.grain, **facts[arm]})
    summary = write_floor_cells(
        out,
        tables,
        cells,
        args.grain,
        split=args.split,
        model_sha256=MODEL_NONE,
        corpus_id=corpus_id,
    )
    sampled = facts.get(ARM_FLOOR_BIGRAM, {})
    return {
        "stage": FLOOR_STAGE,
        "arms": arms,
        "probe": None,
        "noise": NOISE_NONE,
        "grain": args.grain,
        "split": args.split,
        "ordering": args.ordering,
        "rung": args.rung,
        "corpus_id": corpus_id,
        "model_sha256": MODEL_NONE,
        "c": 0,
        "N": 0,
        "g": 0,
        "probe_amp": None,
        "max_len": args.max_len,
        "n_sequences": int(sampled.get("n_sequences", 0)),
        "n_probed": int(sampled.get("n_sequences", 0)),
        "n_skipped_short": int(sampled.get("n_skipped_short", 0)),
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
    p.add_argument("--arms", required=True, nargs="+", choices=FLOOR_ARMS)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument(
        "--num-sequences",
        required=True,
        type=int,
        help="sequences the bigram floor reads; 0 = the whole split (D-SB-14)",
    )
    p.add_argument("--sequence-sample", required=True, choices=SEQUENCE_SAMPLES, help="D-SB-14")
    p.add_argument(
        "--max-len", required=True, type=int, help="cap on real tokens per sequence (D-SB-7)"
    )
    p.add_argument("--seed", required=True, type=int, help="the sample's seed (uniform mode)")
    p.add_argument(
        "--cells",
        required=True,
        nargs="+",
        help=f"KEY=TAU per cell of a test read, or {NO_CELLS!r} on a validation read",
    )
    p.add_argument(
        "--freeze",
        required=True,
        help="the committed floor freeze record, or '' (allowed on val only)",
    )
    p.add_argument("--prior-rung-record", required=True, help="see staging.py (scenario 46)")
    p.add_argument("--staging-ledger", required=True, help="the tracked plans/staging-ledger.json")
    p.add_argument(
        "--replica",
        required=True,
        help="the dated provisioning replica, by file name (scenario 25); 'local' locally",
    )
    p.add_argument(
        "--aws-profile-name",
        required=True,
        help="the AWS profile name (scenario 30); 'none' locally",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.freeze = args.freeze or None
    with RunRecord(args.output_folder, FLOOR_STAGE, vars(args)) as rec:
        res = run_floors(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
