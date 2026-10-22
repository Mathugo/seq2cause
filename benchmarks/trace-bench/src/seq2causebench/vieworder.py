"""The view-order statistic of `plans/floors.md` (addendum 2026-10-22): over the sampled
sequences of one split, the share of co-occurrences of a directed truth edge in which the cause
token comes before the effect token in the view's row order.

    python -m seq2causebench.vieworder --corpus <dir> --ordering start --grain request \
        --split test --num-sequences 0 --sequence-sample head --max-len 64 --seed 0 \
        --output-folder results-vieworder/<rung>/latent/seed=<k>/start-request

Score side, model-free, no staging: it reads the view rows through `Corpus` on the arms' test
sample (`--num-sequences`, `--sequence-sample`, `--max-len`; D-SB-14, D-SB-7), tokens
`<op>:<outcome>` as the floors build them, and the grain's scoring target at its default floor.
In a sequence every ordered pair of positions `(i, j)`, `i != j`, holding the cause and the effect
of one directed truth edge is one co-occurrence of that edge, "cause first" when `i < j`. The
headline is pooled over every truth edge and sequence; the per-edge mean is recorded beside it.
It describes the view, not an arm: reported beside every request-grain orientation number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tracebench.constants import GRAPHS_DIR, SCORING_TARGET_JSON
from tracebench.score import SCORING_TARGET_SESSION_JSON, truth_sets

from .constants import (
    GRAINS,
    ORDERINGS,
    OUTCOME_NAMES,
    SEQUENCE_SAMPLES,
    SEQUENCES_DIR,
    SPLITS,
)
from .corpus import Corpus
from .data import select_sequences
from .floors import OUTCOME_BASE
from .log import log
from .record import RunRecord, read_json, write_json

VIEWORDER_STAGE = "vieworder"
VIEWORDER_JSON = "vieworder.json"
VIEWORDER_SCHEMA = "seq2causebench/vieworder@1"
PAIR_BASE = 1 << 32  # pair key = cause key * PAIR_BASE + effect key
OUTCOME_CLASS = {name: k for k, name in OUTCOME_NAMES.items()}


def token_key(token):
    """`<op>:<outcome>` → `op * OUTCOME_BASE + outcome class`, the floors' integer token."""
    op, outcome = token.rsplit(":", 1)
    return int(op) * OUTCOME_BASE + OUTCOME_CLASS[outcome]


def directed_truth(corpus_dir, grain):
    """The grain's scoring target, its default floor and its directed truth edges as pair keys."""
    gdir = Path(corpus_dir) / GRAPHS_DIR
    target = read_json(
        gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON)
    )
    floor = float(target["default_floor"])
    directed, _bidirected = truth_sets(target, floor)
    keys = np.array(
        sorted(token_key(a) * PAIR_BASE + token_key(b) for a, b in directed), dtype=np.int64
    )
    return keys, floor


def view_order(corpus, split, n, mode, seed, max_len, truth_keys):
    """Cause-first and cause-second co-occurrence counts per directed truth edge over the sample."""
    rng = np.random.default_rng(seed)
    seqs = select_sequences(corpus.sequences(split), n, mode, rng)
    first_chunks, second_chunks, n_with, triu = [], [], 0, {}
    for _tid, ops, outs, _n in seqs:
        t = np.asarray(ops[:max_len], dtype=np.int64) * OUTCOME_BASE + np.asarray(
            outs[:max_len], dtype=np.int64
        )
        if len(t) < 2:
            continue
        i, j = triu.setdefault(len(t), np.triu_indices(len(t), k=1))
        forward = t[i] * PAIR_BASE + t[j]  # earlier token → later token
        backward = t[j] * PAIR_BASE + t[i]
        first = forward[np.isin(forward, truth_keys)]
        second = backward[np.isin(backward, truth_keys)]
        if len(first) or len(second):
            n_with += 1
            first_chunks.append(first)
            second_chunks.append(second)
    empty = np.zeros(0, dtype=np.int64)
    first = np.concatenate(first_chunks) if first_chunks else empty
    second = np.concatenate(second_chunks) if second_chunks else empty
    edges, inverse = np.unique(np.concatenate([first, second]), return_inverse=True)
    n_first = np.bincount(inverse[: len(first)], minlength=len(edges))
    n_both = np.bincount(inverse, minlength=len(edges))
    total = int(n_both.sum())
    return {
        "share_cause_first": (int(n_first.sum()) / total) if total else None,
        "n_cause_first": int(n_first.sum()),
        "n_cooccurrences": total,
        "n_truth_edges": int(len(truth_keys)),
        "n_truth_edges_cooccurring": int(len(edges)),
        "edge_mean_share": float(np.mean(n_first / n_both)) if len(edges) else None,
        "n_sequences": int(len(seqs)),
        "n_sequences_with_cooccurrence": int(n_with),
    }


def run_vieworder(args, rec):
    corpus = Corpus(args.corpus, args.ordering, args.grain)
    truth_keys, floor = directed_truth(args.corpus, args.grain)
    stats = view_order(
        corpus,
        args.split,
        args.num_sequences,
        args.sequence_sample,
        args.seed,
        args.max_len,
        truth_keys,
    )
    target = SCORING_TARGET_JSON if args.grain == "request" else SCORING_TARGET_SESSION_JSON
    result = {
        "schema": VIEWORDER_SCHEMA,
        "corpus_id": corpus.corpus_id,
        "ordering": args.ordering,
        "grain": args.grain,
        "split": args.split,
        "floor": floor,
        "num_sequences": args.num_sequences,
        "sequence_sample": args.sequence_sample,
        "max_len": args.max_len,
        "seed": args.seed,
        "inputs": [
            f"{corpus.view_rel}/{SEQUENCES_DIR}/split={args.split}",
            f"{GRAPHS_DIR}/{target}",
        ],
        **stats,
    }
    write_json(rec.out_dir / VIEWORDER_JSON, result)
    log(
        {
            "event": "vieworder_done",
            "corpus_id": corpus.corpus_id,
            "view": corpus.view,
            "split": args.split,
            "share_cause_first": stats["share_cause_first"],
            "n_cooccurrences": stats["n_cooccurrences"],
        }
    )
    return {"vieworder": VIEWORDER_JSON, **stats}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument(
        "--num-sequences",
        required=True,
        type=int,
        help="the arms' test-read sample; 0 = the whole split (D-SB-14)",
    )
    p.add_argument("--sequence-sample", required=True, choices=SEQUENCE_SAMPLES, help="D-SB-14")
    p.add_argument(
        "--max-len", required=True, type=int, help="cap on real tokens per sequence (D-SB-7)"
    )
    p.add_argument("--seed", required=True, type=int, help="the sample's seed (uniform mode)")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, VIEWORDER_STAGE, vars(args)) as rec:
        res = run_vieworder(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
