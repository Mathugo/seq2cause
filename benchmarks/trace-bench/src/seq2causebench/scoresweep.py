# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Score every sweep cell at every τ with the benchmark's scorer (PRD Interface "scoresweep";
non-negotiable 8; score side).

    python -m seq2causebench.scoresweep --corpus <dir> --sweep-dir out/<sweep-run> --grains request \
        --taus 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 \
        --granger-taus 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1 \
        --quantiles 0.5 0.8 0.9 0.95 0.99 --output-folder out/<run>

Runs only after the method process has exited and the score tier (`graphs/`) has been pulled.
No metric arithmetic of its own: the target and alphabet are loaded once per grain, the universe
built once, and `tracebench.score.score_at_floor` is called at the target's default floor for
every (column, τ). The τ grid is per arm class (`plans/reference-arms.md`, `plans/baselines.md`):
absolute for the divergence arms and Granger, label-free quantiles of the pooled validation scores
for saliency and Shapley (the absolute value is recorded). Per column the full ranking is scored
once more for the threshold-free axes and the coverage rule (the recall of the full ranking is the
fraction of truth edges that co-occur in the probed sample). The cli probes' shipped-cut unions
are scored as their own rows (`cut = shipped`, no τ).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tracebench.constants import ALPHABET_JSON, GRAPHS_DIR, SCORING_TARGET_JSON
from tracebench.score import SCORING_TARGET_SESSION_JSON, build_universe, score_at_floor

from .arms import cell_key
from .constants import (
    ARM_GRANGER,
    ARM_SALIENCY,
    ARM_SHAPLEY,
    CUT_FROZEN,
    CUT_SHIPPED,
    GRAINS,
    MANIFEST_JSON,
    REFERENCE_ARMS,
    VAL_TABLE_JSON,
)
from .corpus import Corpus
from .engine import probe_columns
from .log import log
from .prediction import prediction_document
from .project import read_scores_npz
from .record import RunRecord, read_json, write_json
from .select import ranking, select_edges
from .vocab import Vocab

VAL_TABLE_SCHEMA = "seq2causebench/val-table@1"


def load_target(corpus_dir, grain):
    gdir = Path(corpus_dir) / GRAPHS_DIR
    target = read_json(
        gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON)
    )
    alphabet = read_json(gdir / ALPHABET_JSON)
    ordered, unordered = build_universe(alphabet, target)
    return target, alphabet, ordered, unordered


def manifest_facts(corpus_dir):
    m = read_json(Path(corpus_dir) / MANIFEST_JSON)
    return {
        "tool_version": m.get("tool_version"),
        "config_hash": m.get("config_hash"),
        "instance": m.get("instance"),
        "variant": m.get("variant"),
        "seed": m.get("seed"),
    }


def sweep_files(sweep_dir, grain):
    return sorted(Path(sweep_dir).glob(f"scores-*-{grain}.npz"))


def _strip(r):
    return {
        "directed": r["directed"],
        "skeleton": r["skeleton"],
        "orientation": r["orientation"],
        "shd_mixed": r["shd_mixed"],
        "predictions_outside_universe": r["universe"]["predictions_outside_universe"],
    }


def score_cell(scores, col, tau, vocab, target, alphabet, floor, ordered, unordered):
    src, dst, s = select_edges(scores, col, "max", tau)
    doc = prediction_document(src, dst, s, vocab)
    r = score_at_floor(target, alphabet, doc, floor, ordered, unordered)
    return {"n_edges": int(len(src)), **_strip(r)}


def score_edges(edges, vocab, target, alphabet, floor, ordered, unordered):
    src = np.array([u for u, _ in edges], dtype=np.int64)
    dst = np.array([v for _, v in edges], dtype=np.int64)
    doc = prediction_document(src, dst, np.ones(len(src)), vocab)
    r = score_at_floor(target, alphabet, doc, floor, ordered, unordered)
    return {"n_edges": int(len(src)), **_strip(r)}


def score_ranking(scores, col, vocab, target, alphabet, floor, ordered, unordered):
    src, dst, s = ranking(scores, col, "max")
    doc = prediction_document(src, dst, s, vocab)
    r = score_at_floor(target, alphabet, doc, floor, ordered, unordered)
    inside = int(len(src)) - int(r["universe"]["predictions_outside_universe"])
    return {
        "n_pairs": int(len(src)),
        "auroc": r["auroc"],
        "average_precision": r["average_precision"],
        "coverage": {
            "universe_ordered_pairs": r["universe"]["ordered_pairs"],
            "pairs_cooccurring": inside,
            "universe_fraction_cooccurring": inside / r["universe"]["ordered_pairs"]
            if r["universe"]["ordered_pairs"]
            else None,
            "truth_directed": r["universe"]["truth_directed"],
            "reachable_recall_ceiling": r["directed"]["recall"],
        },
    }


def taus_for(arm, scores, col, args):
    """`[(tau, source)]`: the absolute grid of the arm's class, or quantiles of the pooled scores."""
    if arm in REFERENCE_ARMS:
        return [(float(t), "grid") for t in args.taus]
    if arm == ARM_GRANGER:
        return [(float(t), "grid") for t in args.granger_taus]
    if arm in (ARM_SALIENCY, ARM_SHAPLEY):
        vals = np.asarray(scores[f"max_{col}"], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return []
        return [
            (float(np.quantile(vals, q)), f"quantile p{int(round(q * 100))}")
            for q in args.quantiles
        ]
    raise ValueError(f"no tau grid for arm {arm!r}")


def run_scoresweep(args, rec):
    facts = manifest_facts(args.corpus)
    rows, coverage = [], {}
    model_sha = corpus_id = None
    for grain in args.grains:
        target, alphabet, ordered, unordered = load_target(args.corpus, grain)
        floor = float(target["default_floor"])
        files = sweep_files(args.sweep_dir, grain)
        if not files:
            raise FileNotFoundError(f"no scores-*-{grain}.npz under {args.sweep_dir}")
        for f in files:
            scores = read_scores_npz(f)
            probe, noise = str(scores["probe"]), str(scores["noise"])
            c, N, g = int(scores["c"]), int(scores["N"]), int(scores["g"])
            ordering = str(scores["ordering"])
            model_sha = model_sha or str(scores["model_sha256"])
            corpus_id = corpus_id or str(scores["corpus_id"])
            if str(scores["model_sha256"]) != model_sha:
                raise ValueError(
                    f"{f.name}: mixed models in one sweep ({scores['model_sha256']} vs {model_sha})"
                )
            vocab = Vocab.from_model_vocab(Corpus(args.corpus, ordering, grain).vocab_json())
            shipped = None
            sc_path = f.with_name(f.name.replace("scores-", "shippedcut-").replace(".npz", ".json"))
            if sc_path.exists():
                shipped = read_json(sc_path)
            for arm, paths in probe_columns(probe, noise).items():
                for path, col in paths.items():
                    key = cell_key(arm, path, CUT_FROZEN, grain)
                    coverage[f"{key}/c{c}/N{N}"] = score_ranking(
                        scores, col, vocab, target, alphabet, floor, ordered, unordered
                    )
                    for tau, source in taus_for(arm, scores, col, args):
                        rows.append(
                            {
                                "arm": arm,
                                "path": path,
                                "cut": CUT_FROZEN,
                                "grain": grain,
                                "probe": probe,
                                "noise": noise,
                                "c": c,
                                "N": N,
                                "g": g,
                                "tau": tau,
                                "tau_source": source,
                                "floor": floor,
                                **score_cell(
                                    scores,
                                    col,
                                    tau,
                                    vocab,
                                    target,
                                    alphabet,
                                    floor,
                                    ordered,
                                    unordered,
                                ),
                            }
                        )
                    if shipped is not None and f"{arm}/{path}" in shipped["cells"]:
                        sc = shipped["cells"][f"{arm}/{path}"]
                        edges = [(int(u), int(v)) for u, v in sc["edges"]]
                        rows.append(
                            {
                                "arm": arm,
                                "path": path,
                                "cut": CUT_SHIPPED,
                                "grain": grain,
                                "probe": probe,
                                "noise": noise,
                                "c": c,
                                "N": N,
                                "g": g,
                                "tau": None,
                                "tau_source": "shipped rule",
                                "floor": floor,
                                "threshold_finite": sc["threshold_finite"],
                                "tau_by_lag": sc["tau_by_lag"],
                                **score_edges(
                                    edges, vocab, target, alphabet, floor, ordered, unordered
                                ),
                            }
                        )
            log(
                {
                    "event": "scoresweep_file",
                    "file": f.name,
                    "grain": grain,
                    "probe": probe,
                    "noise": noise,
                    "c": c,
                    "N": N,
                }
            )
    table = {
        "schema": VAL_TABLE_SCHEMA,
        "corpus_id": corpus_id,
        "model_sha256": model_sha,
        **facts,
        "sweep_dir_name": Path(args.sweep_dir).name,
        "grains": list(args.grains),
        "taus": [float(t) for t in args.taus],
        "granger_taus": [float(t) for t in args.granger_taus],
        "quantiles": [float(q) for q in args.quantiles],
        "n_rows": len(rows),
        "coverage": coverage,
        "cells": rows,
    }
    write_json(rec.out_dir / VAL_TABLE_JSON, table)
    return {
        "val_table": VAL_TABLE_JSON,
        "n_rows": len(rows),
        "corpus_id": corpus_id,
        "model_sha256": model_sha,
        **facts,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corpus", required=True, help="a corpus with its score tier pulled (graphs/ present)"
    )
    p.add_argument("--sweep-dir", required=True, help="the sweep run's output folder")
    p.add_argument("--grains", required=True, nargs="+", choices=GRAINS)
    p.add_argument(
        "--taus", required=True, nargs="+", type=float, help="the τ grid of the reference arms"
    )
    p.add_argument(
        "--granger-taus",
        required=True,
        nargs="+",
        type=float,
        help="the τ grid of the Granger baseline",
    )
    p.add_argument(
        "--quantiles",
        required=True,
        nargs="+",
        type=float,
        help="label-free quantiles for saliency / Shapley",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "scoresweep", vars(args)) as rec:
        res = run_scoresweep(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
