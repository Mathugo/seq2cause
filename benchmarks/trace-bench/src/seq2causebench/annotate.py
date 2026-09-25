# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Score one cell of a test read with the benchmark's scorer and add provenance (PRD Interface
"annotate"; non-negotiable 8; scenarios 5, 6, 10, 11, 28, 38, 44, 45).

    python -m seq2causebench.annotate --corpus <dir> --run-dir out/<discover-run> \
        --cell trace/cli/shipped/shipped/request --pretrain-results out/<pretrain>/run/results.json \
        --per-lag-files 8 --output-folder results/xs/latent/seed=0/trace/cli/shipped/shipped/request

Score side, after the method process has exited and `graphs/` has been pulled. Calls
`tracebench.score.score_corpus` — the function behind `python -m tracebench.score` — on the cell's
prediction (structural axes; written verbatim as `score.json`), on the arm-path's full ranking
(threshold-free axes; `score-ranking.json`) and on each per-lag thresholded prediction (per-lag
recall). `annotate.json` then adds, without touching any metric: the structural-limitation field
with the count of bidirected truth edges no arm here could find (scenario 6), the coverage of the
probed sample, the alphabet tokens the views never mint, the corrupted-cell counts of the cell's
path (scenario 10), whether an empty edge set is genuine or induced by a non-finite pooled
threshold (scenario 11), the confounded-pair diagnostic (directed predictions on bidirected truth
pairs, scenario 38), the oracle's regime flag (scenario 45), the benchmark tool version and config
hash, the model hash and the cell's frozen values. `causal_validity` passes through as the scorer
wrote it (scenario 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.constants import ALPHABET_JSON, GRAPHS_DIR, SCORING_TARGET_JSON
from tracebench.score import SCORING_TARGET_SESSION_JSON, score_corpus, truth_sets

from .arms import arm_slug, arm_spec, parse_cell_key
from .constants import (
    CUT_SHIPPED,
    MANIFEST_JSON,
    ORACLE_IN_REGIME,
    PREDICTION_JSON_FMT,
    PREDICTION_LAG_JSON_FMT,
    RANKING_JSON_FMT,
    RESULTS_JSON,
    RUN_DIR,
    SCORES_NPZ_FMT,
)
from .corpus import Corpus
from .log import log
from .prediction import structural_limitation
from .project import read_scores_npz
from .record import RunRecord, read_json, write_json
from .vocab import Vocab

SCORE_JSON = "score.json"
SCORE_RANKING_JSON = "score-ranking.json"
ANNOTATE_JSON = "annotate.json"
ANNOTATE_SCHEMA = "seq2causebench/annotate@1"


def unreachable_tokens(corpus_dir, ordering, grain):
    """Alphabet tokens the view's vocabulary cannot emit (D-SB-6)."""
    alphabet = read_json(Path(corpus_dir) / GRAPHS_DIR / ALPHABET_JSON)
    vocab = Vocab.from_model_vocab(Corpus(corpus_dir, ordering, grain).vocab_json())
    minted = {vocab.token_string(t) for t in vocab.real_ids}
    tokens = [t["token"] for t in alphabet["tokens"]]
    missing = [t for t in tokens if t not in minted]
    return {
        "alphabet_tokens": len(tokens),
        "unreachable": len(missing),
        "unreachable_tokens": missing,
    }


def confounded_pair_count(corpus_dir, grain, prediction, floor):
    """Directed predictions whose unordered pair is a bidirected truth pair at the floor (scenario 38):
    a count from the scorer's own truth sets, never a score."""
    gdir = Path(corpus_dir) / GRAPHS_DIR
    target = read_json(
        gdir / (SCORING_TARGET_JSON if grain == "request" else SCORING_TARGET_SESSION_JSON)
    )
    _directed, bidirected = truth_sets(target, floor)
    hit = 0
    for e in prediction["directed"]:
        if frozenset((e["src"], e["dst"])) in {frozenset(p) for p in bidirected}:
            hit += 1
    return {
        "directed_predictions_on_bidirected_truth_pairs": hit,
        "truth_bidirected_pairs": len(bidirected),
    }


def empty_prediction_kind(n_edges, counts, cell_summary):
    """Genuine (no corrupted cell, every score below the cut) versus induced (corrupted cells and a
    non-finite pooled threshold), from recorded numbers only (scenario 11)."""
    if n_edges > 0:
        return None
    corrupted = int(counts.get("n_cell_collapsed", 0)) + int(counts.get("n_cell_saturated", 0))
    finite = cell_summary.get("threshold_finite", True)
    if corrupted > 0 and not finite:
        return {"kind": "induced", "corrupted_cells": corrupted, "threshold_finite": finite}
    return {"kind": "genuine", "corrupted_cells": corrupted, "threshold_finite": finite}


def run_annotate(args, rec):
    run_dir = Path(args.run_dir)
    disc = read_json(run_dir / RUN_DIR / RESULTS_JSON)
    arm, path, cut, grain = parse_cell_key(args.cell)
    if (
        disc.get("status") != "ok"
        or disc.get("grain") != grain
        or args.cell not in disc.get("cells", {})
    ):
        raise ValueError(f"{run_dir} is not a completed discover run holding cell {args.cell}")
    if disc.get("split") != "test":
        log({"event": "annotate_note", "note": f"annotating a {disc.get('split')} read"})
    summary = disc["cells"][args.cell]
    slug = arm_slug(arm)
    corpus_dir = Path(args.corpus)
    manifest = read_json(corpus_dir / MANIFEST_JSON)
    prediction = read_json(
        run_dir / PREDICTION_JSON_FMT.format(grain=grain, arm=slug, path=path, cut=cut)
    )
    ranking_path = run_dir / RANKING_JSON_FMT.format(grain=grain, arm=slug, path=path)
    if not ranking_path.exists():
        raise ValueError(
            f"{ranking_path.name} missing: a shipped cut's ranking is its frozen sibling's; run the frozen cell in the same read"
        )
    ranking = read_json(ranking_path)
    score = score_corpus(corpus_dir, prediction, grain=grain)
    score_rank = score_corpus(corpus_dir, ranking, grain=grain)
    write_json(rec.out_dir / SCORE_JSON, score)
    write_json(rec.out_dir / SCORE_RANKING_JSON, score_rank)
    per_lag = {}
    if cut != CUT_SHIPPED:
        n_lags = int(summary.get("per_lag_files", 0))
        for lag in range(1, min(int(args.per_lag_files), n_lags) + 1):
            p = run_dir / PREDICTION_LAG_JSON_FMT.format(grain=grain, arm=slug, path=path, lag=lag)
            if not p.exists():
                break
            per_lag[str(lag)] = score_corpus(corpus_dir, read_json(p), grain=grain)["directed"][
                "recall"
            ]
    inside = len(ranking["directed"]) - score_rank["universe"]["predictions_outside_universe"]
    scores = read_scores_npz(
        run_dir / SCORES_NPZ_FMT.format(probe=disc["probe"], noise=disc["noise"], grain=grain)
    )
    pre = read_json(args.pretrain_results)
    oracle = pre.get("oracle", {})
    if pre.get("model_sha256") != disc["model_sha256"]:
        raise ValueError(
            f"pretrain record binds model {pre.get('model_sha256')} but the read used {disc['model_sha256']}"
        )
    limitation = structural_limitation()
    limitation["truth_bidirected_edges"] = int(score["universe"]["truth_bidirected"])
    counts = disc.get("corrupted_cells", {}).get(path, {})
    annotate = {
        "schema": ANNOTATE_SCHEMA,
        "cell": args.cell,
        "arm": arm,
        "path": path,
        "cut": cut,
        "grain": grain,
        "split": disc["split"],
        "rung": disc.get("rung", manifest.get("instance")),
        "variant": score["variant"],
        "seed": manifest.get("seed"),
        "corpus_id": disc["corpus_id"],
        "tool_version": manifest.get("tool_version"),
        "config_hash": manifest.get("config_hash"),
        "model_sha256": disc["model_sha256"],
        "arm_class": arm_spec(arm)["class"],
        "probe": disc["probe"],
        "noise": disc["noise"],
        "probe_amp": disc.get("probe_amp"),
        "frozen": {
            "tau": summary.get("tau"),
            "c": disc["c"],
            "N": disc["N"],
            "g": disc["g"],
            "freeze": disc.get("freeze"),
        },
        "shipped_cut": (
            {
                "rule": disc.get("shipped_rule"),
                "tau_by_lag": summary.get("tau_by_lag"),
                "threshold_finite": summary.get("threshold_finite"),
                "n_sequences_pooled": summary.get("n_sequences_pooled"),
                "n_dropped_special": summary.get("n_dropped_special"),
                "n_within_op": summary.get("n_within_op"),
            }
            if cut == CUT_SHIPPED
            else None
        ),
        "floor": score["floor"],
        "structural_limitation": limitation,
        "coverage": {
            "n_sequences_probed": disc["n_probed"],
            "n_skipped_short": disc["n_skipped_short"],
            "pairs_scored": int(len(ranking["directed"])),
            "pairs_in_universe": int(inside),
            "universe_ordered_pairs": score_rank["universe"]["ordered_pairs"],
            "universe_fraction_cooccurring": (inside / score_rank["universe"]["ordered_pairs"])
            if score_rank["universe"]["ordered_pairs"]
            else None,
            "truth_directed": score_rank["universe"]["truth_directed"],
            "reachable_recall_ceiling": score_rank["directed"]["recall"],
            "n_pairs_in_table": int(len(scores["src"])),
        },
        "unreachable_tokens": unreachable_tokens(corpus_dir, str(scores["ordering"]), grain),
        "corrupted_cells": counts,
        "empty_prediction": empty_prediction_kind(len(prediction["directed"]), counts, summary),
        "confounded_pairs": confounded_pair_count(corpus_dir, grain, prediction, score["floor"]),
        "oracle": {
            "eps_hat": oracle.get("eps_hat"),
            "in_regime": (
                oracle.get("eps_hat") is not None and oracle["eps_hat"] < ORACLE_IN_REGIME
            ),
            "entropy_order": oracle.get("entropy_order"),
            "val_loss": oracle.get("val_loss"),
        },
        "budget": pre.get("budget"),
        "memory": disc.get("memory"),
        "score": score,
        "score_ranking": {
            "auroc": score_rank["auroc"],
            "average_precision": score_rank["average_precision"],
        },
        "per_lag_recall": per_lag,
        "causal_validity": score["causal_validity"],
        "files": {
            "score": SCORE_JSON,
            "score_ranking": SCORE_RANKING_JSON,
            "discover_run": run_dir.name,
        },
    }
    write_json(rec.out_dir / ANNOTATE_JSON, annotate)
    log(
        {
            "event": "annotate_done",
            "cell": args.cell,
            "f1_directed": score["directed"]["f1"],
            "reachable_recall_ceiling": annotate["coverage"]["reachable_recall_ceiling"],
        }
    )
    return {
        "annotate": ANNOTATE_JSON,
        "score": SCORE_JSON,
        "cell": args.cell,
        "corpus_id": disc["corpus_id"],
        "model_sha256": disc["model_sha256"],
        "f1_directed": score["directed"]["f1"],
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument("--run-dir", required=True, help="the discover run's output folder")
    p.add_argument("--cell", required=True, help="<arm>/<path>/<cut>/<grain>, a cell of that read")
    p.add_argument(
        "--pretrain-results",
        required=True,
        help="the pretrain run's results.json (oracle, budget, model hash)",
    )
    p.add_argument(
        "--per-lag-files", required=True, type=int, help="score per-lag predictions for lags 1..K"
    )
    p.add_argument("--output-folder", required=True, help="the results cell directory")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "annotate", vars(args)) as rec:
        res = run_annotate(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
