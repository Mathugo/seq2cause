"""The per-sequence axis (PRD Goal 6, Interface "seqscore", scenarios 35–37;
`plans/per-sequence-rules.md` §3–5): score one cell's stored matrices within each sequence
against the grain's induced truth, and summarise over the sequences of the read.

    python -m seq2causebench.seqscore --corpus <dir> --run-dir out/<discover-run> \
        --cell trace/core/shipped/frozen/request --max-lag 63 --output-folder results/.../seqscore

Score side, after the method has exited. Every formula is the benchmark's own (`prf`, `auroc`,
`average_precision` from `tracebench.score`, the universe from `build_universe`); this instrument
adds only the truth induction (`seqtruth`) and the aggregation over sequences. The frozen cut's
edge set is `S[j, q] > τ`; the shipped cut's is `apply_tau_by_lag(S, tau_by_lag)` with the read's
recorded thresholds. A sequence is scoreable when a truth pair lies among its candidates; the
predict-all audit value `2p / (n + p)` rides every headline. Per-sequence numbers never share a
table with type-level numbers: `seqscore.json` is their own record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from seq2cause.threshold import AdaptiveThreshold
from tracebench.score import auroc, average_precision, prf

from .arms import arm_slug, parse_cell_key
from .constants import (
    CUT_SHIPPED,
    MATRICES_NPZ_FMT,
    RESULTS_JSON,
    RUN_DIR,
    SEQSCORE_JSON,
    SEQUENCES_JSON_FMT,
)
from .corpus import Corpus
from .data import SequenceStore
from .engine import from_triangle
from .log import log
from .record import RunRecord, read_json, write_json
from .seqtruth import RULES_VERSION, truth_for_read
from .vocab import Vocab

SEQSCORE_SCHEMA = "seq2causebench/seqscore@1"


def _summary(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(a) == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "p10": float(np.quantile(a, 0.1)),
        "p50": float(np.quantile(a, 0.5)),
        "p90": float(np.quantile(a, 0.9)),
    }


def _predict_all(p, n):
    return (2 * p / (n + p)) if (n + p) else None


def edge_set(matrix, candidates, cut, tau, tau_by_lag, rule):
    """The cell's per-sequence edge set over the candidates."""
    if cut == CUT_SHIPPED:
        thr = AdaptiveThreshold(**rule)
        import torch

        graph = thr.apply_tau_by_lag(
            torch.as_tensor(matrix, dtype=torch.float32),
            {int(k): float(v) for k, v in tau_by_lag.items()},
        )
        g = graph.numpy()
        return {(j, q) for j, q in candidates if g[j, q]}
    return {(j, q) for j, q in candidates if matrix[j, q] > tau}


def score_sequence(matrix, truth, edges):
    cands = truth["candidates"]
    if not cands:
        return None
    directed_truth, adjacency_truth = truth["directed"], truth["adjacency"]
    scoreable = bool(directed_truth or adjacency_truth)
    tp = len(edges & directed_truth)
    fp = len(edges - directed_truth)
    fn = len(directed_truth - edges)
    stp = len(edges & adjacency_truth)
    sfp = len(edges - adjacency_truth)
    sfn = len(adjacency_truth - edges)
    atp = len(edges & truth["ancestor_directed"])
    afp = len(edges - truth["ancestor_directed"])
    afn = len(truth["ancestor_directed"] - edges)
    scores = np.array([matrix[j, q] for j, q in cands], dtype=np.float64)
    y_dir = np.array([(j, q) in directed_truth for j, q in cands], dtype=np.int64)
    y_adj = np.array([(j, q) in adjacency_truth for j, q in cands], dtype=np.int64)
    out = {
        "scoreable": scoreable,
        "n_candidates": len(cands),
        "n_truth_directed": len(directed_truth),
        "n_truth_adjacency": len(adjacency_truth),
        "n_edges": len(edges),
        "directed": prf(tp, fp, fn),
        "skeleton": prf(stp, sfp, sfn),
        "ancestor_directed": prf(atp, afp, afn),
        "counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "stp": stp,
            "sfp": sfp,
            "sfn": sfn,
            "atp": atp,
            "afp": afp,
            "afn": afn,
        },
        "auroc": auroc(y_dir, scores) if scoreable and 0 < y_dir.sum() < len(y_dir) else None,
        "ap": average_precision(y_dir, scores) if scoreable and y_dir.sum() > 0 else None,
        "auroc_skeleton": auroc(y_adj, scores)
        if scoreable and 0 < y_adj.sum() < len(y_adj)
        else None,
        "predict_all": _predict_all(len(directed_truth), len(cands)),
        "pairs": [
            (
                j,
                q,
                float(matrix[j, q]),
                int((j, q) in directed_truth),
                int((j, q) in adjacency_truth),
            )
            for j, q in cands
        ],
        "lags": {q - j: ((j, q) in edges) for j, q in directed_truth},
    }
    return out


def run_seqscore(args, rec):
    run_dir = Path(args.run_dir)
    disc = read_json(run_dir / RUN_DIR / RESULTS_JSON)
    arm, path, cut, grain = parse_cell_key(args.cell)
    if (
        disc.get("status") != "ok"
        or disc.get("grain") != grain
        or args.cell not in disc.get("cells", {})
    ):
        raise ValueError(f"{run_dir} is not a completed discover run holding cell {args.cell}")
    summary = disc["cells"][args.cell]
    col = f"{arm_slug(arm)}__{path}"
    c = int(disc["c"])
    seqrec = read_json(run_dir / SEQUENCES_JSON_FMT.format(grain=grain))
    with np.load(
        run_dir / MATRICES_NPZ_FMT.format(probe=disc["probe"], noise=disc["noise"], grain=grain)
    ) as z:
        table = {k: z[k] for k in z.files}
    if col not in table:
        raise ValueError(f"the stored matrices carry no column {col!r}")
    # the stored token sequences, re-read exactly as the method read them (ops + outcomes only)
    corpus = Corpus(args.corpus, str(disc.get("ordering", "end")), grain)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    args_rec = read_json(run_dir / RUN_DIR / "arguments.json")["arguments"]
    store = SequenceStore.from_corpus(
        corpus,
        disc["split"],
        vocab,
        int(args_rec["max_len"]),
        n=int(args_rec["num_sequences"]),
        mode=str(args_rec["sequence_sample"]),
        seed=int(args_rec["seed"]),
    )
    stored_seq = [int(s["seq"]) for s in seqrec["sequences"]]
    stored_ids = [store.get(s) for s in stored_seq]
    for entry, ids in zip(seqrec["sequences"], stored_ids, strict=True):
        if len(ids) != entry["length"] or store.trace_ids[entry["seq"]] != entry["trace_id"]:
            raise ValueError(
                "the re-read sequences do not match the read's record; same corpus, split and selection required"
            )
    truth, targets, _ = truth_for_read(
        args.corpus,
        str(disc.get("ordering", "end")),
        grain,
        disc["split"],
        seqrec,
        stored_ids,
        c,
        int(args.max_lag),
    )
    per_seq, pooled = (
        [],
        {"tp": 0, "fp": 0, "fn": 0, "stp": 0, "sfp": 0, "sfn": 0, "atp": 0, "afp": 0, "afn": 0},
    )
    y_all, s_all, y_adj_all = [], [], []
    n_short = n_unscoreable = n_scoreable = 0
    n_truth = n_cand = 0
    lag_hits, lag_tot = {}, {}
    n_links_lost = 0
    for s, ids in zip(stored_seq, stored_ids, strict=True):
        m_rows = table["seq"] == s
        lc = len(ids) - c
        matrix = from_triangle(lc, table["j"][m_rows], table["q"][m_rows], table[col][m_rows])
        t = truth[s]
        n_links_lost += t["n_links_lost"]
        if not t["candidates"]:
            n_short += 1
            continue
        edges = edge_set(
            matrix,
            t["candidates"],
            cut,
            summary.get("tau"),
            summary.get("tau_by_lag"),
            disc.get("shipped_rule"),
        )
        r = score_sequence(matrix, t, edges)
        for k in pooled:
            pooled[k] += r["counts"][k]
        n_truth += r["n_truth_directed"]
        n_cand += r["n_candidates"]
        for _j, _q, sc, yd, ya in r["pairs"]:
            y_all.append(yd)
            s_all.append(sc)
            y_adj_all.append(ya)
        for lag, hit in r["lags"].items():
            lag_tot[lag] = lag_tot.get(lag, 0) + 1
            lag_hits[lag] = lag_hits.get(lag, 0) + int(hit)
        if r["scoreable"]:
            n_scoreable += 1
            per_seq.append(r)
        else:
            n_unscoreable += 1
    n_seq = len(stored_seq)
    y_all = np.asarray(y_all, dtype=np.int64)
    s_all = np.asarray(s_all, dtype=np.float64)
    y_adj_all = np.asarray(y_adj_all, dtype=np.int64)
    result = {
        "schema": SEQSCORE_SCHEMA,
        "axis": "per-sequence",
        "rules_version": RULES_VERSION,
        "cell": args.cell,
        "arm": arm,
        "path": path,
        "cut": cut,
        "grain": grain,
        "split": disc["split"],
        "corpus_id": disc["corpus_id"],
        "rung": disc.get("rung"),
        "model_sha256": disc["model_sha256"],
        "floor": targets["floor"],
        "c": c,
        "max_lag": int(args.max_lag),
        "tau": summary.get("tau"),
        "tau_by_lag": summary.get("tau_by_lag"),
        "n_sequences": n_seq,
        "n_sequences_split": disc.get("n_sequences"),
        "n_skipped_short_by_read": disc.get("n_skipped_short"),
        "n_stored_by_read": disc.get("n_stored"),
        "n_skipped_short": n_short,
        "n_unscoreable": n_unscoreable,
        "n_scoreable": n_scoreable,
        "scoreable_fraction": (n_scoreable / n_seq) if n_seq else None,
        "n_links_lost": n_links_lost,
        "per_sequence": {
            "directed": {
                k: _summary([r["directed"][k] for r in per_seq])
                for k in ("precision", "recall", "f1")
            },
            "skeleton": {
                k: _summary([r["skeleton"][k] for r in per_seq])
                for k in ("precision", "recall", "f1")
            },
            "ancestor_directed": {
                k: _summary([r["ancestor_directed"][k] for r in per_seq])
                for k in ("precision", "recall", "f1")
            },
            "auroc": _summary([r["auroc"] for r in per_seq]),
            "ap": _summary([r["ap"] for r in per_seq]),
            "auroc_skeleton": _summary([r["auroc_skeleton"] for r in per_seq]),
            "predict_all_mean": _summary([r["predict_all"] for r in per_seq])["mean"],
        },
        "pooled": {
            "directed": prf(pooled["tp"], pooled["fp"], pooled["fn"]),
            "skeleton": prf(pooled["stp"], pooled["sfp"], pooled["sfn"]),
            "ancestor_directed": prf(pooled["atp"], pooled["afp"], pooled["afn"]),
            "auroc": auroc(y_all, s_all) if len(y_all) and 0 < y_all.sum() < len(y_all) else None,
            "ap": average_precision(y_all, s_all) if len(y_all) and y_all.sum() > 0 else None,
            "auroc_skeleton": auroc(y_adj_all, s_all)
            if len(y_adj_all) and 0 < y_adj_all.sum() < len(y_adj_all)
            else None,
            "n_candidates": int(n_cand),
            "n_truth_directed": int(n_truth),
        },
        "predict_all": {"pooled": _predict_all(n_truth, n_cand)},
        "per_lag_recall": {str(lag): (lag_hits[lag] / lag_tot[lag]) for lag in sorted(lag_tot)},
    }
    write_json(rec.out_dir / SEQSCORE_JSON, result)
    log(
        {
            "event": "seqscore_done",
            "cell": args.cell,
            "n_scoreable": n_scoreable,
            "scoreable_fraction": result["scoreable_fraction"],
            "pooled_f1": result["pooled"]["directed"]["f1"],
            "predict_all": result["predict_all"]["pooled"],
        }
    )
    return {
        "seqscore": SEQSCORE_JSON,
        "cell": args.cell,
        "n_scoreable": n_scoreable,
        "scoreable_fraction": result["scoreable_fraction"],
        "pooled_directed_f1": result["pooled"]["directed"]["f1"],
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument(
        "--run-dir", required=True, help="the discover run whose stored matrices are scored"
    )
    p.add_argument("--cell", required=True, help="<arm>/<path>/<cut>/<grain>")
    p.add_argument("--max-lag", required=True, type=int, help="candidate pairs within this lag")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "seqscore", vars(args)) as rec:
        res = run_seqscore(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
