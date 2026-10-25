"""Score one RCD rank read (one corpus) on the ranking axis with the run-averaged aggregation (score side).

    python -m tracebenchrca.scorercd --corpus-dir <pulled corpus> --rank-record <rca-rcd>/rank \
        --bootstrap-b 2000 --bootstrap-seed 0 --output-folder <arm folder>

The same aggregation as CausalRCA's score side (`scorecausalrca.score_ranking`; `plans/rca-arms.md`,
addenda 2026-10-23 and 2026-10-25, later): per case AC@k is the share of the runs whose service ranking
places the root cause in the first k (a failed run is a run without a hit; a service absent from a run's
ranking is a miss for that run), the rung's numbers are means over cases, the interval resamples the
cases with the benchmark's draws. Writes `ranking.json` (per case id, every run's ranking) and
`score-rca.json`. No graph axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.score_rca import load_cases

from .constants import ARM_RCD, RANKING_JSON, SCORE_RCA_JSON
from .export import corpus_id_of
from .log import log
from .record import RunRecord, sha256_file, write_json
from .score import read_rank_record
from .scorecausalrca import AGGREGATION, restore_case_ids, score_ranking

STAGE = "score-rca"


def score(corpus_dir, rank_record, b, seed, out):
    rank_res, rank_sha, positional = read_rank_record(rank_record)
    if rank_res.get("arm") != ARM_RCD:
        raise ValueError(f"{rank_record} is an {rank_res.get('arm')!r} read, not {ARM_RCD!r}")
    corpus_id = corpus_id_of(corpus_dir)
    if corpus_id != rank_res["corpus_id"]:
        raise ValueError(f"the rank read is of {rank_res['corpus_id']!r}; this corpus is {corpus_id!r}")
    cases = load_cases(corpus_dir)
    by_id = restore_case_ids(cases, positional, rank_res["cases"])
    out = Path(out)
    write_json(out / RANKING_JSON, {"cases": {cid: c["rankings"] for cid, c in by_id.items()}})
    result = score_ranking(corpus_dir, {cid: c["rankings"] for cid, c in by_id.items()}, b, seed)
    write_json(out / SCORE_RCA_JSON, result)
    return {
        "stage": STAGE,
        "arm": ARM_RCD,
        "rung": rank_res["rung"],
        "corpus_id": corpus_id,
        "rank_record_sha256": rank_sha,
        "ranking_sha256": sha256_file(out / RANKING_JSON),
        "score_rca_sha256": sha256_file(out / SCORE_RCA_JSON),
        "aggregation": AGGREGATION,
        "metrics": result["metrics"],
        "bootstrap": result["bootstrap"],
        "random": result["random"],
        "n_missing": result["n_missing"],
        "n_failed_runs": rank_res["n_failed_runs"],
        "n_unranked_cases": rank_res["n_unranked_cases"],
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True, help="the pulled corpus (tier rca) the export was made from")
    p.add_argument("--rank-record", required=True, help="the rankrcd stage's output folder")
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples (registered: 2000)")
    p.add_argument("--bootstrap-seed", required=True, type=int, help="bootstrap seed (registered: 0)")
    p.add_argument("--output-folder", required=True, help="the arm's result folder: ranking.json, score-rca.json, run/")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = score(args.corpus_dir, args.rank_record, args.bootstrap_b, args.bootstrap_seed, args.output_folder)
        log({"event": "score_rca", "arm": res["arm"], "corpus_id": res["corpus_id"], **res["metrics"]})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
