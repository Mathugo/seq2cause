"""Score one arm's rankings of one corpus on the ranking axis (score side).

    python -m tracebenchrca.score --corpus-dir <pulled corpus> --rank-record <rank folder> \
        --bootstrap-b 2000 --bootstrap-seed 0 --output-folder <arm folder>

Restores the case ids of a method-side ranking (`ranking-positional.json`, keyed by case
directory) by the one positional rule both sides use (`tracebench.export_rca.case_dirs` over
`labels/cases.json`), writes `ranking.json` in `tracebench.score_rca`'s prediction shape and
scores it with the benchmark's scorer, output written verbatim to `score-rca.json` (AC@1 / 3 / 5,
Avg@5, the percentile bootstrap over cases, the analytic random expectation — `control/random`
— and the rows per fault kind). A case the arm left unranked is absent from `ranking.json` and
counts as a miss at every k. For `rca/baro-bocpd` the record adds each case's detected change
point's offset from the labelled injection (plans/rca-arms.md §5).

This module reads labels; it runs after the method process has exited and never hands anything
back to a method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tracebench.export_rca import case_dirs, window_epoch
from tracebench.score_rca import load_cases, score_rca

from .constants import CONFIG_YAML, RANKING_JSON, RESULTS_JSON, RUN_DIR, SCORE_RCA_JSON
from .export import corpus_id_of
from .log import log
from .rank import STAGE as RANK_STAGE
from .record import RunRecord, read_json, sha256_file, write_json

STAGE = "score-rca"


def read_rank_record(rank_record):
    rank_record = Path(rank_record)
    path = rank_record / RUN_DIR / RESULTS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{rank_record} holds no rank record ({RUN_DIR}/{RESULTS_JSON})")
    res = read_json(path)
    if res.get("status") != "ok" or res.get("stage") != RANK_STAGE:
        raise ValueError(f"{path} is not a completed rank record")
    positional_path = rank_record / res["ranking_file"]
    if sha256_file(positional_path) != res["ranking_sha256"]:
        raise ValueError(f"{positional_path} differs from the ranking its record hashed")
    return res, sha256_file(path), read_json(positional_path)["cases"]


def restore_case_ids(cases, positional, rank_cases):
    """`{case_id: ranking}` for the ranked cases; refuses an export whose case directories are not
    exactly the labelled cases' positional names."""
    pairs = case_dirs(cases)
    names = {name for name, _ in pairs}
    ranked_dirs = {c["case_dir"] for c in rank_cases}
    if ranked_dirs != names:
        raise ValueError(
            f"the rank read's case directories {sorted(ranked_dirs)} are not the labelled cases' {sorted(names)}"
        )
    extra = set(positional) - names
    if extra:
        raise ValueError(f"rankings for unknown case directories {sorted(extra)}")
    return {c["case_id"]: list(positional[name]) for name, c in pairs if name in positional}, pairs


def corpus_epoch(corpus_dir):
    """UNIX seconds of the corpus window start — the exporter's own clock (`tracebench.export_rca.window_epoch`)."""
    with open(Path(corpus_dir) / CONFIG_YAML, encoding="utf-8") as f:
        return window_epoch(yaml.safe_load(f))[0]


def change_point_offsets(pairs, rank_cases, epoch_unix):
    """Per case, the first detected change point's offset from the labelled injection (seconds; the
    change point's bin start in the export's UNIX clock against the label's corpus-relative time), or None."""
    by_dir = {c["case_dir"]: c for c in rank_cases}
    out = []
    for name, c in pairs:
        row = by_dir[name]
        t = row.get("change_point_time")
        out.append(
            {
                "case_id": c["case_id"],
                "miss": row.get("miss"),
                "change_point_row": row.get("change_point_row"),
                "change_point_offset_s": None if t is None else int(t) - (epoch_unix + int(float(c["inject_time_s"]))),
            }
        )
    return out


def root_tie_groups(pairs, rank_cases):
    """Per ranked case whose rank record carries PageRank tie groups (`rca/pc-pagerank`): the size of the tie group
    the root cause's best-ranked metric sits in (metrics sharing its score, and the services first ranked inside
    that group), or None when the root cause is unranked. A diagnostic for the findings, never a re-ranking."""
    by_dir = {c["case_dir"]: c for c in rank_cases}
    out = []
    for name, c in pairs:
        row = by_dir[name]
        if row.get("tie_groups") is None:
            continue
        hit = next((g for g in row["tie_groups"] if c["root_cause_component"] in g["services_first_ranked"]), None)
        out.append(
            {
                "case_id": c["case_id"],
                "root_tie_group_metrics": None if hit is None else hit["n_metrics"],
                "root_tie_group_services": None if hit is None else len(hit["services_first_ranked"]),
            }
        )
    return out


def score(corpus_dir, rank_record, b, seed, out):
    rank_res, rank_sha, positional = read_rank_record(rank_record)
    corpus_id = corpus_id_of(corpus_dir)
    if corpus_id != rank_res["corpus_id"]:
        raise ValueError(f"the rank read is of {rank_res['corpus_id']!r}; this corpus is {corpus_id!r}")
    cases = load_cases(corpus_dir)
    ranking, pairs = restore_case_ids(cases, positional, rank_res["cases"])
    out = Path(out)
    write_json(out / RANKING_JSON, {"cases": ranking})
    result = score_rca(corpus_dir, {"cases": ranking}, b, seed)
    write_json(out / SCORE_RCA_JSON, result)
    res = {
        "stage": STAGE,
        "arm": rank_res["arm"],
        "rung": rank_res["rung"],
        "corpus_id": corpus_id,
        "rank_record_sha256": rank_sha,
        "ranking_sha256": sha256_file(out / RANKING_JSON),
        "score_rca_sha256": sha256_file(out / SCORE_RCA_JSON),
        "metrics": result["metrics"],
        "bootstrap": result["bootstrap"],
        "random": result["random"],
        "n_missing": result["n_missing"],
        "n_unranked_cases": rank_res["n_unranked_cases"],
    }
    if any("change_point_time" in c or c.get("miss") for c in rank_res["cases"]):
        res["change_points"] = change_point_offsets(pairs, rank_res["cases"], corpus_epoch(corpus_dir))
    ties = root_tie_groups(pairs, rank_res["cases"])
    if ties:
        res["root_tie_groups"] = ties
        res["n_cases_root_in_tie_group"] = sum(1 for t in ties if (t["root_tie_group_metrics"] or 0) > 1)
    return res


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True, help="the pulled corpus (tier rca) the export was made from")
    p.add_argument("--rank-record", required=True, help="the rank stage's output folder")
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
