"""Score one CausalRCA rank read (one corpus, one series) on the ranking axis and write its learned
service graphs for the graph axis (score side).

    python -m tracebenchrca.scorecausalrca --corpus-dir <pulled corpus> --rank-record <series>/rank \
        --bootstrap-b 2000 --bootstrap-seed 0 --output-folder <arm folder>/<series>

The authors' aggregation (`plans/rca-arms.md`, addenda 2026-10-22, later, and 2026-10-23): each run's
ranking is placed against the labelled root cause by the benchmark's own `case_ranks`; per case, AC@k
is the share of the case's runs that place it in the first k (a failed run is a run without a hit);
the reported AC@1 / 3 / 5 and Avg@5 are the means over cases, and the 95 % interval resamples the cases
with the benchmark's draws (`run_averaged_bootstrap`, equal to `tracebench.score_rca.bootstrap` when the
runs of every case agree). Writes `ranking.json` (per case id, every run's ranking), `score-rca.json`
and `graph.json` (per case id, the mean |adj| over the successful runs as `tracebench.score_service`
predictions over `svc:<index>` keys, every pair with a positive mean listed with it as the score).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tracebench.export_rca import case_dirs
from tracebench.score_rca import AVG_K, KS, LEVEL, case_ranks, load_cases, random_expectation
from tracebench.score_service import deployed_services, service_key

from .constants import ARM_CAUSALRCA, RANKING_JSON, SCORE_RCA_JSON
from .export import corpus_id_of
from .log import log
from .record import RunRecord, sha256_file, write_json
from .score import read_rank_record

STAGE = "score-rca"
GRAPH_JSON = "graph.json"
AGGREGATION = "per-run metrics averaged over the case's runs (the authors' Res-Plot.ipynb), bootstrap over cases"


def restore_case_ids(cases, positional, rank_cases):
    """`{case_id: positional entry}` by the one positional rule both sides use (`case_dirs`); refuses an
    export whose case directories are not exactly the labelled cases' positional names."""
    pairs = case_dirs(cases)
    names = {name for name, _ in pairs}
    if {c["case_dir"] for c in rank_cases} != names or not set(positional) <= names:
        raise ValueError(f"the rank read's case directories are not the labelled cases' {sorted(names)}")
    return {c["case_id"]: positional[name] for name, c in pairs if name in positional}


def run_hits(cases, run_rankings):
    """Per labelled case: the 1-based rank of the root cause in each run's ranking (None = missed or a failed run)."""
    n_runs = max([0] + [len(v) for v in run_rankings.values()])
    ranks = {c["case_id"]: [] for c in cases}
    for r in range(n_runs):
        pred = {cid: runs[r] for cid, runs in run_rankings.items() if runs[r] is not None}
        for row in case_ranks(cases, {"cases": pred}):
            ranks[row["case_id"]].append(row["rank"])
    return ranks


def hit_matrix(per_case_ranks, k_max):
    """cases × k: the share of the case's runs with the root cause in the first k (k = 1 … k_max)."""
    out = np.zeros((len(per_case_ranks), k_max))
    for i, ranks in enumerate(per_case_ranks):
        for k in range(1, k_max + 1):
            out[i, k - 1] = sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0
    return out


def run_averaged_metrics(hits):
    if len(hits) == 0:
        return {**{f"ac@{k}": None for k in KS}, f"avg@{AVG_K}": None, "n_cases": 0}
    mean = hits.mean(axis=0)
    return {
        **{f"ac@{k}": float(mean[k - 1]) for k in KS},
        f"avg@{AVG_K}": float(mean[:AVG_K].mean()),
        "n_cases": len(hits),
    }


def run_averaged_bootstrap(hits, b, seed):
    """`tracebench.score_rca.bootstrap`'s resampling (the same generator, draws and percentiles) applied to
    run-averaged per-case hit rates; identical to it when every run of every case agrees."""
    n = len(hits)
    if n == 0 or b <= 0:
        return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": {}}
    samp = np.random.default_rng(seed).integers(0, n, size=(int(b), n))
    by_k = {k: hits[:, k - 1][samp].mean(axis=1) for k in range(1, AVG_K + 1)}
    lo, hi = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2

    def ci(v):
        return {"low": float(np.quantile(v, lo)), "high": float(np.quantile(v, hi))}

    intervals = {f"ac@{k}": ci(by_k[k]) for k in KS}
    intervals[f"avg@{AVG_K}"] = ci(np.mean([by_k[k] for k in range(1, AVG_K + 1)], axis=0))
    return {"b": int(b), "seed": int(seed), "level": LEVEL, "intervals": intervals}


def by_fault_kind(case_rows, hits):
    kinds = sorted({r["fault_kind"] for r in case_rows if r["fault_kind"] is not None})
    return {
        kind: run_averaged_metrics(hits[[i for i, r in enumerate(case_rows) if r["fault_kind"] == kind]])
        for kind in kinds
    }


def score_ranking(corpus_dir, run_rankings, b, seed):
    cases = load_cases(corpus_dir)
    per_case = run_hits(cases, run_rankings)
    rows = [
        {
            "case_id": c["case_id"],
            "root_cause_component": c["root_cause_component"],
            "fault_kind": c.get("fault_kind"),
            "ranks_per_run": per_case[c["case_id"]],
            "n_runs": len(per_case[c["case_id"]]),
            "n_failed_runs": sum(1 for r in (run_rankings.get(c["case_id"]) or []) if r is None),
            "missing": c["case_id"] not in run_rankings,
        }
        for c in cases
    ]
    hits = hit_matrix([r["ranks_per_run"] for r in rows], AVG_K)
    for r, h in zip(rows, hits, strict=True):
        r["hit_rate"] = {f"ac@{k}": float(h[k - 1]) for k in range(1, AVG_K + 1)}
    return {
        "axis": "ranking",
        "aggregation": AGGREGATION,
        "corpus_dir_name": Path(corpus_dir).name,
        "metrics": run_averaged_metrics(hits),
        "bootstrap": run_averaged_bootstrap(hits, b, seed),
        "random": random_expectation(len(deployed_services(corpus_dir))),
        "n_missing": sum(1 for r in rows if r["missing"]),
        "by_fault_kind": by_fault_kind(rows, hits),
        "cases": rows,
    }


def pool_run_averaged(results, b, seed):
    """One table over the cases of several per-corpus results: their run-averaged per-case hit rates
    concatenated, the same metrics and bootstrap (the counterpart of `tracebench.score_rca.pool`)."""
    rows = [r for res in results for r in res["cases"]]
    hits = np.array([[r["hit_rate"][f"ac@{k}"] for k in range(1, AVG_K + 1)] for r in rows]).reshape(len(rows), AVG_K)
    cands = {(res.get("random") or {}).get("n_candidates") for res in results}
    return {
        "axis": "ranking",
        "aggregation": AGGREGATION,
        "pooled_from": [res.get("corpus_dir_name") for res in results],
        "n_cases_per_result": [len(res["cases"]) for res in results],
        "metrics": run_averaged_metrics(hits),
        "bootstrap": run_averaged_bootstrap(hits, b, seed),
        "random": random_expectation(cands.pop()) if len(cands) == 1 else None,
        "random_per_result": [res.get("random") for res in results],
        "n_missing": sum(1 for r in rows if r["missing"]),
        "n_failed_runs": sum(r["n_failed_runs"] for r in rows),
        "by_fault_kind": by_fault_kind(rows, hits),
    }


def graph_predictions(corpus_dir, positional_by_id):
    """Per case id, the mean |adj| as a service-level prediction: `adj[i, j]` reads as names[i] → names[j]."""
    key_of = {name: service_key(idx) for idx, name, _ in deployed_services(corpus_dir)}
    out = {}
    for cid, case in positional_by_id.items():
        m = case["mean_abs_adj"]
        if m is None:
            out[cid] = None
            continue
        names = case["services"]
        missing = [n for n in names if n not in key_of]
        if missing:
            raise ValueError(f"services {missing} are not deployed services of {corpus_dir}")
        edges = [
            {"src": key_of[names[i]], "dst": key_of[names[j]], "score": float(m[i][j])}
            for i in range(len(names))
            for j in range(len(names))
            if i != j and m[i][j] > 0
        ]
        out[cid] = {"directed": edges, "bidirected": []}
    return out


def score(corpus_dir, rank_record, b, seed, out):
    rank_res, rank_sha, positional = read_rank_record(rank_record)
    if rank_res.get("arm") != ARM_CAUSALRCA:
        raise ValueError(f"{rank_record} is an {rank_res.get('arm')!r} read, not {ARM_CAUSALRCA!r}")
    corpus_id = corpus_id_of(corpus_dir)
    if corpus_id != rank_res["corpus_id"]:
        raise ValueError(f"the rank read is of {rank_res['corpus_id']!r}; this corpus is {corpus_id!r}")
    cases = load_cases(corpus_dir)
    by_id = restore_case_ids(cases, positional, rank_res["cases"])
    out = Path(out)
    write_json(
        out / RANKING_JSON, {"series": rank_res["series"], "cases": {cid: c["rankings"] for cid, c in by_id.items()}}
    )
    result = score_ranking(corpus_dir, {cid: c["rankings"] for cid, c in by_id.items()}, b, seed)
    write_json(out / SCORE_RCA_JSON, result)
    write_json(out / GRAPH_JSON, {"series": rank_res["series"], "cases": graph_predictions(corpus_dir, by_id)})
    return {
        "stage": STAGE,
        "arm": ARM_CAUSALRCA,
        "series": rank_res["series"],
        "rung": rank_res["rung"],
        "corpus_id": corpus_id,
        "rank_record_sha256": rank_sha,
        "ranking_sha256": sha256_file(out / RANKING_JSON),
        "score_rca_sha256": sha256_file(out / SCORE_RCA_JSON),
        "graph_sha256": sha256_file(out / GRAPH_JSON),
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
    p.add_argument("--rank-record", required=True, help="the rankcausalrca stage's output folder")
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples (registered: 2000)")
    p.add_argument("--bootstrap-seed", required=True, type=int, help="bootstrap seed (registered: 0)")
    p.add_argument(
        "--output-folder", required=True, help="<arm folder>/<series>: ranking.json, score-rca.json, graph.json"
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = score(args.corpus_dir, args.rank_record, args.bootstrap_b, args.bootstrap_seed, args.output_folder)
        log(
            {
                "event": "score_rca",
                "arm": ARM_CAUSALRCA,
                "series": res["series"],
                "corpus_id": res["corpus_id"],
                **res["metrics"],
            }
        )
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
