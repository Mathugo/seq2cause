"""CausalRCA's graph axis on one corpus and series (score side): the validation read on seed 0, the
test read on seeds 1–4 (`plans/rca-arms.md`, addenda 2026-10-22 (Q16) and 2026-10-23).

    python -m tracebenchrca.graphaxis --corpus-dir <pulled corpus> --score-record <arm folder>/<series> \
        --split val --freeze none --taus 0 0.03 ... 5 --output-folder <arm folder>/<series>/graph-val
    python -m tracebenchrca.graphaxis ... --split test --freeze freezes/<date>-<rung>-rca-causalrca-<series>-graph.json \
        --taus <the frozen tau> --output-folder <arm folder>/<series>/graph-test

Each case's learned graph (`graph.json`, the mean |adj| over its successful runs) is scored by the
benchmark's `tracebench.score_service` at the service view's default floor: per τ the prediction keeps
the pairs whose mean exceeds τ (directed precision, recall, F1); AUROC is the full listing's. The
deployment topology (`floor/topology`, `tracebench.score_service.topology_reference`) is scored on the
same corpus beside it. The corpus's number is the mean over its cases. Validation reads take the whole
registered grid on seed 0 only; test reads take the frozen τ only, on seeds 1–4 only, from a freeze of
the same rung and series.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tracebench.score_rca import load_cases
from tracebench.score_service import score_service, topology_reference

from .constants import ARM_CAUSALRCA, CAUSALRCA_GRAPH_TAUS, RESULTS_JSON, RUN_DIR
from .export import corpus_id_of
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json
from .scorecausalrca import GRAPH_JSON

STAGE = "graph-axis"
SCORE_SERVICE_JSON = "score-service.json"
VAL_SEED = 0
TEST_SEEDS = (1, 2, 3, 4)
METRICS = ("precision", "recall", "f1")


def seed_of(corpus_id):
    return int(corpus_id.rsplit("seed=", 1)[1])


def directed(res):
    return {m: res["directed"][m] for m in METRICS}


def at_tau(prediction, tau):
    return {"directed": [e for e in prediction["directed"] if e["score"] > tau], "bidirected": []}


def score_case(corpus_dir, prediction, taus):
    if prediction is None:  # every run of the case failed: no learned graph, an empty prediction
        prediction = {"directed": [], "bidirected": []}
    full = score_service(corpus_dir, prediction)
    return {
        "n_pairs_listed": len(prediction["directed"]),
        "auroc": full["auroc"]["directed"],
        "by_tau": {repr(float(t)): directed(score_service(corpus_dir, at_tau(prediction, t))) for t in taus},
    }


def mean_over(rows, key):
    vals = [r[key] for r in rows]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def read(corpus_dir, score_record, split, freeze, taus):
    score_record = Path(score_record)
    sres = read_json(score_record / RUN_DIR / RESULTS_JSON)
    if sres.get("status") != "ok" or sres.get("arm") != ARM_CAUSALRCA:
        raise ValueError(f"{score_record} is not a completed {ARM_CAUSALRCA} score record")
    if sha256_file(score_record / GRAPH_JSON) != sres["graph_sha256"]:
        raise ValueError(f"{score_record / GRAPH_JSON} differs from the file its record hashed")
    corpus_id = corpus_id_of(corpus_dir)
    if corpus_id != sres["corpus_id"]:
        raise ValueError(f"the score read is of {sres['corpus_id']!r}; this corpus is {corpus_id!r}")
    seed = seed_of(corpus_id)
    taus = [float(t) for t in taus]
    if split == "val":
        if seed != VAL_SEED or freeze != "none" or taus != [float(t) for t in CAUSALRCA_GRAPH_TAUS]:
            raise ValueError("a validation read is seed 0, no freeze, the registered grid")
        frozen = None
    else:
        if seed not in TEST_SEEDS or freeze == "none":
            raise ValueError("a test read is seeds 1-4 under a freeze")
        frozen = read_json(freeze)
        if (frozen["rung"], frozen["series"], frozen["arm"]) != (sres["rung"], sres["series"], ARM_CAUSALRCA):
            raise ValueError(f"{freeze} freezes {frozen['rung']} {frozen['series']}, not this read")
        if taus != [float(frozen["tau"])]:
            raise ValueError(f"a test read takes the frozen tau {frozen['tau']} only")
    graphs = read_json(score_record / GRAPH_JSON)["cases"]
    case_ids = [c["case_id"] for c in load_cases(corpus_dir)]
    if set(graphs) != set(case_ids):
        raise ValueError("graph.json's cases are not the labelled cases")
    topo = score_service(corpus_dir, topology_reference(corpus_dir))
    topo_row = {**directed(topo), "auroc": topo["auroc"]["directed"]}
    cases = [{"case_id": cid, **score_case(corpus_dir, graphs[cid], taus)} for cid in case_ids]
    per_tau = {}
    for t in taus:
        rows = [{**c["by_tau"][repr(t)], "auroc": c["auroc"]} for c in cases]
        per_tau[repr(t)] = {m: mean_over(rows, m) for m in (*METRICS, "auroc")}
    return sres, frozen, seed, cases, per_tau, topo_row


def run(corpus_dir, score_record, split, freeze, taus, out):
    sres, frozen, seed, cases, per_tau, topo_row = read(corpus_dir, score_record, split, freeze, taus)
    doc = {
        "axis": "graph",
        "arm": ARM_CAUSALRCA,
        "series": sres["series"],
        "rung": sres["rung"],
        "corpus_id": sres["corpus_id"],
        "seed": seed,
        "split": split,
        "freeze": None
        if frozen is None
        else {"path": str(freeze), "sha256": sha256_file(freeze), "tau": frozen["tau"]},
        "taus": [float(t) for t in taus],
        "mean_over_cases": per_tau,
        "floor_topology": topo_row,
        "n_cases": len(cases),
        "n_cases_without_graph": sum(1 for c in cases if c["n_pairs_listed"] == 0),
        "cases": cases,
    }
    out = Path(out)
    write_json(out / SCORE_SERVICE_JSON, doc)
    return {
        "stage": STAGE,
        "arm": ARM_CAUSALRCA,
        "series": sres["series"],
        "rung": sres["rung"],
        "corpus_id": sres["corpus_id"],
        "split": split,
        "score_record_sha256": sha256_file(Path(score_record) / RUN_DIR / RESULTS_JSON),
        "score_service_sha256": sha256_file(out / SCORE_SERVICE_JSON),
        "mean_over_cases": per_tau,
        "floor_topology": topo_row,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True, help="the pulled corpus (tier rca)")
    p.add_argument("--score-record", required=True, help="the scorecausalrca stage's folder (<arm folder>/<series>)")
    p.add_argument("--split", required=True, choices=("val", "test"), help="val = seed 0; test = seeds 1-4")
    p.add_argument("--freeze", required=True, help="'none' for the validation read; the freeze file for a test read")
    p.add_argument(
        "--taus", required=True, nargs="+", type=float, help="the registered grid (val) or the frozen tau (test)"
    )
    p.add_argument("--output-folder", required=True, help="<arm folder>/<series>/graph-<split>")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = run(args.corpus_dir, args.score_record, args.split, args.freeze, args.taus, args.output_folder)
        log({"event": "graph_axis", "series": res["series"], "corpus_id": res["corpus_id"], "split": res["split"]})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
