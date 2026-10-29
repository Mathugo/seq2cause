"""The GPU/CPU tolerance gate of the per-trace axis (plan of 2026-10-31, H8): the same corpus, cell and selection
probed on two devices must cut the same sample graphs up to a registered edge-set tolerance.

    python -m tracebenchrca.tolerance --gpu-work <runner work dir> --cpu-work <runner work dir> \
        --services <an export case's services.json> --threshold 0.80 --output-folder <run-dir>

Both work directories are `trace_runner.py` outputs (`<case_dir>.json` per case). For every case document present in
both, traces are paired by `trace_id`; the two sides' window positions must be identical (the selection, encoding and
vocabulary are device-free — a difference is a ValueError, not a tolerance), and the cut edge sets `{(j, q)}` are
compared by Jaccard (two empty sets agree: 1.0). Per case: the mean and 10th-percentile Jaccard over the common traces,
the share of identical graphs, and each side's case-level top-1 service under `tracerule.rank_case` with the services
list of `--services`. The gate is `mean Jaccard over every common trace >= --threshold`
(`constants.TRACE_TOLERANCE_JACCARD` is the registered value); it is REPORTED in the record, never enforced by the
exit code — a failed gate exits 0 with `gate.passed = false`, and the plan decides what a failure means.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import CASE_DIR_GLOB, TRACE_NONOK_OUTCOMES
from .log import log
from .record import RunRecord, read_json
from .tracerule import rank_case

STAGE = "tolerance"


def read_services_file(path):
    """The service names of an export's `services.json`, in file order (the export's name order)."""
    doc = read_json(path)
    return [str(s["name"]) for s in doc["services"]]


def case_docs(work_dir):
    """`{case_dir: path}` of the runner's per-case documents in one work directory."""
    return {p.stem: p for p in sorted(Path(work_dir).glob(f"{CASE_DIR_GLOB}.json"))}


def edge_set(trace):
    return {(int(j), int(q)) for j, q, *_ in trace["edges"]}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def top1(doc, services):
    ranked = rank_case(doc["traces"], services, TRACE_NONOK_OUTCOMES)["services"]
    return None if not ranked else ranked[0]


def compare_case(case_dir, gpu_doc, cpu_doc, services):
    """One case: the per-trace Jaccards over the common traces and the case row of the record."""
    gpu = {t["trace_id"]: t for t in gpu_doc["traces"]}
    cpu = {t["trace_id"]: t for t in cpu_doc["traces"]}
    if len(gpu) != len(gpu_doc["traces"]) or len(cpu) != len(cpu_doc["traces"]):
        raise ValueError(f"{case_dir}: duplicate trace ids in a runner document")
    common = [tid for tid in gpu if tid in cpu]
    values = []
    for tid in common:
        if gpu[tid]["positions"] != cpu[tid]["positions"]:
            raise ValueError(f"{case_dir}/{tid}: the two sides' window positions differ; the selection is device-free")
        values.append(jaccard(edge_set(gpu[tid]), edge_set(cpu[tid])))
    row = {
        "case_dir": case_dir,
        "n_traces": len(gpu),
        "n_traces_cpu": len(cpu),
        "n_common": len(common),
        "mean_jaccard": float(np.mean(values)) if values else None,
        "p10_jaccard": float(np.percentile(values, 10)) if values else None,
        "share_identical": float(np.mean([v == 1.0 for v in values])) if values else None,
        "top1_gpu": top1(gpu_doc, services),
        "top1_cpu": top1(cpu_doc, services),
    }
    return row, values


def tolerance(gpu_work, cpu_work, services_file, threshold):
    services = read_services_file(services_file)
    gpu_docs, cpu_docs = case_docs(gpu_work), case_docs(cpu_work)
    common_cases = [c for c in gpu_docs if c in cpu_docs]
    if not common_cases:
        raise FileNotFoundError(f"no case document is present in both {gpu_work} and {cpu_work}")
    per_case, values = [], []
    for case_dir in common_cases:
        row, vals = compare_case(case_dir, read_json(gpu_docs[case_dir]), read_json(cpu_docs[case_dir]), services)
        per_case.append(row)
        values.extend(vals)
        log({"event": "tolerance_case", "case_dir": case_dir, "n_common": row["n_common"], "mean": row["mean_jaccard"]})
    mean = float(np.mean(values)) if values else None
    return {
        "stage": STAGE,
        "gpu_work": str(gpu_work),
        "cpu_work": str(cpu_work),
        "services_file": str(services_file),
        "services": services,
        "n_cases": len(per_case),
        "n_cases_gpu_only": len([c for c in gpu_docs if c not in cpu_docs]),
        "n_cases_cpu_only": len([c for c in cpu_docs if c not in gpu_docs]),
        "n_traces_compared": len(values),
        "per_case": per_case,
        "mean_jaccard": mean,
        "gate": {"threshold": threshold, "passed": mean is not None and mean >= threshold},
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-work", required=True, help="the runner's work directory of the GPU read (per-case documents)")
    p.add_argument("--cpu-work", required=True, help="the runner's work directory of the CPU read of the same cases")
    p.add_argument(
        "--services", required=True, help="an export case's services.json (the deployed services, name order)"
    )
    p.add_argument("--threshold", required=True, type=float, help="the gate: mean edge-set Jaccard >= threshold")
    p.add_argument("--output-folder", required=True, help="where this command's run record is written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = tolerance(args.gpu_work, args.cpu_work, args.services, args.threshold)
        log({"event": STAGE, "mean_jaccard": res["mean_jaccard"], "passed": res["gate"]["passed"]})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
