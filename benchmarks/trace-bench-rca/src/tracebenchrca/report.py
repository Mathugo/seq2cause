"""Pool the landed ranking-axis results per rung and across rungs, and score the hypotheses.

    python -m tracebenchrca.report --results-dir results --rungs xs s m --seeds 0 1 2 3 4 \
        --arms rca/baro rca/baro-bocpd control/rule rca/causalrca --bootstrap-b 2000 --bootstrap-seed 0 --output-folder tables/rca-latent

Reads, per rung, seed and arm, the landed `results/<rung>/latent/seed=<k>/<arm slug>/score-rca.json`
and the two stage records beside it; refuses a rung with fewer seeds than named (a pooled table
is five seeds or nothing). Writes per arm one pooled table per rung and one across the named rungs
(`tracebench.score_rca.pool`: the cases concatenated, the same bootstrap), `summary.json` (every
row with its interval, the analytic random expectation — `control/random` — the miss counts by
reason, for the full pipeline the change-point offsets and for `control/rule` the alerting services
per case) and `rca-latent.md`; the hypotheses
of `plans/rca-arms.md` §6 are scored mechanically from the pooled rows.

`rca/causalrca` is read per series (`results/<rung>/latent/seed=<k>/rca-causalrca/<series>/`), one row
each, pooled from the run-averaged per-case hit rates (`scorecausalrca.pool_run_averaged`). Its graph
axis is reported on the **four test seeds 1–4** (seed 0 is the axis's validation split; the
four-seed exception of `plans/rca-arms.md`, addendum 2026-10-22): per rung and series the mean over
the test cases of the per-case directed precision, recall, F1 and AUROC at the frozen τ, with
`floor/topology` over the same cases, and H-graph-topology scored on them.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from tracebench.score_rca import pool

from .arms import arm_slug
from .constants import (
    ARM_BARO,
    ARM_BARO_BOCPD,
    ARM_CAUSALRCA,
    ARM_CIRCA,
    ARM_CIRCA_TOPOLOGY,
    ARM_CONTROL_RULE,
    ARM_RCD,
    ARM_TRACERCA,
    CAUSALRCA_SERIES,
    RESULTS_JSON,
    ROOT_CAUSE_ARMS,
    RUN_DIR,
    RUNGS,
    SCORE_RCA_JSON,
    SEEDS,
    VARIANT,
)
from .graphaxis import SCORE_SERVICE_JSON, TEST_SEEDS
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json
from .scorecausalrca import pool_run_averaged

STAGE = "report"
AVG = "avg@5"
BOCPD_GAP = 0.10  # H-rca-bocpd's registered tolerance (plans/rca-arms.md §6)
RUN_AVERAGED_ARMS = (
    ARM_CAUSALRCA,
    ARM_RCD,
)  # arms scored by per-run averaging (plans/rca-arms.md, addenda 2026-10-23 / 2026-10-25)
REPORT_ARMS = tuple(a for a in ROOT_CAUSE_ARMS) + (ARM_CONTROL_RULE,)
# the stretch arms' hypotheses: the §6 random / rule rules applied per arm (addenda 2026-10-25)
PER_ARM_HYPOTHESES = {
    ARM_CIRCA: "circa",
    ARM_CIRCA_TOPOLOGY: "circa-topology",
    ARM_RCD: "rcd",
    ARM_TRACERCA: "tracerca",
}


def cell_dir(results_dir, rung, seed, arm, series=None):
    d = Path(results_dir) / rung / VARIANT / f"seed={seed}" / arm_slug(arm)
    return d if series is None else d / series


def load_cell(results_dir, rung, seed, arm, series=None):
    d = cell_dir(results_dir, rung, seed, arm, series)
    score_res = read_json(d / RUN_DIR / RESULTS_JSON)
    rank_res = read_json(d / "rank" / RUN_DIR / RESULTS_JSON)
    if score_res.get("status") != "ok" or rank_res.get("status") != "ok":
        raise ValueError(f"{d} holds a failed record")
    if score_res["arm"] != arm or score_res["rung"] != rung:
        raise ValueError(f"{d} holds an {score_res['arm']!r} rung {score_res['rung']!r} result")
    if sha256_file(d / SCORE_RCA_JSON) != score_res["score_rca_sha256"]:
        raise ValueError(f"{d / SCORE_RCA_JSON} differs from the file its record hashed")
    return read_json(d / SCORE_RCA_JSON), score_res, rank_res


def half_width(row):
    iv = row["bootstrap"]["intervals"][AVG]
    return (iv["high"] - iv["low"]) / 2


def row_of(rung, arm, pooled, cells):
    misses = {}
    for _, _, rank_res in cells:
        for reason, n in rank_res.get("misses_by_reason", {}).items():
            misses[reason] = misses.get(reason, 0) + n
    row = {
        "rung": rung,
        "arm": arm,
        "metrics": pooled["metrics"],
        "bootstrap": pooled["bootstrap"],
        "random": pooled["random"],
        "n_missing": pooled["n_missing"],
        "misses_by_reason": misses,
        "n_empty_rankings": sum(r.get("n_empty_rankings", 0) for _, _, r in cells),
        "n_filled_cells": sum(r["n_filled_cells"] for _, _, r in cells),
        "n_metrics_lt_min_pre": sum(r["n_metrics_lt_min_pre"] for _, _, r in cells),
        "by_fault_kind": pooled["by_fault_kind"],
    }
    if arm == ARM_CAUSALRCA:
        row["series"] = cells[0][1]["series"]
    if arm in RUN_AVERAGED_ARMS:
        row["n_failed_runs"] = sum(r["n_failed_runs"] for _, _, r in cells)
        row["aggregation"] = pooled["aggregation"]
    unranked = [u for _, _, r in cells for c in r["cases"] for u in c.get("n_services_unranked_per_run", [])]
    if unranked:
        row["unranked_services_per_run"] = {
            "mean": statistics.mean(unranked),
            "max": max(unranked),
            "n_runs": len(unranked),
        }
    edges = [c["n_edges"] for _, _, r in cells for c in r["cases"] if "n_edges" in c]
    if edges:
        row["edges_per_case"] = {
            "mean": statistics.mean(edges),
            "min": min(edges),
            "max": max(edges),
            "n_cases": len(edges),
        }
    offsets = [
        cp["change_point_offset_s"]
        for _, s, _ in cells
        for cp in s.get("change_points", [])
        if cp["change_point_offset_s"] is not None
    ]
    alerting = [c["n_alerting_services"] for _, _, r in cells for c in r["cases"] if "n_alerting_services" in c]
    if alerting:
        row["alerting_services_per_case"] = {
            "mean": statistics.mean(alerting),
            "min": min(alerting),
            "max": max(alerting),
            "n_cases": len(alerting),
        }
    if any("change_points" in s for _, s, _ in cells):
        row["change_point_offset_s"] = {
            "n": len(offsets),
            "median": statistics.median(offsets) if offsets else None,
            "min": min(offsets) if offsets else None,
            "max": max(offsets) if offsets else None,
            "n_before_injection": sum(1 for o in offsets if o < 0),
        }
    return row


def hypotheses(rows, rungs, graph_rows=()):
    by = {(r["rung"], r["arm"]): r for r in rows}
    out = {}
    if all((g, ARM_BARO) in by for g in rungs):
        per = {
            g: by[(g, ARM_BARO)]["bootstrap"]["intervals"][AVG]["low"] > by[(g, ARM_BARO)]["random"][AVG] for g in rungs
        }
        out["H-rca-random"] = {"per_rung": per, "verdict": "pass" if all(per.values()) else "fail"}
    if all((g, a) in by for g in rungs for a in (ARM_BARO, ARM_BARO_BOCPD)):
        gaps = {g: by[(g, ARM_BARO_BOCPD)]["metrics"][AVG] - by[(g, ARM_BARO)]["metrics"][AVG] for g in rungs}
        out["H-rca-bocpd"] = {
            "gap_bocpd_minus_baro": gaps,
            "verdict": "pass" if all(abs(v) <= BOCPD_GAP for v in gaps.values()) else "fail",
        }
    if (RUNGS[0], ARM_BARO) in by and (RUNGS[-1], ARM_BARO) in by:
        lo, hi = by[(RUNGS[0], ARM_BARO)], by[(RUNGS[-1], ARM_BARO)]
        rise = hi["metrics"][AVG] - lo["metrics"][AVG]
        bound = max(half_width(lo), half_width(hi))
        out["H-rca-scale"] = {
            "rise_m_minus_xs": rise,
            "wider_half_width": bound,
            "verdict": "fail" if rise > bound else "pass",
        }
    if all((g, a) in by for g in rungs for a in (ARM_BARO, ARM_CONTROL_RULE)):
        per = {g: by[(g, ARM_BARO)]["metrics"][AVG] >= by[(g, ARM_CONTROL_RULE)]["metrics"][AVG] for g in rungs}
        out["H-rca-rule"] = {
            "gap_baro_minus_rule": {
                g: by[(g, ARM_BARO)]["metrics"][AVG] - by[(g, ARM_CONTROL_RULE)]["metrics"][AVG] for g in rungs
            },
            "per_rung": per,
            "verdict": "pass" if all(per.values()) else "fail",
        }
    else:
        out["H-rca-rule"] = {"verdict": "not run", "reason": "needs rca/baro and control/rule on every named rung"}
    for arm, slug in PER_ARM_HYPOTHESES.items():
        if all((g, arm) in by for g in rungs):
            per = {g: by[(g, arm)]["bootstrap"]["intervals"][AVG]["low"] > by[(g, arm)]["random"][AVG] for g in rungs}
            out[f"H-rca-{slug}-random"] = {"per_rung": per, "verdict": "pass" if all(per.values()) else "fail"}
            if all((g, ARM_CONTROL_RULE) in by for g in rungs):
                per = {g: by[(g, arm)]["metrics"][AVG] >= by[(g, ARM_CONTROL_RULE)]["metrics"][AVG] for g in rungs}
                out[f"H-rca-{slug}-rule"] = {
                    "gap_arm_minus_rule": {
                        g: by[(g, arm)]["metrics"][AVG] - by[(g, ARM_CONTROL_RULE)]["metrics"][AVG] for g in rungs
                    },
                    "per_rung": per,
                    "verdict": "pass" if all(per.values()) else "fail",
                }
    if all((g, a) in by for g in rungs for a in (ARM_CIRCA, ARM_CIRCA_TOPOLOGY)):
        per = {g: by[(g, ARM_CIRCA_TOPOLOGY)]["metrics"][AVG] >= by[(g, ARM_CIRCA)]["metrics"][AVG] for g in rungs}
        out["H-circa-topology-gain"] = {
            "gap_topology_minus_pc": {
                g: by[(g, ARM_CIRCA_TOPOLOGY)]["metrics"][AVG] - by[(g, ARM_CIRCA)]["metrics"][AVG] for g in rungs
            },
            "per_rung": per,
            "verdict": "pass" if all(per.values()) else "fail",
        }
    if graph_rows and all(any(g["rung"] == r for g in graph_rows) for r in rungs):
        per = {f"{g['rung']} {g['series']}": g["mean"]["f1"] <= g["floor_topology"]["f1"] for g in graph_rows}
        out["H-graph-topology"] = {
            "gap_causalrca_minus_topology_f1": {
                f"{g['rung']} {g['series']}": g["mean"]["f1"] - g["floor_topology"]["f1"] for g in graph_rows
            },
            "per_rung_and_series": per,
            "seeds": list(TEST_SEEDS),
            "verdict": "pass" if all(per.values()) else "fail",
        }
    else:
        out["H-graph-topology"] = {
            "verdict": "not run",
            "reason": "needs rca/causalrca's graph test reads on every named rung",
        }
    return out


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def graph_row(results_dir, rung, series):
    """The graph axis of one rung and series over the four test seeds: the mean over every test case."""
    cases, freezes, topo = [], set(), []
    for k in TEST_SEEDS:
        d = cell_dir(results_dir, rung, k, ARM_CAUSALRCA, series) / "graph-test"
        rec = read_json(d / RUN_DIR / RESULTS_JSON)
        doc = read_json(d / SCORE_SERVICE_JSON)
        if rec.get("status") != "ok" or sha256_file(d / SCORE_SERVICE_JSON) != rec["score_service_sha256"]:
            raise ValueError(f"{d} holds a failed or changed graph test read")
        (tau,) = doc["taus"]
        freezes.add((doc["freeze"]["sha256"], tau))
        for c in doc["cases"]:
            cases.append({**c["by_tau"][repr(float(tau))], "auroc": c["auroc"]})
            topo.append(doc["floor_topology"])
    if len(freezes) != 1:
        raise ValueError(f"{rung} {series}: the test seeds were read under different freezes {freezes}")
    mean = lambda rows, m: sum(r[m] for r in rows) / len(rows)  # noqa: E731
    metrics = ("precision", "recall", "f1", "auroc")
    return {
        "rung": rung,
        "arm": ARM_CAUSALRCA,
        "series": series,
        "seeds": list(TEST_SEEDS),
        "tau": tau,
        "n_cases": len(cases),
        "mean": {m: mean(cases, m) for m in metrics},
        "floor_topology": {m: mean(topo, m) for m in metrics},
    }


def markdown(rows, hyp, graph_rows=()):
    lines = [
        "| rung | arm | cases | AC@1 | AC@3 | AC@5 | Avg@5 [95 % CI] | random Avg@5 | misses |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m, iv, rnd = r["metrics"], r["bootstrap"]["intervals"].get(AVG, {}), r["random"]
        label = f"`{r['arm']}` ({r['series']})" if "series" in r else f"`{r['arm']}`"
        lines.append(
            f"| {r['rung']} | {label} | {m['n_cases']} | {fmt(m['ac@1'])} | {fmt(m['ac@3'])} | {fmt(m['ac@5'])} | "
            f"{fmt(m[AVG])} [{fmt(iv.get('low'))}, {fmt(iv.get('high'))}] | {fmt(rnd[AVG] if rnd else None)} | "
            f"{r['n_missing']} |"
        )
    edge_rows = [r for r in rows if "edges_per_case" in r]
    if edge_rows:
        lines += ["", "| rung | arm | graph edges per case, mean [min, max] |", "|---|---|---|"]
        for r in edge_rows:
            e = r["edges_per_case"]
            lines.append(f"| {r['rung']} | `{r['arm']}` | {e['mean']:.1f} [{e['min']}, {e['max']}] |")
    rule_rows = [r for r in rows if "alerting_services_per_case" in r]
    if rule_rows:
        lines += ["", "| rung | arm | alerting services per case, mean [min, max] |", "|---|---|---|"]
        for r in rule_rows:
            a = r["alerting_services_per_case"]
            lines.append(f"| {r['rung']} | `{r['arm']}` | {a['mean']:.2f} [{a['min']}, {a['max']}] |")
    if graph_rows:
        lines += [
            "",
            "Graph axis — four test seeds (1–4); seed 0 is the graph axis's validation split. Mean over the test cases.",
            "",
            "| rung | series | τ | cases | precision | recall | F1 | AUROC | `floor/topology` F1 | AUROC |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g in graph_rows:
            m, f = g["mean"], g["floor_topology"]
            lines.append(
                f"| {g['rung']} | {g['series']} | {g['tau']} | {g['n_cases']} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                f"{fmt(m['f1'])} | {fmt(m['auroc'])} | {fmt(f['f1'])} | {fmt(f['auroc'])} |"
            )
    lines += ["", "| hypothesis | verdict |", "|---|---|"]
    lines += [f"| {k} | {v['verdict']} |" for k, v in hyp.items()]
    return "\n".join(lines) + "\n"


def report(results_dir, rungs, seeds, arms, b, seed, out):
    out = Path(out)
    rows, files = [], {}
    graph_rows = []
    for arm in arms:
        for series in CAUSALRCA_SERIES if arm == ARM_CAUSALRCA else (None,):
            pooler = pool_run_averaged if arm in RUN_AVERAGED_ARMS else pool
            stem = arm_slug(arm) if series is None else f"{arm_slug(arm)}-{series}"
            all_results = []
            for rung in rungs:
                cells = [load_cell(results_dir, rung, k, arm, series) for k in seeds]
                pooled = pooler([c[0] for c in cells], b, seed)
                name = f"pooled-{rung}-{stem}.json"
                write_json(out / name, pooled)
                files[name] = sha256_file(out / name)
                rows.append(row_of(rung, arm, pooled, cells))
                all_results += [c[0] for c in cells]
                if arm == ARM_CAUSALRCA:
                    graph_rows.append(graph_row(results_dir, rung, series))
            if len(rungs) > 1:
                name = f"pooled-{'-'.join(rungs)}-{stem}.json"
                write_json(out / name, pooler(all_results, b, seed))
                files[name] = sha256_file(out / name)
    hyp = hypotheses(rows, rungs, graph_rows)
    write_json(out / "summary.json", {"rows": rows, "graph_rows": graph_rows, "hypotheses": hyp})
    (out / "rca-latent.md").write_text(markdown(rows, hyp, graph_rows), encoding="utf-8")
    files["summary.json"] = sha256_file(out / "summary.json")
    files["rca-latent.md"] = sha256_file(out / "rca-latent.md")
    return {
        "stage": STAGE,
        "rungs": list(rungs),
        "seeds": list(seeds),
        "arms": list(arms),
        "files": files,
        "hypotheses": {k: v["verdict"] for k, v in hyp.items()},
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--rungs", required=True, nargs="+", choices=RUNGS)
    p.add_argument("--seeds", required=True, nargs="+", type=int, choices=SEEDS)
    p.add_argument("--arms", required=True, nargs="+", choices=REPORT_ARMS)
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples of the pooled tables")
    p.add_argument("--bootstrap-seed", required=True, type=int)
    p.add_argument("--output-folder", required=True, help="the tables directory and this stage's run record")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if sorted(args.seeds) != list(SEEDS):
        raise SystemExit(f"a pooled table is the five seeds {SEEDS} or nothing; got {args.seeds}")
    rungs = [g for g in RUNGS if g in args.rungs]
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = report(
            args.results_dir, rungs, args.seeds, args.arms, args.bootstrap_b, args.bootstrap_seed, args.output_folder
        )
        log({"event": "report", **res["hypotheses"]})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
