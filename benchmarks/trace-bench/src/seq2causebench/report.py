# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Five-seed tables per cell and paired differences with a significance test (PRD Interface
"report"; non-negotiables 8, 9; scenarios 3, 7, 8, 9, 35, 39, 41).

    python -m seq2causebench.report --results-dir results --rung xs --variant latent \
        --pairs trace/core/shipped/frozen:trace/cli/shipped/frozen trace/cli/shipped/shipped:trace/cli/shipped/frozen \
        --reasons tables/reasons.json --output-folder tables/<run>

Reads `results/<rung>/<variant>/seed=<k>/<arm>/<path>/<cut>/<grain>/{score.json,annotate.json}` and,
where present, `seqscore.json`. Per cell every metric is `tracebench.score.headline` over the five
seeds (refused below five, scenario 3); a cell absent on every seed must carry a reason
(scenario 7); a cell whose seeds mix benchmark tool versions is refused. Bidirected columns carry
the structural-limitation note and count (scenario 6). For every `a:b` pair of `<arm>/<path>/<cut>`
triples the per-seed difference `a − b` is taken only after asserting `(corpus_id, seed,
model_sha256)` equal in that cell (scenarios 9, 41), then headlined, with a **paired t-test** on the
five per-seed differences (`scipy.stats.ttest_rel`; low power at five pairs is stated). A pair
whose two sides differ in cut is allowed only between the same arm and path (the shipped-versus-
frozen row); differencing a core arm against a shipped cut is refused (scenario 39). Per-sequence
results are written to their own tables, never beside a type-level number (scenario 35).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import stats as sps
from tracebench.score import headline

from .arms import cell_key, parse_cell_key
from .constants import ARM_CORE, GRAINS, REFERENCE_ARMS, RUNGS, SEQSCORE_JSON, VARIANTS
from .log import log
from .record import RunRecord, read_json, write_json

SCORE_JSON = "score.json"
ANNOTATE_JSON = "annotate.json"
REPORT_SCHEMA = "seq2causebench/report@1"
PERSEQ_SCHEMA = "seq2causebench/report-perseq@1"

METRICS = (
    "directed.f1",
    "directed.precision",
    "directed.recall",
    "directed.shd",
    "skeleton.f1",
    "skeleton.shd",
    "orientation.accuracy",
    "orientation.accuracy_compelled",
    "bidirected.recall",
    "bidirected.precision",
    "shd_mixed",
    "auroc.directed",
    "average_precision.directed",
    "auroc.skeleton",
    "causal_validity.sid",
    "causal_validity.parent_aid",
    "causal_validity.ancestor_aid",
    "coverage.reachable_recall_ceiling",
    "corrupted_cells.total",
)
PERSEQ_METRICS = (
    "pooled.directed.f1",
    "pooled.directed.precision",
    "pooled.directed.recall",
    "pooled.skeleton.f1",
    "pooled.auroc",
    "pooled.ap",
    "per_sequence.directed.f1.mean",
    "per_sequence.auroc.mean",
    "scoreable_fraction",
    "predict_all.pooled",
    "pooled.ancestor_directed.f1",
)
MIN_SEEDS = 5


class ReportRefusal(RuntimeError):
    pass


def flatten(score, annotate):
    flat = {}
    for axis in ("directed", "skeleton", "bidirected", "orientation"):
        for k, v in score[axis].items():
            flat[f"{axis}.{k}"] = v
    flat["shd_mixed"] = score["shd_mixed"]
    for axis in ("auroc", "average_precision"):
        for k, v in annotate["score_ranking"][axis].items():
            flat[f"{axis}.{k}"] = v
    cv = annotate["causal_validity"]
    if cv.get("value"):
        for k, v in cv["value"].items():
            flat[f"causal_validity.{k}"] = v
        flat["causal_validity.reason"] = None
    else:
        flat["causal_validity.reason"] = cv.get("reason")
    for k, v in annotate["per_lag_recall"].items():
        flat[f"per_lag_recall.{k}"] = v
    flat["coverage.reachable_recall_ceiling"] = annotate["coverage"]["reachable_recall_ceiling"]
    cc = annotate.get("corrupted_cells") or {}
    flat["corrupted_cells.total"] = int(cc.get("n_cell_collapsed", 0)) + int(
        cc.get("n_cell_saturated", 0)
    )
    return flat


def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def load_cells(results_dir, rung, variant):
    """`{cell_key: {seed: {"flat", "annotate", "seqscore"}}}` over every seed directory present."""
    root = Path(results_dir) / rung / variant
    cells = {}
    for seed_dir in sorted(root.glob("seed=*")):
        seed = int(seed_dir.name.split("=")[1])
        for ann_path in sorted(seed_dir.rglob(ANNOTATE_JSON)):
            cell_dir = ann_path.parent
            score_path = cell_dir / SCORE_JSON
            if not score_path.exists():
                raise ReportRefusal(f"{cell_dir} has {ANNOTATE_JSON} but no {SCORE_JSON}")
            annotate = read_json(ann_path)
            score = read_json(score_path)
            key = annotate["cell"]
            parse_cell_key(key)
            if annotate.get("seed") is not None and int(annotate["seed"]) != seed:
                raise ReportRefusal(
                    f"{cell_dir}: annotate says seed {annotate['seed']} but sits under seed={seed}"
                )
            # the job writes `seqscore` into `<cell>/seqscore/` (its own run record beside the
            # annotate record); a copy at the cell root is accepted too (the pipeline test's layout)
            seq_path = cell_dir / SEQSCORE_JSON
            if not seq_path.exists():
                seq_path = cell_dir / "seqscore" / SEQSCORE_JSON
            cells.setdefault(key, {})[seed] = {
                "flat": flatten(score, annotate),
                "annotate": annotate,
                "seqscore": read_json(seq_path) if seq_path.exists() else None,
            }
    return cells


def _headline(per_seed, key):
    vals = [s["flat"].get(key) for s in per_seed]
    if any(v is None for v in vals):
        reasons = sorted(
            {
                str(s["flat"].get("causal_validity.reason") or "value absent on at least one seed")
                for s in per_seed
                if s["flat"].get(key) is None
            }
        )
        return {"metric": key, "n_seeds": len(vals), "reason": "; ".join(reasons)}
    return headline([{"v": v} for v in vals], "v") | {"metric": key}


def paired_test(diffs):
    """`scipy.stats.ttest_rel` on the per-seed differences (equivalently a one-sample t-test of the
    differences against zero): statistic, p-value, df, the mean difference and its standard deviation."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"test": "paired t-test", "n": int(n), "reason": "fewer than two pairs"}
    res = sps.ttest_rel(d, np.zeros_like(d))
    stat = float(res.statistic) if math.isfinite(float(res.statistic)) else None
    p = float(res.pvalue) if math.isfinite(float(res.pvalue)) else None
    return {
        "test": "paired t-test",
        "n": int(n),
        "df": int(n - 1),
        "statistic": stat,
        "pvalue": p,
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=1)),
        "values": d.tolist(),
        "note": f"{n} pairs: low power; the effect size (mean_diff, std_diff) is the primary readout",
    }


def cell_table(per_seed, key):
    seeds = sorted(per_seed)
    if len(seeds) < MIN_SEEDS:
        raise ReportRefusal(
            f"cell {key} has {len(seeds)} seed(s) {seeds}; a headline needs at least {MIN_SEEDS} (PRD scenario 3)"
        )
    versions = {s["annotate"].get("tool_version") for s in per_seed.values()}
    if len(versions) != 1:
        raise ReportRefusal(
            f"cell {key} mixes benchmark tool versions {sorted(map(str, versions))}"
        )
    rows = [per_seed[s] for s in seeds]
    out = {
        "cell": key,
        "seeds": seeds,
        "tool_version": versions.pop(),
        "n_seeds": len(seeds),
        "corpus_ids": [per_seed[s]["annotate"]["corpus_id"] for s in seeds],
        "model_sha256": [per_seed[s]["annotate"]["model_sha256"] for s in seeds],
        "metrics": {m: _headline(rows, m) for m in METRICS},
    }
    lags = sorted(
        {int(k.split(".")[1]) for r in rows for k in r["flat"] if k.startswith("per_lag_recall.")}
    )
    out["per_lag_recall"] = {
        str(k): _headline(rows, f"per_lag_recall.{k}")
        for k in lags
        if all(f"per_lag_recall.{k}" in r["flat"] for r in rows)
    }
    lim = rows[0]["annotate"].get("structural_limitation")
    if lim and lim.get("assumption"):
        out["structural_limitation"] = {
            "assumption": lim["assumption"],
            "truth_bidirected_edges": [
                r["annotate"]["structural_limitation"]["truth_bidirected_edges"] for r in rows
            ],
            "note": lim.get("note"),
        }
    out["in_regime"] = [bool(r["annotate"]["oracle"]["in_regime"]) for r in rows]
    out["empty_prediction"] = [r["annotate"].get("empty_prediction") for r in rows]
    out["confounded_pairs"] = [
        r["annotate"]
        .get("confounded_pairs", {})
        .get("directed_predictions_on_bidirected_truth_pairs")
        for r in rows
    ]
    return out


def _check_pair(ka, kb):
    arm_a, path_a, cut_a, _ = parse_cell_key(ka)
    arm_b, path_b, cut_b, _ = parse_cell_key(kb)
    if cut_a != cut_b and (arm_a, path_a) != (arm_b, path_b):
        raise ReportRefusal(
            f"pair {ka} vs {kb}: cuts differ; a cut difference is reported within one arm and path only, "
            "never across arms (PRD scenario 39)"
        )
    if (arm_a == ARM_CORE or arm_b == ARM_CORE) and "shipped" in (cut_a, cut_b):
        raise ReportRefusal(
            f"pair {ka} vs {kb}: the core arm is differenced against the frozen cut only (PRD scenario 39)"
        )


def paired_table(cells, a_triple, b_triple, grain):
    ka, kb = cell_key(*a_triple, grain), cell_key(*b_triple, grain)
    _check_pair(ka, kb)
    if ka not in cells or kb not in cells:
        return None
    a, b = cells[ka], cells[kb]
    seeds = sorted(set(a) & set(b))
    if len(seeds) < MIN_SEEDS:
        raise ReportRefusal(
            f"pair {ka} vs {kb}: {len(seeds)} shared seed(s); a paired headline needs {MIN_SEEDS} (PRD scenario 3)"
        )
    triples = []
    for s in seeds:
        ta = (
            a[s]["annotate"]["corpus_id"],
            int(a[s]["annotate"]["seed"]),
            a[s]["annotate"]["model_sha256"],
        )
        tb = (
            b[s]["annotate"]["corpus_id"],
            int(b[s]["annotate"]["seed"]),
            b[s]["annotate"]["model_sha256"],
        )
        if ta != tb:
            raise ReportRefusal(
                f"pair {ka} vs {kb} at seed {s}: (corpus_id, seed, model_sha256) differ: {ta} vs {tb} (PRD scenario 9)"
            )
        triples.append(list(ta))
    diffs = []
    for s in seeds:
        fa, fb = a[s]["flat"], b[s]["flat"]
        diffs.append(
            {
                "flat": {
                    k: (fa[k] - fb[k])
                    if (
                        fa.get(k) is not None
                        and fb.get(k) is not None
                        and not isinstance(fa[k], str)
                    )
                    else None
                    for k in set(fa) | set(fb)
                }
            }
        )
    lags = sorted(
        {
            int(k.split(".")[1])
            for d in diffs
            for k in d["flat"]
            if k.startswith("per_lag_recall.") and d["flat"][k] is not None
        }
    )
    tests = {}
    for m in METRICS:
        vals = [d["flat"].get(m) for d in diffs]
        tests[m] = (
            paired_test(vals)
            if all(v is not None for v in vals)
            else {"test": "paired t-test", "reason": "value absent on at least one seed"}
        )
    return {
        "pair": f"{ka} - {kb}",
        "grain": grain,
        "seeds": seeds,
        "triples": triples,
        "metrics": {m: _headline(diffs, m) for m in METRICS},
        "tests": tests,
        "per_lag_recall": {str(k): _headline(diffs, f"per_lag_recall.{k}") for k in lags},
    }


def perseq_tables(cells):
    """Per-sequence headlines per cell and grain, in their own document (scenario 35)."""
    out = {}
    for key, per_seed in cells.items():
        seeds = sorted(s for s in per_seed if per_seed[s]["seqscore"] is not None)
        if not seeds:
            continue
        if len(seeds) < MIN_SEEDS:
            raise ReportRefusal(
                f"per-sequence cell {key} has {len(seeds)} seed(s); a headline needs {MIN_SEEDS}"
            )
        rows = [
            {"flat": {m: _get(per_seed[s]["seqscore"], m) for m in PERSEQ_METRICS}} for s in seeds
        ]
        out[key] = {
            "cell": key,
            "grain": parse_cell_key(key)[3],
            "seeds": seeds,
            "metrics": {m: _headline(rows, m) for m in PERSEQ_METRICS},
            "n_scoreable": [per_seed[s]["seqscore"]["n_scoreable"] for s in seeds],
            "n_sequences": [per_seed[s]["seqscore"]["n_sequences"] for s in seeds],
            "rules_version": per_seed[seeds[0]]["seqscore"]["rules_version"],
        }
    return out


def expected_cells(cells, triples):
    keys = set(cells)
    for t in triples:
        for grain in GRAINS:
            keys.add(cell_key(*t, grain))
    return sorted(keys)


def _triple(spec):
    arm, path, cut = spec.rsplit("/", 2)
    return arm, path, cut


def run_report(args, rec):
    cells = load_cells(args.results_dir, args.rung, args.variant)
    reasons = read_json(args.reasons) if args.reasons else {}
    required = [(a, "shipped", "frozen") for a in REFERENCE_ARMS]
    seen = {parse_cell_key(k)[:3] for k in cells}
    tables, absent = {}, {}
    for key in expected_cells(cells, list(seen | set(required))):
        if key in cells:
            tables[key] = cell_table(cells[key], key)
        elif key in reasons:
            absent[key] = reasons[key]
        else:
            raise ReportRefusal(
                f"cell {key} is absent on every seed and --reasons gives no reason for it (PRD scenario 7)"
            )
    pairs = {}
    for spec in args.pairs:
        a_spec, b_spec = spec.split(":")
        a, b = _triple(a_spec), _triple(b_spec)
        for grain in GRAINS:
            t = paired_table(cells, a, b, grain)
            if t is not None:
                pairs[f"{a_spec}:{b_spec}/{grain}"] = t
    perseq = perseq_tables(cells)
    report = {
        "schema": REPORT_SCHEMA,
        "rung": args.rung,
        "variant": args.variant,
        "n_cells": len(tables),
        "cells": tables,
        "absent": absent,
        "n_absent": len(absent),
        "pairs": pairs,
        "metrics": list(METRICS),
    }
    stem = f"{args.rung}-{args.variant}"
    write_json(rec.out_dir / f"{stem}.json", report)
    (rec.out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    perseq_doc = {
        "schema": PERSEQ_SCHEMA,
        "axis": "per-sequence",
        "rung": args.rung,
        "variant": args.variant,
        "cells": perseq,
        "metrics": list(PERSEQ_METRICS),
    }
    write_json(rec.out_dir / f"{stem}-perseq.json", perseq_doc)
    (rec.out_dir / f"{stem}-perseq.md").write_text(
        render_perseq_markdown(perseq_doc), encoding="utf-8"
    )
    log(
        {
            "event": "report_done",
            "rung": args.rung,
            "variant": args.variant,
            "n_cells": len(tables),
            "n_absent": len(absent),
            "n_pairs": len(pairs),
            "n_perseq_cells": len(perseq),
        }
    )
    return {
        "report": f"{stem}.json",
        "markdown": f"{stem}.md",
        "perseq": f"{stem}-perseq.json",
        "n_cells": len(tables),
        "n_absent": len(absent),
        "n_pairs": len(pairs),
        "n_perseq_cells": len(perseq),
    }


# --- rendering -------------------------------------------------------------------------------------
SHOW = (
    "directed.f1",
    "directed.precision",
    "directed.recall",
    "directed.shd",
    "skeleton.f1",
    "orientation.accuracy",
    "auroc.directed",
    "average_precision.directed",
    "bidirected.recall",
    "shd_mixed",
    "causal_validity.sid",
    "causal_validity.parent_aid",
    "coverage.reachable_recall_ceiling",
    "corrupted_cells.total",
)


def _fmt(h):
    if "reason" in h:
        return f"n/a ({h['reason']})"
    return f"{h['mean']:.3f} ± {h['std']:.3f}"


def _fmt_test(t):
    if "reason" in t:
        return "n/a"
    p = "n/a" if t["pvalue"] is None else f"{t['pvalue']:.3f}"
    return f"{t['mean_diff']:+.3f} ± {t['std_diff']:.3f} (p={p}, n={t['n']})"


def render_markdown(report):
    lines = [f"# {report['rung']} / {report['variant']} — five-seed tables (type-level axis)", ""]
    for grain in GRAINS:
        rows = [(k, t) for k, t in report["cells"].items() if k.endswith(f"/{grain}")]
        if not rows:
            continue
        lines += [
            f"## {grain} grain",
            "",
            "| cell | " + " | ".join(SHOW) + " |",
            "|---|" + "---|" * len(SHOW),
        ]
        for key, t in rows:
            vals = []
            for m in SHOW:
                v = _fmt(t["metrics"][m])
                if m.startswith("bidirected") and "structural_limitation" in t:
                    n = t["structural_limitation"]["truth_bidirected_edges"]
                    v += f" [structural: {t['structural_limitation']['assumption']}; truth bidirected edges {min(n)}–{max(n)}]"
                vals.append(v)
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        lines.append("")
        lag_rows = [(k, t) for k, t in rows if t["per_lag_recall"]]
        if lag_rows:
            lags = sorted({int(lag) for _, t in lag_rows for lag in t["per_lag_recall"]})
            lines += [
                "### per-lag recall at the frozen τ",
                "",
                "| cell | " + " | ".join(f"lag {lag}" for lag in lags) + " |",
                "|---|" + "---|" * len(lags),
            ]
            for key, t in lag_rows:
                lines.append(
                    f"| {key} | "
                    + " | ".join(
                        _fmt(t["per_lag_recall"][str(lag)])
                        if str(lag) in t["per_lag_recall"]
                        else "—"
                        for lag in lags
                    )
                    + " |"
                )
            lines.append("")
    if report["absent"]:
        lines += (
            ["## absent cells", ""] + [f"- `{k}`: {v}" for k, v in report["absent"].items()] + [""]
        )
    if report["pairs"]:
        lines += [
            "## paired differences (a − b per seed, five-seed headline and paired t-test; (corpus_id, seed, model_sha256) equal per cell)",
            "",
        ]
        lines += ["| pair | " + " | ".join(SHOW) + " |", "|---|" + "---|" * len(SHOW)]
        for key, t in report["pairs"].items():
            lines.append(f"| {key} | " + " | ".join(_fmt_test(t["tests"][m]) for m in SHOW) + " |")
        lines.append("")
    return "\n".join(lines)


def render_perseq_markdown(doc):
    lines = [
        f"# {doc['rung']} / {doc['variant']} — per-sequence axis (its own tables; never merged with type-level numbers)",
        "",
    ]
    for grain in GRAINS:
        rows = [(k, t) for k, t in doc["cells"].items() if t["grain"] == grain]
        if not rows:
            continue
        lines += [
            f"## {grain} grain (per-sequence truth of the {grain} grain)",
            "",
            "| cell | " + " | ".join(PERSEQ_METRICS) + " | scoreable / sequences |",
            "|---|" + "---|" * (len(PERSEQ_METRICS) + 1),
        ]
        for key, t in rows:
            frac = ", ".join(
                f"{a}/{b}" for a, b in zip(t["n_scoreable"], t["n_sequences"], strict=True)
            )
            lines.append(
                f"| {key} | "
                + " | ".join(_fmt(t["metrics"][m]) for m in PERSEQ_METRICS)
                + f" | {frac} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True)
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument(
        "--pairs",
        required=True,
        nargs="+",
        help="pairs a:b of <arm>/<path>/<cut> triples to difference",
    )
    p.add_argument(
        "--reasons",
        required=True,
        help="JSON {cell_key: reason} for cells absent on every seed, or '' when none",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.reasons = args.reasons or None
    with RunRecord(args.output_folder, "report", vars(args)) as rec:
        res = run_report(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
