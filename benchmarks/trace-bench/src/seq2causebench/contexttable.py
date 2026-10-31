"""The context table (`plans/reference-arms.md`, addendum 2026-10-31 §5 and §8): the family's cells
beside the landed references, the registered paired differences, the τ drift and the mechanical
verdicts of the family's type-level and per-sequence hypotheses.

    python -m seq2causebench.contexttable --landed-results results --context-results results-context \\
        --rung xs --variant latent \\
        --context-freezes freezes/<date>-xs-latent-s0-context.json ... (five seeds) \\
        --landed-freezes freezes/<date>-xs-latent-s0.json ... (the five landed model freezes) \\
        --test-tables <val-table.json of a scoresweep over each test read> ... (may be empty) \\
        --output-folder tables/context-latent/xs

Per grain: every family cell `<arm>/<path>/<cut>/<grain>` and the three landed reference cells
(`trace/core/fixed-kl/frozen`, `trace/cli-atomic/fixed-kl/frozen`, `trace/cli-atomic/fixed-kl/shipped`)
through `report.cell_table` (five seeds or nothing); the registered pairs through `report.paired_table`
(the per-seed difference and its paired t-test); per frozen cell and seed the frozen (c, N, τ), the
validation F1 at the freeze, the test F1 at the frozen τ (the landed read) and their difference (the
F1 drift), and — when a scoresweep over the test read's score file is given — the test-side argmax τ
and `log10(τ*_test / τ_frozen)` (the τ drift; absent when either τ is 0). Verdicts: H-ctx-cmi-f1,
H-ctx-cmi-tau, H-ctx-keep-perseq, H-atomic-cmi-f1, H-atomic-cmi-tau, H-atomic-cmi-shipped, each with
the registered fail clause; a hypothesis whose inputs are absent is "not computed", never passed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .arms import cell_key, parse_cell_key
from .constants import (
    ARM_CLI_ATOMIC,
    ARM_CLI_ATOMIC_CMI_KEEP,
    ARM_CORE,
    ARM_CORE_CMI_KEEP,
    ARM_CORE_CMI_OVERWRITE,
    ARM_CORE_IG_KEEP,
    ARM_CORE_IG_OVERWRITE,
    CUT_FROZEN,
    CUT_SHIPPED,
    FAMILY_CONTEXT,
    GRAINS,
    PATH_FIXED,
    PATH_FIXED_KL,
    RUNGS,
    VARIANTS,
)
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json
from .report import ReportRefusal, cell_table, load_cells, paired_table, paired_test

TABLE_SCHEMA = "seq2causebench/context-table@1"
F1_BAND = 0.01  # the registered band of the F1 hypotheses
TAU_STEP_DECADES = 0.5  # one grid step of the half-decade τ grid
ALPHA = 0.05
PERSEQ_KEY = "pooled.directed.f1"

IG_OVER = (ARM_CORE_IG_OVERWRITE, PATH_FIXED_KL, CUT_FROZEN)
CMI_OVER = (ARM_CORE_CMI_OVERWRITE, PATH_FIXED_KL, CUT_FROZEN)
IG_KEEP = (ARM_CORE_IG_KEEP, PATH_FIXED_KL, CUT_FROZEN)
CMI_KEEP = (ARM_CORE_CMI_KEEP, PATH_FIXED_KL, CUT_FROZEN)
ATOMIC = (ARM_CLI_ATOMIC_CMI_KEEP, PATH_FIXED_KL, CUT_FROZEN)
ATOMIC_REAL = (ARM_CLI_ATOMIC_CMI_KEEP, PATH_FIXED, CUT_FROZEN)
ATOMIC_SHIPPED = (ARM_CLI_ATOMIC_CMI_KEEP, PATH_FIXED_KL, CUT_SHIPPED)
ATOMIC_REAL_SHIPPED = (ARM_CLI_ATOMIC_CMI_KEEP, PATH_FIXED, CUT_SHIPPED)
LANDED_CORE = (ARM_CORE, PATH_FIXED_KL, CUT_FROZEN)
LANDED_ATOMIC = (ARM_CLI_ATOMIC, PATH_FIXED_KL, CUT_FROZEN)
LANDED_ATOMIC_SHIPPED = (ARM_CLI_ATOMIC, PATH_FIXED_KL, CUT_SHIPPED)
FAMILY_CELLS = (
    IG_OVER,
    CMI_OVER,
    IG_KEEP,
    CMI_KEEP,
    ATOMIC,
    ATOMIC_REAL,
    ATOMIC_SHIPPED,
    ATOMIC_REAL_SHIPPED,
)
REFERENCE_CELLS = (LANDED_CORE, LANDED_ATOMIC, LANDED_ATOMIC_SHIPPED)
# the registered pairs (a − b), by the hypothesis or reading they serve
PAIRS = {
    "H-ctx-cmi-f1": ((CMI_OVER, IG_OVER), (CMI_KEEP, IG_KEEP)),
    "keep-vs-overwrite": ((IG_KEEP, IG_OVER), (CMI_KEEP, CMI_OVER)),
    "identity-i": ((IG_OVER, LANDED_CORE),),
    "H-atomic-cmi-f1": ((ATOMIC, LANDED_ATOMIC),),
    "H-atomic-cmi-shipped": ((ATOMIC_SHIPPED, LANDED_ATOMIC_SHIPPED),),
    "atomic-fixed-vs-fixed-kl": ((ATOMIC_REAL, ATOMIC),),
}


def _triple(t):
    return "/".join(t)


# --- freezes and test-side tables -------------------------------------------------------------------
def load_freezes(paths, family, rung, variant):
    """`{seed: freeze}`; every file must be of one rung, variant and family."""
    out = {}
    for p in paths:
        fz = read_json(p)
        if (fz.get("rung"), fz.get("variant")) != (rung, variant):
            raise ReportRefusal(
                f"{p}: freeze of {fz.get('rung')}/{fz.get('variant')}, not {rung}/{variant}"
            )
        if fz.get("family", "model") != family:
            raise ReportRefusal(f"{p}: family {fz.get('family')!r}, expected {family!r}")
        seed = int(fz["seed"])
        if seed in out:
            raise ReportRefusal(f"two {family} freezes for seed {seed}")
        out[seed] = {"freeze": fz, "path": str(p), "sha256": sha256_file(p)}
    return out


def load_test_tables(paths):
    """`{(seed, arm, path, grain, c, N): [(tau, f1)]}` from scoresweep tables over test reads."""
    out = {}
    for p in paths:
        t = read_json(p)
        seed = t.get("seed")
        if seed is None:
            raise ReportRefusal(f"{p}: the table carries no corpus seed")
        for r in t["cells"]:
            if r["cut"] != CUT_FROZEN:
                continue
            key = (int(seed), r["arm"], r["path"], r["grain"], int(r["c"]), int(r["N"]))
            out.setdefault(key, []).append((float(r["tau"]), float(r["directed"]["f1"])))
    return out


def test_argmax_tau(rows):
    """The argmax of test directed F1 over τ, ties to the larger τ."""
    return max(rows, key=lambda tf: (tf[1], tf[0]))[0] if rows else None


def tau_drift_of(frozen, cells_of_key, seed, test_rows):
    """Per seed: the frozen (c, N, τ), the validation F1 at the freeze, the test F1 at the frozen τ,
    their difference, and the test-side argmax τ with `log10(τ*_test / τ_frozen)` when available."""
    tau = frozen.get("tau")
    test_f1 = (cells_of_key.get(seed) or {}).get("flat", {}).get("directed.f1")
    val_f1 = frozen.get("val_directed_f1")
    tau_test = test_argmax_tau(test_rows) if test_rows is not None else None
    drift = None
    if tau_test is not None and tau and tau > 0 and tau_test > 0:
        drift = math.log10(tau_test / tau)
    return {
        "c": frozen.get("c"),
        "N": frozen.get("N"),
        "tau_frozen": tau,
        "val_f1_at_freeze": val_f1,
        "test_f1_at_frozen_tau": test_f1,
        "f1_drift": (test_f1 - val_f1) if (test_f1 is not None and val_f1 is not None) else None,
        "tau_test": tau_test,
        "tau_drift_log10": drift,
    }


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def drift_table(cells, freezes, triples, grain, test_tables):
    out = {}
    for t in triples:
        key = cell_key(*t, grain)
        per_seed = {}
        for seed, fzrec in sorted(freezes.items()):
            frozen = fzrec["freeze"]["cells"].get(key)
            if frozen is None or frozen.get("tau") is None:
                continue
            rows = test_tables.get((seed, t[0], t[1], grain, int(frozen["c"]), int(frozen["N"])))
            per_seed[seed] = tau_drift_of(frozen, cells.get(key, {}), seed, rows)
        if per_seed:
            out[key] = {
                "per_seed": per_seed,
                "mean_f1_drift": _mean([v["f1_drift"] for v in per_seed.values()]),
                "mean_abs_f1_drift": _mean(
                    [abs(v["f1_drift"]) for v in per_seed.values() if v["f1_drift"] is not None]
                ),
                "mean_abs_tau_drift_log10": _mean(
                    [
                        abs(v["tau_drift_log10"])
                        for v in per_seed.values()
                        if v["tau_drift_log10"] is not None
                    ]
                ),
            }
    return out


# --- pairs and verdicts -----------------------------------------------------------------------------
def pair_tables(cells, grain):
    out = {}
    for name, pairs in PAIRS.items():
        for a, b in pairs:
            t = paired_table(cells, a, b, grain)
            if t is not None:
                out[f"{name}:{_triple(a)}:{_triple(b)}"] = t
    return out


def perseq_pair(cells, a, b, grain):
    """The per-sequence pooled directed F1 difference a − b per seed, with the paired test."""
    ka, kb = cell_key(*a, grain), cell_key(*b, grain)
    if ka not in cells or kb not in cells:
        return None
    diffs = {}
    for s in sorted(set(cells[ka]) & set(cells[kb])):
        sa, sb = cells[ka][s].get("seqscore"), cells[kb][s].get("seqscore")
        if sa is None or sb is None:
            continue
        va, vb = _get(sa, PERSEQ_KEY), _get(sb, PERSEQ_KEY)
        if va is None or vb is None:
            continue
        diffs[s] = float(va) - float(vb)
    if not diffs:
        return None
    return {
        "pair": f"{ka} - {kb}",
        "metric": PERSEQ_KEY,
        "seeds": sorted(diffs),
        "mean_diff": sum(diffs.values()) / len(diffs),
        "test": paired_test(list(diffs.values())),
    }


def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _f1_pair(pairs, name, a, b):
    t = pairs.get(f"{name}:{_triple(a)}:{_triple(b)}")
    if t is None:
        return None
    test = t["tests"].get("directed.f1", {})
    return {"mean_diff": test.get("mean_diff"), "pvalue": test.get("pvalue")}


def _significantly(d, p, below=None, beyond=None):
    """`below`: d < −band with p < α; `beyond`: |d| > band with p < α."""
    if d is None or p is None:
        return None
    if below is not None:
        return d < -below and p < ALPHA
    return abs(d) > beyond and p < ALPHA


def _status(fails, computed):
    if not computed:
        return "not computed"
    return "fail" if any(fails) else "pass"


def verdicts(pairs, perseq, drift, cells, grain):
    out = {}
    # H-ctx-cmi-f1: |CMI − IG| ≤ 0.01 at matched context; fail: |Δ| > 0.01 with p < 0.05
    rows, fails = [], []
    for a, b in PAIRS["H-ctx-cmi-f1"]:
        r = _f1_pair(pairs, "H-ctx-cmi-f1", a, b)
        if r:
            r["fail"] = _significantly(r["mean_diff"], r["pvalue"], beyond=F1_BAND)
            rows.append({"pair": f"{_triple(a)} - {_triple(b)}", **r})
            fails.append(bool(r["fail"]))
    out["H-ctx-cmi-f1"] = {"status": _status(fails, rows), "rows": rows}
    # H-ctx-keep-perseq (session): keep ≥ overwrite on pooled per-sequence F1; fail: below by > 0.01, p < 0.05
    rows, fails = [], []
    if grain == "session":
        for a, b in PAIRS["keep-vs-overwrite"]:
            r = perseq_pair(cells, a, b, grain)
            if r:
                r["fail"] = _significantly(r["mean_diff"], r["test"].get("pvalue"), below=F1_BAND)
                rows.append(r)
                fails.append(bool(r["fail"]))
        out["H-ctx-keep-perseq"] = {"status": _status(fails, rows), "rows": rows}
    else:
        out["H-ctx-keep-perseq"] = {"status": "not applicable (session grain only)", "rows": []}
    # H-atomic-cmi-f1: atomic CMI keep ≥ landed cli-atomic; fail: below by > 0.01, p < 0.05
    r = _f1_pair(pairs, "H-atomic-cmi-f1", ATOMIC, LANDED_ATOMIC)
    if r:
        r["fail"] = _significantly(r["mean_diff"], r["pvalue"], below=F1_BAND)
    out["H-atomic-cmi-f1"] = {"status": _status([bool(r["fail"])] if r else [], bool(r)), "row": r}
    # H-atomic-cmi-shipped: non-empty on every seed and ≥ the landed shipped cut
    key = cell_key(*ATOMIC_SHIPPED, grain)
    empties = {s: v["annotate"].get("empty_prediction") for s, v in cells.get(key, {}).items()}
    r = _f1_pair(pairs, "H-atomic-cmi-shipped", ATOMIC_SHIPPED, LANDED_ATOMIC_SHIPPED)
    fails = []
    if empties:
        fails.append(any(e is not None for e in empties.values()))
    if r:
        r["fail"] = _significantly(r["mean_diff"], r["pvalue"], below=F1_BAND)
        fails.append(bool(r["fail"]))
    out["H-atomic-cmi-shipped"] = {
        "status": _status(fails, bool(empties) or bool(r)),
        "empty_prediction_per_seed": empties,
        "row": r,
    }

    # the τ-drift hypotheses: CMI's mean |τ drift| ≤ IG's; fail: exceeds by > one grid step; with no
    # test-side argmax (l / xl), the F1 drift only: fail when CMI's mean |F1 drift| exceeds IG's by > 0.01
    def _tau_verdict(name, cmi_t, ig_t):
        kc, ki = cell_key(*cmi_t, grain), cell_key(*ig_t, grain)
        dc_, di = drift.get(kc), drift.get(ki)
        if not dc_ or not di:
            return {"status": "not computed", "cmi": kc, "reference": ki}
        tc, ti = dc_["mean_abs_tau_drift_log10"], di["mean_abs_tau_drift_log10"]
        fc, fi = dc_["mean_abs_f1_drift"], di["mean_abs_f1_drift"]
        if tc is not None and ti is not None:
            fail = tc > ti + TAU_STEP_DECADES
            basis = "tau"
        elif fc is not None and fi is not None:
            fail = fc > fi + F1_BAND
            basis = "f1"
        else:
            return {"status": "not computed", "cmi": kc, "reference": ki}
        return {
            "status": "fail" if fail else "pass",
            "basis": basis,
            "cmi": {"cell": kc, "mean_abs_tau_drift_log10": tc, "mean_abs_f1_drift": fc},
            "reference": {"cell": ki, "mean_abs_tau_drift_log10": ti, "mean_abs_f1_drift": fi},
        }

    out["H-ctx-cmi-tau"] = {
        "overwrite": _tau_verdict("H-ctx-cmi-tau", CMI_OVER, IG_OVER),
        "keep": _tau_verdict("H-ctx-cmi-tau", CMI_KEEP, IG_KEEP),
    }
    out["H-atomic-cmi-tau"] = _tau_verdict("H-atomic-cmi-tau", ATOMIC, LANDED_ATOMIC)
    return out


# --- the document -----------------------------------------------------------------------------------
def build(cells, context_freezes, landed_freezes, test_tables, rung, variant):
    grains = {}
    for grain in GRAINS:
        tables = {}
        for t in FAMILY_CELLS + REFERENCE_CELLS:
            key = cell_key(*t, grain)
            if key in cells:
                tables[key] = cell_table(cells[key], key)
        if not tables:
            continue
        pairs = pair_tables(cells, grain)
        drift = drift_table(cells, context_freezes, FAMILY_CELLS, grain, test_tables)
        drift.update(drift_table(cells, landed_freezes, REFERENCE_CELLS, grain, test_tables))
        perseq = {
            f"{_triple(a)}:{_triple(b)}": r
            for a, b in PAIRS["keep-vs-overwrite"]
            if (r := perseq_pair(cells, a, b, grain))
        }
        grains[grain] = {
            "cells": tables,
            "pairs": pairs,
            "perseq_pairs": perseq,
            "tau_drift": drift,
            "verdicts": verdicts(pairs, perseq, drift, cells, grain),
        }
    if not grains:
        raise ReportRefusal("no context-family cell found under the results directories")
    return {
        "schema": TABLE_SCHEMA,
        "rung": rung,
        "variant": variant,
        "family": FAMILY_CONTEXT,
        "context_freezes": {
            s: {"path": v["path"], "sha256": v["sha256"]}
            for s, v in sorted(context_freezes.items())
        },
        "landed_freezes": {
            s: {"path": v["path"], "sha256": v["sha256"]} for s, v in sorted(landed_freezes.items())
        },
        "n_test_tables": len({k[0] for k in test_tables}),
        "grains": grains,
    }


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def render_md(doc):
    lines = [
        f"# Context family — {doc['rung']} / {doc['variant']} (schema {doc['schema']})",
        "",
        "Five-seed means of test directed F1 at the floor 0.05; pairs are per-seed differences with the "
        "paired t-test; τ drift = log10(τ*_test / τ_frozen) where a test-side scoresweep was given, "
        "else the F1 drift test − validation at the frozen τ. Verdicts follow the registered fail clauses.",
        "",
    ]
    for grain, g in doc["grains"].items():
        lines += [
            f"## {grain}",
            "",
            "| cell | F1 mean | F1 sd | reachable F1 | mean F1 drift | mean |τ drift| |",
            "|---|---|---|---|---|---|",
        ]
        for key, t in g["cells"].items():
            m = t["metrics"].get("directed.f1", {})
            rf = t["metrics"].get("reachable.f1", {})
            d = g["tau_drift"].get(key, {})
            lines.append(
                f"| `{key}` | {_fmt(m.get('mean'))} | {_fmt(m.get('std'))} | {_fmt(rf.get('mean'))} | "
                f"{_fmt(d.get('mean_f1_drift'))} | {_fmt(d.get('mean_abs_tau_drift_log10'), 3)} |"
            )
        lines += ["", "| pair (a − b) | Δ directed F1 | p | Δ reachable F1 |", "|---|---|---|---|"]
        for name, t in g["pairs"].items():
            f1 = t["tests"].get("directed.f1", {})
            rf = t["metrics"].get("reachable.f1", {})
            lines.append(
                f"| `{name}` | {_fmt(f1.get('mean_diff'))} | {_fmt(f1.get('pvalue'), 3)} | {_fmt(rf.get('mean'))} |"
            )
        for name, r in g["perseq_pairs"].items():
            lines.append(
                f"| per-sequence `{name}` | {_fmt(r['mean_diff'])} | {_fmt(r['test'].get('pvalue'), 3)} | — |"
            )
        lines += ["", "| hypothesis | verdict |", "|---|---|"]
        for h, v in g["verdicts"].items():
            status = (
                v.get("status")
                if isinstance(v, dict) and "status" in v
                else "; ".join(f"{k}: {x.get('status')}" for k, x in v.items())
            )
            lines.append(f"| {h} | {status} |")
        lines.append("")
    return "\n".join(lines)


def run(args, rec):
    cells = load_cells(args.landed_results, args.rung, args.variant)
    ctx = load_cells(args.context_results, args.rung, args.variant)
    overlap = set(cells) & set(ctx)
    if overlap:
        raise ReportRefusal(f"cells present in both results trees: {sorted(overlap)[:3]} …")
    cells.update(ctx)
    for key in cells:
        parse_cell_key(key)
    context_freezes = load_freezes(args.context_freezes, FAMILY_CONTEXT, args.rung, args.variant)
    landed_freezes = load_freezes(args.landed_freezes, "model", args.rung, args.variant)
    test_tables = load_test_tables(args.test_tables)
    doc = build(cells, context_freezes, landed_freezes, test_tables, args.rung, args.variant)
    out = Path(args.output_folder)
    write_json(out / "context-table.json", doc)
    (out / "context-table.md").write_text(render_md(doc), encoding="utf-8")
    log({"event": "context_table_written", "rung": args.rung, "grains": list(doc["grains"])})
    return {
        "table": "context-table.json",
        "grains": list(doc["grains"]),
        "verdicts": {
            g: {
                h: (
                    v.get("status") if "status" in v else {k: x.get("status") for k, x in v.items()}
                )
                for h, v in d["verdicts"].items()
            }
            for g, d in doc["grains"].items()
        },
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--landed-results", required=True, help="the landed results tree (`results/`)")
    p.add_argument(
        "--context-results", required=True, help="the family's tree (`results-context/`)"
    )
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument(
        "--context-freezes", required=True, nargs="+", help="the family's freezes, one per seed"
    )
    p.add_argument(
        "--landed-freezes", required=True, nargs="+", help="the landed model freezes, one per seed"
    )
    p.add_argument(
        "--test-tables",
        required=True,
        nargs="*",
        help="scoresweep tables over the test reads' score files (the test-side argmax τ); may be empty",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "contexttable", vars(args)) as rec:
        res = run(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
