"""The floors' per-sequence tables (`plans/floors.md`, addendum 2026-10-21, §6): per rung and
grain, every landed floor row beside the reference arm row it was broadcast onto, five-seed
headlines and the paired five-seed difference `floor row − reference arm` with the paired t-test.

    python -m seq2causebench.floorseqtable --results-dir results --rung xs --variant latent \
        --reference-cells trace/core/shipped/frozen trace/cli-atomic/shipped/frozen \
        --output-folder tables/xs-latent-perseq-floors

Reads `results/<rung>/<variant>/seed=<k>/floor/<arm>/none/frozen/<grain>/seqscore-<row>-on-<ref>/
seqscore.json` and the reference cell's `…/<grain>/seqscore/seqscore.json` of the same seed. Its
own document: never beside a type-level number, never merged with `report`'s per-sequence tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .arms import parse_cell_key
from .constants import FLOOR_ARMS, GRAINS, PATH_NONE, RUNGS, SEQSCORE_JSON, VARIANTS
from .floorseq import IDENTITY_KEYS, ROWS_OF_FLOOR, landing_dir_name
from .log import log
from .record import RunRecord, read_json, write_json
from .report import (
    MIN_SEEDS,
    PERSEQ_METRICS,
    ReportRefusal,
    _fmt,
    _fmt_test,
    _get,
    _headline,
    paired_test,
)

TABLE_SCHEMA = "seq2causebench/report-perseq-floors@1"
PAIRED_METRICS = ("pooled.directed.f1", "pooled.auroc")


def load_rows(results_dir, rung, variant, reference_triples):
    """`{(grain, reference, floor cell, row): {seed: (floor seqscore, reference seqscore)}}`."""
    root = Path(results_dir) / rung / variant
    out = {}
    for seed_dir in sorted(root.glob("seed=*")):
        seed = int(seed_dir.name.split("=", 1)[1])
        for grain in GRAINS:
            for ref in reference_triples:
                ref_path = seed_dir / ref / grain / "seqscore" / SEQSCORE_JSON
                for arm in FLOOR_ARMS:
                    for row in ROWS_OF_FLOOR[arm]:
                        cell = f"{arm}/{PATH_NONE}/frozen/{grain}"
                        f = (
                            seed_dir
                            / cell
                            / landing_dir_name(row, f"{ref}/{grain}")
                            / SEQSCORE_JSON
                        )
                        if not f.exists():
                            continue
                        if not ref_path.exists():
                            raise ReportRefusal(f"{f} has no reference seqscore at {ref_path}")
                        fl, rf = read_json(f), read_json(ref_path)
                        if (
                            fl["reference_cell"] != f"{ref}/{grain}"
                            or fl["corpus_id"] != rf["corpus_id"]
                        ):
                            raise ReportRefusal(f"{f} is not paired with {ref_path}")
                        diff = [k for k in IDENTITY_KEYS if _get(fl, k) != _get(rf, k)]
                        if diff:
                            raise ReportRefusal(f"{f} and {ref_path} differ in {diff}")
                        out.setdefault((grain, ref, cell, row), {})[seed] = (fl, rf)
    return out


def build_tables(rows):
    tables = []
    for (grain, ref, cell, row), per_seed in sorted(rows.items()):
        seeds = sorted(per_seed)
        if len(seeds) < MIN_SEEDS:
            raise ReportRefusal(
                f"{cell} {row} on {ref} has {len(seeds)} seed(s); a headline needs {MIN_SEEDS}"
            )
        fl_rows = [{"flat": {m: _get(per_seed[s][0], m) for m in PERSEQ_METRICS}} for s in seeds]
        rf_rows = [{"flat": {m: _get(per_seed[s][1], m) for m in PERSEQ_METRICS}} for s in seeds]
        tests = {}
        for m in PAIRED_METRICS:
            a = [_get(per_seed[s][0], m) for s in seeds]
            b = [_get(per_seed[s][1], m) for s in seeds]
            tests[m] = (
                paired_test([x - y for x, y in zip(a, b, strict=True)])
                if all(v is not None for v in a + b)
                else {"test": "paired t-test", "reason": "value absent on at least one seed"}
            )
        tables.append(
            {
                "grain": grain,
                "reference_cell": f"{ref}/{grain}",
                "floor_cell": cell,
                "row": row,
                "seeds": seeds,
                "tau": [per_seed[s][0]["tau"] for s in seeds],
                "floor": {m: _headline(fl_rows, m) for m in PERSEQ_METRICS},
                "reference": {m: _headline(rf_rows, m) for m in PERSEQ_METRICS},
                "paired": tests,
                "above_predict_all": [
                    per_seed[s][0]["pooled"]["directed"]["f1"]
                    > per_seed[s][0]["predict_all"]["pooled"]
                    for s in seeds
                ],
            }
        )
    return tables


def render_markdown(doc):
    lines = [
        f"# {doc['rung']} / {doc['variant']} — the floors on the per-sequence axis "
        "(its own tables; never merged with type-level numbers)",
        "",
        "Each floor row is broadcast onto the sequences and candidate pairs of the reference arm's "
        "landed test read (`plans/floors.md`, addendum 2026-10-21); the predict-all value and the "
        "scoreable fraction are therefore the reference row's. Paired difference = floor − reference, "
        "five seeds, paired t-test.",
        "",
    ]
    show = (
        "pooled.directed.f1",
        "pooled.directed.precision",
        "pooled.directed.recall",
        "pooled.auroc",
    )
    for grain in GRAINS:
        rows = [t for t in doc["tables"] if t["grain"] == grain]
        if not rows:
            continue
        lines += [
            f"## {grain} grain",
            "",
            "| reference read | row | "
            + " | ".join(show)
            + " | predict-all | Δ F1 (p) | Δ AUROC (p) | > predict-all |",
            "|---|---|" + "---|" * (len(show) + 4),
        ]
        refs = sorted({t["reference_cell"] for t in rows})
        for ref in refs:
            sub = [t for t in rows if t["reference_cell"] == ref]
            r0 = sub[0]["reference"]
            lines.append(
                f"| {ref} | *the arm itself* | "
                + " | ".join(_fmt(r0[m]) for m in show)
                + f" | {_fmt(r0['predict_all.pooled'])} | — | — | — |"
            )
            for t in sub:
                hits = sum(t["above_predict_all"])
                lines.append(
                    f"| {ref} | `{t['floor_cell'].rsplit('/', 3)[0]}` {t['row']} | "
                    + " | ".join(_fmt(t["floor"][m]) for m in show)
                    + f" | {_fmt(t['floor']['predict_all.pooled'])}"
                    + f" | {_fmt_test(t['paired']['pooled.directed.f1'])}"
                    + f" | {_fmt_test(t['paired']['pooled.auroc'])}"
                    + f" | {hits}/{len(t['seeds'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def run_table(args, rec):
    refs = list(args.reference_cells)
    for r in refs:
        parse_cell_key(f"{r}/request")  # an <arm>/<path>/<cut> triple of the registry
    rows = load_rows(args.results_dir, args.rung, args.variant, refs)
    if not rows:
        raise ReportRefusal(
            f"no landed floor per-sequence row under {args.results_dir}/{args.rung}"
        )
    doc = {
        "schema": TABLE_SCHEMA,
        "rung": args.rung,
        "variant": args.variant,
        "reference_cells": refs,
        "tables": build_tables(rows),
    }
    stem = f"{args.rung}-{args.variant}-perseq-floors"
    write_json(rec.out_dir / f"{stem}.json", doc)
    (rec.out_dir / f"{stem}.md").write_text(render_markdown(doc) + "\n")
    log({"event": "floorseqtable_done", "rung": args.rung, "n_rows": len(doc["tables"])})
    return {"tables": f"{stem}.json", "markdown": f"{stem}.md", "n_rows": len(doc["tables"])}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True)
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--reference-cells", required=True, nargs="+", help="<arm>/<path>/<cut> triples")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "floorseqtable", vars(args)) as rec:
        res = run_table(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
