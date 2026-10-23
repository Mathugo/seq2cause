"""The view-order table (`plans/floors.md`, addendum 2026-10-22, the view-order statistic): per
rung, grain and ordering, the five-seed mean, minimum and maximum of `share_cause_first`.

    python -m seq2causebench.viewordertable --results-dir results-vieworder --variant latent \
        --rungs xs s m l xl --output-folder tables/vieworder-latent

Reads `results-vieworder/<rung>/<variant>/seed=<k>/<ordering>-<grain>/vieworder.json`. A rung,
grain and ordering with no read is named "not run"; one with fewer than five seeds is refused
(five seeds or not reported). Every read must be on the test split with the arms' test-read
sample (the whole split at xs; 20,000 request / 10,000 session sequences from s up). The
statistic describes the view, not an arm: no pair, no test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import GRAINS, ORDERINGS, RUNGS, VARIANTS
from .log import log
from .record import RunRecord, read_json, write_json
from .report import MIN_SEEDS, ReportRefusal
from .vieworder import VIEWORDER_JSON

TABLE_SCHEMA = "seq2causebench/vieworder-table@1"
SEEDS = tuple(range(MIN_SEEDS))
TEST_SAMPLE = {"xs": {"request": 0, "session": 0}}
TEST_SAMPLE_ABOVE_XS = {"request": 20000, "session": 10000}


def expected_sample(rung, grain):
    return TEST_SAMPLE.get(rung, TEST_SAMPLE_ABOVE_XS)[grain]


def load_reads(results_dir, rungs, variant):
    """`{(rung, grain, ordering): {seed: vieworder.json}}` for every read on disk."""
    reads = {}
    for rung in rungs:
        for k in SEEDS:
            for ordering in ORDERINGS:
                for grain in GRAINS:
                    p = (
                        Path(results_dir)
                        / rung
                        / variant
                        / f"seed={k}"
                        / f"{ordering}-{grain}"
                        / VIEWORDER_JSON
                    )
                    if not p.exists():
                        continue
                    doc = read_json(p)
                    want = (
                        f"{rung}/{variant}/seed={k}",
                        ordering,
                        grain,
                        "test",
                        expected_sample(rung, grain),
                    )
                    got = (
                        doc["corpus_id"],
                        doc["ordering"],
                        doc["grain"],
                        doc["split"],
                        doc["num_sequences"],
                    )
                    if got != want:
                        raise ReportRefusal(f"{p}: read {got} where the registration asks {want}")
                    reads.setdefault((rung, grain, ordering), {})[k] = doc
    return reads


def build_table(reads, rungs):
    rows = []
    for grain in GRAINS:
        for rung in rungs:
            for ordering in ORDERINGS:
                per_seed = reads.get((rung, grain, ordering), {})
                row = {"rung": rung, "grain": grain, "ordering": ordering, "n_seeds": len(per_seed)}
                if not per_seed:
                    row["status"] = "not run: no score tier on disk for this rung"
                    rows.append(row)
                    continue
                if len(per_seed) < MIN_SEEDS:
                    raise ReportRefusal(
                        f"{rung} {ordering}-{grain}: {len(per_seed)} seeds; five seeds or not reported"
                    )
                shares = [per_seed[k]["share_cause_first"] for k in SEEDS]
                edge_means = [per_seed[k]["edge_mean_share"] for k in SEEDS]
                row.update(
                    status="read",
                    share_cause_first={
                        "mean": sum(shares) / len(shares),
                        "min": min(shares),
                        "max": max(shares),
                        "per_seed": shares,
                    },
                    edge_mean_share_mean=sum(edge_means) / len(edge_means),
                    n_sequences=[per_seed[k]["n_sequences"] for k in SEEDS],
                    n_cooccurrences=[per_seed[k]["n_cooccurrences"] for k in SEEDS],
                    n_truth_edges_cooccurring=[
                        per_seed[k]["n_truth_edges_cooccurring"] for k in SEEDS
                    ],
                    floor=sorted({per_seed[k]["floor"] for k in SEEDS}),
                )
                rows.append(row)
    return rows


def render_markdown(doc):
    out = [
        f"# {doc['variant']} — the view-order statistic (describes the view, not an arm)",
        "",
        "Share of co-occurrences of a directed truth edge (at the default floor) in which the cause token comes "
        "before the effect token in the view's row order, pooled over truth edges and sampled test sequences "
        "(`plans/floors.md`, addendum 2026-10-22). Five-seed mean [min, max]; the unweighted per-edge mean "
        "beside it. Reported beside every request-grain orientation number of both views, never gated.",
    ]
    for grain in GRAINS:
        out += [
            "",
            f"## {grain} grain",
            "",
            "| rung | ordering | share_cause_first | edge_mean_share (mean) | sequences per seed | seeds |",
            "|---|---|---|---|---|---|",
        ]
        for r in doc["rows"]:
            if r["grain"] != grain:
                continue
            if r["status"] != "read":
                out.append(f"| {r['rung']} | {r['ordering']} | {r['status']} | — | — | 0 |")
                continue
            s = r["share_cause_first"]
            n = r["n_sequences"]
            n_txt = f"{min(n)}" if min(n) == max(n) else f"{min(n)}–{max(n)}"
            out.append(
                f"| {r['rung']} | {r['ordering']} | {s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}] "
                f"| {r['edge_mean_share_mean']:.3f} | {n_txt} | {r['n_seeds']} |"
            )
    return "\n".join(out)


def run_table(args, rec):
    reads = load_reads(args.results_dir, args.rungs, args.variant)
    doc = {
        "schema": TABLE_SCHEMA,
        "variant": args.variant,
        "rungs": list(args.rungs),
        "rows": build_table(reads, args.rungs),
    }
    stem = Path(args.output_folder).name
    write_json(rec.out_dir / f"{stem}.json", doc)
    (rec.out_dir / f"{stem}.md").write_text(render_markdown(doc) + "\n")
    log({"event": "viewordertable_done", "rows": len(doc["rows"])})
    return {"table": f"{stem}.json", "n_rows": len(doc["rows"])}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--rungs", required=True, nargs="+", choices=RUNGS)
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "viewordertable", vars(args)) as rec:
        res = run_table(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
