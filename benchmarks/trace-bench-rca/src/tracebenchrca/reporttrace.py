"""Pool the landed per-trace-axis results per rung and across rungs, and score the axis's hypotheses.

    python -m tracebenchrca.reporttrace --results-dir results --rungs xs s m --seeds 0 1 2 3 4 \
        --arms rca/trace rca/trace-core --bootstrap-b 2000 --bootstrap-seed 0 --output-folder tables/rca-trace

`plans/rca-arms.md`, addendum 2026-10-31. Reads, per rung, seed and TRACE arm, the landed
`results/<rung>/latent/seed=<k>/<arm slug>/trace/score-trace.json` and the two stage records beside it
(`run/`, `rank/run/`), refusing a rung with fewer seeds than named (five seeds or nothing) and a file that
does not hash to its record. The floor, control and broadcast rows ride every TRACE arm's cell as copies; they
are read from the first named arm and the other arms' copies must agree on every field outside the window
coordinates (`WINDOW_FIELDS`: the cell's context length c, the position hits, the traces skipped short and the
root-cause positions inside the context all follow the arm's c, which differs between the arms at `xs` and `s`), so
a shared row's position metrics are those of the first named arm's window.
Writes per row one pooled table per rung and one across the named rungs (`tracebench.score_rca_trace.pool`:
the case rows concatenated — the cluster is the (corpus, case) pair — with the same bootstrap),
`summary.json` (every row with its interval and random expectation, the counts, the hypotheses) and
`rca-trace.md`.

Hypotheses (mechanical): **H-trace-perseq-random** — on every rung the hit@1 interval of each trace-reading
row at reading hop1 lies above the per-trace random expectation; **H-trace-perseq-topology** — on every rung
the floor's hit@1 >= each TRACE row's at the same reading; **H-trace-perseq-broadcast** — on every rung each
TRACE row's hit@1 (hop1) >= every broadcast row's; **H-trace-perseq-self** — the hop0 reading changes the
sign of (TRACE − the best broadcast row) on no rung.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.score_rca_trace import pool

from .arms import arm_slug
from .constants import (
    ARM_CONTROL_TRACE_CALLEES,
    ARM_CONTROL_TRACE_RANDOM,
    ARM_FLOOR_TRACE_TOPOLOGY,
    ARM_TRACE,
    ARM_TRACE_CORE,
    RESULTS_JSON,
    RUN_DIR,
    RUNGS,
    SCORE_TRACE_JSON,
    SEEDS,
    STAGE_RANK_TRACE,
    STAGE_SCORE_TRACE,
    TRACE_AXIS_DIR,
    TRACE_READINGS,
    VARIANT,
)
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json

STAGE = "report-trace"
HOP1, HOP0 = TRACE_READINGS
TRACE_ARMS = (ARM_TRACE, ARM_TRACE_CORE)
S1 = "s@1"
METRIC_KEYS = ("s@1", "s@3", "p@1", "p@3", "exact", "mean_set_size")
SHARED_PREFIXES = (ARM_FLOOR_TRACE_TOPOLOGY, ARM_CONTROL_TRACE_RANDOM, ARM_CONTROL_TRACE_CALLEES, "broadcast/")
# the fields of a score row in the window's coordinates (a position p = i − c + 1), which follow the cell's c
WINDOW_FIELDS = frozenset({"c", "p@1", "p@3", "n_skipped", "n_rc_unreachable_context"})


def cell_dir(results_dir, rung, seed, arm):
    return Path(results_dir) / rung / VARIANT / f"seed={seed}" / arm_slug(arm) / TRACE_AXIS_DIR


def load_cell(results_dir, rung, seed, arm):
    d = cell_dir(results_dir, rung, seed, arm)
    score_res = read_json(d / RUN_DIR / RESULTS_JSON)
    rank_res = read_json(d / "rank" / RUN_DIR / RESULTS_JSON)
    if score_res.get("status") != "ok" or rank_res.get("status") != "ok":
        raise ValueError(f"{d} holds a failed record")
    if score_res.get("stage") != STAGE_SCORE_TRACE or rank_res.get("stage") != STAGE_RANK_TRACE:
        raise ValueError(f"{d} does not hold a {STAGE_RANK_TRACE} and a {STAGE_SCORE_TRACE} record")
    if score_res["arm"] != arm or score_res["rung"] != rung or rank_res["arm"] != arm:
        raise ValueError(f"{d} holds an {score_res['arm']!r} rung {score_res['rung']!r} result")
    if sha256_file(d / SCORE_TRACE_JSON) != score_res["score_trace_sha256"]:
        raise ValueError(f"{d / SCORE_TRACE_JSON} differs from the file its record hashed")
    if sha256_file(d / "rank" / RUN_DIR / RESULTS_JSON) != score_res["rank_record_sha256"]:
        raise ValueError(f"{d}: the rank record is not the one the score record names")
    return read_json(d / SCORE_TRACE_JSON)["rows"], score_res, rank_res


def is_shared(row_id):
    return any(row_id.startswith(p) for p in SHARED_PREFIXES)


def without_window_fields(x):
    """A score row (or any part of one) with every `WINDOW_FIELDS` key dropped, at any depth."""
    if isinstance(x, dict):
        return {k: without_window_fields(v) for k, v in x.items() if k not in WINDOW_FIELDS}
    if isinstance(x, list):
        return [without_window_fields(v) for v in x]
    return x


def check_shared_rows_agree(cells_by_arm):
    """The floor, control and broadcast rows must be identical across the TRACE arms' cells of one corpus outside
    the window coordinates."""
    arms = list(cells_by_arm)
    first = cells_by_arm[arms[0]][0]
    for arm in arms[1:]:
        rows = cells_by_arm[arm][0]
        for row_id, r in first.items():
            if not is_shared(row_id):
                continue
            if row_id not in rows or without_window_fields(rows[row_id]) != without_window_fields(r):
                raise ValueError(f"row {row_id!r} differs between the {arms[0]!r} and {arm!r} cells")


def row_of(rung, row_id, pooled, cells):
    counts = {}
    for rows, _, _ in cells:
        for k, v in rows[row_id]["counts"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                counts[k] = counts.get(k, 0) + v
    counts["mean_n_services"] = sum(rows[row_id]["counts"]["mean_n_services"] for rows, _, _ in cells) / len(cells)
    return {
        "rung": rung,
        "row": row_id,
        "mode": pooled["mode"],
        "hop0": pooled["hop0"],
        "metrics": pooled["metrics"],
        "bootstrap": pooled["bootstrap"],
        "random": pooled["random"],
        "by_fault_kind": pooled["by_fault_kind"],
        "counts": counts,
        "n_seeds": len(cells),
    }


def _low(row, key=S1):
    return row["bootstrap"]["intervals"][key]["low"]


def hypotheses(rows, rungs, arms):
    by = {(r["rung"], r["row"]): r for r in rows}
    out = {}
    trace_reading = [f"{a}/{HOP1}" for a in arms] + [f"{ARM_FLOOR_TRACE_TOPOLOGY}/{HOP1}"]
    per = {}
    for row_id in trace_reading:
        for g in rungs:
            r = by.get((g, row_id))
            if r is None:
                continue
            per[f"{g} {row_id}"] = _low(r) > r["random"][S1]
    out["H-trace-perseq-random"] = {"per_rung_and_row": per, "verdict": "pass" if per and all(per.values()) else "fail"}
    per, gaps = {}, {}
    for a in arms:
        for reading in TRACE_READINGS:
            for g in rungs:
                arm_row, floor = by.get((g, f"{a}/{reading}")), by.get((g, f"{ARM_FLOOR_TRACE_TOPOLOGY}/{reading}"))
                if arm_row is None or floor is None:
                    continue
                key = f"{g} {a}/{reading}"
                gaps[key] = arm_row["metrics"][S1] - floor["metrics"][S1]
                per[key] = floor["metrics"][S1] >= arm_row["metrics"][S1]
    out["H-trace-perseq-topology"] = {
        "gap_arm_minus_floor_hit1": gaps,
        "per_rung_and_row": per,
        "verdict": "pass" if per and all(per.values()) else "fail",
    }
    broadcasts = sorted({r["row"] for r in rows if r["row"].startswith("broadcast/")})
    per, best = {}, {}
    for a in arms:
        for g in rungs:
            arm_row = by.get((g, f"{a}/{HOP1}"))
            bc = [by[(g, b)] for b in broadcasts if (g, b) in by]
            if arm_row is None or not bc:
                continue
            top = max(bc, key=lambda r: r["metrics"][S1])
            best[f"{g} {a}"] = {"row": top["row"], S1: top["metrics"][S1]}
            per[f"{g} {a}"] = all(arm_row["metrics"][S1] >= r["metrics"][S1] for r in bc)
    out["H-trace-perseq-broadcast"] = {
        "best_broadcast": best,
        "per_rung_and_arm": per,
        "verdict": "pass" if per and all(per.values()) else "fail",
    }
    per, signs = {}, {}
    for a in arms:
        for g in rungs:
            h1, h0 = by.get((g, f"{a}/{HOP1}")), by.get((g, f"{a}/{HOP0}"))
            bc = [by[(g, b)] for b in broadcasts if (g, b) in by]
            if h1 is None or h0 is None or not bc:
                continue
            top = max(r["metrics"][S1] for r in bc)
            s1, s0 = _sign(h1["metrics"][S1] - top), _sign(h0["metrics"][S1] - top)
            signs[f"{g} {a}"] = {"hop1": s1, "hop0": s0}
            per[f"{g} {a}"] = s1 == s0
    out["H-trace-perseq-self"] = {
        "sign_arm_minus_best_broadcast": signs,
        "per_rung_and_arm": per,
        "verdict": "pass" if per and all(per.values()) else "fail",
    }
    return out


def _sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def markdown(rows, hyp, rungs):
    lines = ["# Per-trace root-cause axis (pooled; B and seed in summary.json)", ""]
    row_ids = []
    for r in rows:
        if r["row"] not in row_ids:
            row_ids.append(r["row"])
    header = "| row | " + " | ".join(f"`{g}` hit@1 [CI] / hit@3 / exact (set) / random@1" for g in rungs) + " |"
    lines += [header, "|---|" + "---|" * len(rungs)]
    by = {(r["rung"], r["row"]): r for r in rows}
    for row_id in row_ids:
        cells = []
        for g in rungs:
            r = by.get((g, row_id))
            if r is None:
                cells.append("—")
                continue
            m, iv = r["metrics"], r["bootstrap"]["intervals"].get(S1)
            ci = f" [{fmt(iv['low'])}, {fmt(iv['high'])}]" if iv else ""
            exact = "n/a" if m.get("exact") is None else f"{fmt(m['exact'])} ({fmt(m.get('mean_set_size'))})"
            cells.append(f"{fmt(m[S1])}{ci} / {fmt(m['s@3'])} / {exact} / {fmt(r['random'][S1])}")
        lines.append(f"| `{row_id}` | " + " | ".join(cells) + " |")
    lines += ["", "## Hypotheses", ""]
    for name, h in hyp.items():
        lines.append(f"- **{name}**: {h['verdict']}")
    return "\n".join(lines) + "\n"


def report(results_dir, rungs, seeds, arms, b, seed, out):
    out = Path(out)
    rows, files = [], {}
    per_row_all = {}
    for rung in rungs:
        cells_by_arm = {arm: [load_cell(results_dir, rung, k, arm) for k in seeds] for arm in arms}
        for arm, cells in cells_by_arm.items():
            if len(cells) != len(seeds):
                raise ValueError(f"rung {rung!r} arm {arm!r}: {len(cells)} seeds, {len(seeds)} named")
        for k in range(len(seeds)):
            check_shared_rows_agree({arm: (cells_by_arm[arm][k][0], None, None) for arm in arms})
        first = arms[0]
        row_ids = [rid for rid in cells_by_arm[first][0][0] if is_shared(rid)]
        for arm in arms:
            row_ids += [rid for rid in cells_by_arm[arm][0][0] if rid.startswith(arm + "/")]
        for row_id in row_ids:
            source_arm = first if is_shared(row_id) else row_id.rsplit("/", 1)[0]
            cells = cells_by_arm[source_arm]
            results = [rows_[row_id] for rows_, _, _ in cells]
            pooled = pool(results, b, seed)
            name = f"pooled-{rung}-{row_id.replace('/', '-')}.json"
            write_json(out / name, pooled)
            files[name] = sha256_file(out / name)
            rows.append(row_of(rung, row_id, pooled, cells))
            per_row_all.setdefault(row_id, []).extend(results)
    if len(rungs) > 1:
        for row_id, results in per_row_all.items():
            name = f"pooled-{'-'.join(rungs)}-{row_id.replace('/', '-')}.json"
            write_json(out / name, pool(results, b, seed))
            files[name] = sha256_file(out / name)
    hyp = hypotheses(rows, rungs, arms)
    write_json(out / "summary.json", {"rows": rows, "hypotheses": hyp})
    (out / "rca-trace.md").write_text(markdown(rows, hyp, rungs), encoding="utf-8")
    files["summary.json"] = sha256_file(out / "summary.json")
    files["rca-trace.md"] = sha256_file(out / "rca-trace.md")
    return {
        "stage": STAGE,
        "rungs": list(rungs),
        "seeds": list(seeds),
        "arms": list(arms),
        "rows": sorted({r["row"] for r in rows}),
        "files": files,
        "hypotheses": {k: v["verdict"] for k, v in hyp.items()},
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--rungs", required=True, nargs="+", choices=RUNGS)
    p.add_argument("--seeds", required=True, nargs="+", type=int, choices=SEEDS)
    p.add_argument("--arms", required=True, nargs="+", choices=TRACE_ARMS)
    p.add_argument("--bootstrap-b", required=True, type=int, help="bootstrap resamples of the pooled tables")
    p.add_argument("--bootstrap-seed", required=True, type=int)
    p.add_argument("--output-folder", required=True, help="the tables directory and this stage's run record")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, STAGE, vars(args)) as rec:
        res = report(
            args.results_dir,
            args.rungs,
            args.seeds,
            args.arms,
            args.bootstrap_b,
            args.bootstrap_seed,
            args.output_folder,
        )
        log({"event": STAGE, "rows": len(res["rows"]), **res["hypotheses"]})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
