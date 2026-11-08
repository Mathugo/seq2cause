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
sign of (TRACE − the best broadcast row) on no rung. Since 2026-11-08 (the l rung; `l_hypotheses`): **H-l-top3**,
**H-l-over-free**, **H-l-hold**, **H-l-vs-rcd**, **H-l-ig-ge-cmi**, **H-l-no-blind-kind**, **H-l-floor-gap** — per trace at
reading hop0 on the keep arms, "not run" unless l is named (H-l-hold also needs m).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tracebench.score_rca_trace import pool

from .arms import arm_slug
from .constants import (
    ARM_CONTROL_TRACE_CALLEES,
    ARM_CONTROL_TRACE_EARLIEST,
    ARM_CONTROL_TRACE_RANDOM,
    ARM_FLOOR_TRACE_TOPOLOGY,
    ARM_RCD,
    L_BLIND_KIND_FACTOR,
    L_FLOOR_GAP,
    L_HOLD_GAP,
    L_HOLD_RUNG,
    L_HYPOTHESIS_ARMS,
    L_OVER_FREE_MARGIN,
    L_RUNG,
    L_TOP3_HIT3,
    RESULTS_JSON,
    RUN_DIR,
    RUNGS,
    SCORE_TRACE_JSON,
    SEEDS,
    STAGE_RANK_TRACE,
    STAGE_SCORE_TRACE,
    TRACE_ARMS_ALL,
    TRACE_AXIS_DIR,
    TRACE_READING_EARLIEST,
    TRACE_READINGS,
    VARIANT,
)
from .log import log
from .record import RunRecord, read_json, sha256_file, write_json

STAGE = "report-trace"
HOP1, HOP0, SOURCE = TRACE_READINGS
TRACE_ARMS = TRACE_ARMS_ALL
REPORT_ARMS = TRACE_ARMS + (ARM_CONTROL_TRACE_EARLIEST,)  # the control's cell carries its own rows only
EARLIEST_ROW = f"{ARM_CONTROL_TRACE_EARLIEST}/{TRACE_READING_EARLIEST}"
S1 = "s@1"
S3 = "s@3"
RANDOM_ROW = ARM_CONTROL_TRACE_RANDOM
CALLEES_HOP0_ROW = f"{ARM_CONTROL_TRACE_CALLEES}/{HOP0}"
RCD_BROADCAST_ROW = f"broadcast/{ARM_RCD}"
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


def has_shared_rows(rows):
    return any(is_shared(rid) for rid in rows)


def check_shared_rows_agree(cells_by_arm):
    """The floor, control and broadcast rows must be identical across the TRACE arms' cells of one corpus outside
    the window coordinates (a cell scored arm-only — the earliest-error control's — carries none and is skipped)."""
    arms = [a for a in cells_by_arm if has_shared_rows(cells_by_arm[a][0])]
    if not arms:
        return
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
    arms = [a for a in arms if a in TRACE_ARMS]  # the control has rows of its own, never an arm's hypotheses
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
    # H-trace-earliest (addendum "2026-10-31, later"): each TRACE arm's hop0 hit@1 >= the earliest-error control's
    per, gaps = {}, {}
    for a in arms:
        for g in rungs:
            arm_row, ctl = by.get((g, f"{a}/{HOP0}")), by.get((g, EARLIEST_ROW))
            if arm_row is None or ctl is None:
                continue
            key = f"{g} {a}/{HOP0}"
            gaps[key] = arm_row["metrics"][S1] - ctl["metrics"][S1]
            per[key] = arm_row["metrics"][S1] >= ctl["metrics"][S1]
    out["H-trace-earliest"] = (
        {
            "gap_arm_minus_earliest_hit1": gaps,
            "per_rung_and_row": per,
            "verdict": "pass" if all(per.values()) else "fail",
        }
        if per
        else {"verdict": "not run", "reason": f"needs {EARLIEST_ROW} and a hop0 row on a named rung"}
    )
    out.update(l_hypotheses(by, rungs, arms))
    return out


def _not_run(reason):
    return {"verdict": "not run", "reason": reason}


def l_hypotheses(by, rungs, arms):
    """The seven hypotheses of the addendum of 2026-11-08 (the l rung; `plans/rca-arms.md`): per trace, reading hop0, hit@1 /
    hit@3, each on the arms of `L_HYPOTHESIS_ARMS` that are named. Every entry is "not run" with its reason unless l is a named
    rung and the rows it reads exist; H-l-hold also needs the m rows (`L_HOLD_RUNG`), so the registered l call names m and l."""
    out = {}
    arms = [a for a in arms if a in L_HYPOTHESIS_ARMS]
    if L_RUNG not in rungs or not arms:
        reason = f"needs rung {L_RUNG!r} and an arm of {L_HYPOTHESIS_ARMS}"
        for h in (
            "H-l-top3",
            "H-l-over-free",
            "H-l-hold",
            "H-l-vs-rcd",
            "H-l-ig-ge-cmi",
            "H-l-no-blind-kind",
            "H-l-floor-gap",
        ):
            out[h] = _not_run(reason)
        return out
    g = L_RUNG
    rows = {a: by.get((g, f"{a}/{HOP0}")) for a in arms}
    missing = [a for a, r in rows.items() if r is None]
    if missing:
        reason = f"no hop0 row at {g} for {missing}"
        for h in (
            "H-l-top3",
            "H-l-over-free",
            "H-l-hold",
            "H-l-vs-rcd",
            "H-l-ig-ge-cmi",
            "H-l-no-blind-kind",
            "H-l-floor-gap",
        ):
            out[h] = _not_run(reason)
        return out
    # H-l-top3: hop0 hit@3 >= L_TOP3_HIT3
    hit3 = {a: rows[a]["metrics"][S3] for a in arms}
    out["H-l-top3"] = {
        "hit3": hit3,
        "threshold": L_TOP3_HIT3,
        "verdict": "pass" if all(v >= L_TOP3_HIT3 for v in hit3.values()) else "fail",
    }
    # H-l-over-free: hop0 hit@1 >= max(earliest, callees hop0, random) + L_OVER_FREE_MARGIN
    free = {
        EARLIEST_ROW: by.get((g, EARLIEST_ROW)),
        CALLEES_HOP0_ROW: by.get((g, CALLEES_HOP0_ROW)),
        RANDOM_ROW: by.get((g, RANDOM_ROW)),
    }
    if any(r is None for r in free.values()):
        out["H-l-over-free"] = _not_run(f"needs {[k for k, r in free.items() if r is None]} at {g}")
    else:
        best_name, best = max(((k, r["metrics"][S1]) for k, r in free.items()), key=lambda kv: kv[1])
        gaps = {a: rows[a]["metrics"][S1] - best for a in arms}
        out["H-l-over-free"] = {
            "best_free_baseline": {"row": best_name, S1: best},
            "free_baselines": {k: r["metrics"][S1] for k, r in free.items()},
            "gap_arm_minus_best_free_hit1": gaps,
            "margin": L_OVER_FREE_MARGIN,
            "verdict": "pass" if all(v >= L_OVER_FREE_MARGIN for v in gaps.values()) else "fail",
        }
    # H-l-hold: |l - m| <= L_HOLD_GAP on hit@1 and hit@3 at hop0
    m_rows = {a: by.get((L_HOLD_RUNG, f"{a}/{HOP0}")) for a in arms}
    if any(r is None for r in m_rows.values()):
        out["H-l-hold"] = _not_run(f"needs the hop0 rows at {L_HOLD_RUNG!r} (name the rungs {L_HOLD_RUNG} {L_RUNG})")
    else:
        diffs = {
            a: {
                S1: rows[a]["metrics"][S1] - m_rows[a]["metrics"][S1],
                S3: rows[a]["metrics"][S3] - m_rows[a]["metrics"][S3],
            }
            for a in arms
        }
        out["H-l-hold"] = {
            "l_minus_m": diffs,
            "max_gap": L_HOLD_GAP,
            "verdict": "pass" if all(abs(v) <= L_HOLD_GAP for d in diffs.values() for v in d.values()) else "fail",
        }
    # H-l-vs-rcd: hop0 hit@1 and hit@3 >= RCD's broadcast row
    rcd = by.get((g, RCD_BROADCAST_ROW))
    if rcd is None:
        out["H-l-vs-rcd"] = _not_run(f"needs {RCD_BROADCAST_ROW} at {g}")
    else:
        gaps = {a: {k: rows[a]["metrics"][k] - rcd["metrics"][k] for k in (S1, S3)} for a in arms}
        out["H-l-vs-rcd"] = {
            "rcd": {k: rcd["metrics"][k] for k in (S1, S3)},
            "gap_arm_minus_rcd": gaps,
            "verdict": "pass" if all(v >= 0 for d in gaps.values() for v in d.values()) else "fail",
        }
    # H-l-ig-ge-cmi: the information-gain arm's hop0 hit@1 and hit@3 >= the CMI arm's
    ig, cmi = L_HYPOTHESIS_ARMS
    if ig in rows and cmi in rows:
        gaps = {k: rows[ig]["metrics"][k] - rows[cmi]["metrics"][k] for k in (S1, S3)}
        out["H-l-ig-ge-cmi"] = {
            "gap_ig_minus_cmi": gaps,
            "verdict": "pass" if all(v >= 0 for v in gaps.values()) else "fail",
        }
    else:
        out["H-l-ig-ge-cmi"] = _not_run(f"needs both {ig} and {cmi}")
    # H-l-no-blind-kind: every fault kind's hop0 hit@1 >= L_BLIND_KIND_FACTOR x the rung's per-trace random hit@1
    per = {}
    for a in arms:
        floor_random = rows[a]["random"][S1] * L_BLIND_KIND_FACTOR
        for kind, v in sorted(rows[a].get("by_fault_kind", {}).items()):
            per[f"{a} {kind}"] = {S1: v[S1], "bound": floor_random, "ok": v[S1] >= floor_random}
    out["H-l-no-blind-kind"] = (
        {
            "factor": L_BLIND_KIND_FACTOR,
            "per_arm_and_kind": per,
            "verdict": "pass" if all(x["ok"] for x in per.values()) else "fail",
        }
        if per
        else _not_run("needs by_fault_kind on the hop0 rows")
    )
    # H-l-floor-gap: hop0 hit@1 >= the call-tree floor's hop0 hit@1 - L_FLOOR_GAP
    floor = by.get((g, f"{ARM_FLOOR_TRACE_TOPOLOGY}/{HOP0}"))
    if floor is None:
        out["H-l-floor-gap"] = _not_run(f"needs {ARM_FLOOR_TRACE_TOPOLOGY}/{HOP0} at {g}")
    else:
        gaps = {a: rows[a]["metrics"][S1] - floor["metrics"][S1] for a in arms}
        out["H-l-floor-gap"] = {
            "floor_hit1": floor["metrics"][S1],
            "gap_arm_minus_floor_hit1": gaps,
            "max_gap": L_FLOOR_GAP,
            "verdict": "pass" if all(v >= -L_FLOOR_GAP for v in gaps.values()) else "fail",
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
        shared_arms = [a for a in arms if has_shared_rows(cells_by_arm[a][0][0])]
        first = shared_arms[0] if shared_arms else None
        row_ids = [rid for rid in cells_by_arm[first][0][0] if is_shared(rid)] if first else []
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
    p.add_argument("--arms", required=True, nargs="+", choices=REPORT_ARMS)
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
