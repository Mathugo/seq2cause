"""The sensitivity table (`plans/reference-arms.md`, addendum 2026-10-22): N = 64 / 128 and a session
context of 16 against the landed seed-0 freeze of each rung — validation only, one corpus per rung,
nothing re-selected.

    python -m seq2causebench.sensitivitytable --freezes freezes/2026-09-25-xs-latent-s0.json ... \
        --landed-val-tables <the freezes' val tables> --sensitivity-val-tables <the sensitivity scoresweeps> \
        --sensitivity-sweep-records <the sensitivity sweep folders> --output-folder tables/sensitivity-latent

Per rung, grain and frozen cell `<arm>/<path>/frozen/<grain>` of the landed freeze: the landed frozen
`(c, N, τ)` and validation F1; for each new `(c, N)` of that arm and path, the validation F1 at the
argmax over τ (ties → the larger τ) and its difference from the landed value; beside them the landed
table's per-`(c, N)` best F1, so each factor reads as a curve. `c = 16` rows carry their probed count
and reachable-recall ceiling. The registered expectation (a gain under 0.01) is reported per cell, not
scored. Every landed table must be the one its freeze hashed; every sensitivity table must bind the
freeze's model and corpus on the end view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .log import log
from .record import RunRecord, read_json, sha256_file, write_json

TABLE_SCHEMA = "seq2causebench/sensitivity@1"
GAIN_BAND = 0.01  # the registered expectation (reported, not scored)
NEW_N = (64, 128)
NEW_C = 16


def best_by_cn(table, arm, path, grain):
    """`{(c, N): (f1, τ)}`: the argmax of directed F1 over τ per (c, N), ties to the larger τ."""
    out = {}
    for r in table["cells"]:
        if (r["arm"], r["path"], r["cut"], r["grain"]) != (arm, path, "frozen", grain):
            continue
        key, cand = (r["c"], r["N"]), (r["directed"]["f1"], r["tau"])
        if key not in out or cand > out[key]:
            out[key] = cand
    return out


def coverage_of(table, arm, path, grain, c, n):
    return (table.get("coverage", {}).get(f"{arm}/{path}/frozen/{grain}/c{c}/N{n}") or {}).get(
        "coverage", {}
    )


def probed_counts(sweep_records):
    """`{(rung, grain, probe, noise, c, N): (n_probed, n_sequences)}` from the sensitivity sweeps."""
    out = {}
    for d in sweep_records:
        res = read_json(Path(d) / "run" / "results.json")
        for cell in res["cells"]:
            key = (res["rung"], cell["grain"], cell["probe"], cell["noise"], cell["c"], cell["N"])
            out[key] = (cell["n_probed"], cell["n_sequences"])
    return out


def load(freezes, landed, sensitivity):
    by_rung = {}
    for f in freezes:
        fz = read_json(f)
        if fz["seed"] != 0 or fz.get("ordering", "end") != "end":
            raise ValueError(f"{f} is not a seed-0 end-view freeze")
        by_rung[fz["rung"]] = {"freeze": fz, "freeze_path": str(f), "landed": {}, "sens": []}
    for p in landed:
        t = read_json(p)
        rung = t["instance"]
        fz = by_rung[rung]["freeze"]
        if sha256_file(p) not in {v["sha256"] for v in fz["val_tables"]}:
            raise ValueError(f"{p} is not a validation table the {rung} freeze hashed")
        (grain,) = t["grains"]
        by_rung[rung]["landed"][grain] = t
    for p in sensitivity:
        t = read_json(p)
        rung = t["instance"]
        fz = by_rung[rung]["freeze"]
        if (t["model_sha256"], t["corpus_id"], t["ordering"]) != (
            fz["model_sha256"],
            fz["corpus_id"],
            "end",
        ):
            raise ValueError(
                f"{p} does not bind the {rung} freeze's model and corpus on the end view"
            )
        by_rung[rung]["sens"].append(t)
    return by_rung


def build(by_rung, probed):
    rows = []
    for rung, d in by_rung.items():
        fz = d["freeze"]
        for key, cell in sorted(fz["cells"].items()):
            arm_, path, cut, grain = key.rsplit("/", 3)
            if cut != "frozen":
                continue
            landed_curve = best_by_cn(d["landed"][grain], arm_, path, grain)
            new = {}
            for t in d["sens"]:
                if t["grains"] != [grain]:
                    continue
                for cn, (f1, tau) in best_by_cn(t, arm_, path, grain).items():
                    cov = coverage_of(t, arm_, path, grain, *cn)
                    cnt = probed.get((rung, grain, cell["probe"], cell["noise"], *cn))
                    new[cn] = {
                        "c": cn[0],
                        "N": cn[1],
                        "tau": tau,
                        "val_f1": f1,
                        "diff_from_landed": f1 - cell["val_directed_f1"],
                        "reachable_recall_ceiling": cov.get("reachable_recall_ceiling"),
                        "n_probed": None if cnt is None else cnt[0],
                        "n_sequences": None if cnt is None else cnt[1],
                    }
            if not new:
                continue
            best_landed = max(f for f, _ in landed_curve.values())
            n_gain = [v["val_f1"] - best_landed for (c, n), v in new.items() if n in NEW_N]
            c_gain = [v["val_f1"] - best_landed for (c, n), v in new.items() if c == NEW_C]
            rows.append(
                {
                    "rung": rung,
                    "cell": key,
                    "landed": {
                        "c": cell["c"],
                        "N": cell["N"],
                        "tau": cell["tau"],
                        "val_f1": cell["val_directed_f1"],
                    },
                    "landed_curve": [
                        {"c": c, "N": n, "val_f1": f, "tau": t}
                        for (c, n), (f, t) in sorted(landed_curve.items())
                    ],
                    "new": [new[k] for k in sorted(new)],
                    "expectation": {
                        "band": GAIN_BAND,
                        "n_gain_over_best_landed": max(n_gain) if n_gain else None,
                        "c16_gain_over_best_landed": max(c_gain) if c_gain else None,
                        "within_band": all(g < GAIN_BAND for g in n_gain + c_gain),
                    },
                }
            )
    return rows


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def render(doc):
    out = [
        "# Sensitivity — N = 64 / 128 and session context 16 (validation only, seed 0, one corpus per rung, not re-selected)",
        "",
        "`plans/reference-arms.md`, addendum 2026-10-22. Each row: the landed frozen cell of the seed-0 freeze, then each new "
        "`(c, N)` at its own best τ on the rung's grids, with the difference from the landed frozen value. The landed table's "
        "best F1 per `(c, N)` is the curve the new cells extend. `c = 16` reads only the longer sessions: its probed count "
        "and reachable-recall ceiling are printed beside its F1. Expectation (reported, not scored): a gain under 0.01 over "
        "the best landed `(c, N)`.",
    ]
    for rung in doc["rungs"]:
        for grain in ("request", "session"):
            rows = [r for r in doc["rows"] if r["rung"] == rung and r["cell"].endswith("/" + grain)]
            if not rows:
                continue
            out += [
                "",
                f"## {rung} / {grain}",
                "",
                "| cell | landed (c, N, τ) F1 | landed curve: best F1 per (c, N) | new (c, N): F1 (Δ landed) [probed / ceiling] | gain over best landed | < 0.01 |",
                "|---|---|---|---|---|---|",
            ]
            for r in rows:
                L = r["landed"]
                curve = ", ".join(
                    f"({p['c']},{p['N']}) {p['val_f1']:.3f}" for p in r["landed_curve"]
                )
                new = ", ".join(
                    f"({p['c']},{p['N']}) {p['val_f1']:.3f} ({p['diff_from_landed']:+.3f})"
                    + (
                        f" [{p['n_probed']}/{p['n_sequences']}, {fmt(p['reachable_recall_ceiling'])}]"
                        if p["c"] == NEW_C
                        else ""
                    )
                    for p in r["new"]
                )
                e = r["expectation"]
                gains = [
                    g
                    for g in (e["n_gain_over_best_landed"], e["c16_gain_over_best_landed"])
                    if g is not None
                ]
                out.append(
                    f"| `{r['cell']}` | ({L['c']}, {L['N']}, {L['tau']:.3g}) {L['val_f1']:.3f} | {curve} | {new} | "
                    f"{', '.join(f'{g:+.3f}' for g in gains)} | {'yes' if e['within_band'] else '**no**'} |"
                )
    return "\n".join(out)


def run(args, rec):
    by_rung = load(args.freezes, args.landed_val_tables, args.sensitivity_val_tables)
    rows = build(by_rung, probed_counts(args.sensitivity_sweep_records))
    doc = {
        "schema": TABLE_SCHEMA,
        "label": "validation only, seed 0, one corpus per rung, not re-selected",
        "rungs": sorted(by_rung, key=["xs", "s", "m", "l", "xl"].index),
        "freezes": {
            r: {"path": d["freeze_path"], "model_sha256": d["freeze"]["model_sha256"]}
            for r, d in by_rung.items()
        },
        "rows": rows,
        "n_outside_band": sum(1 for r in rows if not r["expectation"]["within_band"]),
    }
    stem = Path(args.output_folder).name
    write_json(rec.out_dir / f"{stem}.json", doc)
    (rec.out_dir / f"{stem}.md").write_text(render(doc) + "\n")
    log(
        {"event": "sensitivitytable_done", "rows": len(rows), "outside_band": doc["n_outside_band"]}
    )
    return {"table": f"{stem}.json", "n_rows": len(rows), "n_outside_band": doc["n_outside_band"]}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--freezes", required=True, nargs="+", help="the landed seed-0 freeze of each rung"
    )
    p.add_argument(
        "--landed-val-tables", required=True, nargs="+", help="the val tables those freezes hashed"
    )
    p.add_argument("--sensitivity-val-tables", required=True, nargs="+")
    p.add_argument(
        "--sensitivity-sweep-records", required=True, nargs="+", help="for the probed counts"
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "sensitivitytable", vars(args)) as rec:
        res = run(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
