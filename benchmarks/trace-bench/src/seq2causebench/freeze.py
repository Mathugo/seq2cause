# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Freeze the validation-selected values of one corpus (PRD non-negotiable 10; scenarios 13, 29;
`plans/reference-arms.md` §4 with its 2026-09-25 addendum and `plans/caps.md`'s checkpoint rule).

    python -m seq2causebench.freeze --val-tables out/<scoresweep-request>/val-table.json out/<scoresweep-session>/val-table.json \
        --pretrain-results out/<pretrain>/run/results.json \
        --rung xs --variant latent --seed 0 --freezes-dir freezes --output-folder out/<run>

Per frozen cell `<arm>/<path>/frozen/<grain>` the full-grid argmax of the scorer's directed F1 at
the default floor over `(c, N, τ)`; ties → smaller `N`, then smaller `c`, then larger τ. A row
whose τ is negative is not a candidate (`plans/reference-arms.md`, addendum 2026-10-05: a cut
that keeps zero and negative scores is "every scored pair", the reference line, not a threshold
on the arm's evidence; a quantile of a signed score can land there) — the count of rows set aside
is recorded per cell (`n_rows_negative_tau`). A cli
arm's shipped cut `<arm>/<path>/shipped/<grain>` inherits the frozen sibling's `(c, N, g)` and has
no τ. The model is the one the pretrain record names (`model_choice`, the argmin-validation
checkpoint under the plans' addendum); the val tables must bind that hash. Instrument-soundness
facts are recorded, never gated (`diagnostics`): the pretrain record's `model_choice` and oracle
(chosen and last checkpoint), and per frozen cell the reachable-recall coverage ceiling and an
`at_grid_edge` flag (a τ at either end of its grid — the absolute grids, and since `freeze@3` the
quantile family of saliency and Shapley too), with the F1 of the cut "every scored pair" at the
cell's `(c, N)` beside it (`every_scored_pair_f1`). Its source is named: `scorer` when the val
table carries the scorer's own line for the full ranking, `coverage` when the table was written
without it and the same number is taken from the counts the scorer returned for that ranking
(true positives = ceiling × truth, predictions = the scored pairs inside the universe). Writes
`freezes/<date>-<rung>-<variant>-s<k>.json` with the corpus identity, the benchmark tool version
and config hash, the model hash, the package commit at freeze time, the val tables' sha256, the
grids and the chosen values per cell; refuses to overwrite. The record is then committed (one
freeze commit per rung) and `discover --split test` asserts on that commit. Since 2026-10-09
(`plans/floors.md`, D-SB-16) the document also carries `family` (`model`, or `floor` for a table of
floor cells only — frozen apart, in `…-s<k>-floors.json`, with `--pretrain-results none` and the
model hash `none`), `ordering` (the view the tables were swept on; `-start` in the name when it
is `start`) and `floor_taus`; the schema stays `freeze@3`, the keys are additive.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .arms import arm_spec, cell_key, parse_cell_key
from .constants import (
    ARM_GRANGER,
    CUT_FROZEN,
    CUT_SHIPPED,
    FAMILY_FLOOR,
    FAMILY_MODEL,
    FLOOR_ARMS,
    MODEL_NONE,
    REFERENCE_ARMS,
    RUNGS,
    SEEDS,
    VARIANTS,
)
from .log import log, now_iso
from .record import RunRecord, git_commit, read_json, sha256_file, write_json

FREEZE_SCHEMA = "seq2causebench/freeze@3"


class FreezeExists(FileExistsError):
    pass


class FreezeRefusal(RuntimeError):
    pass


QUANTILE_SOURCE = "quantile p"


def grid_of(arm, tau_source):
    """The val table's grid list a τ came from: `taus` / `granger_taus` for a grid-sourced τ,
    `quantiles` for a quantile-sourced one."""
    if str(tau_source).startswith(QUANTILE_SOURCE):
        return "quantiles"
    if tau_source != "grid":
        return None
    if arm in FLOOR_ARMS:
        return "floor_taus"
    if arm in REFERENCE_ARMS:
        return "taus"
    if arm == ARM_GRANGER:
        return "granger_taus"
    return None


def at_grid_edge(row, grid_name, grid):
    """Whether the chosen row sits at either end of its grid. An absolute grid is compared by τ;
    the quantile family by the quantile the row's `tau_source` names (its τ is data-dependent)."""
    if not grid:
        return None
    if grid_name == "quantiles":
        ends = {f"{QUANTILE_SOURCE}{int(round(q * 100))}" for q in (min(grid), max(grid))}
        return row.get("tau_source") in ends
    return bool(row["tau"] in (min(grid), max(grid)))


def every_scored_pair(ranked):
    """`(f1, source)` of the cut "every scored pair" for one ranking row of a val table. The
    scorer's line when the row carries it. Otherwise the scorer's counts for the same call give
    it exactly: its directed recall on the full ranking is the ceiling, so the true positives are
    `ceiling × truth` (an integer), its predictions are the scored pairs inside the universe, and
    F1 = 2·tp / (predictions + truth). `(None, None)` when the row has neither."""
    line = ranked.get("every_scored_pair")
    if line:
        return line.get("f1"), "scorer"
    cov = ranked.get("coverage") or {}
    counts = [
        cov.get(k) for k in ("reachable_recall_ceiling", "truth_directed", "pairs_cooccurring")
    ]
    if any(v is None for v in counts):
        return None, None
    ceiling, truth, predicted = counts
    tp = round(ceiling * truth)
    return (2 * tp / (predicted + truth) if predicted + truth else 0.0), "coverage"


def select_cells(table):
    """The argmax rows per frozen cell with the plan's tie-breaks; shipped cuts inherit (c, N, g)."""
    frozen_rows = [r for r in table["cells"] if r["cut"] == CUT_FROZEN]
    if not frozen_rows:
        raise FreezeRefusal("the val table has no frozen-cut rows")
    floor = frozen_rows[0]["floor"]
    by_cell = {}
    for r in frozen_rows:
        if r["floor"] != floor:
            raise FreezeRefusal(
                "val table mixes floors; the freeze selects at the default floor only"
            )
        by_cell.setdefault(cell_key(r["arm"], r["path"], CUT_FROZEN, r["grain"]), []).append(r)
    chosen = {}
    coverage = table.get("coverage", {})
    for key, all_rows in by_cell.items():
        rows = [r for r in all_rows if not (r["tau"] is not None and r["tau"] < 0)]
        if not rows:
            raise FreezeRefusal(f"{key}: every swept row has a negative τ; nothing to freeze")
        best = max(rows, key=lambda r: (r["directed"]["f1"], -r["N"], -r["c"], r["tau"]))
        grid_name = grid_of(best["arm"], best.get("tau_source"))
        grid = table.get(grid_name) if grid_name else None
        ranked = coverage.get(f"{key}/c{best['c']}/N{best['N']}", {})
        cov = ranked.get("coverage", {})
        every_f1, every_source = every_scored_pair(ranked)
        chosen[key] = {
            "tau": float(best["tau"]),
            "tau_source": best.get("tau_source"),
            "grid": grid_name,
            "at_grid_edge": at_grid_edge(best, grid_name, grid),
            "reachable_recall_ceiling": cov.get("reachable_recall_ceiling"),
            "every_scored_pair_f1": every_f1,
            "every_scored_pair_source": every_source,
            "c": int(best["c"]),
            "N": int(best["N"]),
            "g": int(best["g"]),
            "probe": best["probe"],
            "noise": best["noise"],
            "val_directed_f1": float(best["directed"]["f1"]),
            "val_directed_precision": float(best["directed"]["precision"]),
            "val_directed_recall": float(best["directed"]["recall"]),
            "n_edges": int(best["n_edges"]),
            "floor": float(best["floor"]),
            "n_rows_considered": len(rows),
            "n_rows_negative_tau": len(all_rows) - len(rows),
        }
        arm, path, grain = best["arm"], best["path"], best["grain"]
        if CUT_SHIPPED in arm_spec(arm)["cuts"]:
            chosen[cell_key(arm, path, CUT_SHIPPED, grain)] = {
                "tau": None,
                "tau_source": "shipped rule",
                "c": int(best["c"]),
                "N": int(best["N"]),
                "g": int(best["g"]),
                "probe": best["probe"],
                "noise": best["noise"],
                "inherits": key,
                "floor": float(best["floor"]),
            }
    return chosen


def family_of(cells):
    """`floor` when every frozen cell is a floor arm's, `model` when none is; a mixture is refused
    (floor cells are frozen apart from model-bound cells, D-SB-16)."""
    arms = {parse_cell_key(k)[0] for k in cells}
    floors = {a for a in arms if a in FLOOR_ARMS}
    if floors and floors != arms:
        raise FreezeRefusal(
            f"floor cells {sorted(floors)} cannot be frozen together with model-bound cells "
            f"{sorted(arms - floors)} (D-SB-16)"
        )
    return FAMILY_FLOOR if floors else FAMILY_MODEL


def freeze_name(date, rung, variant, seed, family, ordering):
    name = f"{date}-{rung}-{variant}-s{seed}"
    if family == FAMILY_FLOOR:
        name += "-floors"
    if ordering == "start":
        name += "-start"
    return name + ".json"


def merge_tables(tables):
    """One table from the per-grain tables of one corpus; refuses mixed corpora, models or grids."""
    first = tables[0]
    for t in tables[1:]:
        for k in (
            "corpus_id",
            "model_sha256",
            "tool_version",
            "config_hash",
            "taus",
            "granger_taus",
            "quantiles",
            "floor_taus",
            "ordering",
        ):
            if t.get(k) != first.get(k):
                raise FreezeRefusal(f"val tables disagree on {k}: {first.get(k)!r} vs {t.get(k)!r}")
    grains = []
    for t in tables:
        for g in t.get("grains", []):
            if g in grains:
                raise FreezeRefusal(f"grain {g} appears in more than one val table")
            grains.append(g)
    return {
        **first,
        "grains": grains,
        "cells": [c for t in tables for c in t["cells"]],
        "coverage": {k: v for t in tables for k, v in t.get("coverage", {}).items()},
    }


PRETRAIN_FACTS = (
    "model_choice",
    "oracle",
    "oracle_last",
    "val_loss_final",
    "val_loss_min",
    "val_final_over_min",
)


def pretrain_facts(pretrain_results, table):
    """The pretrain record's model identity and soundness facts; refuses tables bound to another
    model (the tables must have been swept on the model the record names). A model-free table
    (model hash `none`) takes the literal `none` in place of a record and gets no facts; a real
    record against such a table, or `none` against a model-bound table, is refused (D-SB-16)."""
    if str(pretrain_results) == MODEL_NONE:
        if table["model_sha256"] != MODEL_NONE:
            raise FreezeRefusal(
                f"--pretrain-results none is for model-free tables; these bind model "
                f"{table['model_sha256']} (D-SB-16)"
            )
        return dict.fromkeys(PRETRAIN_FACTS)
    if table["model_sha256"] == MODEL_NONE:
        raise FreezeRefusal(
            "a model-free val table (model hash none) takes --pretrain-results none (D-SB-16)"
        )
    pre = read_json(pretrain_results)
    if table["model_sha256"] != pre.get("model_sha256"):
        raise FreezeRefusal(
            f"the val tables bind model {table['model_sha256']} but the pretrain record's model is "
            f"{pre.get('model_sha256')}"
        )
    return {k: pre.get(k) for k in PRETRAIN_FACTS}


def build_freeze(table, val_table_paths, pretrain, rung, variant, seed):
    paths = [Path(p) for p in val_table_paths]
    cells = select_cells(table)
    family = family_of(cells)
    if (family == FAMILY_FLOOR) != (table.get("model_sha256") == MODEL_NONE):
        raise FreezeRefusal(
            f"a {family} table carries model hash {table.get('model_sha256')!r} (D-SB-16)"
        )
    return {
        "schema": FREEZE_SCHEMA,
        "family": family,
        "ordering": table.get("ordering"),
        "rung": rung,
        "variant": variant,
        "seed": int(seed),
        "corpus_id": table.get("corpus_id"),
        "tool_version": table.get("tool_version"),
        "config_hash": table.get("config_hash"),
        "model_sha256": table.get("model_sha256"),
        "commit": git_commit(),
        "frozen_at": now_iso(),
        "val_tables": [{"name": p.name, "sha256": sha256_file(p)} for p in paths],
        "taus": table.get("taus"),
        "granger_taus": table.get("granger_taus"),
        "quantiles": table.get("quantiles"),
        "floor_taus": table.get("floor_taus"),
        "grains": table.get("grains"),
        "cells": cells,
        "diagnostics": {
            **pretrain,
            "coverage": {
                k: c.get("reachable_recall_ceiling")
                for k, c in cells.items()
                if "inherits" not in c
            },
            "at_grid_edge": [k for k, c in cells.items() if c.get("at_grid_edge")],
        },
    }


def run_freeze(args, rec):
    table = merge_tables([read_json(p) for p in args.val_tables])
    pretrain = pretrain_facts(args.pretrain_results, table)
    freeze = build_freeze(table, args.val_tables, pretrain, args.rung, args.variant, args.seed)
    date = dt.datetime.now(dt.UTC).date().isoformat()
    path = Path(args.freezes_dir) / freeze_name(
        date, args.rung, args.variant, args.seed, freeze["family"], freeze.get("ordering")
    )
    if path.exists():
        raise FreezeExists(
            f"{path} exists; a freeze is never overwritten (a re-freeze is a new dated file with its reason recorded)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, freeze)
    log(
        {
            "event": "freeze_written",
            "path": str(path),
            "n_cells": len(freeze["cells"]),
            "family": freeze["family"],
            "model_step": (pretrain.get("model_choice") or {}).get("step"),
            "at_grid_edge": freeze["diagnostics"]["at_grid_edge"],
        }
    )
    return {
        "freeze": str(path),
        "n_cells": len(freeze["cells"]),
        "cells": freeze["cells"],
        "corpus_id": freeze["corpus_id"],
        "model_sha256": freeze["model_sha256"],
        "diagnostics": freeze["diagnostics"],
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--val-tables",
        required=True,
        nargs="+",
        help="one val-table.json per grain of one corpus, swept on the pretrain record's model",
    )
    p.add_argument(
        "--pretrain-results",
        required=True,
        help="the pretrain run's results.json (model hash, model_choice, oracle: recorded, never gated); "
        "'none' for a table of floor cells (D-SB-16)",
    )
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--freezes-dir", required=True, help="the repository's freezes/ directory")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "freeze", vars(args)) as rec:
        res = run_freeze(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
