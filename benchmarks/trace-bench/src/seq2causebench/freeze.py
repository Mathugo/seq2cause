# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Freeze the validation-selected values of one corpus (PRD non-negotiable 10; scenarios 13, 29;
`plans/reference-arms.md` §4 and `plans/caps.md`'s checkpoint rule).

    python -m seq2causebench.freeze --val-tables out/<scoresweep-request>/val-table.json out/<scoresweep-session>/val-table.json \
        --pretrain-results out/<pretrain>/run/results.json --alt-val-tables '' --checkpoint-choice last \
        --rung xs --variant latent --seed 0 --freezes-dir freezes --output-folder out/<run>

Per frozen cell `<arm>/<path>/frozen/<grain>` the full-grid argmax of the scorer's directed F1 at
the default floor over `(c, N, τ)`; ties → smaller `N`, then smaller `c`, then larger τ. A cli
arm's shipped cut `<arm>/<path>/shipped/<grain>` inherits the frozen sibling's `(c, N, g)` and has
no τ. The checkpoint trigger: if the pretrain record's final validation loss exceeds
`CHECKPOINT_TRIGGER_RATIO` × its minimum, the argmin-val checkpoint must also have been swept
(`--alt-val-tables`, a second scoresweep on that checkpoint's model) and the freeze binds whichever
model won on `trace/core/shipped/frozen/request` directed F1; otherwise the last checkpoint is
frozen and `--alt-val-tables ''`. Writes `freezes/<date>-<rung>-<variant>-s<k>.json` with the
corpus identity, the benchmark tool version and config hash, the model hash, the package commit
at freeze time, the val tables' sha256, the grids and the chosen values per cell; refuses to
overwrite. The record is then committed (one freeze commit per rung) and `discover --split test`
asserts on that commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .arms import arm_spec, cell_key
from .constants import (
    ARM_CORE,
    CHECKPOINT_TRIGGER_RATIO,
    CUT_FROZEN,
    CUT_SHIPPED,
    PATH_SHIPPED,
    RUNGS,
    SEEDS,
    VARIANTS,
)
from .log import log, now_iso
from .record import RunRecord, git_commit, read_json, sha256_file, write_json

FREEZE_SCHEMA = "seq2causebench/freeze@1"
TRIGGER_CELL = (ARM_CORE, PATH_SHIPPED, "request")


class FreezeExists(FileExistsError):
    pass


class FreezeRefusal(RuntimeError):
    pass


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
    for key, rows in by_cell.items():
        best = max(rows, key=lambda r: (r["directed"]["f1"], -r["N"], -r["c"], r["tau"]))
        chosen[key] = {
            "tau": float(best["tau"]),
            "tau_source": best.get("tau_source"),
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


def freeze_name(date, rung, variant, seed):
    return f"{date}-{rung}-{variant}-s{seed}.json"


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


def trigger_facts(pretrain_results):
    pre = read_json(pretrain_results)
    ratio = pre.get("val_final_over_min")
    fired = ratio is not None and ratio > CHECKPOINT_TRIGGER_RATIO
    curve = pre.get("val_curve", [])
    argmin = min(curve, key=lambda v: v["loss"]) if curve else None
    return {
        "ratio": ratio,
        "threshold": CHECKPOINT_TRIGGER_RATIO,
        "fired": bool(fired),
        "argmin_step": None if argmin is None else argmin["step"],
        "last_model_sha256": pre.get("model_sha256"),
    }


def _trigger_f1(table):
    key = cell_key(*TRIGGER_CELL[:2], CUT_FROZEN, TRIGGER_CELL[2])
    cells = select_cells(table)
    if key not in cells:
        raise FreezeRefusal(
            f"the checkpoint trigger compares {key}, which the val tables do not contain"
        )
    return cells[key]["val_directed_f1"]


def choose_tables(main_tables, alt_tables, trigger, checkpoint_choice):
    """The tables to freeze from, per the checkpoint rule; returns `(table, choice facts)`."""
    main = merge_tables(main_tables)
    if main["model_sha256"] != trigger["last_model_sha256"]:
        raise FreezeRefusal(
            f"the val tables bind model {main['model_sha256']} but the pretrain record's last model is "
            f"{trigger['last_model_sha256']}"
        )
    if not trigger["fired"]:
        if alt_tables:
            raise FreezeRefusal("the checkpoint trigger did not fire; --alt-val-tables must be ''")
        if checkpoint_choice != "last":
            raise FreezeRefusal(
                "the checkpoint trigger did not fire; --checkpoint-choice must be 'last'"
            )
        return main, {"choice": "last", "trigger": trigger}
    if not alt_tables:
        raise FreezeRefusal(
            f"the checkpoint trigger fired (val_final / val_min = {trigger['ratio']:.4f} > "
            f"{CHECKPOINT_TRIGGER_RATIO}); sweep the argmin-val checkpoint (step {trigger['argmin_step']}) "
            "and pass its scoresweep tables as --alt-val-tables (plans/caps.md)"
        )
    alt = merge_tables(alt_tables)
    if alt["model_sha256"] == main["model_sha256"]:
        raise FreezeRefusal("--alt-val-tables bind the same model as the main tables")
    f_main, f_alt = _trigger_f1(main), _trigger_f1(alt)
    winner = "argmin" if f_alt > f_main else "last"
    if checkpoint_choice != winner:
        raise FreezeRefusal(
            f"--checkpoint-choice {checkpoint_choice!r} but the trigger cell's val F1 picks {winner!r} "
            f"(last {f_main:.4f} vs argmin {f_alt:.4f})"
        )
    return (alt if winner == "argmin" else main), {
        "choice": winner,
        "trigger": trigger,
        "f1_last": f_main,
        "f1_argmin": f_alt,
    }


def build_freeze(table, val_table_paths, alt_paths, choice, rung, variant, seed):
    paths = [Path(p) for p in val_table_paths]
    alts = [Path(p) for p in alt_paths]
    return {
        "schema": FREEZE_SCHEMA,
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
        "alt_val_tables": [{"name": p.name, "sha256": sha256_file(p)} for p in alts],
        "checkpoint": choice,
        "taus": table.get("taus"),
        "granger_taus": table.get("granger_taus"),
        "quantiles": table.get("quantiles"),
        "grains": table.get("grains"),
        "cells": select_cells(table),
    }


def run_freeze(args, rec):
    trigger = trigger_facts(args.pretrain_results)
    alt = [p for p in args.alt_val_tables if p]
    table, choice = choose_tables(
        [read_json(p) for p in args.val_tables],
        [read_json(p) for p in alt],
        trigger,
        args.checkpoint_choice,
    )
    freeze = build_freeze(table, args.val_tables, alt, choice, args.rung, args.variant, args.seed)
    date = dt.datetime.now(dt.UTC).date().isoformat()
    path = Path(args.freezes_dir) / freeze_name(date, args.rung, args.variant, args.seed)
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
            "checkpoint": choice["choice"],
        }
    )
    return {
        "freeze": str(path),
        "n_cells": len(freeze["cells"]),
        "cells": freeze["cells"],
        "corpus_id": freeze["corpus_id"],
        "model_sha256": freeze["model_sha256"],
        "checkpoint": choice,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--val-tables",
        required=True,
        nargs="+",
        help="one val-table.json per grain of one corpus (the last checkpoint's model)",
    )
    p.add_argument(
        "--pretrain-results",
        required=True,
        help="the pretrain run's results.json (the checkpoint trigger)",
    )
    p.add_argument(
        "--alt-val-tables",
        required=True,
        nargs="+",
        help="the argmin-val checkpoint's tables when the trigger fired, else ''",
    )
    p.add_argument(
        "--checkpoint-choice",
        required=True,
        choices=["last", "argmin"],
        help="must equal what the trigger rule picks",
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
