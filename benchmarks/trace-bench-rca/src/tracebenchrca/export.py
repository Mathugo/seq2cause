"""Export one pulled corpus's fault cases through the benchmark's own exporter and record it.

    python -m tracebenchrca.export --corpus-dir <pulled corpus> --rung xs --out <export root> \
        --workers 1 --extras topology spans --expect-cases none --output-folder <run-dir>

`--expect-cases` (2026-10-31) names a landed export record whose per-case sha256 this export must
reproduce (a re-export on another host is then the same input by construction), or `none`.

`--extras` (2026-10-26; may be empty) asks the exporter for its optional case files — `topology.json`
(`rca/circa-topology`) and `spans.csv` (`rca/tracerca`); a pin whose exporter has no extras refuses a
non-empty request. The three registered files are the same with or without them.

A thin stage around `tracebench.export_rca.export` (score side; it reads the configuration, the
instantiation, `labels/cases.json` — the two window fields only — and the access-log shards):
the benchmark writes the case directories and its own run record under `<out>/run/`; this
stage writes this directory's record beside it (`--output-folder`), carrying the corpus id,
the exporter's results (every input hash, the per-case bin counts, the written files' hashes)
and this repository's commit, so a landed export is a ledger-able record of the harness.
The exporter module arrives with the benchmark pin of `pyproject.toml`; a pin that predates it
fails here with a clear message rather than on the host.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .constants import BIN_WIDTH_S, RUNGS
from .log import log
from .record import RunRecord, read_json, sha256_file


def corpus_id_of(corpus_dir):
    """`<rung>/<variant>/seed=<k>` when the directory follows the dataset host's layout
    (`instances/<rung>/<variant>/seed=<k>`), else the directory name."""
    parts = Path(corpus_dir).resolve().parts
    if len(parts) >= 4 and parts[-4] == "instances" and parts[-1].startswith("seed="):
        return "/".join(parts[-3:])
    return parts[-1]


def _exporter():
    try:
        export_rca = importlib.import_module("tracebench.export_rca")
    except ImportError as e:  # the benchmark pin predates the exporter
        raise ImportError(
            "tracebench.export_rca is not in the installed benchmark package; the pin in pyproject.toml must "
            "name a commit of the publication branch that carries it"
        ) from e
    if dict(export_rca.BIN_WIDTH_S) != dict(BIN_WIDTH_S):
        raise RuntimeError(f"the exporter's widths {export_rca.BIN_WIDTH_S} differ from the registered {BIN_WIDTH_S}")
    return export_rca


def check_expected_cases(results, expect):
    """Per-case content check against a landed export record (`--expect-cases`): every case directory's
    exporter sha256 must equal the landed one, so a re-export on another host is the same input. Returns
    the landed record's path and sha256, or None when `expect` is "none"."""
    if expect == "none":
        return None
    landed = read_json(expect)
    new = {c["dir"]: c["sha256"] for c in results["exporter"]["cases"]}
    old = {c["dir"]: c["sha256"] for c in landed["exporter"]["cases"]}
    if new != old:
        bad = sorted(d for d in set(new) | set(old) if new.get(d) != old.get(d))
        raise ValueError(f"the export differs from the landed record {expect} on {bad}")
    return {"path": str(expect), "sha256": sha256_file(expect), "n_cases": len(new)}


def export(corpus_dir, rung, out, workers, extras=(), expect="none"):
    exporter = _exporter()
    extras = tuple(extras)
    if extras:
        if not hasattr(exporter, "EXTRAS"):
            raise RuntimeError(
                "the pinned exporter has no extras; the pin must move to the commit that carries them (D-TBR-11)"
            )
        results = exporter.export(corpus_dir, rung, out, workers=workers, extras=extras)
    else:
        results = exporter.export(corpus_dir, rung, out, workers=workers)
    res = {
        "corpus_id": corpus_id_of(corpus_dir),
        "export_dir": str(Path(out).resolve()),
        "stage": "export",
        "rung": rung,
        "extras": list(extras),
        "exporter": results,
    }
    res["expected_cases"] = check_expected_cases(res, expect)
    return res


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True, help="a pulled corpus (tier rca)")
    p.add_argument(
        "--rung", required=True, choices=RUNGS, help="selects the registered bin width; must match the corpus"
    )
    p.add_argument("--out", required=True, help="the export root for this corpus (one case-<k>/ per fault)")
    p.add_argument("--workers", required=True, type=int, help="shard parsers in parallel (rows pooled before binning)")
    p.add_argument(
        "--extras",
        required=True,
        nargs="*",
        choices=("topology", "spans"),
        help="the exporter's optional case files; may be empty",
    )
    p.add_argument(
        "--expect-cases",
        required=True,
        help="a landed export record's results.json whose per-case sha256 this export must reproduce, or none",
    )
    p.add_argument("--output-folder", required=True, help="where this stage's run record is written")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "export", vars(args)) as rec:
        res = export(args.corpus_dir, args.rung, args.out, args.workers, tuple(args.extras), args.expect_cases)
        log({"event": "export", "corpus_id": res["corpus_id"], "n_cases": len(res["exporter"]["cases"])})
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
