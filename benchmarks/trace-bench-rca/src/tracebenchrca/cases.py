"""The method's door to an export: what a root-cause arm sees of a corpus.

`tracebench.export_rca` (score side, in the benchmark repository) writes one directory per
injected fault, `case-<k>/`, holding exactly three files — `data.csv` (the per-service series
in the standard suite's shape: one `time` column, UNIX seconds, and `<service>_<metric>`
columns), `window.json` (exactly `inject_time` and `end_time`, the only two label fields a
method receives) and `services.json` (the deployed-service universe). Every read on the
method side goes through `open_case_file`, which refuses any other name and any path that
escapes the case directory, so no code path in a method module can reach the corpus, its
labels, graphs or topology.

`read_table` applies the standard suite's own loading rule to the series before a method
sees them (RCAEval `main.py` at the pinned commit: `±inf → NaN`, forward fill, then 0), and
counts the cells it filled — the exporter leaves an empty bin's latency cells empty rather
than imputing a value. `collapse_to_services` is the suite's coarse-grained collapse of a
metric ranking: each service at the position of its best-ranked metric
(`main.py`: the part before the first `_`, first occurrence kept).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    CASE_DIR_GLOB,
    CASE_FILES,
    DATA_CSV,
    METRIC_SEP,
    METRICS,
    SERVICES_JSON,
    TIME_COLUMN,
    TOPOLOGY_JSON,
    WINDOW_JSON,
    WINDOW_KEYS,
)


class NotCaseReadable(PermissionError):
    pass


def list_cases(export_dir):
    """The case directories of one export, in position order (`case-00`, `case-01`, …)."""
    export_dir = Path(export_dir)
    dirs = sorted(p for p in export_dir.glob(CASE_DIR_GLOB) if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no case directories under {export_dir}")
    return dirs


def open_case_file(case_dir, name, mode="r"):
    """Open one of the three case files on behalf of a method; refuses everything else."""
    case_dir = Path(case_dir).resolve()
    if name not in CASE_FILES:
        raise NotCaseReadable(f"{name!r} is not a case file; a method may read only {CASE_FILES}")
    target = (case_dir / name).resolve()
    if target.parent != case_dir:
        raise NotCaseReadable(f"{name!r} escapes the case directory")
    return open(target, mode, encoding=None if "b" in mode else "utf-8")


def read_window(case_dir):
    with open_case_file(case_dir, WINDOW_JSON) as f:
        window = json.load(f)
    if set(window) != set(WINDOW_KEYS):
        raise ValueError(f"{WINDOW_JSON} must carry exactly {WINDOW_KEYS}, got {sorted(window)}")
    return {k: int(window[k]) for k in WINDOW_KEYS}


def read_services(case_dir):
    """`[(name, kind)]` of the deployed services, in the export's (name) order."""
    with open_case_file(case_dir, SERVICES_JSON) as f:
        doc = json.load(f)
    return [(str(s["name"]), str(s["kind"])) for s in doc["services"]]


def read_topology(case_dir):
    """`[(caller, callee, p_call)]` of the exporter's `topology.json` (the deployed-service call edges, caller →
    callee as stored); only an arm that declares the `topology` input may call this. Refuses an edge that names a
    service outside `services.json`, a within-service edge, or a repeated pair."""
    names = {s for s, _ in read_services(case_dir)}
    with open_case_file(case_dir, TOPOLOGY_JSON) as f:
        doc = json.load(f)
    edges, seen = [], set()
    for e in doc["edges"]:
        a, b = str(e["caller"]), str(e["callee"])
        if a not in names or b not in names or a == b or (a, b) in seen:
            raise ValueError(f"{TOPOLOGY_JSON}: bad edge {a!r} -> {b!r}")
        seen.add((a, b))
        edges.append((a, b, float(e["p_call"])))
    return edges


def suite_fill(df):
    """RCAEval `main.py`'s loading rule at the pinned commit: `±inf → NaN`, forward fill, then 0.
    Returns the filled frame and the number of cells the rule changed."""
    before = df.isna().sum().sum() + np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    out = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return out, int(before)


def read_table(case_dir, fill=True):
    """The case's series as the suite's loader would hand them to a method: a DataFrame with the
    `time` column and one column per `<service>_<metric>`; with `fill`, the suite's rule applied.
    Returns `(frame, n_filled)`."""
    with open_case_file(case_dir, DATA_CSV) as f:
        df = pd.read_csv(f)
    if TIME_COLUMN not in df.columns:
        raise ValueError(f"{DATA_CSV} has no {TIME_COLUMN!r} column")
    expected = {f"{s}{METRIC_SEP}{m}" for s, _ in read_services(case_dir) for m in METRICS}
    present = set(df.columns) - {TIME_COLUMN}
    if present != expected:
        raise ValueError(
            f"{DATA_CSV} columns differ from the services × metrics grid: "
            f"missing {sorted(expected - present)}, extra {sorted(present - expected)}"
        )
    if not fill:
        return df, 0
    return suite_fill(df)


def split_at(df, inject_time):
    """The suite's split: rows strictly before the injection, rows at or after it."""
    return df[df[TIME_COLUMN] < inject_time], df[df[TIME_COLUMN] >= inject_time]


def service_of(column):
    """The service a metric column belongs to (the part before the first separator, as the suite does)."""
    return column.split(METRIC_SEP, 1)[0]


def collapse_to_services(metric_ranking):
    """The suite's coarse-grained collapse: each service at the position of its best-ranked metric,
    first occurrence kept, nothing appended."""
    seen = []
    for col in metric_ranking:
        if col == TIME_COLUMN:
            continue
        s = service_of(col)
        if s not in seen:
            seen.append(s)
    return seen


__all__ = [
    "NotCaseReadable",
    "list_cases",
    "open_case_file",
    "read_window",
    "read_services",
    "read_topology",
    "suite_fill",
    "read_table",
    "split_at",
    "service_of",
    "collapse_to_services",
]
