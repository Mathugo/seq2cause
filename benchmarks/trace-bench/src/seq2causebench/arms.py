# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The arm registry: an arm name → the probe (forward-sharing unit of the shipped
engine) it reads from, the statistic it thresholds, the paths (shipped / fixed
algebra) it runs on and the cuts it is read out through.

A cell of the freeze, the val table and the report is keyed
`<arm>/<path>/<cut>/<grain>`; the type-level score column of an arm on a path
inside a scores table is `<arm slug>__<path>`.
"""

from __future__ import annotations

from .constants import ARMS, CUTS, GRAINS, PATH_SPEC, PATHS


def arm_spec(name):
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; registered: {sorted(ARMS)}")
    return ARMS[name]


def arm_slug(name):
    """File-name form of an arm name (`trace/cli-atomic` → `trace-cli-atomic`)."""
    return arm_spec(name) and name.replace("/", "-")


def probe_of(name):
    return arm_spec(name)["probe"]


def check_path(arm, path):
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    if path not in arm_spec(arm)["paths"]:
        raise ValueError(
            f"arm {arm!r} has no path {path!r}; its paths are {arm_spec(arm)['paths']}"
        )
    return path


def check_cut(arm, cut):
    if cut not in CUTS:
        raise ValueError(f"cut must be one of {CUTS}, got {cut!r}")
    if cut not in arm_spec(arm)["cuts"]:
        raise ValueError(f"arm {arm!r} has no cut {cut!r}; its cuts are {arm_spec(arm)['cuts']}")
    return cut


def column_of(arm, path):
    """The column in a scores table holding this arm's statistic on this path."""
    check_path(arm, path)
    return f"{arm_slug(arm)}__{path}"


def pass_of(arm, path):
    """The (probe, noise) pass whose forward produced this arm's statistic on this path."""
    check_path(arm, path)
    return probe_of(arm), PATH_SPEC[path]["noise"]


def cell_key(arm, path, cut, grain):
    if grain not in GRAINS:
        raise ValueError(f"grain must be one of {GRAINS}, got {grain!r}")
    check_path(arm, path)
    check_cut(arm, cut)
    return f"{arm}/{path}/{cut}/{grain}"


def parse_cell_key(key):
    arm, path, cut, grain = key.rsplit("/", 3)
    cell_key(arm, path, cut, grain)
    return arm, path, cut, grain


def allowed_cells(arm, grains=GRAINS):
    """Every valid cell key of an arm."""
    spec = arm_spec(arm)
    return [cell_key(arm, p, c, g) for p in spec["paths"] for c in spec["cuts"] for g in grains]


__all__ = [
    "arm_spec",
    "arm_slug",
    "probe_of",
    "check_path",
    "check_cut",
    "column_of",
    "pass_of",
    "cell_key",
    "parse_cell_key",
    "allowed_cells",
]
