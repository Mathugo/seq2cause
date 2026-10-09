# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The freeze assertions of a test read (PRD scenarios 13, 29), shared by `discover` and `floors`.

A test read names a committed freeze record; the commit that added it must be an ancestor of
HEAD and predate the run, the freeze must name this run's cells, τ and `(c, N, g)` exactly,
bind the model the run loaded (the literal `none` for a model-free floor, RUN.md D-SB-16), the
corpus it reads and, when the freeze records one, the view ordering it was swept on.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path


class FreezeRefusal(RuntimeError):
    pass


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def freeze_commit(freeze_path):
    """`(sha, committed_at)` of the commit that added the freeze file; refuses an uncommitted or
    modified freeze."""
    path = Path(freeze_path).resolve()
    if not path.exists():
        raise FreezeRefusal(f"freeze {path} does not exist")
    try:
        top = Path(_git(["rev-parse", "--show-toplevel"], cwd=path.parent))
    except subprocess.CalledProcessError as e:
        raise FreezeRefusal(f"freeze {path} is not inside a git repository") from e
    rel = path.relative_to(top).as_posix()
    if _git(["status", "--porcelain", "--", rel], cwd=top):
        raise FreezeRefusal(
            f"freeze {rel} has uncommitted changes; commit it before the test read (PRD scenario 13)"
        )
    lines = _git(["log", "--diff-filter=A", "--format=%H %cI", "--", rel], cwd=top).splitlines()
    if not lines:
        raise FreezeRefusal(
            f"freeze {rel} is not committed; commit it before the test read (PRD scenario 13)"
        )
    sha, committed_at = lines[-1].split()  # the commit that first added the file
    ok = (
        subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=top).returncode == 0
    )
    if not ok:
        raise FreezeRefusal(f"freeze commit {sha[:8]} is not an ancestor of HEAD (PRD scenario 13)")
    return sha, committed_at


def _iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def assert_freeze(freeze, freeze_path, cells, c, N, g, model_sha256, corpus_id, started, ordering):
    """The freeze names this run's values exactly and predates it (scenarios 13, 29). A freeze
    written before orderings were recorded carries none and is accepted on either view."""
    sha, committed_at = freeze_commit(freeze_path)
    if not _iso(committed_at) < _iso(started):
        raise FreezeRefusal(
            f"the run started at {started} but the freeze was committed at {committed_at} (PRD scenario 13)"
        )
    for key, tau in cells.items():
        cell = freeze.get("cells", {}).get(key)
        if cell is None:
            raise FreezeRefusal(f"freeze has no cell {key}")
        want = {"c": int(c), "N": int(N), "g": int(g)}
        if tau != "shipped":
            want["tau"] = float(tau)
        got = {k: cell.get(k) for k in want}
        if got != want:
            raise FreezeRefusal(
                f"freeze cell {key} = {got} but the command line says {want} (PRD scenario 13)"
            )
    if freeze.get("model_sha256") != model_sha256:
        raise FreezeRefusal(
            f"freeze binds model {freeze.get('model_sha256')} but the loaded model hashes to {model_sha256}"
        )
    if freeze.get("corpus_id") != corpus_id:
        raise FreezeRefusal(
            f"freeze is for corpus {freeze.get('corpus_id')!r}, this run reads {corpus_id!r}"
        )
    if freeze.get("ordering") not in (None, ordering):
        raise FreezeRefusal(
            f"freeze was swept on the {freeze.get('ordering')!r} view, this run reads {ordering!r}"
        )
    return {"sha": sha, "committed_at": committed_at, "path": str(freeze_path)}


__all__ = ["FreezeRefusal", "freeze_commit", "assert_freeze"]
