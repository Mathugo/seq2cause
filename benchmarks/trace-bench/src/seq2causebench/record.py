# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Run records (PRD scenarios 22, 23, 24, and the failed-run and refuse-to-overwrite
edge cases).

Every command writes three files into `<out>/run/`:

- `arguments.json` — the command, every parsed argument (complete by
  construction: no knob has a default) and the verbatim `argv`;
- `results.json` — what the command computed;
- `run_meta.json` — provenance and hardware: the repository commit, the shipped
  package's version, torch and CUDA versions, GPU model, host RAM, CPU count,
  wall clock, peak device and resident memory, and a full environment export.
  Never a hostname, an instance type or a storage location (the repository is
  public). Provisioning facts a stage is told about (the replica file name, the
  AWS profile name) arrive through `note()` and are recorded as given.

A record is written on failure too (`status: failed`, with the traceback), so
a directory never looks complete when it is not, and a command refuses to
start into a directory that already holds `results.json`.
"""

from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
import time
import traceback as _tb
from pathlib import Path

from tracebench.record import (  # noqa: F401
    canonical_hash,
    canonical_json,
    read_json,
    sha256_file,
    write_json,
)

from . import __version__
from .constants import ARGUMENTS_JSON, RESULTS_JSON, RUN_DIR, RUN_META_JSON, TOOL_NAME
from .log import now_iso

PACKAGE_DIR = Path(__file__).resolve().parent
ENV_GIT_COMMIT = "SEQ2CAUSEBENCH_GIT_COMMIT"

# The fields of PRD scenario 24 that live in run_meta.json (the seed and the
# invocation live in arguments.json; the corpus provenance and the results in
# results.json). `test_record_fields` asserts every one is present and non-empty.
META_FIELDS = (
    "git_commit",
    "torch",
    "cuda",
    "gpu_name",
    "host_ram_gb",
    "n_cpu",
    "wall_clock_s",
    "peak_device_bytes",
    "peak_rss_bytes",
    "environment_export",
)


def repo_root(start=PACKAGE_DIR):
    """The git toplevel that holds the package (the harness lives in a
    subdirectory of the method's repository), or the package's grandparent."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out:
            return Path(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return Path(start).resolve().parents[2]


REPO_ROOT = repo_root()


def git_commit(repo=None):
    """The repository's commit (and whether the tree was dirty); an environment
    override for installs that carry no `.git`."""
    env = os.environ.get(ENV_GIT_COMMIT)
    if env:
        return {"sha": env, "dirty": None}
    repo = REPO_ROOT if repo is None else repo
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            != ""
        )
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": "unknown", "dirty": None}


def environment_export():
    """`pip freeze` of the executing interpreter, as a list of requirement lines."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return lines or ["<empty pip freeze>"]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [f"<pip freeze unavailable: {type(e).__name__}>"]


def host_ram_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page / 1e9, 2)
    except (ValueError, OSError, AttributeError):
        return -1.0


def peak_rss_bytes():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return int(ru if sys.platform == "darwin" else ru * 1024)


def device_facts():
    """torch / CUDA versions, the GPU model and peak allocated device memory;
    torch is imported lazily so record-only commands work without it."""
    try:
        import torch
    except ImportError:
        return {
            "torch": "not installed",
            "cuda": "none",
            "gpu_name": "none",
            "peak_device_bytes": 0,
        }
    if torch.cuda.is_available():
        return {
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "unknown",
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        }
    return {"torch": torch.__version__, "cuda": "none", "gpu_name": "none", "peak_device_bytes": 0}


def shipped_package_version():
    """The installed seq2cause version (the method under test), or why it is absent."""
    try:
        from importlib.metadata import version

        return version("seq2cause")
    except Exception as e:  # noqa: BLE001 - any failure is reported, not raised
        return f"<not installed: {type(e).__name__}>"


def tool_versions():
    import numpy
    import pyarrow
    import tracebench

    return {
        "tool": TOOL_NAME,
        "tool_version": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "tracebench": tracebench.__version__,
        "seq2cause": shipped_package_version(),
    }


class RunRecord:
    """Writes the three record files of one run into `<out_dir>/run/`.

    Use as a context manager so a failing command still leaves a record:

        with RunRecord(out, "pretrain", vars(args)) as rec:
            ...
            rec.finish(results)
    """

    def __init__(self, out_dir, command, arguments, refuse_complete=True):
        self.out_dir = Path(out_dir)
        self.run_dir = self.out_dir / RUN_DIR
        if refuse_complete and (self.run_dir / RESULTS_JSON).exists():
            raise FileExistsError(
                f"{self.run_dir / RESULTS_JSON} exists: this directory already holds a completed run; "
                "a replay writes to a new directory (PRD scenario 23)"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.command = command
        self.started = now_iso()
        self._t0 = time.monotonic()
        self._meta_extra = {}
        self._finished = False
        write_json(
            self.run_dir / ARGUMENTS_JSON,
            {"command": command, "arguments": _plain(arguments), "argv": list(sys.argv)},
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def note(self, **facts):
        """Facts for run_meta.json (never results.json)."""
        self._meta_extra.update(facts)

    def meta(self, status, traceback=None):
        meta = {
            "command": self.command,
            "status": status,
            "started": self.started,
            "finished": now_iso(),
            "wall_clock_s": round(time.monotonic() - self._t0, 3),
            "git_commit": git_commit(),
            "host_ram_gb": host_ram_gb(),
            "n_cpu": os.cpu_count() or -1,
            "peak_rss_bytes": peak_rss_bytes(),
            "platform": f"{platform.system()} {platform.machine()}",
            "environment_export": environment_export(),
            **device_facts(),
            **tool_versions(),
            **self._meta_extra,
        }
        if traceback is not None:
            meta["traceback"] = traceback
        return meta

    def finish(self, results, status="ok", traceback=None):
        write_json(self.run_dir / RESULTS_JSON, {"status": status, **_plain(results)})
        meta = self.meta(status, traceback)
        write_json(self.run_dir / RUN_META_JSON, meta)
        self._finished = True
        return meta

    def fail(self, exc):
        return self.finish(
            {"error": f"{type(exc).__name__}: {exc}"},
            status="failed",
            traceback="".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and not self._finished:
            self.fail(exc)
        return False


def _plain(obj):
    """Arguments as plain JSON (paths as strings, tuples as lists)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "__fspath__"):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj
