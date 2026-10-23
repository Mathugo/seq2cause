"""CausalRCA's method side: the authors' `train_latency.py` at the pin through a recorded data
substitution (D-TBR-5; `plans/rca-arms.md`, addenda 2026-10-22, later, and 2026-10-23).

Per case and series the method side reads the case through the case door, builds the registered
input (the fault window's rows, the series' columns in `services.json` order, min-max scaled per
column), substitutes the script's data block in memory, and runs `CAUSALRCA_RUNS` trainings, each in
a directory of its own, through `causalrca_runner.py` under the pinned environment's interpreter.
It reads back each run's ranking (the script's last standard-output line) and graph (`adj.npy`, the
runner's conversion of the script's pickle) and averages the absolute graph over the successful
runs. Nothing here opens a label, a graph or the topology of the corpus.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .arms import check_input
from .cases import read_services, read_table, read_window
from .constants import (
    ARM_CAUSALRCA,
    CAUSALRCA_BLOCK_END,
    CAUSALRCA_BLOCK_START,
    CAUSALRCA_COMMIT,
    CAUSALRCA_EPOCHS,
    CAUSALRCA_FILES_SHA256,
    CAUSALRCA_GRAPH_PKL,
    CAUSALRCA_PNG,
    CAUSALRCA_RUNS,
    CAUSALRCA_SCRIPT,
    INPUT_SERIES,
    INPUT_SERVICES,
    INPUT_WINDOW,
    METRIC_SEP,
    TIME_COLUMN,
)

RUNNER = Path(__file__).with_name("causalrca_runner.py")
SUBSTITUTED = "train_latency_substituted.py"
REMOVED = "removed_block.txt"
INPUT_NPY = "input.npy"
ADJ_NPY = "adj.npy"
RUNNER_JSON = "runner.json"
STDOUT_TXT = "stdout.txt"
STDERR_TXT = "stderr.txt"
# no bytecode written into the pinned checkout (its tree must stay clean for the next stage's check)
RUN_ENV = {"MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
CPU_THREAD_ENV = {**RUN_ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_source(src):
    """The pinned checkout: commit, a clean tree, and the four files the script executes or imports."""
    src = Path(src)
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if head.stdout.strip() != CAUSALRCA_COMMIT:
        raise ValueError(f"{src} is at {head.stdout.strip()!r}, not the pin {CAUSALRCA_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError(f"{src} is not a clean checkout: {dirty.splitlines()[:3]}")
    shas = {name: _sha(src / name) for name in CAUSALRCA_FILES_SHA256}
    bad = {n: s for n, s in shas.items() if s != CAUSALRCA_FILES_SHA256[n]}
    if bad:
        raise ValueError(f"{src}: files differ from the pin: {sorted(bad)}")
    return {"commit": CAUSALRCA_COMMIT, "files_sha256": shas}


def input_array(case_dir, series):
    """The registered input of one case and series: the rows at or after the injection, the series'
    columns in `services.json` order, each min-max scaled to [0, 1] (a constant column becomes 0)."""
    check_input(ARM_CAUSALRCA, INPUT_SERIES)
    check_input(ARM_CAUSALRCA, INPUT_SERVICES)
    check_input(ARM_CAUSALRCA, INPUT_WINDOW)
    df, n_filled = read_table(case_dir, fill=True)
    names = [s for s, _ in read_services(case_dir)]
    inject_time = read_window(case_dir)["inject_time"]
    post = df[df[TIME_COLUMN] >= inject_time]
    x = post[[f"{s}{METRIC_SEP}{series}" for s in names]].to_numpy(dtype=np.float64)
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = hi - lo
    scaled = np.where(span > 0, (x - lo) / np.where(span > 0, span, 1.0), 0.0)
    return scaled, names, {"n_rows": int(len(post)), "n_filled": n_filled, "n_constant_columns": int((span == 0).sum())}


def substitute(source, names, series):
    """`train_latency.py` with its data block replaced (each end asserted once); returns the
    substituted source and the removed block."""
    if source.count(CAUSALRCA_BLOCK_START) != 1 or source.count(CAUSALRCA_BLOCK_END) != 1:
        raise ValueError("the data block's ends do not each occur exactly once in the pinned script")
    i = source.index(CAUSALRCA_BLOCK_START)
    j = source.index(CAUSALRCA_BLOCK_END) + len(CAUSALRCA_BLOCK_END)
    if j <= i:
        raise ValueError("the data block's end precedes its start")
    columns = [f"{s}{METRIC_SEP}{series}" for s in names]
    block = "\n".join(
        [
            "# --- data block substituted by trace-bench-rca (D-TBR-5; plans/rca-arms.md, addenda 2026-10-22, later,",
            "# and 2026-10-23): the case's fault-window series, min-max scaled per column ---",
            f"names = {list(names)!r}",
            "idx = 0",
            f"data = pd.DataFrame(np.load({INPUT_NPY!r}), columns={columns!r})",
            f"CONFIG.epochs = {CAUSALRCA_EPOCHS}",
            "# --- end of the substituted block ---",
        ]
    )
    return source[:i] + block + source[j:], source[i:j]


def parse_ranking(stdout, n):
    """The script's last standard-output line → service indices in ranked order, or the reason the run
    failed. A valid line is a list of `(index, score)` pairs covering 0 … n-1 exactly once, with finite scores."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None, "no standard output"
    try:
        pairs = ast.literal_eval(lines[-1].strip())
    except (ValueError, SyntaxError):
        return None, "last line is not a list of (index, score) pairs"
    if not isinstance(pairs, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in pairs):
        return None, "last line is not a list of (index, score) pairs"
    idx = [p[0] for p in pairs]
    if sorted(idx) != list(range(n)):
        return None, "the ranking does not cover every service exactly once"
    if not all(isinstance(p[1], (int, float)) and math.isfinite(p[1]) for p in pairs):
        return None, "a non-finite score"
    return idx, None


def train(job):
    """One training: the runner, in its own process and directory."""
    work = Path(job["work"])
    env = {**os.environ, **job["env"]}
    base = [job["python"], str(RUNNER), "--src", job["src"], "--seed", str(job["seed"])]
    proc = subprocess.run([*base, "--mode", "train"], cwd=work, env=env, capture_output=True, text=True)
    (work / STDOUT_TXT).write_text(proc.stdout, encoding="utf-8")
    (work / STDERR_TXT).write_text(proc.stderr, encoding="utf-8")
    conv = None
    if proc.returncode == 0:  # the script's graph pickle is complete only now that its process has exited
        conv = subprocess.run([*base, "--mode", "convert"], cwd=work, env=env, capture_output=True, text=True)
    row = {"case_dir": job["case_dir"], "run": job["seed"], "seed": job["seed"], "exit_code": proc.returncode}
    if proc.returncode:
        order, reason = None, f"exit code {proc.returncode}"
    elif conv.returncode:
        order, reason = None, "the graph pickle could not be read"
    else:
        order, reason = parse_ranking(proc.stdout, job["n"])
    row["failure"] = reason
    row["ranking_indices"] = order
    row["stdout_sha256"] = _sha(work / STDOUT_TXT)
    for key, name in (
        ("graph_pkl_sha256", CAUSALRCA_GRAPH_PKL),
        ("adj_npy_sha256", ADJ_NPY),
        ("png_sha256", CAUSALRCA_PNG),
    ):
        row[key] = _sha(work / name) if (work / name).exists() else None
    if (work / RUNNER_JSON).exists():
        row["runner"] = json.loads((work / RUNNER_JSON).read_text(encoding="utf-8"))
    return row


def rank_export(case_dirs, series, src, python, work_root, workers, gpu):
    """Every case × run of one export and series. Returns per case: the services, the input's shape, the
    rankings of the successful runs (service names), the mean |adj| over them, and per run its record."""
    src_info = check_source(src)
    source = (Path(src) / CAUSALRCA_SCRIPT).read_text(encoding="utf-8")
    env = dict(RUN_ENV) if gpu else dict(CPU_THREAD_ENV)
    jobs, prepared = [], {}
    for d in case_dirs:
        arr, names, info = input_array(d, series)
        substituted, removed = substitute(source, names, series)
        prepared[d.name] = {
            "names": names,
            "input": info,
            "substituted_sha256": _sha_text(substituted),
            "removed_block_sha256": _sha_text(removed),
            "input_sha256": None,
        }
        for r in range(CAUSALRCA_RUNS):
            work = Path(work_root) / d.name / f"run-{r}"
            work.mkdir(parents=True, exist_ok=False)
            np.save(work / INPUT_NPY, arr)
            (work / SUBSTITUTED).write_text(substituted, encoding="utf-8")
            (work / REMOVED).write_text(removed, encoding="utf-8")
            prepared[d.name]["input_sha256"] = _sha(work / INPUT_NPY)
            jobs.append(
                {
                    "case_dir": d.name,
                    "work": str(work),
                    "python": str(python),
                    "src": str(Path(src).resolve()),
                    "seed": r,
                    "n": len(names),
                    "env": env,
                }
            )
    with ThreadPoolExecutor(max_workers=workers) as pool:  # each training is its own process (the runner)
        rows = list(pool.map(train, jobs))
    cases = []
    for d in case_dirs:
        p = prepared[d.name]
        runs = sorted((r for r in rows if r["case_dir"] == d.name), key=lambda r: r["run"])
        ok = [r for r in runs if r["failure"] is None]
        adjs = [np.load(Path(work_root) / d.name / f"run-{r['run']}" / ADJ_NPY) for r in ok]
        mean_abs = np.mean([np.abs(a) for a in adjs], axis=0) if adjs else None
        cases.append(
            {
                "case_dir": d.name,
                "services": p["names"],
                "input": p["input"],
                "input_sha256": p["input_sha256"],
                "substituted_sha256": p["substituted_sha256"],
                "removed_block_sha256": p["removed_block_sha256"],
                "rankings": [None if r["failure"] else [p["names"][i] for i in r["ranking_indices"]] for r in runs],
                "mean_abs_adj": None if mean_abs is None else mean_abs.tolist(),
                "n_runs": len(runs),
                "n_failed_runs": len(runs) - len(ok),
                "runs": [{k: v for k, v in r.items() if k != "ranking_indices"} for r in runs],
            }
        )
    return src_info, cases


__all__ = ["check_source", "input_array", "substitute", "parse_ranking", "train", "rank_export"]
