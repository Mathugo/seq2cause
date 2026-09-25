# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Check a private provisioning replica against the real command-line parsers.

    python scripts/tfvars_check.py <replica.tfvars>
    python scripts/tfvars_check.py <replica.tfvars> --args-diff <run-dir>

The replica's `train_command` is shlex-split into `&&`-chained segments; every
segment of the form `python -m seq2causebench.<module> ...` is parsed with that
module's `build_parser()`, so a missing or misspelt knob fails here rather
than on the instance (every knob is required, none has a default). With
`--args-diff`, the parsed values are compared to a finished run's
`run/arguments.json`, so the replica and the record agree by construction.

The replica itself is never read into the repository: this script only prints
argument names and values, never the file's provisioning values.
"""

from __future__ import annotations

import importlib
import json
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CHECKED_KEYS = (
    "train_command",
    "github_repo",
    "push_results",
    "output_local_dir",
    "setup_command",
    "repo_subdir",
)
BENCH_SUBDIR = "benchmarks/trace-bench"


def read_tfvars(path):
    """A minimal reader for `key = value` lines (strings, numbers, booleans)."""
    out = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith('"'):
            # a quoted string may contain '#'; take up to the closing quote
            end = val.find('"', 1)
            while end != -1 and val[end - 1] == "\\":
                end = val.find('"', end + 1)
            val = val[1:end] if end != -1 else val[1:]
        elif val in ("true", "false"):
            val = val == "true"
        else:
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                pass
        out[key] = val
    return out


def segments(command):
    return [s.strip() for s in command.split("&&")]


def parse_segment(seg):
    argv = shlex.split(seg)
    if (
        len(argv) < 3
        or argv[0] != "python"
        or argv[1] != "-m"
        or not argv[2].startswith("seq2causebench.")
    ):
        return None
    module = importlib.import_module(argv[2])
    parser = module.build_parser()
    ns = parser.parse_args(argv[3:])
    return argv[2], vars(ns)


def check(path):
    tf = read_tfvars(path)
    problems = []
    for k in CHECKED_KEYS:
        if k not in tf:
            problems.append(f"missing {k}")
    if (
        str(tf.get("github_repo", "")).endswith("/seq2cause")
        and tf.get("push_results") is not False
    ):
        problems.append("push_results must be false for the public repository")
    if tf.get("repo_subdir") != BENCH_SUBDIR:
        problems.append(f"repo_subdir must be {BENCH_SUBDIR!r} (the harness directory)")
    parsed = []
    skipped = []
    for seg in segments(str(tf.get("train_command", ""))):
        try:
            r = parse_segment(seg)
        except SystemExit:
            problems.append(f"argparse rejected: {seg[:120]}...")
            continue
        if r is None:
            skipped.append(seg)  # e.g. the benchmark's own `python -m tracebench.score`
            continue
        parsed.append(r)
    return tf, parsed, problems, skipped


def args_diff(parsed, run_dir):
    """The record's arguments against the replica segment of the same module; when a module
    appears in several segments (the two grains' sweeps, the two pull tiers, the two
    scoresweeps) the segment whose `output_folder` equals the record's is the one compared."""
    rec = json.loads((Path(run_dir) / "run" / "arguments.json").read_text(encoding="utf-8"))
    module = "seq2causebench." + rec["command"]
    candidates = [args for mod, args in parsed if mod == module]
    if not candidates:
        return {"_": (f"no {module} segment in the replica", None)}
    same_folder = [
        a for a in candidates if a.get("output_folder") == rec["arguments"].get("output_folder")
    ]
    args = (same_folder or candidates)[0]
    return {
        k: (args.get(k), rec["arguments"].get(k))
        for k in set(args) | set(rec["arguments"])
        if args.get(k) != rec["arguments"].get(k)
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    tf, parsed, problems, skipped = check(path)
    for mod, args in parsed:
        print(f"OK {mod}: {len(args)} arguments parsed")
    for seg in skipped:
        print(f"SKIP (not a seq2causebench command, unchecked): {seg[:80]}")
    if len(argv) >= 3 and argv[1] == "--args-diff":
        diffs = args_diff(parsed, argv[2])
        if diffs:
            problems.append(f"arguments differ from the record: {diffs}")
        else:
            print("OK args-diff: the record equals the replica's command")
    for p in problems:
        print(f"PROBLEM {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
