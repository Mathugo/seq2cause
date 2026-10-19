# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The public repository contains no account identifier, bucket name, profile name or real
service name, and no generated corpus, export or binary run artifact is tracked.

Scope: every file under this directory that git tracks or would not ignore, plus every file
anywhere in the repository that this work has touched since the shipped commit
(`git diff --name-only <shipped>..HEAD`), so a branch cannot carry a private value into the
author's tree either."""

import re
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
SHIPPED_COMMIT = "67416a2"

# the table is the reference harness's (benchmarks/trace-bench/tests/test_repo_hygiene.py); the literals are
# assembled so that the harness's own repository-wide scan, which skips only its own file, never sees them here
FORBIDDEN = [
    (re.compile("s3:" + "//[a-z0-9]"), "S3 URI"),
    (re.compile("arn:" + "aws"), "AWS ARN"),
    (
        re.compile(r"(?<![0-9a-fA-F.])\d{12}(?![0-9a-fA-F.])"),
        "12-digit AWS account id",
    ),  # not a digit run inside a decimal or a hex hash
    (re.compile("AKIA" + "[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"\b" + "ai" + r"dev\b|--profile" + " [a-z]"), "AWS profile name"),
    (re.compile("lotus" + r"flare|\.mg" + r"mt\.|qla" + r"b0\d|dc" + r"9\d-\d"), "real host or organisation name"),
    (re.compile(r"\bl" + r"f-[a-z]+\b"), "real service name prefix"),
]
BINARY_SUFFIXES = (".parquet", ".npz", ".npy", ".gz", ".pt", ".safetensors", ".jsonl", ".csv")
TEXT_SUFFIXES = (".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".json", ".cff", ".gitignore")


def _git(*args):
    return subprocess.run(["git", *args], cwd=BENCH, check=True, capture_output=True, text=True).stdout


def _repo_root():
    return Path(_git("rev-parse", "--show-toplevel").strip())


def _scoped_files():
    """This directory's tracked-or-unignored files, plus every repository file this
    work has changed since the shipped commit."""
    root = _repo_root()
    bench = {
        BENCH / line for line in _git("ls-files", "--cached", "--others", "--exclude-standard").splitlines() if line
    }
    try:
        touched = {root / line for line in _git("diff", "--name-only", f"{SHIPPED_COMMIT}..HEAD").splitlines() if line}
    except subprocess.CalledProcessError:  # shallow clone without the shipped commit
        touched = set()
    return sorted(p for p in bench | touched if p.exists())


def test_no_private_identifiers_in_tracked_text():
    offenders = []
    root = _repo_root()
    for path in _scoped_files():
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        if path == Path(__file__).resolve() or path.name == "test_repo_hygiene.py":
            continue  # the pattern tables themselves (this file and the reference harness's)
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, what in FORBIDDEN:
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(root)}:{line}: {what} ({m.group(0)!r})")
    assert not offenders, "\n".join(offenders)


def test_no_binary_corpus_or_export_artifacts_tracked():
    tracked = [
        p
        for p in _scoped_files()
        if p.suffix in BINARY_SUFFIXES or "corpora/" in str(p) or "/data/" in str(p) or "exports/" in str(p)
    ]
    assert not tracked, [str(p) for p in tracked]


def test_gitignore_covers_generated_data():
    text = (BENCH / ".gitignore").read_text()
    for pattern in (
        "corpora/",
        "data/",
        "out/",
        "exports/",
        "*.parquet",
        "*.npz",
        "*.gz",
        "*.tfvars",
    ):
        assert pattern in text.split(), f".gitignore must list {pattern}"
    assert "tracerca" not in text and "tracecmi" not in text


def test_no_tracked_file_is_large():
    big = [(str(p), p.stat().st_size) for p in _scoped_files() if p.stat().st_size > 1_000_000]
    assert not big, big
