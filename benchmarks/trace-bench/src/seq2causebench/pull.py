# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Fetch one published corpus from the dataset host and verify it against its manifest.

    python -m seq2causebench.pull --repo chadyuk/trace-bench --revision v0.3.0 \
        --rung s --variant latent --seed 0 --tier method --dest <root> --output-folder <run-dir>

Tiers:
- `method` — `views/**`, `manifest.json`, `COMPLETE`: everything a method
  process needs. Every fetched corpus file must be flagged `method_readable`
  in the manifest (the manifest and the marker are the two unflagged
  exceptions, read here for verification and provenance only).
- `score` — the method tier plus `graphs/**`: what the benchmark's scorer
  needs. Pulled only after the method process has exited.

Files land as real files under `<dest>/instances/<rung>/<variant>/seed=<k>/`
(never symlinks: the benchmark's accessor resolves paths). Every fetched file's
size and sha256 are checked against the manifest; a corpus whose manifest does
not verify, or whose tool version is not the one this repository scores, is
refused. Re-running resumes: files already present with the right hash are
not fetched again.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from huggingface_hub import snapshot_download

from .constants import (
    BENCHMARK_TOOL_VERSION,
    COMPLETE_MARKER,
    GRAPHS_DIR,
    MANIFEST_JSON,
    RUNGS,
    SEEDS,
    VARIANTS,
    VIEWS_DIR,
)
from .log import log
from .record import RunRecord, read_json, sha256_file

TIERS = {
    "method": (f"{VIEWS_DIR}/**", MANIFEST_JSON, COMPLETE_MARKER),
    "score": (f"{VIEWS_DIR}/**", f"{GRAPHS_DIR}/**", MANIFEST_JSON, COMPLETE_MARKER),
}
UNFLAGGED = (MANIFEST_JSON, COMPLETE_MARKER)


def corpus_prefix(rung, variant, seed):
    return f"instances/{rung}/{variant}/seed={seed}"


def tier_patterns(tier, prefix):
    return [f"{prefix}/{p}" for p in TIERS[tier]]


def _matches(rel, patterns):
    """A tier pattern is either a directory prefix (`views/**`) or an exact file name."""
    for p in patterns:
        if p.endswith("/**"):
            if rel.startswith(p[:-2]):
                return True
        elif rel == p or fnmatch.fnmatchcase(rel, p):
            return True
    return False


def verify_pulled(corpus_dir, tier):
    """Check a pulled corpus against its own manifest for the tier's files.
    Returns `(manifest, checked_files, problems)`."""
    corpus_dir = Path(corpus_dir)
    problems = []
    mpath = corpus_dir / MANIFEST_JSON
    if not mpath.exists():
        return None, [], [f"missing {MANIFEST_JSON}"]
    if mpath.is_symlink():
        problems.append(f"{MANIFEST_JSON} is a symlink; the accessor needs real files")
    manifest = read_json(mpath)
    if manifest.get("tool_version") != BENCHMARK_TOOL_VERSION:
        problems.append(
            f"tool_version {manifest.get('tool_version')!r} != {BENCHMARK_TOOL_VERSION!r}"
        )
    if not (corpus_dir / COMPLETE_MARKER).exists():
        problems.append(f"missing {COMPLETE_MARKER}")
    patterns = list(TIERS[tier])
    checked = []
    for f in manifest["files"]:
        rel = f["path"]
        if not _matches(rel, patterns):
            continue
        p = corpus_dir / rel
        if not p.exists():
            problems.append(f"missing {rel}")
            continue
        if p.is_symlink():
            problems.append(f"symlink {rel}")
        if p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            problems.append(f"changed {rel}")
        if tier == "method" and rel not in UNFLAGGED and not f["method_readable"]:
            problems.append(f"not method-readable but fetched: {rel}")
        checked.append(rel)
    # nothing outside the tier may be present on a method-tier pull (the graphs must not be on disk)
    if tier == "method":
        for p in sorted(corpus_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(corpus_dir).as_posix()
                if not _matches(rel, patterns) and rel not in UNFLAGGED:
                    problems.append(f"outside the method tier: {rel}")
    return manifest, checked, problems


def pull(repo, revision, rung, variant, seed, tier, dest, download=None):
    """`download` stands in for the dataset-host client in tests; resolved at call time."""
    download = download or snapshot_download
    prefix = corpus_prefix(rung, variant, seed)
    patterns = tier_patterns(tier, prefix)
    log({"event": "pull_start", "repo": repo, "revision": revision, "corpus": prefix, "tier": tier})
    download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=str(dest),
    )
    corpus_dir = Path(dest) / prefix
    manifest, checked, problems = verify_pulled(corpus_dir, tier)
    for pr in problems:
        log({"event": "pull_problem", "problem": pr})
    result = {
        "corpus": prefix,
        "corpus_dir": str(corpus_dir),
        "tier": tier,
        "n_files_checked": len(checked),
        "problems": problems,
        "manifest": None
        if manifest is None
        else {
            k: manifest.get(k)
            for k in (
                "tool_version",
                "config_hash",
                "instance",
                "variant",
                "seed",
                "alphabet_size_vocab",
                "alphabet_size_realized_train",
                "alphabet_size_potential",
            )
        },
    }
    log(
        {
            "event": "pull_done",
            "corpus": prefix,
            "tier": tier,
            "n_files_checked": len(checked),
            "ok": not problems,
        }
    )
    return result


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", required=True, help="dataset host id, e.g. chadyuk/trace-bench")
    p.add_argument(
        "--revision", required=True, help="the release tag whose corpora this repository scores"
    )
    p.add_argument("--rung", required=True, choices=RUNGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--tier", required=True, choices=sorted(TIERS))
    p.add_argument(
        "--dest",
        required=True,
        help="local root; the corpus lands under instances/<rung>/<variant>/seed=<k>/",
    )
    p.add_argument(
        "--output-folder", required=True, help="where this command's run record is written"
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "pull", vars(args)) as rec:
        res = pull(
            args.repo, args.revision, args.rung, args.variant, args.seed, args.tier, args.dest
        )
        rec.finish(res, status="ok" if not res["problems"] else "failed")
    return 0 if not res["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
