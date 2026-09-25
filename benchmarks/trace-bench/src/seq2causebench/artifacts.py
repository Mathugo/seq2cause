# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Binary run artifacts live in object storage, never in git.

    python -m seq2causebench.artifacts push   <run-dir> --s3-uri <prefix>
    python -m seq2causebench.artifacts pull   <run-dir> --s3-uri <prefix> [--only-records]
    python -m seq2causebench.artifacts verify <run-dir>
    python -m seq2causebench.artifacts status <run-dir>

Shells out to the AWS CLI (the lab's convention; no boto3 dependency). The
object-store prefix comes only from the command line or the
SEQ2CAUSEBENCH_S3_URI environment variable, and the manifest `push` writes
(`artifacts.json`) names every file's size and sha256 but **no location**, so
no bucket name can reach the public repository even if a run directory's
record is committed (PRD non-negotiables 4 and 6).

`push` is what the execution runner calls after its own sync: it re-syncs the
folder (idempotent), writes the manifest from the files that actually exist,
and copies the manifest up beside them, because a public repository never
pushes a results branch and the record must ride the prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .constants import ARTIFACTS_JSON, ARTIFACTS_SCHEMA
from .record import read_json, sha256_file, write_json

ENV_URI = "SEQ2CAUSEBENCH_S3_URI"
RECORD_SUFFIXES = (".json",)


def _run(cmd, check=True):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {proc.stderr.strip()[:400]}")
    return proc


def local_files(folder):
    """Every file under the folder except the manifest itself, with size and sha256."""
    folder = Path(folder)
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name != ARTIFACTS_JSON and "__pycache__" not in p.parts:
            out[p.relative_to(folder).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
    return out


def build_manifest(folder):
    folder = Path(folder)
    return {"schema": ARTIFACTS_SCHEMA, "run_name": folder.name, "files": local_files(folder)}


def push(folder, s3_uri, profile=None):
    folder = Path(folder)
    uri = s3_uri.rstrip("/")
    prof = ["--profile", profile] if profile else []
    _run(
        [
            "aws",
            "s3",
            "sync",
            str(folder),
            uri,
            "--no-progress",
            "--exclude",
            "__pycache__/*",
            "--exclude",
            "*.pyc",
            *prof,
        ]
    )
    manifest = build_manifest(folder)
    path = write_json(folder / ARTIFACTS_JSON, manifest)
    _run(["aws", "s3", "cp", str(path), f"{uri}/{ARTIFACTS_JSON}", "--no-progress", *prof])
    return manifest


def pull(folder, s3_uri, profile=None, only_records=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prof = ["--profile", profile] if profile else []
    cmd = ["aws", "s3", "sync", s3_uri.rstrip("/"), str(folder), "--no-progress", *prof]
    if only_records:
        cmd += ["--exclude", "*"] + [x for s in RECORD_SUFFIXES for x in ("--include", f"*{s}")]
    _run(cmd)
    return verify(folder, only_records=only_records)


def verify(folder, only_records=False):
    """Missing or changed files against the manifest; with `only_records`, only
    the record files are expected to be present."""
    folder = Path(folder)
    m = read_json(folder / ARTIFACTS_JSON)
    problems = []
    for rel, f in sorted(m["files"].items()):
        if only_records and not rel.endswith(RECORD_SUFFIXES):
            continue
        p = folder / rel
        if not p.exists():
            problems.append(f"missing {rel}")
        elif p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            problems.append(f"changed {rel}")
    return problems


def status(folder):
    folder = Path(folder)
    have = local_files(folder)
    m = read_json(folder / ARTIFACTS_JSON) if (folder / ARTIFACTS_JSON).exists() else None
    return {
        "local_files": len(have),
        "local_bytes": sum(f["bytes"] for f in have.values()),
        "manifest": None if m is None else {"run_name": m["run_name"], "n_files": len(m["files"])},
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # transport knobs: where bytes live, never how the method behaves (test_no_defaults allows these two)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("push", "pull"):
        sp = sub.add_parser(name)
        sp.add_argument("folder")
        sp.add_argument(
            "--s3-uri",
            default=os.environ.get(ENV_URI),
            help=f"object-store prefix (or {ENV_URI}); never read from the repository",
        )
        if name == "pull":
            sp.add_argument(
                "--only-records", action="store_true", help="fetch the small record files only"
            )
    for name in ("verify", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("folder")
        if name == "verify":
            sp.add_argument("--only-records", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd in ("push", "pull") and not args.s3_uri:
        print(
            f"an object-store prefix is required (--s3-uri or {ENV_URI}); it is never read from the repository",
            file=sys.stderr,
        )
        return 2
    if args.cmd == "push":
        m = push(args.folder, args.s3_uri, args.profile)
        print(json.dumps({"pushed": len(m["files"]), "run_name": m["run_name"]}))
        return 0
    if args.cmd == "pull":
        problems = pull(args.folder, args.s3_uri, args.profile, only_records=args.only_records)
        print(json.dumps({"problems": problems}))
        return 0 if not problems else 1
    if args.cmd == "verify":
        problems = verify(args.folder, only_records=args.only_records)
        print(json.dumps({"problems": problems}))
        return 0 if not problems else 1
    print(json.dumps(status(args.folder)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
