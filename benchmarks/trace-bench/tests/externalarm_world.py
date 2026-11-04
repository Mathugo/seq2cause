"""A shared builder for the external family's per-method tests (plans/cascade.md addendum
2026-11-03 (later), D-SB-29): one method's topology arm on the fixture corpus end to end through a stub
of the method placed under `--method-src` — validation read (every setting of the grid) →
`scoresweep` on the floor grid → the method's own external freeze in a git repository → the test
read under it (one arm, the frozen setting) → `annotate`. Each `tests/test_externalarm_<method>.py`
writes its stub (the files the method's runner imports) and asserts on what comes back."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fixture_corpus import fixture_corpus

from seq2causebench import annotate as an
from seq2causebench import externalarm as ext
from seq2causebench import freeze as fz
from seq2causebench import scoresweep as ss
from seq2causebench.constants import EXTERNAL_METHODS, GRAINS, VAL_TABLE_JSON
from seq2causebench.record import read_json, write_json

FLOOR_TAUS = ["0", "0.1", "0.5", "0.9"]


def git(args, cwd, env=None):
    e = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x",
        **(env or {}),
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=e
    ).stdout.strip()


def write_stub(root, files):
    """`files`: `{relative path: text}` under `root/stub` (the method's `--method-src`)."""
    src = Path(root) / "stub"
    for rel, text in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return src


def external(
    corpus,
    split,
    grain,
    out,
    ledger,
    method,
    method_src,
    arms,
    settings,
    method_args=(),
    cells=("none",),
    freeze="",
    ordering="end",
    max_len="24",
):
    return ext.main(
        [
            "--corpus",
            str(corpus),
            "--method",
            method,
            "--arms",
            *arms,
            "--ordering",
            ordering,
            "--grain",
            grain,
            "--split",
            split,
            "--rung",
            "xs",
            "--num-sequences",
            "0",
            "--sequence-sample",
            "head",
            "--max-len",
            str(max_len),
            "--seed",
            "0",
            "--settings",
            *[str(s) for s in settings],
            "--method-args",
            *[str(a) for a in method_args],
            "--cells",
            *cells,
            "--freeze",
            str(freeze),
            "--prior-rung-record",
            "smallest-rung",
            "--staging-ledger",
            str(ledger),
            "--replica",
            "local-test",
            "--aws-profile-name",
            "none",
            "--method-python",
            sys.executable,
            "--method-src",
            str(method_src),
            "--output-folder",
            str(out),
        ]
    )


def scoresweep(corpus, sweep_dir, grain, out, arms, floor_taus=FLOOR_TAUS):
    return ss.main(
        [
            "--corpus",
            str(corpus),
            "--sweep-dir",
            str(sweep_dir),
            "--grains",
            grain,
            "--taus",
            "--granger-taus",
            "--quantiles",
            "--floor-taus",
            *floor_taus,
            "--arms",
            *arms,
            "--output-folder",
            str(out),
        ]
    )


def build_world(root, method, stub_files, settings=None, method_args=(), max_len="24"):
    """Run the method's whole chain on the fixture; returns the paths and the freeze document."""
    root = Path(root)
    spec = EXTERNAL_METHODS[method]
    arms = list(spec["arms"])
    settings = [str(s) for s in (settings or spec["grid"])]
    corpus = fixture_corpus("latent")
    stub = write_stub(root, stub_files)
    ledger = root / "staging-ledger.json"
    write_json(ledger, {"schema": "seq2causebench/staging-ledger@1", "entries": []})
    for grain in GRAINS:
        assert (
            external(
                corpus,
                "val",
                grain,
                root / f"val-{grain}",
                ledger,
                method,
                stub,
                arms,
                settings,
                method_args,
                max_len=max_len,
            )
            == 0
        )
        assert scoresweep(corpus, root / f"val-{grain}", grain, root / f"ss-{grain}", arms) == 0
    repo = root / "repo"
    repo.mkdir()
    git(["init", "-q"], repo)
    assert (
        fz.main(
            [
                "--val-tables",
                *[str(root / f"ss-{g}" / VAL_TABLE_JSON) for g in GRAINS],
                "--pretrain-results",
                "none",
                "--rung",
                "xs",
                "--variant",
                "latent",
                "--seed",
                "0",
                "--freezes-dir",
                str(repo / "freezes"),
                "--output-folder",
                str(root / "freeze-run"),
            ]
        )
        == 0
    )
    freeze = next((repo / "freezes").glob(f"*-xs-latent-s0-{spec['freeze_suffix']}.json"))
    git(["add", "-A"], repo)
    git(
        ["commit", "-q", "-m", "external freeze"],
        repo,
        env={
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        },
    )
    doc = read_json(freeze)
    results = root / "results"
    for grain in GRAINS:
        for arm in arms:
            key = f"{arm}/none/frozen/{grain}"
            cell = doc["cells"][key]
            out = root / f"test-{arm.replace('/', '-')}-{grain}"
            assert (
                external(
                    corpus,
                    "test",
                    grain,
                    out,
                    ledger,
                    method,
                    stub,
                    [arm],
                    [str(cell["N"])],
                    method_args,
                    cells=[f"{key}={cell['tau']}"],
                    freeze=freeze,
                    max_len=max_len,
                )
                == 0
            )
            assert (
                an.main(
                    [
                        "--corpus",
                        str(corpus),
                        "--run-dir",
                        str(out),
                        "--cell",
                        key,
                        "--pretrain-results",
                        "none",
                        "--per-lag-files",
                        "0",
                        "--output-folder",
                        str(results / "xs" / "latent" / "seed=0" / key),
                    ]
                )
                == 0
            )
    return {
        "root": root,
        "corpus": corpus,
        "stub": stub,
        "ledger": ledger,
        "repo": repo,
        "freeze": freeze,
        "doc": doc,
        "results": results,
        "arms": arms,
        "settings": settings,
        "method": method,
    }
