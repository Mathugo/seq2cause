"""Tiny fixtures for the test suite, built once per process.

`fixture_corpus()` is a corpus directory in the dataset host's layout holding every file the
`rca` pull tier names (configuration, instantiation, `labels/cases.json`, two access-log shards,
the service view, the alphabet, the topology, the manifest and the completion marker) and the
files it must never fetch (`labels/faults.json`, `labels/sequence-labels.parquet`, the client-side
`sentry.jsonl.gz` shards, a view, an oracle file, the token-level target), with a manifest in the
benchmark's shape at the tool version this directory scores. Contents are placeholders: the
fixture pins the pull contract, not the benchmark's data.

`fixture_export(root)` writes an export in `tracebench.export_rca`'s shape — two case directories
of three services × four metrics on a 30 s grid — for the case-door tests; its numbers are chosen
by hand so a RobustScorer-style read has one obvious answer per case.
"""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="tracebenchrca-tests-"))

SERVICES = [("tb-edge", "bff"), ("tb-beta", "service"), ("ext-gamma", "external")]
METRICS = ("count", "error", "latency-50", "latency-95")
EPOCH = 1_767_603_600  # 2026-01-05T09:00:00Z, the xs window start
WIDTH = 30


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_gz(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(path, "wb"), mtime=0) as gz:
        for ln in lines:
            gz.write((json.dumps(ln, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return path


@functools.cache
def fixture_corpus(variant="latent", seed=0):
    d = _ROOT / f"instances/xs/{variant}/seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    _write(
        d / "config.yaml",
        "name: xs\nrun:\n  window: {start: '2026-01-05T09:00:00Z', duration_hours: 1.0}\n  shard_ticks: 900\n  tick_s: 1\n",
    )
    services = [
        {"index": i, "name": n, "kind": {"bff": 0, "service": 1, "external": 3}[k]} for i, (n, k) in enumerate(SERVICES)
    ]
    services.append({"index": 3, "name": "client", "kind": 2})
    _write(
        d / "instantiation.json",
        json.dumps(
            {
                "instance": "xs",
                "tool_version": "0.3.0",
                "config_hash": "0" * 16,
                "topology": {"services": services, "ops": [], "edges": []},
            },
            indent=1,
        ),
    )
    _write(
        d / "labels/cases.json",
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "xs_tb-beta_degrade_0",
                        "root_cause_component": "tb-beta",
                        "fault_kind": "degrade",
                        "inject_time_s": 1200.0,
                        "end_time_s": 1800.0,
                    },
                    {
                        "case_id": "xs_ext-gamma_crash_1",
                        "root_cause_component": "ext-gamma",
                        "fault_kind": "crash",
                        "inject_time_s": 2700.0,
                        "end_time_s": 3000.0,
                    },
                ]
            },
            indent=1,
        ),
    )
    _write(d / "labels/faults.json", json.dumps({"faults": []}))
    (d / "labels/sequence-labels.parquet").write_bytes(b"PAR1placeholderPAR1")
    for k in range(2):
        _write_gz(
            d / f"raw/shard={k:04d}/vl.jsonl.gz",
            [
                {
                    "kind": "vl.access",
                    "ts": f"2026-01-05T09:{k * 15:02d}:01.000Z",
                    "service": "tb-edge",
                    "path": "/v1/a/b",
                    "status": 200,
                    "request_time": 0.05,
                },
                {"kind": "vl.app", "ts": f"2026-01-05T09:{k * 15:02d}:01.000Z", "service": "tb-edge", "body": "x"},
            ],
        )
        _write_gz(
            d / f"raw/shard={k:04d}/sentry.jsonl.gz", [{"kind": "sentry.error", "ts": "2026-01-05T09:00:02.000Z"}]
        )
        _write(d / f"raw/shard={k:04d}.done", json.dumps({"shard": k}))
        (d / f"oracle/shard={k:04d}").mkdir(parents=True, exist_ok=True)
        (d / f"oracle/shard={k:04d}/spans.parquet").write_bytes(b"PAR1placeholderPAR1")
    _write(
        d / "graphs/views/service.json",
        json.dumps({"level": "service", "grain": "request", "default_floor": 0.05, "directed": [], "bidirected": []}),
    )
    _write(d / "graphs/alphabet.json", json.dumps({"tokens": []}))
    _write(d / "graphs/scoring-target.json", json.dumps({"directed": [], "bidirected": []}))
    _write(d / "topology/prior.json", json.dumps({"edges": []}))
    _write(d / "topology/callgraph.json", json.dumps({"edges": []}))
    _write(d / "views/end-request/model-vocab.json", json.dumps({"vocab_size": 4}))
    files = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "COMPLETE"):
            rel = p.relative_to(d).as_posix()
            files.append(
                {
                    "path": rel,
                    "bytes": p.stat().st_size,
                    "sha256": _sha(p),
                    "method_readable": rel.startswith(("raw/", "views/")),
                }
            )
    _write(
        d / "manifest.json",
        json.dumps(
            {
                "schema": "tracebench/manifest@1",
                "tool_version": "0.3.0",
                "config_hash": "0" * 16,
                "instance": "xs",
                "variant": variant,
                "seed": seed,
                "files": files,
            },
            indent=1,
        ),
    )
    _write(d / "COMPLETE", "")
    return d


def fixture_export(root, n_rows=(60, 40), inject_row=(40, 30)):
    """Two case directories in the exporter's shape. Case 0: `tb-beta`'s p95 jumps after the
    injection; case 1: `ext-gamma`'s count collapses and its error share rises; one empty bin
    (count 0, empty latency cells) in case 1 to exercise the loader's fill rule."""
    root = Path(root)
    dirs = []
    prev_end = 0
    for k, (n, inj) in enumerate(zip(n_rows, inject_row, strict=True)):
        d = root / f"case-{k:02d}"
        d.mkdir(parents=True, exist_ok=False)
        header = ["time"] + [f"{s}_{m}" for s, _ in SERVICES for m in METRICS]
        with open(d / "data.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            for b in range(n):
                row = [EPOCH + (prev_end + b) * WIDTH]
                post = b >= inj
                for s, _ in SERVICES:
                    count, err, p50, p95 = 20, 0.0, 0.010, 0.050
                    if k == 0 and s == "tb-beta" and post:
                        p50, p95 = 0.400, 1.500
                    if k == 1 and s == "ext-gamma" and post:
                        count, err = 2, 0.5
                        if b == inj + 1:
                            count, err, p50, p95 = 0, 0.0, "", ""
                    row += [count, err, p50, p95]
                w.writerow(row)
        inject_time = EPOCH + (prev_end + inj) * WIDTH
        end_time = EPOCH + (prev_end + n) * WIDTH
        _write(
            d / "window.json", json.dumps({"end_time": end_time, "inject_time": inject_time}, indent=1, sort_keys=True)
        )
        _write(
            d / "services.json",
            json.dumps(
                {
                    "services": [{"kind": kd, "name": s} for s, kd in SERVICES],
                    "source": "instantiation.json: the deployed services (the synthetic client excluded)",
                },
                indent=1,
                sort_keys=True,
            ),
        )
        dirs.append(d)
        prev_end += n
    return dirs
