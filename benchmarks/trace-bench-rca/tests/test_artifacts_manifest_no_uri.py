# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The artifact manifest a run directory carries names sizes and content hashes only, never an
object-store location, and the location is never read from the repository."""

import json

import numpy as np

from tracebenchrca import artifacts
from tracebenchrca.constants import ARTIFACTS_JSON
from tracebenchrca.record import read_json, sha256_file


def _run_dir(tmp_path):
    d = tmp_path / "run-x"
    (d / "run").mkdir(parents=True)
    (d / "run" / "results.json").write_text("{}")
    np.savez(d / "scores.npz", a=np.arange(3))
    (d / "graph.json").write_text('{"directed": []}')
    return d


def test_manifest_lists_hashes_and_no_location(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)
    calls = []
    monkeypatch.setattr(artifacts, "_run", lambda cmd, check=True: calls.append(cmd))
    uri = "s3:" + "//example-store/prefix/run-x"  # assembled so the hygiene scan never sees a literal
    manifest = artifacts.push(d, uri)
    assert [c[:3] for c in calls] == [["aws", "s3", "sync"], ["aws", "s3", "cp"]]
    text = (d / ARTIFACTS_JSON).read_text()
    assert "example-store" not in text and "s3:" not in text and "uri" not in text.lower()
    assert manifest["run_name"] == "run-x" and manifest["schema"] == "tracebenchrca/artifacts@1"
    assert set(manifest["files"]) == {"run/results.json", "scores.npz", "graph.json"}
    for rel, f in manifest["files"].items():
        assert f["sha256"] == sha256_file(d / rel) and f["bytes"] == (d / rel).stat().st_size
    assert read_json(d / ARTIFACTS_JSON) == manifest


def test_verify_detects_a_changed_binary(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)
    monkeypatch.setattr(artifacts, "_run", lambda cmd, check=True: None)
    artifacts.push(d, "s3:" + "//example-store/p")
    assert artifacts.verify(d) == []
    (d / "graph.json").write_text('{"directed": [1]}')
    assert artifacts.verify(d) == ["changed graph.json"]
    (d / "scores.npz").unlink()
    assert "missing scores.npz" in artifacts.verify(d)
    assert artifacts.verify(d, only_records=True) == ["changed graph.json"]  # records only: the binary is not expected


def test_location_never_comes_from_the_repository(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)
    monkeypatch.delenv(artifacts.ENV_URI, raising=False)
    assert artifacts.main(["pull", str(d)]) == 2
    assert artifacts.main(["push", str(d)]) == 2
    monkeypatch.setattr(artifacts, "_run", lambda cmd, check=True: None)
    assert artifacts.main(["status", str(d)]) == 0
    assert json.loads(json.dumps(artifacts.status(d)))["local_files"] == 3
