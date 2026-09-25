# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""PRD scenario 24 (the reproducibility fields, none empty), scenario 22
(arguments reconstruct the run), and the failed-run, replay and
refuse-to-overwrite edge cases."""

import json
import platform
from pathlib import Path

import pytest

from seq2causebench.constants import ARGUMENTS_JSON, RESULTS_JSON, RUN_DIR, RUN_META_JSON
from seq2causebench.record import META_FIELDS, RunRecord, read_json


def _empty(v):
    return v is None or v == "" or v == [] or v == {}


def test_run_meta_carries_every_reproducibility_field(tmp_path):
    args = {"corpus": str(tmp_path / "corpus"), "seed": 7, "lr": 3e-4, "grid": (1, 2)}
    with RunRecord(tmp_path, "unit", args) as rec:
        rec.note(tok_pos_per_s=1234.5)
        rec.finish({"answer": 42})
    run = tmp_path / RUN_DIR
    arguments = read_json(run / ARGUMENTS_JSON)
    assert arguments["command"] == "unit"
    assert arguments["arguments"] == {
        "corpus": str(tmp_path / "corpus"),
        "seed": 7,
        "lr": 3e-4,
        "grid": [1, 2],
    }
    assert isinstance(arguments["argv"], list) and arguments["argv"]
    results = read_json(run / RESULTS_JSON)
    assert results == {"status": "ok", "answer": 42}
    meta = read_json(run / RUN_META_JSON)
    for field in META_FIELDS:
        assert field in meta, field
        assert not _empty(meta[field]), field
    assert (
        meta["status"] == "ok"
        and meta["tool"] == "seq2cause-trace-bench"
        and meta["tok_pos_per_s"] == 1234.5
    )
    assert meta["git_commit"]["sha"]
    assert meta["wall_clock_s"] >= 0 and meta["n_cpu"] >= 1 and meta["peak_rss_bytes"] > 0
    assert isinstance(meta["environment_export"], list) and meta["environment_export"]
    # never a hostname, an instance type or a storage location
    text = (run / RUN_META_JSON).read_text()
    assert "host" not in meta or meta.get("host") is None
    assert platform.node() not in text
    assert "s3://" not in text and "instance_type" not in text


def test_failed_run_still_writes_a_record(tmp_path):
    with pytest.raises(RuntimeError):
        with RunRecord(tmp_path, "unit", {"seed": 1}):
            raise RuntimeError("boom at step 3")
    meta = read_json(tmp_path / RUN_DIR / RUN_META_JSON)
    assert meta["status"] == "failed"
    assert "boom at step 3" in meta["traceback"]
    results = read_json(tmp_path / RUN_DIR / RESULTS_JSON)
    assert results["status"] == "failed" and "boom" in results["error"]


def test_refuses_a_directory_that_holds_a_completed_run(tmp_path):
    with RunRecord(tmp_path, "unit", {"seed": 1}) as rec:
        rec.finish({})
    with pytest.raises(FileExistsError):
        RunRecord(tmp_path, "unit", {"seed": 2})
    # a replay goes to a new directory
    with RunRecord(tmp_path / "replay", "unit", {"seed": 2}) as rec:
        rec.finish({})
    assert (tmp_path / "replay" / RUN_DIR / RESULTS_JSON).exists()


def test_arguments_are_plain_json(tmp_path):
    with RunRecord(tmp_path, "unit", {"path": Path("/x/y"), "n": 3}) as rec:
        rec.finish({})
    raw = json.loads((tmp_path / RUN_DIR / ARGUMENTS_JSON).read_text())
    assert raw["arguments"] == {"path": "/x/y", "n": 3}
