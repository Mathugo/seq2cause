# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""`pull --tier rca`: exactly the files this directory reads land as real files under the host
layout, each verified against the corpus manifest; the views, the client-side shards, the
oracle, the token-level target and the two unread label files never land; a corpus that does
not verify, carries the wrong tool version or arrives as symlinks is refused. The dataset host
is stood in for by a local copy."""

import shutil
from pathlib import Path

from fixture_corpus import fixture_corpus

from tracebenchrca import pull as pull_mod
from tracebenchrca.constants import RESULTS_JSON, RUN_DIR
from tracebenchrca.record import read_json, write_json


def _fake_download(src):
    """A stand-in for snapshot_download that copies the allowed patterns from a local corpus."""

    def download(repo_id, repo_type, revision, allow_patterns, local_dir):
        assert repo_type == "dataset"
        prefix = allow_patterns[0].rsplit("/", 1)[0]  # the first pattern is `<prefix>/config.yaml`
        rel_patterns = [p[len(prefix) + 1 :] for p in allow_patterns]
        for p in sorted(Path(src).rglob("*")):
            if p.is_file():
                rel = p.relative_to(src).as_posix()
                if pull_mod._matches(rel, rel_patterns):
                    dst = Path(local_dir) / prefix / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dst)
        return str(local_dir)

    return download


def test_rca_tier_pulls_exactly_what_the_directory_reads(tmp_path):
    res = pull_mod.pull("x/y", "v0.3.0", "xs", "latent", 0, "rca", tmp_path, download=_fake_download(fixture_corpus()))
    assert res["problems"] == []
    d = Path(res["corpus_dir"])
    assert d == tmp_path / "instances/xs/latent/seed=0"
    present = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert present == [
        "COMPLETE",
        "config.yaml",
        "graphs/alphabet.json",
        "graphs/views/service.json",
        "instantiation.json",
        "labels/cases.json",
        "manifest.json",
        "raw/shard=0000/vl.jsonl.gz",
        "raw/shard=0001/vl.jsonl.gz",
        "topology/callgraph.json",
        "topology/prior.json",
    ]
    assert res["n_files_checked"] == 9 and res["manifest"]["tool_version"] == "0.3.0"
    assert all(not p.is_symlink() for p in d.rglob("*"))


def test_corruption_symlink_version_and_unread_labels_are_refused(tmp_path):
    dl = _fake_download(fixture_corpus())
    res = pull_mod.pull("x/y", "v0.3.0", "xs", "latent", 0, "rca", tmp_path, download=dl)
    d = Path(res["corpus_dir"])
    cases = d / "labels/cases.json"
    cases.write_text(cases.read_text() + " ")
    _, _, problems = pull_mod.verify_pulled(d, "rca")
    assert problems == ["changed labels/cases.json"]
    real = d / "graphs/alphabet.json"
    moved = tmp_path / "moved.json"
    shutil.move(real, moved)
    real.symlink_to(moved)
    _, _, problems = pull_mod.verify_pulled(d, "rca")
    assert any(p.startswith("symlink graphs/alphabet.json") for p in problems)
    (d / "labels/faults.json").write_text("{}")
    _, _, problems = pull_mod.verify_pulled(d, "rca")
    assert "never fetched but present: labels/faults.json" in problems
    m = read_json(d / "manifest.json")
    write_json(d / "manifest.json", dict(m, tool_version="0.2.3"))
    _, _, problems = pull_mod.verify_pulled(d, "rca")
    assert any("tool_version" in p for p in problems)


def test_main_writes_a_record_and_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_mod, "snapshot_download", _fake_download(fixture_corpus()))
    out = tmp_path / "run"
    argv = [
        "--repo",
        "x/y",
        "--revision",
        "v0.3.0",
        "--rung",
        "xs",
        "--variant",
        "latent",
        "--seed",
        "0",
        "--tier",
        "rca",
        "--dest",
        str(tmp_path / "corpora"),
        "--output-folder",
        str(out),
    ]
    assert pull_mod.main(argv) == 0
    assert read_json(out / RUN_DIR / RESULTS_JSON)["problems"] == []
    argv[-1] = str(out / "again")
    assert pull_mod.main(argv) == 0  # a resume into a new record folder
