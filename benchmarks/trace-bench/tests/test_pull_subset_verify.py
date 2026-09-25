# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""`pull`: the tier's files land as real files under the host layout, every one
verified against the corpus manifest; a corpus that does not verify, carries
the wrong tool version, arrives as symlinks, or brings the graphs on a
method-tier pull is refused. The dataset host is stood in for by a local copy."""

import shutil
from pathlib import Path

from fixture_corpus import fixture_corpus

from seq2causebench import pull as pull_mod
from seq2causebench.constants import RESULTS_JSON, RUN_DIR
from seq2causebench.record import read_json, write_json


def _fake_download(src):
    """A stand-in for snapshot_download that copies the allowed patterns from a local corpus."""

    def download(repo_id, repo_type, revision, allow_patterns, local_dir):
        assert repo_type == "dataset"
        prefix = (
            allow_patterns[0].split("/views")[0]
            if "/views" in allow_patterns[0]
            else allow_patterns[0].rsplit("/", 1)[0]
        )
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


def test_method_tier_pulls_views_and_verifies(tmp_path):
    res = pull_mod.pull(
        "x/y",
        "v0.3.0",
        "xs",
        "latent",
        0,
        "method",
        tmp_path,
        download=_fake_download(fixture_corpus("latent")),
    )
    assert res["problems"] == []
    d = Path(res["corpus_dir"])
    assert d == tmp_path / "instances/xs/latent/seed=0"
    assert (d / "manifest.json").exists() and (d / "COMPLETE").exists()
    assert (
        not (d / "graphs").exists()
        and not (d / "topology").exists()
        and not (d / "labels").exists()
    )
    assert res["n_files_checked"] > 4 and res["manifest"]["tool_version"] == "0.3.0"
    assert all(not p.is_symlink() for p in d.rglob("*"))


def test_score_tier_adds_the_graphs(tmp_path):
    res = pull_mod.pull(
        "x/y",
        "v0.3.0",
        "xs",
        "twin",
        1,
        "score",
        tmp_path,
        download=_fake_download(fixture_corpus("twin", 1)),
    )
    assert res["problems"] == []
    d = Path(res["corpus_dir"])
    assert (d / "graphs" / "scoring-target.json").exists() and (
        d / "graphs" / "alphabet.json"
    ).exists()
    assert not (d / "labels").exists()


def test_corruption_symlink_and_version_are_refused(tmp_path):
    dl = _fake_download(fixture_corpus("latent"))
    res = pull_mod.pull("x/y", "v0.3.0", "xs", "latent", 0, "method", tmp_path, download=dl)
    d = Path(res["corpus_dir"])
    vocab = d / "views/end-request/model-vocab.json"
    vocab.write_text(vocab.read_text() + " ")
    _, _, problems = pull_mod.verify_pulled(d, "method")
    assert problems == ["changed views/end-request/model-vocab.json"]
    # a symlinked file is refused even when its content matches
    real = d / "views/end-request/export-stats.json"
    moved = tmp_path / "moved.json"
    shutil.move(real, moved)
    real.symlink_to(moved)
    _, _, problems = pull_mod.verify_pulled(d, "method")
    assert any(p.startswith("symlink views/end-request/export-stats.json") for p in problems)
    # the graphs on disk on a method-tier pull
    (d / "graphs").mkdir()
    (d / "graphs" / "alphabet.json").write_text("{}")
    _, _, problems = pull_mod.verify_pulled(d, "method")
    assert "outside the method tier: graphs/alphabet.json" in problems
    # the wrong tool version
    m = read_json(d / "manifest.json")
    write_json(d / "manifest.json", dict(m, tool_version="0.2.3"))
    _, _, problems = pull_mod.verify_pulled(d, "method")
    assert any("tool_version" in p for p in problems)


def test_main_writes_a_record_and_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_mod, "snapshot_download", _fake_download(fixture_corpus("latent")))
    out = tmp_path / "run"
    rc = pull_mod.main(
        [
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
            "method",
            "--dest",
            str(tmp_path / "corpora"),
            "--output-folder",
            str(out),
        ]
    )
    assert rc == 0
    assert read_json(out / RUN_DIR / RESULTS_JSON)["problems"] == []
    # a second pull into the same run folder is refused (completed record), a resume goes to a new folder
    rc2 = pull_mod.main(
        [
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
            "method",
            "--dest",
            str(tmp_path / "corpora"),
            "--output-folder",
            str(out / "again"),
        ]
    )
    assert rc2 == 0
