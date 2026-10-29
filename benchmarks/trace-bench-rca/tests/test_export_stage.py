"""The `export` stage: a thin, recorded wrapper around the benchmark's exporter — the corpus id
from the host layout, the exporter's results in the record, a width mismatch refused, and a
benchmark pin without the exporter failing with a clear message."""

import pytest
from fixture_corpus import fixture_export

from tracebenchrca import export as export_mod
from tracebenchrca.constants import RESULTS_JSON, RUN_DIR
from tracebenchrca.record import read_json


def test_corpus_id_follows_the_host_layout(tmp_path):
    d = tmp_path / "instances/m/latent/seed=3"
    d.mkdir(parents=True)
    assert export_mod.corpus_id_of(d) == "m/latent/seed=3"
    assert export_mod.corpus_id_of(tmp_path / "somewhere") == "somewhere"


class _Stub:
    """Stands in for `tracebench.export_rca`: writes the fixture export and returns an exporter-shaped result."""

    BIN_WIDTH_S = {"xs": 30, "s": 15, "m": 60}

    def __init__(self):
        self.calls = []

    EXTRAS = ("topology", "spans")

    def export(self, corpus_dir, rung, out, workers, extras=()):
        self.calls.append((str(corpus_dir), rung, str(out), workers, tuple(extras)))
        dirs = fixture_export(out)
        return {
            "rung": rung,
            "width_s": self.BIN_WIDTH_S[rung],
            "cases": [{"dir": d.name, "sha256": f"{i:064x}"} for i, d in enumerate(dirs)],
            "inputs": [],
        }


def test_stage_records_the_exporters_results(tmp_path, monkeypatch):
    stub = _Stub()
    monkeypatch.setattr(export_mod, "_exporter", lambda: stub)
    corpus = tmp_path / "instances/xs/latent/seed=0"
    corpus.mkdir(parents=True)
    out = tmp_path / "exports" / "xs-s0"
    rec_dir = tmp_path / "run-export"
    rc = export_mod.main(
        [
            "--corpus-dir",
            str(corpus),
            "--rung",
            "xs",
            "--out",
            str(out),
            "--workers",
            "1",
            "--extras",
            "--expect-cases",
            "none",
            "--output-folder",
            str(rec_dir),
        ]
    )
    assert rc == 0 and stub.calls == [(str(corpus), "xs", str(out), 1, ())]
    res = read_json(rec_dir / RUN_DIR / RESULTS_JSON)
    assert res["status"] == "ok" and res["corpus_id"] == "xs/latent/seed=0" and res["stage"] == "export"
    assert [c["dir"] for c in res["exporter"]["cases"]] == ["case-00", "case-01"]
    assert (out / "case-00" / "data.csv").exists() and res["extras"] == [] and res["expected_cases"] is None


def test_expect_cases_binds_a_re_export_to_the_landed_record(tmp_path, monkeypatch):
    """`--expect-cases` (2026-10-31): a re-export on another host must reproduce the landed export case by case."""
    stub = _Stub()
    monkeypatch.setattr(export_mod, "_exporter", lambda: stub)
    corpus = tmp_path / "instances/xs/latent/seed=0"
    corpus.mkdir(parents=True)
    landed = tmp_path / "landed"
    assert export_mod.main(["--corpus-dir", str(corpus), "--rung", "xs", "--out", str(tmp_path / "e0"), "--workers", "1",
                            "--extras", "--expect-cases", "none", "--output-folder", str(landed)]) == 0
    record = landed / RUN_DIR / RESULTS_JSON
    res = export_mod.export(corpus, "xs", tmp_path / "e1", 1, (), str(record))
    assert res["expected_cases"]["n_cases"] == 2 and res["expected_cases"]["path"] == str(record)
    doc = read_json(record)
    doc["exporter"]["cases"][0]["sha256"] = "f" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(__import__("json").dumps(doc))
    with pytest.raises(ValueError):
        export_mod.export(corpus, "xs", tmp_path / "e2", 1, (), str(tampered))


def test_extras_are_passed_through_and_refused_on_a_pin_without_them(tmp_path, monkeypatch):
    stub = _Stub()
    monkeypatch.setattr(export_mod, "_exporter", lambda: stub)
    corpus = tmp_path / "instances/xs/latent/seed=0"
    corpus.mkdir(parents=True)
    res = export_mod.export(corpus, "xs", tmp_path / "e1", 1, ("topology", "spans"))
    assert stub.calls[-1][-1] == ("topology", "spans") and res["extras"] == ["topology", "spans"]

    class Old:  # a pin whose exporter predates the extras
        BIN_WIDTH_S = _Stub.BIN_WIDTH_S

        def export(self, corpus_dir, rung, out, workers):
            raise AssertionError("must not be called")

    monkeypatch.setattr(export_mod, "_exporter", lambda: Old())
    with pytest.raises(RuntimeError, match="D-TBR-11"):
        export_mod.export(corpus, "xs", tmp_path / "e2", 1, ("spans",))


def test_width_mismatch_and_missing_exporter_are_refused(monkeypatch):
    class Wrong(_Stub):
        BIN_WIDTH_S = {"xs": 60, "s": 15, "m": 60}

    import tracebenchrca.export as mod

    monkeypatch.setattr(mod, "BIN_WIDTH_S", {"xs": 30, "s": 15, "m": 60})
    monkeypatch.setitem(__import__("sys").modules, "tracebench.export_rca", Wrong())
    with pytest.raises(RuntimeError, match="differ"):
        mod._exporter()
    monkeypatch.setitem(__import__("sys").modules, "tracebench.export_rca", None)  # import fails
    with pytest.raises(ImportError, match="pin"):
        mod._exporter()


def test_real_exporter_when_the_pin_carries_it():
    pytest.importorskip("tracebench.export_rca")
    mod = export_mod._exporter()
    assert hasattr(mod, "export") and dict(mod.BIN_WIDTH_S) == {"xs": 30, "s": 15, "m": 60}
