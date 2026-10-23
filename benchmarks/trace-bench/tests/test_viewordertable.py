"""The view-order table (`plans/floors.md`, addendum 2026-10-22): five-seed mean, minimum and
maximum per rung, grain and ordering; an absent rung is named "not run"; fewer than five seeds,
another split or another sample is refused."""

import pytest

from seq2causebench import viewordertable as vt
from seq2causebench.record import read_json, write_json
from seq2causebench.report import ReportRefusal


def _read(root, rung, k, ordering, grain, share, **over):
    doc = {
        "corpus_id": f"{rung}/latent/seed={k}",
        "ordering": ordering,
        "grain": grain,
        "split": "test",
        "num_sequences": vt.expected_sample(rung, grain),
        "floor": 0.05,
        "share_cause_first": share,
        "edge_mean_share": share - 0.01,
        "n_sequences": 100 + k,
        "n_cooccurrences": 1000,
        "n_truth_edges_cooccurring": 40,
    }
    doc.update(over)
    write_json(root / rung / "latent" / f"seed={k}" / f"{ordering}-{grain}" / "vieworder.json", doc)


def _xs(root, seeds=range(5)):
    for k in seeds:
        for grain in ("request", "session"):
            _read(root, "xs", k, "end", grain, 0.90 + 0.01 * k)
            _read(root, "xs", k, "start", grain, 0.05 + 0.01 * k)


def test_table_five_seeds_and_not_run(tmp_path):
    res = tmp_path / "results-vieworder"
    _xs(res)
    out = tmp_path / "tables" / "vieworder-latent"
    assert (
        vt.main(
            [
                "--results-dir",
                str(res),
                "--variant",
                "latent",
                "--rungs",
                "xs",
                "s",
                "--output-folder",
                str(out),
            ]
        )
        == 0
    )
    doc = read_json(out / "vieworder-latent.json")
    rows = {(r["rung"], r["grain"], r["ordering"]): r for r in doc["rows"]}
    assert len(rows) == 8
    end = rows[("xs", "request", "end")]["share_cause_first"]
    assert (
        end["mean"] == pytest.approx(0.92)
        and end["min"] == 0.90
        and end["max"] == pytest.approx(0.94)
    )
    start = rows[("xs", "request", "start")]["share_cause_first"]
    assert start["mean"] == pytest.approx(0.07)
    assert rows[("s", "session", "start")]["status"].startswith("not run")
    md = (out / "vieworder-latent.md").read_text()
    assert "0.920 [0.900, 0.940]" in md and "| s | end | not run" in md


def test_fewer_than_five_seeds_refused(tmp_path):
    res = tmp_path / "results-vieworder"
    _xs(res, seeds=range(4))
    with pytest.raises(ReportRefusal, match="five seeds"):
        vt.build_table(vt.load_reads(res, ["xs"], "latent"), ["xs"])


@pytest.mark.parametrize(
    "over", [{"split": "val"}, {"num_sequences": 20000}, {"corpus_id": "xs/latent/seed=3"}]
)
def test_off_registration_read_refused(tmp_path, over):
    res = tmp_path / "results-vieworder"
    _xs(res)
    _read(res, "xs", 2, "start", "request", 0.07, **over)
    with pytest.raises(ReportRefusal, match="registration"):
        vt.load_reads(res, ["xs"], "latent")
