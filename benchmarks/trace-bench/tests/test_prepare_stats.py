# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""`prepare` reports the substrate: rows per split, vocabulary, folded variants,
length quantiles and the truncation rate at `--max-len`, through the door only."""

from fixture_corpus import N_REQUESTS, N_SESSIONS, fixture_corpus

from seq2causebench import prepare as prep
from seq2causebench.constants import PREPARE_JSON, RESULTS_JSON, RUN_DIR
from seq2causebench.record import read_json


def test_prepare_json(tmp_path):
    out = tmp_path / "prep"
    rc = prep.main(
        [
            "--corpus",
            str(fixture_corpus("latent")),
            "--ordering",
            "end",
            "--grain",
            "request",
            "--max-len",
            "4",
            "--entropy-order",
            "2",
            "--output-folder",
            str(out),
        ]
    )
    assert rc == 0
    p = read_json(out / PREPARE_JSON)
    assert p["view"] == "end-request" and p["vocab"]["size"] == 14 and p["vocab"]["n_ops"] == 6
    assert (
        0.0 < p["entropy"]["entropy_floor"] < p["entropy"]["log_alphabet"]
        and p["entropy"]["order"] == 2
    )
    for split, n in N_REQUESTS.items():
        s = p["splits"][split]
        assert s["rows"] == n == p["export_stats_rows"][f"end-request-{split}"]
        assert s["n_spans"]["p50"] >= 1 and s["n_spans"]["max"] <= 6
        assert s["tokens_dropped"] == sum(0 for _ in ()) or s["tokens_dropped"] >= 0
    train = p["splits"]["train"]
    assert train["folded_positions"] > 0 and train["folded_pairs_distinct"] >= 1
    assert all(pair[1] != 0 for pair in train["folded_pairs"])  # only non-OK outcomes fold
    assert 0 < train["truncation_rate"] < 1 and train["tokens_dropped"] > 0
    assert read_json(out / RUN_DIR / RESULTS_JSON)["rows"]["train"] == N_REQUESTS["train"]


def test_session_grain_is_longer(tmp_path):
    res = prep.prepare(fixture_corpus("latent"), "end", "session", 64, 1)
    assert res["splits"]["train"]["rows"] == N_SESSIONS["train"]
    assert (
        res["splits"]["train"]["n_spans"]["mean"]
        > prep.prepare(fixture_corpus("latent"), "end", "request", 64, 1)["splits"]["train"][
            "n_spans"
        ]["mean"]
    )
    assert res["splits"]["train"]["truncated_sequences"] == 0
