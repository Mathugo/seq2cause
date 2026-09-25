# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The fixture corpus is a valid trace-bench corpus: the benchmark's scorer
scores its target against itself perfectly on both grains and both variants,
and the manifest flags the method-readable set the way the accessor does."""

import pytest
from fixture_corpus import fixture_corpus
from tracebench.allowlist import is_method_readable
from tracebench.score import self_check

from seq2causebench.record import read_json


@pytest.mark.parametrize("variant", ["latent", "twin"])
@pytest.mark.parametrize("grain", ["request", "session"])
def test_target_scores_itself_perfectly(variant, grain):
    res = self_check(fixture_corpus(variant), grain=grain)
    assert res["self_check_passed"], res["self_check_problems"]
    assert res["directed"]["f1"] == 1.0 and res["shd_mixed"] == 0
    if variant == "twin":
        assert res["causal_validity"]["value"]["sid"] == 0.0
        assert res["universe"]["truth_bidirected"] == 0
    else:
        assert (
            res["causal_validity"]["value"] is None
            and "bidirected" in res["causal_validity"]["reason"]
        )
        assert res["universe"]["truth_bidirected"] > 0


def test_manifest_flags_match_the_accessor():
    d = fixture_corpus("latent")
    m = read_json(d / "manifest.json")
    assert m["tool_version"] == "0.3.0"
    for f in m["files"]:
        assert f["method_readable"] == is_method_readable(f["path"]), f["path"]
    readable = {f["path"] for f in m["files"] if f["method_readable"]}
    assert all(p.startswith("views/") for p in readable)
    assert "graphs/scoring-target.json" not in readable and "topology/prior.json" not in readable
    assert (d / "COMPLETE").exists()
