"""The LMT member of the external family (plans/lmt.md; plans/cascade.md addendum 2026-11-04,
D-SB-29): `lmt/topology` on the fixture corpus end to end through the generic read — validation
read (the three `prior_weight_milli` settings) → `scoresweep` on the floor grid → the method's own
`…-external-lmt.json` freeze → the test read under it → `annotate` — with the runner on its
test-only `fit=stub` path (a graded soft output with one edge outside the mask; the runner's
docstring), a stub of the clone's `text_causal.graph_prior` under `--method-src` so the runner's
import is exercised, and, where torch is importable in this environment, one real fit of the
vectorised objective on the fixture's events."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from externalarm_world import FLOOR_TAUS, build_world

from seq2causebench import externalarm as ext
from seq2causebench import floorprior as fp
from seq2causebench.constants import (
    EXTERNAL_METHODS,
    GRAINS,
    RESULTS_JSON,
    RUN_DIR,
    RUN_META_JSON,
    VAL_TABLE_JSON,
)
from seq2causebench.corpus import Corpus
from seq2causebench.record import read_json

METHOD = "lmt"
ARM = "lmt/topology"
SETTINGS = [str(s) for s in EXTERNAL_METHODS[METHOD]["grid"]]
RUNNER = Path(ext.__file__).with_name(EXTERNAL_METHODS[METHOD]["runner"] + ".py")

# a test stand-in for the clone's `chemical_1000_event_repro/text_causal/graph_prior.py`: the
# three terms the runner imports (NOTEARS h, the Bernoulli log-prior, the sparsity sum), written
# here so the test never vendors the unlicensed clone; torch is imported lazily so the stub path
# needs none
STUB_PRIOR = '''
"""Test stand-in for text_causal.graph_prior."""


def dag_penalty_h(A):
    import torch

    return torch.linalg.matrix_exp(A * A).trace() - float(A.shape[0])


def semantic_log_bernoulli(A, q, eps=1e-6):
    import torch

    k = A.shape[0]
    q = q.clamp(eps, 1.0 - eps)
    mask = 1.0 - torch.eye(k, device=A.device, dtype=A.dtype)
    return ((A * torch.log(q) + (1.0 - A) * torch.log(1.0 - q)) * mask).sum()


def sparsity_sum(A):
    import torch

    k = A.shape[0]
    return (A * (1.0 - torch.eye(k, device=A.device, dtype=A.dtype))).sum()
'''
STUB_FILES = {
    "chemical_1000_event_repro/temporal_hawkes_pure_map.py": "# stub of the shipped trainer\n",
    "chemical_1000_event_repro/text_causal/__init__.py": "",
    "chemical_1000_event_repro/text_causal/graph_prior.py": STUB_PRIOR,
}


def _runner_module():
    spec = importlib.util.spec_from_file_location("lmt_runner_under_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("lmt"), METHOD, STUB_FILES, method_args=["fit=stub"])


def _prior_pairs(corpus):
    table = fp.topology_read(corpus)
    ptok = [str(t) for t in table["tokens"]]
    return {(ptok[s], ptok[d]) for s, d in zip(table["src"], table["dst"], strict=True)}


# --- the validation read, the sweep and the freeze -------------------------------------------
def test_val_read_is_a_soft_counted_lmt_read(world):
    for grain in GRAINS:
        r = read_json(world["root"] / f"val-{grain}" / RUN_DIR / RESULTS_JSON)
        assert r["status"] == "ok" and r["stage"] == "lmt" and r["method"] == "lmt"
        assert r["arms"] == [ARM] and r["mask_mode"] == "count" and r["hard_output"] is False
        assert r["setting_name"] == "prior_weight_milli" and r["settings"] == [int(s) for s in SETTINGS]
        assert r["model_sha256"] == "none" and len(r["method_sha256"]) == 64
        assert r["method_args"] == {"fit": "stub"} and r["c"] == r["g"] == 0
        assert r["inputs"][ARM][1].startswith("topology prior through floorprior")
        fits = r["facts"][ARM]["fits"]
        assert sorted(fits) == sorted(SETTINGS)
        for d in SETTINGS:
            f = fits[d]
            assert f["n_edges_outside_mask"] >= 1  # kept and counted, never refused
            assert f["n_allowed_pairs"] == r["facts"][ARM]["n_allowed_pairs"] > 0
            assert f["n_nonzero"] == f["n_allowed_pairs"] + 1  # the stub: mask pairs + one outside
            assert f["method_args"]["fit"] == "stub" and f["method_version"] == "unknown"
            assert f["setting"] == int(d) and f["prior_weight"] == int(d) / 1000
            assert f["wall_seconds"] >= 0 and "method_env" not in f
            z = np.load(world["root"] / f"val-{grain}" / f"scores-lmt-topology-none-c0-N{d}-{grain}.npz")
            assert int(z["N"]) == int(d) and str(z["method"]) == "lmt"
            assert str(z["setting_name"]) == "prior_weight_milli"
            col = z["max_lmt-topology__none"]
            assert 0 < col.min() and col.max() <= 1.0 and len(set(col.tolist())) > 2  # graded
        meta = read_json(world["root"] / f"val-{grain}" / RUN_DIR / RUN_META_JSON)
        assert meta["method"] == "lmt" and meta["method_env"] and all(
            isinstance(ln, str) and ln for ln in meta["method_env"]
        )
        # a soft output: the τ sweep separates the rows of every setting
        t = read_json(world["root"] / f"ss-{grain}" / VAL_TABLE_JSON)
        assert t["arms"] == [ARM]
        for d in SETTINGS:
            rows = [row for row in t["cells"] if row["arm"] == ARM and row["N"] == int(d)]
            assert {row["tau"] for row in rows} == {float(x) for x in FLOOR_TAUS}
            assert len({row["directed"]["f1"] for row in rows}) >= 2


def test_freeze_is_the_methods_own_and_tau_is_a_real_selection(world):
    doc = world["doc"]
    assert world["freeze"].name.endswith("-xs-latent-s0-external-lmt.json")
    assert doc["family"] == "external" and doc["method"] == "lmt"
    assert len(doc["method_sha256"]) == 64 and doc["method_commit"] == "unknown"
    assert set(doc["cells"]) == {f"{ARM}/none/frozen/{g}" for g in GRAINS}
    for grain in GRAINS:
        cell = doc["cells"][f"{ARM}/none/frozen/{grain}"]
        assert cell["grid"] == "floor_taus" and cell["N"] in (1000, 10000, 100000)
        assert cell["tau"] in {float(x) for x in FLOOR_TAUS}
        t = read_json(world["root"] / f"ss-{grain}" / VAL_TABLE_JSON)
        rows = [row for row in t["cells"] if row["arm"] == ARM and row["N"] == cell["N"]]
        f1_at = {row["tau"]: row["directed"]["f1"] for row in rows}
        assert f1_at[cell["tau"]] == pytest.approx(max(f1_at.values()))
        assert min(f1_at.values()) < max(f1_at.values())  # some τ of the grid loses
        assert cell["tau"] < 0.9  # the outside edge at 0.9 and most of the mask survive the cut


# --- the test read and annotate --------------------------------------------------------------
def test_test_read_keeps_the_edge_outside_the_mask(world):
    doc = world["doc"]
    prior = _prior_pairs(world["corpus"])
    for grain in GRAINS:
        key = f"{ARM}/none/frozen/{grain}"
        out = world["root"] / f"test-lmt-topology-{grain}"
        r = read_json(out / RUN_DIR / RESULTS_JSON)
        assert r["stage"] == "lmt" and r["arms"] == [ARM] and r["N"] == doc["cells"][key]["N"]
        assert r["freeze"]["path"] == str(world["freeze"])
        assert r["facts"][ARM]["fits"][str(r["N"])]["n_edges_outside_mask"] >= 1
        assert (out / f"scores-lmt-topology-none-{grain}.npz").exists()
        pred = read_json(next(out.glob(f"prediction-{grain}-lmt-topology-*.json")))
        edges = {(e["src"], e["dst"]) for e in pred["directed"]}
        assert edges and edges - prior, "the soft prior's edge outside the topology was lost"
        assert len(edges - prior) == 1 and len(edges & prior) >= 1
        ann = read_json(world["results"] / "xs" / "latent" / "seed=0" / key / "annotate.json")
        assert ann["arm_class"] == "external" and ann["probe"] == "lmt-topology"
        assert ann["frozen"]["N"] == doc["cells"][key]["N"]


# --- the runner's own rules ------------------------------------------------------------------
def test_runner_q_matrix_stub_and_method_args():
    rn = _runner_module()
    allowed = np.array([[0, 1], [2, 3]], dtype=np.int64)
    q = rn.q_matrix(4, allowed, 0.5, 0.01, 0.5)
    assert q[0, 1] == q[2, 3] == 0.5 and q[1, 0] == q[0, 2] == 0.01
    assert np.all(np.diag(q) == 0.5) and q.shape == (4, 4)
    assert np.all(rn.q_matrix(4, None, 0.5, 0.01, 0.3)[~np.eye(4, dtype=bool)] == 0.3)
    s = rn.stub_scores(4, allowed)
    assert s[0, 1] == pytest.approx(0.3) and s[2, 3] == pytest.approx(1.0) and s[1, 0] == 0.9
    assert np.count_nonzero(s) == 3 and s.dtype == np.float32
    cfg = rn.parse_method_args(["epochs=3", "q_out=0.2", "fit=stub"])
    assert cfg["epochs"] == 3 and cfg["q_out"] == 0.2 and cfg["lr"] == 0.05
    assert cfg["dtype"] == cfg["param_dtype"] == "float64" and cfg["chunk_events"] == 0
    assert cfg["device"] == "cpu" and rn.parse_method_args(["param_dtype=float32"])["param_dtype"] == "float32"
    for bad in (["unknown=1"], ["q_in=1.0"], ["epochs=0"], ["dtype=float16"], ["fit=real"], ["x"]):
        with pytest.raises(SystemExit):
            rn.parse_method_args(bad)


def test_runner_names_no_ground_truth_path():
    src = RUNNER.read_text(encoding="utf-8")
    for word in ("graphs/", "labels/", "topology/", "alphabet.json", "scoring-target", "parent_pos"):
        assert word not in src


def test_torch_fit_runs_on_the_fixture_events(world, tmp_path):
    """The real objective, where this environment has torch: deterministic under the seed, the
    chunked evaluation neutral, the output a graded matrix with a zero diagonal."""
    pytest.importorskip("torch")
    corpus = Corpus(world["corpus"], "end", "request")
    events = ext.events_read(corpus, "val", 0, "head", 0, 24)
    allowed, _ = ext.topology_mask(world["corpus"], events["tokens"])
    n = len(events["tokens"])

    def fit(work, extra):
        return ext.run_method(
            sys.executable, RUNNER, world["stub"], tmp_path / work, events, allowed, 10000,
            ["epochs=3", *extra], False,
        )

    scores, facts = fit("a", ["chunk_events=0"])
    assert scores.shape == (n, n) and scores.dtype == np.float32
    assert np.all(np.diag(scores) == 0) and 0 < scores[~np.eye(n, dtype=bool)].min()
    assert scores.max() <= 1.0 and facts["n_nonzero"] == n * (n - 1)
    assert facts["epochs"] == 3 and facts["final_h"] >= 0 and np.isfinite(facts["final_loglike"])
    assert facts["method_args"]["fit"] == "torch" and facts["prior_weight"] == 10.0
    assert facts["n_above_half_outside_mask"] <= facts["n_above_half"]
    assert facts["loss_last"] < facts["loss_first"]
    again, _ = fit("b", ["chunk_events=0"])
    assert np.array_equal(scores, again)
    chunked, _ = fit("c", ["chunk_events=7"])
    assert np.allclose(scores, chunked, atol=1e-5)
