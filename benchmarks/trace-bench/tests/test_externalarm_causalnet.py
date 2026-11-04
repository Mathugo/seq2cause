"""The CausalNET arm (plans/causalnet.md; the external family's addendum of 2026-11-04, D-SB-29):
`causalnet/topology` on the fixture end to end through `externalarm_world` with a stub of the
method under `--method-src` (`model/STUB_RUNNER.py`, the runner's test-only path, so no torch and
no training): validation read on the registered grid → `scoresweep` on the floor grid → the
method's own freeze → the test read → `annotate`. The stub returns graded scores on every allowed
pair, one edge outside the mask (a soft prior: `mask_mode = count` keeps and counts it) and zero
elsewhere, so the τ sweep is a real selection and the outside edge must survive to the prediction."""

import externalarm_world
import numpy as np
import pytest
from externalarm_world import FLOOR_TAUS, build_world
from fixture_corpus import fixture_corpus

from seq2causebench import floorprior as fp
from seq2causebench.constants import (
    EXTERNAL_METHODS,
    GRAINS,
    RESULTS_JSON,
    RUN_DIR,
    RUN_META_JSON,
    VAL_TABLE_JSON,
)
from seq2causebench.record import read_json

METHOD = "causalnet"
ARM = "causalnet/topology"
SETTINGS = [str(s) for s in EXTERNAL_METHODS[METHOD]["grid"]]
METHOD_ARGS = ["epochs=200", "device=auto", "seed=42", "max_hop=1", "readout=shipped"]

STUB = '''
import numpy as np

def fit(events, allowed, setting, method_args):
    """A stand-in for CausalNET: 0.3 + 0.7 * rank fraction on every allowed pair (rank = the
    pair's position in the sorted mask), 0.9 on the first pair outside the mask whose ops differ,
    0 elsewhere; the same for every setting."""
    n = int(events["event"].max()) + 1
    node_of_type = np.zeros(n, dtype=np.int64)
    node_of_type[events["event"]] = events["node"]
    m = np.zeros((n, n), dtype=np.float32)
    pairs = sorted((int(a), int(b)) for a, b in allowed)
    k = max(len(pairs) - 1, 1)
    for r, (a, b) in enumerate(pairs):
        m[a, b] = 0.3 + 0.7 * r / k
    inside = set(pairs)
    for a in range(n):
        for b in range(n):
            if a != b and (a, b) not in inside and node_of_type[a] != node_of_type[b]:
                m[a, b] = 0.9
                return m
    return m
'''

STUB_FILES = {
    "model/STUB_RUNNER.py": STUB,
    "model/main.py": "# stub clone: hashed by method_sha256 (model/*.py)\n",
    "model/configs/config_m24.yaml": "total_epoch: 200\n",
}



@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(
        tmp_path_factory.mktemp("causalnet"), METHOD, STUB_FILES, method_args=METHOD_ARGS
    )


def test_events_carry_the_nodes_and_the_op_topology(world):
    work = world["root"] / "val-request" / "causalnet-work" / "causalnet-topology-request"
    with np.load(work / "events.npz") as z:
        assert {"event", "timestamp", "node", "n_ops", "op_topology"} <= set(z.files)
        A = z["op_topology"]
        n_ops = int(z["n_ops"])
        assert A.shape == (n_ops, n_ops) and A.dtype == np.int8 and np.array_equal(A, A.T)
        assert A.sum() > 0 and z["node"].max() < n_ops and len(z["node"]) == len(z["event"])
    allowed = np.load(work / "allowed.npy")
    assert allowed.ndim == 2 and allowed.shape[1] == 2 and len(allowed) > 0
    for d in SETTINGS:
        side = read_json(work / f"scores-N{d}.npy.json")
        assert side["setting"] == int(d) and side["stub"] is True
        assert (
            side["method_args"]["epochs"] == "200" and side["method_args"]["readout"] == "shipped"
        )
        assert side["method_version"]["patches"] == ["0001-collapse-decay", "0002-drivable-entry"]
        assert side["n_allowed_pairs"] == len(allowed) and side["n_nonzero"] == len(allowed) + 1


def test_val_read_counts_the_outside_edge_and_sweeps_a_real_tau(world):
    for grain in GRAINS:
        r = read_json(world["root"] / f"val-{grain}" / RUN_DIR / RESULTS_JSON)
        assert r["status"] == "ok" and r["stage"] == "causalnet" and r["method"] == METHOD
        assert r["mask_mode"] == "count" and r["hard_output"] is False
        assert r["setting_name"] == "window_ms" and r["settings"] == [int(s) for s in SETTINGS]
        assert r["method_args"] == {
            "epochs": "200",
            "device": "auto",
            "seed": "42",
            "max_hop": "1",
            "readout": "shipped",
        }
        assert len(r["method_sha256"]) == 64 and r["model_sha256"] == "none"
        fits = r["facts"][ARM]["fits"]
        assert sorted(fits) == sorted(SETTINGS)
        for f in fits.values():
            assert f["n_edges_outside_mask"] >= 1 and f["mask_mode"] == "count"
            assert f["n_pairs"] == r["facts"][ARM]["n_allowed_pairs"] + 1
        meta = read_json(world["root"] / f"val-{grain}" / RUN_DIR / RUN_META_JSON)
        assert meta["method_env"] and all(isinstance(ln, str) for ln in meta["method_env"])
        t = read_json(world["root"] / f"ss-{grain}" / VAL_TABLE_JSON)
        assert t["arms"] == [ARM] and t["model_sha256"] == "none"
        rows = {(row["N"], row["tau"]) for row in t["cells"]}
        assert rows == {(int(d), float(x)) for d in SETTINGS for x in FLOOR_TAUS}
        for d in SETTINGS:
            f1s = {row["directed"]["f1"] for row in t["cells"] if row["N"] == int(d)}
            assert len(f1s) >= 2, (grain, d, f1s)


def test_the_freeze_is_the_methods_own(world):
    doc = world["doc"]
    assert world["freeze"].name.endswith("-xs-latent-s0-external-causalnet.json")
    assert doc["family"] == "external" and doc["method"] == METHOD
    assert set(doc["cells"]) == {f"{ARM}/none/frozen/{g}" for g in GRAINS}
    for cell in doc["cells"].values():
        assert cell["N"] in [int(s) for s in SETTINGS] and 0 <= cell["tau"] < 0.9


def test_test_read_keeps_the_outside_edge_through_annotate(world):
    doc = world["doc"]
    for grain in GRAINS:
        key = f"{ARM}/none/frozen/{grain}"
        out = world["root"] / f"test-causalnet-topology-{grain}"
        r = read_json(out / RUN_DIR / RESULTS_JSON)
        assert r["arms"] == [ARM] and r["N"] == doc["cells"][key]["N"]
        assert r["facts"][ARM]["fits"][str(r["N"])]["n_edges_outside_mask"] >= 1
        ann = read_json(world["results"] / "xs" / "latent" / "seed=0" / key / "annotate.json")
        assert ann["arm_class"] == "external" and ann["probe"] == "causalnet-topology"
    pred = read_json(
        next(
            (world["root"] / "test-causalnet-topology-request").glob(
                "prediction-request-causalnet-topology-*.json"
            )
        )
    )
    table = fp.topology_read(fixture_corpus("latent"))
    ptok = [str(t) for t in table["tokens"]]
    prior = {(ptok[s], ptok[d]) for s, d in zip(table["src"], table["dst"], strict=True)}
    edges = {(e["src"], e["dst"]) for e in pred["directed"]}
    assert edges & prior, "the masked pairs reach the prediction"
    assert edges - prior, "the edge outside the mask survives into the prediction (mask_mode count)"
