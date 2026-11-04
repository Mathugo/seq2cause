"""The THP arm (plans/thp.md; the external family's addendum of 2026-11-04, D-SB-29): `thp/topology`
on the fixture end to end through a stub `castle` package placed under `--method-src` — validation
read (the three `decay_ms` settings) → `scoresweep` on the floor grid → the method's own external
freeze → the test read under it → `annotate` — with the node contract of `events.npz` checked, the
mask asserted, the hard output's degenerate τ sweep, the environment export, and the refusal of an
edge outside the mask. The stub stands in for gCastle: a `TTPM` with the shipped constructor
signature whose `learn` returns every pair the prior leaves open (plus, at `max_hop >= 9`, one
forbidden pair that keeps the graph acyclic) and a `PrioriKnowledge` with the shipped attributes."""

from pathlib import Path

import numpy as np
import pytest
from externalarm_world import FLOOR_TAUS, build_world, external

from seq2causebench import externalarm as ext
from seq2causebench import floorprior as fp
from seq2causebench import thp_runner as tr
from seq2causebench.constants import (
    EXTERNAL_METHODS,
    GRAINS,
    RESULTS_JSON,
    RUN_DIR,
    RUN_META_JSON,
    VAL_TABLE_JSON,
)
from seq2causebench.record import read_json

METHOD = "thp"
ARM = "thp/topology"


SETTINGS = [str(s) for s in EXTERNAL_METHODS[METHOD]["grid"]]  # 100, 1000, 10000
METHOD_ARGS = ["max_hop=1", "penalty=BIC", "max_iter=5000", "epsilon=1"]

STUB_FILES = {
    "castle/__init__.py": '__version__ = "0.0-stub"\n',
    "castle/common/__init__.py": "",
    "castle/common/priori_knowledge.py": '''
import numpy as np


class PrioriKnowledge(object):
    """The shipped attributes: `matrix` (-1 unknown, 0 forbidden, 1 required, 0 on the
    diagonal), `forbidden_edges`, `required_edges`, and the membership tests."""

    def __init__(self, n_nodes):
        self.matrix = -np.ones((n_nodes, n_nodes), dtype=int) + np.eye(n_nodes, dtype=int)
        self.forbidden_edges = []
        self.required_edges = []

    def add_forbidden_edge(self, i, j):
        self.matrix[i, j] = 0
        self.forbidden_edges.append((i, j))

    def add_forbidden_edges(self, edges):
        for i, j in edges:
            self.add_forbidden_edge(i, j)

    def is_forbidden(self, i, j):
        return (i, j) in self.forbidden_edges

    def is_required(self, i, j):
        return (i, j) in self.required_edges
''',
    "castle/algorithms/__init__.py": "from .ttpm import TTPM\n",
    "castle/algorithms/ttpm/__init__.py": "from .ttpm import TTPM\n",
    "castle/algorithms/ttpm/ttpm.py": '''
import numpy as np
import pandas as pd


class TTPM(object):
    """A stand-in for gCastle's TTPM: the shipped constructor; `learn` takes the (event,
    timestamp, node) frame and returns, over the sorted event names, the diagonal plus every
    pair the prior leaves open (the mask); at `max_hop >= 9` one forbidden pair is added that
    cannot close a cycle (its effect is a sink of the open pairs)."""

    def __init__(self, topology_matrix, delta=0.1, epsilon=1, max_hop=0, penalty="BIC",
                 max_iter=20, priori_knowledge=None):
        assert isinstance(topology_matrix, np.ndarray) and topology_matrix.ndim == 2
        assert topology_matrix.shape[0] == topology_matrix.shape[1]
        assert np.array_equal(topology_matrix, topology_matrix.T)
        assert delta > 0 and penalty in ("BIC", "AIC") and max_iter >= 1
        self._topo = topology_matrix
        self._delta = delta
        self._max_hop = max_hop
        self.priori_knowledge = priori_knowledge
        self.causal_matrix = None
        self.n_sweeps_ = None  # the patched search's attributes (0001-decomposable-score)
        self.converged_ = None

    def learn(self, tensor, *args, **kwargs):
        for col in ("event", "timestamp", "node"):
            assert col in tensor.columns
        assert int(tensor["node"].max()) < self._topo.shape[0]
        names = np.array(sorted(set(tensor["event"])))
        n = len(names)
        g = np.eye(n, dtype=int)
        pk = self.priori_knowledge
        if pk is None:
            for i in range(n - 1):
                g[i, i + 1] = 1
        else:
            open_pairs = [(i, j) for i in range(n) for j in range(n) if i != j and pk.matrix[i, j] != 0]
            for i, j in open_pairs:
                g[i, j] = 1
            if self._max_hop >= 9 and open_pairs:
                sinks = [j for j in range(n) if not any(a == j for a, _ in open_pairs)]
                i, j = next((a, b) for b in sinks for a in range(n) if a != b and pk.is_forbidden(a, b))
                g[i, j] = 1
        self.causal_matrix = pd.DataFrame(g, index=names.astype(str), columns=names.astype(str))
        self.n_sweeps_ = int(g.sum() - n) + 1
        self.converged_ = True
''',
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("thp"), METHOD, STUB_FILES, method_args=METHOD_ARGS)


# --- the events contract (node = op) ---------------------------------------------------------
def test_events_npz_carries_the_nodes_and_a_symmetric_op_topology(world):
    work = next((world["root"] / "val-request").glob("thp-work/*request*/events.npz"))
    with np.load(work) as z:
        keys = set(z.files)
        assert {"event", "timestamp", "node", "n_ops", "op_topology"} <= keys
        event, node, n_ops, A = z["event"], z["node"], int(z["n_ops"]), z["op_topology"]
        assert len(node) == len(event) and node.dtype == np.int64
        assert int(node.max()) < n_ops and int(node.min()) >= 0
        assert A.shape == (n_ops, n_ops) and A.dtype == np.int8 and np.array_equal(A, A.T)
        assert A.sum() > 0 and np.all(np.diag(A) == 0)
    r = read_json(world["root"] / "val-request" / RUN_DIR / RESULTS_JSON)
    assert r["facts"][ARM]["n_ops"] == n_ops
    allowed = np.load(work.with_name("allowed.npy"))
    assert (
        allowed.ndim == 2
        and allowed.shape[1] == 2
        and len(allowed) == r["facts"][ARM]["n_allowed_pairs"]
    )


def test_mask_is_the_topology_floors_pairs_over_the_sample(world):
    r = read_json(world["root"] / "val-request" / RUN_DIR / RESULTS_JSON)
    z = np.load(world["root"] / "val-request" / "scores-thp-topology-none-c0-N1000-request.npz")
    tok = [str(t) for t in z["tokens"]]
    got = {(tok[s], tok[d]) for s, d in zip(z["src"], z["dst"], strict=True)}
    table = fp.topology_read(world["corpus"])
    ptok = [str(t) for t in table["tokens"]]
    prior = {(ptok[s], ptok[d]) for s, d in zip(table["src"], table["dst"], strict=True)}
    present = set(tok)
    want = {(a, b) for a, b in prior if a in present and b in present}
    assert got == want and len(got) == r["facts"][ARM]["n_allowed_pairs"]
    assert np.all(z["max_thp-topology__none"] == 1.0) and int(z["N"]) == 1000
    assert int(z["c"]) == int(z["g"]) == 0 and str(z["model_sha256"]) == "none"
    assert len(str(z["method_sha256"])) == 64 and str(z["method"]) == METHOD
    assert str(z["setting_name"]) == "decay_ms"


# --- the validation read, the sweep and the freeze -------------------------------------------
def test_val_read_sweep_and_the_methods_own_freeze(world):
    for grain in GRAINS:
        r = read_json(world["root"] / f"val-{grain}" / RUN_DIR / RESULTS_JSON)
        assert r["status"] == "ok" and r["stage"] == "thp" and r["method"] == METHOD
        assert r["arms"] == [ARM] and r["mask_mode"] == "refuse" and r["hard_output"] is True
        assert r["setting_name"] == "decay_ms" and r["settings"] == [100, 1000, 10000]
        assert r["model_sha256"] == "none" and r["N"] is None and r["c"] == r["g"] == 0
        assert r["method_args"] == {
            "max_hop": "1",
            "penalty": "BIC",
            "max_iter": "5000",
            "epsilon": "1",
        }
        assert r["inputs"][ARM][0] == f"views/end-{grain}/sequences/split=val"
        assert r["inputs"][ARM][1].startswith("topology prior through floorprior")
        fits = r["facts"][ARM]["fits"]
        assert sorted(fits) == sorted(SETTINGS)
        for d, f in fits.items():
            assert f["setting"] == int(d) and f["n_types"] > 0 and f["wall_seconds"] >= 0
            assert f["n_forbidden_pairs"] > 0 and f["n_edges_outside_mask"] == 0
            assert f["acyclic"] is True and f["method_version"].startswith("gcastle")
            assert f["converged"] is True and f["n_sweeps_run"] >= 1 and f["max_iter"] == 5000
            assert f["method_args"] == {
                "max_hop": 1,
                "penalty": "BIC",
                "max_iter": 5000,
                "epsilon": 1,
            }
            assert "method_env" not in f
        meta = read_json(world["root"] / f"val-{grain}" / RUN_DIR / RUN_META_JSON)
        assert isinstance(meta["method_env"], list) and len(meta["method_env"]) > 0
        assert meta["method"] == METHOD and len(meta["method_commit"]) > 0
        t = read_json(world["root"] / f"ss-{grain}" / VAL_TABLE_JSON)
        assert t["model_sha256"] == "none" and t["arms"] == [ARM]
        rows = {(row["arm"], row["N"], row["tau"]) for row in t["cells"]}
        assert rows == {(ARM, int(d), float(x)) for d in SETTINGS for x in FLOOR_TAUS}
        for d in SETTINGS:  # a hard output: every τ of one setting scores the same
            f1s = {row["directed"]["f1"] for row in t["cells"] if row["N"] == int(d)}
            assert len(f1s) == 1
    doc = world["doc"]
    assert world["freeze"].name.endswith("-xs-latent-s0-external-thp.json")
    assert doc["family"] == "external" and doc["method"] == METHOD
    assert doc["model_sha256"] == "none" and doc["ordering"] == "end"
    assert set(doc["cells"]) == {f"{ARM}/none/frozen/{g}" for g in GRAINS}
    for c in doc["cells"].values():
        assert c["grid"] == "floor_taus" and c["c"] == c["g"] == 0 and c["N"] in (100, 1000, 10000)
        assert c["tau"] == 0.9  # ties → larger τ; the cut keeps the whole DAG either way
        assert c["every_scored_pair_f1"] == pytest.approx(c["val_directed_f1"])


# --- the test read and annotate --------------------------------------------------------------
def test_test_read_under_the_freeze_and_annotate(world):
    doc = world["doc"]
    for grain in GRAINS:
        key = f"{ARM}/none/frozen/{grain}"
        out = world["root"] / f"test-thp-topology-{grain}"
        r = read_json(out / RUN_DIR / RESULTS_JSON)
        assert r["arms"] == [ARM] and r["N"] == doc["cells"][key]["N"] and r["stage"] == "thp"
        assert r["freeze"]["path"] == str(world["freeze"])
        assert r["staging"]["stage_matched"] == "discover"
        assert (
            r["cells"][key]["tau"] == 0.9
            and r["cells"][key]["n_edges"] == r["cells"][key]["n_pairs"]
        )
        assert (out / f"scores-thp-topology-none-{grain}.npz").exists()
        cell = world["results"] / "xs" / "latent" / "seed=0" / key
        ann = read_json(cell / "annotate.json")
        assert ann["arm_class"] == "external" and ann["model_sha256"] == "none"
        assert ann["probe"] == "thp-topology" and ann["frozen"]["N"] == doc["cells"][key]["N"]
        assert ann["coverage"]["n_sequences_probed"] == r["n_probed"] > 0
    # the masked arm's test edges lie inside the prior's pairs
    out = world["root"] / "test-thp-topology-request"
    pred = read_json(next(out.glob("prediction-request-thp-topology-*.json")))
    table = fp.topology_read(world["corpus"])
    ptok = [str(t) for t in table["tokens"]]
    prior = {(ptok[s], ptok[d]) for s, d in zip(table["src"], table["dst"], strict=True)}
    assert pred["directed"] and all((e["src"], e["dst"]) in prior for e in pred["directed"])


# --- refusals and the runner's own parsing ---------------------------------------------------
def test_an_edge_outside_the_mask_is_refused(world, tmp_path):
    with pytest.raises(ext.ExternalRefusal, match="outside the mask"):
        external(
            world["corpus"],
            "val",
            "request",
            tmp_path / "outside",
            world["ledger"],
            METHOD,
            world["stub"],
            [ARM],
            ["1000"],
            method_args=["max_hop=9", "penalty=BIC", "max_iter=5000", "epsilon=1"],
        )


def test_runner_parses_the_registered_method_args_only():
    assert tr.parse_method_args([]) == {
        "max_hop": 1,
        "penalty": "BIC",
        "max_iter": 5000,
        "epsilon": 1,
    }
    assert tr.parse_method_args(["max_hop=2", "epsilon=0.5"])["max_hop"] == 2
    assert tr.parse_method_args(["epsilon=0.5"])["epsilon"] == 0.5
    with pytest.raises(ValueError, match="unknown method-arg"):
        tr.parse_method_args(["delta=0.1"])
    with pytest.raises(ValueError, match="penalty"):
        tr.parse_method_args(["penalty=L1"])
    assert tr.is_acyclic(np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]))
    assert not tr.is_acyclic(np.array([[0, 1], [1, 0]]))
    # the runner names no non-method-readable path; the mask arrives as dense ids
    src = Path(tr.__file__).read_text(encoding="utf-8")
    assert "topology/" not in src and "alphabet.json" not in src and "parent_pos" not in src
