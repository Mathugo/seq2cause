"""The external family generalised (plans/cascade.md addendum 2026-11-04; D-SB-29): the generic
read's contracts that do not need a method — the events' dense ops and node topology, the score
table's two mask modes, the method identity, one freeze per method, the parser. The per-method
chains run in `test_externalarm_<method>.py` through `externalarm_world`; CASCADE's own front end
is covered by `test_cascadearm.py`, now on the same module."""

import json
from pathlib import Path

import numpy as np
import pytest

from seq2causebench import externalarm as ext
from seq2causebench import freeze as fz
from seq2causebench.constants import EXTERNAL_METHODS, MASK_COUNT, MASK_REFUSE


class _Fake:
    view_rel = "views/end-request"

    def __init__(self, rows):
        self._rows = rows

    def sequences(self, split, columns=None):
        for i, (ops, outs, t0, off, dur) in enumerate(self._rows):
            yield (f"t{i}", list(ops), list(outs), len(ops), t0, list(off), list(dur))


MS = 1_000_000
ROWS = [
    ([4, 2, 0], [3, 3, 1], 100 * MS, [0, 1 * MS, 4 * MS], [5 * MS, 2 * MS, 5 * MS]),
    ([2, 0], [0, 0], 50 * MS, [0, 0], [1 * MS, 1 * MS]),
    ([5], [3], 10 * MS, [0], [0]),
]


def _events():
    return ext.events_read(_Fake(ROWS), "val", 0, "head", 0, 10)


def test_events_carry_dense_ops_and_the_node_of_every_event():
    ev = _events()
    assert [str(t) for t in ev["tokens"]] == ["0:ok", "0:4xx", "2:ok", "2:err", "4:err", "5:err"]
    assert ev["op_of_token"].tolist() == [0, 0, 2, 2, 4, 5]
    assert ev["op_id_of_token"].tolist() == [0, 0, 1, 1, 2, 3] and ev["n_ops"] == 4
    node = ev["op_id_of_token"][ev["event"]]
    assert node.tolist() == [3, 1, 0, 1, 2, 0]  # 5:err, 2:ok, 0:ok, 2:err, 4:err, 0:4xx
    assert ev["facts"]["n_ops"] == 4


def test_op_topology_is_the_undirected_span_of_the_allowed_pairs():
    ev = _events()
    allowed = np.array([[2, 0], [3, 1], [4, 2]], dtype=np.int64)  # 2:ok→0:ok, 2:err→0:4xx, 4:err→2:ok
    A = ext.op_topology(allowed, ev["op_id_of_token"], ev["n_ops"])
    assert A.dtype == np.int8 and A.shape == (4, 4)
    assert np.array_equal(A, A.T)
    assert A[1, 0] == A[0, 1] == 1 and A[2, 1] == A[1, 2] == 1 and A[2, 0] == 0 and A[3].sum() == 0
    assert ext.op_topology(None, ev["op_id_of_token"], ev["n_ops"]).sum() == 0


def test_score_table_refuses_or_counts_an_edge_outside_the_mask():
    ev = _events()
    allowed = np.array([[2, 0], [3, 1]], dtype=np.int64)
    m = np.zeros((6, 6), dtype=np.float32)
    m[2, 0] = 0.9
    m[3, 1] = 0.4
    m[3, 3] = 1.0  # the diagonal is ignored
    t = ext.score_table(m, ev, allowed, MASK_REFUSE)
    assert list(zip(t["src"].tolist(), t["dst"].tolist(), strict=True)) == [(2, 0), (3, 1)]
    assert t["score"].tolist() == [pytest.approx(0.9), pytest.approx(0.4)]
    assert t["facts"] == {
        "n_edges_returned": 2,
        "n_within_op_dropped": 0,
        "n_edges_outside_mask": 0,
        "mask_mode": "refuse",
    }
    m[4, 2] = 0.5  # 4:err → 2:ok, outside the mask
    m[0, 1] = 0.7  # within-op (0:ok → 0:4xx): never in a mask; dropped at projection under `count`
    with pytest.raises(ext.ExternalRefusal, match="outside the mask"):
        ext.score_table(m, ev, allowed, MASK_REFUSE)
    t = ext.score_table(m, ev, allowed, MASK_COUNT)
    pairs = list(zip(t["src"].tolist(), t["dst"].tolist(), strict=True))
    assert (4, 2) in pairs and (0, 1) not in pairs
    assert t["facts"]["n_edges_outside_mask"] == 2 and t["facts"]["mask_mode"] == "count"
    assert t["facts"]["n_edges_returned"] == 4 and t["facts"]["n_within_op_dropped"] == 1
    t = ext.score_table(m, ev, None, MASK_REFUSE)  # no mask: nothing is outside it
    assert t["facts"]["mask_mode"] is None and t["facts"]["n_edges_outside_mask"] == 0
    with pytest.raises(ext.ExternalRefusal, match="within"):
        ext.score_table(m * 2, ev, allowed, MASK_COUNT)
    with pytest.raises(ext.ExternalRefusal, match="matrix"):
        ext.score_table(m[:5, :5], ev, allowed, MASK_COUNT)
    with pytest.raises(ValueError):
        ext.score_table(m, ev, allowed, "drop")


def test_method_identity_hashes_registered_files_and_the_pin(tmp_path):
    src = tmp_path / "pkg"
    (src / "castle" / "algorithms" / "ttpm").mkdir(parents=True)
    (src / "castle" / "common").mkdir(parents=True)
    (src / "castle" / "algorithms" / "ttpm" / "ttpm.py").write_text("a = 1\n")
    (src / "castle" / "common" / "priori_knowledge.py").write_text("b = 2\n")
    runner = tmp_path / "r.py"
    runner.write_text("print()\n")
    sha0, pin0 = ext.method_sha256("thp", src, runner)
    assert pin0 == "unknown" and len(sha0) == 64
    (src / "gcastle-1.0.4.dist-info").mkdir()
    (src / "gcastle-1.0.4.dist-info" / "direct_url.json").write_text(
        json.dumps({"url": "https://x", "vcs_info": {"vcs": "git", "commit_id": "abc123"}})
    )
    sha1, pin1 = ext.method_sha256("thp", src, runner)
    assert pin1 == "abc123" and sha1 != sha0
    (src / "castle" / "algorithms" / "ttpm" / "ttpm.py").write_text("a = 2\n")
    assert ext.method_sha256("thp", src, runner)[0] != sha1
    (src / "castle" / "other.py").write_text("c = 3\n")  # not a registered file
    assert ext.method_sha256("thp", src, runner)[0] == ext.method_sha256("thp", src, runner)[0]
    runner.write_text("print(1)\n")  # the runner's own bytes are in the hash
    assert ext.method_sha256("thp", src, runner)[0] != sha1


def test_one_freeze_per_method():
    thp = "thp/topology/none/frozen/request"
    cas = "cascade/topology/none/frozen/request"
    with pytest.raises(fz.FreezeRefusal, match="D-SB-29"):
        fz.family_of({thp: {}, cas: {}})
    with pytest.raises(fz.FreezeRefusal, match="D-SB-27"):
        fz.family_of({thp: {}, "trace/core/shipped/frozen/request": {}})
    assert fz.family_of({thp: {}}) == "external" and fz.external_method_of({thp: {}}) == "thp"
    assert fz.external_method_of({cas: {}}) == "cascade"
    assert fz.external_method_of({"floor/topology/none/frozen/request": {}}) is None
    assert fz.freeze_name("2026-01-01", "xs", "latent", 0, "external", "end", "thp") == (
        "2026-01-01-xs-latent-s0-external-thp.json"
    )
    assert fz.freeze_name("2026-01-01", "xs", "latent", 0, "external", "end", "cascade") == (
        "2026-01-01-xs-latent-s0-external.json"
    )
    assert fz.freeze_name("2026-01-01", "xs", "latent", 0, "external", "end") == (
        "2026-01-01-xs-latent-s0-external.json"
    )
    assert fz.freeze_name("2026-01-01", "s", "latent", 2, "external", "start", "lmt") == (
        "2026-01-01-s-latent-s2-external-lmt-start.json"
    )


def test_parser_requires_every_flag_and_parses_method_args():
    p = ext.build_parser()
    for a in p._actions:
        if a.dest != "help":
            assert a.required, a.dest
    assert set(ext.GENERIC_METHODS) == set(EXTERNAL_METHODS) - {"cascade"}
    with pytest.raises(SystemExit):
        p.parse_args(["--method", "cascade"])  # CASCADE keeps its own front end
    assert ext.parse_method_args(["max_hop=1", "penalty=BIC"]) == {"max_hop": "1", "penalty": "BIC"}
    assert ext.parse_method_args([]) == {}
    with pytest.raises(ValueError):
        ext.parse_method_args(["max_hop"])
    with pytest.raises(ValueError):
        ext.parse_method_args(["a=1", "a=2"])


def test_the_generic_module_names_no_non_method_readable_path():
    src = Path(ext.__file__).read_text(encoding="utf-8")
    assert "topology/" not in src and "alphabet.json" not in src and "parent_pos" not in src
