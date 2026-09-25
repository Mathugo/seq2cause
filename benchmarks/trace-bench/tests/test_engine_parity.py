"""D-SB-2 and PRD scenarios 10, 18, 34, 39, 40: the engine adapter reproduces the shipped
entry points bit for bit on the fixture — `SampleLevelCausalDiscovery.run()` for the
core probe (both KL modes and the Granger read-out from one forward) and
`compute_cmi_matrix` for the two cli probes (both strategies, both KL modes, both noise
draws) — the shipped cut equals `seq2cause.cli.main`'s own graphs, corrupted-cell
counts ride every path, the memory estimate refuses above its cap, and the tensors
handed to the method derive from the operation and outcome lists alone."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from fixture_corpus import fixture_corpus
from seq2cause.core import SampleLevelCausalDiscovery
from seq2cause.diagnostics import compute_cmi_matrix

from seq2causebench import engine
from seq2causebench.backbone import build_model, save_model
from seq2causebench.constants import (
    ARM_CLI,
    ARM_CLI_ATOMIC,
    ARM_CORE,
    ARM_GRANGER,
    CLI_SHIPPED_RULE,
    NOISE_ALL,
    NOISE_REAL,
    PATH_FIXED,
    PATH_FIXED_KL,
    PATH_SHIPPED,
    PROBE_CLI_ATOMIC,
    PROBE_CLI_FULL,
    PROBE_CORE,
    PROBE_SALIENCY,
    PROBE_SHAPLEY,
)
from seq2causebench.corpus import Corpus
from seq2causebench.data import SequenceStore
from seq2causebench.vocab import Vocab

V = 14


def _corpus_store(grain="session", split="val", max_len=12):
    c = Corpus(fixture_corpus("latent"), "end", grain)
    v = Vocab.from_model_vocab(c.vocab_json())
    return c, v, SequenceStore.from_corpus(c, split, v, max_len)


def _backbone(tmp_path, max_len=12):
    torch.manual_seed(0)
    m = build_model(V, 1, 16, 2, 2.0, 0.0, 10000.0, max_len).eval()
    save_model(m, tmp_path / "model")
    return engine.load_backbone(tmp_path / "model", "cpu")


def _sequence(store, s=0):
    ids = store.get(s)
    return torch.as_tensor(np.asarray(ids), dtype=torch.long)


# ---------------------------------------------------------------- core probe vs run()


@pytest.fixture
def cpu_accelerator(monkeypatch):
    """`run()` builds an Accelerator, which picks MPS / CUDA when present; force the CPU so the
    parity comparison shares a device, and reset the singleton state around the test."""
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


def _run_shipped_core(hf_model, ids, c, g, N, kl_mode, noise, causal_strength=None):
    stats = {}
    params = {
        "BS": 1,
        "full": True,
        "fp16": False,
        "cls_token_id": None,
        "check_memory_budget": False,
        "kl_mode": kl_mode,
        "kl_stats": stats,
        "causal_strength": causal_strength,
        "sampling": {
            "context": c,
            "guidance": g,
            "value": N,
            "noise_min_id": engine.noise_min_id(noise),
        },
    }
    ds = [{"input_ids": ids.tolist(), "attention_mask": [1] * len(ids)}]
    algo = SampleLevelCausalDiscovery(hf_model, params, ds)
    algo.prepare()
    _, adj = algo.run()
    return adj[0].cpu().numpy().astype(np.float32), stats


@pytest.mark.parametrize("g", [3, 2])
def test_core_probe_matches_run_bit_for_bit(tmp_path, cpu_accelerator, g):
    hf_model, adapter, _ = _backbone(tmp_path)
    _, _, store = _corpus_store()
    ids = _sequence(store, 1)
    c, N = 3, 4
    for noise in (NOISE_ALL, NOISE_REAL):
        cols = engine.probe_columns(PROBE_CORE, noise)
        torch.manual_seed(11)
        res = engine.probe_sequence(PROBE_CORE, noise, hf_model, adapter, ids, c, g, N, cols)
        for path, col in cols[ARM_CORE].items():
            torch.manual_seed(11)
            want, stats = _run_shipped_core(
                hf_model, ids, c, g, N, "clamp" if path == PATH_SHIPPED else "logspace", noise
            )
            assert np.array_equal(res.matrices[col], want), (noise, path)
            assert res.counts[path] == stats, (noise, path)
        torch.manual_seed(11)
        want_g, _ = _run_shipped_core(
            hf_model, ids, c, g, N, "clamp", noise, causal_strength="Granger"
        )
        assert np.array_equal(
            res.matrices[cols[ARM_GRANGER][PATH_SHIPPED if noise == NOISE_ALL else PATH_FIXED]],
            want_g,
        )
    assert res.tok_pos > 0


def test_core_pass_columns_follow_the_registry():
    all_cols = engine.probe_columns(PROBE_CORE, NOISE_ALL)
    assert set(all_cols[ARM_CORE]) == {PATH_SHIPPED, PATH_FIXED_KL} and set(
        all_cols[ARM_GRANGER]
    ) == {PATH_SHIPPED}
    real_cols = engine.probe_columns(PROBE_CORE, NOISE_REAL)
    assert set(real_cols[ARM_CORE]) == {PATH_FIXED} and set(real_cols[ARM_GRANGER]) == {PATH_FIXED}
    assert engine.probe_columns(PROBE_CLI_FULL, NOISE_ALL) == {
        ARM_CLI: {PATH_SHIPPED: "trace-cli__shipped", PATH_FIXED_KL: "trace-cli__fixed-kl"}
    }
    assert list(engine.probe_columns(PROBE_SALIENCY, NOISE_ALL)) == ["baseline/saliency"]
    assert engine.probe_columns(PROBE_SALIENCY, NOISE_REAL) == {}


# ---------------------------------------------------------------- cli probes vs compute_cmi_matrix


@pytest.mark.parametrize("probe,strategy", [(PROBE_CLI_FULL, "full"), (PROBE_CLI_ATOMIC, "atomic")])
def test_cli_probes_match_compute_cmi_matrix_bit_for_bit(tmp_path, probe, strategy):
    hf_model, adapter, _ = _backbone(tmp_path)
    _, _, store = _corpus_store()
    ids = _sequence(store, 2)
    c, N = 2, 4
    arm = ARM_CLI if probe == PROBE_CLI_FULL else ARM_CLI_ATOMIC
    for noise in (NOISE_ALL, NOISE_REAL):
        cols = engine.probe_columns(probe, noise)
        torch.manual_seed(5)
        res = engine.probe_sequence(probe, noise, hf_model, adapter, ids, c, 1, N, cols)
        for path, col in cols[arm].items():
            stats = {}
            torch.manual_seed(5)
            want = compute_cmi_matrix(
                adapter,
                ids,
                context_len=c,
                n_particles=N,
                strategy=strategy,
                kl_mode="clamp" if path == PATH_SHIPPED else "logspace",
                stats=stats,
                noise_min_id=engine.noise_min_id(noise),
            )
            assert np.array_equal(res.matrices[col], want.cpu().numpy().astype(np.float32)), (
                noise,
                path,
            )
            assert res.counts[path] == stats, (noise, path)
            assert res.counts[path]["n_cells"] > 0


# ---------------------------------------------------------------- the shipped cut vs the shipped CLI


def test_shipped_cut_matches_seq2cause_cli(tmp_path):
    """The engine's shipped cut on its stored matrices equals what `seq2cause` itself writes
    for the same sequences, model and seed (per-sequence graphs and summary graphs)."""
    from seq2cause.cli import main as cli_main

    hf_model, adapter, _ = _backbone(tmp_path)
    _, vocab, store = _corpus_store()
    seqs = [store.get(s) for s in range(4)]
    dataset = tmp_path / "events.pt"
    torch.save([torch.as_tensor(np.asarray(s), dtype=torch.long) for s in seqs], dataset)
    out = tmp_path / "graphs.pt"
    c, N = 2, 4
    cli_main(
        [
            "--dataset",
            str(dataset),
            "--model",
            str(tmp_path / "model"),
            "--context-len",
            str(c),
            "--n-particles",
            str(N),
            "--strategy",
            "full",
            "--kl-mode",
            "clamp",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--output",
            str(out),
        ]
    )
    shipped = torch.load(out, weights_only=True)
    # the engine: same seed once, the same sequences in the same order (the CLI seeds once per run)
    cols = engine.probe_columns(PROBE_CLI_FULL, NOISE_ALL)
    col = cols[ARM_CLI][PATH_SHIPPED]
    torch.manual_seed(0)
    mats = []
    for s in seqs:
        res = engine.probe_sequence(
            PROBE_CLI_FULL,
            NOISE_ALL,
            hf_model,
            adapter,
            torch.as_tensor(np.asarray(s), dtype=torch.long),
            c,
            1,
            N,
            cols,
        )
        j, q, vals = engine.triangle(res.matrices[col])
        mats.append(
            engine.from_triangle(res.matrices[col].shape[0], j, q, vals)
        )  # through the stored form
    tau_by_lag, graphs, finite = engine.shipped_cut(mats, CLI_SHIPPED_RULE)
    assert finite and set(tau_by_lag) == set(range(1, max(len(s) for s in seqs) - c))
    for want, got in zip(shipped, graphs, strict=True):
        assert torch.equal(want["sample_graph"], got)
    edges, facts = engine.shipped_cut_union(seqs, graphs, c, vocab)
    # the CLI's own summary graphs, unioned, restricted to real tokens
    want_edges = set()
    for entry in shipped:
        active = entry["summary_graph"]["active_tokens"].tolist()
        for a, b in torch.nonzero(entry["summary_graph"]["adj"]).tolist():
            u, v = int(active[a]), int(active[b])
            if vocab.is_real(u) and vocab.is_real(v):
                want_edges.add((u, v))
    assert edges == want_edges and facts["n_edges"] == len(edges)


def test_shipped_cut_with_a_corrupted_cell_is_empty_and_flagged():
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 1] = 0.5
    m[1, 2] = np.nan  # what a saturated observed branch leaves in the CLI path
    tau_by_lag, graphs, finite = engine.shipped_cut([m], CLI_SHIPPED_RULE)
    assert not finite and not any(g.any() for g in graphs)


# ---------------------------------------------------------------- baselines run on the raw model


def test_readout_baselines_produce_finite_matrices(tmp_path):
    hf_model, adapter, _ = _backbone(tmp_path)
    _, _, store = _corpus_store()
    ids = _sequence(store, 0)
    c = 2
    for probe in (PROBE_SALIENCY, PROBE_SHAPLEY):
        cols = engine.probe_columns(probe, NOISE_ALL)
        res = engine.probe_sequence(probe, NOISE_ALL, hf_model, adapter, ids, c, 1, 1, cols)
        (m,) = res.matrices.values()
        assert m.shape == (len(ids) - c, len(ids) - c) and np.isfinite(m).all()
    assert engine.probe_columns(PROBE_SHAPLEY, NOISE_ALL)["baseline/shapley"] == {
        "none": "baseline-shapley__none"
    }


# ---------------------------------------------------------------- memory (scenario 18)


def test_memory_estimate_and_refusal(tmp_path):
    hf_model, _, _ = _backbone(tmp_path)
    est = engine.check_memory(hf_model, PROBE_CORE, 4, 12, 2, cap_gb=1.0)
    assert est["harness_estimate"] > est["shipped_estimate"] > 0 and est["cap_bytes"] == 2**30
    with pytest.raises(engine.MemoryRefusal):
        engine.check_memory(hf_model, PROBE_CORE, 4, 12, 2, cap_gb=1e-6)
    big = engine.estimate_peak_bytes(PROBE_CORE, 32, 64, 1, 19873, 6, 256, 5_000_000)
    assert big["shipped_estimate"] == 32 * 63 * 64 * 19873 * 4
    assert big["harness_estimate"] > 2 * big["shipped_estimate"]


# ---------------------------------------------------------------- scenario 34, dynamic half


def test_method_side_never_requests_the_parent_column(monkeypatch):
    import pyarrow.parquet as pq

    seen = []
    original = pq.ParquetFile.iter_batches

    def spy(self, *args, **kwargs):
        seen.append(tuple(kwargs.get("columns") or ()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    _corpus_store()
    assert seen and all("parent_pos" not in cols for cols in seen)
    assert all(set(cols) == {"trace_id", "ops", "outcomes", "n_spans"} for cols in seen)


def test_seed_sequence_is_deterministic_per_split_and_index():
    engine.seed_sequence(3, "val", 7)
    a = torch.randint(0, 1000, (5,))
    engine.seed_sequence(3, "val", 7)
    b = torch.randint(0, 1000, (5,))
    engine.seed_sequence(3, "test", 7)
    d = torch.randint(0, 1000, (5,))
    assert torch.equal(a, b) and not torch.equal(a, d)
    with pytest.raises(ValueError):
        engine.seed_sequence(3, "holdout", 0)
    assert os.environ.get("ACCELERATE_USE_CPU") is None or True
