# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""D-SB-13: the order-k floor is a plug-in conditional entropy — zero for a
deterministic process, log V for i.i.d. uniform tokens, non-increasing in k —
and the oracle score is read against it, never against the validation loss."""

import math

import numpy as np
import pytest

from seq2causebench.constants import BOS, EOS
from seq2causebench.data import SequenceStore
from seq2causebench.entropy import oracle_score, order_k_floor


def _store(seqs):
    tokens, offsets = [], [0]
    for s in seqs:
        tokens += [BOS, *s, EOS]
        offsets.append(len(tokens))
    return SequenceStore(
        tokens, offsets, [str(i) for i in range(len(seqs))], 0, 0, [len(s) for s in seqs]
    )


def test_deterministic_process_has_zero_floor():
    store = _store([[4, 5, 6, 7]] * 50)
    for k in (1, 2, 3):
        assert order_k_floor(store, k, 10)["entropy_floor"] == pytest.approx(0.0, abs=1e-12)
    assert (
        order_k_floor(store, 0, 10)["entropy_floor"] > 0.5
    )  # the marginal over 5 symbols is not degenerate


def test_uniform_iid_tokens_approach_log_v():
    rng = np.random.default_rng(0)
    V_real = 8
    seqs = [list(rng.integers(4, 4 + V_real, size=30)) for _ in range(2000)]
    store = _store(seqs)
    h1 = order_k_floor(store, 1, 4 + V_real)
    # events plus EOS: 9 symbols, EOS at 1/31 of positions; plug-in entropy sits a little below the mixture entropy
    p_eos = 1 / 31
    mixture = -(p_eos * math.log(p_eos) + (1 - p_eos) * math.log((1 - p_eos) / V_real))
    assert 0.9 * mixture < h1["entropy_floor"] <= mixture + 1e-9
    assert h1["n_positions"] == 2000 * 31


def test_floor_is_non_increasing_in_k():
    rng = np.random.default_rng(1)
    seqs = []
    for _ in range(500):
        s = [int(rng.integers(4, 9))]
        for _ in range(11):
            s.append(4 + (s[-1] - 4 + int(rng.integers(0, 2))) % 5)  # a Markov chain
        seqs.append(s)
    store = _store(seqs)
    hs = [order_k_floor(store, k, 9)["entropy_floor"] for k in (0, 1, 2, 3)]
    assert hs[0] >= hs[1] >= hs[2] >= hs[3] >= 0
    assert hs[0] - hs[1] > 0.3  # order 1 explains the chain


def test_order_bounds_and_oracle_score():
    store = _store([[4, 5]])
    with pytest.raises(ValueError):
        order_k_floor(store, 4, 10)
    o = oracle_score(val_loss=1.0, entropy_floor=0.5, log_alphabet=2.5, in_regime_below=0.1)
    assert o["eps_hat"] == pytest.approx(0.25) and not o["in_regime"]
    assert oracle_score(0.55, 0.5, 2.5, 0.1)["in_regime"]
