# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""D-SB-7: PAD is masked from attention and the loss, so a sequence's log-probs
and its loss contribution do not depend on how much padding its batch carries;
batching, sampling and encoding behave as specified."""

import numpy as np
import torch
from fixture_corpus import fixture_corpus

from seq2causebench.backbone import build_model, log_probs_at, loss
from seq2causebench.constants import BOS, EOS, PAD
from seq2causebench.corpus import Corpus
from seq2causebench.data import SequenceStore, collate, encode_sequence, select_sequences
from seq2causebench.vocab import Vocab


def _model(V=20):
    torch.manual_seed(1)
    return build_model(V, 2, 32, 4, 2.0, 0.0, 10000.0, max_len=16).eval()


def test_padding_does_not_change_log_probs_or_loss():
    m = _model()
    short = np.array([BOS, 5, 6, 7, EOS])
    long = np.array([BOS, 8, 9, 10, 11, 12, 13, 14, EOS])
    ids_a, pad_a = collate([short])
    ids_b, pad_b = collate([short, long])
    lp_a = log_probs_at(m, ids_a, pad_a)[0, :5]
    lp_b = log_probs_at(m, ids_b, pad_b)[0, :5]
    assert torch.allclose(lp_a, lp_b, atol=1e-5)
    loss_a, n_a = loss(m, ids_a, pad_a)
    ids_s, pad_s = collate([short, short])
    loss_s, n_s = loss(m, ids_s, pad_s)
    assert n_a == 4 and n_s == 8 and torch.allclose(loss_a, loss_s, atol=1e-5)


def test_encode_truncates_and_folds():
    c = Corpus(fixture_corpus("latent"), "end", "request")
    v = Vocab.from_model_vocab(c.vocab_json())
    ids, truncated, folded = encode_sequence(v, [4, 2, 5, 0], [3, 0, 3, 1], max_len=3)
    # folds are counted on the whole sequence (vocabulary coverage), the cap then drops the tail
    assert ids[0] == BOS and ids[-1] == EOS and len(ids) == 5 and truncated and folded == 2
    store = SequenceStore.from_corpus(c, "train", v, max_len=64)
    assert len(store) == 160 and store.n_tokens == sum(store.lengths) and (store.lengths >= 3).all()
    assert store.n_truncated == 0 and store.n_folded > 0
    assert store.get(0)[0] == BOS and store.get(0)[-1] == EOS


def test_head_and_uniform_selection():
    rows = [(f"t{i}", [i], [0], 1) for i in range(100)]
    assert [r[0] for r in select_sequences(iter(rows), 5, "head", np.random.default_rng(0))] == [
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
    ]
    a = select_sequences(iter(rows), 10, "uniform", np.random.default_rng(3))
    b = select_sequences(iter(rows), 10, "uniform", np.random.default_rng(3))
    assert a == b and len(a) == 10 and len({r[0] for r in a}) == 10 and a != rows[:10]
    assert select_sequences(iter(rows), 0, "head", np.random.default_rng(0)) == rows


def test_batches_cover_every_sequence_once_and_bucket_by_length():
    c = Corpus(fixture_corpus("latent"), "end", "session")
    v = Vocab.from_model_vocab(c.vocab_json())
    store = SequenceStore.from_corpus(c, "train", v, 64)
    seen = []
    for b in store.batches(8, np.random.default_rng(0)):
        seen += b
        ids, pad = store.collate(b)
        assert ids.shape[1] == max(store.lengths[b]) and (ids[:, 0] == BOS).all()
        assert ((ids == PAD) == pad).all()
    assert sorted(seen) == list(range(len(store)))
    det = list(store.batches(8))
    assert sorted(x for b in det for x in b) == list(range(len(store)))
    assert all(
        store.lengths[det[i]].max() <= store.lengths[det[i + 1]].min() for i in range(len(det) - 1)
    )
