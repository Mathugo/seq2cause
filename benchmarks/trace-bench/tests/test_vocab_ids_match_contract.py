# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""D-SB-6: token ids follow the views' `model-vocab.json` contract exactly —
specials 0–3, base ops in listed order, minted variants in listed order — and
an unminted variant folds to its base op."""

import pytest
from fixture_corpus import MINTED, OPS, fixture_corpus
from tracebench.constants import BOS, EOS, N_SPECIALS, OUTCOME_IDS, PAD, UNK

from seq2causebench.corpus import Corpus
from seq2causebench.vocab import Vocab


def _vocab():
    return Vocab.from_model_vocab(Corpus(fixture_corpus("latent"), "end", "request").vocab_json())


def test_ids_follow_the_contract():
    v = _vocab()
    assert v.size == N_SPECIALS + len(OPS) + len(MINTED) == 14
    for i, op in enumerate(sorted(OPS)):
        assert v.id_of(op, OUTCOME_IDS["ok"]) == (N_SPECIALS + i, False)
        assert v.token_string(N_SPECIALS + i) == f"{op}:ok"
    for k, (op, o) in enumerate(sorted(MINTED)):
        tid = N_SPECIALS + len(OPS) + k
        assert v.id_of(op, o) == (tid, False)
        assert v.op_of(tid) == op and v.outcome_of(tid) == o
        assert v.token_string(tid).startswith(f"{op}:")
    assert [v.token_string(t) for t in (PAD, BOS, EOS, UNK)] == ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def test_unminted_variant_folds_to_base_op():
    v = _vocab()
    assert v.id_of(5, OUTCOME_IDS["err"]) == (v.base_id(5), True)
    assert v.id_of(0, OUTCOME_IDS["4xx"]) == (v.base_id(0), True)
    ids, folded = v.encode(
        [4, 2, 5, 0],
        [OUTCOME_IDS["err"], OUTCOME_IDS["ok"], OUTCOME_IDS["err"], OUTCOME_IDS["4xx"]],
    )
    assert folded == 2 and ids[2] == v.base_id(5) and ids[3] == v.base_id(0)


def test_real_and_predictable_ids():
    v = _vocab()
    assert list(v.real_ids) == list(range(N_SPECIALS, v.size))
    assert v.predictable_ids == [EOS, *range(N_SPECIALS, v.size)]
    assert not v.is_real(BOS) and v.is_real(N_SPECIALS)
    assert len(v.tokens()) == len(OPS) + len(MINTED) and "3:slow" in v.tokens()


def test_contract_violations_are_refused():
    d = Corpus(fixture_corpus("latent"), "end", "request").vocab_json()
    bad = dict(d, vocab_size=d["vocab_size"] + 1)
    with pytest.raises(ValueError):
        Vocab.from_model_vocab(bad)
    bad = dict(d, specials={"PAD": 0, "BOS": 2, "EOS": 1, "UNK": 3})
    with pytest.raises(ValueError):
        Vocab.from_model_vocab(bad)
    bad = dict(d, variants=d["variants"] + [{"op_id": 99, "outcome": "err", "outcome_id": 3}])
    with pytest.raises(ValueError):
        Vocab.from_model_vocab(dict(bad, vocab_size=bad["vocab_size"] + 1))
