# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The token space: the shipped `model-vocab.json` of a view, as-is (D-SB-6).

Ids follow the benchmark's consumer contract: the four specials 0–3
(PAD, BOS, EOS, UNK), then every base operation in listed order — the token
`<op>:ok` — then every minted non-OK variant in listed order. A non-OK
outcome the correlator did not mint (below its `min_count`) folds to the base
operation; the caller counts folds. The scoring token of a real id is
`f"{op}:{outcome}"`, which is exactly the node name the benchmark's scorer
uses, so a prediction over these ids needs no mapping table.
"""

from __future__ import annotations

from .constants import BOS, EOS, N_SPECIALS, OUTCOME_IDS, OUTCOME_NAMES, OUTCOME_OK, PAD, UNK

SPECIAL_NAMES = {PAD: "<PAD>", BOS: "<BOS>", EOS: "<EOS>", UNK: "<UNK>"}


class Vocab:
    def __init__(self, base_ops, variants):
        self.base_ops = list(base_ops)  # [{id, service, name, kind}] in listed order
        self.variants = list(variants)  # [{op_id, outcome, outcome_id}] in listed order
        self._base_index = {op["id"]: i for i, op in enumerate(self.base_ops)}
        if len(self._base_index) != len(self.base_ops):
            raise ValueError("duplicate op id in base_ops")
        self._variant_index = {}
        for k, v in enumerate(self.variants):
            key = (v["op_id"], v["outcome_id"])
            if key in self._variant_index:
                raise ValueError(f"duplicate variant {key}")
            if v["outcome_id"] == OUTCOME_OK or OUTCOME_NAMES[v["outcome_id"]] != v["outcome"]:
                raise ValueError(f"variant {v} is not a named non-OK outcome")
            if v["op_id"] not in self._base_index:
                raise ValueError(f"variant {v} names an op that is not a base op")
            self._variant_index[key] = N_SPECIALS + len(self.base_ops) + k
        self.size = N_SPECIALS + len(self.base_ops) + len(self.variants)
        # id -> (op, outcome_id) for real ids
        self._decode = {}
        for i, op in enumerate(self.base_ops):
            self._decode[N_SPECIALS + i] = (op["id"], OUTCOME_OK)
        for (op, o), tid in self._variant_index.items():
            self._decode[tid] = (op, o)

    @classmethod
    def from_model_vocab(cls, d):
        specials = d["specials"]
        expected = {"PAD": PAD, "BOS": BOS, "EOS": EOS, "UNK": UNK}
        if specials != expected or d["n_specials"] != N_SPECIALS:
            raise ValueError(
                f"special-token contract mismatch: {specials} / n_specials={d['n_specials']}"
            )
        v = cls(d["base_ops"], d["variants"])
        if v.size != d["vocab_size"]:
            raise ValueError(
                f"vocab_size {d['vocab_size']} != {v.size} (specials + base ops + variants)"
            )
        return v

    # --- encoding -------------------------------------------------------------------------------
    def base_id(self, op):
        return N_SPECIALS + self._base_index[op]

    def id_of(self, op, outcome_id):
        """Token id of an `(op, outcome)` pair; an unminted non-OK variant folds to the base op.
        Returns `(id, folded)`."""
        if outcome_id == OUTCOME_OK:
            return self.base_id(op), False
        tid = self._variant_index.get((op, outcome_id))
        if tid is None:
            return self.base_id(op), True
        return tid, False

    def encode(self, ops, outcomes):
        """Ids for one sequence plus the number of folded positions."""
        ids, folded = [], 0
        for op, o in zip(ops, outcomes, strict=False):
            tid, f = self.id_of(op, o)
            ids.append(tid)
            folded += f
        return ids, folded

    # --- decoding -------------------------------------------------------------------------------
    def is_real(self, tid):
        return N_SPECIALS <= tid < self.size

    def op_of(self, tid):
        return self._decode[tid][0]

    def outcome_of(self, tid):
        return self._decode[tid][1]

    def token_string(self, tid):
        """`"<op>:<outcome>"` for a real id — the scorer's node name; `<PAD>`-style names for specials."""
        if tid in SPECIAL_NAMES:
            return SPECIAL_NAMES[tid]
        op, o = self._decode[tid]
        return f"{op}:{OUTCOME_NAMES[o]}"

    @property
    def real_ids(self):
        """Every event token: what the uniform do-operator draws from (D-SB-5)."""
        return range(N_SPECIALS, self.size)

    @property
    def predictable_ids(self):
        """What the model may emit as a next token: every event token and EOS."""
        return [EOS, *self.real_ids]

    @property
    def n_ops(self):
        return len(self.base_ops)

    def tokens(self):
        return [self.token_string(t) for t in self.real_ids]

    def __len__(self):
        return self.size


def outcome_id(name):
    return OUTCOME_IDS[name]
