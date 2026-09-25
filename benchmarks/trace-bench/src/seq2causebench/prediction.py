# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The prediction files the benchmark's scorer consumes (D-SB-10, PRD scenario 6).

Two files per read, because the scorer treats every listed edge as present:

- `prediction-<grain>-<arm>-<path>-<cut>.json` — the edges the cut selects,
  for the structural axes;
- `ranking-<grain>-<arm>-<path>.json` — every scored pair, for AUROC / AP;
  plus `ranking-…-lag<k>.json` per lag for per-lag recall.

Both carry `bidirected: []` and a top-level `structural_limitation` field
naming the assumption every arm here rests on; `annotate` fills in the count
of bidirected truth edges the arm could never have found. Unknown top-level
keys are ignored by the scorer.
"""

from __future__ import annotations

from .constants import STRUCTURAL_LIMITATION
from .record import write_json


def structural_limitation():
    return {
        "assumption": STRUCTURAL_LIMITATION,
        "truth_bidirected_edges": None,
        "note": "no arm here can emit a bidirected edge (temporal precedence and no hidden confounders are "
        "assumptions of the method); its bidirected recall on a latent target is zero by construction",
    }


def edge_list(src, dst, score, vocab):
    return [
        {"src": vocab.token_string(int(u)), "dst": vocab.token_string(int(v)), "score": float(s)}
        for u, v, s in zip(src, dst, score, strict=True)
    ]


def prediction_document(src, dst, score, vocab, **meta):
    return {
        "directed": edge_list(src, dst, score, vocab),
        "bidirected": [],
        "structural_limitation": structural_limitation(),
        "meta": meta,
    }


def write_prediction(out_path, src, dst, score, vocab, **meta):
    """`meta` may carry `path` (the arm's shipped / fixed path), hence the file argument's name."""
    doc = prediction_document(src, dst, score, vocab, **meta)
    write_json(out_path, doc)
    return len(doc["directed"])


__all__ = ["structural_limitation", "edge_list", "prediction_document", "write_prediction"]
