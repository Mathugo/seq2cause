# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""Report the substrate the method can see of one corpus view (PRD Behavior, "Preparing").

    python -m seq2causebench.prepare --corpus <dir> --ordering end --grain session \
        --max-len 64 --output-folder <run-dir>

Reads only through the corpus adapter (the raw feed and the views): rows per
split, vocabulary size, the count of unminted variants folded to their base
operation (D-SB-6), `n_spans` quantiles, and the truncation rate at
`--max-len` (D-SB-7). Writes `prepare.json` beside the run record.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from .constants import GRAINS, ORDERINGS, PREPARE_JSON, SPLITS
from .corpus import Corpus
from .data import SequenceStore
from .entropy import order_k_floor
from .log import log
from .record import RunRecord, write_json
from .vocab import Vocab

QUANTILES = (0.5, 0.9, 0.99)


def split_stats(corpus, vocab, split, max_len):
    n = 0
    spans = []
    folded_positions = 0
    folded_pairs = Counter()
    truncated = 0
    tokens_total = 0
    tokens_kept = 0
    for _tid, ops, outcomes, n_spans in corpus.sequences(split):
        n += 1
        spans.append(n_spans)
        _ids, folded = vocab.encode(ops, outcomes)
        folded_positions += folded
        if folded:
            for op, o in zip(ops, outcomes, strict=False):
                if vocab.id_of(op, o)[1]:
                    folded_pairs[(op, o)] += 1
        tokens_total += len(ops)
        kept = min(len(ops), max_len)
        tokens_kept += kept
        truncated += len(ops) > max_len
    a = np.asarray(spans, dtype=np.int64) if spans else np.zeros(0, dtype=np.int64)
    q = {f"p{int(x * 100)}": (float(np.quantile(a, x)) if len(a) else None) for x in QUANTILES}
    return {
        "rows": n,
        "n_spans": {
            **q,
            "max": int(a.max()) if len(a) else None,
            "mean": float(a.mean()) if len(a) else None,
        },
        "folded_positions": folded_positions,
        "folded_pairs_distinct": len(folded_pairs),
        "folded_pairs": sorted([[int(op), int(o), int(c)] for (op, o), c in folded_pairs.items()]),
        "truncated_sequences": truncated,
        "truncation_rate": (truncated / n) if n else None,
        "tokens_total": tokens_total,
        "tokens_dropped": tokens_total - tokens_kept,
    }


def prepare(corpus_dir, ordering, grain, max_len, entropy_order):
    corpus = Corpus(corpus_dir, ordering, grain)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    stats = corpus.export_stats()
    out = {
        "corpus_dir_name": corpus.dir.name,
        "view": corpus.view,
        "ordering": ordering,
        "grain": grain,
        "max_len": max_len,
        "vocab": {
            "size": vocab.size,
            "n_ops": vocab.n_ops,
            "n_variants": len(vocab.variants),
            "n_real_tokens": len(vocab.real_ids),
            "export_stats_vocab_size": stats.get("vocab_size"),
        },
        "export_stats_rows": stats.get("rows"),
        "splits": {},
    }
    for split in SPLITS:
        out["splits"][split] = split_stats(corpus, vocab, split, max_len)
        log(
            {
                "event": "prepare_split",
                "split": split,
                **{k: v for k, v in out["splits"][split].items() if k != "folded_pairs"},
            }
        )
    train = SequenceStore.from_corpus(corpus, "train", vocab, max_len)
    out["entropy"] = order_k_floor(train, entropy_order, vocab.size)
    out["entropy"]["log_alphabet"] = math.log(len(vocab.predictable_ids))
    log({"event": "prepare_entropy", **out["entropy"]})
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grain", required=True, choices=GRAINS)
    p.add_argument(
        "--max-len", required=True, type=int, help="cap on real tokens per sequence (D-SB-7)"
    )
    p.add_argument(
        "--entropy-order",
        required=True,
        type=int,
        help="order k of the n-gram entropy floor (D-SB-13)",
    )
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "prepare", vars(args)) as rec:
        res = prepare(args.corpus, args.ordering, args.grain, args.max_len, args.entropy_order)
        write_json(rec.out_dir / PREPARE_JSON, res)
        rec.finish(
            {
                "prepare_json": PREPARE_JSON,
                "vocab_size": res["vocab"]["size"],
                "rows": {s: res["splits"][s]["rows"] for s in SPLITS},
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
