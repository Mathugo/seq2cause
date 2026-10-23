"""The floors on the per-sequence axis (`plans/floors.md` and `plans/per-sequence-rules.md`,
addenda of 2026-10-21; RUN.md D-SB-17): a floor's type-level score broadcast onto the sequences,
window and candidate pairs of a landed arm read, scored by `seqscore`'s own aggregation.

    python -m seq2causebench.floorseq --corpus <dir> \
        --read-dir results/<rung>/latent/seed=<k>/test-read/discover-<grain>-core-all-c2-N2 \
        --reference-cell trace/core/shipped/frozen/<grain> \
        --reference-seqscore results/.../trace/core/shipped/frozen/<grain>/seqscore/seqscore.json \
        --floor-read results/<rung>/latent/seed=<k>/test-read/floors-<grain> \
        --floor-cell floor/bigram/none/frozen/<grain> --row adjacent --max-lag 63 \
        --output-folder results/.../floor/bigram/none/frozen/<grain>/seqscore-adjacent-on-trace-core

Score side, model-free. The arm read's stored sequences are re-derived from its recorded
selection exactly as `discover` chose them and asserted against its record; the floor's score
table is recomputed from the corpus with the floor test read's recorded arguments and asserted
against that read's `n_pairs` / `n_edges`; the cut is the floor's frozen τ (strict). The matrix is
`S[j, q] = score(tok_j → tok_q)` over the candidates (`broadcast`), or that score on `q = j + 1`
only (`adjacent`, the bigram floor). The candidate, truth and scoreable counts and the predict-all
value must equal the arm's landed `seqscore.json`, or the run refuses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .arms import arm_slug, parse_cell_key
from .constants import (
    ARGUMENTS_JSON,
    ARM_FLOOR_BIGRAM,
    ARM_FLOOR_TOPOLOGY,
    CUT_FROZEN,
    FLOOR_ARMS,
    FLOOR_STAGE,
    MODEL_NONE,
    PATH_NONE,
    RESULTS_JSON,
    RUN_DIR,
    SEQSCORE_JSON,
    SEQUENCES_JSON_FMT,
)
from .corpus import Corpus
from .data import SequenceStore
from .floorprior import topology_read
from .floors import bigram_read
from .log import log
from .record import RunRecord, read_json, write_json
from .seqscore import SEQSCORE_SCHEMA, aggregate_read, read_view
from .seqtruth import RULES_VERSION, truth_for_read
from .vocab import Vocab

ROW_BROADCAST = "broadcast"
ROW_ADJACENT = "adjacent"
ROWS = (ROW_BROADCAST, ROW_ADJACENT)
ROWS_OF_FLOOR = {ARM_FLOOR_TOPOLOGY: (ROW_BROADCAST,), ARM_FLOOR_BIGRAM: ROWS}
# the counts that make the floor row and the arm row the same read (per-sequence-rules addendum)
IDENTITY_KEYS = (
    "n_sequences",
    "n_scoreable",
    "n_unscoreable",
    "n_skipped_short",
    "n_links_lost",
    "pooled.n_candidates",
    "pooled.n_truth_directed",
    "predict_all.pooled",
)


class FloorSeqRefusal(RuntimeError):
    pass


def _get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def stored_sequences(store, c, cells_sequences):
    """The store indices `discover` probes and stores: every sequence with at least two window
    positions after `c`, the first `cells_sequences` of them (all when 0)."""
    out, n_skipped = [], 0
    for s in range(len(store)):
        if len(store.get(s)) - c < 2:
            n_skipped += 1
            continue
        if cells_sequences == 0 or len(out) < cells_sequences:
            out.append(s)
    return out, n_skipped


def floor_table(arm, corpus_dir, ordering, grain, floor_args):
    """The floor's score table as its landed test read computed it: `{(src token, dst token): score}`."""
    if arm == ARM_FLOOR_TOPOLOGY:
        table = topology_read(corpus_dir)
    elif arm == ARM_FLOOR_BIGRAM:
        table = bigram_read(
            Corpus(corpus_dir, ordering, grain),
            floor_args["split"],
            int(floor_args["num_sequences"]),
            str(floor_args["sequence_sample"]),
            int(floor_args["seed"]),
            int(floor_args["max_len"]),
        )
    else:
        raise FloorSeqRefusal(f"arm {arm!r} is not a floor arm {FLOOR_ARMS}")
    toks = table["tokens"]
    return {
        (str(toks[a]), str(toks[b])): float(v)
        for a, b, v in zip(table["src"], table["dst"], table["score"], strict=True)
    }


def floor_matrix(lc, candidates, tok, scores, row):
    """The broadcast (or adjacent-only) window matrix of one sequence; 0 off the floor's pairs."""
    m = np.zeros((lc, lc), dtype=np.float64)
    for j, q in candidates:
        if row == ROW_ADJACENT and q != j + 1:
            continue
        m[j, q] = scores.get((tok[j], tok[q]), 0.0)
    return m


def window_tokens(ids, c, vocab):
    """Window index → the token string the per-sequence truth is induced on (`seqtruth.induce`)."""
    return {
        t: vocab.token_string(int(ids[c + t]))
        for t in range(len(ids) - c)
        if vocab.is_real(int(ids[c + t]))
    }


def _assert_read(disc, args_rec, reference_cell, grain):
    if disc.get("status") != "ok" or disc.get("grain") != grain:
        raise FloorSeqRefusal(f"the arm read is not a completed {grain} read")
    if reference_cell not in disc.get("cells", {}):
        raise FloorSeqRefusal(f"the arm read holds no cell {reference_cell}")
    if args_rec.get("max_len") is None or args_rec.get("cells_sequences") is None:
        raise FloorSeqRefusal("the arm read's arguments carry no --max-len / --cells-sequences")


def _assert_floor_read(fres, floor_cell, grain, disc, ordering):
    if fres.get("status") != "ok" or fres.get("stage") != FLOOR_STAGE:
        raise FloorSeqRefusal("the floor read is not a completed floors read")
    if fres.get("grain") != grain or floor_cell not in fres.get("cells", {}):
        raise FloorSeqRefusal(f"the floor read holds no cell {floor_cell}")
    if not fres.get("freeze") or not fres["freeze"].get("sha"):
        raise FloorSeqRefusal("the floor read carries no freeze binding (a test read is required)")
    for k, arm_value in (
        ("split", disc.get("split")),
        ("corpus_id", disc.get("corpus_id")),
        ("ordering", ordering),
    ):
        if fres.get(k) != arm_value:
            raise FloorSeqRefusal(
                f"floor read and arm read differ in {k}: {fres.get(k)!r} vs {arm_value!r}"
            )


def run_floorseq(args, rec):
    arm, path, cut, grain = parse_cell_key(args.floor_cell)
    if arm not in FLOOR_ARMS or path != PATH_NONE or cut != CUT_FROZEN:
        raise FloorSeqRefusal(f"{args.floor_cell} is not a floor's frozen cell")
    if args.row not in ROWS_OF_FLOOR[arm]:
        raise FloorSeqRefusal(f"row {args.row!r} is not registered for {arm} {ROWS_OF_FLOOR[arm]}")
    ref_arm, _rp, _rc, ref_grain = parse_cell_key(args.reference_cell)
    if ref_grain != grain:
        raise FloorSeqRefusal(f"reference cell {args.reference_cell} is not on grain {grain!r}")
    read_dir, floor_dir = Path(args.read_dir), Path(args.floor_read)
    disc = read_json(read_dir / RUN_DIR / RESULTS_JSON)
    args_rec = read_json(read_dir / RUN_DIR / ARGUMENTS_JSON)["arguments"]
    _assert_read(disc, args_rec, args.reference_cell, grain)
    ordering = read_view(args_rec, read_dir)
    fres = read_json(floor_dir / RUN_DIR / RESULTS_JSON)
    fargs = read_json(floor_dir / RUN_DIR / ARGUMENTS_JSON)["arguments"]
    _assert_floor_read(fres, args.floor_cell, grain, disc, ordering)
    ref = read_json(args.reference_seqscore)
    if ref.get("cell") != args.reference_cell or ref.get("corpus_id") != disc["corpus_id"]:
        raise FloorSeqRefusal("the reference seqscore is not the reference cell's on this corpus")
    if int(ref["max_lag"]) != int(args.max_lag) or int(ref["c"]) != int(disc["c"]):
        raise FloorSeqRefusal(
            f"--max-lag {args.max_lag} / c {disc['c']} differ from the reference seqscore's"
        )
    c = int(disc["c"])
    tau = float(fres["cells"][args.floor_cell]["tau"])

    # the floor's table, recomputed as its landed test read computed it
    scores = floor_table(arm, args.corpus, ordering, grain, {**fargs, "split": fres["split"]})
    n_edges = sum(1 for v in scores.values() if v > tau)
    want = fres["cells"][args.floor_cell]
    if len(scores) != int(want["n_pairs"]) or n_edges != int(want["n_edges"]):
        raise FloorSeqRefusal(
            f"the recomputed floor table ({len(scores)} pairs, {n_edges} edges at τ {tau}) is not the "
            f"landed read's ({want['n_pairs']}, {want['n_edges']})"
        )

    # the arm read's stored sequences, re-derived from its recorded selection
    corpus = Corpus(args.corpus, ordering, grain)
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    store = SequenceStore.from_corpus(
        corpus,
        disc["split"],
        vocab,
        int(args_rec["max_len"]),
        n=int(args_rec["num_sequences"]),
        mode=str(args_rec["sequence_sample"]),
        seed=int(args_rec["seed"]),
    )
    stored_seq, n_skipped = stored_sequences(store, c, int(args_rec["cells_sequences"]))
    got = {"n_sequences": len(store), "n_skipped_short": n_skipped, "n_stored": len(stored_seq)}
    for k, v in got.items():
        if disc.get(k) != v:
            raise FloorSeqRefusal(f"re-derived {k} = {v}, the read recorded {disc.get(k)}")
    seqrec = {
        "grain": grain,
        "split": disc["split"],
        "context": c,
        "sequences": [
            {
                "seq": int(s),
                "trace_id": store.trace_ids[s],
                "length": int(store.lengths[s]),
                "n_spans": int(store.n_spans[s]),
            }
            for s in stored_seq
        ],
    }
    seq_file = read_dir / SEQUENCES_JSON_FMT.format(grain=grain)
    if seq_file.exists() and read_json(seq_file)["sequences"] != seqrec["sequences"]:
        raise FloorSeqRefusal("the re-derived sequences differ from the read's sequences record")
    stored_ids = [store.get(s) for s in stored_seq]
    truth, targets, _ = truth_for_read(
        args.corpus, ordering, grain, disc["split"], seqrec, stored_ids, c, int(args.max_lag)
    )

    toks = {}

    def matrix_of(s, ids):
        toks[s] = window_tokens(ids, c, vocab)
        return floor_matrix(len(ids) - c, truth[s]["candidates"], toks[s], scores, args.row)

    agg = aggregate_read(
        stored_seq,
        stored_ids,
        truth,
        matrix_of,
        lambda matrix, cands: {(j, q) for j, q in cands if matrix[j, q] > tau},
    )
    mismatch = {
        k: (_get(agg, k), _get(ref, k)) for k in IDENTITY_KEYS if _get(agg, k) != _get(ref, k)
    }
    if mismatch:
        raise FloorSeqRefusal(f"not the reference cell's read (floor row vs reference): {mismatch}")
    result = {
        "schema": SEQSCORE_SCHEMA,
        "axis": "per-sequence",
        "rules_version": RULES_VERSION,
        "cell": args.floor_cell,
        "arm": arm,
        "path": path,
        "cut": cut,
        "grain": grain,
        "split": disc["split"],
        "corpus_id": disc["corpus_id"],
        "rung": disc.get("rung"),
        "model_sha256": MODEL_NONE,
        "floor": targets["floor"],
        "c": c,
        "max_lag": int(args.max_lag),
        "tau": tau,
        "tau_by_lag": None,
        "n_sequences_split": disc.get("n_sequences"),
        "n_skipped_short_by_read": disc.get("n_skipped_short"),
        "n_stored_by_read": disc.get("n_stored"),
        "row": args.row,
        "reference_cell": args.reference_cell,
        "reference_read": read_dir.name,
        "reference_model_sha256": disc.get("model_sha256"),
        "floor_read_freeze": fres["freeze"],
        "floor_table_pairs": len(scores),
        "floor_table_edges": n_edges,
        **agg,
    }
    write_json(rec.out_dir / SEQSCORE_JSON, result)
    log(
        {
            "event": "floorseq_done",
            "cell": args.floor_cell,
            "row": args.row,
            "reference_cell": args.reference_cell,
            "pooled_f1": result["pooled"]["directed"]["f1"],
            "reference_pooled_f1": ref["pooled"]["directed"]["f1"],
            "predict_all": result["predict_all"]["pooled"],
        }
    )
    return {
        "seqscore": SEQSCORE_JSON,
        "cell": args.floor_cell,
        "row": args.row,
        "reference_cell": args.reference_cell,
        "n_scoreable": agg["n_scoreable"],
        "pooled_directed_f1": result["pooled"]["directed"]["f1"],
    }


def landing_dir_name(row, reference_cell):
    """`seqscore-<row>-on-<reference arm slug>`: beside the floor cell, a directory `report` does not read."""
    return f"seqscore-{row}-on-{arm_slug(parse_cell_key(reference_cell)[0])}"


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument("--read-dir", required=True, help="the landed arm read (its run/ records)")
    p.add_argument(
        "--reference-cell", required=True, help="<arm>/<path>/<cut>/<grain> of that read"
    )
    p.add_argument(
        "--reference-seqscore", required=True, help="the reference cell's landed seqscore.json"
    )
    p.add_argument(
        "--floor-read", required=True, help="the floor's landed test read (its run/ records)"
    )
    p.add_argument("--floor-cell", required=True, help="floor/<name>/none/frozen/<grain>")
    p.add_argument("--row", required=True, choices=ROWS)
    p.add_argument("--max-lag", required=True, type=int, help="the reference seqscore's --max-lag")
    p.add_argument("--output-folder", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "floorseq", vars(args)) as rec:
        res = run_floorseq(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
