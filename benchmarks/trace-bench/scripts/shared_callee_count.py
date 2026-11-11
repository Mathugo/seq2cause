"""Count the sample-level pairs the recorded-parent rule drops at a shared callee (score side).

An operation with several callers is invoked once per request and records one parent, yet every
caller's outcome depends on it. `seqtruth.induce` marks `(j, q)` directed truth only when
`parent_pos[o_j] == o_q`, so the pair (shared callee, another caller present in the request) is
never truth. This script re-derives the exact sample of a landed per-sequence cell, re-induces its
truth with the landed code, checks the totals against the landed `seqscore.json`, and counts:

- dropped pairs: candidate pairs `(j, q)` of one request where the operation of `q` is a
  call-graph caller of the operation of `j` and the operation of `j`'s recorded parent is not;
- dropped truth pairs: those whose token pair is a directed target edge at the floor (the pairs a
  call-graph rule would add to the truth);
- reversed occurrences: the same dependence where the callee is ordered after the other caller,
  which no rule over `j < q` candidates can reach.

It reads the views' parent column and the graphs, so it is a score-side tool; it runs no method
and writes one JSON record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seq2causebench.corpus import Corpus
from seq2causebench.data import SequenceStore
from seq2causebench.floorprior import column_to_op, op_pairs_from_prior, prior_rel
from seq2causebench.floorseq import stored_sequences
from seq2causebench.record import read_json
from seq2causebench.seqtruth import induce, load_targets, roots, rows_by_trace_id
from seq2causebench.vocab import Vocab

CELL_FMT = "trace/core/shipped/frozen/{grain}"


def find_read(seed_dir, cell):
    """The landed discover read that holds `cell`: `(run dir, results, arguments)`."""
    hits = []
    for res in sorted((Path(seed_dir) / "test-read").glob("discover-*/run/results.json")):
        disc = read_json(res)
        if cell in disc.get("cells", {}):
            hits.append((res.parent.parent, disc))
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} landed reads hold cell {cell!r} under {seed_dir}")
    run_dir, disc = hits[0]
    return run_dir, disc, read_json(run_dir / "run" / "arguments.json")["arguments"]


def call_edges(corpus_dir, alphabet, instantiation):
    """`{(callee_op, caller_op): critical | None}` from the shipped prior; criticality from the
    instantiation record when one is given (its edge set must equal the prior's)."""
    pairs = op_pairs_from_prior(read_json(Path(corpus_dir) / prior_rel()), column_to_op(alphabet))
    edges = dict.fromkeys(pairs)
    facts = {"n_call_edges": len(edges), "criticality": instantiation is not None}
    if instantiation is not None:
        inst = {}
        for e in read_json(instantiation)["topology"]["edges"]:
            k = (int(e["callee"]), int(e["caller"]))
            inst[k] = bool(e["critical"]) or inst.get(k, False)
        facts["n_instantiation_edges"] = len(inst)
        facts["n_prior_not_in_instantiation"] = len(set(edges) - set(inst))
        facts["n_instantiation_not_in_prior"] = len(set(inst) - set(edges))
        edges = {k: inst.get(k) for k in edges}
    callers = {}
    for callee, _caller in edges:
        callers[callee] = callers.get(callee, 0) + 1
    facts["n_callee_ops"] = len(callers)
    facts["n_shared_callee_ops"] = sum(1 for v in callers.values() if v > 1)
    return edges, facts


def count_sequence(ids, ops, parent_pos, c, vocab, grain, targets, max_lag, edges, truth):
    """The dropped / reversed counts of one stored sequence, on the landed candidate set."""
    lc = len(ids) - c
    kept = min(len(ops), len(ids) - 2)
    tok, op_index = {}, {}
    for t in range(lc):
        o = c + t - 1
        if 0 <= o < kept and vocab.is_real(int(ids[c + t])):
            tok[t] = vocab.token_string(int(ids[c + t]))
            op_index[t] = o
    root = roots(parent_pos) if grain == "session" else [0] * len(ops)
    occ = {}
    for o, op in enumerate(ops):
        occ[(root[o], op)] = occ.get((root[o], op), 0) + 1

    def parent_op(o):
        return ops[parent_pos[o]] if parent_pos[o] >= 0 else None

    k = dict.fromkeys(
        (
            "dropped",
            "dropped_truth",
            "dropped_truth_critical",
            "dropped_truth_noncritical",
            "dropped_truth_retry",
            "reversed",
            "reversed_truth",
        ),
        0,
    )
    for j, q in truth["candidates"]:
        oj, oq = op_index[j], op_index[q]
        if root[oj] != root[oq]:
            continue
        # forward: j the callee, q a caller that is not the recorded parent's operation
        e = (ops[oj], ops[oq])
        if e in edges and parent_op(oj) != ops[oq]:
            assert (j, q) not in truth["directed"]
            k["dropped"] += 1
            if (tok[j], tok[q]) in targets["directed"]:
                k["dropped_truth"] += 1
                if edges[e] is True:
                    k["dropped_truth_critical"] += 1
                elif edges[e] is False:
                    k["dropped_truth_noncritical"] += 1
                if occ[(root[oj], ops[oj])] > 1 or occ[(root[oq], ops[oq])] > 1:
                    k["dropped_truth_retry"] += 1
        # reversed: the caller j is ordered before its callee q, and is not q's recorded parent
        r = (ops[oq], ops[oj])
        if r in edges and parent_op(oq) != ops[oj]:
            k["reversed"] += 1
            if (tok[q], tok[j]) in targets["directed"]:
                k["reversed_truth"] += 1
    return k


def run(args):
    grain = args.grain
    cell = CELL_FMT.format(grain=grain)
    run_dir, disc, args_rec = find_read(args.seed_dir, cell)
    landed = read_json(Path(args.seed_dir) / cell / "seqscore" / "seqscore.json")
    c, max_lag, ordering = int(disc["c"]), int(landed["max_lag"]), str(args_rec["ordering"])
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
    stored, n_skipped = stored_sequences(store, c, int(args_rec["cells_sequences"]))
    got = {"n_sequences": len(store), "n_skipped_short": n_skipped, "n_stored": len(stored)}
    for key, v in got.items():
        if disc.get(key) != v:
            raise ValueError(f"re-derived {key} = {v}, the read recorded {disc.get(key)}")
    targets = load_targets(args.corpus, grain)
    edges, facts = call_edges(args.corpus, targets["alphabet"], args.instantiation)
    rows = rows_by_trace_id(corpus, disc["split"], [store.trace_ids[s] for s in stored])

    total = {}
    n_truth = n_cand = n_with_cand = n_any = n_any_truth = 0
    for s in stored:
        ids = np.asarray(store.get(s))
        ops, outcomes, parent_pos = rows[store.trace_ids[s]]
        truth = induce(ids, ops, outcomes, parent_pos, c, vocab, grain, targets, max_lag)
        if not truth["candidates"]:
            continue
        n_with_cand += 1
        n_truth += len(truth["directed"])
        n_cand += len(truth["candidates"])
        k = count_sequence(ids, ops, parent_pos, c, vocab, grain, targets, max_lag, edges, truth)
        for name, v in k.items():
            total[name] = total.get(name, 0) + v
        n_any += k["dropped"] > 0
        n_any_truth += k["dropped_truth"] > 0

    pooled = landed["pooled"]
    check = {
        "n_truth_directed": (n_truth, pooled["n_truth_directed"]),
        "n_candidates": (n_cand, pooled["n_candidates"]),
        "n_sequences": (len(stored), landed["n_sequences"]),
        "n_skipped_short": (len(stored) - n_with_cand, landed["n_skipped_short"]),
    }
    bad = {key: v for key, v in check.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"not the landed cell's truth (recomputed, landed): {bad}")
    return {
        "schema": "seq2causebench/shared-callee-count@1",
        "cell": cell,
        "corpus_id": landed["corpus_id"],
        "grain": grain,
        "split": disc["split"],
        "ordering": ordering,
        "c": c,
        "max_lag": max_lag,
        "floor": targets["floor"],
        "model_sha256": landed["model_sha256"],
        "read": run_dir.name,
        "call_graph": facts,
        "n_sequences": len(stored),
        "n_sequences_with_candidates": n_with_cand,
        "n_candidates": n_cand,
        "n_truth_directed": n_truth,
        "n_sequences_dropped": int(n_any),
        "n_sequences_dropped_truth": int(n_any_truth),
        **{f"n_{name}": int(v) for name, v in total.items()},
        "share_sequences_dropped_truth": n_any_truth / n_with_cand,
        "share_dropped_truth_of_truth": total["dropped_truth"] / n_truth,
        "share_dropped_truth_of_candidates": total["dropped_truth"] / n_cand,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True, help="a corpus with its score tier pulled")
    p.add_argument("--seed-dir", required=True, help="results/<rung>/latent/seed=<k> (landed)")
    p.add_argument("--grain", required=True, choices=("request", "session"))
    p.add_argument(
        "--instantiation",
        required=True,
        help="the corpus's instantiation.json (edge criticality), or 'none'",
    )
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.instantiation == "none":
        args.instantiation = None
    result = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k.startswith(("n_", "share_"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
