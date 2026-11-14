"""Score the model-free sample-level rows of one read under both truth rules (score side).

The call-graph reference (`floor/topology`, broadcast), the frequency baseline (`floor/bigram`,
broadcast) and predict-every-pair need no stored score matrix: their tables are recomputed from the
corpus exactly as their landed test read computed them. This script scores them on the stored
sequences of a discover read under `first-caller-parent` — refusing unless that reproduces the
landed floor records — and under `forced-ok-counterfactual`, from a truth file `truthbuild` wrote.
It is the first look at the new rule before any stored matrix is pulled; the arms go through
`seqscore` once the rule is wired into it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seq2causebench.constants import ARM_FLOOR_BIGRAM, ARM_FLOOR_TOPOLOGY, RESULTS_JSON, RUN_DIR
from seq2causebench.corpus import Corpus
from seq2causebench.data import SequenceStore
from seq2causebench.floorseq import (
    ROW_BROADCAST,
    floor_matrix,
    floor_table,
    stored_sequences,
    window_tokens,
)
from seq2causebench.record import read_json
from seq2causebench.seqscore import aggregate_read
from seq2causebench.seqtruth import RULE_FIRST_CALLER_PARENT, RULE_FORCED_OK, truth_for_read
from seq2causebench.vocab import Vocab

FLOORS = {"call_graph": ARM_FLOOR_TOPOLOGY, "frequency": ARM_FLOOR_BIGRAM}
LANDED_FMT = "{arm}/none/frozen/{grain}/seqscore-broadcast-on-trace-core/seqscore.json"


def run(args):
    read_dir, floor_dir, seed_dir = Path(args.read_dir), Path(args.floor_read), Path(args.seed_dir)
    disc = read_json(read_dir / RUN_DIR / RESULTS_JSON)
    args_rec = read_json(read_dir / RUN_DIR / "arguments.json")["arguments"]
    fres = read_json(floor_dir / RUN_DIR / RESULTS_JSON)
    fargs = read_json(floor_dir / RUN_DIR / "arguments.json")["arguments"]
    grain, c, ordering = str(disc["grain"]), int(disc["c"]), str(args_rec["ordering"])
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
    stored, _ = stored_sequences(store, c, int(args_rec["cells_sequences"]))
    seqrec = {
        "grain": grain,
        "split": disc["split"],
        "sequences": [
            {"seq": int(s), "trace_id": store.trace_ids[s], "length": int(store.lengths[s])}
            for s in stored
        ],
    }
    stored_ids = [store.get(s) for s in stored]
    tables = {}
    for name, arm in FLOORS.items():
        cell = f"{arm}/none/frozen/{grain}"
        tables[name] = (
            floor_table(arm, args.corpus, ordering, grain, {**fargs, "split": fres["split"]}),
            float(fres["cells"][cell]["tau"]),
        )
    out = {
        "corpus_id": disc["corpus_id"],
        "grain": grain,
        "c": c,
        "read": read_dir.name,
        "rules": {},
    }
    for rule in (RULE_FIRST_CALLER_PARENT, RULE_FORCED_OK):
        truth, _targets, _ = truth_for_read(
            args.corpus, ordering, grain, disc["split"], seqrec, stored_ids, c, int(args.max_lag),
            rule=rule, truth_file=args.truth_file,
        )  # fmt: skip
        rows = {}
        for name, (scores, tau) in tables.items():

            def matrix_of(s, ids, scores=scores, truth=truth):
                return floor_matrix(
                    len(ids) - c,
                    truth[s]["candidates"],
                    window_tokens(ids, c, vocab),
                    scores,
                    ROW_BROADCAST,
                )

            agg = aggregate_read(
                stored, stored_ids, truth, matrix_of,
                lambda matrix, cands, tau=tau: {(j, q) for j, q in cands if matrix[j, q] > tau},
            )  # fmt: skip
            rows[name] = {**agg["pooled"]["directed"], "tau": tau}
            base = {
                k: agg[k]
                for k in ("n_sequences", "n_skipped_short", "n_scoreable", "n_unscoreable")
            }
            base["n_candidates"] = agg["pooled"]["n_candidates"]
            base["n_truth_directed"] = agg["pooled"]["n_truth_directed"]
            base["predict_every_pair_f1"] = agg["predict_all"]["pooled"]
            if rule == RULE_FIRST_CALLER_PARENT:
                landed = read_json(seed_dir / LANDED_FMT.format(arm=FLOORS[name], grain=grain))
                want = landed["pooled"]["directed"]
                if any(rows[name][k] != want[k] for k in ("tp", "fp", "fn")):
                    raise ValueError(
                        f"{name}: {rule} does not reproduce the landed record: {rows[name]} vs {want}"
                    )
        out["rules"][rule] = {**base, **rows}
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--seed-dir", required=True, help="results/<rung>/latent/seed=<k> (landed)")
    p.add_argument(
        "--read-dir", required=True, help="the discover read of trace/core/shipped/frozen"
    )
    p.add_argument("--floor-read", required=True, help="the floors' landed test read of this grain")
    p.add_argument("--truth-file", required=True)
    p.add_argument("--max-lag", required=True, type=int)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
