# Adapted from alex-chadyuk/trace-cmi-bench@f54776a (MIT) -- see NOTICE
"""The blind validation sweep: one probing pass per (probe, noise, c, N), τ never applied
(PRD Interface "sweep"; scenarios 10, 18, 46).

    python -m seq2causebench.sweep --corpus <dir> --ordering end --grains request --split val \
        --rung xs --model <model dir> --probes core cli-full cli-atomic saliency shapley --noises all real \
        --contexts 1 2 3 --particles-grid 2 8 32 --guidance 3 --num-sequences 0 \
        --shipped-method percentile ... --shipped-min-group-size 8 \
        --max-len 64 --max-lag 63 --sequence-sample head --memory-cap-gb 20 --probe-amp none \
        --prior-rung-record smallest-rung --staging-ledger plans/staging-ledger.json \
        --replica <tfvars> --aws-profile-name default --seed 0 --device cuda --output-folder out/<run>

Refuses `--split test` (the test split is read once, by `discover`, after a committed freeze).
The effective guidance per cell is `g = min(--guidance, c)`; the saliency and Shapley probes have
no particle axis and run once per `c` at `N = 0`; a pass with no read-out under a noise (saliency,
Shapley, and `real` for nothing else) is skipped. Emits `scores-<probe>-<noise>-c<c>-N<N>-<grain>.npz`
per cell and, for the cli probes, `shippedcut-…json` with the tool's own rule fitted on that cell's
matrices (informational: the shipped cut is never selected on validation, but a NaN pooled threshold
is visible before any test read).
"""

from __future__ import annotations

import argparse

from .constants import (
    GRAINS,
    NOISES,
    ORDERINGS,
    PARTICLE_PROBES,
    PROBE_CLI_ATOMIC,
    PROBE_CLI_FULL,
    PROBES,
    SHIPPED_CUT_JSON_FMT,
    SPLITS,
    SWEEP_SCORES_NPZ_FMT,
)
from .corpus import Corpus
from .data import SequenceStore
from .discover import (
    add_probe_knobs,
    add_shipped_rule_knobs,
    is_v019_rule,
    probe_store,
    shipped_rule,
)
from .engine import load_backbone, probe_columns, shipped_cut, shipped_cut_union
from .log import log
from .project import write_scores_npz
from .record import RunRecord, write_json
from .staging import require_prior_rung
from .vocab import Vocab


class SweepRefusal(RuntimeError):
    pass


def run_sweep(args, rec):
    if args.split == "test":
        raise SweepRefusal(
            "sweep runs on validation only; the test split is read once by discover after a committed freeze"
        )
    staging = require_prior_rung(args.prior_rung_record, args.rung, "sweep", args.staging_ledger)
    rec.note(replica=args.replica, aws_profile=args.aws_profile_name)
    if len(args.num_sequences) not in (1, len(args.grains)):
        raise ValueError(
            "--num-sequences takes one value, or one per grain in the order of --grains"
        )
    hf_model, adapter, model_sha = load_backbone(args.model, args.device)
    rule = shipped_rule(args)
    cells = []
    for gi, grain in enumerate(args.grains):
        n_seq = args.num_sequences[0] if len(args.num_sequences) == 1 else args.num_sequences[gi]
        corpus = Corpus(args.corpus, args.ordering, grain)
        if "/" in corpus.corpus_id and not corpus.corpus_id.startswith(args.rung + "/"):
            raise ValueError(f"corpus {corpus.corpus_id!r} is not at rung {args.rung!r}")
        vocab = Vocab.from_model_vocab(corpus.vocab_json())
        if vocab.size != hf_model.config.vocab_size:
            raise ValueError(
                f"{grain}: the view's vocabulary has {vocab.size} ids but the model was trained on {hf_model.config.vocab_size}"
            )
        store = SequenceStore.from_corpus(
            corpus,
            args.split,
            vocab,
            args.max_len,
            n=n_seq,
            mode=args.sequence_sample,
            seed=args.seed,
        )
        for probe in args.probes:
            for noise in args.noises:
                if not probe_columns(probe, noise):
                    continue
                n_grid = args.particles_grid if probe in PARTICLE_PROBES else [0]
                for c in args.contexts:
                    g = min(args.guidance, c)
                    for N in n_grid:
                        keep = (
                            0 if probe in (PROBE_CLI_FULL, PROBE_CLI_ATOMIC) else 1
                        )  # cli probes keep every matrix for the shipped cut
                        scores, _, stored, facts = probe_store(
                            hf_model,
                            adapter,
                            vocab,
                            store,
                            probe,
                            noise,
                            c,
                            g,
                            N,
                            args.max_lag,
                            args.memory_cap_gb,
                            args.probe_amp,
                            args.seed,
                            args.split,
                            keep,
                        )
                        name = SWEEP_SCORES_NPZ_FMT.format(
                            probe=probe, noise=noise, c=c, n=N, grain=grain
                        )
                        write_scores_npz(
                            rec.out_dir / name,
                            scores,
                            probe=probe,
                            noise=noise,
                            c=c,
                            N=N,
                            g=g,
                            grain=grain,
                            split=args.split,
                            ordering=args.ordering,
                            model_sha256=model_sha,
                            corpus_id=corpus.corpus_id,
                            rung=args.rung,
                            n_sequences=facts["n_probed"],
                            max_lag=args.max_lag,
                            probe_amp=args.probe_amp,
                        )
                        cell = {
                            "probe": probe,
                            "noise": noise,
                            "c": c,
                            "N": N,
                            "g": g,
                            "grain": grain,
                            "file": name,
                            "columns": facts["columns"],
                            "n_pairs": int(len(scores["src"])),
                            **{
                                k: facts[k]
                                for k in (
                                    "n_sequences",
                                    "n_probed",
                                    "n_skipped_short",
                                    "n_cells",
                                    "tok_pos",
                                    "tok_pos_per_s",
                                    "wall_clock_s",
                                    "counts",
                                )
                            },
                        }
                        if probe in (PROBE_CLI_FULL, PROBE_CLI_ATOMIC):
                            stored_ids = [store.get(s) for s in facts["stored_seq"]]
                            sc = {}
                            for arm, paths in probe_columns(probe, noise).items():
                                for path, col in paths.items():
                                    tau_by_lag, graphs, finite = shipped_cut(stored[col], rule)
                                    edges, ufacts = shipped_cut_union(stored_ids, graphs, c, vocab)
                                    sc[f"{arm}/{path}"] = {
                                        "threshold_finite": finite,
                                        "tau_by_lag": tau_by_lag,
                                        "edges": sorted([int(u), int(v)] for u, v in edges),
                                        **ufacts,
                                    }
                            write_json(
                                rec.out_dir
                                / SHIPPED_CUT_JSON_FMT.format(
                                    probe=probe, noise=noise, c=c, n=N, grain=grain
                                ),
                                {
                                    "probe": probe,
                                    "noise": noise,
                                    "c": c,
                                    "N": N,
                                    "g": g,
                                    "grain": grain,
                                    "rule": rule,
                                    "rule_is_v019": is_v019_rule(rule),
                                    "cells": sc,
                                },
                            )
                            cell["shipped_cut"] = {
                                k: {
                                    "threshold_finite": v["threshold_finite"],
                                    "n_edges": v["n_edges"],
                                }
                                for k, v in sc.items()
                            }
                        cells.append(cell)
                        log(
                            {
                                "event": "sweep_cell_done",
                                "probe": probe,
                                "noise": noise,
                                "c": c,
                                "N": N,
                                "grain": grain,
                                "file": name,
                            }
                        )
    if not cells:
        raise ValueError("the sweep produced no cell: check --probes / --noises")
    total_tok = sum(c["tok_pos"] for c in cells)
    total_wall = sum(c["wall_clock_s"] for c in cells)
    rec.note(tok_pos_per_s=total_tok / max(total_wall, 1e-9))
    return {
        "stage": "sweep",
        "split": args.split,
        "rung": args.rung,
        "model_sha256": model_sha,
        "probes": list(args.probes),
        "noises": list(args.noises),
        "contexts": list(args.contexts),
        "particles_grid": list(args.particles_grid),
        "grains": list(args.grains),
        "n_cells": len(cells),
        "probe_wall_clock_s": round(total_wall, 3),
        "tok_pos_per_s": total_tok / max(total_wall, 1e-9),
        "shipped_rule": rule,
        "staging": staging,
        "cells": cells,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--ordering", required=True, choices=ORDERINGS)
    p.add_argument("--grains", required=True, nargs="+", choices=GRAINS)
    p.add_argument("--split", required=True, choices=SPLITS, help="val; test is refused")
    p.add_argument("--model", required=True)
    p.add_argument("--probes", required=True, nargs="+", choices=PROBES)
    p.add_argument("--noises", required=True, nargs="+", choices=NOISES)
    p.add_argument("--contexts", required=True, nargs="+", type=int, help="c grid, BOS-counted")
    p.add_argument(
        "--particles-grid", required=True, nargs="+", type=int, help="N grid (particle probes)"
    )
    p.add_argument("--guidance", required=True, type=int, help="g; the cell uses min(g, c)")
    p.add_argument(
        "--num-sequences",
        required=True,
        nargs="+",
        type=int,
        help="per grain (or one value); 0 = whole split",
    )
    add_shipped_rule_knobs(p)
    add_probe_knobs(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with RunRecord(args.output_folder, "sweep", vars(args)) as rec:
        res = run_sweep(args, rec)
        rec.finish(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
