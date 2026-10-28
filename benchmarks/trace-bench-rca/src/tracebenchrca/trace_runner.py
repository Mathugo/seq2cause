"""Probe the selected traces of every case of one corpus with one frozen TRACE cell and write each trace's sample
graph (method side; runs in the reference harness's environment, `plans/rca-arms.md` addendum 2026-10-30, D-TBR-13;
the per-trace axis's selection rule, plan of 2026-10-31).

    <seq2cause-trace-bench python> trace_runner.py --corpus-dir <pulled corpus> --cases <cases.json> \
        --model <backbone dir> --freeze <committed freeze> --cell trace/cli-atomic/shipped/frozen/request \
        --select post-nonok --cap 0 --max-len 64 --device cpu --seed 0 --out-dir <dir>

Imports `seq2causebench` (the reference harness: the corpus door, the vocabulary, the encoder and the engine
adapter `probe_sequence`) and nothing from `tracebenchrca`. `cases.json` is `[{"case_dir", "inject_time",
"end_time"}]` (UNIX seconds, the two window fields). Per case, the rows of the three splits of the end-ordered
request view are selected by `--select`:

- `window` (the landed case-level runs): `start_ns` in `[inject_time, end_time)`, no outcome test;
- `post-nonok` (rule R of the per-trace axis, label-free): `inject_ns <= start_ns`, the trace's last span ends
  before `end_ns` (`start_ns + max(offset + duration) < end_ns`, integer nanoseconds) and the trace holds a position
  whose RAW view outcome is non-OK (5xx, err, slow — never 4xx) on a non-client operation.

Selected rows are ordered by `(start_ns, trace_id)`; `--cap 0` probes every one, `--cap N` the first N. Each is
encoded as the frozen cell's reads were (`BOS + ids + EOS`, `--max-len` real tokens), seeded per trace, probed once
with the cell's `(probe, noise, c, N, g)` on the loaded backbone (its weights hash must equal the freeze's
`model_sha256`), and cut at the cell's frozen tau: the edges `j -> q` (`j < q`, `S[j, q] > tau`, both tokens real) of
the window positions (window position `p` is `ops[c + p - 1]`). Writes `<out-dir>/<case_dir>.json` — the probed
traces with, per window position, `(op, outcome, service, is_client)` from the view's `model-vocab.json` and the
cut edges, the selected and skipped trace ids, and the count of window positions whose raw outcome is non-OK but
whose token folded to the base op (`n_nonok_folded`) — and `<out-dir>/summary.json` (the cell, the model and
freeze hashes, versions, import paths, throughput).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

SELECT_WINDOW = "window"
SELECT_POST_NONOK = "post-nonok"
SPLITS = ("train", "val", "test")
SEQUENCE_COLUMNS = ("trace_id", "ops", "outcomes", "offsets", "durations", "n_spans", "start_ns")


def trace_end_ns(start_ns, offsets, durations):
    """The trace's last span end, integer nanoseconds (the row start when it has no span)."""
    ends = [int(o) + int(d) for o, d in zip(offsets, durations, strict=True)]
    return int(start_ns) + (max(ends) if ends else 0)


def is_window_trace(start_ns, inject_ns, end_ns):
    return inject_ns <= int(start_ns) < end_ns


def is_r_trace(start_ns, t_end_ns, ops, outcomes, inject_ns, end_ns, nonok_ids, is_client):
    """Rule R: the whole trace inside `[inject_ns, end_ns)` and a non-client position with a raw non-OK outcome."""
    if not (inject_ns <= int(start_ns) and t_end_ns < end_ns):
        return False
    return any(int(o) in nonok_ids and not is_client(int(op)) for op, o in zip(ops, outcomes, strict=True))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--cases", required=True, help="JSON list of {case_dir, inject_time, end_time}")
    p.add_argument("--model", required=True, help="the seed's frozen backbone directory")
    p.add_argument("--freeze", required=True, help="the committed freeze record of the same corpus and backbone")
    p.add_argument("--cell", required=True, help="the frozen cell key, <arm>/<path>/frozen/request")
    p.add_argument(
        "--select",
        required=True,
        choices=(SELECT_WINDOW, SELECT_POST_NONOK),
        help="window: start in [inject, end); post-nonok: rule R (whole trace inside, a non-client non-OK span)",
    )
    p.add_argument("--cap", required=True, type=int, help="selected traces probed per case, in start order; 0 = all")
    p.add_argument("--max-len", required=True, type=int, help="real tokens per sequence, as the cell's reads")
    p.add_argument("--device", required=True, choices=("cpu", "cuda", "mps"))
    p.add_argument("--seed", required=True, type=int, help="the corpus seed; the per-trace noise seed derives from it")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    if args.cap < 0:
        raise ValueError("--cap must be >= 0 (0 = every selected trace)")

    import seq2cause
    import seq2causebench
    from seq2causebench.arms import column_of, parse_cell_key
    from seq2causebench.corpus import Corpus
    from seq2causebench.data import encode_sequence
    from seq2causebench.engine import load_backbone, probe_columns, probe_sequence
    from seq2causebench.record import git_commit, sha256_file
    from seq2causebench.vocab import Vocab
    from tracebench.constants import KIND_CLIENT, OUTCOME_5XX, OUTCOME_ERR, OUTCOME_NAMES, OUTCOME_OK, OUTCOME_SLOW

    nonok_ids = (OUTCOME_5XX, OUTCOME_ERR, OUTCOME_SLOW)  # = tracebenchrca.constants.TRACE_NONOK_OUTCOMES by name
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    arm, path, cut, grain = parse_cell_key(args.cell)
    if cut != "frozen" or grain != "request":
        raise ValueError(f"the cell must be a frozen request cell, got {args.cell}")
    cell = freeze["cells"][args.cell]
    c, N, g, tau, probe, noise = cell["c"], cell["N"], cell["g"], float(cell["tau"]), cell["probe"], cell["noise"]
    hf_model, adapter, model_sha = load_backbone(args.model, args.device)
    if model_sha != freeze["model_sha256"]:
        raise ValueError(f"the backbone hashes to {model_sha} but the freeze binds {freeze['model_sha256']}")
    corpus = Corpus(args.corpus_dir, freeze.get("ordering") or "end", "request")
    if corpus.corpus_id != freeze["corpus_id"]:
        raise ValueError(f"the freeze is for {freeze['corpus_id']!r}, the corpus is {corpus.corpus_id!r}")
    vocab = Vocab.from_model_vocab(corpus.vocab_json())
    if vocab.size != hf_model.config.vocab_size:
        raise ValueError(f"the view has {vocab.size} ids, the backbone {hf_model.config.vocab_size}")
    op_service = {op["id"]: (str(op["service"]), int(op["kind"]) == KIND_CLIENT) for op in vocab.base_ops}

    def is_client(op):
        return op_service[op][1]

    columns = probe_columns(probe, noise)
    col = column_of(arm, path)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    windows = [(k["case_dir"], int(k["inject_time"]) * 10**9, int(k["end_time"]) * 10**9) for k in cases]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # one pass over the three splits: the selected rows per case (every split; the backbone is frozen); the
    # legacy window count is kept under both rules so `n_traces_window` keeps one meaning
    selected = {name: [] for name, _, _ in windows}
    n_window = dict.fromkeys(selected, 0)
    n_rows = 0
    for split in SPLITS:
        for tid, ops, outcomes, offsets, durations, n_spans, start_ns in corpus.sequences(
            split, columns=SEQUENCE_COLUMNS
        ):
            n_rows += 1
            start_ns = int(start_ns)
            t_end = trace_end_ns(start_ns, offsets, durations) if args.select == SELECT_POST_NONOK else None
            for name, a, b in windows:
                in_window = is_window_trace(start_ns, a, b)
                n_window[name] += in_window
                if args.select == SELECT_WINDOW:
                    take = in_window
                else:
                    take = is_r_trace(start_ns, t_end, ops, outcomes, a, b, nonok_ids, is_client)
                if take:
                    selected[name].append((start_ns, str(tid), list(ops), list(outcomes), int(n_spans)))
    device = next(hf_model.parameters()).device
    totals = {
        "n_traces_window": 0,
        "n_traces_candidates": 0,
        "n_traces_probed": 0,
        "tok_pos": 0,
        "probe_wall_clock_s": 0.0,
    }
    t_all = time.monotonic()
    for name, _, _ in windows:
        candidates = sorted(selected[name])
        rows = candidates[: args.cap] if args.cap > 0 else candidates
        traces, skipped, n_trunc, n_fold, n_nonok_fold = [], [], 0, 0, 0
        t0 = time.monotonic()
        tok_pos = 0
        for i, (start_ns, tid, ops, outcomes, n_spans) in enumerate(rows):
            ids, truncated, folded = encode_sequence(vocab, ops, outcomes, args.max_len)
            n_trunc += truncated
            n_fold += folded
            if len(ids) - c < 2:
                skipped.append(tid)
                continue
            torch.manual_seed((int(args.seed) * 1000003 + 3) * 1000003 + i)
            ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=device)
            res = probe_sequence(probe, noise, hf_model, adapter, ids_t, c, g, N, columns, "none")
            tok_pos += res.tok_pos
            m = np.asarray(res.matrices[col], dtype=np.float64)
            window = ids[c:]
            positions = []
            for pidx, t in enumerate(window):
                if vocab.is_real(t):
                    svc, client = op_service[vocab.op_of(t)]
                    outcome = OUTCOME_NAMES[vocab.outcome_of(t)]
                    positions.append({"op": vocab.op_of(t), "outcome": outcome, "service": svc, "is_client": client})
                    # window position p <-> ops index c + p - 1: a raw non-OK outcome whose token is the base op folded
                    if outcome == OUTCOME_NAMES[OUTCOME_OK] and int(outcomes[c + pidx - 1]) in nonok_ids:
                        n_nonok_fold += 1
                else:
                    positions.append(None)
            real = np.array([pos is not None for pos in positions])
            jj, qq = np.triu_indices(len(window), k=1)
            keep = (m[jj, qq] > tau) & real[jj] & real[qq]
            edges = [[int(j), int(q), float(m[j, q])] for j, q in zip(jj[keep], qq[keep], strict=True)]
            traces.append(
                {
                    "trace_id": tid,
                    "start_ns": start_ns,
                    "n_spans": n_spans,
                    "n_window": len(window),
                    "positions": positions,
                    "edges": edges,
                }
            )
        wall = time.monotonic() - t0
        doc = {
            "case_dir": name,
            "select": args.select,
            "n_traces_window": n_window[name],
            "n_traces_candidates": len(candidates),
            "n_traces_selected": len(rows),
            "n_traces_probed": len(traces),
            "n_skipped_short": len(skipped),
            "n_truncated": n_trunc,
            "n_folded": n_fold,
            "n_nonok_folded": n_nonok_fold,
            "selected_trace_ids": [tid for _, tid, *_ in rows],
            "skipped_trace_ids": skipped,
            "tok_pos": tok_pos,
            "probe_wall_clock_s": round(wall, 3),
            "traces": traces,
        }
        (out / f"{name}.json").write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        totals["n_traces_window"] += n_window[name]
        totals["n_traces_candidates"] += len(candidates)
        totals["n_traces_probed"] += len(traces)
        totals["tok_pos"] += tok_pos
        totals["probe_wall_clock_s"] += wall
        print(
            json.dumps(
                {
                    "event": "trace_case",
                    "case_dir": name,
                    "select": args.select,
                    "n_window": n_window[name],
                    "n_candidates": len(candidates),
                    "n_probed": len(traces),
                    "wall_s": round(wall, 1),
                }
            ),
            flush=True,
        )
    summary = {
        "cell": args.cell,
        "arm": arm,
        "path": path,
        "probe": probe,
        "noise": noise,
        "column": col,
        "c": c,
        "N": N,
        "g": g,
        "tau": tau,
        "model_sha256": model_sha,
        "freeze_sha256": sha256_file(args.freeze),
        "freeze_corpus_id": freeze["corpus_id"],
        "corpus_id": corpus.corpus_id,
        "vocab_size": vocab.size,
        "n_rows_read": n_rows,
        "device": str(device),
        "select": args.select,
        "cap": args.cap,
        "max_len": args.max_len,
        "seed": args.seed,
        "seed_stream": 3,
        **totals,
        "wall_clock_s": round(time.monotonic() - t_all, 3),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seq2cause_version": getattr(seq2cause, "__version__", None),
        "seq2cause_file": os.path.abspath(seq2cause.__file__),
        "seq2causebench_file": os.path.abspath(seq2causebench.__file__),
        "seq2causebench_commit": git_commit(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
