"""Command-line entry point: recover a causal graph from a tokenized
event-sequence dataset using seq2cause's do-intervention CI-test.

Wraps exactly the same three calls as the README Quick Start
(`HFModelAdapter` -> `compute_cmi_matrix` -> `AdaptiveThreshold`), so you can
run seq2cause against your own tokenized data without writing any Python:

    seq2cause --dataset events.txt --model gpt2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from seq2cause.adapters import HFModelAdapter
from seq2cause.diagnostics import compute_cmi_matrix, summary_graph
from seq2cause.threshold import AdaptiveThreshold
from seq2cause.utils import check_memory_budget, estimate_tensor_bytes, format_bytes

__all__ = ["build_arg_parser", "load_tokenized_dataset", "main"]


def load_tokenized_dataset(path: str | Path) -> list[torch.Tensor]:
    """Loads a tokenized event-sequence dataset from `path` into a list of 1D
    `LongTensor`s (one per sequence -- sequences may have different lengths,
    since `compute_cmi_matrix` processes one sequence at a time).

    Supported formats (picked by file extension):
      - `.pt`/`.pth`: a `torch.save`d `Tensor` (`[N, L]`, or `[L]` for a
        single sequence) or a list/tuple of 1D tensors/sequences. Loaded
        with `weights_only=True` where supported, to avoid executing
        arbitrary code from an untrusted checkpoint-shaped file.
      - `.npy`: a `numpy.save`d array, same shapes as above (an object
        array of variable-length sequences is also supported).
      - anything else: a plain text file, one sequence per line, token ids
        separated by whitespace and/or commas (e.g. `"3 17 2 91"` or
        `"3,17,2,91"`).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".pt", ".pth"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older torch versions don't support the weights_only kwarg.
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            sequences = [obj] if obj.dim() == 1 else list(obj)
        elif isinstance(obj, (list, tuple)):
            sequences = list(obj)
        else:
            raise ValueError(
                f"Unsupported contents in {path}: expected a Tensor or a list/tuple "
                f"of sequences, got {type(obj)}"
            )
        return [torch.as_tensor(seq, dtype=torch.long) for seq in sequences]

    if suffix == ".npy":
        import numpy as np

        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object or arr.ndim > 1:
            return [torch.as_tensor(row, dtype=torch.long) for row in arr]
        return [torch.as_tensor(arr, dtype=torch.long)]

    sequences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = [int(tok) for tok in line.replace(",", " ").split()]
            sequences.append(torch.tensor(tokens, dtype=torch.long))
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    return sequences


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seq2cause",
        description=(
            "Recover a causal graph from a tokenized event-sequence dataset, using "
            "seq2cause's do-intervention conditional-independence test."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a tokenized dataset: a .pt/.pth (Tensor or list of sequences), "
        ".npy array, or a plain text file with one whitespace/comma-separated "
        "sequence of token ids per line.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="A HuggingFace causal LM to use as the density estimator -- a model id "
        "(e.g. 'gpt2') or a local checkpoint path. If omitted, a small randomly "
        "initialized model is used (for quick experimentation only -- see README "
        "Quick Start). Either way, vocab_size is inferred automatically (from the "
        "model's config, or otherwise from the dataset's own token ids) -- never "
        "passed explicitly.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=4,
        help="Length of the fixed, always-real prefix (default: 4).",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=32,
        help="Number of do-intervention noise particles per candidate cause (default: 32).",
    )
    parser.add_argument(
        "--strategy",
        choices=["atomic", "full"],
        default="atomic",
        help="Do-intervention construction strategy (default: 'atomic', recommended -- "
        "see README).",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["otsu", "mad", "percentile", "gmm"],
        default="otsu",
        help="Unsupervised cutoff `AdaptiveThreshold` anchors tau on, since no labeled "
        "ground truth is available here (default: 'otsu' -- see README Threshold "
        "Selection).",
    )
    parser.add_argument(
        "--graph-level",
        choices=["sample", "summary", "both"],
        default="both",
        help="Which graph(s) to report (default: 'both'). 'sample': the raw "
        "position-indexed (time-step) causal graph for each sequence. 'summary': "
        "projects that down to an event-TYPE-indexed graph per sequence (an edge "
        "u -> v exists iff some position with type u causally affected a later "
        "position with type v at least once in that sequence). See README.",
    )
    parser.add_argument(
        "--self-loops",
        action="store_true",
        help="When --graph-level includes 'summary', keep u -> u edges (an event type "
        "causing itself). Off by default.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Process at most this many sequences from the dataset (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the recovered graph(s) to (torch.save, a list of "
        "dicts, one per sequence, with a 'sample_graph' and/or 'summary_graph' key "
        "depending on --graph-level).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (default: 'cuda' if available, else 'cpu').",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sequences = load_tokenized_dataset(args.dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    if args.model:
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(args.model).eval().to(device)
        vocab_size = hf_model.config.vocab_size
    else:
        from transformers import LlamaConfig, LlamaForCausalLM

        vocab_size = max(int(seq.max()) for seq in sequences) + 1
        max_len = max(seq.numel() for seq in sequences)
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=max_len + 8,
        )
        hf_model = LlamaForCausalLM(config).eval().to(device)
        print(
            f"No --model given: using a small, randomly initialized LlamaForCausalLM "
            f"(vocab_size={vocab_size}, inferred from the dataset) -- for real causal "
            "discovery, pass a trained checkpoint via --model.\n"
        )

    adapter = HFModelAdapter(hf_model, vocab_size=vocab_size)
    threshold = AdaptiveThreshold(method=args.threshold_method)
    want_sample = args.graph_level in ("sample", "both")
    want_summary = args.graph_level in ("summary", "both")

    print(
        f"seq2cause: {len(sequences)} event sequence(s), "
        f"do-intervention strategy={args.strategy!r}, threshold={args.threshold_method!r}"
    )
    first = sequences[0]
    first_lc = first.numel() - args.context_len
    est_bytes = estimate_tensor_bytes(args.n_particles, first_lc, first.numel(), vocab_size)
    print(f"est. VRAM/RAM per sequence (approx.): {format_bytes(est_bytes)}\n")

    results = []
    total_sample_edges = 0
    total_summary_edges = 0
    top_score = float("-inf")
    t0 = time.perf_counter()
    for sequence in tqdm(sequences, desc="CI-tests (do-intervention)", unit="seq"):
        sequence = sequence.to(device)
        lc = sequence.numel() - args.context_len
        seq_est_bytes = estimate_tensor_bytes(args.n_particles, lc, sequence.numel(), vocab_size)
        check_memory_budget(seq_est_bytes, device)
        cmi_matrix = compute_cmi_matrix(
            adapter,
            sequence,
            context_len=args.context_len,
            n_particles=args.n_particles,
            strategy=args.strategy,
        )
        causal_graph = threshold.causal_graph(cmi_matrix)
        top_score = max(top_score, cmi_matrix.max().item())

        entry = {}
        if want_sample:
            entry["sample_graph"] = causal_graph
            total_sample_edges += int(causal_graph.sum())
        if want_summary:
            active_tokens, summary_adj = summary_graph(
                sequence, causal_graph, context_len=args.context_len, self_loops=args.self_loops
            )
            entry["summary_graph"] = {"active_tokens": active_tokens, "adj": summary_adj}
            total_summary_edges += int(summary_adj.sum())
        results.append(entry)
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.1f}s. Top CMI score: {top_score:.2e}")
    if want_sample:
        print(
            f"Sample-level (time-step): {total_sample_edges} candidate causal edges "
            f"across {len(sequences)} sequence(s)."
        )
    if want_summary:
        print(
            f"Summary graph (event type): {total_summary_edges} candidate causal edges "
            f"across {len(sequences)} sequence(s)."
        )

    if args.output:
        torch.save(results, args.output)
        print(f"Saved {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
