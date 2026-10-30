"""End-to-end check of `SampleLevelCausalDiscovery` context modes and causal strengths.

Trains a tiny Llama on a synthetic `NonlinearSCM` until its exact oracle score eps_hat <= 0.05,
then runs the full pipeline (`prepare()` + `run()`) with

    context_mode in {"overwrite", "prepend", "none"}  x  causal_strength in {info gain, "CMI"}

and reports, for every run, the adjacency shape, the observed positions it covers, and the
average precision against the exact SCM ground truth:
  * info gain vs. GT-IG  (`diagnostics.ground_truth_adjacency`, atomic intervention, KL > delta),
  * CMI       vs. GT-CMI (exact CMI of the binary cause indicator, top-k with k = #GT-IG edges),
both on the pairs the run tests and on ALL pairs of the sequence (untested pairs score 0).

Usage:
    python scripts/validate_context_modes.py [--n-seq 20] [--particles 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from compare_cmi_and_proposals import (  # noqa: E402
    average_precision_and_best_f1,
    exact_cmi_matrix,
    train_model,
)

from seq2cause.core import SampleLevelCausalDiscovery  # noqa: E402
from seq2cause.diagnostics import ground_truth_adjacency  # noqa: E402
from seq2cause.scm import NonlinearSCM  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vocab-size", type=int, default=50)
    p.add_argument("--memory", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-seq", type=int, default=20)
    p.add_argument("--particles", type=int, default=32)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    V, m, L = args.vocab_size, args.memory, args.seq_len
    scm = NonlinearSCM(vocab_size=V, memory=m, sparsity=0.7, decay_rate=0.5, seed=args.seed)
    targs = argparse.Namespace(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        seq_len=L + args.context,
        lr=2e-3,
        max_steps=12000,
        eval_every=500,
        batch_size=128,
        target_eps=0.05,
    )
    model, oracle = train_model(scm, targs)
    print(f"trained tiny Llama: eps_hat={oracle['epsilon_hat']:.4f} after {oracle['steps']} steps")

    gen = torch.Generator().manual_seed(args.seed + 1)
    seqs = scm.sample_sequence(L, batch_size=args.n_seq, generator=gen)
    gt_ig = torch.stack(
        [ground_truth_adjacency(scm, s, threshold=args.delta, generator=gen) for s in seqs]
    )
    cmi = torch.stack([exact_cmi_matrix(scm, s, 0) for s in seqs])
    allp = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    k = int(gt_ig[:, allp].sum())
    gt_cmi = torch.zeros_like(gt_ig)
    flat = cmi[:, allp].reshape(-1)
    sel = torch.zeros_like(flat, dtype=torch.bool)
    sel[torch.argsort(flat, descending=True)[:k]] = True
    gt_cmi[:, allp] = sel.view(args.n_seq, -1)
    ds = [{"input_ids": s.tolist(), "attention_mask": [1] * L} for s in seqs]

    runs = [
        ("overwrite", args.context, m),  # v0.1.9 default: first m kept, m..c-1 overwritten
        ("prepend", args.context, 0),  # sampled past before the full sequence (no CLS token)
        ("none", 1, 1),  # no sampled past, first token as anchor
    ]
    print(
        f"\n|X|={V}, memory={m}, L={L}, {args.n_seq} sequences, N={args.particles}, "
        f"{k} GT-IG edges of {int(allp.sum()) * args.n_seq} pairs\n"
        f"{'context_mode':<12}{'c':>3}{'g':>3}{'score':>7}{'adjacency':>16}{'positions':>11}"
        f"{'AP tested':>11}{'AP all':>9}"
    )
    for mode, c, g in runs:
        for cs, name, gt in ((None, "ig", gt_ig), ("CMI", "cmi", gt_cmi)):
            params = {
                "BS": 4,
                "full": True,
                "fp16": False,
                "cls_token_id": None,
                "causal_strength": cs,
                "check_memory_budget": False,
                "sampling": {
                    "type": "naive",
                    "clamping": 1e-9,
                    "context": c,
                    "guidance": g,
                    "value": args.particles,
                    "context_mode": mode,
                },
            }
            torch.manual_seed(args.seed + 7)
            algo = SampleLevelCausalDiscovery(model, params, ds)
            algo.prepare()
            _, adj = algo.run()
            s0 = algo.first_position()
            n = adj.shape[-1]
            full = torch.zeros(args.n_seq, L, L)
            full[:, s0 : s0 + n, s0 : s0 + n] = adj.float().cpu()
            tested = torch.zeros(L, L, dtype=torch.bool)
            tested[s0 + 1 : s0 + n, :] = True  # causes s0+1.. (row 0 is never a cause)
            tested &= allp
            ap_t, _ = average_precision_and_best_f1(
                full[:, tested].reshape(-1).double(), gt[:, tested].reshape(-1)
            )
            ap_a, _ = average_precision_and_best_f1(
                full[:, allp].reshape(-1).double(), gt[:, allp].reshape(-1)
            )
            print(
                f"{mode:<12}{c:>3}{g:>3}{name:>7}{str(tuple(adj.shape)):>16}{f'{s0}..{s0 + n - 1}':>11}"
                f"{ap_t:11.3f}{ap_a:9.3f}"
            )


if __name__ == "__main__":
    main()
