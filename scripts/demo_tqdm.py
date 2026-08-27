"""Demo: seq2cause's tqdm progress bar and VRAM/RAM estimate while
recovering a causal graph from a synthetic event sequence.

Used to record the README demo GIF (see demo.tape, run via
`vhs demo.tape`). Not a test, just a visual demo -- deliberately sized to
run for a few seconds so the progress bar is visible when recorded.
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from seq2cause.causal_strength import calc_neural_saliency  # noqa: E402

torch.manual_seed(0)
vocab_size, seq_len, context = 500, 60, 5
config = LlamaConfig(
    vocab_size=vocab_size, hidden_size=768, intermediate_size=1536,
    num_hidden_layers=4, num_attention_heads=8, max_position_embeddings=seq_len + 4,
)
model = LlamaForCausalLM(config).eval()
batch = {"input_ids": torch.randint(0, vocab_size, (4, seq_len))}
params = {"sampling": {"context": context}}

print("seq2cause: recovering a causal graph from a synthetic event sequence\n")
t0 = time.perf_counter()
_, adj = calc_neural_saliency(model, batch, params)
elapsed = time.perf_counter() - t0

top = torch.topk(adj.flatten(), k=3).values
print(f"\nDone in {elapsed:.1f}s. Causal-strength matrix: {tuple(adj.shape)}")
print(f"Top causal-strength scores: {[round(v.item(), 3) for v in top]}")
