"""Demo: seq2cause recovering causal graphs from several synthetic event
sequences using the recommended "atomic" do-intervention strategy, with a
tqdm progress bar (and a VRAM/RAM estimate) while it runs.

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
from tqdm import tqdm  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from seq2cause.adapters import HFModelAdapter  # noqa: E402
from seq2cause.diagnostics import compute_cmi_matrix  # noqa: E402
from seq2cause.threshold import AdaptiveThreshold  # noqa: E402
from seq2cause.utils import estimate_tensor_bytes, format_bytes  # noqa: E402

torch.manual_seed(0)
vocab_size, seq_len, context_len, n_particles = 400, 40, 4, 64
n_sequences = 8

config = LlamaConfig(
    vocab_size=vocab_size, hidden_size=256, intermediate_size=512,
    num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=seq_len + 4,
)
model = LlamaForCausalLM(config).eval()
adapter = HFModelAdapter(model, vocab_size=vocab_size)
sequences = torch.randint(0, vocab_size, (n_sequences, seq_len))

lc = seq_len - context_len
est_bytes = estimate_tensor_bytes(n_particles, lc, seq_len, vocab_size)

print(f"seq2cause: {n_sequences} event sequences, do-intervention strategy='atomic'")
print(f"est. VRAM/RAM per sequence: {format_bytes(est_bytes)}\n")

total_edges = 0
top_scores = []
t0 = time.perf_counter()
for sequence in tqdm(sequences, desc="CI-tests (atomic intervention)", unit="seq"):
    cmi_matrix = compute_cmi_matrix(
        adapter, sequence, context_len=context_len, n_particles=n_particles, strategy="atomic",
    )
    causal_graph = AdaptiveThreshold().causal_graph(cmi_matrix)
    total_edges += int(causal_graph.sum())
    top_scores.append(cmi_matrix.max().item())
elapsed = time.perf_counter() - t0

print(f"\nDone in {elapsed:.1f}s. {total_edges} candidate causal edges across {n_sequences} sequences.")
print(f"Top CMI score: {max(top_scores):.2e} (across all sequences)")
