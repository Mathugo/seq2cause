[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19068730.svg)](https://doi.org/10.5281/zenodo.19068730)

# seq2cause
seq2cause: Turns any discrete sequence of events into a causal graph using autoregressive models (LLaMA, GPT, RNN, Mamba).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19068730.svg)](https://doi.org/10.5281/zenodo.19068730)
[![PyPI version](https://img.shields.io/pypi/v/seq2cause.svg)](https://pypi.org/project/seq2cause/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**seq2cause** is a Python library for **Causal Discovery on Discrete Event Sequences**. It bridges the gap between Autoregressive Models (Language Models, RNN, Mamba) and Causal Discovery by treating autoregressive models as density estimators to perform parallelized CI-tests on GPUs.

## 🚀 Key Features

- **Bring Your Own Model:** Plug in any HuggingFace/PyTorch model (`GPT-2`, `LLaMA`, `RNN`) trained on your discrete sequences (logs, codes, symbols).
- **Scaling:** To thousands of events: The memory complexity scales linearly with the vocabulary and sequence length. Optimized for sparse, high-dimensional streams (e.g., Vehicle Diagnostics, Server Logs, User Journeys).
- **Multiple GPUs Acceleration:** Batch processing for analyzing thousands of events in seconds using multiple GPUs.
- **Delayed Effects:** Are identifiable up to the sequence length
- **Causal Relationships Type**: We explain event-to-event, event-to-outcome causal graphs from single sequences and also an aggregation of global event-to-outcome scenarios with instance time causal graphs and summary causal graph.

## 📦 Installation

```bash
pip install seq2cause
```

## ⚡ Quick Start

Recover a causal graph from your own sequence of discrete events (log/event
ids) with your own model -- no known data-generating process, no labeled
ground truth required. `HFModelAdapter` accepts any HuggingFace causal LM
(`GPT-2`, `LLaMA`, a fine-tuned checkpoint, ...); we use a small randomly
initialized `LlamaForCausalLM` below purely so this snippet runs standalone
-- swap it for `LlamaForCausalLM.from_pretrained(...)` or your own trained
model. This exact example is run as part of the test suite
(`tests/test_readme_example.py`).

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from seq2cause.adapters import HFModelAdapter
from seq2cause.diagnostics import compare_intervention_strategies
from seq2cause.threshold import otsu_threshold

# 1. Your model (any HF causal LM) and your own event sequence (token ids).
vocab_size = 20
model = LlamaForCausalLM(LlamaConfig(
    vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
)).eval()
adapter = HFModelAdapter(model, vocab_size=vocab_size)
sequence = torch.randint(0, vocab_size, (20,))  # <- your real event sequence

# 2. Per-(cause, effect) Conditional Mutual Information, via the recommended
#    "atomic" intervention strategy (see "Alternative Intervention Constructions").
#    No ground truth is known here, so pass an empty placeholder adjacency --
#    only `results["atomic"].cmi_matrix` is used below.
placeholder_adjacency = torch.zeros(len(sequence), len(sequence), dtype=torch.bool)
results = compare_intervention_strategies(
    adapter, sequence, context_len=3, adjacency=placeholder_adjacency, n_particles=32, max_lag=3,
)
cmi_matrix = results["atomic"].cmi_matrix  # [L-c, L-c]

# 3. Turn CMI scores into a binary causal graph WITHOUT labeled data, via an
#    unsupervised anomaly-detection-style cutoff (see "Threshold Selection").
causal_graph = cmi_matrix >= otsu_threshold(cmi_matrix.flatten())
```

## 🧪 Evaluation (against a known generator)

To validate the method itself (e.g. when developing/benchmarking, or
reproducing our own experiments), `seq2cause.scm.NonlinearSCM` is a
synthetic oracle generator with a known ground-truth causal graph, letting
you measure recall/F1 directly and pick a threshold by maximizing F1 on a
held-out validation split instead of an unsupervised cutoff:

```python
import torch
from seq2cause.scm import create_scm
from seq2cause.diagnostics import ground_truth_adjacency, compare_intervention_strategies
from seq2cause.threshold import select_threshold_by_validation

torch.manual_seed(0)

# 1. A sequence of discrete events from a KNOWN generator (for validation).
scm, sequences = create_scm(vocab_size=15, memory=3, length=20, seed=0, sparsity=0.5)
sequence = sequences[0]

# 2. Ground-truth edges -- only available/needed because the generator is known.
adjacency = ground_truth_adjacency(scm, sequence, threshold=0.05, n_counterfactuals=16)

# 3. Same recovery step as Quick Start (the SCM satisfies the same
#    `.vocab_size` + `forward(input_ids=...)` interface as `HFModelAdapter`).
results = compare_intervention_strategies(
    scm, sequence, context_len=3, adjacency=adjacency, n_particles=32, max_lag=3,
)
cmi_matrix = results["atomic"].cmi_matrix  # [L-c, L-c]

# 4. With labels available, select tau by maximizing F1 on this validation
#    split instead of an unsupervised cutoff (see "Threshold Selection" --
#    never hardcode a constant tau).
scores, labels = cmi_matrix.flatten(), adjacency[3:, 3:].flatten()
result = select_threshold_by_validation(scores, labels, delta=0.05)
causal_graph = cmi_matrix >= result.tau
print(result.summary())  # e.g. F1=0.800, precision=0.800, recall=0.800
```


## 📚 How It Works

seq2cause implements the **TRACE** framework (Temporal Reconstruction via
Autoregressive Causal Estimation) for event-to-event causal discovery: an
autoregressive model's own next-token conditionals are used as a density
estimator to run parallelized conditional-independence tests, comparing
predicted probabilities with and without a candidate cause intervened on
(`seq2cause.sampling.do_interventions`) via Conditional Mutual Information
(`seq2cause.diagnostics`).

## Graph Types

- **Event-to-Event (per sequence):** implemented here -- the **TRACE**
  algorithm using Conditional Mutual Information (CMI) approximation (see
  Quick Start above).
- **Event-to-Outcome (per sequence / global):** the **OSCAR**/**CARGO**
  algorithms described in the paper for event-to-outcome and aggregated
  global causal graphs are not yet implemented in this repository -- see
  "Future works".

## Future works

- **Event-to-outcome (OSCAR) and global aggregation (CARGO)**: as described
  in the paper, but not yet implemented in this codebase.
- **Time series**: causal discovery for time series using autoregressive
  models (normalizing flows, AR models).

## 🎯 Threshold Selection

Fixed CMI thresholds (e.g. the `tau=3e-5` printed in our paper's example
configs) do not transfer across generator/backbone setups. Following an
independent replication -- Chadyuk, Zhang, and Kucukates, "Replicating
TRACE: A Practitioner's Guide to Its Threshold and Particle Budget"
(LotusFlare Inc., Aug 2026) -- the recommended practice is to *select* the
threshold by maximizing F1 on a held-out validation split instead of
hardcoding a constant:

```python
from seq2cause.threshold import select_threshold_by_validation

result = select_threshold_by_validation(
    cmi_scores,      # per-pair CMI on a held-out validation split
    labels,           # aligned boolean ground-truth edge labels
    delta=0.05,       # the KL margin used to define a ground-truth edge, if known
    hardcoded_defaults={"paper_table_2": 3e-5},
)
print(result.summary())
tau = result.tau
```

`select_threshold_by_validation` reports the selected `tau` *relative to*
the truth margin `delta` (Chadyuk et al. found the blind validation optimum
typically lands within roughly `[delta/2, delta]` for a level-calibrated
estimator) rather than as a bare number, and warns if the selection lands
suspiciously close to a hardcoded default -- a sign the validation split may
be too small or unrepresentative. `resolve_threshold({"type": "validation_sweep", ...})`
makes this the default when resolving a `params["threshold"]` config, while
`{"type": "static", "value": ...}` remains available as an explicit
opt-out/override for reproducing a specific prior run.

## 🔬 Alternative Intervention Constructions

The same replication note found that the default "full" staircase
intervention construction (`seq2cause.sampling.do_interventions`) collapses
recall for cause-effect lags >= 2 to near zero: the position immediately
preceding a distant effect is randomized on both sides of the CMI contrast,
destroying the local context the model relies on most. `do_interventions`
now supports pluggable `strategy=` options to diagnose and work around this,
all opt-in (the default remains `strategy="full"`, unchanged):

- `"full"` (default): the original staircase construction.
- `"atomic"`: only the candidate-cause position is randomized; every other
  position, including mediators, stays real.
- `"windowed"`: preserves a configurable trailing local-context radius
  (`window_k`) before each candidate effect.
- `"independent_mediator"`: draws the cause and mediator noise from two
  statistically independent tensors instead of one shared tensor.

`seq2cause.sampling.unigram_sample` provides an "in-distribution-noise"
alternative to `uniform_sample` for the do-intervention proposal itself.

Run `python scripts/snr_diagnostic.py` for a self-contained (no trained
model required) comparison of per-lag recall and CMI magnitude across all
of the above on a synthetic oracle SCM (`seq2cause.scm.NonlinearSCM`) --
useful for confirming whether the lag>=2 collapse reproduces on your own
generator/backbone before assuming it transfers.

## 🔗 Citation
If you use seq2cause in your research, please cite our works:

```bibtex
@software{math2026seq2cause,
  author = {Math, Hugo},
  title = {seq2cause: Sample- and Population-Level Causal Discovery from Event Sequences using Autoregressive Models},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19068730},
  url = {https://doi.org/10.5281/zenodo.19068730},
  version = {0.1.4}
}
```

```bibtex
@inproceedings{
math2026your,
title={Your Autoregressive Model Already Reveals the Causal Graph},
author={Hugo Math and Rainer Lienhart},
booktitle={ICML 2026 Workshop on Structured Probabilistic Inference {\&} Generative Modeling},
year={2026},
url={https://openreview.net/forum?id=Q66hINx9fA}
}
```

```bibtex
@inproceedings{
math2025oneshot,
title={One-Shot Multi-Label Causal Discovery in High-Dimensional Event Sequences},
author={Hugo Math and Robin Sch{\"o}n and Rainer Lienhart},
booktitle={NeurIPS 2025 Workshop on CauScien: Uncovering Causality in Science},
year={2025},
url={https://openreview.net/forum?id=z7NT8vGWC2}
}
```

```bibtex
@inproceedings{
math2025towards,
title={Towards Practical Multi-label Causal Discovery in High-Dimensional Event Sequences via One-Shot Graph Aggregation},
author={Hugo Math and Rainer Lienhart},
booktitle={NeurIPS 2025 Workshop on Structured Probabilistic Inference {\&} Generative Modeling},
year={2025},
url={https://openreview.net/forum?id=1HZfpuDVeW}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Building
Ruff is used to add only clean code. A pre-commit will be automatically run.

```
pre-commit run --all-files
git add .
git commit -m "corrected import jaxtyping"
git push origin main
git push origin v0.1.4
```
