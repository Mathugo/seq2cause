# seq2cause
seq2cause: Turns any discrete sequence of events into a causal graph using autoregressive models (LLaMA, GPT, RNN, Mamba).

[![PyPI version](https://img.shields.io/pypi/v/seq2cause.svg)](https://pypi.org/project/seq2cause/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**seq2cause** is a Python library for **Causal Discovery on Discrete Event Sequences**. It bridges the gap between Autoregressive Models (Language Models, RNN, Mambda) and Causal Discovery by treating autoregressive models as density estimators to perform parallelized CI-tests on GPUs.

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
Recover the causal graph from your logs in 3 lines of code.

## 📚 How It Works

seq2cause implements the **TRACE** framework (Temporal Reconstruction via Autoregressive Causal Estimation) for the event-to-event causal discovery and **OSCAR** for the event-to-outcome. <talk abvout cmi>

## Graph Types
You can precise the graph types, which includes [redo graph namming and parameters in packages, put time instrance, summary graph]:

- **Event-to-Event (per sequence):** Implements the **TRACE** algorithm using Conditional Mutual Information (CMI) approximation.
- **Event-to-Outcome (per sequence):** Implements the **OSCAR** algorithm which target event-to-outcome relationships using a second autoregressive models to predict outcomes.
- **Event-to-Outcome (global):** Implements the **CARGO** algorithm which aggregate the per-sequence causal graph to provide a global causal relationship of observational data.

## Future works

- **Time series**: Implements causal discovery for time series using autoregressive models (normalizing flows, AR models)

## 🔗 Citation
If you use seq2cause in your research, please cite our works:

```bash
@misc{math2026tracescalableamortizedcausal,
      title={TRACE: Scalable Amortized Causal Discovery from Single Sequences via Autoregressive Density Estimation},
      author={Hugo Math and Rainer Lienhart},
      year={2026},
      eprint={2602.01135},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.01135},
}
```

```bash
@inproceedings{
math2025oneshot,
title={One-Shot Multi-Label Causal Discovery in High-Dimensional Event Sequences},
author={Hugo Math and Robin Sch{\"o}n and Rainer Lienhart},
booktitle={NeurIPS 2025 Workshop on CauScien: Uncovering Causality in Science},
year={2025},
url={https://openreview.net/forum?id=z7NT8vGWC2}
}
```

```bash
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
