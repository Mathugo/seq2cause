# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- **`AdaptiveThreshold`'s default `method` is now `"mad"`, not `"otsu"`.**
  Otsu maximizes between-class variance, which on a CMI score distribution
  with one dominant outlier and a long near-zero tail (common on a single,
  short sequence) often isolates JUST that one outlier as its own "class"
  rather than the true/false-edge break. Measured on one worked example
  (`NonlinearSCM`, memory 2, vocab 12, length 40): F1 = 0.13 for `otsu` vs.
  0.57 for `mad`/`gmm` on the *same* CMI matrix. If you were relying on the
  old default, pass `AdaptiveThreshold(method="otsu")` explicitly.
- The `seq2cause` CLI now fits its threshold ONCE on CMI scores pooled
  across every sequence in `--dataset` (via the new
  `AdaptiveThreshold.apply_tau_by_lag`), instead of re-fitting a fresh
  cutoff per sequence. Measured across 8 sequences from the same
  generator, pooling reduces the standard deviation of per-sequence F1
  from 0.12 to 0.09 (`mad`) and 0.18 to 0.08 (`gmm`), i.e. a materially
  more consistent result run to run. `--threshold-method`'s CLI default
  is also now `mad` (previously `otsu`), for the same reason as above.

### Added
- `AdaptiveThreshold.apply_tau_by_lag(cmi_matrix, tau_by_lag)`: applies an
  externally fit `tau_by_lag` (e.g. from scores pooled across several
  sequences) to a single sequence's own CMI matrix, without re-fitting.
  `causal_graph` is now implemented in terms of it.
- `seq2cause.diagnostics.summary_graph`: projects a single sequence's
  sample-level (time-step/position) causal graph down to a summary graph
  whose nodes are event TYPES instead -- an edge `u -> v` exists iff some
  position holding type `u` causally affected a later position holding
  type `v` at least once in that sequence (union/"at least once"
  aggregation). Scales with the sequence length, not `vocab_size`. See
  README Quick Start.
- A `seq2cause` command-line interface (`seq2cause.cli`, registered as a
  console script): recovers a causal graph from a tokenized event-sequence
  dataset (`--dataset`, accepting a plain text file, `.pt`, or `.npy`)
  using `compute_cmi_matrix`'s `"atomic"` do-intervention strategy and
  `AdaptiveThreshold`, wrapping any HuggingFace causal LM (`--model`) or a
  small randomly-initialized one for quick experimentation. `vocab_size`
  is always inferred automatically (from `--model`'s config, or otherwise
  from the dataset's own token ids) -- there is no `--vocab-size` flag.
  `--threshold-method` selects `AdaptiveThreshold`'s unsupervised cutoff
  (`otsu`/`mad`/`percentile`/`gmm`). `--graph-level` (`sample`/`summary`/
  `both`, default `both`) picks which of the sample-level and summary
  graphs to compute/report/save; `--self-loops` keeps `u -> u` edges in
  the summary graph. See README "Command-line interface".
- A demo GIF at the top of the README (`assets/demo.gif`), showing
  `seq2cause --help` (the available CLI arguments), then the CLI loading a
  tokenized example dataset (`examples/event_sequences.txt`) and a real
  pretrained HuggingFace model (`distilgpt2`), with a `tqdm` progress bar
  and VRAM/RAM estimate running live, then the recovered CMI results.
  Generated with [vhs](https://github.com/charmbracelet/vhs) from
  `demo.tape`.
- `seq2cause.utils.check_memory_budget` and `get_available_memory_bytes`:
  a pre-flight check that compares an estimated tensor size against the
  memory currently available on a device, and raises a clear `MemoryError`
  before an allocation that would likely run out of memory, instead of
  crashing deep inside a forward pass. Works on CUDA
  (`torch.cuda.mem_get_info`) and on CPU with `psutil` installed; a no-op
  on MPS, which has no public free-memory query. Wired into
  `SampleLevelCausalDiscovery`'s batch loop.
- README "Avoiding an out-of-memory crash" section.
- Tests for `check_memory_budget`/`get_available_memory_bytes`, including
  the fallback behavior when the available memory can't be determined.

## [0.1.7] - 2026-08-27

### Added
- `seq2cause.diagnostics.compute_cmi_matrix_sparse`: a bounded-memory
  ("sparse") variant of `compute_cmi_matrix` for generators with a known or
  assumed finite causal lag, such as `NonlinearSCM(memory=m)`. It slides a
  short local window across the sequence instead of running the "full"
  staircase once on the whole sequence, so each step is much cheaper. This
  is exact, not an approximation, whenever `memory` truly bounds the lag.
  Empirically it lands within about 0.01 to 0.02 pooled F1 of the unbounded
  computation, with a 2x to 5x speedup that grows with sequence length
  (`scripts/evaluate_sparse_vs_full.py`).
- README "Sparse / Bounded-Memory Construction" section.
- Tests for `compute_cmi_matrix_sparse`: shape and guard checks, plus an
  empirical full-vs-sparse comparison (CMI-cell correlation and pooled F1)
  on a decayed, memory-bounded `NonlinearSCM`.

### Changed
- README rewritten for readability: shorter, plainer prose, no more
  em-dashes or emoji section headings. Cut from 329 to 218 lines.

## [0.1.6] - 2026-08-27

### Added
- `seq2cause.diagnostics.compute_cmi_matrix`: a lightweight, single-strategy,
  no-ground-truth-required entry point for computing the CMI matrix
  (defaults to `strategy="atomic"`) -- the new recommended way to run
  causal discovery on a real model with no known generator, without paying
  for `compare_intervention_strategies`' full 5-strategy comparison.
- `seq2cause.threshold.AdaptiveThreshold`: a configurable, label-free
  `tau(lag)` construction (`method`, `per_lag`, `decay`, `decay_type`,
  `decay_rate`/`exponent`, `floor`). Default recipe -- Otsu fit once
  globally (not per lag), anchored at lag=1, then decayed exponentially
  toward an Otsu-fit floor -- is now the **recommended default unsupervised
  threshold**, with a `.causal_graph(cmi_matrix)` one-call convenience
  method. Matches/approaches labeled validation-based thresholds in our own
  testing, with zero labels used.
- `seq2cause.threshold`: exponential/power-law/joint lag x vocab-size
  threshold-decay fits (`fit_exponential_lag_threshold`,
  `fit_power_law_lag_threshold`, `fit_joint_lag_vocab_threshold`),
  `select_thresholds_by_group` reliability warnings for small per-group
  sample counts, and `resolve_threshold` for `{"type": "validation_sweep" |
  "static"}` configs.
- `seq2cause.adapters.HFModelAdapter`: a public adapter wrapping any real
  HuggingFace causal LM (pretrained or trained) so it satisfies the small
  interface `compute_cmi_matrix`/`do_interventions` expect.
- Pluggable `do_interventions` intervention-construction strategies:
  `"atomic"` (recommended), `"full"` (paper default, kept for backward
  compatibility/reproducing Table 2), `"windowed"`, `"independent_mediator"`.
- Multi-GPU correctness testing: `tests/test_multi_gpu.py` +
  `scripts/multi_process_check.py` spawn 2 real processes (`gloo` backend)
  to verify Accelerate's dataloader sharding and `gather()` logic without
  needing real GPU hardware.
- `tqdm` progress bars ("CI-tests (batch x context)") with a live estimated
  VRAM/RAM footprint (`seq2cause.utils.estimate_tensor_bytes`/
  `format_bytes`) in `core.py`'s batch loop and `causal_strength.py`'s
  per-position Saliency/Shapley loops.
- README documents tested backends: CPU, Apple Silicon (MPS), NVIDIA CUDA.
- Tests for previously-uncovered modules: `core.py`, `causal_strength.py`,
  `utils.py`. Overall coverage 77% -> 95%.

### Fixed
- `seq2cause.core.SampleLevelCausalDiscovery` was completely non-functional:
  `__init__` set `self._params` but every method read `self.params`
  (`AttributeError` on the first call to `.prepare()`/`.run()`); the final
  causal-strength call site passed an extra `tfx` argument that
  `calc_lag_info_gain`/`calc_granger_score` don't accept; the do-intervention
  proposal call splatted the whole sampling dict into `uniform_sample`,
  which has no `**kwargs` catch-all. All fixed and covered by new tests.
- `seq2cause.sampling.ancestral_sampling`: unconditional
  `torch.cuda.synchronize()` crashed with "Torch not compiled with CUDA
  enabled" on any CPU-only machine; now guarded by `torch.cuda.is_available()`.
- `seq2cause.sampling.NonlinearSCM.sample_sequence`: `torch.multinomial`
  wasn't forwarding the caller-supplied `generator`, silently falling back
  to global RNG state for every token after the memory burn-in prefix --
  made sequence sampling genuinely reproducible regardless of unrelated
  prior global-RNG consumption (e.g. whether a training loop ran first).
- `seq2cause.causal_strength.calc_granger_score`: an invalid `mode` used to
  silently fall through to a `NameError`; now raises `ValueError`.
- Per-lag Otsu thresholding (fitting an independent cutoff at every lag)
  cost ~15-18 F1 points versus a single global fit -- removed in favor of
  `AdaptiveThreshold`/`otsu_global`.

### Changed
- `"atomic"` is now the recommended intervention strategy in the docs
  (`do_interventions` itself still defaults to `"full"` for backward
  compatibility).
- README split into a no-ground-truth "Quick Start" (real model, unknown
  generator) and an "Evaluation" section (known synthetic generator,
  validation-selected threshold, F1 measurement).

### Removed
- `mad`/`gmm`/`power_law`/per-lag-`otsu` threshold schemes from the
  comparison tooling in `scripts/train_and_test_lagged_effects.py`
  (consistently the worst performers or redundant with kept schemes).
- Internal/confidential experiment reports that were not intended for the
  public repo (`reports/response_to_chadyuk_*.md`,
  `reports/results_vocab100_*.json`, `reports/particle_sweep/`,
  `reports/results_threshold_law_sweep.json`) -- only `reports/figures/`
  (plots, no raw data) remains tracked; `.gitignore` updated to keep future
  raw results/reports local-only.

[0.1.7]: https://github.com/Mathugo/seq2cause/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/Mathugo/seq2cause/compare/v0.1.5...v0.1.6
