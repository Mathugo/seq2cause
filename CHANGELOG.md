# Changelog

All notable changes to this project are documented in this file.

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

[0.1.6]: https://github.com/Mathugo/seq2cause/compare/v0.1.5...v0.1.6
