# RUN — environment, deviation record, verification, run registry

The reproducibility record for `seq2cause-trace-bench`, the harness that
scores the shipped seq2cause package (TRACE, Math & Lienhart 2026,
arXiv:2602.01135) on the trace-bench corpora: what runs where, every numbered
departure from the paper or from the shipped code (each dated **before** the
run that depends on it, PRD non-negotiable 12), how a result is verified, and
one registry row per run. "The shipped code" is seq2cause v0.1.9 (commit
`67416a2`); "the sibling" is `alex-chadyuk/trace-cmi-bench` at `f54776a`, the
source of the copied and adapted harness modules (see `NOTICE`).

## Environment

- **Local:** one conda env from `environment.yml` (python 3.11; torch, numpy 2,
  pyarrow 25, scipy, huggingface_hub, safetensors, transformers, accelerate,
  captum, jaxtyping, datasets, seaborn, tqdm and `trace-bench[sid]` at tag
  `v0.3.0` via `requirements.txt`), with the method under test installed from
  the repository root **without its dependency pins**:
  `pip install --no-deps -e ../..` (D-SB-1). Local use is the test suite on
  tiny fixtures and record rendering only: **nothing runs locally, the
  smallest rung included** (PRD non-negotiable 2, an operating rule of the
  owner; the package itself runs anywhere).
- **Runs:** cloud GPU instances provisioned by the workspace's private
  terraform, one dated replica file per run, no command-line overrides (PRD
  non-negotiable 3). The replica carries the verbatim invocation; the run
  record (`run/arguments.json`) republishes every algorithm value.
  Provisioning values never appear here.
- **Records:** every run writes `run/{arguments,results,run_meta}.json`
  (`run_meta` = repository commit, the shipped package's version, torch/CUDA,
  GPU model, host RAM, CPU count, wall clock, peak device and resident memory,
  `pip freeze`, and the replica name and AWS profile name a stage is told);
  binaries ride object storage under content hashes with a location-free
  `artifacts.json` manifest.
- **Benchmark:** trace-bench corpora at tool version **0.3.0** only; `report`
  refuses a cell that mixes tool versions.

## Deviations from the paper and from the shipped code (D-SB-n)

Each entry states what it **follows** and what it **deviates from**. Dated
2026-09-24 unless noted; all predate the first run.

| id | follows | deviates from |
|---|---|---|
| D-SB-1 | the shipped `pyproject.toml` (`numpy<2.0`, `pyarrow<17`, Python ≥ 3.8) | the package is installed `--no-deps` under numpy 2.4.6 / pyarrow 25.0.1 / torch 2.14.0 / transformers 5.17.0 / captum 0.9.0 (the shipped suite passes 160/160 under this set, verified 2026-09-24); retired when fix PR #4 (`fix-relax-pins`) reaches the author's main line. |
| D-SB-2 | `SampleLevelCausalDiscovery.run()` and `diagnostics.compute_cmi_matrix` as the entry points | the harness calls the shipped functions in the shipped order from **one** forward pass per `(probe, noise, c, N)` — `ancestral_sampling`, `uniform_sample`, `do_interventions`, the model forward, then `calc_lag_info_gain` / `calc_granger_score` (core) or `_predicted_true_token_(log_)probs` and `_cmi_matrix_from_*` (cli) — because each entry point returns a single coerced matrix while three read-outs (shipped KL, fixed KL, Granger) need the same tensor, and `run()` builds an `Accelerator` per instance. Bit-equal parity tests against both entry points on the fixture, and the shipped cut is checked against `seq2cause.cli.main`'s own boolean graphs. |
| D-SB-3 | the v0.1.9 Bernoulli KL (`clamp(p, 1e-9, 1 − 1e-9)` in float32) | fix PR #1 (`fix-fp32-clamp-logspace-kl`, `1cc3d83`) merged into the harness branch: `kl_mode="logspace"` evaluates the divergence from `log_softmax` in float64 (`seq2cause.kl`); the `fixed-kl` path runs it on the same forward and draws as the `shipped` path (`kl_mode="clamp"`, pinned bit for bit by `tests/fixtures/lag_info_gain_v019.json` in the shipped suite). The corrupted-cell counts (collapsed / saturated) are recorded on every path. |
| D-SB-4 | the v0.1.9 atomic construction and batch loop | fix PR #2 (`fix-atomic-and-batch`, `aaa2970`) merged: the atomic mask lives on the KL's device (as shipped it raised on every accelerator), `run()` processes every batch, batch size > 1 shapes, `strategy="atomic"` through `core` refused. No measured quantity changes at batch size 1 with `strategy="full"`; the harness moves the small atomic tensors to the CPU so `trace/cli-atomic` runs with or without the merge. |
| D-SB-5 | the v0.1.9 noise draw over `[0, V)` (specials included) | fix PR #3 (`fix-noise-real-ids`, `9ce172a`) merged: `noise_min_id = N_SPECIALS` draws over real event ids only on the `fixed` path (a separate forward, paired by corpus, seed and model hash but not by draw); the `shipped` and `fixed-kl` paths keep `noise_min_id = 0`. |
| D-SB-6 | bare integer ids, vocabulary inferred from `max(id) + 1` | tokens are the shipped `views/<view>/model-vocab.json` as-is (specials 0–3, base ops = `op:ok`, minted non-OK variants above the correlator's `min_count`); unminted variants fold to their base op (fold count reported by `prepare`) and the number of scoring-universe tokens the method can never emit is reported per corpus (sibling D-CB-16). |
| D-SB-7 | one dataset, one model | one model per corpus, trained on the `end-session` view; both grains are probed from that frozen model (request sequences from `end-request`, session sequences from `end-session` under `--max-len 64`); `end` ordering (callee precedes caller = the direction outcomes propagate in the target), `start` one flag away (sibling D-CB-17, D-CB-11). Sequence = `BOS + ids + EOS`, BOS counts toward `c`. |
| D-SB-8 | `--context-len 4` | `c` is a required knob swept blind on validation per grain (request {1, 2, 3}, session {2, 4, 8}) and frozen with τ and N; `guidance g = min(3, c)` for the core probe (sibling D-CB-4). |
| D-SB-9 | `--n-particles 32` | `N` is swept on {2, 8, 32} (xl: {2, 8, 16}) and frozen; the saliency and Shapley baselines have no particle axis. |
| D-SB-10 | the boolean per-sequence union projection (`summary_graph`) | the type-level score of a token pair is the **max** over every within-sequence occurrence (mean and count recorded beside it); within-operation pairs and special tokens are dropped at projection; two files per read (thresholded prediction + full ranking) because the scorer treats every listed edge as present (sibling D-CB-8, D-CB-12, D-CB-18). |
| D-SB-11 | one threshold rule (the CLI's pooled percentile with lag decay) | two cuts of one score table for the cli arms: `shipped` (the tool's rule, every knob passed explicitly and recorded, fitted on the full strict-upper triangle of every stored matrix exactly as `cli.py` pools it, no truth) and `frozen` (the validation-swept τ every arm uses). The shipped cut inherits `(c, N)` from its frozen sibling cell. |
| D-SB-12 | the shipped memory estimate (`cli.estimate_tensor_bytes`, the logits tensor only) | the harness estimate is twice the shipped one (logits plus the softmax) plus an activation term, refused above the declared cap of `plans/caps.md` (PRD scenario 18); shipped estimate, harness estimate, cap and measured peak are recorded per run. |
| D-SB-13 | no trainer in the package (the author's research script is toy-scale) | the sibling's clean-room trainer adapted to a Hugging Face `LlamaForCausalLM` built from a `LlamaConfig`, saved per checkpoint in the Hugging Face directory format; the model hash is the sha256 of `model.safetensors`; the oracle score ε̂ is taken against an independent order-2 n-gram entropy floor, never the minimum validation loss (sibling D-CB-7). The frozen model is the checkpoint `--model-choice` names: **`argmin-val`** (the validated step with the smallest validation loss, validated and checkpointed every 250 steps) since the 2026-09-25 addendum of `plans/caps.md`; the 2026-09-24 rule (last checkpoint + a `1.01×` trigger) was withdrawn after the xs calibration run, whose last checkpoint sat at 1.498× the curve minimum with the argmin at the first checkpoint. The oracle is reported at the chosen and at the last checkpoint. |
| D-SB-14 | the whole split | `--num-sequences` and `--sequence-sample {head, uniform}` per rung, recorded; the Shapley baseline probes its own, smaller sample; the reachable-recall coverage ceiling is reported per cell (freeze `diagnostics`, `annotate`), not gated — the 2026-09-24 rule (raise the sample below 0.90) was withdrawn on 2026-09-25 after the whole xs validation split gave 0.48 / 0.59. |

Deviations for later stages are numbered here before the run that depends
on them; the arm plans under `plans/` reference these ids.

## Verification

- **Every commit:** `pytest tests/` green in the harness env;
  `tests/test_no_defaults.py` (no algorithm knob has a default),
  `tests/test_repo_hygiene.py` (no private identifier, location or binary in
  any file this work touches) and `tests/test_notice_present.py` (copied code
  keeps its notice) fail the build. The shipped suite (`pytest tests/` at the
  repository root, 190 tests after the four fix merges) stays green.
- **M5 / M6 (`tests/test_engine_parity.py`, `tests/test_m6_pipeline.py`):** the engine
  adapter reproduces `SampleLevelCausalDiscovery.run()` and `compute_cmi_matrix` bit for
  bit on the fixture and the shipped cut equals `seq2cause`'s own CLI output; the fixture
  runs prepare → pretrain → sweep → scoresweep → freeze in a git repository → three test
  reads under the freeze → annotate → seqscore → report, and every refusal of scenarios
  3, 7, 9, 13, 22, 23, 29, 39, 46 and 47 is asserted; `score.json` is byte-identical to a
  direct `score_corpus` call (scenario 28). Since 2026-09-25: `pretrain --model-choice
  argmin-val` selects the argmin-validation checkpoint (`tests/test_pretrain_smoke_cpu.py`),
  the freeze records the soundness diagnostics (model choice, oracle, coverage ceiling,
  grid-edge flags) and refuses only tables bound to another model, and the replica checker's
  `--args-diff` compares a repeated module by output folder
  (`tests/test_tfvars_check_parses.py`); a replica may name a tracked job script
  (`bash scripts/jobs/<replica>.sh`, one command per line) when its chain would cross the
  provisioning user-data cap, and the checker follows it with the same parsers.
- **Per run:** the executing host's log shows the command exiting with
  status 0 and the output sync completing; `run/*.json` carry every
  scenario-24 field, the replica name and the profile; a registry row lands
  below.
- **Per rung:** `report` builds every cell from five seeds or records the
  reason; the estimator-difference and cut-difference tables exist per axis
  and per lag with the paired test; every bidirected column carries the
  structural-limitation note; per-sequence tables stand alone per grain with
  the scoreable fraction and the predict-all value; the wall clock lands within
  twice its estimate before the next rung opens.

## Run registry

One row per run, newest last. Object-store locations are never written here;
the run directory's `artifacts.json` (hashes only) and the private replica
are the pointers.

| date (UTC) | run | command | rung/variant/seed | split | model sha256 (8) | exit | wall clock | gpu | outcome |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | `xs-latent-s0-val-26-09-24` (Job V, replica `seq2cause-bench-26-09-24-xs-latent-s0-val.tfvars`) | pull → prepare → pretrain 12k × 256 → sweep request c{1,2,3} → sweep session c{2,4,8} (5 probes × 2 noises × N{2,8,32}, whole val split) → pull score → scoresweep × 2 | xs / latent / 0 | val | `c5c9d641` (last checkpoint, `--model-choice` predates this knob) | 0 (all 8 stages) | 2 h 41 min (pretrain 305 s; sweeps 59 + 95 min, Shapley 74 % / 89 %) | A10G, peak 3.07 GB | **calibration only, not frozen**: validation loss rose 2.20 → 3.30 from step 1000 (≈ 750 epochs), ε̂ = 0.62; coverage ceiling 0.48 / 0.59 on the whole split; session τ argmax at the grid top. Records verified against the manifest; `--args-diff` clean. Package commit `1832f45` (pre-rebase; tree = `cd1e48c` up to line wrapping in `cli.py` / `diagnostics.py` and the CHANGELOG order). Led to the 2026-09-25 addenda (`plans/`); superseded by the next xs replica. |
| 2026-09-25 | `xs-latent-s0-val-26-09-25` (Job V, second replica, `seq2cause-bench-26-09-25-xs-latent-s0-val.tfvars`) | the same chain with `--val-every 250 --checkpoint-every 250 --model-choice argmin-val` and the widened τ grids (plans' 2026-09-25 addenda) | xs / latent / 0 | val | `ba7725c4` (argmin-val checkpoint, step 500) | 0 (all 8 stages) | 2 h 40 min (pretrain 302 s; sweeps 58 + 92 min) | A10G, peak 3.07 GB | **frozen**: `freezes/2026-09-25-xs-latent-s0.json` (38 cells, no grid-edge τ). Backbone in regime: val loss 1.586 at step 500 vs the order-2 floor 1.608 (ε̂ = −0.008; the floor is a plug-in estimate, not a bound; last checkpoint 3.33). Zero corrupted cells; shipped vs fixed-kl ≤ 0.0013 F1. Val directed F1 (floor 0.05): trace arms 0.316–0.327 request, 0.341–0.367 session; granger 0.32–0.34; saliency 0.26 / 0.34; Shapley 0.27 / 0.30; predict-all 0.21. Shipped cut 5–65 edges (request), 15–162 (session). Records verified against the manifest; `--args-diff` clean on all 8 stages; package commit `56d6122`. Ledger: two `sweep` entries. Job T = `scripts/jobs/seq2cause-bench-26-09-25-xs-latent-s0-test.sh` (20 discover reads, 38 annotate, 38 seqscore). |
| 2026-09-25 | `xs-latent-s0-test-26-09-25` (Job T, `seq2cause-bench-26-09-25-xs-latent-s0-test.tfvars`, `scripts/jobs/seq2cause-bench-26-09-25-xs-latent-s0-test.sh`) | pull method → 20 × `discover --split test` under `freezes/2026-09-25-xs-latent-s0.json` → pull score → 38 × annotate → 38 × seqscore | xs / latent / 0 | test | `ba7725c4` | 0 (96 stages) | 61 min (discover 52 min, Shapley 1036 + 1756 s) | A10G, peak 3.07 GB | **landed**: records under `results/xs/latent/seed=0/` (cells, `test-read/`, `val-sweep/`, manifests), findings in `findings/xs-latent-s0.md` (one seed: H-estimator / H-fix / H-regime / H-baselines / H-granger-agree / H-shapley-cost pass; H-construction / H-lag / H-sat / H-perseq-request fail; H-hazard / H-cut n/a). Test F1 within −0.03 of the frozen validation values. 0 corrupted cells; `--args-diff` clean on all 98 records; package commit `6aaa3b4`. Ledger: 20 `discover` entries. `tables/` wait for seeds 1–4. |
| 2026-09-26 | `xs-latent-s1-val-26-09-25` (Job V, `seq2cause-bench-26-09-25-xs-latent-s1-val.tfvars`) | the seed-0 second-replica chain (argmin-val every 250 steps, widened τ grids, whole val split) | xs / latent / 1 | val | `3d9aaef0` (argmin-val checkpoint, step 500) | 0 (all 8 stages) | 174 min | A10G, peak 3.07 GB | **frozen**: `freezes/2026-09-26-xs-latent-s1.json` (38 cells, no grid-edge τ); ε̂ = −0.005 at the frozen checkpoint (in regime); 0 corrupted cells; shipped vs fixed-kl ≤ 0.002 F1; records verified against the manifest; `--args-diff` clean on all 8 stages; package commit `5ebbd02`. Ledger: two `sweep` entries. Job T = `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s1-test.sh`. |
| 2026-09-26 | `xs-latent-s2-val-26-09-25` (Job V, `seq2cause-bench-26-09-25-xs-latent-s2-val.tfvars`) | the seed-0 second-replica chain (argmin-val every 250 steps, widened τ grids, whole val split) | xs / latent / 2 | val | `7e39da00` (argmin-val checkpoint, step 500) | 0 (all 8 stages) | 182 min | A10G, peak 3.07 GB | **frozen**: `freezes/2026-09-26-xs-latent-s2.json` (38 cells, no grid-edge τ); ε̂ = +0.015 at the frozen checkpoint (in regime); 0 corrupted cells; shipped vs fixed-kl ≤ 0.002 F1; records verified against the manifest; `--args-diff` clean on all 8 stages; package commit `5ebbd02`. Ledger: two `sweep` entries. Job T = `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s2-test.sh`. |
| 2026-09-26 | `xs-latent-s3-val-26-09-25` (Job V, `seq2cause-bench-26-09-25-xs-latent-s3-val.tfvars`) | the seed-0 second-replica chain (argmin-val every 250 steps, widened τ grids, whole val split) | xs / latent / 3 | val | `0c6345ca` (argmin-val checkpoint, step 500) | 0 (all 8 stages) | 117 min | A10G, peak 3.07 GB | **frozen**: `freezes/2026-09-26-xs-latent-s3.json` (38 cells, no grid-edge τ); ε̂ = −0.003 at the frozen checkpoint (in regime); 0 corrupted cells; shipped vs fixed-kl ≤ 0.002 F1; records verified against the manifest; `--args-diff` clean on all 8 stages; package commit `5ebbd02`. Ledger: two `sweep` entries. Job T = `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s3-test.sh`. |
| 2026-09-26 | `xs-latent-s4-val-26-09-25` (Job V, `seq2cause-bench-26-09-25-xs-latent-s4-val.tfvars`) | the seed-0 second-replica chain (argmin-val every 250 steps, widened τ grids, whole val split) | xs / latent / 4 | val | `3478a111` (argmin-val checkpoint, step 500) | 0 (all 8 stages) | 166 min | A10G, peak 3.07 GB | **frozen**: `freezes/2026-09-26-xs-latent-s4.json` (38 cells, no grid-edge τ); ε̂ = +0.092 at the frozen checkpoint (in regime); 0 corrupted cells; shipped vs fixed-kl ≤ 0.002 F1; records verified against the manifest; `--args-diff` clean on all 8 stages; package commit `5ebbd02`. Ledger: two `sweep` entries. Job T = `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s4-test.sh`. |
| 2026-09-26 | `xs-latent-s1-test-26-09-26` (Job T, `seq2cause-bench-26-09-26-xs-latent-s1-test.tfvars`, `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s1-test.sh`) | pull method → 20 × `discover --split test` under `freezes/2026-09-26-xs-latent-s1.json` → pull score → 38 × annotate → 38 × seqscore | xs / latent / 1 | test | `3d9aaef0` | 0 (98 stages) | 77 min (discover 68 min, Shapley 1204 + 2487 s); annotate 6 min, seqscore 27 s | A10G, peak 3.07 GB | **landed**: records under `results/xs/latent/seed=1/` (cells, `test-read/`, `val-sweep/`, manifests); 0 corrupted cells; test F1 within −0.037 of the frozen validation values; `--args-diff` clean on all 98 records; package commit `38bd891`. Ledger: 20 `discover` entries. Five-seed tables `tables/xs-latent/`; findings `findings/xs-latent.md`. |
| 2026-09-26 | `xs-latent-s2-test-26-09-26` (Job T, `seq2cause-bench-26-09-26-xs-latent-s2-test.tfvars`, `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s2-test.sh`) | pull method → 19 × `discover --split test` under `freezes/2026-09-26-xs-latent-s2.json` → pull score → 38 × annotate → 38 × seqscore | xs / latent / 2 | test | `7e39da00` | 0 (97 stages) | 55 min (discover 47 min, Shapley 953 + 1496 s); annotate 106 s, seqscore 23 s | A10G, peak 3.07 GB | **landed**: records under `results/xs/latent/seed=2/` (cells, `test-read/`, `val-sweep/`, manifests); 0 corrupted cells; test F1 within −0.037 of the frozen validation values; `--args-diff` clean on all 97 records; package commit `38bd891`. Ledger: 19 `discover` entries. Five-seed tables `tables/xs-latent/`; findings `findings/xs-latent.md`. |
| 2026-09-26 | `xs-latent-s3-test-26-09-26` (Job T, `seq2cause-bench-26-09-26-xs-latent-s3-test.tfvars`, `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s3-test.sh`) | pull method → 18 × `discover --split test` under `freezes/2026-09-26-xs-latent-s3.json` → pull score → 38 × annotate → 38 × seqscore | xs / latent / 3 | test | `0c6345ca` | 0 (96 stages) | 51 min (discover 44 min, Shapley 862 + 1443 s); annotate 89 s, seqscore 18 s | A10G, peak 3.07 GB | **landed**: records under `results/xs/latent/seed=3/` (cells, `test-read/`, `val-sweep/`, manifests); 0 corrupted cells; test F1 within −0.037 of the frozen validation values; `--args-diff` clean on all 96 records; package commit `38bd891`. Ledger: 18 `discover` entries. Five-seed tables `tables/xs-latent/`; findings `findings/xs-latent.md`. |
| 2026-09-26 | `xs-latent-s4-test-26-09-26` (Job T, `seq2cause-bench-26-09-26-xs-latent-s4-test.tfvars`, `scripts/jobs/seq2cause-bench-26-09-26-xs-latent-s4-test.sh`) | pull method → 20 × `discover --split test` under `freezes/2026-09-26-xs-latent-s4.json` → pull score → 38 × annotate → 38 × seqscore | xs / latent / 4 | test | `3478a111` | 0 (98 stages) | 63 min (discover 54 min, Shapley 822 + 2110 s); annotate 109 s, seqscore 21 s | A10G, peak 3.07 GB | **landed**: records under `results/xs/latent/seed=4/` (cells, `test-read/`, `val-sweep/`, manifests); 0 corrupted cells; test F1 within −0.037 of the frozen validation values; `--args-diff` clean on all 98 records; package commit `38bd891`. Ledger: 20 `discover` entries. Five-seed tables `tables/xs-latent/`; findings `findings/xs-latent.md`. |
