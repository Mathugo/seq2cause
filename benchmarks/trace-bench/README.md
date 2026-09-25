# seq2cause-trace-bench

The benchmark harness that scores **the shipped `seq2cause` package** — TRACE
(Math & Lienhart 2026, *Your Autoregressive Model Already Reveals the Causal
Graph*, arXiv:2602.01135) — on the
[trace-bench](https://github.com/alex-chadyuk/trace-bench) corpora
(dataset [`chadyuk/trace-bench`](https://huggingface.co/datasets/chadyuk/trace-bench),
CC BY 4.0) through the benchmark's own scorer. It lives on the
`trace-bench-harness` branch of the author's repository, under this
directory, and never touches the package outside of bug-fix pull requests
opened against the author's main line.

## What this directory claims

A reference result for the author's own implementation, not a
state-of-the-art claim: the shipped code, run as shipped, on every rung,
variant, grain and seed of the benchmark, on every axis its scorer computes,
on a type-level axis (token pairs) and a per-sequence axis (position pairs
within one trace), under a validation → freeze → test protocol, with every
departure from the paper or from the shipped code numbered and dated in
[`RUN.md`](RUN.md). The two shipped estimators are two arms and their
difference is a result; the float32 clamp hazard is counted, not assumed;
each fixed bug runs on both its shipped and its fixed path, paired. The
results are co-published with the trace-bench dataset paper.

## Status

| milestone | what lands | state |
|---|---|---|
| M0 | the four bug-fix branches (Bug Policy), merged into the harness branch | done 2026-09-24 |
| M1 | scaffold: packaging, constants and arm registry, records, artifacts, hygiene / no-defaults / notice gates, deviation register | built 2026-09-24 |
| M2 | corpus adapter, vocabulary, `pull`, `prepare`, entropy floor, fixture corpus | built 2026-09-24 |
| M3 | `pretrain` (Hugging Face Llama backbone) | built 2026-09-24 |
| M4 | pre-registered plans (arms, baselines, per-sequence rules, caps, staging ledger) | committed 2026-09-24, before the engine code |
| M5 | engine adapter over the shipped functions, projection, selection, the two cuts | built 2026-09-24 (bit-equal parity against `run()` and `compute_cmi_matrix`; the shipped cut equals the shipped CLI's output) |
| M6 | `sweep`, `scoresweep`, `freeze`, `discover`, `annotate`, `seqscore`, `report` | built 2026-09-24 (`tests/test_m6_pipeline.py`) |
| M7 | xs: Job V (validation sweep) per corpus → freeze → Job T (test read) | xs `latent/seed=0` calibration run landed 2026-09-25 (not frozen: the backbone overfit under the 12k-step budget; see `RUN.md` registry). Protocol amended by dated addenda in `plans/` (argmin-validation checkpoint, coverage ceiling reported not gated, wider τ grids); second xs replica landed 2026-09-25 with an in-regime backbone and **frozen** (`freezes/2026-09-25-xs-latent-s0.json`); Job T authored (`scripts/jobs/`) |
| M8+ | the ladder s → m → l → xl | — |

## Install

```bash
# from this directory
conda env create -f environment.yml
conda activate seq2cause-trace-bench
pip install --no-deps -e ../..        # the method under test, without its dependency pins (RUN.md D-SB-1)
pip install -e ".[dev]"
pytest tests/
```

The benchmark package is pinned to the release tag whose corpora this
directory scores (`trace-bench[sid] @ v0.3.0`); `sid` pulls `gadjid` for the
causal-validity axis on the observable twin.

## Commands

Every command runs as `python -m seq2causebench.<command>`. **Every knob is a
required flag; nothing has a default** (`tests/test_no_defaults.py` fails the
build otherwise), so a run's `run/arguments.json` reconstructs it exactly.

| command | what it does |
|---|---|
| `pull` | fetch one corpus in its method tier (views only) or score tier (adds the graphs, after the method has exited) from the dataset host and verify every file against the corpus manifest |
| `prepare` | report the substrate the method can see: rows per split, vocabulary size, folded outcomes, length quantiles, entropy floor |
| `pretrain` | train one Llama backbone on a corpus's training split by next-token prediction; report the oracle score and periodic checkpoints |
| `sweep` / `freeze` | blind grid on the validation split (arms × context × particles), both grains; a dated, committed record of the chosen values |
| `discover` | run one probe on one corpus with one frozen model; write every sequence's score matrix, the type-level scores, and the prediction and ranking files for every arm, path and cut of that probe; a test read asserts on a committed freeze |
| `scoresweep` / `annotate` | thin wrappers that call the benchmark's scorer and add provenance, coverage and corruption fields; no metric arithmetic |
| `seqscore` | the per-sequence axis: score a read's stored matrices within each sequence against the grain's induced truth |
| `report` | five-seed tables per cell, paired estimator / construction / cut / fix differences with a paired t-test |
| `artifacts` | object-store push/pull/verify of a run directory; the manifest it writes carries hashes, never a location |

Scoring is not a command here: `tracebench.score` is called on the
prediction files this tool writes, and its report is consumed unchanged.

## What the method may read

A method reads a corpus only through the benchmark's accessor
(`tracebench.allowlist.open_for_method`), which permits the raw feed and the
correlated views and refuses the graphs, labels, oracle and manifest. The
vocabulary is the views' own `model-vocab.json`, never the alphabet record.
The views' parent column, although method-readable, is never handed to the
method: only the per-sequence scorer reads it, on the score side.

## Layout

```
src/seq2causebench/  the package (one build_parser()/main(argv) per command module)
tests/               flat pytest suite; tests/fixture_corpus.py writes a tiny corpus that passes the scorer's self-check
RUN.md               environment, numbered deviations D-SB-n, verification, run registry
NOTICE               the sibling harness's MIT notice and commit for every copied or adapted file
plans/ findings/     pre-registered plans and scored findings per arm; the staging ledger
freezes/             dated frozen-threshold records, one per corpus
results/ tables/     per-run score, annotation and per-sequence records; five-seed tables
scripts/             replica checker for the private provisioning layer
```

Trained models, score matrices, prediction and ranking files live in object
storage under content hashes and are never committed; corpora never enter the
repository. Provisioning values (instance types, storage locations, account
identifiers) live outside it; `tests/test_repo_hygiene.py` fails the build if
any reach a file this work touches.

## Provenance

The harness's method-agnostic layer is copied or adapted from
[trace-cmi-bench](https://github.com/alex-chadyuk/trace-cmi-bench) at commit
`f54776a` (MIT; see `NOTICE`). The engine it drives is the shipped
`seq2cause` package, called in process; no reimplementation of the discovery
algorithm and no code from any private predecessor enters this directory.

## Licence

MIT (see `LICENSE`). The corpora are CC BY 4.0 under their own release.
