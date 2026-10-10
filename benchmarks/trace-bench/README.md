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

**How to read the two axes** (`plans/per-sequence-rules.md`, addendum 2026-10-04). For a
root-cause-analysis reading — which events of the trace in hand caused which — the
**per-sequence axis leads**: it scores each probed trace on its own position pairs. The
type-level axis answers how much of the system's graph a fixed number of traces recovers; its
headline is the benchmark scorer's output, and every type-level table also carries precision,
recall and F1 against the **reachable** truth only (the true edges whose two event types occur
together in the probed sample), because the benchmark F1 falls with the reachable share as the
vocabulary grows. The frozen τ is not re-selected on the reachable columns. Cross-rung reading:
[`findings/rca-reading-latent.md`](findings/rca-reading-latent.md).

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
| M7 | xs: Job V (validation sweep) per corpus → freeze → Job T (test read) | xs `latent/seed=0` calibration run landed 2026-09-25 (not frozen: the backbone overfit under the 12k-step budget; see `RUN.md` registry). Protocol amended by dated addenda in `plans/` (argmin-validation checkpoint, coverage ceiling reported not gated, wider τ grids); second xs replica landed 2026-09-25 with an in-regime backbone and **frozen** (`freezes/2026-09-25-xs-latent-s0.json`); Job T landed 2026-09-25 — `results/xs/latent/seed=0/`, `findings/xs-latent-s0.md` (one seed, provisional); seeds 1–4 validation runs landed and **frozen** 2026-09-26 (`freezes/2026-09-26-xs-latent-s{1..4}.json`), their Job T scripts authored (`scripts/jobs/`); **all four test reads landed 2026-09-26** — `results/xs/latent/seed={1..4}/`, five-seed tables `tables/xs-latent/` (`report`, 38 cells, 12 pairs, per-sequence document), findings scored on the five-seed paired means in `findings/xs-latent.md` (supersedes `xs-latent-s0.md`). Next: the `s` rung |
| M8 | s: Job V per corpus (10k / 5k validation sample, Shapley 2k / 500 via `sweep --shapley-sequences`) → freeze → Job T | seed-0 Job V landed 2026-09-27 (13 h; the per-sequence probe costs equal xs, the benchmark scorer's quadratic mixed-SHD loop cost 4.5 h — fixed upstream) and **frozen** (`freezes/2026-09-27-s-latent-s0.json`, step 6750, ε̂ −0.10); Job T script authored (`scripts/jobs/`); **seed-0 test read landed 2026-09-27** — `results/s/latent/seed=0/`, `findings/s-latent-s0.md` (one seed, provisional; test F1 above validation on every frozen cell — the registered 10k / 5k vs 20k / 10k sample asymmetry, `plans/caps.md` addendum 2026-09-27); seeds 1–4 Job V landed 2026-09-28 (8 h 19 min – 10 h 28 min at the 12 h cap; scoresweeps minutes under the pinned scorer) and **frozen** (`freezes/2026-09-28-s-latent-s{1..4}.json`, argmin steps 4000 / 7500 / 5500 / 8000, all in regime); their Job T scripts authored (`scripts/jobs/`); **all four test reads landed 2026-09-28** — `results/s/latent/seed={1..4}/`, five-seed tables `tables/s-latent/` (`report`, 38 cells, 12 pairs, per-sequence document), findings scored on the five-seed paired means in `findings/s-latent.md` (supersedes `s-latent-s0.md`; headline: Granger beats `trace/core` at the session grain on all five seeds, p = 0.045 — H-baselines fails at `s`). Next: the `m` rung |
| M9 | m: Job V per corpus → freeze → Job T | seed-0 Job V landed 2026-09-29 (13 h 28 min at the owner-raised 24 h cap; the linear scorer costs 5.5 h on the ≈ 40× m universe — scoring now rivals the sweeps; probe costs stay flat in V) and **frozen** (`freezes/2026-09-29-m-latent-s0.json`, step 7500, ε̂ −0.02, **0 corrupted cells at the first H-hazard-live rung**); Job T script authored (`scripts/jobs/`, 16 reads, cap 12 h); seeds 1–4 Job V replicas authored; **seed-0 test read landed 2026-09-30** (13 h 46 min — annotate 9.9 h is the m-rung cost driver; the applied 18 h cap saved the run over the addendum's 12, re-anchored `plans/caps.md` 2026-09-30) — `results/m/latent/seed=0/`, `findings/m-latent-s0.md` (one seed, provisional; test F1 above validation on every frozen cell +0.027 … +0.068 — the registered sample-size note; H-baselines fails by sign at both grains; H-hazard fails on this corpus — 0 corrupted cells at the first live rung; H-perseq flips to pass); **seeds 1–4 Job V landed 2026-09-29** (14 h 25 min – 15 h 24 min at the 24 h cap; 0 corrupted cells) and **frozen 2026-09-30** (`freezes/2026-09-30-m-latent-s{1..4}.json`, argmin steps 4000 / 4250 / 6500 / 7750, ε̂ −0.004 … −0.027, all in regime, no grid-edge τ); their Job T scripts authored (`scripts/jobs/`, 16 / 17 / 16 / 16 reads, cap 18 h); **all four test reads landed 2026-10-01** (15 h 49 min – 16 h 50 min — annotate 680–745 min is the m cost driver; the slowest margin at the 18 h cap was 1.07×) — `results/m/latent/seed={1..4}/`, five-seed tables `tables/m-latent/`, findings scored on the five-seed paired means in `findings/m-latent.md` (supersedes `m-latent-s0.md`; headline: Granger ≥ `trace/core` by sign at both grains, session p = 0.050; **H-hazard fails at `m`** — 0 corrupted cells on every read of all five latent corpora; H-perseq passes everywhere but saliency-request). Next: the `l` rung |
| M10 | l: Job V per corpus → freeze → Job T | seed-0 Job V landed 2026-10-02 (46 h 23 min, of which 39 h is the universe-bound scorer the 2026-10-01 re-pin removes; probe costs still flat in V) and **frozen** (`freezes/2026-10-02-l-latent-s0.json`, step 1250, ε̂ +0.04, 0 corrupted cells; the validation sample reaches 6.6 % / 10.3 % of the truth edges, so validation F1 is ceiling-bound at 0.09 / 0.11 and the Granger request τ sits at its grid bottom — recorded); Job T script authored (`scripts/jobs/`, 16 reads, cap 24 h); **seeds 1–4 Job V landed 2026-10-02** on the sparse-context scorer (8 h 24 min – 10 h 06 min at the 48 h cap; scoresweeps 11–12 + 31–37 min against seed 0's 11.9 + 27.4 h; 0 corrupted cells) and **frozen 2026-10-03** (`freezes/2026-10-03-l-latent-s{1..4}.json`, argmin steps 1750 / 750 / 1500 / 3000, ε̂ +0.04 … +0.06, all in regime; seed 2 freezes five request cells at a grid bottom — recorded); their Job T scripts authored (`scripts/jobs/`, 16 / 18 / 17 / 18 reads, cap 24 h); **all five test reads completed discover 2026-10-03 and were stopped in annotate** (a per-edge rebuild of the truth pair set in the confounded-pair count: hours per cell at `l`; fixed, byte-identical, 68 s against 155 min on the landed cell) — the scoring half ran again from the synced reads (`scripts/jobs/…-annotate.sh`; a 16 GB host was killed for memory on the first session cell, a 32 GB CPU host completed in 2.6–3.3 h) and **landed 2026-10-04**: `results/l/latent/seed={0..4}/`, five-seed tables `tables/l-latent/`, findings on the paired means in `findings/l-latent.md` (headline: all arms but Shapley within 0.006 at request and 0.013 at session; **saliency tops request on every seed** — the first significant H-baselines failure not involving Granger; **H-hazard fails at `l`** — 0 corrupted cells on every read; H-construction's lag gradient is gone; the validation sample reaches 7–12 % of the truth edges, so both jobs are ceiling-bound). Next: `xl` |
| M11 | xl: Job V per corpus as two instances (training half on the GPU, scoring half on a CPU host) → freeze → Job T as two instances (discover on the GPU, annotate + seqscore on a CPU host) | training halves landed on all five seeds 2026-10-04 (7 h 22 – 7 h 58 min, 0 corrupted cells, probe costs still flat in V); scoring halves 2026-10-05, twice — on the registered grid (every request-grain cell at the bottom of its τ grid, the quantile baselines at `p50`) and on the extended grids of the 2026-10-05 addenda (re-scored on the CPU, 2 h 17 – 2 h 34 min, 41–48 GiB; identical on every shared row); **frozen** 2026-10-05 (`freezes/2026-10-05-xl-latent-s{0..4}.json`, `freeze@3`: the every-scored-pair reference line beside each cell; a negative τ — Shapley's `p0` — is not a candidate). **Job T landed on all five seeds 2026-10-06/07**: discover halves 3 h 15 – 3 h 36 min on the GPU (0 corrupted cells), scoring halves 14 h 56 min – 20 h 02 min on a 256 GiB host (annotate 30–36 GiB per request cell, 70–81 GiB per session cell — the class the 2× rule asked for); `tables/xl-latent/`, `findings/xl-latent.md`: at request every arm sits on the every-observed-pair reference line (0.084–0.089; saliency and cli-atomic, which *are* that cut, top it — H-baselines fails), at session `trace/core` is the top arm (0.097; above cli, cli-atomic and Granger, p ≤ 0.039); H-hazard fails on the fifth rung (0 corrupted cells); the shipped cut −0.09; `findings/rca-reading-latent.md` carries the `xl` column. **M11 closed; the rung ladder is complete.** |
| M12 | floors: the `floor` arm class (`floor/topology`, `floor/bigram`; `plans/floors.md`, D-SB-15/16) → floors on xs … xl (five seeds each, floor freezes, test reads, landing beside the arms) → the start-ordered-view read on xs and s | class built 2026-10-09 (`floors`, `floorprior`, `freezecheck`; `scoresweep --floor-taus`; `freeze` families; model-free exemptions in `annotate` and `report`; `tests/test_floors.py`); scorer pin moved to `a3eb104` (`v0.3.0-score3`) 2026-10-10. **xs floors landed 2026-10-10** (local: five floor freezes `freezes/2026-10-10-xs-latent-s{0..4}-floors.json`, test reads, `tables/xs-latent/` regenerated with the floor rows and pairs, `findings/floors-latent.md`): the topology floor's request F1 0.820 ± 0.072 is above every learned arm on every seed (sample-free: validation and test reads coincide except for τ), the bigram floor ties `trace/core` at both grains (+0.018 / +0.001, n.s.), and the learned arms' session-AUROC advantage over the bigram floor holds for every arm but Shapley. **s floors landed 2026-10-10** (same chain, 10k / 5k validation and 20k / 10k test samples): topology 0.839 ± 0.014 request / 0.740 session, bigram 0.306 / 0.342 — the same three verdicts; the bigram floor's test F1 lands +0.02 … +0.05 above its frozen validation value (the registered sample-size note, measured). Next: m locally, then l and xl replicas. |

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
| `floors` | the two model-free floor arms (`plans/floors.md`): `floor/topology` reads the shipped deployment-topology prior on the score side (never through `Corpus`), `floor/bigram` reads the views only; one call writes both arms' score tables for one split and grain, and under a committed floor freeze the prediction and ranking files of a test read |
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
