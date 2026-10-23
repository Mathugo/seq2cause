# trace-bench-rca

The root-cause harness of the trace-bench dataset paper: **published root-cause methods, run
as published from pinned commits, and their controls**, scored on the
[trace-bench](https://github.com/alex-chadyuk/trace-bench) corpora (dataset
[`chadyuk/trace-bench`](https://huggingface.co/datasets/chadyuk/trace-bench), CC BY 4.0,
release `v0.3.0`) through the benchmark's own scorers. It is the sibling of the reference
harness `benchmarks/trace-bench` (which scores the author's own seq2cause / TRACE on the token
level) and copies its method-agnostic layer — records, pull, artifacts, staging, hygiene and
no-defaults gates, `NOTICE` — so the two directories share one run ledger and one freeze tag.
Nothing in it touches the package outside this directory.

## What this directory claims

A reference result for two published root-cause methods on a benchmark whose root causes are
labelled by construction, not a state-of-the-art claim: BARO (two rows — the standard suite's
injection-time configuration and the full change-point pipeline) and CausalRCA (ranking and the
learned service graph), beside an analytic random control, a trivial rule control and the
data-free deployment topology, on the three smallest rungs (3, 10 and 50 services; 10, 15 and 25
fault cases per rung over five generator seeds). Every departure from the published code or
from the test plans sent to the authors is numbered and dated in [`RUN.md`](RUN.md); every
arm, input, pin, rule and hypothesis is fixed in [`plans/`](plans/) before the first run.

**How the methods see the benchmark.** The benchmark ships no metric series, so the methods'
metrics are log-derived: the benchmark's own exporter (`tracebench.export_rca`, score side)
turns the raw access-log lines into per-service request count, 5xx share, latency p50 and p95
in fixed-width bins (xs 30 s, s 15 s, m 60 s, by a traffic rule registered before any ranking
was read) and writes one case directory per injected fault in the standard suite's shape —
`data.csv`, `window.json` (the injection and end times, nothing else from the labels) and
`services.json` (the candidate universe). A method opens those three files through one door
(`tracebenchrca.cases`) and nothing else; the root cause never leaves the labels.

## Status

| milestone | what lands | state |
|---|---|---|
| B1 | the exporter in the benchmark repository (`tracebench.export_rca`, branch `clear-2027`) | built 2026-10-14; PR open, owner merges; this directory's benchmark pin moves to the merge commit (D-TBR-1) |
| B3 | this scaffold: packaging, constants and arm registry (three classes, six arms), records, `pull --tier rca`, the case door, the `export` stage, artifacts, staging, hygiene / no-defaults / notice gates, `plans/rca-arms.md`, `plans/controls.md`, fixtures and tests | built 2026-10-14 |
| B4 | `rca/baro`, `rca/baro-bocpd`, `control/random`, `control/rule` on xs, s, m × five seeds: adapters, runs, landing (results, ledger, tables with intervals and pooled rows) | **landed 2026-10-21** for both BARO rows and `control/random` (`findings/rca-latent.md`, `tables/rca-latent/`); **`control/rule` landed 2026-10-23** (`rank --arm control/rule`, `plans/controls.md` §2 with addenda 2026-10-22 / 2026-10-23): Avg@5 0.640 / 0.680 / 0.272 against `rca/baro`'s 0.720 / 0.467 / 0.176 — H-rca-rule fails at s and m (every interval overlaps) |
| B5 | `rca/causalrca`: adapter, graph-axis threshold sweep → freeze → test read; ranking and service-graph axes | after B4; Q7, Q16, Q17 decided 2026-10-22 (`plans/rca-arms.md` addendum: seed 0 = graph-axis validation, four test seeds; 1000 epochs = D-TBR-4; GPU at m) |
| B6 | the service-level graph table: CausalRCA beside `floor/topology` | after B5 |
| B8 | `findings/rca-latent.md`: hypotheses scored, deviation notes, the outreach record | at the freeze |

## Install

```bash
# from this directory
conda env create -f environment.yml
conda activate trace-bench-rca
pip install -e ".[dev,baro,rcaeval]"     # the methods under test from their pinned commits
pytest tests/
```

The benchmark package is pinned to a commit of its publication branch (`pyproject.toml`); the
run records name the installed revision of every method package through `pip freeze`.
CausalRCA has no package: its pinned clone is named by the record (`plans/rca-arms.md` §2).

## Commands

Every command runs as `python -m tracebenchrca.<command>`. **Every knob is a required flag;
nothing has a default** (`tests/test_no_defaults.py` fails the build otherwise), so a run's
`run/arguments.json` reconstructs it exactly.

| command | what it does |
|---|---|
| `pull` | fetch one corpus in the `rca` tier — the configuration, the instantiation, `labels/cases.json`, the access-log shards, the service view, the alphabet and the topology, nothing else — from the dataset host and verify every file against the corpus manifest |
| `export` | run the benchmark's exporter on a pulled corpus (one case directory per fault) and record it with this repository's commit |
| `artifacts` | object-store push / pull / verify of a run directory; the manifest it writes carries hashes, never a location |
| `rank` | method side: one ranking arm (`rca/baro`, `rca/baro-bocpd`) over every case of one export, through the case door and only the arm's declared inputs; rankings keyed by case directory |
| `score` | score side: restores the case ids by position, writes `ranking.json` and the benchmark's `score-rca.json` (AC@k, Avg@5, bootstrap, the random expectation = `control/random`), and the full pipeline's change-point offsets |
| `report` | pools five seeds per rung and across rungs, writes `tables/rca-latent/` and scores the hypotheses of `plans/rca-arms.md` §6 |

The CausalRCA command lands with B5; `control/rule` is a `rank` arm with no third-party code. Scoring is the
benchmark's: `tracebench.score_rca` on the per-case rankings, `tracebench.score_service` on the
service graphs, their reports consumed unchanged.

## Layout

`src/tracebenchrca/` (the package), `tests/` (fixtures and gates), `plans/` (pre-registrations),
`RUN.md` (environment, deviations, verification, run registry), `results/`, `tables/`,
`findings/` (land with the runs). Exports and corpora are local and git-ignored.
