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
| B5 | `rca/causalrca`: adapter, graph-axis threshold sweep → freeze → test read; ranking and service-graph axes | **landed 2026-10-23** (`rankcausalrca` → `scorecausalrca` → `graphaxis` / `graphfreeze`, code `88b3b325`; 1,000 local CPU trainings, no GPU needed): Avg@5 at or below random on every rung (`latency-50` / `latency-95`: xs 0.596 / 0.622, s 0.248 / 0.204, m 0.040 / 0.034); service graph F1 0.14–0.27, AUROC 0.43–0.52 on the four test seeds, against `floor/topology` ≥ 0.997 — **H-graph-topology passes** |
| B6 | the service-level graph table: CausalRCA beside `floor/topology` | **landed 2026-10-23** with B5 (`tables/rca-latent/`, graph axis; four test seeds) |
| B7 | stretch arms (publication plan B7; owner decisions 2026-10-25): `rca/circa` (the suite's port, verbatim, D-TBR-8), `rca/circa-topology` (CIRCA's structural graph from the shipped topology, D-TBR-9), `rca/rcd` (the suite's RCD environment, ten seeded runs, D-TBR-6 / D-TBR-7), `rca/tracerca` (the suite's reimplementation on the view's spans, D-TBR-10) on xs, s, m × five seeds | registered 2026-10-25 (`plans/rca-arms.md` addenda); **`rca/circa` and `rca/rcd` landed 2026-10-25** (`findings/rca-latent.md`): `rca/rcd` Avg@5 0.818 / 0.900 / 0.518 — above `control/rule` on every rung (H-rca-rcd-rule passes, the first arm to do so) and above random at m; `rca/circa` 0.620 / 0.427 / 0.000 (the port's 1 s head on 15–60 s bins). **`rca/circa-topology` and `rca/tracerca` landed 2026-10-26** on the re-exported corpora with the extras (pin `fcb7ddb3`, D-TBR-11; the three registered case files byte-identical to the 2026-10-21 exports): `rca/circa-topology` 0.500 / 0.253 / 0.024 — below the PC cell at xs and s (H-circa-topology-gain fails); `rca/tracerca` 0.640 / 0.520 / 0.088 — ties the rule control at xs, below it at s and m, near random at m |
| B8 | `findings/rca-latent.md`: hypotheses scored, deviation notes, the outreach record | at the freeze |
| B10 | stretch arms `rca/trace` and `rca/trace-core` (owner go 2026-10-30; publication plan B4b): the reference algorithm's own root-cause rule — the seed's frozen backbone and frozen request cell (`trace/cli-atomic`, `trace/core`), the window traces' sample graphs, parents of each non-OK token, upstream closure, traces-upstream counts (D-TBR-13) on xs, s, m × five seeds | registered 2026-10-30 (`plans/rca-arms.md` addendum); **landed 2026-10-30** (`findings/rca-latent.md`): `rca/trace` Avg@5 0.540 / 0.333 / 0.080, `rca/trace-core` 0.560 / 0.307 / 0.144 at xs / s / m — below `control/rule` on every rung (H-rca-trace-random / -rule and the core pair fail), pool exhaustion at s found (0.72) and the m degrade faults the best m cells; the sample graphs reach every service, the root cause's median rank 14–15 of 52 at m |
| B11 | the **per-trace root-cause axis** (owner decision 2026-10-30, plan approved 2026-10-31): `rca/trace` and `rca/trace-core` scored per failed request against `labels/sequence-labels.parquet` (hit@1 / hit@3 of the root-cause service and position, the exact candidate set; readings `hop1` / `hop0`), beside `floor/trace-topology` (the rule on the true call tree), `control/trace-random`, `control/trace-callees` and the eleven landed case rankings broadcast to the trace; axis `trace`, stages `ranktrace --axis trace` (uncapped selection R, GPU replica, D-TBR-15), `tolerance`, `scoretrace`, `reporttrace`; scorer `tracebench.score_rca_trace` (D-TBR-14) | registered 2026-10-31 (`plans/rca-arms.md` addendum); landed 2026-11-01 on xs, s, m × five seeds (g5 replica for the probes; `tables/rca-trace/`, `findings/rca-trace.md`): the four hypotheses fail; the tolerance gate fails on one run of 30 (disclosed, owner decision) |
| B9 | stretch arm `rca/pc-pagerank` (owner go 2026-10-28): the suite's PC graph — the one CIRCA's port builds — under the PageRank head CausalRCA uses, named-dataset preprocessing (D-TBR-12), one deterministic run per case, no window; on xs, s, m × five seeds | registered 2026-10-28 (`plans/rca-arms.md` addendum); landing in `findings/rca-latent.md` |

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
| `export` | run the benchmark's exporter on a pulled corpus (one case directory per fault; `--extras topology spans` for the stretch arms' files) and record it with this repository's commit |
| `artifacts` | object-store push / pull / verify of a run directory; the manifest it writes carries hashes, never a location |
| `rank` | method side: one ranking arm (`rca/baro`, `rca/baro-bocpd`, `rca/circa`, `rca/circa-topology`, `rca/tracerca`, `rca/pc-pagerank`, `control/rule`) over every case of one export, through the case door and only the arm's declared inputs (`rca/circa-topology` reads the exporter's `topology.json` extra, `rca/tracerca` its `spans.csv`); rankings keyed by case directory |
| `score` | score side: restores the case ids by position, writes `ranking.json` and the benchmark's `score-rca.json` (AC@k, Avg@5, bootstrap, the random expectation = `control/random`), and the full pipeline's change-point offsets |
| `ranktrace` | method side: `rca/trace` or `rca/trace-core`, every case of one export — the reference harness's environment runs `trace_runner.py` once per corpus (the seed's frozen backbone, the arm's frozen request cell of the committed freeze, the window traces of the end-ordered request view through the benchmark's method accessor, one probe pass per trace, the frozen cut), then the rule (`tracerule.py`: parents of each non-OK token, upstream closure, positions → services, traces-upstream counts with mean-hop ties; D-TBR-13) writes the ranking; scored by `score` |
| `rankrcd` / `scorercd` | method side: `rca/rcd`, every case of one export — the suite's RCD (named-dataset preprocessing, D-TBR-6) ten times per case with seed = run index through `rcd_runner.py` in the pinned RCD environment (`environment-rcd.yml`, D-TBR-7); score side: the run-averaged aggregation of `scorecausalrca` |
| `rankcausalrca` | method side: `rca/causalrca`, one series, every case of one export — the authors' `train_latency.py` at the pin with its data block substituted (D-TBR-5), ten seeded trainings per case through `causalrca_runner.py` in the pinned environment (`environment-causalrca.yml`); per case every run's ranking and the mean `|adj|` |
| `scorecausalrca` | score side: per-run ranks by the benchmark's `case_ranks`, hits averaged over runs per case, the bootstrap on the benchmark's draws; `ranking.json`, `score-rca.json`, `graph.json` (`svc:<index>` predictions) |
| `graphaxis` | score side: each case's learned service graph scored by `tracebench.score_service` beside `floor/topology` — the τ grid on seed 0 (validation) or the frozen τ on seeds 1–4 (test) |
| `graphfreeze` | the graph axis's τ per rung and series from the seed-0 validation read (argmax of the mean F1, ties → the larger τ), `freezes/<date>-<rung>-rca-causalrca-<series>-graph.json` |
| `report` | pools five seeds per rung and across rungs (`rca/causalrca`'s graph axis: the four test seeds), writes `tables/rca-latent/` and scores the hypotheses of `plans/rca-arms.md` §6 |

CausalRCA runs through `rankcausalrca` (its own stage: the pinned environment, ten trainings per case and series); the two TRACE rows through `ranktrace` (the reference harness's environment `seq2cause-trace-bench`, `environment.yml` of `benchmarks/trace-bench`); `control/rule` is a `rank` arm with no third-party code. Scoring is the
benchmark's: `tracebench.score_rca` on the per-case rankings, `tracebench.score_service` on the
service graphs, their reports consumed unchanged.

## Layout

`src/tracebenchrca/` (the package), `tests/` (fixtures and gates), `plans/` (pre-registrations),
`RUN.md` (environment, deviations, verification, run registry), `results/`, `tables/`,
`findings/` (land with the runs). Exports and corpora are local and git-ignored.
