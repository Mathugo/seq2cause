# RUN — environment, deviation record, verification, run registry

The reproducibility record for `trace-bench-rca`, the harness that runs published root-cause
methods and their controls on the trace-bench corpora: what runs where, every numbered departure
from the published code or from the test plans sent to the methods' authors (each dated
**before** the run that depends on it), how a result is verified, and one registry row per run.
"The methods" are BARO (`phamquiluan/baro` `e35f4ec`, through the standard suite's runner
`phamquiluan/RCAEval` `259ea41`) and CausalRCA (`AXinx/CausalRCA_code` `21f4eb0`); "the sibling"
is `benchmarks/trace-bench` (seq2cause-trace-bench) at `14edb93c`, the source of the copied
method-agnostic modules (see `NOTICE`), which in turn adapted them from
`alex-chadyuk/trace-cmi-bench` at `f54776a`.

## Environment

- **Local:** one conda env from `environment.yml` (python 3.11; numpy 2, pyarrow 25, pandas,
  scikit-learn, huggingface_hub and the benchmark package at its pinned commit via
  `requirements.txt`), with the methods installed from their pinned commits as extras
  (`pip install -e ".[baro,rcaeval]"`). BARO and the controls run locally (seconds per case);
  CausalRCA runs locally at xs and s and on one GPU box at m if a case exceeds ~30 minutes
  (publication plan Q7, owner applies).
- **Runs:** every run writes `run/{arguments,results,run_meta}.json` (`run_meta` = repository
  commit, the installed method packages' versions, torch / CUDA when present, host RAM, CPU count,
  wall clock, peak memory, `pip freeze`, and the replica name and profile a stage is told);
  binaries ride object storage under content hashes with a location-free `artifacts.json`.
  Provisioning values never appear here.
- **Benchmark:** trace-bench corpora at tool version **0.3.0** only, pulled in the `rca` tier
  from the published release; the exporter and the two scorers come from the benchmark package
  at the commit `pyproject.toml` pins (D-TBR-1).
- **Staging ledger:** shared with the sibling (`../trace-bench/plans/staging-ledger.json`), so one
  freeze tag covers both directories (publication plan Q5 — pending the owner's confirmation).

## Deviations from the published code and the sent test plans (D-TBR-n)

Each entry states what it **follows** and what it **deviates from**. Dated 2026-10-14; all
predate the first run.

| id | follows | deviates from |
|---|---|---|
| D-TBR-1 | the sibling's benchmark pin `a3eb104` (`v0.3.0-score3`) | this directory needs the publication-branch commit that carries `tracebench.export_rca`; `pyproject.toml` pins `a3eb104` until the owner merges the exporter and then moves to that commit. The scorers (`score_rca`, `score_service`) are identical at both commits (both additive to the token scorer). **Moved 2026-10-19:** the pin is `ace384b4` (`clear-2027`, merge of trace-bench PR #12, 2026-10-19); `score.py` is unchanged between `a3eb104` and `ace384b4`, so every reference-harness cell stays byte-identical. |
| D-TBR-2 | RCAEval's `main.py` loading path (the suite's own evaluation driver) | not run: it slices `--length` minutes around the injection, drops every `_latency-50` column, derives the root cause from the case directory's name and needs the suite's directory naming. The adapter reads the export through `tracebenchrca.cases.read_table`, applies that loader's fill rule (`±inf → NaN`, forward fill, then 0; counted) and calls the suite's method function `e2e/baro.py::baro` on the whole case table with `dataset=None`, whose `preprocess` is the identity; the `time` column, which that path then scores as a metric, is removed from the returned ranking before the service collapse (`plans/rca-arms.md` §2). |
| D-TBR-3 | the suite's case directories `<service>_<fault>_<rep>/`, whose service part is the root cause | positional case directories `case-<k>/` (injection order) holding `data.csv`, `window.json` (`inject_time`, `end_time` only) and `services.json`; the pairing back to `labels/cases.json` is the same positional rule on the score side (`tracebench.export_rca.case_dirs`). |

Deviations for later stages are numbered here before the run that depends on them; the plans
under `plans/` reference these ids.

## Verification

- **Every commit:** `pytest tests/` green in the `trace-bench-rca` env (or the sibling's env with
  the benchmark's `src/` on `PYTHONPATH` while the pin predates the exporter);
  `tests/test_no_defaults.py` (no algorithm knob has a default), `tests/test_repo_hygiene.py`
  (no private identifier, location, corpus, export or binary in any file this work touches),
  `tests/test_notice_present.py` (copied code keeps its notice) and
  `tests/test_plans_registered.py` (the plans exist, are dated, and agree with the registry)
  fail the build.
- **Scaffold (2026-10-14):** the `rca` pull tier lands exactly the eleven files the directory
  reads and refuses corruption, symlinks, the wrong tool version and the presence of the two unread
  label files (`tests/test_pull_subset_verify.py`); the case door refuses every path but the three
  case files, demands exactly the two window fields, checks the services × metrics grid and
  applies the suite's fill rule with a count (`tests/test_cases_door.py`); the arm registry declares
  three classes and six arms with their inputs, axes and full-sha pins, the BOCPD row without the
  window, and the registered widths equal to the exporter's when the pin carries it
  (`tests/test_arm_registry.py`); the `export` stage records the exporter's results under the
  corpus id and refuses a width mismatch or a pin without the exporter
  (`tests/test_export_stage.py`); records carry every reproducibility field and the method
  packages' versions (`tests/test_record_fields.py`).
- **Per run (B4 onward):** the host's log shows the command exiting with status 0 and the output
  sync completing; `run/*.json` carry every field; a registry row lands below.
- **Per rung:** every arm's table from five seeds with the bootstrap interval, the random
  expectation beside it and the pooled row; the BOCPD row's no-change-point count; the hypotheses
  of `plans/rca-arms.md` §6 scored pass or fail in `findings/rca-latent.md`.

## Run registry

| date | run | rung | seeds | arm(s) | record | notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | no run yet (the scaffold predates the first run) |
