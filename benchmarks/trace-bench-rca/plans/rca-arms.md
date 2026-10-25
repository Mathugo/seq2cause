# Root-cause arm plan (pre-registered 2026-10-14, before the first root-cause run)

Committed before any arm code exists. Registers the root-cause arms of the trace-bench dataset
paper (publication plan §3, stream B) as this directory's arm registry (`constants.ARMS`), the
export they read, the bin-width rule, the protocol and the hypotheses. Every number the paper
quotes for a root-cause arm comes from a read governed by this file. Sources, in order of
authority: the two test plans sent to the method authors on 2026-10-14 (the BARO plan to Luan
Pham, confirmed by his reply the same day — both rows and the naming of the injection-time
configuration as "BARO"; the CausalRCA plan to Peng Chen and Zhiming Zhao, no reply yet), the
owner's bin-width decision of 2026-10-14 (N = 10), the benchmark's pre-registered scorers
(`docs/scorers-service-rca.md`, 2026-10-09) and the facts read from the pinned code on
2026-10-14 (§2). Changes are dated addenda at the end, never edits in place. Deviation ids are
RUN.md's D-TBR-n. Open questions for the owner are marked **OPEN** and collected in §9.

## 1. Arms

| arm | class | declared inputs | axes | what it is |
|---|---|---|---|---|
| `rca/baro` | `root-cause` | `series`, `window`, `services` | ranking | **BARO, injection time given**: the standard suite's runner `RCAEval/e2e/baro.py::baro` at `259ea41` over `phamquiluan/baro` `e35f4ec`, unmodified, published defaults. The table is split at the labelled injection time; RobustScorer (sklearn `RobustScaler`, median and interquartile range of each metric on the pre-fault rows) scores each metric by the largest scaled value after the split and ranks metrics by it; no threshold, no p-value. The paper's `RobustScorer[t_inject]` row (Tables 3 and 4); labelled "BARO" as the suite does — confirmed by the author 2026-10-14. |
| `rca/baro-bocpd` | `root-cause` | `series`, `services` | ranking | **BARO, full pipeline**: `baro.anomaly_detection.bocpd(data)` at the published defaults on the same table (the detector locates the change point — this row never sees the injection time), then `baro.root_cause_analysis.robust_scorer(data, anomalies=<change points>)` from the first detected change point, as in Algorithm 1 of the paper. Cases in which the detector returns no change point are counted and reported beside the row (§5). |
| `rca/causalrca` | `root-cause` | `series`, `window`, `services` | ranking, graph | **CausalRCA**, `AXinx/CausalRCA_code` `21f4eb0` (master, 2 May 2023), the coarse-grained (service-level) configuration of the paper: the DAG-GNN structure learner (`train_latency.py`'s procedure — the VAE with the acyclicity constraint, Adam, `lr 1e-3`, `--gamma 0.25 --eta 10` as the script's defaults) on the window of per-service latency series after the labelled injection, the learned graph's edges reversed and absolute weights taken, PageRank (α = 0.85) over that graph for the service ranking; 10 runs per case averaged (the repository's own repetition count, `test_latency_per.sh`); one graph per fault case; run once on `latency-50` and once on `latency-95`, each reported. Licence line: "author repository, no licence file; authors contacted 2026-10-14; reply: <…>". |
| `control/random` | `floor-control` | `services` | ranking | the analytic expectation of a uniformly random ranking of the n deployed services: AC@k = min(k, n) / n (`tracebench.score_rca.random_expectation`); never sampled. `plans/controls.md` §1. |
| `control/rule` | `floor-control` | `series`, `window`, `services` | ranking | the trivial three-rule alert counter on the same series. `plans/controls.md` §2 — **OPEN**: the registration text is proposed there and awaits the owner (publication plan Q8). |
| `floor/topology` | `floor-control` | `topology` (score side) | graph | the deployment call topology coarsened to services, callee → caller, score = the largest `p_call` of the pair's op-level edges: `tracebench.score_service --prediction topology`. The data-free reference of the graph axis; the service-level counterpart of the reference harness's `floor/topology`. |

The `discovery` class (series only, no window; a learned structure) is declared with no member:
no root-cause arm of the paper is window-free. Stretch arms (`rca/tracerca`, `rca/rcd`,
`rca/circa`; publication plan B7) enter by dated addendum with their own code pins.

## 2. Code pins and the facts read from the code (2026-10-14)

Pins (`constants.py`; the installed revisions ride every record's `pip freeze`):
`phamquiluan/baro` `e35f4ec1095e5cac891d52de9ad18a5b32a37ec8` (MIT; `fse-baro` extra);
`phamquiluan/RCAEval` `259ea4167160256a74ad30004aa3d96a998c846e` (MIT; `RCAEval` extra);
`AXinx/CausalRCA_code` `21f4eb0d8139aa5046402f15e66fd56b65fd3844` (no licence file; a script
repository, cloned at the pin, named by the record, never installed or vendored). Nothing is
re-tuned or re-implemented; the adapters hand the exported table to the pinned functions.

Facts that fix the adapters (read, not assumed):

- `e2e/baro.py::baro(data, inject_time, dataset=None)`: `normal_df = data[data["time"] < inject_time]`,
  `anomal_df = data[data["time"] >= inject_time]`; `preprocess(dataset=None)` returns its input
  unchanged (the suite drops `time` and constant columns only when a dataset name is given) — so
  the `time` column is a scored "metric" of that path. **Registered:** the adapter passes
  `dataset=None` as the suite does for an unnamed dataset and drops the `time` column from the
  returned ranking before collapsing (it is not a candidate); nothing else is removed. The score
  is `max(zscores)` — the largest scaled value, signed, not the largest absolute deviation.
  A constant pre-fault column has IQR 0 and `RobustScaler` scales it by 1.
- The "fewer than four pre-fault bins are skipped" rule lives in `robust_scorer_dict`
  (the `metrics.json` path of `run_bench.py`), not in the DataFrame path the suite runs. Our
  windows hold ≥ 30 pre-fault bins at xs, ≥ 960 at s and ≥ 216 at m (§3), so the count the mail
  promised to report is 0 by construction and is reported as such.
- `baro.anomaly_detection.bocpd(data)`: selects the columns whose names contain `latency` or
  `_error` — this directory's `<service>_latency-50`, `<service>_latency-95` and
  `<service>_error` are read, `<service>_count` and `time` are not; min-max normalises each
  selected column; `drop_constant`; `online_changepoint_detection(constant_hazard(50),
  MultivariateT)`; `find_cps(maxes)` returns the row indices where the run-length argmax jumps
  by more than one. `robust_scorer(data, anomalies=cps)` then splits the table at the **first**
  detected row index (`data.head(anomalies[0])`). With an empty list `anomalies[0]` raises
  `IndexError`: **registered — a case without a change point is counted and scored as a miss
  (no ranking); the adapter never falls back to the injection time.**
- RCAEval's `main.py` loading path is **not** run (D-TBR-2): it slices `--length` minutes around
  the injection, drops every `_latency-50` column, derives the root cause from the case
  directory's name and needs the suite's directory naming. The adapter reads the export through
  `tracebenchrca.cases.read_table`, which applies that loader's own fill rule to the series
  (`±inf → NaN`, forward fill, then 0 — `main.py` at the pin) and counts the cells it filled,
  then calls the pinned `baro()` on the whole case table.
- CausalRCA: `config.py` sets `epochs = 500`, `batch_size 50`, `lr 1e-3`, `graph_threshold 0.3`
  (used only for the training-time edge count); `train_latency.py` reads a pickle of per-service
  `ctn_latency` series, builds the DAG-GNN on `data / data.max()`, runs the augmented-Lagrangian
  loop (`--gamma`, `--eta`), thresholds the learned matrix at 0.3 and ranks by PageRank over
  the transposed absolute graph (the RCAEval port `e2e/causalrca.py` is the same procedure).
  **OPEN (Q17):** the test plan sent to the authors said "1000 epochs"; the repository's default
  is 500. The adapter runs the repository's default unmodified and records the discrepancy;
  the owner decides whether to set 1000 (a one-line config change, recorded as a deviation).

## 3. The export and the bin width

The arms never see a corpus. `tracebench.export_rca` (benchmark repository, score side) writes
one case directory per injected fault, `exports/<corpus>/case-<k>/`, holding exactly three files
— `data.csv`, `window.json`, `services.json` — and the method side opens nothing else
(`tracebenchrca.cases.open_case_file`, the one door). Case directories are named by position
in injection order (`case-00`, …), never by the labelled case id (which carries the root-cause
service and the fault kind — D-TBR-3); the pairing back to `labels/cases.json` is the same
positional rule on the score side (`tracebench.export_rca.case_dirs`).

- **Source.** The raw feed's access-log lines (`raw/shard=*/vl.jsonl.gz`, `kind: vl.access`,
  one per request handled at a service's edge proxy; `ts`, `service`, `status`, `request_time`),
  `/healthz` probes dropped, pooled over the corpus before binning (per-pod clock skew moves
  lines across shard boundaries). Lines whose skewed timestamp falls outside the corpus window
  are dropped and counted. Owner decision 2026-10-14: the raw lines, not the correlated views.
- **Series.** Per deployed service and bin: `count` (requests), `error` (the share of responses
  with a 5xx status — 500 ≤ status < 600; the network-error outcome emits status 0 and is **not**
  in this share: 6.6 % of the xs seed-0 access lines, recorded per export as `n_status_0`),
  `latency-50` and `latency-95` (quantiles of `request_time`, linear interpolation, six
  decimals). Column names `<service>_<metric>`, the suite's shape (the service is the part before
  the first `_`; no benchmark service name carries one). `time` = UNIX seconds of the bin start.
  An empty bin has count 0, error 0 and empty latency cells; the loader's fill rule (§2) is the
  suite's, applied on the method side and counted.
- **Width rule (registered; plan Q9, owner 2026-10-14, N = 10).** w(rung) = the smallest grid
  width in {5, 10, 15, 30, 60, 90, 180, 300} s at which, on every seed, the sparsest candidate
  service's pre-fault median request count per bin reaches N = 10 and its 10th percentile reaches
  1, with at least five bins inside the shortest fault → **xs 30 s, s 15 s, m 60 s**. Source: the
  bin-traffic job of 2026-10-14 over xs/s/m × seeds 0–4 (sparsest-service pre-fault median count
  per bin, minimum over seeds — 30 s: xs 17 / s 28 / m 5; 60 s: 35 / 56 / 11; the sparsest m
  services are never root causes; shortest faults xs 300 s, s and m 900 s). Rejected: one minute
  everywhere (five bins in the xs crash; kept as an optional BARO sensitivity row, by addendum),
  N = 5 (m at 30 s has empty bins on the sparsest service), N = 20 (xs at 60 s). The widths are
  constants of the exporter and of this directory (`BIN_WIDTH_S`, asserted equal); every width
  divides the 900 s shard and every shipped fault boundary of xs, s and m. The rule was fixed
  before any ranking was read because the 2026-10-14 prototype's root-cause rank flipped between
  10 s and 30 s bins on xs seed 0.
- **Case window.** From the end of the previous fault (or the corpus start) to the labelled end
  of the fault; `window.json` carries `inject_time` and `end_time` only (UNIX seconds). Pre-fault
  bins: xs 40 / 30, s 960 / 1 320 / 1 140, m 216 / 258 / 273 / 220 / 190; fault bins: xs 20 / 10,
  s 120 / 60 / 80, m 30 / 15 / 25 / 40 / 20 (from the shipped schedules at the registered widths).
- **Universe.** `services.json` = the deployed services of `instantiation.json` (BFF, backend,
  external; the synthetic client excluded) — the n that `tracebench.score_rca` takes for the
  random expectation. The external service (`ext-*`, one per corpus) keeps its series in the
  table and is a candidate like any other.
- **Record.** The exporter's run record hashes every input it read (configuration,
  instantiation, `labels/cases.json`, every access-log shard) and names the two label fields
  read; `labels/faults.json` and `labels/sequence-labels.parquet` are never read. The sibling
  stage `tracebenchrca.export` wraps the call and records the same facts with this repository's
  commit (the benchmark's own record under `<export>/run/` carries the host name as every
  benchmark record does and is not landed).

## 4. Protocol

Per corpus (xs, s, m × seeds 0–4 of `chadyuk/trace-bench` at `v0.3.0`, latent variant):
`pull --tier rca` → `export` → each arm writes `ranking.json` (`{"cases": {"<case_id>": […]}}`,
the case id restored on the score side by position) and, for graph arms, `graph.json` →
`tracebench.score_rca` (B = 2000, seed 0) per corpus → pooled per rung (five seeds) and across
rungs with `--pool`. Ranking-only arms run at published defaults; the xs seed-0 export is their
smoke test (no value is chosen on it). Staging: `--prior-rung-record` names the previous rung's
completed record of the same stage and arm, through the ledger shared with the reference
harness (`constants.STAGING_LEDGER`; **OPEN** Q5 — the owner confirms the sharing). Compute:
BARO and the controls are seconds per case, local; CausalRCA is local at xs and s, m on one GPU
box if a case exceeds ~30 minutes locally (**OPEN** Q7; owner applies). The method environment is
`trace-bench-rca` (`environment.yml` + the `baro`/`rcaeval` extras), created before the first
run; every record names it through `pip freeze`.

Graph axis (CausalRCA only): the learned service graph as a `tracebench.score_service`
prediction (`svc:<index>` nodes through `services.json` ↔ `instantiation.json`, scores = the
absolute learned weights), scored against `graphs/views/service.json` at the view's default floor,
beside `floor/topology`. The edge threshold is swept on a validation split and frozen in a dated
freeze file before the test read. **OPEN (Q16):** the validation split. Proposal: the seed-0
corpus of every rung is the validation split of the graph axis (τ frozen per rung on its
cases' learned graphs, argmax of directed F1, ties → larger τ) and seeds 1–4 are its test read;
the ranking axis, which has no threshold, keeps all five seeds.

## 5. What is reported

Per arm, per rung and pooled across rungs: AC@1, AC@3, AC@5, Avg@5 with the 95 % percentile
bootstrap over cases (B = 2000, seed 0), the random expectation beside every table, rows per
fault kind. BARO's metric ranking is collapsed to services by each service's best-ranked metric
(the suite's coarse-grained collapse, first occurrence kept; the `time` column removed). For
`rca/baro-bocpd`: the number of cases without a change point (scored as misses) and the detected
change point's offset from the labelled injection, per case. For every BARO row: the number of
metrics with fewer than four pre-fault bins (0 by construction). For every case: the cells the
loader's fill rule changed. For CausalRCA: directed precision, recall, F1 and AUROC on the
service graph at the frozen τ, beside `floor/topology`, and the ranking metrics above; the
licence line with the authors' reply or its absence. The contact dates (2026-10-14, both mails)
and the replies (Pham 2026-10-14; Chen and Zhao: none yet) go to `findings/rca-latent.md` and
the paper's appendix.

## 6. Hypotheses (scored pass or fail per rung in `findings/rca-latent.md`)

- **H-rca-random.** `rca/baro`'s pooled Avg@5 at every rung lies above the analytic random
  expectation, with the lower end of its 95 % interval above it. *Fail:* any rung where the
  interval reaches the expectation.
- **H-rca-rule.** `rca/baro`'s pooled Avg@5 is at or above `control/rule`'s at every rung.
  *Fail:* any rung where the rule control's Avg@5 exceeds BARO's.
- **H-rca-bocpd.** The full pipeline's pooled Avg@5 is within 0.10 of the injection-time row's
  at every rung, with its no-change-point count reported. *Fail:* a gap beyond 0.10 either way.
- **H-rca-scale.** `rca/baro`'s Avg@5 does not rise from xs to m (more candidates, same
  evidence per candidate). *Fail:* m above xs by more than the wider interval's half-width.
- **H-graph-topology.** CausalRCA's directed service-graph F1 at the frozen τ is at or below
  `floor/topology`'s on every rung. *Fail:* any rung where the learned graph beats the
  deployment topology.

## 7. Deviations (RUN.md rows, numbered before the first run)

- **D-TBR-1** — the benchmark pin: the reference harness pins `a3eb104` (`v0.3.0-score3`); this
  directory needs the publication-branch commit that carries `tracebench.export_rca` and moves
  its pin there when the owner merges it; scoring (`score_rca`, `score_service`) is unchanged
  between the two (both are additive to the token scorer).
- **D-TBR-2** — the standard suite's `main.py` loading path is not run (§2); the adapter calls
  the suite's method function on the exporter's table after the loader's own fill rule.
- **D-TBR-3** — positional case directories (`case-<k>`) in place of the suite's
  `<service>_<fault>_<rep>` names, whose service part is the root cause; the method receives the
  table, the two window fields and the service universe, and nothing that names the answer.

## 8. Provenance (not results)

- 2026-10-14, the xs seed-0 prototype (session scratch): a RobustScorer-equivalent at the
  labelled injection time ranked the root cause 2/5 and 2/5 at 10 s bins, 1/5 (degrade) and 4/5
  (crash) at 30 s — the reason the width is fixed by rule before any test read. The exporter's
  table reproduces the 30 s ranks through the loader shape (checked 2026-10-14 with a numpy
  RobustScaler; the pinned `baro()` runs in the `trace-bench-rca` env at B4).
- 2026-10-14, the bin-traffic job (c7i.4xlarge, 973 s, xs/s/m × seeds 0–4, widths 5–300 s) — the
  per-service traffic tables behind §3; the exporter's 30 s series on xs seed 0 equal that job's
  (counts and 5xx counts exactly; quantiles within 1.3e-6, the job having binned `request_time`
  in float32).
- 2026-10-14, Luan Pham's reply: both rows confirmed, the naming confirmed, assistance offered.

## 9. Open questions for the owner

- **Q5** — one staging ledger shared with the reference harness (as coded) or a separate file?
- **Q7** — a GPU box for CausalRCA at m if local runs exceed ~30 minutes per case.
- **Q8** — the `control/rule` registration text (`plans/controls.md` §2).
- **Q16** — the validation split of CausalRCA's graph-axis threshold (§4 proposal).
- **Q17** — CausalRCA epochs: the repository's 500 (run as shipped) or the mail's 1000.

## Addenda

(none)

## Addenda

### 2026-10-19 — D-TBR-1 pin moved (owner merged trace-bench PR #12)

- **§7 D-TBR-1.** The owner merged `export-rca-26-10-15` into `clear-2027` on 2026-10-19 (merge
  commit `ace384b4`). `pyproject.toml` and `requirements.txt` now pin `ace384b4` in place of
  `a3eb104`; the pinned tree differs from `a3eb104` only by `tracebench.export_rca` and its
  tests (`score.py`, `score_rca`, `score_service` unchanged), so no landed cell moves. Whether
  `ace384b4` receives a score-side tag is the owner's call (publication plan Q10 family); the
  directory pins the commit either way. Nothing in `benchmarks/trace-bench/` changes.

### 2026-10-21 — B4 adapters, the method environment, and a second no-ranking case of the full pipeline (before any test read is landed)

Written after the xs seed-0 smoke (§4: "the xs seed-0 export is their smoke test; no value is
chosen on it") and before any read of any other corpus.

- **Method environment (§4).** `trace-bench-rca` is created from `environment.yml` with
  **Python 3.12** in place of 3.11: at the RCAEval pin, `RCAEval/e2e/__init__.py` imports its
  method modules leniently (a module whose dependencies are absent is skipped) only on Python
  3.10, 3.12 and 3.14; on 3.11 it imports `rcd`, `mmrcd` and `torai` unconditionally, whose
  dependencies are not installed, so `RCAEval.e2e.baro` cannot be imported there. 3.12 is the
  version RCAEval's README names as its default environment. RCAEval's optional dependency sets
  (`default`, `rcd`, `eventadl`) are not installed; its base install declares none, and the
  `baro` path needs only numpy, pandas, scikit-learn, requests and tqdm, which the `baro` extra
  brings. Versions resolved on 2026-10-21 and named by every record's `pip freeze`: `fse-baro`
  at `e35f4ec`, `RCAEval` 1.8.0 at `259ea41`, pandas 3.0.6, scikit-learn 1.9.1, numpy 2.5.4,
  scipy 1.18.1. The package itself runs from its source tree (`PYTHONPATH=src`), not as an
  editable install, so no local path enters a record's `pip freeze`.
- **Commands.** `tracebenchrca.rank` (method side; `--arm rca/baro | rca/baro-bocpd`) reads an
  export through the case door and writes `ranking-positional.json`, keyed by case directory;
  `tracebenchrca.score` (score side) restores the case ids by `tracebench.export_rca.case_dirs`,
  writes `ranking.json` and `score-rca.json` (`tracebench.score_rca`, B = 2000, seed 0) and, for
  the full pipeline, each case's change-point offset; `tracebenchrca.report` pools per rung and
  across rungs. `control/random` is the `random` block of every `score-rca.json` (analytic, §1);
  it has no stage. `control/rule` is not run until the owner confirms `plans/controls.md` §2 (Q8).
- **Staging.** The `rank` stage at s and m names the previous rung's `rank` record of the same
  arm and seed (record form); landed records enter the shared ledger as stage `rank` with the arm.
- **`rca/baro`, `time` column.** The rank of the `time` column in the returned metric ranking is
  recorded per case (on the smoke: 13th and 12th of 21); it is removed before the collapse, as
  registered.
- **`rca/baro-bocpd`, change point after the last row (new; the counterpart of §2's empty list).**
  `online_changepoint_detection` returns a run-length argmax array one entry longer than the
  table, so `find_cps` can return the index `n_rows`. `robust_scorer(data, anomalies=cps)` then
  splits off zero fault rows (`data.tail(0)`) and `RobustScaler.transform` raises `ValueError`.
  **Registered: a case whose first change point is at or after the table's last row is counted
  and scored as a miss with its own reason ("change point at the table's end"), exactly as a case
  without a change point; the adapter never moves to a later or earlier change point and never
  falls back to the injection time.** Any other exception of the pinned code fails the run.
  Disclosure: the smoke surfaced it — xs seed 0, case 2 of 2 (the crash case), whose only change
  point is row 40 of 40. Under the registered code no ranking exists for that case under any
  rule, so no alternative was scored or compared; `rca/baro`'s two smoke ranks (1/5, 4/5) were
  seen and are the values the 2026-10-14 prototype gave.
- **Change-point offset (§5).** Score side: the first change point's bin start (UNIX seconds, the
  export's clock) minus the labelled injection (`tracebench.export_rca.window_epoch` of the
  corpus configuration plus `inject_time_s`); positive = detected late. On the smoke: +450 s (15
  bins) on the degrade case.
- **`rca/baro-bocpd`, the scorer's own preprocessing (code fact read at B4; not a deviation).**
  The full pipeline's `baro.root_cause_analysis.robust_scorer` runs BARO's own `preprocess`
  (`root_cause_analysis.py:96-118`), which applies `drop_constant(drop_time(...))` for every
  `dataset`, unlike RCAEval's `preprocess(dataset=None)` on the `rca/baro` path (§2, identity). So
  on this row the `time` column is never scored, and a metric constant **within either segment**
  (before or after the first change point) is dropped from the intersection; a fault segment of one
  row leaves no metric, and the pinned function returns an empty ranking. The adapter passes that
  output through unchanged; the benchmark's scorer counts a case whose ranking lacks the root cause
  as a miss; the record counts these cases (`n_empty_rankings`). Found after the xs rung's five
  seeds had been read in an uncommitted dry run of these adapters (xs seed 4, the crash case:
  change point at row 39 of 40); that dry run is discarded and every read is re-run from the
  committed code. This bullet changes no rule — the empty output was already passed through and
  scored as a miss — and adds only the count.
- **`rca/baro-bocpd`, the detector raises `LinAlgError` (new; the third no-ranking case of the full
  pipeline).** `MultivariateT.pdf` (`baro/_bocpd.py:101`) calls `scipy.stats.multivariate_t.pdf`,
  whose `_PSD` runs `scipy.linalg.eigh` with its default driver (`dsyevr`). At m (≈ 156 selected
  latency / error columns) the predictive's shape matrix is finite and positive definite (smallest
  eigenvalue exactly 1 / (df − 1)) but carries large clusters of identical eigenvalues, on which
  `dsyevr` returns its internal-error code and scipy raises `LinAlgError: Internal Error.`. The
  same matrix passes with the `evd`, `ev` and `evx` drivers and `numpy.linalg.eigh`, and fails
  identically with the conda-forge OpenBLAS build of scipy 1.18.1 and the pip wheel (Apple
  Accelerate), so it is not one machine's library. **Registered: a case on which the pinned
  detector raises `numpy.linalg.LinAlgError` is counted and scored as a miss with its own reason
  ("detector raised LinAlgError"); no other exception is caught; the pinned code is not patched
  (choosing another driver would modify it) and nothing falls back to the injection time.**
  Disclosure: found in the uncommitted dry run at m seed 0 (3 of 5 cases), after the xs and s rungs
  had been read in that dry run; every read is re-run from the committed code. Whether to tell the
  method's author is the owner's call.

### 2026-10-22 — owner decisions on CausalRCA (publication plan Q7, Q16, Q17; before any CausalRCA code or run)

- **Q16 — the graph axis's validation split, with a four-seed exception.** The proposal of §4 is
  registered: on every rung the **seed-0 corpus is the validation split of the graph axis** — the
  edge threshold τ is chosen on the seed-0 cases' learned service graphs (argmax of directed F1
  against `graphs/views/service.json` at the view's default floor; ties → the larger τ) and frozen in
  a dated freeze file per rung before any test read — and **seeds 1–4 are its test read**.
  **Four-seed exception:** every graph-axis number of `rca/causalrca` (per rung and pooled) is
  computed on the four test seeds 1–4, an exception to the five-seed rule that governs every other
  table of this directory and of the reference harness; each such table and sentence says "four
  test seeds (1–4); seed 0 is the graph axis's validation split". `floor/topology` is set beside it
  on the same four seeds, so that H-graph-topology compares paired seeds. The ranking axis has no
  threshold and keeps all five seeds, seed 0 included. The `report` stage's five-seed refusal
  stays for the ranking axis; the graph axis gets its own four-seed check. The τ grid is registered
  by addendum before the first validation read.
- **Q17 — 1000 epochs.** CausalRCA runs with **1000 epochs**, the value of the test plan sent to
  the authors on 2026-10-14; the repository's `config.py` sets `CONFIG.epochs = 500` and has no
  command-line flag for it. Recorded as **D-TBR-4** (follows the sent test plan; deviates from the
  repository default by this one value). How the value is set, preferably without editing the
  pinned files, is registered with the adapter at B5; every record names it.
- **Q7 — compute.** A GPU box for CausalRCA at m is approved (the owner applies the replica). The
  registered rungs of this directory remain xs, s and m (`constants.RUNGS`; the exporter's width
  table has no l / xl entry and refuses them; the test plans sent to both author teams name the
  three smallest rungs). The owner asks for a sizing of l and xl from the m timings, as a proposal
  for approval: nothing runs at l or xl before a dated addendum that registers (a) their bin widths
  by the N = 10 rule of §3 from a traffic measurement on l and xl, (b) the exporter change on the
  benchmark's publication branch and the pin move, and (c) the compute, and before the owner
  approves it.

### 2026-10-22, later — CausalRCA's code path: the authors' script through a recorded data shim (publication plan Q28, owner decision; D-TBR-5; before any CausalRCA code or run)

Branch note: registered on top of the 2026-10-22 decisions above while their pull request was
still open.

- **Decision.** `rca/causalrca` runs the authors' `train_latency.py` at `21f4eb0` with its
  **data block substituted and every other line unchanged**. The data block is exactly the
  statements from `names = ['front-end', 'user', 'catalogue', 'orders', 'carts', 'payment',
  'shipping']` to `data = all_data[name]`: the fixed sock-shop service list, the metric list,
  `idx` and `atype` from the command line, the pickle open and the `<name>_ctn_latency`
  selection — the file's only sock-shop-specific lines. The replacement sets `names` to the
  case's services (`services.json` order), `idx = 0` (read only by the final `print`), `data` to
  the case's input table (below) and `CONFIG.epochs = 1000` (D-TBR-4). The pinned file is never
  edited on disk: the adapter reads it, asserts that each end of the block occurs exactly once,
  substitutes the block in memory, writes the substituted source and the removed block beside the
  training's record, and executes the substituted source. RUN.md **D-TBR-5**. The sent test plan
  said "run unmodified"; the paper says "the authors' script with its data-loading block replaced
  by the benchmark's series, training code unchanged".
- **Two corrections to §2** (read 2026-10-22 at the pins).
  - §2 says RCAEval's `e2e/causalrca.py` "is the same procedure". It is not: the script runs
    PageRank on `|adj.T|` (the learned edges reversed), the port on `|graph|` after a double
    transpose; called with `dataset = None` the port neither cuts the post-injection window nor
    drops `time`; and it scales by the column maximum. The port is not run.
  - §2 says the script "builds the DAG-GNN on `data / data.max()`". That is the port. The script
    trains on the pickled columns as stored; the authors' collection notebook
    (`data_collection_all_services.ipynb` at the pin) min-max scales every column with
    `sklearn.preprocessing.MinMaxScaler`, drops `time`, drops duplicate columns and pickles the
    result.
- **Input (the substituted block's `data`).** Per case and series (`latency-50`, `latency-95`):
  the exported table read through `tracebenchrca.cases.read_table` (the loader's fill rule,
  D-TBR-2); the rows with `time ≥ inject_time` (the export ends at `end_time`, so these are the
  labelled fault window: 10–20 rows at xs, 60–120 at s, 15–40 at m); the `<service>_<series>`
  columns in `services.json` order; each column min-max scaled to [0, 1] as the authors' notebook
  does (a constant column becomes 0, `MinMaxScaler`'s convention). Not chosen: the notebook's
  duplicate-column drop, which would remove services from the graph and from the ranking — here
  every deployed service is a node and is ranked; the port's `data / data.max()`; unscaled series.
- **Command line.** `--gamma 0.25 --eta 10`: the script's defaults and the sent plan's values.
  The repository's `test_latency_per.sh` passes `--gamma 0.5`; recorded, not used.
  PageRank is the script's own `sknetwork.ranking.PageRank()`, whose damping factor at 0.24.0 is
  0.85 by default, the sent plan's α.
- **Repetitions and seeds.** Ten trainings per case and series (the shell loop's count, and the
  sent plan's "averaged over 10 runs"). The script never seeds (its `torch.manual_seed` calls are
  commented out); before executing the substituted source the adapter seeds `torch` and `numpy`
  with the run index `r` ∈ {0, …, 9}, so a training is repeatable on the same platform. The seed
  rides the record.
- **Working directory.** Each training runs in a directory of its own, because the script writes
  fixed file names into its working directory. The adapter reads two of the script's outputs and
  keeps the third as written:
  - the ranking, the script's last standard-output line: `(index, score)` pairs sorted by the
    PageRank score, descending, equal scores in index order (a stable sort), mapped to services
    through `names`; the adapter asserts every service appears exactly once;
  - the learned graph, `cpu-hog_front-end_adj.pkl` (the script's own file name): the learned
    adjacency after the script's own cut (`|w| < 0.3` set to 0). `adj[i, j] ≠ 0` reads as the
    service edge `names[i] → names[j]` (DAG-GNN's linear SEM `X = AᵀX + Z`; `modules.py`'s
    encoder applies `I − Aᵀ`), so PageRank on `|adj.T|` walks against the learned edges, toward
    causes;
  - `metrics_causality.png`, kept.
- **Aggregation — the authors' own: per-run metrics averaged.** The authors' results notebook
  (`Res-Plot.ipynb` at the pin) lists one Avg@5 per run for each fault case and reports their
  mean. So on the ranking axis AC@1, AC@3, AC@5 and Avg@5 are computed on each run's ranking and
  averaged over the ten runs of the case; the bootstrap over cases (B = 2000, seed 0) resamples
  these run-averaged per-case values. On the graph axis the sent plan's "one graph per fault case"
  is the elementwise mean of `|adj|` over the ten runs, scored by `tracebench.score_service` with
  the mean absolute weight as the edge score.
- **Graph-axis τ grid** (the Q16 addendum above requires it before the first validation read).
  Strict cut `score > τ` on the mean absolute weight over
  `0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 5` (15 values). τ = 0 is every
  edge learned in any run; a single run's surviving weight is at least 0.3, so a mean below 0.03
  cannot occur. The freeze rule is Q16's (argmax of directed F1 at the service view's default
  floor on the seed-0 cases, ties → the larger τ, frozen per rung and per series before any test
  read). A τ at either end is flagged `at_grid_edge` and recorded, not refused.
- **Environment.** The repository's `requirements.txt` exactly, in a conda environment of its own
  (`causalrca-21f4eb0`, Python 3.9, the one version all ten pins share), never `trace-bench-rca`;
  the script calls `networkx.from_numpy_matrix`, which networkx 3 removed. On macOS arm64 the
  environment is x86_64 under Rosetta 2: scikit-network 0.24.0 publishes macOS wheels for x86_64
  only, and its source build pins numpy 1.20.0, which does not build on arm64. On the m GPU host
  it is the Linux x86_64 wheels, torch 1.10.2 with CUDA 11.3. Premise of 2026-10-22, before this
  addendum (not a read, nothing kept): the pins install and import in the script's own import
  order (`import torch` alone faults under Rosetta, numpy first does not, and the script imports
  numpy first), and one substituted training on an xs window of 10 rows × 5 services at 1000
  epochs ran in 27 s at a 0.19 GB peak footprint and printed a ranking of all five services.
- **What each record names.** The pin and the sha256 of the pinned `train_latency.py`, of the
  substituted source and of the removed block; the environment's `pip freeze` and platform; per
  training the case, series, run index, seed, wall clock and the sha256 of each output read.

### 2026-10-23 — CausalRCA's adapter: what D-TBR-5 leaves to the code (before any CausalRCA code or read)

Fixed before the adapter is written and before any training of a registered read:

- **Two rows.** `rca/causalrca` is read once per series, `latency-50` and `latency-95`, and each is
  its own row on both axes (a table row and a graph-axis freeze per rung and series). Records land
  under `results/<rung>/latent/seed=<k>/rca-causalrca/<series>/` (`rank/` the method side, the score
  files beside it).
- **The pinned source.** `rank --arm rca/causalrca --causalrca-src <dir>` refuses a directory whose
  git commit is not `21f4eb0` or whose tree is not clean, and refuses unless the sha256 of
  `train_latency.py`, `modules.py`, `utils.py` and `config.py` equal the values read at the pin on
  2026-10-23 (`6e86acfe…`, `5e1237aa…`, `c394ed6f…`, `451459d9…`; full values in
  `constants.CAUSALRCA_FILES_SHA256`).
- **The hand-off.** Per training, the method side (the `trace-bench-rca` environment) writes into the
  training's own directory the input array `input.npy` (float64, rows × services, the D-TBR-5 input)
  and the substituted source, whose data block reads `data = pd.DataFrame(np.load('input.npy'),
  columns=[...])`. A standalone runner, `tracebenchrca/causalrca_runner.py`, executes it in the
  pinned environment: it imports numpy before torch, seeds `torch` and `numpy` with the run index,
  executes the source with the registered command line, and, after the script has exited, converts
  the script's pickled graph to `adj.npy` in the same environment (numpy 1.20 wrote it). It writes
  nothing to standard output of its own, so the last line is the script's.
- **Threads and parallelism.** On CPU each training runs single-threaded (`OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` = 1, recorded), and a stage runs its trainings in parallel processes
  (`--workers`). On a GPU host the script's own `CONFIG.cuda = torch.cuda.is_available()` holds.
- **A failed run.** A training that exits non-zero, or whose last standard-output line is not a list
  of `(index, score)` pairs covering every service exactly once (a NaN score, a crash), is a failed
  run: on the ranking axis it is a run with no hit (counted per case, never refilled from another
  run); on the graph axis it is left out of the case's mean and counted.
- **The ranking axis's numbers.** Per case, AC@k for k = 1 … 5 is the share of the ten runs whose
  ranking places the root cause in the first k (the root cause found through the benchmark's
  `case_ranks`); AC@1 / 3 / 5 and Avg@5 are the means over cases; the 95 % interval resamples the
  cases with the benchmark's own draws (`numpy.random.default_rng(0).integers(0, n, (2000, n))`) and
  averages the resampled per-case values, so it equals `tracebench.score_rca.bootstrap` when every
  run of a case agrees. Pooling over seeds and rungs concatenates the per-case values.
- **The graph axis's numbers.** A case's graph is the elementwise mean of `|adj|` over its
  successful runs, nodes mapped by service name to the benchmark's `svc:<index>` keys
  (`tracebench.score_service.deployed_services`); every pair with a positive mean is listed with the
  mean as its score (the full ranking), and the prediction at τ keeps the pairs whose mean exceeds τ.
  Each case is scored by `tracebench.score_service` at the view's default floor; a rung's number is
  the mean over its cases of the per-case directed precision, recall, F1 and AUROC (each case is
  one learned graph), the per-case values recorded. `floor/topology`'s number is the same mean over
  the same cases (its prediction is the corpus's deployment topology, one per corpus). The freeze
  is the argmax over the τ grid of the mean directed F1 over the seed-0 cases, ties → the larger τ.

### 2026-10-25 — stretch arms `rca/circa` and `rca/circa-topology` (publication plan B7; owner decisions 2026-10-25; before any CIRCA code or read)

Owner decisions of 2026-10-25 (stretch plans §2 and the follow-up questions): CIRCA runs as the
standard suite's port, **verbatim and disclosed**; a second cell builds CIRCA's own structural graph
from the shipped deployment topology, **now**, beside the port. Both are stretch arms of the paper
(appendix A5); the required set is unchanged.

- **Arms.** `rca/circa` — class `root-cause`, inputs `series, window, services`, axis ranking:
  **CIRCA (PC graph, suite port)**, `RCAEval/e2e/circa.py::circa(data, inject_time, dataset=None)` at
  `259ea41` (MIT), a port of `NetManAIOps/CIRCA` (BSD-3-Clause; the repository itself is not run).
  `rca/circa-topology` — class `root-cause`, inputs `series, window, services, topology`, axis
  ranking: **CIRCA (structural graph)**, the same hypothesis-testing head
  (`RCAEval/graph_heads/rht.py::rht`) on the structural graph of the paper's Algorithm 1 built from the
  deployment call topology. The topology reaches the method through a fourth case file,
  `topology.json`, written by the exporter (`tracebench.export_rca --extras topology`, D-TBR-9 below);
  `tracebenchrca.cases.open_case_file` stays the one door. It is a declared **topology-informed** arm
  and is reported beside the floors as a topology-reading discovery arm would be.
- **Facts read at the pin (2026-10-25).** `circa()`: `preprocess(dataset=None)` is the identity; the
  `time` column is dropped for graph construction and re-added for the head (so, unlike `rca/baro`, the
  clock is never a scored variable); `pc_default(pc_input, dataset="ob")` calls causal-learn's `pc()`
  with its defaults (Fisher-z, α 0.05, `stable=True`, `uc_rule 0`, `uc_priority 2`) and no background
  knowledge; `rht(adj, inject_time, data)` builds `Node(service, metric)` from each column split at
  its first `_`, adds the PC edges, `reverse()`s the graph, and scores every node with `RHTScorer()`
  (`tau_max 0`, linear regression on the node's parents; `ANMRegressor`) — the descendant adjustment
  is not applied. Its data window is **hard-coded for 1 s metrics**: `CaseData(interval = 1 s,
  lookup_window = 120, detect_window = 10)` and `current = inject_time + 300`: the loader keeps the
  table's points in `[inject − 120 s, inject + 300 s]`, resamples them to 1 s (`resample().mean()`,
  time interpolation in both directions, back-fill) → 421 points, of which the first 111 are the
  reference (training) window and the last 10 the test window. On this directory's bins that is
  **4 / 8 / 2 pre-fault bins (xs 30 s / s 15 s / m 60 s), interpolated**, tested on the 10 s ending
  five minutes after the injection. `sli = np.random.choice(nodes)` is drawn but unused by the head
  (every node is scored), so the row is deterministic given PC (stable PC is deterministic). The
  port's `@rca` wrapper catches any exception, prints one line to standard error and returns the
  **unranked column order** as the ranking with `adj = []`.
- **D-TBR-8 — the port's fallback is a miss.** A case on which `circa()` returns its exception
  fallback (`adj` is an empty list) is counted and scored as a miss, reason "port fallback", with the
  wrapper's standard-error line kept in the record; the column order is never scored. Any other
  exception fails the run.
- **Registered reading.** The arm is labelled "CIRCA (PC graph, suite port)"; the paper states the
  head's window facts above beside the result (it is what the suite runs on every dataset; on 1 s
  metrics the reference window holds 111 samples, here 2–8 bins). No value is chosen on any read;
  defaults as the port's; the xs seed-0 export is the smoke test.
- **D-TBR-9 — the structural graph from the shipped topology (`rca/circa-topology`).** Nodes = every
  metric column. Meta metrics (Algorithm 1, four golden signals): Traffic = `count`; Latency =
  `latency-50` and `latency-95`; Errors = `error`; Saturation empty (the benchmark ships no resource
  metrics; an empty meta metric passes its parents' metrics through). Within a service: Traffic →
  each Latency, Traffic → Errors. Across each call edge of `topology.json` (caller → callee, the
  deployed-service coarsening of `instantiation.json`'s `topology.edges`, the client excluded,
  within-service calls dropped, one edge per ordered pair): caller Traffic → callee Traffic; callee
  Latency (both quantiles) → caller Latency (both); callee Errors → caller Errors. The adjacency is
  handed to the port's own `rht()` in the encoding **the port reads**: for a cause column `c` and an
  effect column `e`, `adj[c, e] = 1` and `adj[e, c] = −1`, which after the port's edge construction
  and `reverse()` makes `MemoryGraph.parents(e) ∋ c` (asserted by a test that observes
  `graph.parents` inside the port's own scoring). No PC runs on this cell; nothing else differs from
  `rca/circa`. The record carries the structural graph's edge count per case.
- **Hypotheses (scored mechanically in `report`).** H-rca-circa-random and H-rca-circa-rule: the
  H-rca-random / H-rca-rule rules of §6 applied to `rca/circa`. **H-circa-topology-gain:**
  `rca/circa-topology`'s pooled Avg@5 is at or above `rca/circa`'s on every rung (the paper's claim
  that the structural graph carries the method). *Fail:* any rung where the PC cell is above.
- **Environment.** The port needs causal-learn and networkx, absent from `trace-bench-rca` until
  now: `pyproject.toml` gains the extra `circa = ["causal-learn>=0.1.4", "networkx>=3"]`; resolved
  2026-10-25 as causal-learn 0.1.4.8, networkx 3.7 (pure-Python, numpy 2.5.4 / pandas 3.0.6
  unchanged); every record's `pip freeze` names them. RCAEval's `default` extra is not installed.
- **Premise checks (2026-10-25, xs seed-0 and m seed-0 exports of 2026-10-21; nothing kept).** PC
  on the 20-column xs table: 0.1 s; on the 208-column, 246-row m case 0: 90 s (376 undirected
  edges); the head 0.2 s; every service ranked on every case tried; no fallback; the head saw 421
  points, 111 training, 10 test, as read from the code. Expected cost: ≈ 40 min for the 25 m cases,
  local.
- **Staging, protocol, reporting.** As §4 and §5: `rank` per arm, xs → s → m by
  `--prior-rung-record`, `tracebench.score_rca` (B = 2000, seed 0), five seeds or nothing, the
  random expectation and `control/rule` beside every row; per case the record keeps `n_edges`
  (PC or structural), `n_services_unranked`, the fallback flag and the loader's filled cells.

### 2026-10-25, later — stretch arm `rca/rcd` (publication plan B7; owner decisions 2026-10-25; before any RCD code or read)

Owner decisions of 2026-10-25: RCD goes, in the suite's own RCD environment, with **ten seeded runs
per case averaged**. A stretch arm (appendix A5).

- **Arm.** `rca/rcd` — class `root-cause`, inputs `series, window, services`, axis ranking: **RCD**,
  `RCAEval/e2e/rcd.py::rcd` at `259ea41` (MIT), which carries the authors' code (`azamikram/rcd`,
  MIT) and runs on the suite's patched causal-learn. The failure as a soft intervention on an unknown
  target: an F-node (0 = the rows before the labelled injection, 1 = the rows from it to the labelled
  end) is appended, every metric is discretised (`KBinsDiscretizer`, 5 bins, ordinal, k-means, fitted
  on the concatenated table), and the hierarchical, localised Ψ-PC (χ² tests involving the F-node
  only; random chunks of γ = 5 variables; α from 0.001 in steps of 0.1 until a neighbour survives;
  two phases) returns the F-node's neighbours in order — a **partial ranking of metrics**.
- **Facts read at the pin (2026-10-25).** `rcd(data, inject_time, dk_select_useful=False, gamma=5,
  localized=True, bins=5, dataset=None, seed=None)` splits on `time < inject_time`; with
  `dataset=None` it hands the table — `time` included — straight to Ψ-PC (the `rca/baro` trap); with
  any other dataset name but `"sock-shop"` it runs the suite's named-dataset preprocessing on each
  segment: `drop_time` → `convert_mem_mb` (a no-op here: no `_mem` column) → `drop_constant`, then
  `_match_columns` (the columns present in both segments). `create_chunks` permutes the columns with
  `np.random.permutation`, so the output is stochastic; `seed` seeds numpy once. Phase 2 can return
  fewer metrics than services. The method module imports causal-learn, networkx, scikit-learn,
  matplotlib, pandas, numpy and `RCAEval.io.time_series`; the suite's `SETUP.md` runs RCD only in its
  *RCD mode*: Python 3.8, `requirements_rcd.lock` (causal-learn 0.1.2.3, numpy 1.23.5, pandas
  2.0.1, networkx 2.5, scikit-learn 1.2.2, scipy 1.10.1), then `script/link.sh` overlays four patched
  causal-learn files from the suite's `lib/causallearn/` (`utils/PCUtils/SkeletonDiscovery.py`, which
  adds `local_skeleton_discovery` and the per-test `p_values`; `utils/Fas.py`;
  `search/ConstraintBased/FCI.py`; `graph/GraphClass.py`). Upstream causal-learn has no
  `local_skeleton_discovery`, so RCD cannot run in `trace-bench-rca`.
- **D-TBR-6 — the suite's named-dataset preprocessing.** The adapter calls `rcd(table, inject_time,
  dataset="trace-bench", seed=r)` with the port's defaults for everything else: the dataset name
  selects the suite's own named-dataset branch (time dropped, constant columns dropped per segment,
  columns matched between segments), the code the suite runs on every dataset it publishes RCD
  numbers for; nothing is re-implemented. Follows: the suite's named-dataset path. Deviates from: the
  unnamed path, which would make `time` a PC variable. The record counts the columns the branch
  removed per case (table columns minus the ranked universe is not observable from the output; the
  adapter recomputes the branch's column set with the suite's own functions for the count only).
- **D-TBR-7 — the RCD environment and runner.** A second pinned conda environment,
  `rcaeval-rcd-259ea41` (`environment-rcd.yml`): **Python 3.10** (not the suite's 3.8: on 3.8 the
  suite's `e2e/__init__.py` imports every method module unconditionally and therefore every
  method's stack; on 3.10 it imports leniently, as on 3.12), the lock's versions of exactly the
  packages RCD imports — causal-learn 0.1.2.3, numpy 1.23.5, pandas 2.0.1, networkx 2.5,
  scikit-learn 1.2.2, scipy 1.10.1, matplotlib 3.7.1, tqdm 4.65.0, requests 2.30.0 — RCAEval at the
  pin installed without extras, and the four patched files copied over the installed causal-learn
  exactly as `script/link.sh` links them (the sha256 of each overlaid file, and of the file it
  replaced, ride the record). The method side (`tracebenchrca.rankrcd`, in `trace-bench-rca`) writes
  each case's filled table (`read_table`, D-TBR-2) as `input.csv` with the window's `inject_time`
  into a run directory and executes `tracebenchrca/rcd_runner.py` in the pinned environment once
  per run (`--seed r`); the runner imports nothing from this package, calls the pinned `rcd()` as in
  D-TBR-6 and writes the returned metric ranking to `ranking.json`. Created 2026-10-25; the
  environment's `pip freeze` rides every record.
- **Repetitions, partial rankings, failures.** Ten runs per case, `seed = r ∈ {0, …, 9}` (the paper
  reports averages over runs; the CausalRCA aggregation of the 2026-10-23 addendum applies: per
  case, AC@k is the share of the ten runs whose service ranking places the root cause in the first
  k, the rung's numbers are means over cases, the interval resamples cases with the benchmark's
  draws). A run's metric ranking is collapsed to services by first occurrence (the suite's coarse
  collapse); **services absent from a run's ranking are misses at every k for that run**
  (`n_services_unranked` per run recorded) — the BOCPD precedent, never refilled. A run that raises
  is a failed run with no hit (counted per case with its reason; never re-run with another seed).
  Defaults: `gamma 5, localized True, bins 5, dk_select_useful False`; no sweep (ranking-only arm;
  the xs seed-0 export is the smoke test, no value chosen on it).
- **Hypotheses (mechanical in `report`).** H-rca-rcd-random and H-rca-rcd-rule: the H-rca-random /
  H-rca-rule rules of §6 applied to `rca/rcd`.
- **Premise checks (2026-10-25, the 2026-10-21 exports; nothing kept).** The environment above
  imports `RCAEval.e2e.rcd` with `local_skeleton_discovery` present. xs seed 0 through the
  named-dataset path: no constant column in either segment on either case; the same seed gives the
  same ranking twice (seeds 0 and 1, both cases); different seeds give different orders; the port
  ranked 1 of 5 services on case 0 and 3 of 5 on case 1 (partial rankings are the norm); 0.1 s per
  run. s seed 0 case 0 (1,080 rows × 48 columns): 3.0 s, 8 of 12 services ranked. m seed 0 case 0
  (246 rows × 208 columns): 3.0 s, 4 of 52 services ranked. Expected cost: ≈ 25 cases × 10 runs ×
  3 s ≈ 13 min per rung at m, local, single process.
- **Staging, protocol, reporting.** As §4, §5 and the 2026-10-23 addendum's run-averaged scoring
  (the score side generalises `scorecausalrca`'s per-run averaging to any rank record that carries
  runs; CausalRCA's landed numbers must stay byte-identical); the random expectation and
  `control/rule` beside every row; per case the record keeps every run's ranked-service count, the
  columns the preprocessing removed, the loader's filled cells, and the runner's wall clock.
