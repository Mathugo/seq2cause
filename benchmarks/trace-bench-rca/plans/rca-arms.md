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
- **Correction (2026-10-25, found by the adapter's test before any read).** "Every service is ranked
  (the head scores every node)" above is wrong for the PC cell: the port builds its graph by adding
  edges, so a metric with no PC edge is not a node and is never scored; such metrics are absent from
  the ranking and their services, if none of their metrics is ranked, are **misses at every k**,
  counted per case as `n_services_unranked` (the BOCPD precedent; nothing is appended). A second
  source of unranked metrics applies to **both** cells: the head's `CaseData(prune=True)` drops every
  node whose resampled series inside its window (`[inject − 120 s, inject + 300 s]`) takes a single
  value, so a service all of whose metrics are constant there (a quiet service: count 0, error 0,
  empty latency cells filled to 0) is unranked — a miss, counted the same way. On the 2026-10-21
  exports no case has a constant column over its whole table, and every service was ranked on the
  cases tried (premise above); inside the head's window the constant columns number 0.4 / 0.6 / 6.1
  of 20 / 48 / 208 per case on average (xs / s / m), so a few m services may be unranked. The rule
  stands for any case where it bites; the structural-graph cell gives every metric at least one
  edge, so only the pruning can leave a service unranked there.

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

### 2026-10-25, later (2) — stretch arm `rca/tracerca` (publication plan B7; owner decisions 2026-10-25; before any TraceRCA code or read)

Owner decisions of 2026-10-25: TraceRCA's spans come from the **views** (the only shipped source that
carries trace membership); the code is the standard suite's **port**, not the authors' unlicensed
repository. A stretch arm (appendix A5), declared **call-tree-informed**: it reads the reconstructed
traces, so the correlator's measured loss (paper §3) is part of its input.

- **Arm.** `rca/tracerca` — class `root-cause`, inputs `spans, window, services`, axis ranking:
  **TraceRCA (suite implementation)**, `RCAEval/e2e/tracerca.py::tracerca(data, inject_time)` at
  `259ea41` (MIT; the file states it was refactored from the MicroRank repository), a
  reimplementation of `tracerca-li-2021` — the authors' repository `NetManAIOps/TraceRCA` carries no
  licence file and is cited, not run (**D-TBR-10**; "author repository, no licence file; authors
  contacted <date>" in the paper, with the port named as what ran).
- **Facts read at the pin (2026-10-25).** The port takes a span table with `serviceName`,
  `methodName` (filled from `operationName` where absent), `startTime` and `duration`; it forms
  `operation = serviceName_methodName`; splits spans at `startTime + duration < inject_time` (the
  same unit as the table — microseconds: the SLO divides `duration` by 1,000); per operation the
  pre-fault mean and standard deviation of `duration / 1000` define its threshold; a post-fault span
  is abnormal if its duration reaches mean + 3 SD (an operation with one pre-fault span has an
  undefined SD and is never abnormal); per operation support = its abnormal spans over all abnormal
  spans and confidence = its abnormal spans over its post-fault spans; score `2sc / (s + c)`;
  operations ranked by score, undefined scores (no abnormal span) dropped. It has no trace-level
  anomaly-detection stage and counts spans, not traces. **It raises `KeyError`** when a post-fault
  span names an operation absent from the pre-fault spans (`normal_slo[op]`). It mutates its input
  (the adapter passes a copy) and ignores `dataset`.
- **Input (the exporter's `spans.csv`, `tracebench.export_rca --extras spans`, a fifth case file
  through the one door).** One row per span of every request trace of the **end-ordered request
  view** whose last span ends inside the case window `[window start, end_time)` — every split's rows
  (the root-cause read is not a test-split read on the discovery axis) — `trace_id, service,
  operation, start_us, duration_us, parent_pos` (microseconds from the view's `start_ns + offsets`
  and `durations`; service and operation from the view's `model-vocab.json` base ops). The adapter
  renames the columns to the port's (`serviceName`, `methodName`, `operationName`, `startTime`,
  `duration`), passes `inject_time × 10⁶`, and collapses the returned `serviceName_methodName`
  strings to services by first occurrence **through the spans' own (service, operation) pairs**, not
  by splitting at an underscore (operation names carry underscores). `parent_pos` and `trace_id` are
  in the file (the port reads neither); labels are not read.
- **Miss rules (registered).** A case on which the port raises `KeyError` (an operation first seen
  after the injection) is counted and scored as a miss with its own reason, never patched; a case
  whose post-fault spans contain no abnormal span yields an empty ranking (every score undefined) —
  passed through, scored as a miss by the benchmark's scorer and counted (`n_empty_rankings`);
  services absent from the ranking are misses at every k (`n_services_unranked`). Any other
  exception fails the run. No value is chosen on any read (ranking-only; defaults as the port's; the
  xs seed-0 export is the smoke test).
- **Hypotheses (mechanical in `report`).** H-rca-tracerca-random and H-rca-tracerca-rule: the §6
  rules applied to `rca/tracerca`.
- **Pull tier.** `pull --tier rca` gains `views/end-request/**` (the sequence files and
  `model-vocab.json`) for the exporter's spans extra; the method side never opens a view. The
  benchmark pin moves to the merge commit of the exporter's extras (D-TBR-11, the D-TBR-1 pattern).
- **Premise checks (2026-10-25, xs seed 0 end-request view pulled from the release; nothing kept).**
  12,578 request traces, 20 base ops; case 1 window (0–1,800 s): 6,243 traces, 20,595 pre- and
  10,309 post-injection spans; case 2 (1,800–3,000 s): 4,307 traces, 15,976 / 5,093 spans; **every
  operation seen after the injection was seen before it** on both cases, so the `KeyError` rule is not
  expected to fire at xs (it stays registered for any case where it does). The port's own import
  needs nothing beyond `trace-bench-rca`'s pandas and numpy.
- **Staging, protocol, reporting.** As §4 and §5 (`rank` per arm, xs → s → m, `score`, five seeds
  or nothing, random expectation and `control/rule` beside every row); per case the record keeps the
  spans and traces read, the abnormal-span count the port implies (recomputed for the count only
  from the port's own threshold rule on the same table), `n_services_unranked`, the miss reason and
  the loader's filled cells of `data.csv` (read for the record's row count only, not by the method).

### 2026-10-26 — D-TBR-11 pin moved (owner merged trace-bench PR #14)

- **§7 D-TBR-11.** The owner merged `export-rca-extras-26-10-26` into `clear-2027` on 2026-10-26
  (merge commit `fcb7ddb3`). `pyproject.toml` and `requirements.txt` now pin `fcb7ddb3` in place of
  `ace384b4`; the pinned tree differs from `ace384b4` only by `tracebench.export_rca` (the extras
  `topology` and `spans`, `EXTRAS`), its tests, README and RUN.md (`score.py`, `score_rca`,
  `score_service` byte-identical), so no landed cell moves. The three registered case files
  (`data.csv`, `window.json`, `services.json`) are written by the same code with or without the
  extras; the fresh exports made for the stretch arms assert their sha256 against the 2026-10-21
  export records before any read. Nothing in `benchmarks/trace-bench/` changes.

### 2026-10-26, later — correction to `rca/tracerca`: the synthetic client's spans (before any TraceRCA ranking)

- **What the door found.** The first xs read (seed 0, the registered smoke test) stopped at
  `tracebenchrca.cases.read_spans`, which refused `spans.csv` for naming a service outside
  `services.json`: `client`. The view reconstructs the synthetic client's root spans (its page loads,
  operations `/page/…`) and the exporter writes every span of a trace, so the file carries them —
  xs seed 0 case 1: 915 of 30,904 spans in 700 of 6,243 traces, every one a root (`parent_pos` −1).
  The 2026-10-25 premise counts (20,595 / 10,309 spans) include them. No ranking was recorded; the
  failed record was removed.
- **Rule.** The client is not a deployed service — `services.json` and `topology.json` exclude it,
  the scorer's candidate set never contains it — and its span is the whole request, so handed to the
  port it would be an "operation" whose spans are abnormal whenever anything downstream is slow,
  entering every operation's support denominator. The adapter therefore **drops the client's rows
  before the port** (the door admits exactly that one extra service name, `constants.CLIENT_SERVICE`,
  the benchmark's `Service(name="client", kind=KIND_CLIENT)`; any other unknown service still
  refuses the file) and records per case `n_spans_read`, `n_spans_client`, `n_traces_read` beside
  `n_spans` / `n_traces` (the rows handed to the port). Nothing else in the port's input changes;
  the miss rules stand. An alternative — passing the client's rows and deleting `client` from the
  returned ranking — was not taken for the denominator reason above.
- **Scope.** `rca/tracerca` only; `rca/circa-topology` reads `topology.json`, which already excludes
  the client. Registered 2026-10-26 before the first TraceRCA ranking; `tests/test_tracerca.py`
  gains the case.

### 2026-10-26, later (2) — l under the registered rule (stretch plans §4a; the bin-traffic job of 2026-10-26; no root-cause arm runs at l)

- **Measurement.** The 2026-10-14 bin-traffic script (sha256 `830176b3…`, unchanged) re-run on rung l ×
  seeds 0–4 only (c7i.4xlarge, 839 s of compute, 2026-10-26 18:01–18:15 UTC; HF
  `chadyuk/trace-bench@v0.3.0`, raw access lines with `/healthz` dropped; 152 candidate services + 1
  external per seed; widths 5 / 10 / 15 / 30 / 60 / 90 / 180 / 300 s). As on 2026-10-14: per service and
  width, the median, p10 and empty-bin share of the request count over the pre-fault stretches. No
  method ran; no ranking was produced; labels were read for the fault windows and root-cause flags only.
  Not a harness run: no ledger entry, no freeze.
- **The sparsest candidate (non-external) service, pre-fault, minimum over the five seeds.**

  | width | median count / bin | p10 | max empty-bin share | divides every l fault boundary (49 020 / 62 220 / 76 020 s) | bins in the 900 s crash |
  |---|---|---|---|---|---|
  | 5 s | 0 | 0 | 0.86 | yes | 180 |
  | 10 s | 0 | 0 | 0.75 | yes | 90 |
  | 15 s | 0 | 0 | 0.65 | yes | 60 |
  | 30 s | 1 | 0 | 0.42 | yes | 30 |
  | 60 s | 2 | 0 | 0.19 | yes | 15 |
  | 90 s | 3 | 1 | 0.10 | no | 10 |
  | 180 s | 5 | 2 | 0.02 | no | 5 |
  | 300 s | 9 | 4 | 0.002 | no | 3 |

  Per-seed sparsest medians at 60 s: 2 / 3 / 3 / 2 / 2; at 300 s: 9 / 17 / 16 / 13 / 11. The binding
  service is never a root cause on any seed. The whole-window scope gives the same table to within
  rounding.
- **Outcome under the §3 rule (N = 10: median ≥ 10 and p10 ≥ 1 on every seed, ≥ 5 bins in the shortest
  fault, a width that divides every fault boundary).** **No width passes.** The floor is not reached at
  any width on the grid — the widest, 300 s, reaches a minimum median of 9 and is excluded by both
  arithmetic constraints anyway; the widest aligned width, 60 s, has a sparsest median of 2 with empty
  bins at p10. Registered outcome, as the tfvars header of 2026-10-25 stated it either way: **the
  registered rule has no solution at l** — no root-cause arm runs at l at this submission (the paper's
  appendix A5.3, one sentence); the §3 widths for xs / s / m (30 / 15 / 60 s) and every landed cell are
  untouched. Nothing is chosen on this read: the rule was fixed on 2026-10-14, the l measurement was
  authorised on 2026-10-25 before any l number existed, and this addendum records the number.

### 2026-10-28 — stretch arm `rca/pc-pagerank` (beside CausalRCA; owner go 2026-10-28; before any PC-PageRank code or read)

Owner go of 2026-10-28 on the proposal of 2026-10-27 ("PC-PageRank beside CausalRCA"), with its
recommendations: the named-dataset preprocessing path (D-TBR-12), no graph axis, scikit-network as an
extra of this environment. This re-opens the stretch list (closed 2026-10-26) for one arm; the required
set is unchanged. CausalRCA is the only root-cause row whose ranking is a **learned graph + PageRank**
(DAG-GNN on the post-injection latency window, edges reversed, PageRank α 0.85, ten trainings averaged;
§1 and the addendum of 2026-10-22, later); PC-PageRank is the standard suite's classical counterpart —
**the same PageRank head on a PC graph**. The row separates two questions the table cannot: under the
PageRank head, does a classical learner do better than the neural one (PC vs DAG-GNN); and under the same
PC graph, does the head matter (PageRank vs CIRCA's regression-based hypothesis testing) — because the
port's PC call on this path is, case by case, the PC call of `rca/circa` (P5 below).

- **Arm.** `rca/pc-pagerank` — class `root-cause`, inputs `series, services` (**no window**: the port
  never reads the injection time), axis ranking: **PC-PageRank (suite port)**,
  `RCAEval/e2e/pc_pagerank.py::pc_pagerank(data, dataset="trace-bench")` at `259ea41` (MIT). One
  deterministic run per case, scored by the standard `tracebench.score_rca` path (the CIRCA precedent),
  not by the per-run averaging of RCD / CausalRCA.
- **Facts read at the pin (2026-10-27).** `pc_pagerank(data, inject_time=None, dataset=None,
  dk_select_useful=False, with_bg=False, n_iter=10, **kwargs)`: `preprocess(data, dataset,
  dk_select_useful)` → causal-learn `pc(data.to_numpy())` with its defaults (Fisher-z, α 0.05,
  `stable=True`, `uc_rule 0`, `uc_priority 2`, no background knowledge; `show_progress=True`, a progress
  bar on standard error) → a `networkx.DiGraph` with **every node kept** ("keep isolated nodes so the
  scores stay aligned"); `adj[i, j] == −1` adds `i → j`, `adj[i, j] == 1` adds `j → i` (a directed
  causal-learn edge yields one arc, an undirected one both) → `sknetwork.ranking.PageRank().fit_predict(adj.T)`
  (walks against the edges, toward causes — CausalRCA's reading) → metrics sorted by score, descending, by
  Python's stable sort (**ties keep the column order**, i.e. the export's). Returns `adj`, `node_names`,
  `ranks` — names only; the scores are not returned. **`inject_time`, `with_bg` and `n_iter` are never
  read**: the graph is learned on the whole case table, pre- and post-fault rows pooled, and PageRank is
  unpersonalised, so the ranking is the stationary centrality of the pooled PC graph. PageRank defaults
  at the installed version: `damping_factor 0.85, solver "piteration", n_iter 10, tol 1e-6`.
  Preprocessing (`io/time_series.py::preprocess`): `dataset=None` is the identity (the `time` column
  would enter PC and PageRank as a variable — the `rca/baro` / RCD trap); any other name but
  `"causalrca-sock-shop"` runs `drop_time` → `drop_constant` (whole-table) → `convert_mem_mb` (no `_mem`
  column here: a no-op). `rca/circa`'s port runs `pc_default(table minus time)` = `pc()` with the same
  defaults, so on the named path the two PC graphs coincide whenever `drop_constant` drops nothing. The
  `@rca` wrapper catches any exception, prints one line to standard error and returns the unranked
  column order with `adj = []`.
- **D-TBR-12 — the named-dataset preprocessing.** The adapter calls `pc_pagerank(table, dataset="trace-bench")`
  and passes nothing else (the D-TBR-6 rule, applied to this port): the suite's own path on every dataset
  it publishes, which keeps the clock out of the graph and makes the PC graph CIRCA's. The alternative —
  `dataset=None` with `time` removed from the returned ranking (D-TBR-2) — is rejected: the clock would
  still shape the graph every other metric is ranked on. The record counts the columns the path drops.
- **Choices.** Inputs `series, services` only — the door never opens `window.json` (as for
  `rca/baro-bocpd`). One run per case (stable PC + Fisher-z + power-iteration PageRank are deterministic;
  P3). Rankings are complete (isolated nodes kept) unless the wrapper falls back; collapsed to services by
  first occurrence (`cases.collapse_to_services`); `n_services_unranked` recorded (expected 0; a service
  unranked because its every column is constant over the table is a miss, the BOCPD precedent). **Ties =
  the port's order** (the export's column order); recorded per case — the number of distinct PageRank
  scores and the tie groups in rank order with the services first ranked inside each, re-derived from the
  port's returned adjacency with the port's own PageRank call (the port returns names only); on the score
  side the size of the tie group holding the root cause's best-ranked metric, and the count of cases where
  that group holds more than one metric. A diagnostic, never a re-ranking. **Fallback = D-TBR-8's rule:**
  `adj == []` → miss "port fallback", the wrapper's line kept; any other exception fails the run. Defaults
  as the port's; **no sweep**; the xs seed-0 export is the smoke test (no value chosen on it). Not
  proposed: a CausalRCA-matched variant (PC on the post-injection latency window only) — a non-suite
  recipe, i.e. a new method; and the graph axis (a service coarsening of the PC graph) — owner's call,
  no, at this date.
- **Hypotheses (mechanical in `report`).** **H-rca-pc-pagerank-random** and **H-rca-pc-pagerank-rule**: the
  §6 H-rca-random / H-rca-rule rules applied to `rca/pc-pagerank`. Two descriptive gaps go to the findings,
  not to hypotheses: PC-PageRank − CausalRCA (each series; the learner under the same head) and
  PC-PageRank − CIRCA (the head under the same graph).
- **Environment.** `pyproject.toml` gains the extra `pagerank = ["scikit-network>=0.31.0"]` (the pin's own
  floor for its default mode; its changelog records a PageRank fix for scikit-network ≥ 0.32). Resolved
  2026-10-27 into `trace-bench-rca`: scikit-network **0.33.5** (the cp312 macOS arm64 wheel; the package's
  `__version__` string prints 0.33.0), beside numpy 2.5.4, scipy 1.18.1, causal-learn 0.1.4.8, networkx
  3.7, unchanged; every record's `pip freeze` names it. The RCD environment is not needed (no
  `local_skeleton_discovery`; D-TBR-7 does not apply) and the CausalRCA environment is not touched.
- **Premise checks (2026-10-27, on fresh xs and m seed-0 exports of the three registered files, no
  extras; every case file sha256-equal to the 2026-10-26 export records; nothing kept).**
  **P1** — in a clone of `trace-bench-rca` with the wheel installed, `RCAEval.e2e.pc_pagerank` imports
  under numpy 2.5.4 / scipy 1.18.1 (the suite's lenient import had silently skipped the module), PageRank
  defaults as read above, and `rca/baro` and `rca/circa` rankings on both xs seed-0 cases are identical
  with and without the wheel. **P2** — xs seed 0, both cases: complete rankings over the 20 metric
  columns, no fallback, `time` absent, 0 constant columns dropped; PC graphs of 19 / 16 undirected edges;
  17 / 13 distinct PageRank scores of 20, the ties being isolated nodes at the tail (groups of 4 and 5)
  and, on case 1, three pairs mid-ranking. **P3** — two calls on the same table: identical `ranks` and
  `adj` on every case tried (xs both, m case 0). **P4** — m seed 0 case 0 (208 columns × 246 rows): one
  port call 94.9 s wall, peak memory footprint 0.60 GB (resident 0.67 GB) — the CIRCA PC call's cost;
  expected ≈ 35–40 min for the 25 m cases, local. **P5** — the port's `adj` equals the networkx
  adjacency built by the port's edge rule from `rca/circa`'s `pc_default(table minus time)` on xs both
  cases and m case 0 (376 undirected edges at m — the CIRCA premise's number); `drop_constant` dropped
  0 columns on every case tried.
- **Staging, protocol, reporting.** As §4 and §5: `rank` per arm, xs → s → m by `--prior-rung-record`,
  `score` (`tracebench.score_rca`, B = 2000, seed 0), five seeds or nothing, the random expectation and
  `control/rule` beside every row; per case the record keeps `n_edges`, `n_metrics_dropped`,
  `n_distinct_scores`, `tie_groups`, `n_services_unranked`, the fallback flag and the loader's filled
  cells. Paper placement is the paper's decision; nothing here depends on it.

### 2026-10-30 — stretch arms `rca/trace` and `rca/trace-core` (the reference algorithm's own root-cause rule; publication plan B4b; owner go 2026-10-30; before any TRACE root-cause code or read)

Owner go of 2026-10-30 on the draft of 2026-10-21 ("proceed with the rule as recorded"). The rule is the
one the owner proposed to the method's author in the mail of 2026-10-21 (point 2), as the lab's reading of
the author's own ask of 2026-10-19 (point 2: "TRACE evaluated on AC@k with its own RCA rule (Markov boundary
of the outcome, then upstream traversal of the sample graph), next to BARO and CausalRCA"). The author
neither confirmed nor declined the proposed reading by this date; the owner registers it as proposed and the
author reviews the draft. A different rule text from the author before the freeze enters by a dated addendum
and a re-run, never by an edit here. The row answers the question the author asked: with its own sample
graph and its own rule, how does TRACE rank the deployed services of a fault case, under the same scorer
and intervals as the published root-cause methods.

- **Arms.** `rca/trace` — class `root-cause`, inputs `views, window, services`, axis ranking: **TRACE
  (Math & Lienhart 2026), the atomic construction** — the seed's frozen latent backbone and the frozen cell
  `trace/cli-atomic/shipped/frozen/request` of the seed's committed freeze (the reference harness's
  `freezes/<date>-<rung>-latent-s<k>.json`): the per-cause intervention construction, best within-trace on
  every rung up to l; the shipped arithmetic and the frozen cut — the one path and one cut the paper reports
  (owner 2026-10-28). `rca/trace-core` — the same with `trace/core/shipped/frozen/request` (the full
  ancestral-history construction, the second row). Code: the author's `Mathugo/seq2cause` at the shipped
  commit `67416a2f0ba6ef675a39028518f7a45bbf327291` (v0.1.9, MIT in `pyproject.toml`), driven through the
  reference harness's engine adapter `seq2causebench.engine` (RUN.md D-SB-2 of `benchmarks/trace-bench`: the
  shipped functions in the shipped order from one forward pass; bit-equal to both shipped entry points on the
  fixture) — the code path that produced every landed TRACE cell — so the sample graph a trace gets here is
  the one the reference harness's per-sequence axis scored. Grain: request (the call tree is where a fault
  propagates upstream); the end-ordered view (every landed TRACE cell; the start view's per-sequence truth is
  empty by construction, `plans/per-sequence-rules.md` addendum 2026-10-23).
- **The declared input `views`.** The end-ordered request view of the pulled corpus:
  `views/end-request/sequences/split=*/…/part-*.parquet` rows (`trace_id`, `ops`, `outcomes`, `n_spans`,
  `start_ns`) and `views/end-request/model-vocab.json` (`base_ops` with `{id, service, name, kind}`, the
  minted variants) — opened through the benchmark's method accessor (`tracebench.allowlist.open_for_method`,
  via `seq2causebench.corpus.Corpus`), which refuses the graphs, the labels, the oracle and the topology.
  `parent_pos` is never read (the reference harness's scenario 34). The exporter is not changed: the view is
  method-readable by the benchmark's own contract (`raw/` and `views/`), and this arm is declared
  trace-reading and call-tree-informed by construction, as `rca/tracerca` is. **Window traces:** every row
  of the three splits with `inject_time · 1e9 ≤ start_ns < end_time · 1e9` (`window.json` through the case
  door; `start_ns` is the trace's earliest span start, UNIX ns). The backbone was trained on the train split
  of the same corpus, so a window trace may be one the backbone saw during pretraining: the root-cause read is
  not a held-out read on the discovery axis — the method trains nothing here (the backbone is frozen), and the
  statement travels with the row. Order: `start_ns`, then `trace_id`; **the first 500 are probed** (the cap,
  fixed here before any read; `n_traces_window` and `n_traces_probed` recorded per case).
- **The sequence and the sample graph.** Encoded exactly as the frozen cell's reads were
  (`seq2causebench.data.encode_sequence`: `BOS + ids + EOS`, at most 64 real tokens, an unminted non-OK
  variant folded to its base op and counted); the frozen `(c, N, g, τ)` of the cell and the seed's backbone,
  whose weights hash must equal the freeze's `model_sha256` and whose vocabulary size the view's
  `model-vocab.json`; one probe pass per trace (`engine.probe_sequence`, noise `all`, no autocast, the
  shipped KL); the shipped samplers draw from torch's global generator, seeded per trace as the reference
  harness seeds per sequence (`(seed · 1000003 + 3) · 1000003 + i`: stream 3 = root-cause reads, seed = the
  corpus seed, i = the trace's index in the case's probed list). A trace with fewer than two window
  positions is skipped and counted (`discover`'s rule). The sample graph of a trace is the frozen cut of its
  score matrix: a directed edge j → q for every window position pair j < q with `S[j, q] > τ` (strict, the
  cell's frozen τ; window position t is token `ids[c + t]`; `S[j, q]` with j before q is the statistic the
  per-sequence axis scores as j → q), both tokens real (PAD, BOS, EOS, UNK never enter). The first c tokens
  (BOS and the context) are outside the matrix: they are neither effects nor causes, and a non-OK token
  among them is counted (`n_nonok_in_context`), never ranked.
- **The rule (the lab's reading of the author's rule, as sent 2026-10-21).** A **non-OK position** is a
  window position whose token's outcome is `5xx`, `err` or `slow` in the model vocabulary (`4xx` is a
  client-side outcome, excluded; a folded rare variant is `:ok` and invisible — the fold share at xs–m is
  stated in the datasheet). For each non-OK position q, its **parents** are `{j < q : S[j, q] > τ}` (its
  Markov boundary in the sample graph, which has only backward-in-time edges, is its parent set), and its
  **upstream set** is the transitive closure of the parent relation, with the hop distance of a position = the
  length of the shortest parent chain from q. Positions map to operations through the token (`op_of`) and to
  services through `model-vocab.json`'s `base_ops`; the synthetic client's operations (service `client`,
  kind client — not a candidate, absent from `services.json`) are dropped and counted. **Per case, a
  service's score = the number of probed traces in which it is upstream (hop ≥ 1) of at least one non-OK
  position;** ties by the mean, over those traces, of the trace's smallest hop distance to the service
  (smaller first); then `services.json` order. Services never upstream of a non-OK position follow, in
  `services.json` order, so every deployed service is ranked (`n_services_reached` recorded). A case with no
  probed trace, no non-OK position in any probed trace, or no upstream service is **a counted miss** (reasons
  `no window trace`, `no non-OK position`, `no upstream service`) — the export's order is never scored. The
  non-OK position's own service is not scored by its own failure (hop 0): its count comes only from traces
  in which it is upstream of another non-OK position; the hop-0 counts are recorded as a diagnostic
  (`self_counts`), never ranked.
- **Choices.** No sweep: every knob is the frozen cell's or fixed here; the xs seed-0 export is the smoke
  test (no value chosen on it). Device: CPU (the reference harness's CPU reads are bit-identical; a GPU or
  Apple-silicon device, if the m cost forces one, is recorded and disclosed with the row). Both rows run on
  xs, s, m × seeds 0–4; the per-case record keeps `n_traces_window`, `n_traces_probed`, `n_skipped_short`,
  `n_truncated`, `n_folded`, `n_nonok_positions`, `n_nonok_in_context`, `n_traces_with_nonok`,
  `n_traces_with_upstream`, `n_edges` (cut edges over the probed traces), `n_client_positions_dropped`,
  `n_services_reached`, the per-service counts and mean hops, `self_counts`, the probe's wall clock and
  token-positions per second, and the miss reason. Not proposed: the session grain (journey and retry
  edges cross requests; the rule is a within-request traversal), the fixed paths (the paper reports the
  shipped arithmetic), the shipped cut (its union is a type-level rule), a hop-weighted score (a new rule).
- **Hypotheses (mechanical in `report`).** **H-rca-trace-random**, **H-rca-trace-rule**,
  **H-rca-trace-core-random**, **H-rca-trace-core-rule**: the §6 H-rca-random / H-rca-rule rules applied to
  each row. Descriptive gaps for the findings, not hypotheses: `rca/trace` − `rca/tracerca` (the two
  trace-reading arms), `rca/trace` − `rca/trace-core` (the construction), `rca/trace` − `rca/rcd` (the
  best published arm).
- **Environment and code path.** The probe runs in the reference harness's environment
  `seq2cause-trace-bench` (Python 3.11, torch 2.14.1, transformers 5.19.0, seq2cause 0.1.9, numpy 2.4.6 — the
  environment of every landed TRACE cell, RUN.md of `benchmarks/trace-bench`), through a runner script
  (`trace_runner.py`, imports `seq2causebench` and `seq2cause` from the same commit as this directory) that
  this directory's `ranktrace` stage executes once per corpus and arm (the `rankrcd` pattern: the backbone
  loaded once; one JSON per case holding every probed trace's cut edges and, per window position, its
  operation, outcome and service, plus the counts above); the rule and the ranking run on this side
  (`tracerule.py`, pure Python, tested on hand-built sample graphs), never in the runner. The record names
  both environments' `pip freeze`, the freeze file's path, sha256 and commit, the cell's `(c, N, g, τ)`, the
  model sha256, the runner's `seq2causebench` and `seq2cause` import paths and commit, and the device.
- **D-TBR-13** (RUN.md): follows the author's rule as he stated it and the frozen cells as landed; deviates
  from TRACE as published, which has no root-cause stage in the X→X regime the harness exercises (the paper's
  Markov-boundary rule is stated for the outcome head of the X→Y regime, which this benchmark does not
  provide — the owner's mail of 2026-10-21, point 6): the rule registered here is the lab's reading in the X→X
  regime (the Markov boundary of each non-OK event token in the sample graph, then the upstream traversal);
  the probe is driven by the harness's engine adapter rather than the shipped CLI (as every landed cell);
  the window traces are read from all three splits. Dated 2026-10-30, before any TRACE root-cause run.
- **Premise checks (to run on the xs seed-0 export and corpus before the first ranking; results in the
  findings).** P1 — the view's rows carry `start_ns` and the window selection finds traces in every case.
  P2 — the number of window traces per case at xs, s and m seed 0 against the cap. P3 — the probe's cost
  per trace on the CPU at xs and m (the m cell's cost sizes the five-seed run). P4 — the runner's sample
  graph of one trace equals the reference harness's stored matrix cut at τ for a trace of the landed test
  read, where that trace is in the window (a spot check of the code path; if no landed test-read trace falls
  in a window, the check is on the matrix shape and the cut only).
- **Staging, protocol, reporting.** As §4 and §5: `ranktrace` per arm, xs → s → m by `--prior-rung-record`
  on the same arm, `score` (`tracebench.score_rca`, B = 2000, seed 0), five seeds or nothing, the random
  expectation and `control/rule` beside every row. Paper placement is the paper's decision; nothing here
  depends on it.

### 2026-10-31 — the per-trace root-cause axis for `rca/trace` and `rca/trace-core`, with `floor/trace-topology`, `control/trace-random` and `control/trace-callees` (publication plan of 2026-10-31, approved by the owner the same day; before any per-trace code or read)

Owner decision 2026-10-30 ("a per-trace root-cause axis goes into this submission") and the owner's
answers of 2026-10-31 on the plan's four forks (uncapped selection on a GPU box; floors on the whole
trace; case-weighted aggregation; the full row set plus an exact-set line). The ranking axis above
asks, per fault case, which deployed service is the root; this axis asks, per **failed request**,
which service in *this* trace is the root — the method author's own framing (2026-10-19: the Markov
boundary of the outcome, then the upstream traversal of the sample graph) applied trace by trace
instead of aggregated over a case's traces. The benchmark ships the truth for it in
`labels/sequence-labels.parquet` (one row per request trace and injected fault: the fault phase of
the trace, the end-view positions of the root-cause service's operations, whether one of them failed),
which nothing read before this date. Scorer: `tracebench.score_rca_trace` (benchmark branch
`clear-2027`, `docs/scorers-service-rca.md` §4, dated 2026-10-31), through this directory's
`scoretrace` and `reporttrace` stages.

- **Axis `trace`** (a third value beside `ranking` and `graph`). Request grain only (the only grain the
  labels cover). Arms scored on it: `rca/trace` and `rca/trace-core` (their declared inputs unchanged:
  `views, window, services`), each under two **readings** — `hop1` (the author's rule as registered on
  2026-10-30: a service is a candidate when it is upstream, hop ≥ 1, of a failed position) and `hop0`
  (the failed positions' own services enter at hop 0) — the primary row is `hop1`, the `hop0` row is
  reported beside it; three model-free score-side rows: `floor/trace-topology` (the same rule run on
  the trace's **true call tree**: for every failed non-client span its call-tree descendants —
  `parent_pos` followed downwards, hop = chain length — ranked by the number of failed spans whose
  subtree holds the service, then the smallest hop, then name order; both readings), `control/trace-random`
  (the analytic expectation of a uniformly random ranking of the trace's distinct non-client services:
  min(k, n_t) / n_t) and `control/trace-callees` (the analytic expectation of a random ranking restricted
  to the failed spans' descendant services, the predict-all audit of this axis; both readings); and
  **broadcast rows** for the eleven landed case-level rankings (`control/rule`, `rca/baro`,
  `rca/baro-bocpd`, `rca/causalrca` p50 and p95, `rca/pc-pagerank`, `rca/circa`, `rca/circa-topology`,
  `rca/tracerca`, `rca/rcd`): the case ranking filtered to the trace's own services, order kept, services
  the case ranking lacks appended in name order, an unranked case a miss; for the run-averaged arms the
  per-trace hit is the mean over the ten runs (a failed run 0). The broadcast rows cannot localise within
  a trace by construction and are reported as references, as the discovery axis's floors are.
- **Selection rule R (method side, label-free; the score side recomputes it and refuses a mismatch).**
  Per case, every request trace of the end-ordered view, all three splits, that lies wholly inside the
  fault window — `inject_time · 1e9 ≤ start_ns` and `start_ns + max(offsets + durations) < end_time · 1e9`,
  integer nanoseconds, the two `window.json` fields — and holds at least one **failed non-client span**
  (raw view `outcomes` in {5xx, err, slow}; 4xx is a client-side outcome, as above). Ordered by
  `start_ns`, then `trace_id`; **no cap** (`--cap 0`; the owner's decision of 2026-10-31 — the case
  rule's first 500 would have left 28 scored traces for the xs crash fault). The runner records the
  selected and the skipped (fewer than two window positions) trace ids per case. The case-level
  selection of 2026-10-30 (start in the window, first 500) stays available as `--select window` for the
  landed rows and the tolerance check below. Measured 2026-10-31 on seed 0 (label side, before
  registration): R holds 1,128 / 758 traces of the two xs windows, 7,463–35,450 per s window and
  4,031–22,597 per m window; R's phase test agrees with the labels' POST phase on every trace of xs, s
  and m seed 0 (0 mismatches).
- **Scored set C (score side).** The selected traces in which **the fault took**: a trace whose
  root-cause positions (`rc_positions_end`: the end-view positions whose operation belongs to the
  fault's service and endpoints) hold at least one outcome in {5xx, err, slow} — recomputed from the view
  row, not the label's `rc_took`, which also counts 4xx (0–8 traces per case at seed 0; the implication
  took ⇒ `rc_took` is asserted). A selected trace the runner skipped is a counted miss. Measured on seed 0:
  |C| = 623 / 48 (xs), 1,844 / 1,442 / 7,754 (s), 2,565 / **30** / 889 / 305 / 1,058 (m) — the m crash
  fault's 30 traces are disclosed with the row. Splits of C at seed 0: 79 % train, 10 % val, 11 % test
  (the backbone saw the train split; the per-split hit rate is reported as the training-overlap audit,
  never as the axis). Not proposed: STRADDLE / PRE / NONE traces, OK traces, a "no root cause" class, the
  session grain.
- **Prediction, truth, metrics.** Per scored trace a ranking of the trace's distinct non-client
  **services** (every row) and, for the trace-reading rows, of **positions** (the arms: window positions;
  the floor: end-view positions). Truth: `root_cause_component`; for positions `rc_positions_end`, mapped
  for an arm to window position `i − c + 1` where reachable (`c − 1 ≤ i < min(len(ops), 64)`); a
  root-cause position inside the context or beyond the length cap is unreachable and counted
  (`n_rc_unreachable_context`, `n_rc_unreachable_truncation`; the floor and the controls are defined on
  the whole trace, so the arms' context cost is a reported diagnostic, owner decision 2026-10-31).
  **hit@1** and **hit@3** of the root-cause service (positions: of any root-cause position); the **exact
  set** (owner addition 2026-10-31; trace-reading rows only): the method's native candidate set — the
  union of non-client services upstream (hop ≥ 1; hop 0 added in the `hop0` row) of the trace's failed
  positions under the frozen τ, for the floor the descendant services under the true call tree — scores 1
  iff it equals {root-cause service}; reported with the mean set size; "n/a" for the broadcast and
  random rows. **Random expectation per trace**: services min(k, n_t) / n_t; positions 1 − C(m − r, k′) /
  C(m, k′), k′ = min(k, m), 0 when r = 0. Per trace the candidate universe has 3.9–6.7 services at seed 0,
  so hit@3 is near its ceiling; hit@1 and the exact set carry the comparison. **Aggregation (owner
  2026-10-31): per case the mean over its scored traces; the point is the mean over cases** (the ranking
  axis's unit, equal fault weight); **intervals = the percentile bootstrap over cases** (`numpy.random
  .default_rng(seed).integers(0, n, (b, n))`, B = 2000, seed 0, the run-averaged machinery of 2026-10-23),
  pooled per rung over the five seeds and across rungs by concatenating the case rows (the cluster is the
  (seed, case) pair; xs has 10 clusters, disclosed). Secondary points, no interval: the trace-weighted
  mean, the mean candidate count, per fault kind, per split.
- **Hypotheses (mechanical in `reporttrace`).** **H-trace-perseq-random**: on every rung the hit@1
  interval of each trace-reading row (`rca/trace`, `rca/trace-core`, `floor/trace-topology`, reading
  `hop1`) lies above the per-trace random expectation. **H-trace-perseq-topology**: on every rung the
  floor's hit@1 ≥ each TRACE row's (same reading). **H-trace-perseq-broadcast**: on every rung each TRACE
  row's hit@1 ≥ every broadcast row's (the author's claim, stated as the pre-registered expectation).
  **H-trace-perseq-self**: the `hop0` reading changes the sign of (TRACE − the best broadcast row) on no
  rung. The exact set is descriptive. Descriptive gaps for the findings: TRACE − floor, `hop0` − `hop1`,
  the context-unreachable share, the per-split gap.
- **Environment, device and the tolerance check (D-TBR-15).** The 30 probe runs (xs, s, m × seeds 0–4 ×
  two arms; ≈ 1.1 M traces, ≈ 4 h at the measured GPU probe cost) run on one g5-class EC2 replica of the
  reference harness's provisioning (`infra/algos/seq2cause-bench-26-10-31-rca-trace-axis.tfvars`, job
  script `scripts/jobs/rca-trace-axis-26-10-31.sh`: both environments built on the box, the fifteen
  frozen backbones as input artifacts and hash-checked against the freezes, corpora pulled and exported
  on the box, exports hash-checked against the landed `export-trace` records, the per-trace outputs synced
  back), **device `cuda`**. CPU and CUDA draw different noise under the same seed, so a GPU probe is a
  **disclosed replica of the frozen cells, not a bit-identical reproduction** of the landed CPU rows
  (which stay as landed; nothing under `rca-trace{,-core}/` is rewritten). In place of the CPU
  bit-identity check (P4 of 2026-10-30) the replica's first stage re-probes the 30 landed case-level
  selections (`--select window`, cap 500) on the GPU and `tolerance` compares them trace by trace with
  the landed CPU work files: positions identical (refused otherwise), the Jaccard of the cut edge sets per
  trace; **gate: mean per-trace Jaccard ≥ 0.80 per run** (`TRACE_TOLERANCE_JACCARD`); the per-case
  top-1 service of the GPU re-probe against the landed one is reported beside it. A failed gate is a
  disclosed deviation and an owner decision, never a silent block or a silent pass. The per-trace noise
  seed is the trace's index in the selected list, so the uncapped selection gives a trace a different
  draw than the capped one — a second reason the two reads are not compared trace by trace.
- **Files and records.** The per-trace rankings (`ranking-trace.json`: per case the selected and skipped
  ids and, per probed trace, the reached services as indices into the export's name-ordered
  `services.json` and the reached positions, both readings) and the per-trace score rows
  (`score-trace-traces.json`) are too large to track uncapped (up to ≈ 60 MB per m run) and live on the
  bucket and in the local scratch; **only their sha256 ride the records** (`rank-trace` and
  `score-trace`), as the case-level `ranking-positional.json` does. Tracked: the `rank-trace`,
  `tolerance` and `score-trace` records, `score-trace.json` (aggregates and per-case rows) under
  `results/<rung>/latent/seed=<k>/<arm slug>/trace/`, and `tables/rca-trace/` from `reporttrace`.
  `pull --tier trace` fetches `labels/sequence-labels.parquet` for the score side only; the runner's source
  never names it and never reads `parent_pos` (a source test), so the method side's contract is unchanged.
- **D-TBR-14** (RUN.md): the benchmark pin moves from `fcb7ddb3` to the `clear-2027` commit that carries
  `tracebench.score_rca_trace` once the owner merges it; `score.py`, `score_rca`, `score_service` and the
  exporter are unchanged, so no landed cell moves. **D-TBR-15**: the axis, the selection rule R without a
  cap, the GPU device as a disclosed replica with the tolerance check, the per-trace files out of git.
- **Premise checks.** P1–P3 run 2026-10-31 on seed 0 of xs, s and m from the released labels before this
  registration (the counts above; the identity `rc_positions_end` ↔ the root-cause service's operations
  holds on every label row; `parent_pos` points to a later position on 97–98 % of links, inversions kept):
  a root-cause span is the only failed span of its trace in 1–14 % of C at xs and s and in 29 % / 37 % /
  68 % of C for the m degrade (`tb-sitizo`), cache-flush and edge-service (`tb-edge`, whose only caller is
  the client) faults — where the `hop1` reading cannot name it and the `hop0` row carries the comparison.
  P4 = the tolerance check above, run on the box's first stage and scored before any per-trace read is
  landed. No hit rate of any row was computed before this registration.
- **Staging, protocol, reporting.** `ranktrace --axis trace` per arm, xs → s → m by `--prior-rung-record`
  on the same arm and stage (`rank-trace`); `scoretrace` (every row of the axis, B = 2000, seed 0);
  `reporttrace` (five seeds or nothing; `tables/rca-trace/`); the findings in `findings/rca-trace.md`.
  Paper placement is the paper's decision; nothing here depends on it.

### 2026-10-31, later — the context family's root-cause arms `rca/trace-ig-overwrite`, `rca/trace-cmi-overwrite`, `rca/trace-ig-keep`, `rca/trace-cmi-keep`, `rca/trace-atomic-cmi-keep`; the hop-0 and source readings; the pre-fault subtraction; the control `control/trace-earliest` (the method author's proposals of 2026-10-30; owner decisions 2026-10-30 / 31; before any such code or read)

The method author's mail of 2026-10-30 on the root-cause rule: the current rule "counts every upstream
ancestor of a non-OK token, healthy ones included, and with hop ≥ 1 a failing service can never credit
itself"; the hop-0 numbers "look like the natural reading"; he proposes to score only the **sources** of the
non-OK subgraph, to **subtract** the same count on traces from a pre-fault window, to add an **"earliest
error in the trace" control** beside the call-tree floor, and to call the current rule the
**upstream-closure rule** rather than "the author's rule" ("since we did not design it. If you want an
author's rule, the source rule is the one we would sign"). From this addendum on, the rule registered on
2026-10-30 is the **upstream-closure rule** (reading `hop1`: a service is credited when it is upstream,
hop ≥ 1, of a non-OK position); the earlier name stays in the dated texts above as written. Owner
decisions: hop 0 is the **primary** reading on both root-cause axes, hop ≥ 1 reported beside it; the three
proposals are registered here with their definitions and run as new rows; the family's arms are the
reference harness's context family (`benchmarks/trace-bench/plans/reference-arms.md`, addendum of
2026-10-31; seq2cause v0.3.1 `280ed948`). Nothing landed moves: the landed `rca/trace` and `rca/trace-core`
rows stand as landed and gain new rows beside them.

- **Arms.** Five new arms, class `root-cause`, inputs `views, window, services` (declared, read and bounded
  exactly as `rca/trace`, 2026-10-30), axes `ranking` and `trace`, each reading one frozen request cell of the
  seed's committed **context freeze** (`freezes/<date>-<rung>-latent-s<k>-context.json`): `rca/trace-ig-overwrite`
  → `trace/core-ig-overwrite/fixed-kl/frozen/request`; `rca/trace-cmi-overwrite` →
  `trace/core-cmi-overwrite/fixed-kl/frozen/request`; `rca/trace-ig-keep` → `trace/core-ig-keep/fixed-kl/frozen/request`;
  `rca/trace-cmi-keep` → `trace/core-cmi-keep/fixed-kl/frozen/request`; `rca/trace-atomic-cmi-keep` →
  `trace/cli-atomic-cmi-keep/fixed-kl/frozen/request`. The family reads the `fixed-kl` path (the author's
  released default for the CMI, log-space KL; the landed `rca/trace` rows read the `shipped` path, the one
  path the paper reports — the difference is stated with every table). Code: the author's `Mathugo/seq2cause`
  at v0.3.1, commit `280ed948` (`CONTEXT_FAMILY_SEQ2CAUSE_COMMIT`, the full sha in `constants.py`), through
  the reference harness's engine adapter as every landed TRACE cell (the frozen `(c, N, g, τ)`,
  `context_mode` and `window_offset` of the cell; the backbone hash asserted against the freeze; one probe
  pass per trace; the per-trace seed stream of 2026-10-30; the keep arms at `c = 1` test every position,
  `n_nonok_in_context = 0` by construction). The information-gain and CMI cells of one core layout share
  one forward (the runner probes a trace once per layout and cuts both score matrices at their own τ; the
  work file carries both cuts; recorded `cells`); the atomic CMI arm is its own forward. xs, s, m × seeds 0–4.
- **Readings (every TRACE arm, both axes; `--reading` required on `ranktrace`).** `hop1` — the
  upstream-closure rule as registered on 2026-10-30 (the landed reading). `hop0` — the same closure plus the
  non-OK position's own service at hop 0 (the reading the per-trace axis reports since 2026-10-31, lifted to
  the case count: a service's case score = the number of probed traces in which it is upstream, hop ≥ 0, of
  a non-OK position; ties by the mean smallest hop, then `services.json` order). **`hop0` is the primary
  row on both axes from this date; `hop1` is reported beside it.** `source` — the author's source rule: a
  non-OK position is a **source** when none of its parents in the cut sample graph (`{j < q : S[j, q] > τ}`)
  is a non-OK position (an isolated non-OK position, with no parents at all, is a source); a service's case
  score = the number of probed traces in which it holds at least one source; ties by the number of sources
  held (more first), then `services.json` order; per trace the source services ranked the same way, the rest
  appended in name order. Recorded per case: `n_sources`, `n_sources_isolated`. The landed `rca/trace` and
  `rca/trace-core` get case-level `hop0` and `source` rows **re-ranked from their landed 2026-10-30 work
  files** (`ranktrace --from-work-dir`, the files' sha256 recorded against the landed `rank` records; no
  re-probe) if those files still exist, else from a CPU re-probe of the landed selection at the cap of 500,
  disclosed; their per-trace rows keep `hop1` / `hop0` as landed (`source` per trace for the new arms only).
- **Pre-fault subtraction (case level only; the author's proposal, defined here and flagged to him).**
  Per case a **pre-fault stretch** `[pre_start, inject_time)`, with `pre_start` the preceding case's
  `end_time` in injection order, or the corpus start (0 s) for the first case of a corpus (`pre_source`
  recorded as `previous-fault-end` or `corpus-start`; the two window fields enter through the case door as
  `window.json` does, `cases.json` gaining `pre_start_time`). The runner's phase `pre` selects the stretch's
  traces by the case rule of 2026-10-30 (start inside the stretch, all splits), ordered by `start_ns`
  **descending** from the injection (the traces nearest the fault) and capped at 500 as the post selection is,
  probes them under the same cell, cap, seed stream and device (the index in the pre list seeds the trace),
  and records `n_traces_pre`, `n_traces_pre_probed`. `ranktrace --subtract pre`: a service's score =
  `count_post / n_post_probed − count_pre / n_pre_probed` under the same reading (the per-trace credit rates;
  both counts and both denominators recorded); ties by the post rate, then `services.json` order; a case
  whose pre stretch holds no probed trace keeps the post score with `n_pre_probed = 0` and is counted
  (`n_cases_no_pre`). Rows `…/<reading>-subpre/` for the five new arms (not for the landed arms: no pre
  probe exists for them; reason recorded). Alternatives not chosen, so the choice is not re-made on
  outcomes: a ratio of rates, a floor at zero, a subtraction at the per-trace level (undefined per trace), an
  uncapped pre selection. The definition is flagged to the author with this addendum; a different text from
  him enters by a dated addendum and a re-run.
- **`control/trace-earliest` — the earliest-error control (model-free, label-free, method-side as
  `control/rule`; `plans/controls.md` §3).** Per selected trace (the trace axis's rule R; the case axis's
  window selection at the cap of 500) the non-client services ordered by the start time of their earliest
  non-OK span (`start_ns + offset`, the raw view), the remaining services appended in name order; positions
  likewise (the ops indices in that order; under c = 1 the window position is the ops index). Case level: a
  service's score = the number of selected traces in which it holds the trace's earliest non-OK span (ties
  within a trace: every tied service), ties by `services.json` order → `ranking-positional.json`, scored by
  `tracebench.score_rca` as every case ranking; per trace → the mode `arm` of `tracebench.score_rca_trace`
  through `scoretrace` (an `arm_only` row path, so the floor and broadcast rows are not duplicated). No
  dataset-repository change. The control answers the author's question whether the sample graph adds
  anything over "the first thing that failed".
- **Row set (bounded).** Case level: the five new arms × readings {`hop1`, `hop0`, `source`} × {post,
  post − pre}; the landed two arms × {`hop0`, `source`} re-ranked; `control/trace-earliest`. Per trace: the
  five new arms × {`hop1`, `hop0`, `source`}; the landed two as landed; `control/trace-earliest` beside
  `floor/trace-topology`. The `hop0` row is the primary row and the first printed; the pre-fault rows and
  the l / xl rungs (which have no root-cause arm) are untouched by this addendum. `reporttrace`'s shared-row
  check extends to the new cells; the landed RCA tables are regenerated and asserted byte-equal.
- **Hypotheses (mechanical in `report` / `reporttrace`; Avg@5 at case level, hit@1 per trace, the
  registered bootstrap B = 2000, seed 0).** **H-rca-hop0**: for every TRACE arm (landed and new) the `hop0`
  reading's Avg@5 ≥ the `hop1` reading's on every rung; *fail:* any arm and rung where `hop0` < `hop1`.
  **H-rca-ctx-rule**: each new arm's `hop0` Avg@5 ≥ `control/rule`'s on every rung; **H-rca-ctx-rcd**: ≥
  `rca/rcd`'s (the best published arm); *fail:* otherwise on any rung. **H-rca-source**: `source` ≥ `hop1`
  for every TRACE arm and rung; *fail:* otherwise. **H-rca-presub**: post − pre ≥ post under the `hop0`
  reading for every new arm and rung; *fail:* otherwise. **H-rca-atomic-cmi**: `rca/trace-atomic-cmi-keep`'s
  `hop0` Avg@5 ≥ `rca/trace`'s `hop0` (its landed sibling on the same construction) on every rung; *fail:*
  otherwise. **H-trace-earliest**: each TRACE arm's `hop0` hit@1 per trace ≥ `control/trace-earliest`'s on
  every rung; *fail:* otherwise. Descriptive gaps for the findings: keep − overwrite and CMI − information
  gain per reading and axis; `source` − `hop0`; the context-unreachable share (0 for the keep arms by
  construction; the P3 shares of the reference harness's addendum printed beside it).
- **Environment, cost, staging.** As 2026-10-30 / 31: the case-level probes on the CPU (`ranktrace` per arm
  and reading; the per-layout dedup above; xs → s → m by `--prior-rung-record`); the per-trace probes on the
  GPU replica of D-TBR-15 (three layouts: core overwrite, core keep, atomic CMI keep; ≈ 1.1 M traces per core
  layout; the atomic layout measured first on the CPU at m, P4′ of the reference harness's addendum), the
  tolerance check of D-TBR-15 not applicable (no landed CPU row of these cells; the replica is disclosed as
  such); `score` / `scoretrace` (B = 2000, seed 0); five seeds or nothing; the random expectation,
  `control/rule` and `control/trace-earliest` beside every row. Paper placement is the paper's decision.
- **Deviations (RUN.md).** **D-TBR-16** the five arms on v0.3.1 cells of the context freeze, the `fixed-kl`
  path, the second code pin, the per-layout dedup; **D-TBR-17** the readings (`hop0` primary, `source`; the
  landed arms' `hop0` / `source` rows re-ranked from their work files); **D-TBR-18** the pre-fault phase and
  subtraction; **D-TBR-19** the earliest-error control scored through the per-trace scorer's mode `arm`.
