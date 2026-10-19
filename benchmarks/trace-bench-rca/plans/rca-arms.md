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
