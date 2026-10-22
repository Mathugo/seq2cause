# Controls for the root-cause arms (pre-registered 2026-10-14, before the first root-cause run)

The two controls every root-cause table carries beside the methods (`plans/rca-arms.md` §1,
class `floor-control`). Both are fixed here before any ranking is read; neither learns anything.
Changes are dated addenda at the end.

## 1. `control/random` — the analytic random ranking

A uniformly random ranking of the n candidate services places the root cause among the first k
with probability min(k, n) / n, so AC@k = min(k, n) / n and Avg@5 is the mean over k = 1 … 5.
n = the deployed services of the corpus (`services.json`: BFF, backend and external services;
the synthetic client excluded), the same universe every method is scored over. Computed by
`tracebench.score_rca.random_expectation` and reported beside every table; never sampled, so
it carries no interval. Per rung: xs n = 5, s n = 12, m n = 52 (one external service each).

## 2. `control/rule` — the three-rule alert counter (**OPEN: proposed text, owner confirms — publication plan Q8**)

The trivial ranker in the spirit of a three-rule alerting policy, on the same exported series
a method reads, with the fault window's injection time as the split (the same information
`rca/baro` receives):

- **Series.** Per service, three of the four exported series: `latency-95`, `error` and `count`
  (`latency-50` is not used: a median alert duplicates the p95 one in this feed).
- **Baseline.** Per series, the mean and standard deviation of the pre-fault rows
  (`time < inject_time`, after the suite's fill rule), as a classic z-score alert; a series with
  zero standard deviation never alerts.
- **Alert.** A series alerts at the first post-fault bin (`time ≥ inject_time`) whose value lies
  more than 3 standard deviations from the baseline mean, in either direction (a count may fall
  under a crash); the first alerting bin is its alert time.
- **Ranking.** Services ranked by the number of alerting series (3 … 0), descending; ties broken
  by the earliest alert time among the service's alerting series; remaining ties by service name
  (the export's order). Every service appears in the ranking.
- **Reported.** AC@1 / 3 / 5 and Avg@5 with the same bootstrap as the methods; the number of
  alerting services per case beside it.

Not chosen (recorded so the choice is not re-made on outcomes): robust (median / IQR) baselines
— that is RobustScorer itself; a threshold other than 3; the median latency as a fourth rule;
ranking by the largest z-score (that is the n-sigma method of the BARO paper, not a control).

## Addenda

(none)

### 2026-10-22 — §2 confirmed by the owner (publication plan Q8; before any `control/rule` run)

- **`control/rule` is registered exactly as §2 states it**, and the **OPEN** mark on §2 is
  superseded. Per service, three exported series — `latency-95`, `error`, `count`; per series a
  mean / standard-deviation baseline over the pre-fault rows (after the suite's fill rule); a series
  alerts at the first post-fault bin more than 3 standard deviations from its baseline mean, in
  either direction; a zero-standard-deviation series never alerts. Services are ranked by the
  number of alerting series, ties by the earliest alert time among the service's alerting series,
  then by service name; every service appears in the ranking. Reported as AC@1 / 3 / 5 and Avg@5
  with the methods' bootstrap (B = 2000, seed 0), with the number of alerting services per case.
- The "not chosen" list of §2 stays as written. `plans/rca-arms.md` §6 H-rca-rule is scored once
  the row has landed on xs, s and m × five seeds.
