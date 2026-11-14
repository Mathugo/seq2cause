#!/usr/bin/env bash
# seq2cause-bench — the l RUNG of the root-cause work, seed 0 (one box per seed; the five scripts rca-l-s0 … s4-26-11-08.sh differ by the
# seed only); generated 2026-11-08 from the per-trace job rca-ctx-trace-26-11-06.sh (the box form). plans/rca-arms.md addendum
# 2026-11-08 (D-TBR-21 … 23): pull (tier rca, from the public dataset host) → export at the relaxed rule's 180 s width (N = 5; three
# of l's ten fault boundaries floored to their bin, offsets recorded; --expect-cases none: no landed l export) → the case-level
# broadcast sources of the l row set: rankrcd (RCD, 10 runs per case, the pinned env built by the replica's setup_command), rank
# rca/baro, rank control/rule → the per-trace probe of the two keep arms in ONE layout (every l context freeze has both keep cells at
# c = 1, N = 8, g = 1: the information-gain arm probes with --probe-arms both, the CMI arm re-ranks the shared --work-dir,
# `runner_reused`; --axis trace --select post-nonok --cap 0 = rule R, every failed request trace wholly inside the fault window; every
# reading written per trace, --reading hop0 the primary) → control/trace-earliest per trace (D-TBR-19) → the raw shards deleted.
# Staging: every l row names the m ledger entry of the same arm and stage (form ledger; the entries landed by 2026-11-08). The
# backbone is an input artifact (input/l-s0/model), hash-checked against the context freeze. Probes on the A10G (--device cuda): a
# disclosed GPU replica of the frozen cells (D-TBR-15's class; no CPU row of these cells). Scoring and the landing run locally.
# Environments (2026-11-09, after the second apply stopped at rca/baro on every box): the chain runs in the provisioning `train`
# environment (Python 3.11, the package without extras); `rank --arm rca/baro` runs under $BP = the harness's own method
# environment `trace-bench-rca` built by scripts/envs/rca-env.sh (Python 3.12, the baro + rcaeval extras at their pins — on 3.11
# RCAEval's e2e package cannot be imported), as rankrcd drives the RCD environment through --rcd-python $RP.
# Last line: the sibling's artifacts manifest. One command per line (scripts/tfvars_check.py parses every tracebenchrca command
# with the real parsers, the $BP line included); the runner's cwd is benchmarks/trace-bench-rca; set -e stops at the first failure.
set -euo pipefail
TP=$HOME/miniforge3/envs/seq2cause-trace-bench/bin/python
RP=$HOME/miniforge3/envs/rcaeval-rcd-259ea41/bin/python
BP=$HOME/miniforge3/envs/trace-bench-rca/bin/python
HR=../..
L=../trace-bench/plans/staging-ledger.json

python -m tracebenchrca.pull --repo chadyuk/trace-bench --revision v0.3.0 --rung l --variant latent --seed 0 --tier rca --dest corpora --output-folder out/rca-l/l-s0/pull-rca-l
python -m tracebenchrca.export --corpus-dir corpora/instances/l/latent/seed=0 --rung l --out exports/l-s0 --workers 4 --extras --expect-cases none --output-folder out/rca-l/l-s0/export-l
python -m tracebenchrca.rankrcd --export-record out/rca-l/l-s0/export-l --rung l --rcd-python $RP --workers 4 --work-dir out/rca-l/l-s0/work/rcd --prior-rung-record 9fe9b7a4b83159924998991418dd65f24a34daa6858dc38b978eb7b5611c89ba --staging-ledger $L --output-folder out/rca-l/l-s0/rca-rcd/rank
$BP -m tracebenchrca.rank --arm rca/baro --export-record out/rca-l/l-s0/export-l --rung l --prior-rung-record a1e8c09e8744fb8d3ab50750f3c1b566e7befb41460f779bf62988ef76c34440 --staging-ledger $L --output-folder out/rca-l/l-s0/rca-baro/rank
python -m tracebenchrca.rank --arm control/rule --export-record out/rca-l/l-s0/export-l --rung l --prior-rung-record aeac7bb8e857006817735b6a6ee28569a71f599e43d9bef9a11e43802aeffb86 --staging-ledger $L --output-folder out/rca-l/l-s0/control-rule/rank
python -m tracebenchrca.ranktrace --arm rca/trace-ig-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-l/l-s0/export-l --corpus-dir corpora/instances/l/latent/seed=0 --model-dir input/l-s0/model --freeze ../trace-bench/freezes/2026-11-02-l-latent-s0-context.json --rung l --seed 0 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-l/l-s0/work/core-keep --prior-rung-record 54282475c25d61b40e3a9701815dcda9e73d78210cd0371ac681a51a3cb7f1de --staging-ledger $L --output-folder out/rca-l/l-s0/rca-trace-ig-keep/trace/rank
python -m tracebenchrca.ranktrace --arm rca/trace-cmi-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-l/l-s0/export-l --corpus-dir corpora/instances/l/latent/seed=0 --model-dir input/l-s0/model --freeze ../trace-bench/freezes/2026-11-02-l-latent-s0-context.json --rung l --seed 0 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-l/l-s0/work/core-keep --prior-rung-record edd03c14a5d1303a921afd06a75f5bce4364e70e8c88f09cadf09461dab25c45 --staging-ledger $L --output-folder out/rca-l/l-s0/rca-trace-cmi-keep/trace/rank
python -m tracebenchrca.earliest --axis trace --select post-nonok --cap 0 --export-record out/rca-l/l-s0/export-l --corpus-dir corpora/instances/l/latent/seed=0 --rung l --seed 0 --prior-rung-record 8064e4b7ae927dd0a5776fb189883e1c2fbb70abeb7a507ed5e0ff2cdc05061b --staging-ledger $L --output-folder out/rca-l/l-s0/control-trace-earliest/trace/rank
rm -rf corpora/instances/l/latent/seed=0/raw
python -m tracebenchrca.artifacts manifest out
