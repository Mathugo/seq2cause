#!/usr/bin/env bash
# seq2cause-bench — the l RUNG of the root-cause work, seed 4 (one box per seed; the five scripts rca-l-s0 … s4-26-11-08.sh differ by the
# seed only); generated 2026-11-08 from the per-trace job rca-ctx-trace-26-11-06.sh (the box form). plans/rca-arms.md addendum
# 2026-11-08 (D-TBR-21 … 23): pull (tier rca, from the public dataset host) → export at the relaxed rule's 180 s width (N = 5; three
# of l's ten fault boundaries floored to their bin, offsets recorded; --expect-cases none: no landed l export) → the case-level
# broadcast sources of the l row set: rankrcd (RCD, 10 runs per case, the pinned env built by the replica's setup_command), rank
# rca/baro, rank control/rule → the per-trace probe of the two keep arms in ONE layout (every l context freeze has both keep cells at
# c = 1, N = 8, g = 1: the information-gain arm probes with --probe-arms both, the CMI arm re-ranks the shared --work-dir,
# `runner_reused`; --axis trace --select post-nonok --cap 0 = rule R, every failed request trace wholly inside the fault window; every
# reading written per trace, --reading hop0 the primary) → control/trace-earliest per trace (D-TBR-19) → the raw shards deleted.
# Staging: every l row names the m ledger entry of the same arm and stage (form ledger; the entries landed by 2026-11-08). The
# backbone is an input artifact (input/l-s4/model), hash-checked against the context freeze. Probes on the A10G (--device cuda): a
# disclosed GPU replica of the frozen cells (D-TBR-15's class; no CPU row of these cells). Scoring and the landing run locally.
# Last line: the sibling's artifacts manifest. One command per line (scripts/tfvars_check.py parses every tracebenchrca command
# with the real parsers); the runner's cwd is benchmarks/trace-bench-rca; set -e stops at the first failure.
set -euo pipefail
TP=$HOME/miniforge3/envs/seq2cause-trace-bench/bin/python
RP=$HOME/miniforge3/envs/rcaeval-rcd-259ea41/bin/python
HR=../..
L=../trace-bench/plans/staging-ledger.json

python -m tracebenchrca.pull --repo chadyuk/trace-bench --revision v0.3.0 --rung l --variant latent --seed 4 --tier rca --dest corpora --output-folder out/rca-l/l-s4/pull-rca-l
python -m tracebenchrca.export --corpus-dir corpora/instances/l/latent/seed=4 --rung l --out exports/l-s4 --workers 4 --extras --expect-cases none --output-folder out/rca-l/l-s4/export-l
python -m tracebenchrca.rankrcd --export-record out/rca-l/l-s4/export-l --rung l --rcd-python $RP --workers 4 --work-dir out/rca-l/l-s4/work/rcd --prior-rung-record 7a05426b41b6940c71bc61419dac4aef1f2f14c5a27bb483c05fd010cdde0b11 --staging-ledger $L --output-folder out/rca-l/l-s4/rca-rcd/rank
python -m tracebenchrca.rank --arm rca/baro --export-record out/rca-l/l-s4/export-l --rung l --prior-rung-record 7e1e7fa674fd02fc3159f281dac2106a4066d4fb0b94ac9e1248733ec54ed92a --staging-ledger $L --output-folder out/rca-l/l-s4/rca-baro/rank
python -m tracebenchrca.rank --arm control/rule --export-record out/rca-l/l-s4/export-l --rung l --prior-rung-record 5afbd00db0cdbe054a45d904c1c219284b35e9858dd88f9166400a8033204fdc --staging-ledger $L --output-folder out/rca-l/l-s4/control-rule/rank
python -m tracebenchrca.ranktrace --arm rca/trace-ig-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-l/l-s4/export-l --corpus-dir corpora/instances/l/latent/seed=4 --model-dir input/l-s4/model --freeze ../trace-bench/freezes/2026-11-02-l-latent-s4-context.json --rung l --seed 4 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-l/l-s4/work/core-keep --prior-rung-record f41ecb6d62c1cd8cae6fd9de50c08f5e29f02c2210110aecfbb995a6b38d7c17 --staging-ledger $L --output-folder out/rca-l/l-s4/rca-trace-ig-keep/trace/rank
python -m tracebenchrca.ranktrace --arm rca/trace-cmi-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-l/l-s4/export-l --corpus-dir corpora/instances/l/latent/seed=4 --model-dir input/l-s4/model --freeze ../trace-bench/freezes/2026-11-02-l-latent-s4-context.json --rung l --seed 4 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-l/l-s4/work/core-keep --prior-rung-record 196a9ab725e819d896ca46cc47317ee2855a288e0a709fb95196e3a1ab4f1853 --staging-ledger $L --output-folder out/rca-l/l-s4/rca-trace-cmi-keep/trace/rank
python -m tracebenchrca.earliest --axis trace --select post-nonok --cap 0 --export-record out/rca-l/l-s4/export-l --corpus-dir corpora/instances/l/latent/seed=4 --rung l --seed 4 --prior-rung-record 15c7a14fa1b64e2c8817abcf34c05880061463f4b8440636600cf23c9da8701a --staging-ledger $L --output-folder out/rca-l/l-s4/control-trace-earliest/trace/rank
rm -rf corpora/instances/l/latent/seed=4/raw
python -m tracebenchrca.artifacts manifest out
