#!/usr/bin/env bash
# seq2cause-bench — the xl RUNG of the root-cause work, seed 2 (one box per seed; the five scripts rca-xl-s0 … s4-26-11-10.sh differ by
# the seed only); generated 2026-11-10 from the l rung's job rca-l-s2-26-11-08.sh — the same chain one rung up. plans/rca-arms.md
# addendum 2026-11-10 (D-TBR-24 … 26): pull (tier rca, from the public dataset host) → export at 180 s (N = 2 as measured; the
# xl fault schedule has l's boundaries, so the same three of ten are floored to their bin, offsets recorded; --expect-cases none:
# no landed xl export) → the case-level broadcast sources of the xl row set: rankrcd (RCD, 10 runs per case, the pinned env built by
# the replica's setup_command), rank rca/baro, rank control/rule → the per-trace probe of the two keep arms in ONE layout (every xl
# context freeze has both keep cells at c = 1, N = 8, g = 1: the information-gain arm probes with --probe-arms both, the CMI arm
# re-ranks the shared --work-dir, `runner_reused`; --axis trace --select post-nonok --cap 0 = rule R, every failed request trace
# wholly inside the fault window; every reading written per trace, --reading hop0 the primary) → control/trace-earliest per trace
# (D-TBR-19) → the raw shards deleted. Staging: every xl row names the l ledger entry of the same arm and stage (form ledger; landed
# 2026-11-09). The backbone is an input artifact (input/xl-s2/model), hash-checked against the context freeze. Probes on the A10G
# (--device cuda): a disclosed GPU replica of the frozen cells (D-TBR-15's class; no CPU row of these cells). No hypothesis is
# registered at xl: the rows are reported beside m and l. Scoring and the landing run off the box.
# Environments (as at l since 2026-11-09): the chain runs in the provisioning `train` environment (Python 3.11, the package without
# extras); `rank --arm rca/baro` runs under $BP = the harness's own method environment `trace-bench-rca` built by
# scripts/envs/rca-env.sh (Python 3.12, the baro + rcaeval extras at their pins), as rankrcd drives the RCD environment through
# --rcd-python $RP.
# Last line: the sibling's artifacts manifest. One command per line (scripts/tfvars_check.py parses every tracebenchrca command
# with the real parsers, the $BP line included); the runner's cwd is benchmarks/trace-bench-rca; set -e stops at the first failure.
set -euo pipefail
TP=$HOME/miniforge3/envs/seq2cause-trace-bench/bin/python
RP=$HOME/miniforge3/envs/rcaeval-rcd-259ea41/bin/python
BP=$HOME/miniforge3/envs/trace-bench-rca/bin/python
HR=../..
L=../trace-bench/plans/staging-ledger.json

python -m tracebenchrca.pull --repo chadyuk/trace-bench --revision v0.3.0 --rung xl --variant latent --seed 2 --tier rca --dest corpora --output-folder out/rca-xl/xl-s2/pull-rca-xl
python -m tracebenchrca.export --corpus-dir corpora/instances/xl/latent/seed=2 --rung xl --out exports/xl-s2 --workers 4 --extras --expect-cases none --output-folder out/rca-xl/xl-s2/export-xl
python -m tracebenchrca.rankrcd --export-record out/rca-xl/xl-s2/export-xl --rung xl --rcd-python $RP --workers 4 --work-dir out/rca-xl/xl-s2/work/rcd --prior-rung-record 64286cbc3b34fd3e521ce97165a02940c8563430a30a216c4447f49aa8c69375 --staging-ledger $L --output-folder out/rca-xl/xl-s2/rca-rcd/rank
$BP -m tracebenchrca.rank --arm rca/baro --export-record out/rca-xl/xl-s2/export-xl --rung xl --prior-rung-record 72c7c120f8fe9cb466f75c83ef1633feaa173e00b0bb464a4ee0db3f4cc45cef --staging-ledger $L --output-folder out/rca-xl/xl-s2/rca-baro/rank
python -m tracebenchrca.rank --arm control/rule --export-record out/rca-xl/xl-s2/export-xl --rung xl --prior-rung-record 1e77159d60bd82c3d6ed052f9fab6d2b266a05cb52afc2ac325d3db9a1142f7c --staging-ledger $L --output-folder out/rca-xl/xl-s2/control-rule/rank
python -m tracebenchrca.ranktrace --arm rca/trace-ig-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-xl/xl-s2/export-xl --corpus-dir corpora/instances/xl/latent/seed=2 --model-dir input/xl-s2/model --freeze ../trace-bench/freezes/2026-11-03-xl-latent-s2-context.json --rung xl --seed 2 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-xl/xl-s2/work/core-keep --prior-rung-record 84f1a9a962260e5c84e2690b1f579018b14836424907e5b4f20ef212eb2293ea --staging-ledger $L --output-folder out/rca-xl/xl-s2/rca-trace-ig-keep/trace/rank
python -m tracebenchrca.ranktrace --arm rca/trace-cmi-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-xl/xl-s2/export-xl --corpus-dir corpora/instances/xl/latent/seed=2 --model-dir input/xl-s2/model --freeze ../trace-bench/freezes/2026-11-03-xl-latent-s2-context.json --rung xl --seed 2 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-xl/xl-s2/work/core-keep --prior-rung-record 727c40c52d288464172678cb536800e0533c821402d2ccb952288095d9154cd5 --staging-ledger $L --output-folder out/rca-xl/xl-s2/rca-trace-cmi-keep/trace/rank
python -m tracebenchrca.earliest --axis trace --select post-nonok --cap 0 --export-record out/rca-xl/xl-s2/export-xl --corpus-dir corpora/instances/xl/latent/seed=2 --rung xl --seed 2 --prior-rung-record d6c44866143d244f7c0fe1b778b82aefb9393a3bd2079b0ec68e3e363a7cfedd --staging-ledger $L --output-folder out/rca-xl/xl-s2/control-trace-earliest/trace/rank
rm -rf corpora/instances/xl/latent/seed=2/raw
python -m tracebenchrca.artifacts manifest out
