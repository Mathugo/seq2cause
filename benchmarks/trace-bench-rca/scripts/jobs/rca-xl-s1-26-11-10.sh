#!/usr/bin/env bash
# seq2cause-bench — the xl RUNG of the root-cause work, seed 1 (one box per seed; the five scripts rca-xl-s0 … s4-26-11-10.sh differ by
# the seed only); generated 2026-11-10 from the l rung's job rca-l-s1-26-11-08.sh — the same chain one rung up. plans/rca-arms.md
# addendum 2026-11-10 (D-TBR-24 … 26): pull (tier rca, from the public dataset host) → export at 180 s (N = 2 as measured; the
# xl fault schedule has l's boundaries, so the same three of ten are floored to their bin, offsets recorded; --expect-cases none:
# no landed xl export) → the case-level broadcast sources of the xl row set: rankrcd (RCD, 10 runs per case, the pinned env built by
# the replica's setup_command), rank rca/baro, rank control/rule → the per-trace probe of the two keep arms in ONE layout (every xl
# context freeze has both keep cells at c = 1, N = 8, g = 1: the information-gain arm probes with --probe-arms both, the CMI arm
# re-ranks the shared --work-dir, `runner_reused`; --axis trace --select post-nonok --cap 0 = rule R, every failed request trace
# wholly inside the fault window; every reading written per trace, --reading hop0 the primary) → control/trace-earliest per trace
# (D-TBR-19) → the raw shards deleted. Staging: every xl row names the l ledger entry of the same arm and stage (form ledger; landed
# 2026-11-09). The backbone is an input artifact (input/xl-s1/model), hash-checked against the context freeze. Probes on the A10G
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

python -m tracebenchrca.pull --repo chadyuk/trace-bench --revision v0.3.0 --rung xl --variant latent --seed 1 --tier rca --dest corpora --output-folder out/rca-xl/xl-s1/pull-rca-xl
python -m tracebenchrca.export --corpus-dir corpora/instances/xl/latent/seed=1 --rung xl --out exports/xl-s1 --workers 4 --extras --expect-cases none --output-folder out/rca-xl/xl-s1/export-xl
python -m tracebenchrca.rankrcd --export-record out/rca-xl/xl-s1/export-xl --rung xl --rcd-python $RP --workers 4 --work-dir out/rca-xl/xl-s1/work/rcd --prior-rung-record 9a4e2d462934662cba7194f12c1b624b03d400e6421d53ba47369e85788b321c --staging-ledger $L --output-folder out/rca-xl/xl-s1/rca-rcd/rank
$BP -m tracebenchrca.rank --arm rca/baro --export-record out/rca-xl/xl-s1/export-xl --rung xl --prior-rung-record 03343003ebc600b5b7bcfbc912271af61b36171bd8b97426f504b60a30c7e0a7 --staging-ledger $L --output-folder out/rca-xl/xl-s1/rca-baro/rank
python -m tracebenchrca.rank --arm control/rule --export-record out/rca-xl/xl-s1/export-xl --rung xl --prior-rung-record 567c4cf5f4847ee55d9c11fd5a4a5837c48d765bf0a12e5cc0ca86bd09ede2da --staging-ledger $L --output-folder out/rca-xl/xl-s1/control-rule/rank
python -m tracebenchrca.ranktrace --arm rca/trace-ig-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-xl/xl-s1/export-xl --corpus-dir corpora/instances/xl/latent/seed=1 --model-dir input/xl-s1/model --freeze ../trace-bench/freezes/2026-11-03-xl-latent-s1-context.json --rung xl --seed 1 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-xl/xl-s1/work/core-keep --prior-rung-record fddf56660c56030a76a445bab581f86ababeeb54e13562577126a78fad89c994 --staging-ledger $L --output-folder out/rca-xl/xl-s1/rca-trace-ig-keep/trace/rank
python -m tracebenchrca.ranktrace --arm rca/trace-cmi-keep --probe-arms rca/trace-ig-keep rca/trace-cmi-keep --reading hop0 --phase post --subtract none --pre-record none --axis trace --select post-nonok --cap 0 --export-record out/rca-xl/xl-s1/export-xl --corpus-dir corpora/instances/xl/latent/seed=1 --model-dir input/xl-s1/model --freeze ../trace-bench/freezes/2026-11-03-xl-latent-s1-context.json --rung xl --seed 1 --trace-python $TP --harness-root $HR --device cuda --work-dir out/rca-xl/xl-s1/work/core-keep --prior-rung-record e9a78667c66d6319d64ec7cf64c008f7d05cdbc73d9bb82ff110bcc602c2f815 --staging-ledger $L --output-folder out/rca-xl/xl-s1/rca-trace-cmi-keep/trace/rank
python -m tracebenchrca.earliest --axis trace --select post-nonok --cap 0 --export-record out/rca-xl/xl-s1/export-xl --corpus-dir corpora/instances/xl/latent/seed=1 --rung xl --seed 1 --prior-rung-record 64e57a7c74428cb910a3ca19ce1f2dc13540fb35bdc1d2f26dc0b380fa8d50ca --staging-ledger $L --output-folder out/rca-xl/xl-s1/control-trace-earliest/trace/rank
rm -rf corpora/instances/xl/latent/seed=1/raw
python -m tracebenchrca.artifacts manifest out
