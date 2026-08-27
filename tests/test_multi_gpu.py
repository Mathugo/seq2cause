"""Multi-process (simulated multi-GPU) correctness test.

Spawns 2 real, independent Python processes (via `torch.multiprocessing`,
initialized as a `gloo` process group -- the same mechanism `accelerate
launch --num_processes=2 --cpu` uses under the hood) and runs
`scripts/multi_process_check.py`'s `main()` in each one. This is the
practical way to test Accelerate-based dataloader-sharding/`gather()` logic
without needing real multi-GPU hardware in CI.

This is deliberately NOT implemented by shelling out to `accelerate launch`
as a subprocess: on some platforms/environments `accelerate launch`'s own
process-spawning heuristics silently fall back to 1 process (observed on
macOS in this repo's dev environment) or `torchrun`'s rendezvous can hang --
`torch.multiprocessing.spawn` with an explicit `gloo` init is more portable
and gives a clear pass/fail (any assertion failure in a child process is
re-raised in the parent by `mp.spawn`).
"""

import os
import socket
import sys
from pathlib import Path

import pytest
import torch.multiprocessing as mp

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int) -> None:
    # Fresh interpreter (spawn start method) -- redo path setup + env vars
    # that `Accelerator()` reads to know its rank/world size.
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    import multi_process_check

    multi_process_check.main()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="torch multiprocessing stdout/stderr redirects are not supported on Windows",
)
def test_multi_process_sharding_and_gather_matches_single_process():
    """Runs `scripts/multi_process_check.py` across 2 real processes: any
    assertion failure inside a worker (bad adjacency shape, duplicated
    sequences across processes, wrong gathered size, ...) is re-raised here
    by `mp.spawn`, failing this test."""
    world_size = 2
    port = _free_port()
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)
