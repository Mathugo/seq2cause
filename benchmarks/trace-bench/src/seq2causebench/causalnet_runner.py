"""Run CausalNET once on an events file (plans/causalnet.md §2; D-SB-29). Executed by the method's
own interpreter (the `causalnet` conda environment: torch 2.x, omegaconf, pandas), never imported
by the harness; it puts `<method-src>/model` at the head of `sys.path` and imports the pinned,
patched clone from there (`algos/causalnet-zhu-2024/CausalNET` at `aaccce6` with the lab's patches
`0001-collapse-decay`, `0002-drivable-entry` and `0003-init-off`, recorded in that directory's RUN.md):

    <causalnet python> -I causalnet_runner.py --method-src <clone> --events <events.npz> \\
        --allowed <pairs.npy | none> --setting <window_ms> --method-args k=v ... --out <scores.npy>

`events.npz` holds `event` (dense type ids), `timestamp` (integer milliseconds, sorted), `node`
(the dense op id of every event: node = op), `n_ops` and `op_topology` (undirected op adjacency).
The runner writes the clone's dataset format next to `<out>` (`alarm.csv` with `alarm_id` = type,
`device_id` = node, `start_timestamp` = `end_timestamp` = the millisecond time; `topology.npy` =
the op adjacency; `causal_prior.npy` = 1 on an allowed pair, 0 elsewhere, −1 everywhere without a
mask; no `true_graph.npy`), loads it through the clone's own `get_formated_dataset`, trains
in-process with the shipped `config_m24.yaml` (`max_time_lag` = the setting in milliseconds, the
collapsed decay tensor, the mask edge init, batches streamed per epoch, no prune, no figures) and
saves the clone's unpruned read-out `final_prob` — `p_ij · max_{k,u,v: A_k[u,v] ≠ 0} σ(θ)` (row =
cause), already in [0, 1] — as the float32 score matrix with the diagonal zeroed, plus the sidecar
JSON (`n_types`, `n_events`, `n_allowed_pairs`, `n_nonzero`, `wall_seconds`, `setting`,
`method_args` (every value, defaults filled), `method_version` (the clone's commit and the patch
names), `method_env` (the `pip freeze` lines of this interpreter), `epochs_run`, `device`, the
per-epoch wall clock and the last epoch's loss terms).

Method-args (`k=v`, every default recorded): `epochs` (200, the shipped `total_epoch`), `device`
(`auto` = cuda, else mps, else cpu), `seed` (42, the shipped config), `max_hop` (1), `readout`
(`shipped` = the saved `final_prob`; `gated` = the paper's Eq. 16 over the trained slots only),
`time_scale` (`auto` = setting / 2: the shipped 120 s window reached the model as 120 / 60 = 2
encoded units, and a window of ξ ms reaches it as the same 2 units), `graph_init` (`mask`; `zero`
= the shipped init), `batch_size` (512), `init_off` (−1: the logit of every off-mask pair at the
mask init, D-CN-4; the on-mask logit is 0; `plans/causalnet.md` addendum 2026-11-07, D-CN-10).

Test-only stub path: when `<method-src>/model/STUB_RUNNER.py` exists the runner imports it instead
of the clone and returns `fit(events, allowed, setting, method_args)`; the harness's own tests use
it to run the chain without torch. A real clone never carries that file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PATCHES = ("0001-collapse-decay", "0002-drivable-entry", "0003-init-off")
STUB_FILE = "STUB_RUNNER.py"
DEFAULTS = {
    "epochs": "200",
    "device": "auto",
    "seed": "42",
    "max_hop": "1",
    "readout": "shipped",
    "time_scale": "auto",
    "graph_init": "mask",
    "batch_size": "512",
    "init_off": "-1",
}


def pip_freeze():
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        return [ln for ln in out.splitlines() if ln.strip()] or ["<empty pip freeze>"]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [f"<pip freeze unavailable: {type(e).__name__}>"]


def git_head(root):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_method_args(items):
    args = dict(DEFAULTS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"method-args entries are KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if k not in DEFAULTS:
            raise ValueError(f"unknown method-arg {k!r}; known: {sorted(DEFAULTS)}")
        args[k] = v
    return args


def pick_device(torch, name):
    """`auto` = cuda if available, else mps if a small op runs there, else cpu."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:
            x = torch.ones(2, 2, device="mps")
            torch.matrix_exp(x).sum().item()
            return "mps"
        except Exception:  # noqa: BLE001 - any failure means the backend is unusable here
            return "cpu"
    return "cpu"


def write_dataset(root, name, events, allowed):
    """The clone's dataset directory for one fit (see the module docstring)."""
    import numpy as np
    import pandas as pd

    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    n_types = int(events["event"].max()) + 1
    pd.DataFrame(
        {
            "alarm_id": events["event"].astype(np.int64),
            "device_id": events["node"].astype(np.int64),
            "start_timestamp": events["timestamp"].astype(np.int64),
            "end_timestamp": events["timestamp"].astype(np.int64),
        }
    ).to_csv(d / "alarm.csv", index=False)
    np.save(d / "topology.npy", np.asarray(events["op_topology"], dtype=np.float64))
    if allowed is None:
        prior = np.full((n_types, n_types), -1, dtype=np.int64)
    else:
        prior = np.zeros((n_types, n_types), dtype=np.int64)
        prior[allowed[:, 0], allowed[:, 1]] = 1
        np.fill_diagonal(prior, 0)
    np.save(d / "causal_prior.npy", prior)
    return d, n_types


def fit_real(method_src, events, allowed, setting, margs, out):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(Path(method_src) / "model"))
    import main as cn  # the clone's entry module (patched)
    from utils.util_dataset import get_formated_dataset

    if not hasattr(cn, "node_of_type_from_data") or not hasattr(cn.CausalNET, "final_prob"):
        raise RuntimeError(f"the clone under {method_src} lacks the lab patches {PATCHES}")
    device = pick_device(torch, margs["device"])
    epochs = int(margs["epochs"])
    time_scale = (
        float(setting) / 2.0 if margs["time_scale"] == "auto" else float(margs["time_scale"])
    )
    data_root = Path(out).parent / "causalnet-data"
    name = f"N{int(setting)}"
    _d, n_types = write_dataset(data_root, name, events, allowed)
    data, topology, _true_cm, prior_cm, num_types = get_formated_dataset(str(data_root) + "/", name)
    if int(num_types) != n_types:
        raise RuntimeError(f"the clone read {num_types} types, the events hold {n_types}")
    node_of_type = cn.node_of_type_from_data(data, num_types)

    opt = OmegaConf.load(str(Path(method_src) / "model" / "configs" / "config_m24.yaml"))
    opt.model.dataset.n_nodes = int(num_types)
    opt.model.dataset.n_topos = int(topology.shape[0])
    opt.model.dataset.max_hop = int(margs["max_hop"])
    opt.model.dataset.max_time_lag = int(setting)
    opt.model.dataset.time_scale = time_scale
    opt.model.dataset.theta_mode = "collapsed"
    opt.model.graph_init = margs["graph_init"] if allowed is not None else "zero"
    opt.model.readout = margs["readout"]
    opt.model.graph_init_off = float(margs["init_off"])
    opt.model.batch_size = int(margs["batch_size"])
    opt.model.total_epoch = epochs
    opt.model.show_graph_every = 0

    class NoLog:
        def log_metrics(self, *a, **k):
            pass

        def log_figures(self, *a, **k):
            pass

    cn.reproduc(
        int(margs["seed"])
    )  # after the data load: `read_dag` draws when true_graph.npy is absent
    model = cn.CausalNET(
        num_types, topology, prior_cm, opt.model, NoLog(), device=device, node_of_type=node_of_type
    )
    epoch_wall = []
    t_last = [time.monotonic()]

    def on_epoch(i, facts):
        now = time.monotonic()
        epoch_wall.append(now - t_last[0])
        t_last[0] = now
        print(
            f"epoch {i + 1}/{epochs} wall={epoch_wall[-1]:.1f}s batches={facts.get('n_batches')} "
            f"s1_loss={facts.get('latent_data_pred/data_pred_loss'):.3f} "
            f"s2_loss={facts.get('graph_discov/total_loss'):.3f} "
            f"sparsity={facts.get('graph_discov/sparsity_loss'):.4f} dag={facts.get('graph_discov/dag_loss'):.4f}",
            flush=True,
        )

    final, last = model.train(
        name, data, topology, None, stream=True, on_epoch=on_epoch, save=False
    )
    scores = np.asarray(final, dtype=np.float64)
    if scores.shape != (n_types, n_types):
        raise RuntimeError(f"the read-out is {scores.shape}, expected {(n_types, n_types)}")
    facts = {
        "epochs_run": epochs,
        "device": device,
        "torch": torch.__version__,
        "time_scale": time_scale,
        "n_batches_per_epoch": int(last.get("n_batches", 0)),
        "epoch_wall_seconds": epoch_wall,
        "final_loss_terms": {k: v for k, v in last.items() if k != "n_batches"},
        "theta_shape": list(model.theta.shape),
        "graph_init": str(opt.model.graph_init),
        "readout": str(opt.model.readout),
        "init_off": float(opt.model.graph_init_off),
    }
    return scores, facts


def fit_stub(method_src, events, allowed, setting, margs):
    """Test-only: `<method-src>/model/STUB_RUNNER.py::fit(events, allowed, setting, method_args)`."""
    spec = importlib.util.spec_from_file_location(
        "STUB_RUNNER", Path(method_src) / "model" / STUB_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit(events, allowed, int(setting), dict(margs)), {
        "stub": True,
        "device": "none",
        "epochs_run": 0,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method-src", required=True, help="the pinned, patched CausalNET clone")
    p.add_argument("--events", required=True)
    p.add_argument(
        "--allowed", required=True, help="an (k, 2) int64 .npy of allowed pairs, or 'none'"
    )
    p.add_argument(
        "--setting",
        required=True,
        type=int,
        help="ξ, the causal-attention window, in the events' unit (ms)",
    )
    p.add_argument("--method-args", required=True, nargs="*")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import numpy as np

    margs = parse_method_args(args.method_args)
    with np.load(args.events) as z:
        events = {k: z[k] for k in z.files}
    for key in ("event", "timestamp", "node", "n_ops", "op_topology"):
        if key not in events:
            raise ValueError(f"events.npz lacks {key!r} (the method needs nodes)")
    n_types = int(events["event"].max()) + 1 if len(events["event"]) else 0
    allowed = None
    if args.allowed != "none":
        allowed = np.load(args.allowed).astype(np.int64).reshape(-1, 2)
    t0 = time.monotonic()
    if (Path(args.method_src) / "model" / STUB_FILE).exists():
        scores, facts = fit_stub(args.method_src, events, allowed, args.setting, margs)
    else:
        scores, facts = fit_real(args.method_src, events, allowed, args.setting, margs, args.out)
    wall = time.monotonic() - t0
    scores = np.clip(np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0), 0.0, 1.0)
    np.fill_diagonal(scores, 0.0)
    scores = scores.astype(np.float32)
    np.save(args.out, scores)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_types": n_types,
                "n_events": int(len(events["event"])),
                "n_ops": int(events["n_ops"]),
                "n_allowed_pairs": None if allowed is None else int(len(allowed)),
                "n_nonzero": int((scores > 0).sum()),
                "wall_seconds": wall,
                "setting": int(args.setting),
                "method_args": margs,
                "method_version": {"commit": git_head(args.method_src), "patches": list(PATCHES)},
                "method_env": pip_freeze(),
                **facts,
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
