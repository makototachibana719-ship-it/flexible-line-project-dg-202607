#!/usr/bin/env python3
"""One-update Peg JAX/XLA differential probe; never trains formally or saves checkpoints."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/peg_insertion/peg_rlpd_baseline_seed0.yaml"
CANDIDATE = ROOT / "results/checkpoints/peg_rlpd_gpu_formal_candidate_seed0/checkpoint_0"
EXAMPLE = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
ENV_PARENT = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
FIELDS = ("actions", "dones", "masks", "next_observations", "observations", "rewards")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def array_signature(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.all(np.isfinite(array))) if np.issubdtype(array.dtype, np.number) else True,
    }


def prepare_batch(batch_path: Path, result_dir: Path) -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    metadata_path = CANDIDATE.parent / f"{CANDIDATE.name}_resume_metadata.json"
    replay_path = CANDIDATE.parent / f"{CANDIDATE.name}_online_replay.pkl"
    metadata = json.loads(metadata_path.read_text())
    if metadata["online_replay_sha256"] != sha256(replay_path):
        raise ValueError("candidate replay hash mismatch")
    demo_path = ROOT / cfg["demo"]["path"]
    if sha256(demo_path) != cfg["demo"]["sha256"]:
        raise ValueError("Demo hash mismatch")
    replay = pickle.loads(replay_path.read_bytes())
    demos = pickle.loads(demo_path.read_bytes())
    if int(replay["size"]) != 121 or len(demos) != 5512:
        raise ValueError("replay population mismatch")
    generator = np.random.default_rng(20260903)
    online_indices = generator.choice(int(replay["size"]), size=32, replace=False)
    demo_indices = generator.choice(len(demos), size=32, replace=False)
    batch = {}
    for field in FIELDS:
        online = np.asarray(replay["dataset_dict"][field])[online_indices]
        demo = np.stack([np.asarray(demos[int(index)][field]) for index in demo_indices])
        if online.shape[1:] != demo.shape[1:] or online.dtype != demo.dtype:
            raise ValueError(f"batch contract mismatch: {field}")
        batch[field] = np.concatenate((online, demo), axis=0)
    if batch["observations"].shape != (64, 20) or batch["actions"].shape != (64, 7):
        raise ValueError("frozen RLPD batch shape mismatch")
    np.savez(batch_path, **batch)
    write_json(
        result_dir / "batch_manifest.json",
        {
            "batch_file": str(batch_path),
            "batch_sha256": sha256(batch_path),
            "demo_sha256": sha256(demo_path),
            "online_replay_sha256": sha256(replay_path),
            "online_indices": online_indices.tolist(),
            "demo_indices": demo_indices.tolist(),
            "fields": {key: array_signature(value) for key, value in batch.items()},
            "composition": "32 online + 32 demo",
        },
    )
    print("PEG_DIFFERENTIAL_BATCH_PREPARE=PASS")
    print("BATCH_SHA256:", sha256(batch_path))


def gpu_memory() -> dict:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    return {"raw": output}


def tree_signature(jax, tree: object) -> dict:
    pairs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    leaves = []
    for path, leaf in pairs:
        shape = list(getattr(leaf, "shape", ()))
        dtype = str(getattr(leaf, "dtype", None))
        device = getattr(leaf, "device", None)
        if callable(device):
            device = device()
        sharding = getattr(leaf, "sharding", None)
        host = np.asarray(jax.device_get(leaf)) if hasattr(leaf, "dtype") else None
        leaves.append(
            {
                "path": str(path),
                "type": type(leaf).__name__,
                "shape": shape,
                "dtype": dtype,
                "device": str(device),
                "sharding_type": type(sharding).__name__ if sharding is not None else None,
                "sharding": str(sharding),
                "finite": bool(np.all(np.isfinite(host))) if host is not None and np.issubdtype(host.dtype, np.number) else True,
            }
        )
    return {"treedef": str(treedef), "leaf_count": len(leaves), "leaves": leaves}


def run_probe(mode: str, batch_path: Path, run_dir: Path, timeout_seconds: int) -> None:
    import jax
    from flax.core import freeze

    sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

    phase = {"value": "bootstrap"}
    stack_timers = []
    watchdog = None

    def dump_stack(delay: int) -> None:
        print(f"[STACK] delay={delay}s phase={phase['value']}", file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def hard_timeout() -> None:
        print(f"PEG_DIFFERENTIAL_{mode}_TIMEOUT phase={phase['value']} limit={timeout_seconds}s", file=sys.stderr, flush=True)
        os._exit(124)

    try:
        with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
            environment = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
            agent = make_sac_agent(
                seed=0,
                sample_obs=environment.observation_space.sample(),
                sample_action=environment.action_space.sample(),
            )
            environment.close()
            if mode == "B":
                agent = agent.replace(state=restore_fresh_state(CANDIDATE.parent, agent.state, 0))
            batch_archive = np.load(batch_path, allow_pickle=False)
            host_batch = {field: np.asarray(batch_archive[field]) for field in FIELDS}
            batch_archive.close()
            batch = freeze(jax.tree_util.tree_map(jax.device_put, host_batch))
            phase["value"] = "preflight_state_sync"
            state_signature = tree_signature(jax, agent.state)
            batch_signature = tree_signature(jax, batch)
            optimizer_leaves = [
                leaf for leaf in state_signature["leaves"] if "opt" in leaf["path"].lower()
            ]
            preflight = {
                "mode": mode,
                "pid": os.getpid(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "batch_sha256": sha256(batch_path),
                "batch_signature": batch_signature,
                "agent_state_signature": state_signature,
                "optimizer_state_structure": optimizer_leaves,
                "rng": np.asarray(jax.device_get(agent.state.rng), dtype=np.uint32).tolist(),
                "gpu_memory_before": gpu_memory(),
                "xla_flags": os.environ.get("XLA_FLAGS"),
                "jax_log_compiles": os.environ.get("JAX_LOG_COMPILES"),
                "checkpoint_restored": mode == "B",
            }
            write_json(run_dir / "preflight.json", preflight)
            print("PROBE_MODE:", mode, flush=True)
            print("BATCH_SHA256:", preflight["batch_sha256"], flush=True)
            print("STATE_LEAVES:", state_signature["leaf_count"], flush=True)
            print("OPTIMIZER_LEAVES:", len(optimizer_leaves), flush=True)
            print("RNG_BEFORE:", preflight["rng"], flush=True)
            phase["value"] = "dispatch"
            for delay in (30, 90):
                timer = threading.Timer(delay, dump_stack, args=(delay,))
                timer.daemon = True
                timer.start()
                stack_timers.append(timer)
            watchdog = threading.Timer(timeout_seconds, hard_timeout)
            watchdog.daemon = True
            watchdog.start()
            true_start = time.perf_counter()
            dispatch_start = time.perf_counter()
            new_agent, update_info = agent.update_high_utd(batch, utd_ratio=1)
            dispatch_seconds = time.perf_counter() - dispatch_start
            print(f"DISPATCH_SECONDS: {dispatch_seconds:.9f}", flush=True)
            phase["value"] = "full_block_until_ready"
            sync_start = time.perf_counter()
            for leaf in jax.tree_util.tree_leaves((new_agent, update_info)):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()
            sync_seconds = time.perf_counter() - sync_start
            true_seconds = time.perf_counter() - true_start
            phase["value"] = "postflight"
            post_signature = tree_signature(jax, new_agent.state)
            losses = {}
            for path, leaf in jax.tree_util.tree_flatten_with_path(update_info)[0]:
                array = np.asarray(jax.device_get(leaf))
                if "loss" in str(path).lower() and array.size == 1:
                    losses[str(path)] = float(array.reshape(()))
            result = {
                "mode": mode,
                "status": "PASS",
                "dispatch_seconds": dispatch_seconds,
                "sync_seconds": sync_seconds,
                "true_update_seconds": true_seconds,
                "rng_before": preflight["rng"],
                "rng_after": np.asarray(jax.device_get(new_agent.state.rng), dtype=np.uint32).tolist(),
                "losses": losses,
                "losses_finite": bool(losses) and all(np.isfinite(value) for value in losses.values()),
                "agent_state_after_signature": post_signature,
                "gpu_memory_after": gpu_memory(),
            }
            write_json(run_dir / "result.json", result)
            print(f"SYNC_SECONDS: {sync_seconds:.9f}", flush=True)
            print(f"TRUE_UPDATE_SECONDS: {true_seconds:.9f}", flush=True)
            print(f"PEG_DIFFERENTIAL_{mode}=PASS", flush=True)
    finally:
        phase["value"] = "cleanup"
        if watchdog is not None:
            watchdog.cancel()
        for timer in stack_timers:
            timer.cancel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "A", "B"), required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=130)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    batch_path = args.batch.resolve()
    if ROOT not in result_dir.parents or ROOT not in batch_path.parents:
        raise ValueError("diagnostic output outside project")
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        if batch_path.exists():
            raise FileExistsError(batch_path)
        prepare_batch(batch_path, result_dir)
        return 0
    run_dir = result_dir / args.mode
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    run_probe(args.mode, batch_path, run_dir, args.timeout_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
