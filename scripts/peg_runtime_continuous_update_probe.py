#!/usr/bin/env python3
"""Fresh-agent continuous update probe with no W&B, resume, or checkpoint I/O."""
from __future__ import annotations

import argparse
import faulthandler
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

from peg_jax_xla_differential_probe import (
    CANDIDATE,
    CONFIG,
    ENV_PARENT,
    EXAMPLE,
    FIELDS,
    ROOT,
    gpu_memory,
    sha256,
    write_json,
)


EXPECTED_FIXED_BATCH_SHA256 = "854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c"
EXPECTED_ONLINE_REPLAY_SHA256 = "21fe7eeb0885b0ba3f75d1daa3633499620b7e2f6051beabeca5f5c5fa6f9561"


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rng_value(jax, state: object) -> list[int]:
    return np.asarray(jax.device_get(state.rng), dtype=np.uint32).tolist()


def block_tree(jax, tree: object) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def losses(jax, update_info: object) -> dict[str, float]:
    values = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(update_info)[0]:
        array = np.asarray(jax.device_get(leaf))
        if "loss" in str(path).lower() and array.size == 1:
            values[str(path)] = float(array.reshape(()))
    return values


def latency_classification(index: int, seconds: float) -> str:
    if index == 0:
        return "INITIAL_COMPILE"
    if seconds < 1.0:
        return "NORMAL"
    if seconds < 10.0:
        return "WARNING"
    if seconds <= 120.0:
        return "ANOMALOUS"
    return "TIMEOUT"


def batch_contract(batch: object) -> dict:
    return {
        field: {"shape": list(batch[field].shape), "dtype": str(batch[field].dtype)}
        for field in FIELDS
    }


def snapshot_30s(result_dir: Path, update_index: int, phase: dict) -> None:
    prefix = result_dir / f"update_{update_index:02d}_30s"
    with (prefix.with_suffix(".python_stack.txt")).open("w", encoding="utf-8") as stream:
        stream.write(f"update={update_index} phase={phase['value']}\n")
        faulthandler.dump_traceback(file=stream, all_threads=True)
    with (prefix.with_suffix(".gpu.txt")).open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["nvidia-smi"], stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=10, check=False
        )
    with (prefix.with_suffix(".cpu_threads.txt")).open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["ps", "-L", "-p", str(os.getpid()), "-o", "pid,tid,psr,pcpu,stat,wchan:32,comm"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False,
        )
    print(f"UPDATE_30S_SNAPSHOT: index={update_index} phase={phase['value']}", flush=True)


def populate_dynamic_buffers(environment, make_replay_buffer):
    cfg = yaml.safe_load(CONFIG.read_text())
    online_path = CANDIDATE.parent / f"{CANDIDATE.name}_online_replay.pkl"
    demo_path = ROOT / cfg["demo"]["path"]
    if sha256(online_path) != EXPECTED_ONLINE_REPLAY_SHA256:
        raise ValueError("online replay snapshot hash mismatch")
    if sha256(demo_path) != cfg["demo"]["sha256"]:
        raise ValueError("Demo hash mismatch")
    online = pickle.loads(online_path.read_bytes())
    demos = pickle.loads(demo_path.read_bytes())
    online_size = int(online["size"])
    if online_size != 121 or len(demos) != 5512:
        raise ValueError("dynamic replay population mismatch")
    online_buffer = make_replay_buffer(environment, capacity=1000000, type="replay_buffer")
    demo_buffer = make_replay_buffer(
        environment, capacity=max(10000, len(demos) + 100), type="replay_buffer"
    )
    for index in range(online_size):
        online_buffer.insert({field: np.asarray(online["dataset_dict"][field][index]) for field in FIELDS})
    for transition in demos:
        demo_buffer.insert(transition)
    return online_buffer, demo_buffer, {
        "online_size": online_size,
        "demo_size": len(demos),
        "online_sha256": sha256(online_path),
        "demo_sha256": sha256(demo_path),
    }


def run(mode: str, fixed_batch_path: Path, result_dir: Path) -> None:
    import jax
    from flax.core import freeze
    from serl_launcher.utils.launcher import make_replay_buffer

    sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
    from async_sac_state_sim import concat_rlpd_batches, make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

    updates_path = result_dir / "updates.jsonl"
    phase = {"value": "bootstrap"}
    current_update = {"value": None}
    watchdog: threading.Timer | None = None
    thirty_timer: threading.Timer | None = None

    def timeout() -> None:
        payload = {
            "status": "TIMEOUT",
            "mode": mode,
            "update_index": current_update["value"],
            "phase": phase["value"],
            "timeout_seconds": 120,
            "timestamp_ns": time.time_ns(),
        }
        write_json(result_dir / "timeout.json", payload)
        print(
            f"PEG_RUNTIME_{mode}_TIMEOUT update={current_update['value']} phase={phase['value']}",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(124)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        environment = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        agent = make_sac_agent(
            seed=0,
            sample_obs=environment.observation_space.sample(),
            sample_action=environment.action_space.sample(),
        )
        replay_metadata = None
        replay_iterator = None
        demo_iterator = None
        if mode == "R2":
            online_buffer, demo_buffer, replay_metadata = populate_dynamic_buffers(
                environment, make_replay_buffer
            )
            sharding = jax.sharding.PositionalSharding(jax.local_devices()).replicate()
            replay_iterator = online_buffer.get_iterator(
                sample_args={"batch_size": 32}, device=sharding
            )
            demo_iterator = demo_buffer.get_iterator(
                sample_args={"batch_size": 32}, device=sharding
            )
        environment.close()

        fixed_batch = None
        if mode == "R1":
            archive = np.load(fixed_batch_path, allow_pickle=False)
            host_batch = {field: np.asarray(archive[field]) for field in FIELDS}
            archive.close()
            fixed_batch = freeze(jax.tree_util.tree_map(jax.device_put, host_batch))
            block_tree(jax, fixed_batch)
        preflight = {
            "mode": mode,
            "pid": os.getpid(),
            "fresh_initialization": True,
            "resume": False,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "initial_rng": rng_value(jax, agent.state),
            "fixed_batch_sha256": sha256(fixed_batch_path),
            "dynamic_replay": replay_metadata,
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "jax_log_compiles": os.environ.get("JAX_LOG_COMPILES"),
            "max_updates": 20,
            "timeout_seconds_per_update": 120,
            "gpu_memory_before": gpu_memory(),
        }
        write_json(result_dir / "preflight.json", preflight)
        print("PROBE_MODE:", mode, flush=True)
        print("FRESH_INITIALIZATION: True", flush=True)
        print("RESUME: False", flush=True)
        print("RNG_INITIAL:", preflight["initial_rng"], flush=True)

        completed = 0
        first_contract = None
        try:
            for index in range(20):
                current_update["value"] = index
                phase["value"] = "sample"
                sample_start = time.perf_counter()
                if mode == "R1":
                    batch = fixed_batch
                else:
                    online_batch = next(replay_iterator)
                    demo_batch = next(demo_iterator)
                    batch = concat_rlpd_batches(online_batch, demo_batch)
                sample_seconds = time.perf_counter() - sample_start
                contract = batch_contract(batch)
                if contract["observations"]["shape"] != [64, 20] or contract["actions"]["shape"] != [64, 7]:
                    raise ValueError(f"batch contract mismatch at update {index}: {contract}")
                if first_contract is None:
                    first_contract = contract
                elif contract != first_contract:
                    raise ValueError(f"batch shape/dtype changed at update {index}")
                rng_before = rng_value(jax, agent.state)
                print(f"UPDATE_START index={index}", flush=True)
                append_jsonl(
                    updates_path,
                    {
                        "event": "start",
                        "index": index,
                        "sample_seconds": sample_seconds,
                        "rng_before": rng_before,
                        "timestamp_ns": time.time_ns(),
                    },
                )
                phase["value"] = "dispatch"
                thirty_timer = threading.Timer(30, snapshot_30s, args=(result_dir, index, phase))
                thirty_timer.daemon = True
                thirty_timer.start()
                watchdog = threading.Timer(120, timeout)
                watchdog.daemon = True
                watchdog.start()
                true_start = time.perf_counter()
                dispatch_start = time.perf_counter()
                new_agent, update_info = agent.update_high_utd(batch, utd_ratio=1)
                dispatch_seconds = time.perf_counter() - dispatch_start
                phase["value"] = "full_block_until_ready"
                sync_start = time.perf_counter()
                block_tree(jax, (new_agent, update_info))
                sync_seconds = time.perf_counter() - sync_start
                true_seconds = time.perf_counter() - true_start
                watchdog.cancel()
                watchdog = None
                thirty_timer.cancel()
                thirty_timer = None
                phase["value"] = "record"
                loss_map = losses(jax, update_info)
                finite = bool(loss_map) and all(np.isfinite(value) for value in loss_map.values())
                rng_after = rng_value(jax, new_agent.state)
                event = {
                    "event": "complete",
                    "index": index,
                    "sample_seconds": sample_seconds,
                    "dispatch_seconds": dispatch_seconds,
                    "sync_seconds": sync_seconds,
                    "true_update_seconds": true_seconds,
                    "latency_classification": latency_classification(index, true_seconds),
                    "rng_before": rng_before,
                    "rng_after": rng_after,
                    "losses": loss_map,
                    "losses_finite": finite,
                    "timestamp_ns": time.time_ns(),
                }
                append_jsonl(updates_path, event)
                print("UPDATE_COMPLETE:", json.dumps(event, sort_keys=True), flush=True)
                if not finite:
                    raise FloatingPointError(f"non-finite loss at update {index}")
                if rng_before == rng_after:
                    raise RuntimeError(f"RNG did not advance at update {index}")
                agent = new_agent
                completed += 1
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if thirty_timer is not None:
                thirty_timer.cancel()

        result = {
            "mode": mode,
            "status": "PASS",
            "completed_updates": completed,
            "initial_rng": preflight["initial_rng"],
            "final_rng": rng_value(jax, agent.state),
            "batch_contract": first_contract,
            "gpu_memory_after": gpu_memory(),
        }
        write_json(result_dir / "result.json", result)
        print(f"PEG_RUNTIME_{mode}=PASS updates={completed}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("R1", "R2"), required=True)
    parser.add_argument("--fixed-batch", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    batch = args.fixed_batch.resolve()
    result_dir = args.result_dir.resolve()
    if ROOT not in batch.parents or ROOT not in result_dir.parents:
        raise ValueError("diagnostic path outside project")
    if sha256(batch) != EXPECTED_FIXED_BATCH_SHA256:
        raise ValueError("fixed batch hash mismatch")
    if result_dir.exists():
        raise FileExistsError(result_dir)
    result_dir.mkdir(parents=True)
    run(args.mode, batch, result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
