#!/usr/bin/env python3
"""Bounded restore-placement causal probe; no W&B, checkpoint save, or formal launch."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from peg_jax_xla_differential_probe import (
    CANDIDATE,
    ENV_PARENT,
    EXAMPLE,
    FIELDS,
    ROOT,
    gpu_memory,
    sha256,
    tree_signature,
    write_json,
)


EXPECTED_BATCH_SHA256 = "854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c"


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rng_value(jax, state: object) -> list[int]:
    return np.asarray(jax.device_get(state.rng), dtype=np.uint32).tolist()


def placement_counts(signature: dict) -> dict:
    leaves = signature["leaves"]
    return {
        "leaf_count": len(leaves),
        "cuda_jax_array_count": sum(
            leaf["type"] == "ArrayImpl" and leaf["device"] == "cuda:0" for leaf in leaves
        ),
        "host_numpy_count": sum(
            leaf["type"] == "ndarray" and leaf["device"] == "None" for leaf in leaves
        ),
    }


def block_tree(jax, tree: object) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def loss_values(jax, update_info: object) -> dict[str, float]:
    values = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(update_info)[0]:
        array = np.asarray(jax.device_get(leaf))
        if "loss" in str(path).lower() and array.size == 1:
            values[str(path)] = float(array.reshape(()))
    return values


def run(mode: str, batch_path: Path, result_dir: Path, timeout_seconds: int, max_updates: int) -> None:
    import jax
    from flax.core import freeze

    sys.path[:0] = [str(ROOT), str(EXAMPLE), str(ENV_PARENT)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

    progress_path = result_dir / "updates.jsonl"
    phase = {"value": "bootstrap"}
    current_update = {"value": None}
    watchdog: threading.Timer | None = None
    stack_timers: list[threading.Timer] = []

    def dump_stack(delay: int) -> None:
        print(
            f"[STACK] delay={delay}s update={current_update['value']} phase={phase['value']}",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def timeout() -> None:
        payload = {
            "status": "TIMEOUT",
            "mode": mode,
            "update_index": current_update["value"],
            "phase": phase["value"],
            "timeout_seconds": timeout_seconds,
            "timestamp_ns": time.time_ns(),
        }
        write_json(result_dir / "timeout.json", payload)
        print(
            f"PEG_RESTORE_PLACEMENT_{mode}_TIMEOUT update={current_update['value']} "
            f"phase={phase['value']} limit={timeout_seconds}s",
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
        environment.close()
        agent = agent.replace(state=restore_fresh_state(CANDIDATE.parent, agent.state, 0))

        phase["value"] = "signature_before_placement"
        signature_before = tree_signature(jax, agent.state)
        placement_before = placement_counts(signature_before)
        if placement_before != {
            "leaf_count": 198,
            "cuda_jax_array_count": 0,
            "host_numpy_count": 198,
        }:
            raise RuntimeError(f"unexpected restored-state placement: {placement_before}")

        phase["value"] = "explicit_gpu_placement" if mode == "D2" else "placement_control"
        placement_seconds = 0.0
        if mode == "D2":
            placement_start = time.perf_counter()
            placed_state = jax.tree_util.tree_map(jax.device_put, agent.state)
            block_tree(jax, placed_state)
            placement_seconds = time.perf_counter() - placement_start
            agent = agent.replace(state=placed_state)

        signature_after = tree_signature(jax, agent.state)
        placement_after = placement_counts(signature_after)
        expected_after = (
            {"leaf_count": 198, "cuda_jax_array_count": 198, "host_numpy_count": 0}
            if mode == "D2"
            else placement_before
        )
        if placement_after != expected_after:
            raise RuntimeError(f"unexpected post-placement state: {placement_after}")

        archive = np.load(batch_path, allow_pickle=False)
        host_batch = {field: np.asarray(archive[field]) for field in FIELDS}
        archive.close()
        batch = freeze(jax.tree_util.tree_map(jax.device_put, host_batch))
        block_tree(jax, batch)
        initial_rng = rng_value(jax, agent.state)
        preflight = {
            "mode": mode,
            "pid": os.getpid(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "checkpoint": str(CANDIDATE),
            "checkpoint_rng": initial_rng,
            "batch_sha256": sha256(batch_path),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "jax_log_compiles": os.environ.get("JAX_LOG_COMPILES"),
            "max_updates": max_updates,
            "timeout_seconds_per_update": timeout_seconds,
            "placement_seconds": placement_seconds,
            "state_signature_before_placement": signature_before,
            "state_signature_after_placement": signature_after,
            "placement_counts_before": placement_before,
            "placement_counts_after": placement_after,
            "gpu_memory_before_updates": gpu_memory(),
        }
        write_json(result_dir / "preflight.json", preflight)
        print("PROBE_MODE:", mode, flush=True)
        print("PLACEMENT_BEFORE:", placement_before, flush=True)
        print("PLACEMENT_AFTER:", placement_after, flush=True)
        print("PLACEMENT_SECONDS:", f"{placement_seconds:.9f}", flush=True)
        print("RNG_INITIAL:", initial_rng, flush=True)

        completed = 0
        try:
            for update_index in range(max_updates):
                current_update["value"] = update_index
                rng_before = rng_value(jax, agent.state)
                append_jsonl(
                    progress_path,
                    {
                        "event": "start",
                        "update_index": update_index,
                        "rng_before": rng_before,
                        "gpu_memory": gpu_memory(),
                        "timestamp_ns": time.time_ns(),
                    },
                )
                phase["value"] = "dispatch"
                for delay in (30, 90):
                    timer = threading.Timer(delay, dump_stack, args=(delay,))
                    timer.daemon = True
                    timer.start()
                    stack_timers.append(timer)
                watchdog = threading.Timer(timeout_seconds, timeout)
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
                for timer in stack_timers:
                    timer.cancel()
                stack_timers.clear()
                phase["value"] = "record_update"
                losses = loss_values(jax, update_info)
                finite = bool(losses) and all(np.isfinite(value) for value in losses.values())
                rng_after = rng_value(jax, new_agent.state)
                event = {
                    "event": "complete",
                    "update_index": update_index,
                    "rng_before": rng_before,
                    "rng_after": rng_after,
                    "dispatch_seconds": dispatch_seconds,
                    "sync_seconds": sync_seconds,
                    "true_update_seconds": true_seconds,
                    "losses": losses,
                    "losses_finite": finite,
                    "gpu_memory": gpu_memory(),
                    "timestamp_ns": time.time_ns(),
                }
                append_jsonl(progress_path, event)
                print("UPDATE_RESULT:", json.dumps(event, sort_keys=True), flush=True)
                if not finite:
                    raise FloatingPointError(f"non-finite losses at update {update_index}")
                agent = new_agent
                completed += 1
        finally:
            if watchdog is not None:
                watchdog.cancel()
            for timer in stack_timers:
                timer.cancel()

        phase["value"] = "finalize"
        final_signature = tree_signature(jax, agent.state)
        result = {
            "mode": mode,
            "status": "PASS",
            "completed_updates": completed,
            "rng_initial": initial_rng,
            "rng_final": rng_value(jax, agent.state),
            "final_state_placement": placement_counts(final_signature),
            "gpu_memory_after_updates": gpu_memory(),
        }
        write_json(result_dir / "result.json", result)
        print(f"PEG_RESTORE_PLACEMENT_{mode}=PASS updates={completed}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("D1", "D2"), required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-updates", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch = args.batch.resolve()
    result_dir = args.result_dir.resolve()
    if ROOT not in batch.parents or ROOT not in result_dir.parents:
        raise ValueError("diagnostic path outside project")
    if sha256(batch) != EXPECTED_BATCH_SHA256:
        raise ValueError("fixed batch hash mismatch")
    if args.timeout_seconds != 120 or args.max_updates != 5:
        raise ValueError("causal diagnostic bounds are immutable")
    if result_dir.exists():
        raise FileExistsError(result_dir)
    result_dir.mkdir(parents=True)
    run(args.mode, batch, result_dir, args.timeout_seconds, args.max_updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
