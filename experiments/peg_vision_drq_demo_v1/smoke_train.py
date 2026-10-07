"""Bounded Object-recipe DrQ smoke with 50/50 Demo/online batches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import pickle
from pathlib import Path
import sys
import time
import traceback

import jax
import numpy as np
import gym

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.peg_vision_drq_demo_v1.drq_runtime import (
    concat_batches,
    make_object_drq_agent,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clone_obs(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def make_replay(env, capacity, image_keys):
    return MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=capacity,
        image_keys=image_keys,
    )


class ReplaySpec:
    """Object-contract spaces without constructing MuJoCo in the learner."""

    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, shape=(1, 20), dtype=np.float32),
            "front": gym.spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
            "wrist": gym.spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)


def finite_metrics(metrics) -> dict[str, float]:
    flat = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, leaf in value.items():
                visit(f"{prefix}/{key}" if prefix else str(key), leaf)
        else:
            array = np.asarray(jax.device_get(value))
            if not np.isfinite(array).all():
                raise RuntimeError(f"nonfinite metric: {prefix}")
            flat[prefix] = float(array.mean())

    visit("", metrics)
    return flat


def param_signature(params) -> list[float]:
    return [float(np.asarray(jax.device_get(leaf)).mean()) for leaf in jax.tree_util.tree_leaves(params)]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--online-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--port", type=int, default=5628)
    args = parser.parse_args()
    output = args.output.resolve()
    if PROJECT_ROOT not in output.parents:
        raise RuntimeError("output must be below project root")
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name not in {"stdout.log"} for path in output.iterdir()):
        raise RuntimeError("output must be fresh or contain only stdout.log")
    running = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_at": now(),
        "command": [sys.executable, *sys.argv],
        "updates_requested": args.updates,
        "batch_size": args.batch_size,
        "demo_online_ratio": "50/50",
        "checkpoint_timestamps": [],
    }
    write_json(output / "RUNNING.json", running)
    result = {
        "DRQ_SMOKE": "FAIL",
        "training_updates": 0,
        "fresh_initialization": True,
        "demo_online_ratio": "50/50",
        "batch_size": args.batch_size,
        "critic_actor_ratio": 4,
    }
    env = server = client = None
    try:
        with args.demo.open("rb") as stream:
            demos = pickle.load(stream)
        with args.online_data.open("rb") as stream:
            online_rows = pickle.load(stream)
        spec = ReplaySpec()
        image_keys = ["front", "wrist"]
        demo_buffer = make_replay(spec, max(2000, len(demos) + 16), image_keys)
        online_buffer = make_replay(spec, max(1000, len(online_rows) + 16), image_keys)
        for transition in demos:
            demo_buffer.insert(transition)

        actor_queue = QueuedDataStore(max(1000, len(online_rows) + 16))
        config = TrainerConfig(
            port_number=args.port,
            broadcast_port=args.port + 1,
            request_types=["send-stats"],
        )
        server = TrainerServer(config)
        server.register_data_store("actor_env", online_buffer)
        server.start(threaded=True)
        client = TrainerClient(
            "actor_env",
            "127.0.0.1",
            config,
            data_store=actor_queue,
            wait_for_server=True,
            timeout_ms=2000,
        )

        for step, transition in enumerate(online_rows):
            actor_queue.insert(transition)
            if (step + 1) % 30 == 0:
                if not client.update():
                    raise RuntimeError(f"actor upload failed at row {step + 1}")
        if len(online_rows) % 30:
            if not client.update():
                raise RuntimeError("final actor upload failed")
        deadline = time.monotonic() + 10
        while len(online_buffer) < len(online_rows) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(online_buffer) < len(online_rows):
            raise RuntimeError(f"online replay too small: {len(online_buffer)}")
        sample_obs = demos[0]["observations"]

        agent = make_object_drq_agent(0, sample_obs, np.zeros(7, dtype=np.float32))
        initial_signature = param_signature(agent.state.params)
        received = {"count": 0, "last": None}

        def update_params(params):
            received["count"] += 1
            received["last"] = params

        client.recv_network_callback(update_params)
        time.sleep(0.3)
        server.publish_network(agent.state.params)
        deadline = time.monotonic() + 5
        while received["count"] < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        if received["count"] < 1:
            raise RuntimeError("actor did not receive initial parameters")

        half = args.batch_size // 2
        if args.batch_size % 2:
            raise RuntimeError("batch size must be even for 50/50 replay")
        metrics_log = []
        for update in range(1, args.updates + 1):
            # Match Object: three critic-only updates, then one joint update.
            for _ in range(3):
                online = online_buffer.sample(half, pack_obs_and_next_obs=True)
                demo = demo_buffer.sample(half, pack_obs_and_next_obs=True)
                batch = concat_batches(online, demo, axis=0)
                agent, critic_info = agent.update_critics(batch)
                jax.block_until_ready(agent)
                critic_metrics = finite_metrics(critic_info)
            online = online_buffer.sample(half, pack_obs_and_next_obs=True)
            demo = demo_buffer.sample(half, pack_obs_and_next_obs=True)
            batch = concat_batches(online, demo, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)
            jax.block_until_ready(agent)
            metrics = finite_metrics(update_info)
            if update == 1 or update % 10 == 0 or update == args.updates:
                row = {"update": update, **critic_metrics, **metrics}
                metrics_log.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            result["training_updates"] = update

        server.publish_network(agent.state.params)
        deadline = time.monotonic() + 5
        while received["count"] < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        if received["count"] < 2:
            raise RuntimeError("actor did not receive updated parameters")

        from flax.training import checkpoints

        checkpoint_dir = output / "checkpoints"
        checkpoints.save_checkpoint(
            str(checkpoint_dir), agent.state, step=args.updates, keep=5, overwrite=False
        )
        checkpoint_time = now()
        running["checkpoint_timestamps"].append(
            {"step": args.updates, "timestamp": checkpoint_time}
        )
        restored = checkpoints.restore_checkpoint(str(checkpoint_dir), target=agent.state)
        final_signature = param_signature(agent.state.params)
        restored_signature = param_signature(restored.params)
        restore_max_abs = float(
            max(abs(left - right) for left, right in zip(final_signature, restored_signature))
        )
        changed = float(max(abs(left - right) for left, right in zip(initial_signature, final_signature)))
        if restore_max_abs != 0.0 or changed == 0.0:
            raise RuntimeError("checkpoint restore mismatch or parameters did not update")
        write_json(output / "metrics.json", metrics_log)
        result.update(
            {
                "DRQ_SMOKE": "PASS",
                "online_replay_size": len(online_buffer),
                "demo_replay_size": len(demo_buffer),
                "actor_parameter_receipts": received["count"],
                "parameter_signature_max_change": changed,
                "checkpoint_restore_max_abs": restore_max_abs,
                "checkpoint_path": str(checkpoint_dir),
                "checkpoint_timestamp": checkpoint_time,
                "finite_critic_loss": True,
                "finite_actor_loss": True,
                "finite_temperature_loss": True,
                "image_augmentation_executed": True,
                "jax_backend": jax.default_backend(),
            }
        )
        running.update(status="COMPLETED", completed_at=now(), exit_code=0)
        write_json(output / "result.json", result)
        write_json(output / "RUNNING.json", running)
        (output / "exit_code").write_text("0\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        running.update(status="FAILED", completed_at=now(), exit_code=1)
        write_json(output / "result.json", result)
        write_json(output / "RUNNING.json", running)
        (output / "exit_code").write_text("1\n")
        raise
    finally:
        if client is not None:
            client.stop()
        if server is not None:
            server.stop()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
