"""Independent Cable policy evaluation with explicit success and return metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import config as flax_config
from flax.training import checkpoints

from training.cable.run_cable_training import (
    DEFAULT_CONFIG,
    assert_observation,
    load_config,
    load_delivered_module,
)
from tools.cable_demo import make_cable_training_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/single_skill/cable"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_agent(config: dict[str, Any], delivered: Any, env: Any) -> Any:
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    agent = delivered.make_drq_agent(
        seed=int(config["seed"]),
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=config["training"]["encoder_type"],
    )
    return jax.device_put(jax.tree_map(jnp.array, agent), delivered.sharding.replicate())


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    lengths = np.asarray([row["episode_length"] for row in rows], dtype=np.float64)
    distances = np.asarray([row["final_distance_to_slot"] for row in rows], dtype=np.float64)
    times = np.asarray([row["execution_time_seconds"] for row in rows], dtype=np.float64)
    successes = int(sum(bool(row["success"]) for row in rows))
    return {
        "episodes": len(rows),
        "success_count": successes,
        "success_rate": successes / len(rows),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_std": float(lengths.std()),
        "execution_time_mean_seconds": float(times.mean()),
        "final_distance_mean": float(distances.mean()),
        "final_distance_std": float(distances.std()),
        "success_metric": 'terminal info["success"]',
        "final_task_metric": 'terminal info["distance_to_slot"]',
    }


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    config = load_config(args.config)
    delivered = load_delivered_module()
    env = make_cable_training_env()
    try:
        observation, info = env.reset(seed=args.seed)
        assert_observation(observation, config)
        if args.dry_run:
            next_observation, reward, terminated, truncated, step_info = env.step(
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            )
            assert_observation(next_observation, config)
            print("EVALUATION_ENTRY: PREPARED")
            print('SUCCESS_METRIC: terminal info["success"]')
            print('RETURN_METRIC: sum of environment rewards per episode')
            print('FINAL_TASK_METRIC: terminal info["distance_to_slot"]')
            print(f"DRY_RUN_REWARD_TYPE: {type(reward).__name__}")
            print(f"DRY_RUN_TERMINATED: {terminated}")
            print(f"DRY_RUN_TRUNCATED: {truncated}")
            print(f"DRY_RUN_INFO_KEYS: {sorted(step_info)}")
            print("EVALUATION_RUNTIME: PREPARED / CHECKPOINT_UNVERIFIED")
            return 0
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --dry-run is used")

        checkpoint_dir = args.checkpoint.expanduser().resolve()
        agent = make_agent(config, delivered, env)
        old_orbax = flax_config.flax_use_orbax_checkpointing
        try:
            flax_config.update("flax_use_orbax_checkpointing", False)
            state = checkpoints.restore_checkpoint(
                checkpoint_dir, agent.state, step=args.checkpoint_step
            )
        finally:
            flax_config.update("flax_use_orbax_checkpointing", old_orbax)
        agent = agent.replace(state=state)

        rows: list[dict[str, Any]] = []
        rng = jax.random.PRNGKey(args.seed)
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            episode_steps = 0
            started = time.perf_counter()
            while True:
                action = agent.sample_actions(
                    observations=jax.device_put(observation), argmax=True
                )
                observation, reward, terminated, truncated, info = env.step(
                    np.asarray(jax.device_get(action))
                )
                episode_return += float(reward)
                episode_steps += 1
                if terminated or truncated:
                    break
            rows.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "success": bool(info.get("success", False)),
                    "return": episode_return,
                    "episode_length": episode_steps,
                    "execution_time_seconds": time.perf_counter() - started,
                    "final_distance_to_slot": float(info["distance_to_slot"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir = output_dir.resolve()
        if PROJECT_ROOT not in output_dir.parents:
            raise ValueError("evaluation output must remain inside project")
        output_dir.mkdir(parents=True, exist_ok=False)
        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = summary(rows)
        result.update(
            {
                "checkpoint": str(checkpoint_dir),
                "checkpoint_step": args.checkpoint_step,
                "seed": args.seed,
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
