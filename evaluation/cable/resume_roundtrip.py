"""Bounded cross-process validation of Cable safe-resume state and sidecars."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from flax import config as flax_config
from flax.training import checkpoints

from evaluation.cable.checkpoint_roundtrip import make_agent
from training.cable.resume import (
    canonical_hash,
    install_actor_step_tracking,
    load_resume_bundle,
    restore_online_replay,
    sha256_file,
    sidecar_paths,
    snapshot_online_replay,
    write_resume_metadata,
)
from training.cable.run_cable_training import (
    DEMO_MANIFEST,
    load_config,
    load_delivered_module,
    make_resume_checkpoint_saver,
    make_wandb_init_kwargs,
    populate_demo_buffer_sequentially,
)
from tools.cable_demo import make_cable_training_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime/cable_handoff/resume_roundtrip"


class CountingHorizonOneStore:
    def __init__(self):
        self.size = 0
        self.first = True

    def insert(self, transition):
        if self.first:
            self.size += 1
        self.size += 1
        self.first = bool(transition["dones"])

    def __len__(self):
        return self.size


def legacy_save(directory: Path, target, step: int) -> str:
    old = flax_config.flax_use_orbax_checkpointing
    flax_config.update("flax_use_orbax_checkpointing", False)
    try:
        return checkpoints.save_checkpoint(
            directory, target, step=step, keep=20, overwrite=False
        )
    finally:
        flax_config.update("flax_use_orbax_checkpointing", old)


def legacy_restore(directory: Path, target, step: int):
    old = flax_config.flax_use_orbax_checkpointing
    flax_config.update("flax_use_orbax_checkpointing", False)
    try:
        return checkpoints.restore_checkpoint(directory, target, step=step)
    finally:
        flax_config.update("flax_use_orbax_checkpointing", old)


def make_small_replay(delivered, env):
    from serl_launcher.utils.launcher import make_replay_buffer

    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    return make_replay_buffer(
        env, capacity=32, type="memory_efficient_replay_buffer", image_keys=image_keys
    )


def add_transitions(buffer, env, count: int = 4) -> None:
    observation, _ = env.reset(seed=0)
    for _ in range(count):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        buffer.insert(
            {
                "observations": observation,
                "actions": action,
                "next_observations": next_observation,
                "rewards": np.asarray(reward, dtype=np.float32),
                "masks": np.asarray(1.0 - terminated, dtype=np.float32),
                "dones": bool(terminated or truncated),
            }
        )
        observation = next_observation


def expected_bundle(config, checkpoint_path: Path, run_id: str):
    return load_resume_bundle(
        checkpoint_path,
        expected_environment=config["environment"],
        expected_algorithm=config["algorithm"],
        expected_seed=int(config["seed"]),
        expected_config_hash=canonical_hash(config),
        expected_demo_manifest_hash=sha256_file(DEMO_MANIFEST),
        expected_observation=config["observation"],
        expected_action=config["action"],
        expected_wandb_project=config["wandb"]["project"],
        expected_wandb_name=config["wandb"]["smoke_run_name"],
        explicit_wandb_run_id=run_id,
    )


def create_phase(config_path: Path, output: Path) -> int:
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    delivered = load_delivered_module()
    env = make_cable_training_env()
    try:
        agent = make_agent(config, delivered, env)
        checkpoint_path = Path(legacy_save(output, agent.state, 123))
        replay = make_small_replay(delivered, env)
        install_actor_step_tracking(replay, 0)
        add_transitions(replay, env)
        _, replay_path = sidecar_paths(checkpoint_path)
        replay_info = snapshot_online_replay(replay, replay_path)
        metadata_path, _ = sidecar_paths(checkpoint_path)
        write_resume_metadata(
            metadata_path,
            {
                "format_version": 1,
                "checkpoint_step": 123,
                "learner_update_steps": 123,
                "actor_steps": replay_info["actor_steps"],
                "seed": int(config["seed"]),
                "environment": config["environment"],
                "algorithm": config["algorithm"],
                "observation": config["observation"],
                "action": config["action"],
                "wandb_run_id": "resume-unit-run-id",
                "wandb_project": config["wandb"]["project"],
                "wandb_name": config["wandb"]["smoke_run_name"],
                "demo_manifest_hash": sha256_file(DEMO_MANIFEST),
                "config_hash": canonical_hash(config),
                "source_snapshot": {"test": "bounded-real-agent"},
                "online_replay_file": replay_info["path"],
                "online_replay_sha256": replay_info["sha256"],
                "online_replay_size": replay_info["size"],
                "online_replay_capacity": replay_info["capacity"],
            },
        )
        print("INITIAL_CHECKPOINT_STEP: 123")
        print(f"INITIAL_ACTOR_STEPS: {replay_info['actor_steps']}")
        return 0
    finally:
        env.close()


def resume_phase(config_path: Path, output: Path) -> int:
    config = load_config(config_path)
    checkpoint_path = output / "checkpoint_123"
    original_hash = sha256_file(checkpoint_path)
    bundle = expected_bundle(config, checkpoint_path, "resume-unit-run-id")
    if int(bundle["learner_update_steps"]) != 123:
        raise AssertionError("learner resume step mismatch")

    delivered = load_delivered_module()
    env = make_cable_training_env()
    try:
        agent = make_agent(config, delivered, env)
        restored_state = legacy_restore(output, agent.state, 123)
        agent = agent.replace(state=restored_state)
        action = np.asarray(
            jax.device_get(agent.sample_actions(observations=env.reset(seed=0)[0], argmax=True))
        )
        if action.shape != (3,) or not np.isfinite(action).all():
            raise AssertionError("restored agent inference failed")

        replay = make_small_replay(delivered, env)
        replay_info = restore_online_replay(replay, bundle["online_replay_path"])
        if replay_info["actor_steps"] != int(bundle["actor_steps"]):
            raise AssertionError("actor progress mismatch")
        if replay_info["size"] != int(bundle["online_replay_size"]):
            raise AssertionError("online replay size mismatch")

        demo_store = CountingHorizonOneStore()
        raw_total = populate_demo_buffer_sequentially(demo_store, config)
        if raw_total != 10521 or len(demo_store) != 10526:
            raise AssertionError("Demo replay rebuild mismatch")

        context = dict(bundle)
        context.update(
            {
                "enabled": True,
                "smoke": True,
                "wandb_run_id": "resume-unit-run-id",
                "active_wandb_run_id": "resume-unit-run-id",
            }
        )
        wandb_kwargs = make_wandb_init_kwargs(
            config, smoke=True, log_dir=output / "wandb", resume_context=context
        )
        if wandb_kwargs.get("id") != "resume-unit-run-id" or wandb_kwargs.get("resume") != "must":
            raise AssertionError("W&B resume reconstruction mismatch")
        saver = make_resume_checkpoint_saver(
            checkpoints.save_checkpoint,
            config=config,
            replay_buffer=replay,
            resume_context=context,
        )
        skipped = saver(output, agent.state, step=0, keep=20)
        if Path(skipped).name != "checkpoint_123":
            raise AssertionError("resume first-step skip mismatch")
        saved = Path(saver(output, agent.state, step=1, keep=20))
        if saved.name != "checkpoint_125":
            raise AssertionError(f"unexpected next checkpoint: {saved.name}")
        if sha256_file(checkpoint_path) != original_hash:
            raise AssertionError("original checkpoint was modified")
        try:
            saver(output, agent.state, step=1, keep=20)
        except FileExistsError:
            collision_pass = True
        else:
            collision_pass = False
        if not collision_pass:
            raise AssertionError("checkpoint collision was not rejected")

        result = {
            "initial_checkpoint_step": 123,
            "restored_learner_step": 123,
            "restored_actor_steps": replay_info["actor_steps"],
            "online_replay_size": replay_info["size"],
            "online_replay_resume": "RESTORED",
            "demo_raw_transitions": raw_total,
            "demo_replay_size": len(demo_store),
            "demo_replay_resume": "REBUILT",
            "wandb_run_id": "resume-unit-run-id",
            "wandb_resume_policy": "must",
            "next_checkpoint_step": 125,
            "original_checkpoint_unchanged": True,
            "collision_protection": "PASS",
            "restored_policy_inference": "PASS",
            "training_resume": "VALIDATED",
        }
        (output / "resume_roundtrip_report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("create", "resume"))
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if PROJECT_ROOT not in output.parents:
        raise ValueError("resume test output must remain inside project")
    if args.phase == "create":
        return create_phase(config_path, output)
    if args.phase == "resume":
        return resume_phase(config_path, output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    command = [
        sys.executable,
        "-m",
        "evaluation.cable.resume_roundtrip",
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    environment = os.environ.copy()
    subprocess.run(command + ["--phase", "create"], check=True, env=environment)
    subprocess.run(command + ["--phase", "resume"], check=True, env=environment)
    print("PROCESS_RECONSTRUCTION: PASS")
    print("TRAINING_RESUME: VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
