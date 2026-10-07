"""Project-side Cable DrQ launcher with explicit W&B and multi-demo inputs.

Dry-run mode never initializes W&B and never enters an actor or learner loop.
The real modes reuse the delivered Cable actor/learner functions without editing
Cable V2 or SERL, while injecting only project-owned integration boundaries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/cable/cable_baseline_5demo.yaml"
CABLE_ROOT = (
    PROJECT_ROOT
    / "external/packages/cable_routing_v2"
    / "CableRoute_最终交接包_V2_20260820_221714"
)
DELIVERED_ENTRY = CABLE_ROOT / "async_cable_route_sim/async_drq_sim.py"
FRANKA_SIM_PARENT = CABLE_ROOT / "franka_sim"
EXPECTED_KEYS = frozenset(("state", "front", "wrist"))
TRANSITION_KEYS = frozenset(
    ("observations", "actions", "next_observations", "rewards", "masks", "dones")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--role", choices=("learner", "actor"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ip", default="localhost")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("baseline config must be a mapping")
    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_observation(observation: dict[str, Any], config: dict[str, Any]) -> None:
    if not isinstance(observation, dict) or frozenset(observation) != EXPECTED_KEYS:
        raise AssertionError(
            f"observation keys must be {sorted(EXPECTED_KEYS)}, got "
            f"{sorted(observation) if isinstance(observation, dict) else type(observation)}"
        )
    expected = config["observation"]
    checks = {
        "state": (tuple(expected["state_shape"]), np.dtype(expected["state_dtype"])),
        "front": (tuple(expected["front_shape"]), np.dtype(expected["image_dtype"])),
        "wrist": (tuple(expected["wrist_shape"]), np.dtype(expected["image_dtype"])),
    }
    for key, (shape, dtype) in checks.items():
        value = np.asarray(observation[key])
        if value.shape != shape or value.dtype != dtype:
            raise AssertionError(
                f"{key} expected shape/dtype {shape}/{dtype}, got {value.shape}/{value.dtype}"
            )


def assert_transition(transition: Any, config: dict[str, Any]) -> None:
    if not isinstance(transition, dict) or frozenset(transition) != TRANSITION_KEYS:
        raise AssertionError("demo transition keys do not match learner schema")
    assert_observation(transition["observations"], config)
    assert_observation(transition["next_observations"], config)
    action = np.asarray(transition["actions"])
    if action.shape != tuple(config["action"]["shape"]):
        raise AssertionError(f"action shape mismatch: {action.shape}")
    if action.dtype != np.dtype(config["action"]["dtype"]):
        raise AssertionError(f"action dtype mismatch: {action.dtype}")
    if not np.isfinite(action).all() or np.any(action < -1.0) or np.any(action > 1.0):
        raise AssertionError("action is non-finite or outside [-1, 1]")


def demo_entries(config: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    entries = config["demos"]["files"]
    if len(entries) != int(config["demos"]["expected_count"]):
        raise AssertionError("demo file count does not match config")
    return [(resolve_project_path(item["path"]), item) for item in entries]


def validate_demo_files(config: dict[str, Any], *, inspect_transitions: bool) -> int:
    total = 0
    verify_hashes = bool(config["demos"].get("verify_sha256", False))
    for path, entry in demo_entries(config):
        if not path.is_file():
            raise FileNotFoundError(f"required demo not found: {path}")
        if verify_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != entry["sha256"]:
                raise AssertionError(f"SHA256 mismatch: {path}")
        if inspect_transitions:
            with path.open("rb") as stream:
                transitions = pickle.load(stream)
            if not isinstance(transitions, list) or not transitions:
                raise AssertionError(f"demo must be a non-empty transition list: {path}")
            for transition in transitions:
                assert_transition(transition, config)
            count = len(transitions)
            del transitions
            gc.collect()
        else:
            count = int(entry["transitions"])
        if count != int(entry["transitions"]):
            raise AssertionError(f"transition count mismatch for {path}: {count}")
        total += count
        print(f"DEMO_LOAD_CHECK: {path.name} transitions={count} PASS", flush=True)
    if total != int(config["demos"]["expected_raw_transitions"]):
        raise AssertionError(f"total transitions mismatch: {total}")
    return total


def populate_demo_buffer_sequentially(data_store: Any, config: dict[str, Any]) -> int:
    """Load one pickle at a time and release its top-level list after insertion."""
    raw_total = 0
    for path, entry in demo_entries(config):
        with path.open("rb") as stream:
            transitions = pickle.load(stream)
        if len(transitions) != int(entry["transitions"]):
            raise AssertionError(f"transition count mismatch for {path}")
        for transition in transitions:
            assert_transition(transition, config)
            data_store.insert(transition)
        raw_total += len(transitions)
        del transitions
        gc.collect()
        print(
            f"DEMO_INSERT: {path.name} raw_total={raw_total} replay_size={len(data_store)}",
            flush=True,
        )
    if raw_total != int(config["demos"]["expected_raw_transitions"]):
        raise AssertionError(f"inserted raw transition mismatch: {raw_total}")
    expected_replay = int(config["demos"]["expected_replay_size"])
    if len(data_store) != expected_replay:
        raise AssertionError(
            f"horizon-one replay size mismatch: expected {expected_replay}, got {len(data_store)}"
        )
    return raw_total


def assert_output_isolation(
    config: dict[str, Any], smoke: bool, *, allow_existing: bool = False
) -> tuple[Path, Path]:
    outputs = config["outputs"]
    checkpoint_key = "smoke_checkpoint_dir" if smoke else "checkpoint_dir"
    log_key = "smoke_log_dir" if smoke else "log_dir"
    checkpoint_dir = resolve_project_path(outputs[checkpoint_key])
    log_dir = resolve_project_path(outputs[log_key])
    for path in (checkpoint_dir, log_dir):
        if PROJECT_ROOT not in path.parents:
            raise AssertionError(f"output must remain inside project: {path}")
        if not allow_existing and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    forbidden = ("cable_route_smoke", "cable_route_shutdown_test")
    if any(name in str(checkpoint_dir) for name in forbidden):
        raise AssertionError("checkpoint directory overlaps a delivered checkpoint name")
    return checkpoint_dir, log_dir


def assert_config(
    config: dict[str, Any], smoke: bool, *, allow_existing_outputs: bool = False
) -> tuple[Path, Path]:
    if config["environment"] != "PandaCableRouteVision-v0":
        raise AssertionError("unexpected environment")
    if config["algorithm"] != "DrQ" or int(config["seed"]) != 0:
        raise AssertionError("baseline algorithm/seed mismatch")
    wandb_config = config["wandb"]
    if wandb_config["mode"] != "online":
        raise AssertionError("W&B mode must be online")
    run_name = wandb_config["smoke_run_name"] if smoke else wandb_config["run_name"]
    if not wandb_config["project"] or not run_name:
        raise AssertionError("W&B project and run name must be non-empty")
    training = config["training"]
    if int(training["replay_capacity"]) < int(config["demos"]["expected_replay_size"]):
        raise AssertionError("demo replay capacity is too small")
    return assert_output_isolation(
        config, smoke, allow_existing=allow_existing_outputs
    )


def load_delivered_module() -> Any:
    if not DELIVERED_ENTRY.is_file():
        raise FileNotFoundError(DELIVERED_ENTRY)
    sys.path.insert(0, str(FRANKA_SIM_PARENT))
    sys.path.insert(0, str(DELIVERED_ENTRY.parent))
    spec = importlib.util.spec_from_file_location("cable_v2_async_drq", DELIVERED_ENTRY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import delivered entry: {DELIVERED_ENTRY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flattened(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flattened(value, full_key))
        else:
            result[full_key] = value
    return result


class ProjectWandBLogger:
    """Minimal project logger with explicit project and visible run name."""

    def __init__(self, config: dict[str, Any], *, smoke: bool, log_dir: Path):
        import wandb

        wandb_config = config["wandb"]
        name = wandb_config["smoke_run_name"] if smoke else wandb_config["run_name"]
        log_dir.mkdir(parents=True, exist_ok=False)
        self.run = wandb.init(
            project=wandb_config["project"],
            name=name,
            mode="online",
            dir=str(log_dir),
            config={
                "environment": config["environment"],
                "algorithm": config["algorithm"],
                "seed": config["seed"],
                "demo_count": config["demos"]["expected_count"],
                "demo_transitions": config["demos"]["expected_raw_transitions"],
                **config["training"],
            },
        )
        print(f"WANDB_RUN_ID: {self.run.id}", flush=True)
        print(f"WANDB_PROJECT: {wandb_config['project']}", flush=True)
        print(f"WANDB_NAME: {name}", flush=True)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.run.log(flattened(data), step=step)


def make_legacy_checkpoint_saver(original_save):
    """Use Flax's compatible non-Orbax path without changing the environment."""
    from flax import config as flax_config

    def save_checkpoint(*args, **kwargs):
        old_value = flax_config.flax_use_orbax_checkpointing
        flax_config.update("flax_use_orbax_checkpointing", False)
        try:
            return original_save(*args, **kwargs)
        finally:
            flax_config.update("flax_use_orbax_checkpointing", old_value)

    return save_checkpoint


def make_runtime_flags(
    config: dict[str, Any], role: str, smoke: bool, ip: str, checkpoint_dir: Path
) -> SimpleNamespace:
    training = config["training"]
    max_steps = (
        int(config["smoke"]["learner_updates" if role == "learner" else "actor_steps"])
        if smoke
        else int(training["max_steps"])
    )
    checkpoint_period = (
        int(config["smoke"]["checkpoint_period"])
        if smoke
        else int(training["checkpoint_period"])
    )
    return SimpleNamespace(
        env=config["environment"],
        agent="drq",
        exp_name=(
            config["wandb"]["smoke_run_name"]
            if smoke
            else config["wandb"]["run_name"]
        ),
        max_traj_length=1000,
        seed=int(config["seed"]),
        save_model=True,
        batch_size=int(training["batch_size"]),
        critic_actor_ratio=int(training["critic_actor_ratio"]),
        max_steps=max_steps,
        replay_buffer_capacity=int(training["replay_capacity"]),
        random_steps=int(training["random_steps"]),
        training_starts=int(training["training_starts"]),
        steps_per_update=int(training["steps_per_update"]),
        log_period=int(training["log_period"]),
        eval_period=int(training["eval_period"]),
        eval_n_trajs=int(training["eval_n_trajs"]),
        learner=role == "learner",
        actor=role == "actor",
        render=False,
        ip=ip,
        encoder_type=training["encoder_type"],
        demo_path=None,
        checkpoint_period=checkpoint_period,
        checkpoint_path=str(checkpoint_dir),
        debug=False,
        log_rlds_path=None,
        preload_rlds_path=None,
    )


def dry_run(config: dict[str, Any], smoke: bool) -> int:
    checkpoint_dir, log_dir = assert_config(config, smoke)
    total = validate_demo_files(config, inspect_transitions=True)
    delivered = load_delivered_module()
    from tools.cable_demo import make_cable_training_env

    env = make_cable_training_env()
    try:
        observation, _ = env.reset(seed=int(config["seed"]))
        assert_observation(observation, config)
        if env.action_space.shape != tuple(config["action"]["shape"]):
            raise AssertionError(f"action space mismatch: {env.action_space}")
        import jax

        devices = jax.devices()
        if not devices:
            raise RuntimeError("no JAX device available")
        print(f"DELIVERED_IMPORT: PASS ({delivered.__name__})")
        print(f"OBSERVATION_ADAPTER: PASS keys={sorted(observation)}")
        print(f"ACTION_SHAPE: {env.action_space.shape} PASS")
        print(f"DEMO_RAW_TRANSITIONS: {total} PASS")
        print(
            f"DEMO_REPLAY_CAPACITY: {config['training']['replay_capacity']} "
            f">= {config['demos']['expected_replay_size']} PASS"
        )
        print(f"JAX_DEVICES: {[str(device) for device in devices]} PASS")
        print(f"WANDB_PROJECT: {config['wandb']['project']}")
        print(
            "WANDB_NAME: "
            + (config["wandb"]["smoke_run_name"] if smoke else config["wandb"]["run_name"])
        )
        print(f"WANDB_MODE: {config['wandb']['mode']} (construction only)")
        print(f"CHECKPOINT_DIR: {checkpoint_dir} PASS")
        print(f"LOG_DIR: {log_dir} PASS")
        print("WANDB_INIT: NOT_CALLED")
        print("LEARNER_UPDATE: NOT_STARTED")
        print("DRY_RUN_RESULT: PASS")
        return 0
    finally:
        env.close()


def launch(config: dict[str, Any], role: str, smoke: bool, ip: str) -> int:
    # The learner owns output creation. The actor must be allowed to connect
    # after those directories become non-empty, but never writes checkpoints or W&B logs.
    checkpoint_dir, log_dir = assert_config(
        config, smoke, allow_existing_outputs=role == "actor"
    )
    validate_demo_files(config, inspect_transitions=False)

    os.environ["MUJOCO_GL"] = config["runtime"]["mujoco_gl"]
    delivered = load_delivered_module()
    from tools.cable_demo import CableTrainingObservationWrapper, make_cable_training_env

    # The delivered actor creates its own eval environment through these globals.
    delivered.SERLObsWrapper = CableTrainingObservationWrapper
    delivered.FLAGS = make_runtime_flags(config, role, smoke, ip, checkpoint_dir)
    delivered.make_wandb_logger = lambda **_: ProjectWandBLogger(
        config, smoke=smoke, log_dir=log_dir
    )
    # Flax 0.8.5 + Orbax 0.6.4 emits TensorStore `assume_config`, which the
    # installed TensorStore 0.1.51 rejects. Keep the fix project-local and use
    # Flax's legacy serializer until the environment is changed explicitly.
    delivered.checkpoints.save_checkpoint = make_legacy_checkpoint_saver(
        delivered.checkpoints.save_checkpoint
    )

    import jax
    import jax.numpy as jnp
    from agentlace.data.data_store import QueuedDataStore
    from serl_launcher.utils.launcher import make_replay_buffer

    env = make_cable_training_env()
    observation, _ = env.reset(seed=int(config["seed"]))
    assert_observation(observation, config)
    if env.action_space.shape != tuple(config["action"]["shape"]):
        raise AssertionError("runtime action shape mismatch")

    rng = jax.random.PRNGKey(int(config["seed"]))
    rng, sampling_rng = jax.random.split(rng)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    agent = delivered.make_drq_agent(
        seed=int(config["seed"]),
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=config["training"]["encoder_type"],
    )
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), delivered.sharding.replicate()
    )

    if role == "learner":
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        replay_buffer = make_replay_buffer(
            env,
            capacity=int(config["training"]["replay_capacity"]),
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )
        demo_buffer = make_replay_buffer(
            env,
            capacity=int(config["training"]["replay_capacity"]),
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )
        populate_demo_buffer_sequentially(demo_buffer, config)
        sampling_rng = jax.device_put(sampling_rng, device=delivered.sharding.replicate())
        delivered.learner(sampling_rng, agent, replay_buffer, demo_buffer=demo_buffer)
    else:
        sampling_rng = jax.device_put(sampling_rng, delivered.sharding.replicate())
        delivered.actor(agent, QueuedDataStore(2000), env, sampling_rng)
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run and args.role:
        raise ValueError("--dry-run must not be combined with --role")
    if not args.dry_run and not args.role:
        raise ValueError("select --dry-run or one --role")
    config = load_config(args.config)
    if args.dry_run:
        return dry_run(config, args.smoke)
    return launch(config, args.role, args.smoke, args.ip)


if __name__ == "__main__":
    raise SystemExit(main())
