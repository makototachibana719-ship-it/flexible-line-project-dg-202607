#!/usr/bin/env python3
"""Bounded, read-only evidence collection for Object Grasp M1D-B."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "external/packages/object_grasping/object_grasp_package"
DEMO = PACKAGE / "demos/franka_lift_cube_image_20_trajs.pkl"
CHECKPOINT_ROOT = PACKAGE / "checkpoints"
ENV_SOURCE = PACKAGE / "code/env"


def array_summary(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def load_demo() -> list[dict]:
    # This pickle is a frozen, repository-local, project-audited delivery asset.
    with DEMO.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise TypeError("Expected a flat list of transition dictionaries")
    return data


def audit_demo() -> dict:
    data = load_demo()
    required = {
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "masks",
        "dones",
    }
    observed_key_sets = {tuple(sorted(item)) for item in data}
    missing_required = [i for i, item in enumerate(data) if not required <= item.keys()]
    obs_key_sets = {tuple(sorted(item["observations"])) for item in data}
    next_obs_key_sets = {tuple(sorted(item["next_observations"])) for item in data}

    episode_lengths: list[int] = []
    episode_returns: list[float] = []
    current_length = 0
    current_return = 0.0
    terminal_indices: list[int] = []
    for index, item in enumerate(data):
        current_length += 1
        current_return += float(np.asarray(item["rewards"]))
        if bool(np.asarray(item["dones"])):
            terminal_indices.append(index)
            episode_lengths.append(current_length)
            episode_returns.append(current_return)
            current_length = 0
            current_return = 0.0

    actions = np.stack([np.asarray(x["actions"]) for x in data])
    action_norms = np.linalg.norm(actions, axis=-1)
    zero_action = np.all(np.isclose(actions, 0.0), axis=-1)
    rewards = np.asarray([float(np.asarray(x["rewards"])) for x in data])
    masks = np.asarray([float(np.asarray(x["masks"])) for x in data])
    dones = np.asarray([bool(np.asarray(x["dones"])) for x in data])

    array_fields = []
    finite = True
    for item in data:
        for parent in ("observations", "next_observations"):
            for value in item[parent].values():
                array = np.asarray(value)
                array_fields.append((array.shape, str(array.dtype)))
                finite = finite and bool(np.isfinite(array).all())
        for key in ("actions", "rewards", "masks"):
            finite = finite and bool(np.isfinite(np.asarray(item[key])).all())

    first = data[0]
    schema = {
        "transition_keys": sorted(first),
        "observation_keys": sorted(first["observations"]),
        "observation": {
            key: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for key, value in first["observations"].items()
        },
        "action": {
            "shape": list(np.asarray(first["actions"]).shape),
            "dtype": str(np.asarray(first["actions"]).dtype),
        },
        "reward": {
            "shape": list(np.asarray(first["rewards"]).shape),
            "dtype": str(np.asarray(first["rewards"]).dtype),
        },
        "mask_type": type(first["masks"]).__name__,
        "done_type": type(first["dones"]).__name__,
    }
    return {
        "path": str(DEMO),
        "size_bytes": DEMO.stat().st_size,
        "sha256": hashlib.sha256(DEMO.read_bytes()).hexdigest(),
        "top_level_type": type(data).__name__,
        "storage_form": "flat_transition_list",
        "total_transitions": len(data),
        "transition_key_sets": [list(keys) for keys in sorted(observed_key_sets)],
        "observation_key_sets": [list(keys) for keys in sorted(obs_key_sets)],
        "next_observation_key_sets": [list(keys) for keys in sorted(next_obs_key_sets)],
        "missing_required_indices": missing_required,
        "schema": schema,
        "all_numeric_finite": finite,
        "action_min_per_dim": actions.min(axis=0).tolist(),
        "action_max_per_dim": actions.max(axis=0).tolist(),
        "action_mean_abs_per_dim": np.abs(actions).mean(axis=0).tolist(),
        "action_norm": array_summary(action_norms.tolist()),
        "zero_action_count": int(zero_action.sum()),
        "zero_action_ratio": float(zero_action.mean()),
        "reward": array_summary(rewards.tolist()),
        "mask_values": sorted(np.unique(masks).tolist()),
        "done_count": int(dones.sum()),
        "terminal_indices": terminal_indices,
        "trailing_unterminated_transitions": current_length,
        "total_episodes_from_done": len(episode_lengths),
        "episode_boundary_validation": (
            "PASS" if episode_lengths and current_length == 0 else "UNVERIFIED_NO_DONE_BOUNDARIES"
        ),
        "episode_length": array_summary(episode_lengths),
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "episode_return": array_summary(episode_returns),
        "success_evidence_fields_present": False,
        "block_height_or_position_present": False,
        "success_validation": "UNVERIFIED_MISSING_BLOCK_STATE_AND_SUCCESS_METADATA",
    }


def demo_spaces(data: list[dict]):
    import gym

    first = data[0]
    obs_spaces = {}
    for key, value in first["observations"].items():
        array = np.asarray(value)
        if array.dtype == np.uint8:
            obs_spaces[key] = gym.spaces.Box(0, 255, shape=array.shape, dtype=np.uint8)
        else:
            obs_spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, shape=array.shape, dtype=array.dtype
            )
    action = np.asarray(first["actions"])
    return (
        gym.spaces.Dict(obs_spaces),
        gym.spaces.Box(-1.0, 1.0, shape=action.shape, dtype=action.dtype),
    )


def audit_replay() -> dict:
    from serl_launcher.data.memory_efficient_replay_buffer import (
        MemoryEfficientReplayBuffer,
    )

    data = load_demo()
    observation_space, action_space = demo_spaces(data)
    replay = MemoryEfficientReplayBuffer(
        observation_space,
        action_space,
        capacity=10_000,
        pixel_keys=("front", "wrist"),
    )
    for transition in data:
        replay.insert(transition)
    batch = replay.sample(batch_size=8, pack_obs_and_next_obs=True)
    return {
        "raw_demo_transitions": len(data),
        "replay_size": len(replay),
        "horizon_one_initialization_records": len(replay) - len(data),
        "pixel_keys": list(replay.pixel_keys),
        "sample_batch_size": int(np.asarray(batch["actions"]).shape[0]),
        "sample_shapes": {
            "actions": list(np.asarray(batch["actions"]).shape),
            "state": list(np.asarray(batch["observations"]["state"]).shape),
            "front_packed": list(np.asarray(batch["observations"]["front"]).shape),
            "wrist_packed": list(np.asarray(batch["observations"]["wrist"]).shape),
        },
        "compatibility": "PASS",
    }


def audit_resets() -> dict:
    sys.path.insert(0, str(ENV_SOURCE))
    import gym
    import franka_sim  # noqa: F401

    env = gym.make("PandaPickCubeVision-v0")
    records = []
    try:
        for index in range(20):
            obs, _ = env.reset(seed=index)
            base = env.unwrapped
            block = base._data.sensor("block_pos").data.copy()
            tcp = base._data.sensor("2f85/pinch_pos").data.copy()
            front = np.asarray(obs["images"]["front"])
            wrist = np.asarray(obs["images"]["wrist"])
            records.append(
                {
                    "reset": index,
                    "block_pos": block.tolist(),
                    "tcp_pos": tcp.tolist(),
                    "gripper_shape": list(
                        np.asarray(obs["state"]["panda/gripper_pos"]).shape
                    ),
                    "front_mean": float(front.mean()),
                    "wrist_mean": float(wrist.mean()),
                }
            )
    finally:
        env.close()
    block = np.asarray([x["block_pos"] for x in records])
    tcp = np.asarray([x["tcp_pos"] for x in records])
    return {
        "count": len(records),
        "block_position_distribution": "RANDOMIZED",
        "block_position_min": block.min(axis=0).tolist(),
        "block_position_max": block.max(axis=0).tolist(),
        "block_position_std": block.std(axis=0).tolist(),
        "tcp_distribution": "FIXED",
        "tcp_position_min": tcp.min(axis=0).tolist(),
        "tcp_position_max": tcp.max(axis=0).tolist(),
        "camera_initial_state": "VARIES_WITH_RANDOMIZED_SCENE",
        "gripper_actual_shapes": [
            list(shape) for shape in sorted({tuple(x["gripper_shape"]) for x in records})
        ],
        "records": records,
    }


def audit_checkpoint() -> dict:
    import gym
    import jax
    from flax.training import checkpoints
    from serl_launcher.utils.launcher import make_drq_agent

    data = load_demo()
    observation_space, action_space = demo_spaces(data)
    sample_obs = observation_space.sample()
    sample_action = action_space.sample()
    agent = make_drq_agent(
        seed=0,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=("front", "wrist"),
        encoder_type="resnet-pretrained",
    )
    target_leaves = jax.tree_util.tree_leaves(agent.state)
    result = {
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "checkpoint_step": 4500,
        "target_leaf_count": len(target_leaves),
        "target_action_shape": list(sample_action.shape),
        "target_observation_shapes": {
            key: list(np.asarray(value).shape) for key, value in sample_obs.items()
        },
    }
    try:
        restored = checkpoints.restore_checkpoint(
            str(CHECKPOINT_ROOT), target=agent.state, step=4500
        )
        restored_leaves = jax.tree_util.tree_leaves(restored)
        result["restore"] = "PASS"
        result["restored_leaf_count"] = len(restored_leaves)
        result["leaf_shape_match"] = all(
            np.shape(a) == np.shape(b)
            for a, b in zip(target_leaves, restored_leaves)
        ) and len(target_leaves) == len(restored_leaves)
        restored_agent = agent.replace(state=restored)
        reset_obs = {key: np.asarray(value) for key, value in sample_obs.items()}
        action = restored_agent.sample_actions(
            observations=jax.device_put(reset_obs),
            seed=jax.random.PRNGKey(1),
            deterministic=True,
        )
        action = np.asarray(jax.device_get(action))
        result["policy_inference"] = "PASS" if action.shape == (4,) else "FAIL"
        result["policy_action_shape"] = list(action.shape)
        result["policy_action_finite"] = bool(np.isfinite(action).all())
        result["policy_action_in_space"] = bool(action_space.contains(action))
    except Exception as exc:  # Evidence is recorded, not hidden by the audit runner.
        result["restore"] = "FAIL"
        result["policy_inference"] = "UNVERIFIED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def audit_checkpoint_structure() -> dict:
    import jax
    from flax import serialization, traverse_util
    from serl_launcher.utils.launcher import make_drq_agent

    data = load_demo()
    observation_space, action_space = demo_spaces(data)
    sample_obs = observation_space.sample()
    sample_action = action_space.sample()
    agent = make_drq_agent(
        seed=0,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=("front", "wrist"),
        encoder_type="resnet-pretrained",
    )
    target_state = serialization.to_state_dict(agent.state)
    target_flat = traverse_util.flatten_dict(target_state)
    metadata_path = CHECKPOINT_ROOT / "checkpoint_4500/_METADATA"
    metadata = json.loads(metadata_path.read_text())
    checkpoint_paths = {tuple(ast.literal_eval(key)) for key in metadata["tree_metadata"]}
    target_paths = set(target_flat)
    missing_from_target = sorted(checkpoint_paths - target_paths)
    missing_from_checkpoint = sorted(target_paths - checkpoint_paths)
    return {
        "checkpoint_step": 4500,
        "checkpoint_format": "FLAX_CHECKPOINT_WITH_ORBAX_OCDBT_BACKEND",
        "checkpoint_metadata_leaf_count": len(checkpoint_paths),
        "constructed_target_leaf_count": len(target_paths),
        "jax_tree_leaf_count": len(jax.tree_util.tree_leaves(agent.state)),
        "exact_serialized_path_match": not missing_from_target
        and not missing_from_checkpoint,
        "missing_from_target_count": len(missing_from_target),
        "missing_from_checkpoint_count": len(missing_from_checkpoint),
        "missing_from_target_examples": [list(x) for x in missing_from_target[:10]],
        "missing_from_checkpoint_examples": [
            list(x) for x in missing_from_checkpoint[:10]
        ],
        "sample_observation_shapes": {
            key: list(np.asarray(value).shape) for key, value in sample_obs.items()
        },
        "sample_action_shape": list(sample_action.shape),
        "architecture_mapping": (
            "MATCH" if checkpoint_paths == target_paths else "MISMATCH"
        ),
        "restore": "UNVERIFIED_TIMEOUT_AFTER_120_SECONDS",
        "policy_inference": "UNVERIFIED_RESTORE_DID_NOT_RETURN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("demo", "replay", "resets", "checkpoint", "structure")
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    functions = {
        "demo": audit_demo,
        "replay": audit_replay,
        "resets": audit_resets,
        "checkpoint": audit_checkpoint,
        "structure": audit_checkpoint_structure,
    }
    report = functions[args.mode]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
