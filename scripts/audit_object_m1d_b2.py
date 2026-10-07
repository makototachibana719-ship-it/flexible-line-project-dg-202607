#!/usr/bin/env python3
"""Bounded Object M1D-B2 optimizer-state and checkpoint closure audit."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "external/packages/object_grasping/object_grasp_package"
DEMO = PACKAGE / "demos/franka_lift_cube_image_20_trajs.pkl"
CHECKPOINT = PACKAGE / "checkpoints/checkpoint_4500"
ENV_SOURCE = PACKAGE / "code/env"


def load_demo() -> list[dict]:
    with DEMO.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or not data:
        raise TypeError("Expected the frozen flat transition list")
    return data


def demo_spaces(data):
    import gym

    obs_spaces = {}
    for key, value in data[0]["observations"].items():
        array = np.asarray(value)
        if array.dtype == np.uint8:
            obs_spaces[key] = gym.spaces.Box(0, 255, array.shape, dtype=array.dtype)
        else:
            obs_spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, array.shape, dtype=array.dtype
            )
    action = np.asarray(data[0]["actions"])
    return gym.spaces.Dict(obs_spaces), gym.spaces.Box(
        -1.0, 1.0, action.shape, dtype=action.dtype
    )


def build_agent():
    from serl_launcher.utils.launcher import make_drq_agent

    data = load_demo()
    observation_space, action_space = demo_spaces(data)
    agent = make_drq_agent(
        seed=0,
        sample_obs=observation_space.sample(),
        sample_action=action_space.sample(),
        image_keys=("front", "wrist"),
        encoder_type="resnet-pretrained",
    )
    return agent, observation_space, action_space


def serialized_tree(state, *, keep_empty_nodes=False):
    from flax import serialization, traverse_util

    return traverse_util.flatten_dict(
        serialization.to_state_dict(state), keep_empty_nodes=keep_empty_nodes
    )


def checkpoint_paths():
    metadata = json.loads((CHECKPOINT / "_METADATA").read_text())
    return {tuple(ast.literal_eval(key)) for key in metadata["tree_metadata"]}


def checkpoint_metadata():
    metadata = json.loads((CHECKPOINT / "_METADATA").read_text())
    return {
        tuple(ast.literal_eval(key)): value["value_metadata"]
        for key, value in metadata["tree_metadata"].items()
    }


def initialize_lazy_optimizer_state(agent):
    """Run one bounded zero-gradient optimizer update; no learner/loss loop."""
    import jax

    zero_grads = jax.tree_util.tree_map(jax.numpy.zeros_like, agent.state.params)
    grads = {
        "actor": zero_grads,
        "critic": zero_grads,
        "temperature": zero_grads,
    }
    return agent.replace(state=agent.state.apply_gradients(grads=grads))


def value_signature(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def optimizer_audit():
    agent, observation_space, action_space = build_agent()
    pre = serialized_tree(agent.state)
    pre_structural = serialized_tree(agent.state, keep_empty_nodes=True)
    post_agent = initialize_lazy_optimizer_state(agent)
    post = serialized_tree(post_agent.state)
    post_structural = serialized_tree(post_agent.state, keep_empty_nodes=True)
    checkpoint = checkpoint_paths()
    metadata = checkpoint_metadata()
    checkpoint_non_array = sorted(
        path
        for path, value in metadata.items()
        if value["value_type"] != "jax.Array"
    )
    added = sorted(set(post_structural) - set(pre_structural))
    return {
        "initialization_method": "ONE_ZERO_GRADIENT_APPLY_GRADIENTS_ALL_THREE_OPTIMIZERS",
        "learner_loop_started": False,
        "loss_evaluated": False,
        "pre_update_array_paths": len(pre),
        "post_update_array_paths": len(post),
        "pre_update_structural_paths": len(pre_structural),
        "post_update_structural_paths": len(post_structural),
        "checkpoint_paths": len(checkpoint),
        "checkpoint_array_paths": sum(
            value["value_type"] == "jax.Array" for value in metadata.values()
        ),
        "checkpoint_non_array_paths": [list(path) for path in checkpoint_non_array],
        "checkpoint_non_array_metadata": {
            "/".join(path): metadata[path] for path in checkpoint_non_array
        },
        "paths_added_by_first_optimizer_update": [list(path) for path in added],
        "post_checkpoint_path_match": set(post_structural) == checkpoint,
        "missing_from_post": [
            list(x) for x in sorted(checkpoint - set(post_structural))
        ],
        "missing_from_checkpoint": [
            list(x) for x in sorted(set(post_structural) - checkpoint)
        ],
        "sample_observation_shapes": {
            key: list(space.shape) for key, space in observation_space.spaces.items()
        },
        "sample_action_shape": list(action_space.shape),
        "architecture_mapping": (
            "MATCH" if set(post_structural) == checkpoint else "MISMATCH"
        ),
        "optimizer_state_explanation": (
            "THREE_OPTAX_EMPTY_STATE_SENTINELS_NOT_LAZY_ARRAY_STATE"
        ),
    }


class RestoreTimeout(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise RestoreTimeout("checkpoint restore exceeded bounded timeout")


def adapt_native_observation(env, observation):
    from gym.spaces import flatten

    state_space = env.unwrapped.observation_space["state"]
    return {
        "state": np.expand_dims(flatten(state_space, observation["state"]), axis=0),
        "front": np.expand_dims(observation["images"]["front"], axis=0),
        "wrist": np.expand_dims(observation["images"]["wrist"], axis=0),
    }


def restore_audit(timeout_seconds: int):
    import jax
    from flax.training import checkpoints

    agent, _observation_space, action_space = build_agent()
    target_agent = initialize_lazy_optimizer_state(agent)
    target_flat = serialized_tree(target_agent.state)
    target_structural = serialized_tree(target_agent.state, keep_empty_nodes=True)
    expected_paths = checkpoint_paths()
    report = {
        "restore_backend": "flax.training.checkpoints.restore_checkpoint_ORBAX_OCDBT",
        "timeout_seconds": timeout_seconds,
        "target_paths": len(target_flat),
        "target_structural_paths": len(target_structural),
        "checkpoint_paths": len(expected_paths),
        "architecture_path_match_before_restore": (
            set(target_structural) == expected_paths
        ),
    }
    if not report["architecture_path_match_before_restore"]:
        report["restore"] = "NOT_ATTEMPTED_ARCHITECTURE_MISMATCH"
        return report

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    start = time.monotonic()
    try:
        restored = checkpoints.restore_checkpoint(
            str(CHECKPOINT.parent), target=target_agent.state, step=4500
        )
        restored = jax.block_until_ready(restored)
        elapsed = time.monotonic() - start
        restored_flat = serialized_tree(restored)
        path_match = set(restored_flat) == set(target_flat)
        signatures_match = path_match and all(
            np.shape(restored_flat[path]) == np.shape(target_flat[path])
            and np.asarray(restored_flat[path]).dtype
            == np.asarray(target_flat[path]).dtype
            for path in target_flat
        )
        report.update(
            {
                "restore": "PASS",
                "elapsed_seconds": elapsed,
                "restored_serialized_leaf_count": len(restored_flat),
                "restored_jax_leaf_count": len(jax.tree_util.tree_leaves(restored)),
                "path_match": path_match,
                "shape_dtype_match": signatures_match,
                "restored_step": int(np.asarray(restored.step)),
            }
        )
    except RestoreTimeout as exc:
        report.update(
            {
                "restore": "UNVERIFIED_TIMEOUT",
                "elapsed_seconds": time.monotonic() - start,
                "error": str(exc),
                "policy_inference": "NOT_RUN",
            }
        )
        return report
    except Exception as exc:
        report.update(
            {
                "restore": "FAIL",
                "elapsed_seconds": time.monotonic() - start,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "policy_inference": "NOT_RUN",
            }
        )
        return report
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    if report["restore"] != "PASS" or not report["shape_dtype_match"]:
        report["policy_inference"] = "NOT_RUN"
        return report

    sys.path.insert(0, str(ENV_SOURCE))
    import gym
    import franka_sim  # noqa: F401

    env = gym.make("PandaPickCubeVision-v0")
    try:
        native_obs, _ = env.reset(seed=0)
        obs = adapt_native_observation(env, native_obs)
        restored_agent = target_agent.replace(state=restored)
        action = restored_agent.sample_actions(
            observations=jax.device_put(obs),
            seed=jax.random.PRNGKey(1),
            argmax=True,
        )
        action = np.asarray(jax.device_get(action))
        report.update(
            {
                "policy_inference": "PASS"
                if action.shape == (4,)
                and np.isfinite(action).all()
                and action_space.contains(action)
                else "FAIL",
                "policy_action_shape": list(action.shape),
                "policy_action_dtype": str(action.dtype),
                "policy_action_finite": bool(np.isfinite(action).all()),
                "policy_action_in_space": bool(action_space.contains(action)),
            }
        )
    finally:
        env.close()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("optimizer", "restore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    report = (
        optimizer_audit()
        if args.mode == "optimizer"
        else restore_audit(args.timeout_seconds)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
