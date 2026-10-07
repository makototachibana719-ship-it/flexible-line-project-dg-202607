"""Project-owned Cable Routing demo recording utilities."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import gym
import numpy as np
from gym.spaces import flatten, flatten_space


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CABLE_V2_ROOT = (
    PROJECT_ROOT
    / "external/packages/cable_routing_v2/"
    / "CableRoute_最终交接包_V2_20260820_221714"
)
FRANKA_SIM_PARENT = CABLE_V2_ROOT / "franka_sim"
ENVIRONMENT_ID = "PandaCableRouteVision-v0"
REQUIRED_TRANSITION_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
)
OBSERVATION_KEYS = ("state", "front", "wrist")


class CableTrainingObservationWrapper(gym.ObservationWrapper):
    """Expose Cable V2 observations in the DrQ replay-buffer layout."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        native = env.observation_space
        self._state_space = native["state"]
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self._state_space),
                "front": native["images"]["front"],
                "wrist": native["images"]["wrist"],
            }
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "state": np.asarray(
                flatten(self._state_space, observation["state"]),
                dtype=np.float32,
            ),
            "front": np.asarray(observation["images"]["front"], dtype=np.uint8),
            "wrist": np.asarray(observation["images"]["wrist"], dtype=np.uint8),
        }


def make_cable_training_env(render_mode: str = "rgb_array") -> gym.Env:
    """Create the real isolated V2 environment with the intended DrQ layout."""
    sys.path.insert(0, str(FRANKA_SIM_PARENT))
    import franka_sim  # noqa: F401 - import performs Gym registration
    from serl_launcher.wrappers.chunking import ChunkingWrapper

    env = gym.make(ENVIRONMENT_ID, render_mode=render_mode)
    env = CableTrainingObservationWrapper(env)
    return ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)


def copy_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Detach all observation arrays from mutable environment buffers."""
    return {
        key: np.asarray(copy.deepcopy(observation[key])).copy()
        for key in OBSERVATION_KEYS
    }


def make_transition(
    observation: dict[str, Any],
    action: np.ndarray,
    next_observation: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    """Build the exact six-field transition consumed by Cable DrQ."""
    return {
        "observations": copy_observation(observation),
        "actions": np.asarray(action, dtype=np.float32).copy(),
        "next_observations": copy_observation(next_observation),
        "rewards": np.asarray(reward, dtype=np.float32),
        "masks": np.asarray(1.0 - float(terminated), dtype=np.float32),
        "dones": np.asarray(bool(terminated or truncated), dtype=np.bool_),
    }


def schema_metadata() -> dict[str, Any]:
    return {
        "transition_keys": list(REQUIRED_TRANSITION_KEYS),
        "observation_keys": list(OBSERVATION_KEYS),
        "observations": {
            "state": {"shape": [1, 7], "dtype": "float32"},
            "front": {"shape": [1, 128, 128, 3], "dtype": "uint8"},
            "wrist": {"shape": [1, 128, 128, 3], "dtype": "uint8"},
        },
        "action": {
            "shape": [3],
            "dtype": "float32",
            "low": -1.0,
            "high": 1.0,
            "meaning": "normalized xyz delta; gripper fixed closed",
        },
        "reward_dtype": "float32",
        "mask_dtype": "float32",
        "done_dtype": "bool",
        "mask_semantics": "1 - terminated",
        "done_semantics": "terminated or truncated",
        "success_semantics": "info['success'] is True",
    }
