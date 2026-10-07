"""Training-only Peg curriculum resets from exact MuJoCo snapshots.

This module is deliberately separate from every evaluation entry point.  It
does not change goals, rewards, success criteria, controls, or action scaling.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


SNAPSHOT_BOUNDARY = "AFTER_ENV_STEP_RETURN_THEN_MJ_FORWARD_THEN_OBSERVATION"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
PRODUCTION_DEMO_SHA256 = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
LEVELS = ("C0", "C1", "C2", "C3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainingOnlyExactStateCurriculum:
    """Delegate normal dynamics to a canonical env; replace only train reset."""

    def __init__(
        self,
        canonical_env,
        snapshot_directory: str | Path,
        level: str = "C3",
    ) -> None:
        self.env = canonical_env
        self.snapshot_directory = Path(snapshot_directory)
        self.level = self._validate_level(level)
        self.action_space = canonical_env.action_space
        self.observation_space = canonical_env.observation_space
        self.metadata = canonical_env.metadata

    @staticmethod
    def _validate_level(level: str) -> str:
        normalized = str(level).upper()
        if normalized not in LEVELS:
            raise ValueError(f"curriculum level must be one of {LEVELS}, got {level!r}")
        return normalized

    def set_level(self, level: str) -> None:
        self.level = self._validate_level(level)

    def _load_snapshot(self, level: str) -> tuple[dict, dict[str, np.ndarray]]:
        stem = level.lower() + "_exact_state"
        manifest_path = self.snapshot_directory / f"{stem}_manifest.json"
        snapshot_path = self.snapshot_directory / f"{stem}.npz"
        manifest = json.loads(manifest_path.read_text())
        if manifest["snapshot_boundary"] != SNAPSHOT_BOUNDARY:
            raise RuntimeError("snapshot observation boundary mismatch")
        if manifest["MUJOCO_STATE_SPEC"] != "mjSTATE_INTEGRATION":
            raise RuntimeError("snapshot does not use mjSTATE_INTEGRATION")
        if int(manifest["MUJOCO_STATE_SPEC_VALUE"]) != int(STATE_SPEC):
            raise RuntimeError("MuJoCo state spec value mismatch")
        if manifest["production_demo_sha256"] != PRODUCTION_DEMO_SHA256:
            raise RuntimeError("snapshot production-demo identity mismatch")
        if _sha256(snapshot_path) != manifest["snapshot_file_sha256"]:
            raise RuntimeError("snapshot SHA256 mismatch")
        with np.load(snapshot_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        if arrays["mujoco_state_vector"].size != int(manifest["MUJOCO_STATE_VECTOR_SIZE"]):
            raise RuntimeError("snapshot state-vector size mismatch")
        return manifest, arrays

    def reset(self, *, seed=None, options=None):
        if self.level == "C3":
            observation, info = self.env.reset(seed=seed, options=options)
            result_info = dict(info)
            result_info.update(curriculum_level="C3", curriculum_restored=False)
            return observation, result_info

        # Canonical reset initializes the same model and object graph.  Its
        # physical and Python state is then wholly replaced by the snapshot.
        self.env.reset(seed=seed, options=options)
        manifest, arrays = self._load_snapshot(self.level)
        raw = self.env.env
        task = raw.task

        mujoco.mj_setState(
            raw.model,
            raw.data,
            arrays["mujoco_state_vector"],
            STATE_SPEC,
        )
        wrapper_state = manifest["WRAPPER_STATE"]
        env_state = manifest["ENV_STATE"]
        task_state = manifest["TASK_STATE"]

        self.env._target_pose = arrays["adapter_target_pose"].copy()
        self.env._workspace_low = arrays["adapter_workspace_low"].copy()
        self.env._workspace_high = arrays["adapter_workspace_high"].copy()
        raw._elapsed_steps = int(env_state["raw_elapsed_steps"])
        raw._episode_count = int(env_state["raw_episode_count"])
        raw._desired_goal = arrays["raw_desired_goal"].copy()
        task.current_step = int(task_state["current_step"])
        task.is_success = bool(task_state["is_success"])
        task._insertion_axis = arrays["task_insertion_axis"].copy()
        task.last_metrics = copy.deepcopy(task_state["last_metrics"])
        raw.controller.q_initial = arrays["controller_q_initial"].copy()

        # These duplicated values make schema drift fail closed.
        checks = (
            (self.env._target_pose, wrapper_state["adapter_target_pose"], "target pose"),
            (self.env._workspace_low, wrapper_state["adapter_workspace_low"], "workspace low"),
            (self.env._workspace_high, wrapper_state["adapter_workspace_high"], "workspace high"),
        )
        for actual, expected, label in checks:
            if not np.array_equal(actual, np.asarray(expected, dtype=actual.dtype)):
                raise RuntimeError(f"snapshot {label} schema mismatch")

        # Exactly one forward operation defines the proved observation boundary.
        mujoco.mj_forward(raw.model, raw.data)
        observation = self.env._state(raw._get_obs())
        if not np.array_equal(observation, arrays["boundary_observation"]):
            delta = float(np.max(np.abs(observation - arrays["boundary_observation"])))
            raise RuntimeError(f"exact restored observation mismatch: max_delta={delta}")

        info = {
            "curriculum_level": self.level,
            "curriculum_restored": True,
            "snapshot_boundary": SNAPSHOT_BOUNDARY,
            "snapshot_source_transition_index": int(manifest["source_transition_index"]),
        }
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_training_exact_state_curriculum_env(
    canonical_factory: Callable,
    snapshot_directory: str | Path,
    *,
    level: str = "C3",
    **canonical_kwargs,
) -> TrainingOnlyExactStateCurriculum:
    """Create a TRAIN_ENV wrapper around the explicitly supplied factory."""
    canonical_env = canonical_factory(**canonical_kwargs)
    return TrainingOnlyExactStateCurriculum(canonical_env, snapshot_directory, level)

