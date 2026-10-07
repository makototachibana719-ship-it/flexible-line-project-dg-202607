#!/usr/bin/env python3
"""Static preflight for the training-only exact-state Peg curriculum."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from peg_env_adapter import make_peg_state_env
from training.peg_insertion.exact_state_curriculum import (
    PRODUCTION_DEMO_SHA256,
    SNAPSHOT_BOUNDARY,
    STATE_SPEC,
    TrainingOnlyExactStateCurriculum,
    make_training_exact_state_curriculum_env,
)


SNAPSHOTS = ROOT / "results/closure/peg/f3_v2_exact_state_restore_20260907_001"
OUT_PARENT = ROOT / "results/closure/peg"
DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
EVAL_CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
EVAL_CONFIG_SHA256 = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
THRESHOLD = 7.41773975814


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_output() -> Path:
    for index in range(1, 1000):
        candidate = OUT_PARENT / f"f3_v2_curriculum_static_preflight_20260907_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("no unique curriculum-preflight output")


def imports_curriculum(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any("exact_state_curriculum" in alias.name for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and "exact_state_curriculum" in node.module:
                return True
    return False


def main() -> None:
    demo_before = sha256(DEMO)
    config_before = sha256(EVAL_CONFIG)
    if demo_before != PRODUCTION_DEMO_SHA256 or config_before != EVAL_CONFIG_SHA256:
        raise RuntimeError("protected input identity failed before preflight")

    level_results = {}
    for level in ("C0", "C1", "C2"):
        env = make_training_exact_state_curriculum_env(
            make_peg_state_env,
            SNAPSHOTS,
            level=level,
            render_mode="rgb_array",
            max_episode_steps=500,
        )
        try:
            observation, info = env.reset(seed=0)
            with np.load(SNAPSHOTS / f"{level.lower()}_exact_state.npz", allow_pickle=False) as loaded:
                expected_obs = loaded["boundary_observation"]
                expected_state = loaded["mujoco_state_vector"]
            actual_state = np.empty(mujoco.mj_stateSize(env.env.env.model, STATE_SPEC))
            mujoco.mj_getState(env.env.env.model, env.env.env.data, actual_state, STATE_SPEC)
            level_results[level] = {
                "observation_max_delta": float(np.max(np.abs(observation - expected_obs))),
                "integration_state_max_delta": float(np.max(np.abs(actual_state - expected_state))),
                "goal_matches_manifest": bool(np.array_equal(
                    env.env.env._desired_goal,
                    np.load(SNAPSHOTS / f"{level.lower()}_exact_state.npz")["raw_desired_goal"],
                )),
                "curriculum_restored": bool(info["curriculum_restored"]),
                "status": "PASS" if np.array_equal(observation, expected_obs)
                and np.array_equal(actual_state, expected_state) else "FAIL",
            }
        finally:
            env.close()

    curriculum_c3 = make_training_exact_state_curriculum_env(
        make_peg_state_env,
        SNAPSHOTS,
        level="C3",
        render_mode="rgb_array",
        max_episode_steps=500,
    )
    canonical_c3 = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
    try:
        curriculum_obs, curriculum_info = curriculum_c3.reset(seed=0)
        canonical_obs, canonical_info = canonical_c3.reset(seed=0)
        c3_delta = float(np.max(np.abs(curriculum_obs - canonical_obs)))
        c3_goal_delta = float(np.max(np.abs(
            curriculum_c3.env.env._desired_goal - canonical_c3.env._desired_goal
        )))
    finally:
        curriculum_c3.close()
        canonical_c3.close()

    eval_entries = [
        ROOT / "training/peg_insertion/run_peg_training.py",
        ROOT / "training/peg_insertion/formal_policy.py",
        ROOT / "training/peg_insertion/formal_preflight.py",
    ]
    eval_imports = {str(path.relative_to(ROOT)): imports_curriculum(path) for path in eval_entries}
    all_levels = all(item["status"] == "PASS" and item["goal_matches_manifest"]
                     for item in level_results.values())
    c3_pass = c3_delta == 0.0 and c3_goal_delta == 0.0 and not curriculum_info["curriculum_restored"]
    eval_separate = not any(eval_imports.values())
    protected_ok = sha256(DEMO) == demo_before and sha256(EVAL_CONFIG) == config_before
    passed = all_levels and c3_pass and eval_separate and protected_ok

    result = {
        "TRAINING_CURRICULUM_STATIC_PREFLIGHT": "PASS" if passed else "FAIL",
        "CURRICULUM_TRAINING_READY": "YES" if passed else "NO",
        "CANONICAL_SNAPSHOT_BOUNDARY": SNAPSHOT_BOUNDARY,
        "LEVEL_RESULTS": level_results,
        "C3_CANONICAL_RESET_OBSERVATION_MAX_DELTA": c3_delta,
        "C3_CANONICAL_GOAL_MAX_DELTA": c3_goal_delta,
        "C3_CALLS_ORIGINAL_CANONICAL_RESET": "YES" if c3_pass else "NO",
        "TRAIN_ENV_FACTORY": f"{make_training_exact_state_curriculum_env.__module__}.{make_training_exact_state_curriculum_env.__name__}",
        "TRAIN_ENV_WRAPPER": f"{TrainingOnlyExactStateCurriculum.__module__}.{TrainingOnlyExactStateCurriculum.__name__}",
        "EVAL_ENV_FACTORY": f"{make_peg_state_env.__module__}.{make_peg_state_env.__name__}",
        "EVAL_ENTRY_CURRICULUM_IMPORTS": eval_imports,
        "EVAL_RESET_CANONICAL": "YES" if eval_separate else "NO",
        "EVAL_GOAL_CANONICAL": "YES" if config_before == EVAL_CONFIG_SHA256 else "NO",
        "EVAL_CURRICULUM_ACTIVE": "NO" if eval_separate else "YES",
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES" if sha256(DEMO) == PRODUCTION_DEMO_SHA256 else "NO",
        "ONLINE_ADMISSION_THRESHOLD": THRESHOLD,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO" if protected_ok else "YES",
        "NEXT": "STOP_BEFORE_TRAINING" if passed else "STOP_PREFLIGHT_FAILED",
    }
    output = unique_output()
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    print(output / "result.json")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
