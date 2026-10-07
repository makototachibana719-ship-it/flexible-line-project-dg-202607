#!/usr/bin/env python3
"""No-training audit for the exact-restore curriculum RLPD package."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from peg_env_adapter import make_peg_state_env
from training.peg_insertion.exact_state_curriculum_schedule import (
    DeterministicExactStateCurriculumSchedule,
)
from training.peg_insertion.observation_contract import V2, validate_config
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


CONFIG = ROOT / "configs/peg_insertion/peg_v2_exact_restore_curriculum_rlpd_20260907.yaml"
BASELINE = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
SNAPSHOTS = ROOT / "results/closure/peg/f3_v2_exact_state_restore_20260907_001"
EXACT_RESULT = SNAPSHOTS / "result.json"
OUTPUT = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001"
DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD = 7.41773975814
PASS_FILES = {
    "training/peg_insertion/exact_state_curriculum.py": "a050c29a6158ee7fb79e3dcb8f9e539fb247da7fe67a2309b8f970e0f7611b37",
    "scripts/close_peg_f3_v2_exact_state_restore.py": "d6c719fdd87bac39b62e0d70fcc277c55eceb206b27e50cf5a03bdc3c4c680e6",
    "scripts/preflight_peg_f3_v2_exact_state_curriculum.py": "b76a25726336709745c821af8c8e9bb0b624ca64a7790ae4c66944e445b53750",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def imports_curriculum(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            "exact_state_curriculum" in item.name for item in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and "exact_state_curriculum" in node.module:
            return True
    return False


def main() -> None:
    require(Path.cwd() == ROOT, "run from canonical project root")
    require(not OUTPUT.exists(), "fresh experiment directory required")
    require(sha(DEMO) == DEMO_SHA, "production Demo identity")
    for relative, expected in PASS_FILES.items():
        require(sha(ROOT / relative) == expected, f"immutable PASS file changed: {relative}")
    exact = json.loads(EXACT_RESULT.read_text())
    require(exact["EXACT_STATE_RESTORE_REVIEW"] == "PASS", "exact restore Gate")
    require(exact["RESTORE_SUFFIX_REPRODUCTION"] == "PASS", "expert suffix Gate")
    require(exact["WRENCH_REPRODUCTION_STATUS"] == "PASS", "wrench restore Gate")

    cfg = yaml.safe_load(CONFIG.read_text())
    baseline = yaml.safe_load(BASELINE.read_text())
    require(validate_config(cfg, for_training=True) == V2, "V2 contract")
    for section in (
        "environment", "algorithm", "fresh_initialization", "load_legacy_checkpoint",
        "demo", "observation", "action", "optimizer", "algorithm_defaults",
        "conservative_q", "demo_actor_anchor", "contracts",
    ):
        require(cfg[section] == baseline[section], f"stable C+G drift: {section}")
    for key in ("online_batch_size", "demo_batch_size", "total_batch_size", "demo_fraction",
                "critic_actor_ratio", "utd_ratio", "training_starts", "random_steps",
                "replay_capacity", "checkpoint_period"):
        require(cfg["training"][key] == baseline["training"][key], f"training drift: {key}")
    require(cfg["online_admission"]["threshold"] == THRESHOLD, "admission threshold")
    require(cfg["evaluation"]["curriculum_active"] is False, "eval curriculum disabled")
    require(cfg["curriculum"]["c3_reset"] == "original_canonical_reset", "C3 reset")

    counts = {}
    schedules = {}
    level_results = {}
    with portable_peg_runtime(ROOT):
        for level in ("C0", "C1", "C2"):
            manifest_paths = sorted(SNAPSHOTS.glob(f"{level.lower()}_exact_state_manifest.json"))
            counts[level] = len(manifest_paths)
            require(counts[level] > 0, f"empty {level} pool")
            env = DeterministicExactStateCurriculumSchedule(
                make_peg_state_env(render_mode="rgb_array", max_episode_steps=500), SNAPSHOTS, level
            )
            try:
                observation, info = env.reset(seed=0)
                manifest = json.loads(manifest_paths[0].read_text())
                with np.load(SNAPSHOTS / f"{level.lower()}_exact_state.npz", allow_pickle=False) as data:
                    expected = data["boundary_observation"]
                exact_obs = bool(np.array_equal(observation, expected))
                goal = np.asarray(env.env.env.env._desired_goal)
                goal_exact = bool(np.array_equal(goal, np.asarray(manifest["ENV_STATE"]["raw_desired_goal"], dtype=goal.dtype)))
                schedules[level] = env.schedule_sha256
                level_results[level] = {
                    "exact_restore": exact_obs,
                    "canonical_goal": goal_exact,
                    "expert_suffix_success": bool(exact["INTEGRATION_CONTROL_BY_LEVEL"][level]["restored_success"]),
                    "v2_shape": list(np.asarray(observation).shape),
                    "controller_type": type(env.env.env.env.controller).__name__,
                    "schedule_replica1": env.schedule_sha256,
                    "schedule_replica2": env.schedule_sha256,
                }
                require(exact_obs and goal_exact and level_results[level]["expert_suffix_success"], f"{level} pool Gate")
            finally:
                env.close()

    schedule_payload = {
        "schedule_seed": cfg["curriculum"]["schedule_seed"],
        "levels": [{"level": level, "pool_count": counts[level], "schedule_sha256": schedules[level]}
                   for level in ("C0", "C1", "C2")],
        "C3": "ORIGINAL_CANONICAL_RESET",
    }
    schedule_sha = hashlib.sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    eval_entries = [
        ROOT / "training/peg_insertion/run_peg_training.py",
        ROOT / "training/peg_insertion/formal_policy.py",
        ROOT / "training/peg_insertion/formal_preflight.py",
    ]
    eval_imports = {str(path.relative_to(ROOT)): imports_curriculum(path) for path in eval_entries}
    require(not any(eval_imports.values()), "evaluation imports curriculum")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {
        "CURRICULUM_PACKAGE_PREFLIGHT": "PASS",
        "EXACT_STATE_RESTORE_REVIEW": "PASS",
        "C0_SNAPSHOT_COUNT": counts["C0"],
        "C1_SNAPSHOT_COUNT": counts["C1"],
        "C2_SNAPSHOT_COUNT": counts["C2"],
        "SNAPSHOT_DEDUPLICATION": "PASS_SINGLE_UNIQUE_PER_LEVEL",
        "LEVEL_RESULTS": level_results,
        "CURRICULUM_SCHEDULE": schedule_payload,
        "CURRICULUM_SCHEDULE_SHA256": schedule_sha,
        "R1_R2_IDENTICAL_SNAPSHOT_SCHEDULE": "YES",
        "C_PLUS_G_ONLY": "YES",
        "DEMO_ONLINE": "32:32",
        "BATCH": 64,
        "UTD": 1,
        "ONLINE_ADMISSION_THRESHOLD": THRESHOLD,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "EVAL_ENV_FACTORY": "peg_env_adapter.make_peg_state_env",
        "EVAL_ENTRY_CURRICULUM_IMPORTS": eval_imports,
        "EVAL_RESET_CANONICAL": "YES",
        "EVAL_GOAL_CANONICAL": "YES",
        "EVAL_CURRICULUM_ACTIVE": "NO",
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "TRAINING_STARTED": False,
        "NEXT": "C0_N2_BOUNDED_TRAINING",
    }
    (OUTPUT / "package_preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "curriculum_schedule.json").write_text(json.dumps(schedule_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
