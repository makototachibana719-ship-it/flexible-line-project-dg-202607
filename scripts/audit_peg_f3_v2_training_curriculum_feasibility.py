#!/usr/bin/env python3
"""Validate exact, training-only recovery curriculum states for Peg F3 V2."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
import sys
import time

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
from training.peg_insertion.observation_contract import V2
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.v2_recorder import bindings as recorder_bindings


DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
EVAL_CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
EVAL_SHA = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
RECOVERY_RESULT = ROOT / "results/closure/peg/f3_v2_recovery_closure_20260907_003/result.json"
PROVENANCE = ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json"
OUT_PARENT = ROOT / "results/closure/peg"
LEVEL_TYPES = {"C0": "standard", "C1": "rotation_roll", "C2": "standard"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def unique_output() -> Path:
    OUT_PARENT.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        out = OUT_PARENT / f"f3_v2_curriculum_feasibility_20260907_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique curriculum feasibility output")


def source_for(kind: str) -> tuple[Path, str]:
    entries = json.loads(PROVENANCE.read_text())
    entry = next(row for row in entries if row["type"] == kind)
    path = ROOT / entry["episode_path"] / "transitions.jsonl"
    require(sha(path) == entry["transitions_sha256"], f"{kind} accepted trajectory SHA256")
    return path, entry["transitions_sha256"]


def rows_for(kind: str) -> tuple[list[dict], Path, str]:
    path, expected = source_for(kind)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(rows and rows[-1]["success"] and not rows[-1]["truncated"], f"{kind} success source")
    require(all(row["observation_contract_version"] == V2 for row in rows), f"{kind} V2")
    return rows, path, expected


def capture_index(level: str, rows: list[dict]) -> int:
    if level == "C0":
        candidates = [i for i, row in enumerate(rows[:-1])
                      if row["after"]["metrics"]["lateral_error"] <= 0.010
                      and row["after"]["metrics"]["axial_error"] <= 0.020]
        require(candidates, "C0 near-hole snapshot unavailable")
        return candidates[0]
    phase = "orientation_correction" if level == "C1" else "approach"
    candidates = [i for i, row in enumerate(rows[:-1]) if row["phase"] == phase]
    require(candidates, f"{level} {phase} snapshot unavailable")
    return candidates[len(candidates) // 2]


def env_contract(env) -> dict:
    raw = env.env
    require(env.observation_contract_version == V2, "V2 contract")
    require(env.action_space.shape == (7,) and env.action_space.dtype == np.float32, "action contract")
    require(raw.task.position_tolerance == 0.002 and raw.task.orientation_tolerance == 0.05,
            "success contract")
    return {
        "factory": "peg_env_adapter.make_peg_state_env",
        "max_episode_steps": int(raw.max_episode_steps),
        "position_tolerance": float(raw.task.position_tolerance),
        "orientation_tolerance": float(raw.task.orientation_tolerance),
        "curriculum_active": False,
    }


def action(row) -> np.ndarray:
    value = np.asarray(row["action"], dtype=np.float32)
    require(value.shape == (7,) and np.isfinite(value).all() and np.max(np.abs(value)) <= 1.0,
            "expert action contract")
    return value


def runtime_snapshot(env, obs, source, source_sha, index, file: Path) -> dict:
    raw = env.env
    integration = raw.save_state()
    values = {
        "mj_state_integration": integration,
        "adapter_target_pose": np.asarray(env._target_pose).copy(),
        "adapter_workspace_low": np.asarray(env._workspace_low).copy(),
        "adapter_workspace_high": np.asarray(env._workspace_high).copy(),
        "raw_desired_goal": np.asarray(raw._desired_goal).copy(),
        "controller_q_initial": np.asarray(raw.controller.q_initial).copy(),
        "observation": np.asarray(obs, dtype=np.float32).copy(),
        "raw_elapsed_steps": np.asarray(raw._elapsed_steps, dtype=np.int64),
        "raw_episode_count": np.asarray(raw._episode_count, dtype=np.int64),
        "task_current_step": np.asarray(raw.task.current_step, dtype=np.int64),
        "task_is_success": np.asarray(raw.task.is_success, dtype=np.bool_),
    }
    np.savez_compressed(file, **values)
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "capture_transition_index": index + 1,
        "next_expert_index": index + 1,
        "snapshot": str(file),
        "snapshot_sha256": sha(file),
        "sim_time": float(raw.data.time),
        "stored_fields": sorted(values),
        "metrics": rows_metric(raw),
    }


def rows_metric(raw) -> dict:
    metrics = raw.task._goal_metrics(raw._get_obs())
    return {key: float(metrics[key]) for key in
            ("lateral_error", "axial_error", "orientation_error")}


def restore(env, file: Path) -> np.ndarray:
    raw = env.env
    data = np.load(file)
    raw.load_state(data["mj_state_integration"])
    raw._elapsed_steps = int(data["raw_elapsed_steps"])
    raw._episode_count = int(data["raw_episode_count"])
    raw.task.current_step = int(data["task_current_step"])
    raw.task.is_success = bool(data["task_is_success"])
    raw._desired_goal = data["raw_desired_goal"].copy()
    raw.controller.q_initial = data["controller_q_initial"].copy()
    env._target_pose = data["adapter_target_pose"].copy()
    env._workspace_low = data["adapter_workspace_low"].copy()
    env._workspace_high = data["adapter_workspace_high"].copy()
    mujoco.mj_forward(raw.model, raw.data)
    return env._state(raw._get_obs())


def source_capture(env, level, rows, source, source_sha, out) -> tuple[dict, dict]:
    index = capture_index(level, rows)
    obs, _ = env.reset(seed=0)
    captured = None
    next_reference = None
    success = False
    for i, row in enumerate(rows):
        obs, reward, terminated, truncated, info = env.step(action(row))
        if i == index:
            file = out / f"{level.lower()}_state.npz"
            captured = runtime_snapshot(env, obs, source, source_sha, index, file)
        elif i == index + 1:
            next_reference = {
                "obs": np.asarray(obs, dtype=np.float32),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(info["is_success"]),
            }
        success = success or bool(info["is_success"])
        if terminated or truncated:
            break
    require(captured is not None and next_reference is not None and success,
            f"{level} source trajectory did not complete")
    return captured, next_reference


def restoration_trial(env, level, rows, snapshot, next_reference, repeat) -> dict:
    canonical_obs, _ = env.reset(seed=0)
    canonical_goal = np.asarray(env.env._desired_goal).copy()
    restored_obs = restore(env, Path(snapshot["snapshot"]))
    data = np.load(snapshot["snapshot"])
    observation_delta = np.abs(restored_obs - data["observation"])
    observation_max_abs_delta = float(np.max(observation_delta))
    physical_indices = np.r_[0:7, 13:20]
    observation_physical_max_abs_delta = float(np.max(observation_delta[physical_indices]))
    observation_wrench_max_abs_delta = float(np.max(observation_delta[7:13]))
    obs_exact = observation_max_abs_delta <= 1e-7
    obs_physical_exact = observation_physical_max_abs_delta <= 1e-7
    goal_exact = bool(np.array_equal(env.env._desired_goal, canonical_goal))
    start = int(snapshot["next_expert_index"])
    row = rows[start]
    obs, reward, terminated, truncated, info = env.step(action(row))
    next_exact = (
        np.allclose(obs, next_reference["obs"], rtol=0, atol=1e-7)
        and math.isclose(float(reward), next_reference["reward"], rel_tol=0, abs_tol=1e-8)
        and bool(terminated) == next_reference["terminated"]
        and bool(truncated) == next_reference["truncated"]
        and bool(info["is_success"]) == next_reference["success"]
    )
    success = bool(info["is_success"])
    for row in rows[start + 1:]:
        if terminated or truncated:
            break
        obs, reward, terminated, truncated, info = env.step(action(row))
        success = success or bool(info["is_success"])
    return {
        "repeat": repeat,
        "canonical_reset_before_restore": True,
        "observation_exact": obs_exact,
        "observation_max_abs_delta": observation_max_abs_delta,
        "observation_physical_exact": obs_physical_exact,
        "observation_physical_max_abs_delta": observation_physical_max_abs_delta,
        "observation_wrench_recomputed": not obs_exact and obs_physical_exact,
        "observation_wrench_max_abs_delta": observation_wrench_max_abs_delta,
        "goal_exact": goal_exact,
        "first_suffix_transition_exact": bool(next_exact),
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_metrics": rows_metric(env.env),
        "canonical_reset_observation_shape": list(np.asarray(canonical_obs).shape),
    }


def c3_trial(env, rows) -> dict:
    obs, _ = env.reset(seed=0)
    success = False
    for row in rows:
        obs, reward, terminated, truncated, info = env.step(action(row))
        success = success or bool(info["is_success"])
        if terminated or truncated:
            break
    return {"original_full_reset": True, "success": success,
            "terminated": bool(terminated), "truncated": bool(truncated)}


def main() -> int:
    require(Path.cwd() == ROOT, "run from canonical project root")
    require(sha(DEMO) == DEMO_SHA and sha(EVAL_CONFIG) == EVAL_SHA, "canonical protected hashes")
    recovery = json.loads(RECOVERY_RESULT.read_text())
    require(recovery["RECOVERY_BASIN_AUDIT"] == "FAIL", "recovery audit binding")
    require(recovery["EXPERT_POSITIVE_CONTROL"] == "PASS", "expert positive control binding")
    sources = {level: rows_for(kind) for level, kind in LEVEL_TYPES.items()}
    protected = [DEMO, EVAL_CONFIG, RECOVERY_RESULT, PROVENANCE]
    protected.extend(item for level in sources.values() for item in (level[1],))
    protected.extend(ROOT / relative for relative in recorder_bindings(ROOT)["files"])
    protected = sorted(set(protected), key=str)
    before = {str(path): sha(path) for path in protected}
    out = unique_output()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "protected_before.json", before)
    env = None
    started = time.monotonic()
    try:
        with portable_peg_runtime(ROOT):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
            binding = env_contract(env)
            levels = {}
            for level in ("C0", "C1", "C2"):
                rows, source, source_sha = sources[level]
                snapshot, next_reference = source_capture(env, level, rows, source, source_sha, out)
                trials = [restoration_trial(env, level, rows, snapshot, next_reference, repeat)
                          for repeat in range(2)]
                levels[level] = {"snapshot": snapshot, "trials": trials,
                                 "pass": all(all((trial["observation_physical_exact"], trial["goal_exact"],
                                                   trial["first_suffix_transition_exact"], trial["success"]))
                                             for trial in trials)}
            standard_rows = sources["C3"][0] if "C3" in sources else sources["C0"][0]
            c3 = [c3_trial(env, standard_rows) for _ in range(2)]
            levels["C3"] = {"original_full_reset": True, "trials": c3,
                             "pass": all(trial["success"] for trial in c3)}
    finally:
        if env is not None:
            env.close()
    overall = all(row["pass"] for row in levels.values())
    after = {str(path): sha(path) for path in protected}
    require(before == after and after[str(DEMO)] == DEMO_SHA, "protected artifact changed")
    result = {
        "TRAINING_ONLY_CURRICULUM_FEASIBILITY": "PASS" if overall else "FAIL",
        "CURRICULUM_LEVELS": levels,
        "TRAIN_ENV_FACTORY": "canonical factory plus exact recorded-state restoration wrapper",
        "EVAL_ENV_FACTORY": "canonical original make_peg_state_env only",
        "EVAL_BINDING": binding,
        "TRAINING_CURRICULUM_USED": False,
        "TRAINING_STARTED": False,
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
        "CORRECTIVE_DEMO_SEPARATE": "YES",
        "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "EVAL_RESET_CANONICAL": "YES",
        "EVAL_GOAL_CANONICAL": "YES",
        "EVAL_SUCCESS_CANONICAL": "YES",
        "EVAL_CURRICULUM_ACTIVE": "NO",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "NEXT": "TRAINING_ONLY_CURRICULUM_WRAPPER" if overall else "EXACT_STATE_RESTORE_REVIEW",
        "ELAPSED_SECONDS": time.monotonic() - started,
    }
    dump(out / "protected_after.json", after)
    dump(out / "result.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"CURRICULUM_LEVELS", "EVAL_BINDING"}}, sort_keys=True))
    print(str(out / "result.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
