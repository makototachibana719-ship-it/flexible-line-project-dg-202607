#!/usr/bin/env python3
"""Diagnose and close Peg F3 V2 exact curriculum-state restoration.

No training is performed.  Successful expert trajectories are replayed in
fresh canonical environments.  The script compares full mjData copying at the
original post-observation boundary with official mjSTATE_INTEGRATION round
trips at an explicitly forwarded snapshot boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
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
PROVENANCE = ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json"
OLD_FEASIBILITY = ROOT / "results/closure/peg/f3_v2_curriculum_feasibility_20260907_003/result.json"
OUT_PARENT = ROOT / "results/closure/peg"
LEVEL_TYPES = {"C0": "standard", "C1": "rotation_roll", "C2": "standard"}
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
OBS_SOURCES = {
    0: "tcp_position_x", 1: "tcp_position_y", 2: "tcp_position_z",
    3: "tcp_quaternion_w", 4: "tcp_quaternion_x", 5: "tcp_quaternion_y", 6: "tcp_quaternion_z",
    7: "wrench_force_x", 8: "wrench_force_y", 9: "wrench_force_z",
    10: "wrench_torque_x", 11: "wrench_torque_y", 12: "wrench_torque_z",
    13: "relative_goal_position_x", 14: "relative_goal_position_y", 15: "relative_goal_position_z",
    16: "goal_quaternion_w", 17: "goal_quaternion_x", 18: "goal_quaternion_y", 19: "goal_quaternion_z",
}
NON_WRENCH = np.r_[0:7, 13:20]


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
        out = OUT_PARENT / f"f3_v2_exact_state_restore_20260907_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique exact-restore output")


def trajectory(kind: str) -> tuple[list[dict], Path, str]:
    entries = json.loads(PROVENANCE.read_text())
    entry = next(row for row in entries if row["type"] == kind)
    path = ROOT / entry["episode_path"] / "transitions.jsonl"
    require(sha(path) == entry["transitions_sha256"], f"{kind} source SHA256")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(rows and rows[-1]["success"] and not rows[-1]["truncated"], f"{kind} successful source")
    return rows, path, entry["transitions_sha256"]


def capture_index(level: str, rows: list[dict]) -> int:
    if level == "C0":
        candidates = [i for i, row in enumerate(rows[:-1])
                      if row["after"]["metrics"]["lateral_error"] <= 0.010
                      and row["after"]["metrics"]["axial_error"] <= 0.020]
    else:
        phase = "orientation_correction" if level == "C1" else "approach"
        candidates = [i for i, row in enumerate(rows[:-1]) if row["phase"] == phase]
    require(candidates, f"{level} capture point")
    return candidates[0] if level == "C0" else candidates[len(candidates) // 2]


def action(row) -> np.ndarray:
    value = np.asarray(row["action"], dtype=np.float32)
    require(value.shape == (7,) and np.isfinite(value).all() and np.max(np.abs(value)) <= 1.0,
            "expert action contract")
    return value


def obs(env) -> np.ndarray:
    return env._state(env.env._get_obs())


def python_state(env) -> dict:
    raw = env.env
    task = raw.task
    controller = raw.controller
    return {
        "adapter_target_pose": np.asarray(env._target_pose).copy(),
        "adapter_workspace_low": np.asarray(env._workspace_low).copy(),
        "adapter_workspace_high": np.asarray(env._workspace_high).copy(),
        "raw_elapsed_steps": int(raw._elapsed_steps),
        "raw_episode_count": int(raw._episode_count),
        "raw_desired_goal": np.asarray(raw._desired_goal).copy(),
        "task_current_step": int(task.current_step),
        "task_is_success": bool(task.is_success),
        "task_insertion_axis": np.asarray(task._insertion_axis).copy(),
        "task_last_metrics": copy.deepcopy(getattr(task, "last_metrics", None)),
        "controller_q_initial": np.asarray(controller.q_initial).copy(),
    }


def restore_python_state(env, state: dict) -> None:
    raw = env.env
    env._target_pose = state["adapter_target_pose"].copy()
    env._workspace_low = state["adapter_workspace_low"].copy()
    env._workspace_high = state["adapter_workspace_high"].copy()
    raw._elapsed_steps = state["raw_elapsed_steps"]
    raw._episode_count = state["raw_episode_count"]
    raw._desired_goal = state["raw_desired_goal"].copy()
    raw.task.current_step = state["task_current_step"]
    raw.task.is_success = state["task_is_success"]
    raw.task._insertion_axis = state["task_insertion_axis"].copy()
    raw.task.last_metrics = copy.deepcopy(state["task_last_metrics"])
    raw.controller.q_initial = state["controller_q_initial"].copy()


def state_inventory(env) -> dict:
    raw = env.env
    controller = raw.controller
    joint = controller.joint_controller
    integration_members = [
        "time", "qpos", "qvel", "act", "plugin_state", "qacc_warmstart", "ctrl",
        "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat", "userdata",
    ]
    present = {name: bool(hasattr(raw.data, name)) for name in integration_members}
    return {
        "MUJOCO_INTEGRATION_STATE": present,
        "ENV_PYTHON_STATE": ["_elapsed_steps", "_episode_count", "_desired_goal"],
        "CONTROLLER_STATE": {
            "mutable": ["q_initial"],
            "configuration_only": ["ik_regularization", "ik_radius", "joint_limits",
                                   "joint_controller.kp", "joint_controller.kd",
                                   "joint_controller.use_target_as_ctrl"],
            "runtime_types": [type(controller).__name__, type(joint).__name__],
        },
        "WRAPPER_STATE": ["_target_pose", "_workspace_low", "_workspace_high"],
        "TASK_STATE": ["current_step", "is_success", "_insertion_axis", "last_metrics"],
        "DERIVED_STATE": ["site_xpos", "site_xmat/site quaternion", "sensordata/wrench",
                          "other mjData forward outputs"],
        "WRAPPER_CHAIN": [type(env).__name__, type(raw).__name__],
        "MISSING_FROM_OLD_SNAPSHOT": ["task._insertion_axis", "task.last_metrics"],
    }


def run_to(env, rows, index) -> np.ndarray:
    current, _ = env.reset(seed=0)
    for i in range(index + 1):
        current, _reward, terminated, truncated, _info = env.step(action(rows[i]))
        require(not terminated and not truncated, "source ended before snapshot")
    return np.asarray(current, dtype=np.float32)


def step_pair(source, restored, row) -> dict:
    so, sr, st, sx, si = source.step(action(row))
    ro, rr, rt, rx, ri = restored.step(action(row))
    sqpos, rqpos = source.env.data.qpos.copy(), restored.env.data.qpos.copy()
    sqvel, rqvel = source.env.data.qvel.copy(), restored.env.data.qvel.copy()
    return {
        "source_obs": np.asarray(so), "restored_obs": np.asarray(ro),
        "source_reward": float(sr), "restored_reward": float(rr),
        "source_terminated": bool(st), "restored_terminated": bool(rt),
        "source_truncated": bool(sx), "restored_truncated": bool(rx),
        "source_success": bool(si["is_success"]), "restored_success": bool(ri["is_success"]),
        "qpos_delta": float(np.max(np.abs(sqpos - rqpos))),
        "qvel_delta": float(np.max(np.abs(sqvel - rqvel))),
        "obs_delta": float(np.max(np.abs(np.asarray(so) - np.asarray(ro)))),
        "obs_non_wrench_delta": float(np.max(np.abs(np.asarray(so)[NON_WRENCH] - np.asarray(ro)[NON_WRENCH]))),
        "reward_delta": abs(float(sr) - float(rr)),
    }


def compact_step(value: dict) -> dict:
    return {key: item for key, item in value.items() if not key.startswith("source_obs")
            and not key.startswith("restored_obs")}


def full_copy_control(rows, index) -> dict:
    source = restored = None
    try:
        source = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        restored = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        source_obs = run_to(source, rows, index)
        restored.reset(seed=0)
        state = python_state(source)
        mujoco.mj_copyData(restored.env.data, restored.env.model, source.env.data)
        restore_python_state(restored, state)
        copied_obs = obs(restored)
        current_delta = float(np.max(np.abs(source_obs[NON_WRENCH] - copied_obs[NON_WRENCH])))
        next_step = step_pair(source, restored, rows[index + 1])
        passed = current_delta == 0.0 and all(next_step[key] <= 1e-12 for key in
                                               ("obs_non_wrench_delta", "qpos_delta", "qvel_delta", "reward_delta"))
        return {"status": "PASS" if passed else "FAIL",
                "CURRENT_OBS_MAX_DELTA_NON_WRENCH": current_delta,
                "NEXT_OBS_MAX_DELTA_NON_WRENCH": next_step["obs_non_wrench_delta"],
                "NEXT_QPOS_MAX_DELTA": next_step["qpos_delta"],
                "NEXT_QVEL_MAX_DELTA": next_step["qvel_delta"],
                "REWARD_DELTA": next_step["reward_delta"]}
    finally:
        if source is not None: source.close()
        if restored is not None: restored.close()


def mismatch_dimensions(source_values, restored_values, indices, minimum=0.0) -> list[dict]:
    result = []
    for index in indices:
        delta = abs(float(source_values[index]) - float(restored_values[index]))
        if delta > minimum:
            result.append({"index": int(index), "semantic_source": OBS_SOURCES[int(index)],
                           "source_value": float(source_values[index]),
                           "restored_value": float(restored_values[index]),
                           "absolute_delta": delta})
    return result


def save_snapshot(out, level, env, state_vector, state, source_path, source_sha, index, boundary_obs):
    npz = out / f"{level.lower()}_exact_state.npz"
    np.savez_compressed(
        npz,
        mujoco_state_vector=state_vector,
        adapter_target_pose=state["adapter_target_pose"],
        adapter_workspace_low=state["adapter_workspace_low"],
        adapter_workspace_high=state["adapter_workspace_high"],
        raw_desired_goal=state["raw_desired_goal"],
        task_insertion_axis=state["task_insertion_axis"],
        controller_q_initial=state["controller_q_initial"],
        boundary_observation=boundary_obs,
    )
    manifest = {
        "MUJOCO_STATE_SPEC": "mjSTATE_INTEGRATION",
        "MUJOCO_STATE_SPEC_VALUE": int(STATE_SPEC),
        "MUJOCO_STATE_VECTOR_SIZE": int(state_vector.size),
        "ENV_STATE": {"raw_elapsed_steps": state["raw_elapsed_steps"],
                      "raw_episode_count": state["raw_episode_count"],
                      "raw_desired_goal": state["raw_desired_goal"].tolist()},
        "CONTROLLER_STATE": {"q_initial": state["controller_q_initial"].tolist()},
        "WRAPPER_STATE": {key: state[key].tolist() for key in
                          ("adapter_target_pose", "adapter_workspace_low", "adapter_workspace_high")},
        "TASK_STATE": {"current_step": state["task_current_step"],
                       "is_success": state["task_is_success"],
                       "insertion_axis": state["task_insertion_axis"].tolist(),
                       "last_metrics": state["task_last_metrics"]},
        "source_trajectory": str(source_path), "source_trajectory_sha256": source_sha,
        "source_transition_index": index + 1, "next_expert_index": index + 1,
        "snapshot_boundary": "AFTER_ENV_STEP_RETURN_THEN_MJ_FORWARD_THEN_OBSERVATION",
        "model_environment_binding_sha256": recorder_bindings(ROOT)["env_hash"],
        "canonical_eval_config_sha256": EVAL_SHA,
        "production_demo_sha256": DEMO_SHA,
        "observation_contract_version": V2,
        "canonical_goal_identity": hashlib.sha256(state["raw_desired_goal"].tobytes()).hexdigest(),
        "snapshot_file": str(npz), "snapshot_file_sha256": sha(npz),
    }
    dump(out / f"{level.lower()}_exact_state_manifest.json", manifest)
    return manifest


def integration_control(out, level, rows, source_path, source_sha, index) -> dict:
    source = restored = None
    try:
        source = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        restored = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        returned_obs = run_to(source, rows, index)
        state = python_state(source)
        mujoco.mj_forward(source.env.model, source.env.data)
        source_boundary_obs = obs(source)
        size = mujoco.mj_stateSize(source.env.model, STATE_SPEC)
        vector = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(source.env.model, source.env.data, vector, STATE_SPEC)
        manifest = save_snapshot(out, level, source, vector, state, source_path, source_sha,
                                 index, source_boundary_obs)
        restored.reset(seed=0)
        mujoco.mj_setState(restored.env.model, restored.env.data, vector, STATE_SPEC)
        restore_python_state(restored, state)
        restored_vector = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(restored.env.model, restored.env.data, restored_vector, STATE_SPEC)
        integration_delta = float(np.max(np.abs(vector - restored_vector)))
        mujoco.mj_forward(restored.env.model, restored.env.data)
        restored_boundary_obs = obs(restored)
        immediate_non_wrench = float(np.max(np.abs(
            source_boundary_obs[NON_WRENCH] - restored_boundary_obs[NON_WRENCH])))
        immediate_all = float(np.max(np.abs(source_boundary_obs - restored_boundary_obs)))
        old_boundary_delta = float(np.max(np.abs(returned_obs[NON_WRENCH] - source_boundary_obs[NON_WRENCH])))
        old_mismatches = mismatch_dimensions(returned_obs, source_boundary_obs, NON_WRENCH)
        canonical_mismatches = mismatch_dimensions(source_boundary_obs, restored_boundary_obs, range(20))
        suffix = {}
        maxima = {"obs": 0.0, "obs_non_wrench": 0.0, "qpos": 0.0, "qvel": 0.0, "reward": 0.0}
        source_success = restored_success = False
        source_terminated = restored_terminated = False
        checkpoints = {1, 5, 10, 25}
        for offset, row in enumerate(rows[index + 1:], 1):
            step = step_pair(source, restored, row)
            maxima["obs"] = max(maxima["obs"], step["obs_delta"])
            maxima["obs_non_wrench"] = max(maxima["obs_non_wrench"], step["obs_non_wrench_delta"])
            maxima["qpos"] = max(maxima["qpos"], step["qpos_delta"])
            maxima["qvel"] = max(maxima["qvel"], step["qvel_delta"])
            maxima["reward"] = max(maxima["reward"], step["reward_delta"])
            source_success |= step["source_success"]
            restored_success |= step["restored_success"]
            source_terminated = step["source_terminated"]
            restored_terminated = step["restored_terminated"]
            if offset in checkpoints:
                suffix[str(offset)] = compact_step(step)
            if source_terminated or restored_terminated or step["source_truncated"] or step["restored_truncated"]:
                break
        single = suffix["1"]
        single_pass = all(single[key] <= 1e-12 for key in
                          ("obs_delta", "qpos_delta", "qvel_delta", "reward_delta"))
        suffix_pass = (all(value <= 1e-12 for value in maxima.values())
                       and source_success and restored_success
                       and source_terminated == restored_terminated)
        passed = integration_delta == 0.0 and immediate_all == 0.0 and single_pass and suffix_pass
        return {
            "status": "PASS" if passed else "FAIL",
            "INTEGRATION_STATE_SIZE": int(size),
            "INTEGRATION_STATE_MAX_DELTA_AFTER_RESTORE": integration_delta,
            "IMMEDIATE_OBS_MAX_DELTA": immediate_all,
            "IMMEDIATE_NON_WRENCH_MAX_DELTA": immediate_non_wrench,
            "OLD_BOUNDARY_TO_FORWARDED_NON_WRENCH_MAX_DELTA": old_boundary_delta,
            "OLD_BOUNDARY_MISMATCH_DIMENSIONS": old_mismatches,
            "CANONICAL_BOUNDARY_MISMATCH_DIMENSIONS": canonical_mismatches,
            "WRENCH_REPRODUCTION_STATUS": "PASS" if immediate_all == immediate_non_wrench == 0.0 else
                                          "PASS" if np.max(np.abs(source_boundary_obs[7:13] - restored_boundary_obs[7:13])) == 0 else "FAIL",
            "single_step": single,
            "suffix_checkpoints": suffix,
            "suffix_maxima": maxima,
            "source_success": source_success,
            "restored_success": restored_success,
            "RESTORE_SINGLE_STEP": "PASS" if single_pass else "FAIL",
            "RESTORE_SUFFIX_REPRODUCTION": "PASS" if suffix_pass else "FAIL",
            "snapshot_manifest": manifest,
        }
    finally:
        if source is not None: source.close()
        if restored is not None: restored.close()


def main() -> int:
    require(Path.cwd() == ROOT, "run from canonical project root")
    require(sha(DEMO) == DEMO_SHA and sha(EVAL_CONFIG) == EVAL_SHA, "protected hashes")
    old = json.loads(OLD_FEASIBILITY.read_text())
    require(old["TRAINING_ONLY_CURRICULUM_FEASIBILITY"] == "FAIL", "old failure binding")
    sources = {level: trajectory(kind) for level, kind in LEVEL_TYPES.items()}
    protected = [DEMO, EVAL_CONFIG, PROVENANCE, OLD_FEASIBILITY]
    protected.extend(value[1] for value in sources.values())
    protected.extend(ROOT / relative for relative in recorder_bindings(ROOT)["files"])
    protected = sorted(set(protected), key=str)
    before = {str(path): sha(path) for path in protected}
    out = unique_output()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "protected_before.json", before)
    started = time.monotonic()
    with portable_peg_runtime(ROOT):
        probe = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        try:
            inventory = state_inventory(probe)
            require(probe.observation_contract_version == V2, "V2 contract")
            require(probe.env.task.position_tolerance == .002 and
                    probe.env.task.orientation_tolerance == .05, "success contract")
        finally:
            probe.close()
        full_copy = {}
        integration = {}
        for level, (rows, source_path, source_sha) in sources.items():
            index = capture_index(level, rows)
            full_copy[level] = full_copy_control(rows, index)
            integration[level] = integration_control(out, level, rows, source_path, source_sha, index)
    full_copy_pass = all(row["status"] == "PASS" for row in full_copy.values())
    integration_pass = all(row["status"] == "PASS" for row in integration.values())
    timing_proven = full_copy_pass and integration_pass and all(
        row["OLD_BOUNDARY_TO_FORWARDED_NON_WRENCH_MAX_DELTA"] > 0 for row in integration.values())
    after = {str(path): sha(path) for path in protected}
    require(before == after and after[str(DEMO)] == DEMO_SHA, "protected artifact changed")
    result = {
        "EXACT_STATE_RESTORE_REVIEW": "PASS" if timing_proven else "FAIL",
        "CANONICAL_SNAPSHOT_BOUNDARY": "AFTER_ENV_STEP_RETURN_THEN_MJ_FORWARD_THEN_OBSERVATION",
        "SOURCE_OLD_BOUNDARY": "AFTER_ENV_STEP_RETURN_USING_STEP_GENERATED_OBSERVATION_BEFORE_EXTRA_MJ_FORWARD",
        "STATE_INVENTORY": inventory,
        "MISSING_FROM_CURRENT_SNAPSHOT": inventory["MISSING_FROM_OLD_SNAPSHOT"],
        "FULL_MJDATA_COPY_CONTROL": "PASS" if full_copy_pass else "FAIL",
        "FULL_MJDATA_COPY_BY_LEVEL": full_copy,
        "INTEGRATION_CONTROL_BY_LEVEL": integration,
        "INTEGRATION_STATE_SIZE": {level: row["INTEGRATION_STATE_SIZE"] for level, row in integration.items()},
        "INTEGRATION_STATE_MAX_DELTA_AFTER_RESTORE": {
            level: row["INTEGRATION_STATE_MAX_DELTA_AFTER_RESTORE"] for level, row in integration.items()},
        "PRIMARY_RESTORE_MISMATCH_SOURCE": "OBSERVATION_TIMING" if timing_proven else "UNRESOLVED",
        "OBS_MISMATCH_DIMENSIONS": {
            level: row["OLD_BOUNDARY_MISMATCH_DIMENSIONS"] for level, row in integration.items()},
        "WRENCH_REPRODUCTION_STATUS": "PASS" if all(
            row["WRENCH_REPRODUCTION_STATUS"] == "PASS" for row in integration.values()) else "FAIL",
        "C0_IMMEDIATE_NON_WRENCH_MAX_DELTA": integration["C0"]["IMMEDIATE_NON_WRENCH_MAX_DELTA"],
        "C1_IMMEDIATE_NON_WRENCH_MAX_DELTA": integration["C1"]["IMMEDIATE_NON_WRENCH_MAX_DELTA"],
        "C2_IMMEDIATE_NON_WRENCH_MAX_DELTA": integration["C2"]["IMMEDIATE_NON_WRENCH_MAX_DELTA"],
        "C0_NEXT_TRANSITION_MAX_DELTA": integration["C0"]["single_step"]["obs_delta"],
        "C1_NEXT_TRANSITION_MAX_DELTA": integration["C1"]["single_step"]["obs_delta"],
        "C2_NEXT_TRANSITION_MAX_DELTA": integration["C2"]["single_step"]["obs_delta"],
        "RESTORE_SINGLE_STEP": "PASS" if all(row["RESTORE_SINGLE_STEP"] == "PASS" for row in integration.values()) else "FAIL",
        "RESTORE_SUFFIX_REPRODUCTION": "PASS" if all(row["RESTORE_SUFFIX_REPRODUCTION"] == "PASS" for row in integration.values()) else "FAIL",
        "TRAINING_CURRICULUM_STATIC_PREFLIGHT": "NOT_RUN" if not timing_proven else "READY",
        "CURRICULUM_TRAINING_READY": "NO",
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
        "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "EVAL_RESET_CANONICAL": "YES",
        "EVAL_GOAL_CANONICAL": "YES",
        "EVAL_CURRICULUM_ACTIVE": "NO",
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "FIRST_BAD_GATE": "NONE_EXACT_RESTORE" if timing_proven else "EXACT_STATE_RESTORE",
        "PRIMARY_ROOT_CAUSE": "BOUNDARY_MISMATCH_PRE_FORWARD_SOURCE_VS_POST_FORWARD_RESTORE" if timing_proven else "UNRESOLVED",
        "NEXT": "TRAINING_ONLY_CURRICULUM_STATIC_PREFLIGHT" if timing_proven else "EXACT_STATE_RESTORE_REVIEW",
        "ELAPSED_SECONDS": time.monotonic() - started,
    }
    dump(out / "protected_after.json", after)
    dump(out / "result.json", result)
    keys = ("EXACT_STATE_RESTORE_REVIEW", "CANONICAL_SNAPSHOT_BOUNDARY", "FULL_MJDATA_COPY_CONTROL",
            "PRIMARY_RESTORE_MISMATCH_SOURCE", "WRENCH_REPRODUCTION_STATUS",
            "C0_IMMEDIATE_NON_WRENCH_MAX_DELTA", "C1_IMMEDIATE_NON_WRENCH_MAX_DELTA",
            "C2_IMMEDIATE_NON_WRENCH_MAX_DELTA", "C0_NEXT_TRANSITION_MAX_DELTA",
            "C1_NEXT_TRANSITION_MAX_DELTA", "C2_NEXT_TRANSITION_MAX_DELTA",
            "RESTORE_SINGLE_STEP", "RESTORE_SUFFIX_REPRODUCTION", "NEXT")
    print(json.dumps({key: result[key] for key in keys}, sort_keys=True))
    print(str(out / "result.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
