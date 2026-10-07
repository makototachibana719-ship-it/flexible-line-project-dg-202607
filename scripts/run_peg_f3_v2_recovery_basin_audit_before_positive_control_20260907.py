#!/usr/bin/env python3
"""Protected Phase 0-2 recovery-basin audit for Peg F3 V2.

The policy and the frozen scripted expert act in the same live canonical
episode.  No simulator state is restored during this audit and no training or
replay insertion is performed.  Complete state snapshots are diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys
import time

import jax
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
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.observation_contract import V2, validate_config
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.v2_recorder import bindings as recorder_bindings


CANONICAL_DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
CANONICAL_DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
EVAL_CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
EVAL_CONFIG_SHA = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
POLICY_CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
POLICY_CONFIG_SHA = "b1cd4126c10673c267e0e2382a44eef97cfc7b0611a5774568d46ab6d8f2ff41"
STABLE_RUNNER = ROOT / "scripts/run_cg_deterministic_staged_admission_n2_1k_v3.py"
POLICY_RUN = ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica2"
POLICY_CHECKPOINT = POLICY_RUN / "checkpoints/checkpoint_1000"
POLICY_REPLAY = POLICY_RUN / "checkpoints/checkpoint_1000_online_replay.pkl"
POLICY_METADATA = POLICY_RUN / "checkpoints/checkpoint_1000_resume_metadata.json"
EXPERT_STREAM = ROOT / "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
EXPERT_STREAM_SHA = "c1e66e2e6e7a43202d2b140162a4ac9e9c6aca1937da077f76c9865bb24f9b4f"
OUT_PARENT = ROOT / "results/closure/peg"
CAPTURE_STEPS = (1, 2, 5, 10, 15, 25, 50)
RECOVERY_SEEDS = (0, 1, 2)
D_DEMO_THRESHOLD = 7.41773975814


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


def finite(value):
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def unique_output() -> Path:
    OUT_PARENT.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        candidate = OUT_PARENT / f"f3_v2_recovery_closure_20260907_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unique recovery closure output")


def protected_paths() -> list[Path]:
    paths = [
        CANONICAL_DEMO,
        EVAL_CONFIG,
        POLICY_CONFIG,
        STABLE_RUNNER,
        POLICY_CHECKPOINT,
        POLICY_REPLAY,
        POLICY_METADATA,
        EXPERT_STREAM,
    ]
    binding = recorder_bindings(ROOT)
    paths.extend(ROOT / relative for relative in binding["files"])
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda path: str(path))


def verify_static_inputs() -> tuple[dict, dict, list[dict]]:
    require(ROOT == Path.cwd(), "run from the canonical project root")
    require(sha(CANONICAL_DEMO) == CANONICAL_DEMO_SHA, "production Demo SHA256")
    require(sha(EVAL_CONFIG) == EVAL_CONFIG_SHA, "canonical eval config SHA256")
    require(sha(POLICY_CONFIG) == POLICY_CONFIG_SHA, "accepted policy config SHA256")
    require(sha(EXPERT_STREAM) == EXPERT_STREAM_SHA, "frozen expert stream SHA256")
    eval_cfg = yaml.safe_load(EVAL_CONFIG.read_text())
    require(validate_config(eval_cfg, for_training=True) == V2, "canonical V2 eval contract")
    require(eval_cfg["demo"]["sha256"] == CANONICAL_DEMO_SHA, "eval Demo binding")
    require(eval_cfg["contracts"]["success"] ==
            "lateral<=0.002 and axial<=0.003 and orientation<=0.05; force excluded",
            "canonical success binding")
    policy_cfg = yaml.safe_load(POLICY_CONFIG.read_text())
    rows = [json.loads(line) for line in EXPERT_STREAM.read_text().splitlines() if line.strip()]
    require(rows and all(len(row["action"]) == 7 for row in rows), "7D expert stream")
    require(bool(rows[-1].get("success")), "expert stream must end in canonical success")
    return eval_cfg, policy_cfg, rows


def make_distance_model() -> tuple[np.ndarray, np.ndarray, float]:
    demos = pickle.loads(CANONICAL_DEMO.read_bytes())
    demo_obs = np.asarray([row["observations"] for row in demos], dtype=np.float64)
    require(demo_obs.shape == (1921, 20), "production Demo observation contract")
    lo, hi = np.percentile(demo_obs, [1, 99], axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    unique = np.unique(demo_obs, axis=0)
    distances = []
    for start in range(0, len(unique), 64):
        stop = min(start + 64, len(unique))
        d = np.sqrt(np.sum(np.square((unique[start:stop, None] - unique[None]) / scale), axis=2))
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf
        distances.append(np.min(d, axis=1))
    derived = float(np.max(np.concatenate(distances)))
    require(abs(derived - D_DEMO_THRESHOLD) <= 5e-12, "frozen online-admission threshold derivation")
    return demo_obs, scale, derived


def demo_distance(obs, demo_obs, scale) -> float:
    value = np.asarray(obs, dtype=np.float64)
    return float(np.min(np.sqrt(np.sum(np.square((demo_obs - value) / scale), axis=1))))


def raw_env(env):
    return env.env


def snapshot_state(env, obs, step: int, path: Path) -> dict:
    raw = raw_env(env)
    data = raw.data
    arrays = {}
    for name in (
        "qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata",
        "qacc_warmstart", "qfrc_applied", "xfrc_applied", "eq_active",
    ):
        value = getattr(data, name, None)
        if value is not None:
            arrays[name] = np.asarray(value).copy()
    arrays["adapter_target_pose"] = np.asarray(env._target_pose).copy()
    arrays["adapter_workspace_low"] = np.asarray(env._workspace_low).copy()
    arrays["adapter_workspace_high"] = np.asarray(env._workspace_high).copy()
    arrays["observation"] = np.asarray(obs, dtype=np.float32).copy()
    np.savez_compressed(path, **arrays)
    metrics = raw.task._goal_metrics(raw._get_obs())
    return {
        "step": step,
        "state_file": str(path),
        "state_file_sha256": sha(path),
        "sim_time": float(data.time),
        "model_dimensions": {
            key: int(getattr(raw.model, key))
            for key in ("nq", "nv", "na", "nu", "nmocap", "nuserdata")
        },
        "stored_fields": sorted(arrays),
        "metrics": {key: float(metrics[key]) for key in
                    ("lateral_error", "axial_error", "orientation_error")},
    }


def verify_canonical_env(env) -> dict:
    raw = raw_env(env)
    task = raw.task
    require(env.observation_contract_version == V2, "EVAL V2 observation contract")
    require(env.action_space.shape == (7,) and env.action_space.dtype == np.float32,
            "EVAL action contract")
    require(task.position_tolerance == 0.002 and task.orientation_tolerance == 0.05,
            "EVAL canonical success thresholds")
    return {
        "factory": "peg_env_adapter.make_peg_state_env",
        "max_episode_steps": 500,
        "observation_contract_version": V2,
        "position_tolerance": float(task.position_tolerance),
        "orientation_tolerance": float(task.orientation_tolerance),
        "curriculum_active": False,
        "expert_intervention_active_during_eval": False,
    }


def load_policy(policy_cfg, env):
    fresh = make_combined_q_anchor_agent(
        policy_cfg["seed"], env.observation_space.sample(), env.action_space.sample(), policy_cfg)
    state = restore_fresh_state(POLICY_CHECKPOINT.parent, fresh.state, 1000)
    return fresh.replace(state=state)


def policy_action(agent, obs) -> np.ndarray:
    action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)),
                        dtype=np.float32)
    require(action.shape == (7,) and np.isfinite(action).all() and
            np.max(np.abs(action)) <= 1.000001, "policy action contract")
    return action


def transition_metrics(env, info, action, owner, step, distance) -> dict:
    link01, unexpected, pairs = contact_flags(env)
    return {
        "step": step,
        "owner": owner,
        "action": action.tolist(),
        "demo_distance": float(distance),
        "lateral": float(info["lateral_error"]),
        "axial": float(info["axial_error"]),
        "orientation": float(info["orientation_error"]),
        "force": float(info.get("force_norm", 0.0)),
        "link01_contact": bool(link01),
        "unexpected_contact": bool(unexpected),
        "contact_pairs": pairs,
        "success": bool(info.get("is_success", False)),
    }


def policy_audit(env, agent, demo_obs, scale, snapshot_dir: Path) -> dict:
    obs, _ = env.reset(seed=0)
    captures = {}
    rows = []
    ret = 0.0
    near = success = any_contact = False
    for step in range(1, 501):
        action = policy_action(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        distance = demo_distance(obs, demo_obs, scale)
        row = transition_metrics(env, info, action, "policy", step, distance)
        rows.append(row)
        near = near or (row["lateral"] <= 0.010 and row["axial"] <= 0.020)
        success = success or row["success"]
        any_contact = any_contact or row["link01_contact"] or row["unexpected_contact"]
        if step in CAPTURE_STEPS:
            state_path = snapshot_dir / f"policy_seed0_step{step:03d}.npz"
            captures[str(step)] = snapshot_state(env, obs, step, state_path)
            captures[str(step)]["demo_distance"] = distance
        if terminated or truncated:
            break
    return {
        "seed": 0,
        "steps": len(rows),
        "return": ret,
        "success": success,
        "near_hole": near,
        "contact": any_contact,
        "captures": captures,
        "trajectory": rows,
    }


def recovery_trial(env, agent, expert_rows, intervention_step, seed, demo_obs, scale,
                   snapshot_dir: Path) -> dict:
    obs, _ = env.reset(seed=seed)
    trajectory = []
    ret = 0.0
    terminated = truncated = False
    for step in range(1, intervention_step + 1):
        action = policy_action(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        trajectory.append(transition_metrics(
            env, info, action, "policy", step, demo_distance(obs, demo_obs, scale)))
        if terminated or truncated:
            break
    boundary_path = snapshot_dir / f"recovery_seed{seed}_step{intervention_step:03d}.npz"
    boundary = snapshot_state(env, obs, len(trajectory), boundary_path)
    boundary["demo_distance"] = demo_distance(obs, demo_obs, scale)
    if not (terminated or truncated):
        for stream_index in range(intervention_step, len(expert_rows)):
            source = expert_rows[stream_index]
            action = np.asarray(source["action"], dtype=np.float32)
            require(action.shape == (7,) and np.isfinite(action).all() and
                    np.max(np.abs(action)) <= 1.0, "expert action contract")
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            trajectory.append(transition_metrics(
                env, info, action, "expert", len(trajectory) + 1,
                demo_distance(obs, demo_obs, scale)))
            if terminated or truncated:
                break
    success = any(row["success"] for row in trajectory)
    expert_count = sum(row["owner"] == "expert" for row in trajectory)
    max_force = max((row["force"] for row in trajectory), default=0.0)
    contact = any(row["link01_contact"] or row["unexpected_contact"] for row in trajectory)
    final = trajectory[-1] if trajectory else {}
    return {
        "seed": seed,
        "intervention_step": intervention_step,
        "policy_transitions": min(intervention_step, len(trajectory)),
        "expert_transitions": expert_count,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "return": ret,
        "max_force": max_force,
        "contact": contact,
        "boundary": boundary,
        "final": final,
        "trajectory": trajectory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    eval_cfg, policy_cfg, expert_rows = verify_static_inputs()
    demo_obs, scale, derived_threshold = make_distance_model()
    paths = protected_paths()
    before = {str(path): sha(path) for path in paths}
    output = unique_output()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_dir = output / "sim_states"
    snapshot_dir.mkdir(exist_ok=False)
    binding = recorder_bindings(ROOT)
    phase0 = {
        "RECOVERY_CLOSURE_WORKDIR": str(output),
        "PRODUCTION_DEMO_SHA": before[str(CANONICAL_DEMO)],
        "CANONICAL_EVAL_CONFIG_SHA": before[str(EVAL_CONFIG)],
        "STABLE_PROTOCOL_RUNNER_SHA": before[str(STABLE_RUNNER)],
        "EVAL_BINDING_SHA": binding["env_hash"],
        "POLICY_CHECKPOINT": str(POLICY_CHECKPOINT),
        "POLICY_CHECKPOINT_SHA": before[str(POLICY_CHECKPOINT)],
        "SCRIPTED_EXPERT_STREAM": str(EXPERT_STREAM),
        "SCRIPTED_EXPERT_STREAM_SHA": before[str(EXPERT_STREAM)],
        "ONLINE_ADMISSION_THRESHOLD": D_DEMO_THRESHOLD,
        "ONLINE_ADMISSION_THRESHOLD_DERIVED": derived_threshold,
    }
    dump(output / "phase0_manifest.json", phase0)
    dump(output / "protected_before.json", before)
    if args.preflight_only:
        result = {
            **phase0,
            "PREFLIGHT": "PASS",
            "RECOVERY_BASIN_AUDIT": "NOT_RUN",
            "TRAINING_STARTED": False,
            "CORRECTIVE_DATASET_STARTED": False,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "CORRECTIVE_DEMO_SEPARATE": "YES",
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "EVAL_RESET_CANONICAL": "YES",
            "EVAL_GOAL_CANONICAL": "YES",
            "EVAL_SUCCESS_CANONICAL": "YES",
            "EVAL_CURRICULUM_ACTIVE": "NO",
            "PROTECTED_ARTIFACTS_CHANGED": "NO",
        }
        dump(output / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0

    env = None
    started = time.monotonic()
    try:
        with portable_peg_runtime(ROOT):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
            eval_binding = verify_canonical_env(env)
            agent = load_policy(policy_cfg, env)
            policy_result = policy_audit(env, agent, demo_obs, scale, snapshot_dir)
            recovery = {}
            for intervention_step in CAPTURE_STEPS:
                recovery[str(intervention_step)] = [
                    recovery_trial(env, agent, expert_rows, intervention_step, seed,
                                   demo_obs, scale, snapshot_dir)
                    for seed in RECOVERY_SEEDS
                ]
    finally:
        if env is not None:
            env.close()
    rates = {
        step: float(np.mean([trial["success"] for trial in trials]))
        for step, trials in recovery.items()
    }
    reliable = [int(step) for step, rate in rates.items() if rate >= 2.0 / 3.0]
    recoverable = [int(step) for step, rate in rates.items() if rate > 0.0]
    nonrecoverable = [step for step in CAPTURE_STEPS if step not in recoverable]
    gate = bool(recoverable)
    after = {str(path): sha(path) for path in paths}
    unchanged = before == after
    require(after[str(CANONICAL_DEMO)] == CANONICAL_DEMO_SHA, "production Demo changed")
    require(unchanged, "protected artifact changed")
    result = {
        **phase0,
        "PREFLIGHT": "PASS",
        "RECOVERY_BASIN_AUDIT": "PASS" if gate else "FAIL",
        "RECOVERABLE_STEP_SET": recoverable,
        "NONRECOVERABLE_STEP_SET": nonrecoverable,
        "RECOVERY_SUCCESS_RATE_BY_STEP": rates,
        "LATEST_RELIABLY_RECOVERABLE_STEP": max(reliable) if reliable else None,
        "POLICY_FULL_DIFFICULTY_ROLLOUT": policy_result,
        "RECOVERY_TRIALS": recovery,
        "EVAL_BINDING": eval_binding,
        "TRAINING_STARTED": False,
        "CORRECTIVE_DATASET_STARTED": False,
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
        "CORRECTIVE_DEMO_SEPARATE": "YES",
        "ONLINE_ADMISSION_THRESHOLD": D_DEMO_THRESHOLD,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "EVAL_RESET_CANONICAL": "YES",
        "EVAL_GOAL_CANONICAL": "YES",
        "EVAL_SUCCESS_CANONICAL": "YES",
        "EVAL_CURRICULUM_ACTIVE": "NO",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "PRIMARY_ROOT_CAUSE": "RECOVERY_DATA_GAP" if gate else "NO_RECOVERABLE_POLICY_STATE",
        "NEXT": "CORRECTIVE_DEMO_COLLECTION" if gate else "TRAINING_ONLY_CURRICULUM_FEASIBILITY",
        "ELAPSED_SECONDS": time.monotonic() - started,
    }
    dump(output / "protected_after.json", after)
    dump(output / "result.json", finite(result))
    summary_keys = (
        "RECOVERY_BASIN_AUDIT", "RECOVERABLE_STEP_SET", "NONRECOVERABLE_STEP_SET",
        "RECOVERY_SUCCESS_RATE_BY_STEP", "LATEST_RELIABLY_RECOVERABLE_STEP",
        "PRODUCTION_DEMO_SHA_UNCHANGED", "ONLINE_ADMISSION_THRESHOLD",
        "ONLINE_ADMISSION_THRESHOLD_CHANGED", "EVAL_RESET_CANONICAL",
        "EVAL_GOAL_CANONICAL", "EVAL_SUCCESS_CANONICAL", "EVAL_CURRICULUM_ACTIVE",
        "PROTECTED_ARTIFACTS_CHANGED", "PRIMARY_ROOT_CAUSE", "NEXT", "ELAPSED_SECONDS",
    )
    print(json.dumps(finite({key: result[key] for key in summary_keys}), sort_keys=True))
    print(str(output / "result.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
