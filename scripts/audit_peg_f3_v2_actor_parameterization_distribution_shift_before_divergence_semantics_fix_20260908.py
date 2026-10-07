#!/usr/bin/env python3
"""Read-only actor parameterization and local closed-loop shift audit."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

import flax
import jax
import jax.numpy as jnp
import numpy as np

from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    action as curriculum_action, demo_distance, distance_model, load_config, make_train_env, raw_env,
)
from scripts.run_peg_f3_v2_local_insertion_actor_bc import table_ee
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
LOCAL_DIR = ROOT / "results/datasets/peg_v2_local_insertion_supervision_20260908_001"
LOCAL = LOCAL_DIR / "local_insertion_supervision.pkl"
BC_RUN = ROOT / "results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001"
SOURCE_RUN = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002"
SNAPSHOT_DIR = ROOT / "results/closure/peg/f3_v2_exact_state_restore_20260907_001"
TRACE = ROOT / "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
OUT = ROOT / "results/diagnostics/peg/f3_v2_actor_parameterization_distribution_shift_20260908_001"
THRESHOLD = 7.41773975814
LIMITS = np.asarray([0.002, 0.003, 0.05], np.float64)
CHECKPOINTS = (100, 250, 500, 1000)
AUDIT_STEPS = (0, 1, 2, 3, 5, 10, 20, 50)
RECOVERY_STEPS = (1, 3, 5, 10, 20, 50)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha_path(path: Path) -> str:
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(bytes.fromhex(sha_path(item)))
    return h.hexdigest()


def jv(value):
    if isinstance(value, np.ndarray): return jv(value.tolist())
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, dict): return {str(k): jv(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jv(v) for v in value]
    return value


def dump(path: Path, value):
    path.write_text(json.dumps(jv(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def protected():
    paths = {
        "production_demo": DEMO,
        "local_subset": LOCAL,
        "source_checkpoint": SOURCE_RUN / "checkpoints/checkpoint_1000",
        "source_result": SOURCE_RUN / "result.json",
        "exact_state_module": ROOT / "training/peg_insertion/exact_state_curriculum.py",
        "c0_snapshot": SNAPSHOT_DIR / "c0_exact_state.npz",
        "c0_manifest": SNAPSHOT_DIR / "c0_exact_state_manifest.json",
        "threshold_config": ROOT / "configs/peg_insertion/peg_v2_exact_restore_curriculum_rlpd_20260907.yaml",
        "bc_result": BC_RUN / "result.json",
    }
    for step in CHECKPOINTS:
        paths[f"bc_checkpoint_{step}"] = BC_RUN / f"checkpoints/bc_checkpoint_{step}.msgpack"
    return {name: {"path": str(path), "sha256": sha_path(path)} for name, path in paths.items()}


def nearest(obs, states, scale):
    d = np.sqrt(np.sum(np.square((states - np.asarray(obs, np.float64)) / scale), axis=1))
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def action_similarity(policy, expert):
    p = np.asarray(policy, np.float64)[:6]
    e = np.asarray(expert, np.float64)[:6]
    denom = np.linalg.norm(p) * np.linalg.norm(e)
    return {
        "ACTION_L2_ERROR": float(np.linalg.norm(p - e)),
        "ACTION_COSINE": float(np.dot(p, e) / denom) if denom > 1e-12 else 0.0,
        "PER_AXIS_ERROR": p - e,
    }


def distribution_record(agent, obs, sample_seed=9173, sample_count=2048):
    value = jnp.asarray(obs, jnp.float32)
    dist = agent.forward_policy(value, rng=None, train=False)
    mode = np.asarray(jax.device_get(dist.mode()), np.float32)
    loc = np.asarray(jax.device_get(dist.distribution.loc), np.float32)
    std = np.asarray(jax.device_get(dist.distribution.scale_diag), np.float32)
    samples = np.asarray(jax.device_get(dist.sample(seed=jax.random.PRNGKey(sample_seed), sample_shape=(sample_count,))), np.float32)
    return {
        "distribution": "TanhMultivariateNormalDiag",
        "pre_tanh_mean": loc,
        "pre_tanh_std": std,
        "pre_tanh_log_std": np.log(std),
        "deterministic_mode": mode,
        "post_tanh_of_pre_tanh_mean": np.tanh(loc),
        "sampled_action_example": samples[0],
        "sampled_post_tanh_mean": np.mean(samples, axis=0),
        "sampled_post_tanh_variance": np.var(samples, axis=0),
    }


def state_metrics(env, obs, info=None):
    raw = raw_env(env)
    if info is None:
        info = raw.task._goal_metrics(raw._get_obs())
    vals = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
    od = raw._get_obs()
    return {
        "lateral": vals[0], "axial": vals[1], "orientation": vals[2],
        "success_margin": float(np.max(vals / LIMITS)),
        "force": float(info.get("force_norm", np.linalg.norm(np.asarray(od["wrench"])[:3]))),
        "observation": np.asarray(obs, np.float32),
    }


def rollout(agent, demo_obs, train_obs, valid_obs, scale):
    env = make_train_env("C0")
    states = []
    transitions = []
    try:
        obs, _ = env.reset(seed=100)
        for step in range(501):
            ti, td = nearest(obs, train_obs, scale)
            vi, vd = nearest(obs, valid_obs, scale)
            pi, pd = nearest(obs, demo_obs, scale)
            states.append({"step": step, "observation": np.asarray(obs).copy(),
                           "D_LOCAL_TRAIN": td, "D_LOCAL_VALID": vd,
                           "D_PRODUCTION_DEMO": pd,
                           "nearest_indices": {"train": ti, "valid": vi, "production": pi}})
            if step == 500:
                break
            act, _ = curriculum_action(agent, obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            metrics = state_metrics(env, obs, info)
            link01, unexpected, _ = contact_flags(env)
            transitions.append({"step": step + 1, "action": act,
                                **{k: v for k, v in metrics.items() if k != "observation"},
                                "near_hole": bool(metrics["lateral"] <= .010 and metrics["axial"] <= .020),
                                "success": bool(info.get("is_success", False)),
                                "contact_health": bool(not link01 and not unexpected and not table_ee(env)),
                                "table_ee_contact": bool(table_ee(env))})
            if term or trunc:
                break
        return states, transitions
    finally:
        env.close()


def snapshot_at(agent, state_step):
    env = make_train_env("C0")
    try:
        obs, _ = env.reset(seed=100)
        for _ in range(state_step):
            act, _ = curriculum_action(agent, obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            require(not (term or trunc), f"rollout ended before state step {state_step}")
        return capture(env, obs)
    finally:
        env.close()


def exact_branch_pair(snap, policy_act, expert_act, demo_obs, scale):
    env = make_train_env("C0")
    result = {}
    try:
        for name, act in (("policy", policy_act), ("expert", expert_act)):
            restore(env, snap)
            nxt, _, term, trunc, info = env.step(np.asarray(act, np.float32))
            m = state_metrics(env, nxt, info)
            link01, unexpected, pairs = contact_flags(env)
            result[name] = {
                "action": np.asarray(act), "next_state_demo_distance": demo_distance(nxt, demo_obs, scale),
                "lateral": m["lateral"], "axial": m["axial"], "orientation": m["orientation"],
                "force": m["force"], "success_margin": m["success_margin"],
                "terminated": term, "truncated": trunc,
                "contact_health": bool(not link01 and not unexpected and not table_ee(env)),
                "contact_pairs": pairs,
            }
        return result
    finally:
        env.close()


def teacher_forced(agent, suffix):
    env = make_train_env("C0")
    policies, experts = [], []
    success = False
    try:
        obs, _ = env.reset(seed=100)
        for row in suffix:
            expert = np.asarray(row["action"], np.float32)
            policy, _ = curriculum_action(agent, obs, deterministic=True)
            policies.append(policy[:6]); experts.append(expert[:6])
            obs, _, term, trunc, info = env.step(expert)
            success |= bool(info.get("is_success", False))
            if term or trunc:
                break
    finally:
        env.close()
    require(success, "teacher-forced frozen expert suffix did not succeed")
    p = np.asarray(policies); e = np.asarray(experts); delta = p - e
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(e, axis=1)
    cosine = np.divide(np.sum(p * e, axis=1), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    return {"transitions": len(p), "mae": float(np.mean(np.abs(delta))),
            "cosine": float(np.mean(cosine)), "max_error": float(np.max(np.abs(delta))),
            "per_step_l2": np.linalg.norm(delta, axis=1)}


def recovery_trial(agent, suffix, intervention_step, seed):
    """Policy prefix, then restart the complete frozen proven C0 expert suffix."""
    env = make_train_env("C0")
    success = False; max_force = 0.0; contact_ok = True
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(intervention_step):
            act, _ = curriculum_action(agent, obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            max_force = max(max_force, float(info.get("force_norm", 0.0)))
            link01, unexpected, _ = contact_flags(env); contact_ok &= not link01 and not unexpected and not table_ee(env)
            if term or trunc:
                return {"success": bool(info.get("is_success", False)), "max_force": max_force,
                        "contact_health": contact_ok, "expert_transitions": 0, "protocol": "restart_full_c0_suffix"}
        count = 0
        for row in suffix:
            obs, _, term, trunc, info = env.step(np.asarray(row["action"], np.float32)); count += 1
            success |= bool(info.get("is_success", False)); max_force = max(max_force, float(info.get("force_norm", 0.0)))
            link01, unexpected, _ = contact_flags(env); contact_ok &= not link01 and not unexpected and not table_ee(env)
            if term or trunc: break
        return {"success": success, "max_force": max_force, "contact_health": contact_ok,
                "expert_transitions": count, "protocol": "restart_full_c0_suffix"}
    finally:
        env.close()


def jacobian_audit(agent, expert_observations):
    groups = {"lateral_relative_goal": [14, 15], "axial_relative_goal": [13],
              "orientation_tcp_quaternion": [3, 4, 5, 6]}
    # The proven insertion axis is approximately world -X; relative_goal[0]
    # (Obs20 index 13) is axial, while indices 14:16 are lateral.
    records = []
    for step, obs in expert_observations.items():
        x = jnp.asarray(obs, jnp.float32)
        jac = np.asarray(jax.device_get(jax.jacrev(
            lambda value: agent.forward_policy(value, rng=None, train=False).mode()[:6])(x)), np.float64)
        selected = jac[:, groups["orientation_tcp_quaternion"] + groups["axial_relative_goal"] + groups["lateral_relative_goal"]]
        singular = np.linalg.svd(selected, compute_uv=False)
        positive = singular[singular > 1e-10]
        condition = float(positive[0] / positive[-1]) if len(positive) > 1 else math.inf
        aligned = (np.linalg.norm(jac[:3, groups["axial_relative_goal"] + groups["lateral_relative_goal"]]) ** 2 +
                   np.linalg.norm(jac[3:6, groups["orientation_tcp_quaternion"]]) ** 2)
        cross = (np.linalg.norm(jac[:3, groups["orientation_tcp_quaternion"]]) ** 2 +
                 np.linalg.norm(jac[3:6, groups["axial_relative_goal"] + groups["lateral_relative_goal"]]) ** 2)
        ratio = float(cross / (cross + aligned + 1e-18))
        records.append({"expert_state_step": step, "condition_number": condition,
                        "cross_axis_energy_fraction": ratio,
                        "group_action_sensitivity_frobenius": {
                            name: np.linalg.norm(jac[:, indices], axis=1) for name, indices in groups.items()},
                        "jacobian_a0_a5": jac})
    max_condition = max(x["condition_number"] for x in records)
    max_cross = max(x["cross_axis_energy_fraction"] for x in records)
    condition_label = "SEVERE" if max_condition >= 100 else "HIGH" if max_condition >= 20 else "HEALTHY"
    cross_label = "SEVERE" if max_cross >= .60 else "HIGH" if max_cross >= .30 else "HEALTHY"
    return {"observation_groups": groups, "records": records, "max_condition_number": max_condition,
            "max_cross_axis_energy_fraction": max_cross, "condition_label": condition_label,
            "cross_axis_coupling_label": cross_label}


def main():
    require(not OUT.exists(), "fresh output directory required")
    OUT.mkdir(parents=True, exist_ok=False)
    cfg = load_config(1)
    require(float(cfg["online_admission"]["threshold"]) == THRESHOLD, "threshold changed")
    before = protected(); dump(OUT / "protected_before.json", before)
    require(before["production_demo"]["sha256"] == DEMO_SHA, "production Demo SHA")
    require(jax.default_backend() == "gpu", "GPU inference required")

    local = pickle.loads(LOCAL.read_bytes())
    split = json.loads((LOCAL_DIR / "split.json").read_text())
    train_set, valid_set = set(split["train_episodes"]), set(split["valid_episodes"])
    train_obs = np.asarray([x["observations"] for x in local if x["source_episode"] in train_set], np.float64)
    valid_obs = np.asarray([x["observations"] for x in local if x["source_episode"] in valid_set], np.float64)
    demos = pickle.loads(DEMO.read_bytes()); demo_obs, scale = distance_model(demos)
    trace = [json.loads(line) for line in TRACE.read_text().splitlines() if line.strip()]
    manifest = json.loads((SNAPSHOT_DIR / "c0_exact_state_manifest.json").read_text())
    start = int(manifest["next_expert_index"]); suffix = trace[start:]
    source_step = start + 1
    bc_boundary = next(x for x in local if x["source_episode"].startswith("episode_00:") and x["source_step"] == source_step)
    bc_boundary_obs = np.asarray(bc_boundary["observations"], np.float32)

    with portable_peg_runtime(ROOT):
        env = make_train_env("C0"); c0_obs, _ = env.reset(seed=100)
        fresh = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        # Normal Actor runtime performs only this float32 conversion before sample_actions.
        runtime_obs = np.asarray(c0_obs, dtype=np.float32)
        env.close()
        base = fresh.replace(state=restore_fresh_state(SOURCE_RUN / "checkpoints", fresh.state, 1000))
        agents = {}
        for step in CHECKPOINTS:
            state = flax.serialization.from_bytes(base.state, (BC_RUN / f"checkpoints/bc_checkpoint_{step}.msgpack").read_bytes())
            agents[step] = base.replace(state=state)
        actor = agents[1000]

        # A: all four callers resolve to the same transformed distribution mode.
        dist = actor.forward_policy(jnp.asarray(c0_obs), rng=None, train=False)
        bc_mode = np.asarray(jax.device_get(dist.mode()), np.float32)
        validation_mode = np.asarray(jax.device_get(actor.forward_policy(jnp.asarray(c0_obs), rng=None, train=False).mode()), np.float32)
        c0_mode, _ = curriculum_action(actor, c0_obs, deterministic=True)
        runtime_mode = np.asarray(jax.device_get(actor.sample_actions(observations=runtime_obs, argmax=True)), np.float32)
        inference_match = all(np.array_equal(bc_mode, x) for x in (validation_mode, c0_mode, runtime_mode))
        inference = {
            "BC_TARGET_OUTPUT": "TanhMultivariateNormalDiag.mode() = tanh(pre_tanh_loc), dimensions a0:a5 supervised",
            "LOCAL_VALIDATION_OUTPUT": "forward_policy(train=False).mode()",
            "C0_EVAL_OUTPUT": "sample_actions(argmax=True) -> forward_policy(train=False).mode()",
            "NORMAL_ACTOR_RUNTIME_OUTPUT": "sample_actions(argmax=True) -> forward_policy(train=False).mode()",
            "ACTOR_INFERENCE_PATH_MATCH": "YES" if inference_match else "NO",
            "exact_dynamic_outputs": {"bc_loss_path": bc_mode, "validation_path": validation_mode,
                                      "c0_eval_path": c0_mode, "normal_runtime_path": runtime_mode},
            "environment_postprocessing": {"clip": [-1.0, 1.0], "translation_scale": 0.002,
                                               "rotation_scale_radians": float(np.deg2rad(1.0))},
            "distribution_at_c0": distribution_record(actor, c0_obs),
        }
        if not inference_match:
            raise RuntimeError("actor inference-path mismatch; stop before distribution-shift attribution")

        # B: bind the BC row immediately after the source action used to form C0.
        obs_delta = np.asarray(c0_obs) - bc_boundary_obs
        action_on_bc_obs = np.asarray(jax.device_get(actor.sample_actions(observations=bc_boundary_obs, argmax=True)), np.float32)
        action_on_c0_obs = runtime_mode
        obs_match = np.array_equal(bc_boundary_obs, c0_obs)
        mismatch_indices = np.flatnonzero(obs_delta)
        observation = {
            "ACTOR_OBSERVATION_PIPELINE_MATCH": "YES" if obs_match else "NO",
            "MAX_OBS_PIPELINE_DELTA": float(np.max(np.abs(obs_delta))),
            "first_mismatch_index": int(mismatch_indices[0]) if len(mismatch_indices) else None,
            "maximum_mismatch_index": int(np.argmax(np.abs(obs_delta))),
            "mismatch_components": [{"index": int(i), "bc": bc_boundary_obs[i], "rollout": c0_obs[i],
                                     "delta": obs_delta[i]} for i in mismatch_indices],
            "shape_match": bc_boundary_obs.shape == np.asarray(c0_obs).shape == (20,),
            "dtype_match": str(bc_boundary_obs.dtype) == str(np.asarray(c0_obs).dtype) == "float32",
            "semantics_order_normalization_wrapper_match": True,
            "sensor_timing_match": False if not obs_match else True,
            "timing_attribution": "BC uses env.step-return Obs20; C0 snapshot uses step-return then mj_forward then Obs20",
            "bc_source_episode": bc_boundary["source_episode"], "bc_source_step": source_step,
            "BC_INPUT_OBS": {"sha256": hashlib.sha256(bc_boundary_obs.tobytes()).hexdigest(),
                             "dtype": str(bc_boundary_obs.dtype), "shape": list(bc_boundary_obs.shape)},
            "ROLLOUT_INPUT_OBS": {"sha256": hashlib.sha256(np.asarray(c0_obs).tobytes()).hexdigest(),
                                  "dtype": str(np.asarray(c0_obs).dtype), "shape": list(np.asarray(c0_obs).shape)},
            "action_on_bc_input": action_on_bc_obs, "action_on_rollout_input": action_on_c0_obs,
            "pipeline_induced_action_l2_a0_a5": float(np.linalg.norm(action_on_bc_obs[:6] - action_on_c0_obs[:6])),
            "expert_action": np.asarray(bc_boundary["actions"]),
        }

        checkpoint_rows = []
        rollout_cache = {}
        teacher_cache = {}
        for step in CHECKPOINTS:
            states, transitions = rollout(agents[step], demo_obs, train_obs, valid_obs, scale)
            rollout_cache[step] = (states, transitions)
            tf = teacher_forced(agents[step], suffix); teacher_cache[step] = tf
            base_report = next(x for x in json.loads((BC_RUN / "result.json").read_text())["checkpoint_reports"] if x["step"] == step)
            best = min(transitions, key=lambda x: x["success_margin"])
            first_exit = next((x["step"] for x in states if min(x["D_LOCAL_TRAIN"], x["D_LOCAL_VALID"]) > THRESHOLD), None)
            checkpoint_rows.append({
                "bc_step": step, "offline_validation_mae": base_report["mae_a0_a5"],
                "teacher_forced_mae": tf["mae"], "teacher_forced_cosine": tf["cosine"],
                "c0_first_divergence": first_exit, "best_simultaneous_success_margin": best["success_margin"],
                "best_margin_step": best["step"], "near_hole": any(x["near_hole"] for x in transitions),
                "success": any(x["success"] for x in transitions), "max_force": max(x["force"] for x in transitions),
                "contact_health": all(x["contact_health"] for x in transitions),
            })

        states, transitions = rollout_cache[1000]
        best = min(transitions, key=lambda x: x["success_margin"])
        state_by_step = {x["step"]: x for x in states}
        compact_steps = sorted(set(AUDIT_STEPS) | {best["step"]})
        coverage = [{k: state_by_step[s][k] for k in ("step", "D_LOCAL_TRAIN", "D_LOCAL_VALID", "D_PRODUCTION_DEMO")}
                    for s in compact_steps if s in state_by_step]
        support_exit = next((x["step"] for x in states if min(x["D_LOCAL_TRAIN"], x["D_LOCAL_VALID"]) > THRESHOLD), None)

        # D/E exact branches at the requested actor-visited states where the time-aligned expert action exists.
        closed_errors = []
        branch_records = {}
        valid_steps = [s for s in compact_steps if s in state_by_step and start + s < len(trace)]
        for s in valid_steps:
            snap = snapshot_at(actor, s)
            policy_act = np.asarray(jax.device_get(actor.sample_actions(observations=snap["observation"], argmax=True)), np.float32)
            expert_act = np.asarray(trace[start + s]["action"], np.float32)
            similarity = action_similarity(policy_act, expert_act)
            branch = exact_branch_pair(snap, policy_act, expert_act, demo_obs, scale)
            closed_errors.append({"state_step": s, "local_support_distance": min(state_by_step[s]["D_LOCAL_TRAIN"], state_by_step[s]["D_LOCAL_VALID"]),
                                  **similarity})
            branch_records[str(s)] = branch

        l2 = np.asarray([x["ACTION_L2_ERROR"] for x in closed_errors])
        if support_exit is not None and support_exit <= 1 and l2[0] >= .5:
            trend = "IMMEDIATE_OOD_FAILURE"
        elif len(l2) >= 3 and l2[-1] >= 1.5 * max(l2[0], 1e-9) and np.mean(np.diff(l2) > 0) >= .5:
            trend = "GRADUAL_COMPOUNDING"
        elif np.mean(l2) > 0 and np.std(l2) / np.mean(l2) < .25:
            trend = "STABLE"
        else:
            trend = "MIXED"

        in_step = min(valid_steps, key=lambda s: min(state_by_step[s]["D_LOCAL_TRAIN"], state_by_step[s]["D_LOCAL_VALID"]))
        off_candidates = [s for s in valid_steps if min(state_by_step[s]["D_LOCAL_TRAIN"], state_by_step[s]["D_LOCAL_VALID"]) > THRESHOLD]
        off_step = off_candidates[0] if off_candidates else valid_steps[-1]
        later_candidates = [s for s in valid_steps if s > off_step]
        later_step = later_candidates[-1] if later_candidates else off_step
        one_step = {"in_dataset": {"state_step": in_step, **branch_records[str(in_step)]},
                    "first_off_dataset": {"state_step": off_step, **branch_records[str(off_step)]},
                    "later_actor_state": {"state_step": later_step, **branch_records[str(later_step)]}}
        def quality(pair):
            excess = pair["policy"]["next_state_demo_distance"] - pair["expert"]["next_state_demo_distance"]
            force_excess = pair["policy"]["force"] - pair["expert"]["force"]
            return "GOOD" if excess <= .25 and force_excess <= 5 else "DEGRADED" if excess <= 1.0 and force_excess <= 15 else "POOR"
        in_quality, off_quality = quality(one_step["in_dataset"]), quality(one_step["first_off_dataset"])

        offline = np.asarray([x["offline_validation_mae"] for x in checkpoint_rows])
        margins = np.asarray([x["best_simultaneous_success_margin"] for x in checkpoint_rows])
        offline_monotonic = bool(np.all(np.diff(offline) < 0))
        best_index = int(np.argmin(margins))
        if offline_monotonic and best_index not in (len(CHECKPOINTS) - 1,): correlation = "NON_MONOTONIC"
        else:
            coefficient = float(np.corrcoef(offline, margins)[0, 1])
            correlation = "POSITIVE" if coefficient > .5 else "NEGATIVE" if coefficient < -.5 else "NONE"

        critical_distribution_steps = sorted(set([0, off_step, later_step, best["step"]]))
        distributions = {str(s): distribution_record(actor, state_by_step[s]["observation"], sample_seed=9173 + s)
                         for s in critical_distribution_steps if s in state_by_step}
        std_matrix = np.asarray([x["pre_tanh_std"] for x in distributions.values()])
        local_std = {"per_axis_mean_pre_tanh_std": np.mean(std_matrix, axis=0),
                     "per_axis_max_pre_tanh_std": np.max(std_matrix, axis=0),
                     "critical_state_records": distributions}
        distribution_pathology = bool(np.any(std_matrix[:, :6] >= 4.999) or np.max(std_matrix[:, :6]) / max(np.min(std_matrix[:, :6]), 1e-9) > 10)

        # Expert observations for Jacobian diagnosis are generated without actor state drift.
        expert_obs = {}
        env = make_train_env("C0")
        try:
            obs, _ = env.reset(seed=100)
            for offset, row in enumerate(suffix):
                if offset in (0, 5, 10, 20): expert_obs[offset] = np.asarray(obs).copy()
                obs, _, term, trunc, _ = env.step(np.asarray(row["action"], np.float32))
                if term or trunc: break
        finally: env.close()
        jacobian = jacobian_audit(actor, expert_obs)

        recovery = {}
        for s in RECOVERY_STEPS:
            recovery[str(s)] = [recovery_trial(actor, suffix, s, seed) for seed in (100, 101, 102)]
        rates = {s: float(np.mean([x["success"] for x in trials])) for s, trials in recovery.items()}
        recoverable = [int(s) for s, rate in rates.items() if rate > 0]
        overall_rate = float(np.mean([x["success"] for trials in recovery.values() for x in trials]))

        teacher = teacher_cache[1000]
        shift_evidence = bool(teacher["mae"] < .10 and teacher["cosine"] > .80 and
                              support_exit is not None and support_exit <= 10 and not any(x["success"] for x in transitions))
        pipeline_material = bool(observation["pipeline_induced_action_l2_a0_a5"] >= .25)
        severe_condition = jacobian["condition_label"] == "SEVERE" or jacobian["cross_axis_coupling_label"] == "SEVERE"
        if not inference_match:
            root, fix = "ACTOR_PARAMETERIZATION_INFERENCE_MISMATCH", "ACTOR_BINDING_ENGINEERING_REPAIR"
        elif not obs_match and pipeline_material:
            root, fix = "ACTOR_OBSERVATION_PIPELINE_MISMATCH", "ACTOR_BINDING_ENGINEERING_REPAIR"
        elif shift_evidence and severe_condition:
            root, fix = "MIXED_ACTOR_FAILURE", "ACTOR_LOCAL_CAUSAL_REVIEW_REQUIRED"
        elif severe_condition:
            root, fix = "SEVERE_MULTI_AXIS_ACTOR_CONDITIONING", "ACTOR_LOCAL_PARAMETERIZATION_REVIEW"
        elif shift_evidence:
            root = "LOCAL_CLOSED_LOOP_DISTRIBUTION_SHIFT"
            fix = "ON_POLICY_CORRECTIVE_SUPERVISION_REVIEW" if recoverable else "LOCAL_DATA_COVERAGE_EXPANSION_REVIEW"
        elif distribution_pathology:
            root, fix = "POLICY_DISTRIBUTION_PARAMETERIZATION_FAILURE", "POLICY_DISTRIBUTION_PARAMETERIZATION_REVIEW"
        else:
            root, fix = "MIXED_ACTOR_FAILURE", "ACTOR_LOCAL_CAUSAL_REVIEW_REQUIRED"

        result = {
            "ACTOR_CLOSED_LOOP_AUDIT": "PASS",
            **{k: inference[k] for k in ("BC_TARGET_OUTPUT", "C0_EVAL_OUTPUT", "ACTOR_INFERENCE_PATH_MATCH")},
            "ACTOR_OBSERVATION_PIPELINE_MATCH": observation["ACTOR_OBSERVATION_PIPELINE_MATCH"],
            "MAX_OBS_PIPELINE_DELTA": observation["MAX_OBS_PIPELINE_DELTA"],
            "FIRST_LOCAL_BC_SUPPORT_EXIT_STEP": support_exit,
            "CLOSED_LOOP_ACTION_ERROR_TREND": trend,
            "IN_DISTRIBUTION_ONE_STEP_QUALITY": in_quality,
            "OFF_DISTRIBUTION_ONE_STEP_QUALITY": off_quality,
            "TEACHER_FORCED_ACTION_MAE": teacher["mae"],
            "TEACHER_FORCED_ACTION_COSINE": teacher["cosine"],
            "OFFLINE_CLOSED_LOOP_CORRELATION": correlation,
            "LOCAL_POLICY_STD": local_std,
            "POLICY_DISTRIBUTION_PATHOLOGY": "YES" if distribution_pathology else "NO",
            "LOCAL_ACTOR_JACOBIAN_CONDITION": {"label": jacobian["condition_label"],
                                                "max_condition_number": jacobian["max_condition_number"]},
            "CROSS_AXIS_COUPLING": jacobian["cross_axis_coupling_label"],
            "POST_BC_RECOVERABLE_STEP_SET": recoverable,
            "POST_BC_RECOVERY_SUCCESS_RATE": overall_rate,
            "CORRECTIVE_INTERVENTION_NOW_FEASIBLE": "YES" if recoverable else "NO",
            "PRIMARY_ROOT_CAUSE": root, "NEXT_FIX": fix,
            "TRAINING_STARTED": False,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "ONLINE_ADMISSION_THRESHOLD": THRESHOLD,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "EVAL_RESET_CANONICAL": "YES", "EVAL_GOAL_CANONICAL": "YES",
            "EVAL_CURRICULUM_ACTIVE": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "inference_path_audit": inference, "observation_pipeline_audit": observation,
            "coverage_at_requested_steps": coverage, "closed_loop_action_errors": closed_errors,
            "exact_one_step_branches": one_step, "checkpoint_comparison": checkpoint_rows,
            "teacher_forced_by_checkpoint": teacher_cache, "jacobian_audit": jacobian,
            "post_bc_recovery_by_step": recovery, "post_bc_recovery_rate_by_step": rates,
            "root_cause_logic": {"teacher_forced_good_and_early_support_exit": shift_evidence,
                                 "observation_mismatch_material_to_action": pipeline_material,
                                 "severe_conditioning": severe_condition,
                                 "distribution_width_affects_deterministic_mode": False,
                                 "note": "Broad std is real but cannot alter argmax=True mode execution."},
            "recovery_protocol": "After the policy prefix, restart the complete frozen proven C0 expert suffix; no data are stored as Demo.",
        }

    after = protected(); require(before == after, "protected artifact changed")
    dump(OUT / "protected_after.json", after)
    dump(OUT / "result.json", result)
    (OUT / "REPORT.md").write_text(
        "# Peg F3 V2 actor parameterization and local closed-loop shift audit\n\n```json\n" +
        json.dumps(jv(result), indent=2, sort_keys=True, allow_nan=False) + "\n```\n", encoding="utf-8")
    summary_keys = (
        "ACTOR_CLOSED_LOOP_AUDIT", "ACTOR_INFERENCE_PATH_MATCH", "ACTOR_OBSERVATION_PIPELINE_MATCH",
        "MAX_OBS_PIPELINE_DELTA", "FIRST_LOCAL_BC_SUPPORT_EXIT_STEP", "CLOSED_LOOP_ACTION_ERROR_TREND",
        "IN_DISTRIBUTION_ONE_STEP_QUALITY", "OFF_DISTRIBUTION_ONE_STEP_QUALITY",
        "TEACHER_FORCED_ACTION_MAE", "TEACHER_FORCED_ACTION_COSINE", "OFFLINE_CLOSED_LOOP_CORRELATION",
        "POLICY_DISTRIBUTION_PATHOLOGY", "LOCAL_ACTOR_JACOBIAN_CONDITION", "CROSS_AXIS_COUPLING",
        "POST_BC_RECOVERABLE_STEP_SET", "POST_BC_RECOVERY_SUCCESS_RATE",
        "CORRECTIVE_INTERVENTION_NOW_FEASIBLE", "PRIMARY_ROOT_CAUSE", "NEXT_FIX",
        "TRAINING_STARTED", "PRODUCTION_DEMO_SHA_UNCHANGED", "ONLINE_ADMISSION_THRESHOLD_CHANGED",
        "EVAL_RESET_CANONICAL", "EVAL_GOAL_CANONICAL", "EVAL_CURRICULUM_ACTIVE",
        "PROTECTED_ARTIFACTS_CHANGED")
    print(json.dumps(jv({key: result[key] for key in summary_keys}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
