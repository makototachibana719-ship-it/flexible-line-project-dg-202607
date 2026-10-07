#!/usr/bin/env python3
"""Read-only causal audit of the failed C0 near-hole policy.

This script never updates an agent or changes task definitions.  Every causal
branch restores mjSTATE_INTEGRATION plus the proved Python/controller state.
"""
from __future__ import annotations

import argparse
import copy
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
    str(ROOT), str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from peg_env_adapter import make_peg_state_env
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    action as policy_action, distance_model, load_config, make_train_env, raw_env,
)
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD = 7.41773975814
LIMITS = np.asarray([0.002, 0.003, 0.05], np.float64)
FORCE_SPIKE_N = 50.0
ACTION_DIVERGENCE_LINF = 0.25
SNAPSHOT_DIR = ROOT / "results/closure/peg/f3_v2_exact_state_restore_20260907_001"
RUN = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002"
TRACE = ROOT / "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
OUT_ROOT = ROOT / "results/diagnostics/peg"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def require(value, message):
    if not value:
        raise RuntimeError(message)


def jvalue(value):
    if isinstance(value, np.ndarray): return jvalue(value.tolist())
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, dict): return {str(k): jvalue(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jvalue(v) for v in value]
    return value


def dump(path: Path, value):
    path.write_text(json.dumps(jvalue(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_path(path: Path) -> str:
    if path.is_file(): return sha_file(path)
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(bytes.fromhex(sha_file(item)))
    return h.hexdigest()


def protected_hashes(cfg):
    paths = {
        "production_demo": ROOT / cfg["demo"]["path"],
        "online_threshold_config": ROOT / "configs/peg_insertion/peg_v2_exact_restore_curriculum_rlpd_20260907.yaml",
        "exact_restore_module": ROOT / "training/peg_insertion/exact_state_curriculum.py",
        "exact_restore_closure": ROOT / "scripts/close_peg_f3_v2_exact_state_restore.py",
        "c0_snapshot": SNAPSHOT_DIR / "c0_exact_state.npz",
        "c0_manifest": SNAPSHOT_DIR / "c0_exact_state_manifest.json",
        "expert_trace": TRACE,
        "checkpoint_1000": RUN / "checkpoints/checkpoint_1000",
        "training_result": RUN / "result.json",
    }
    return {name: {"path": str(path), "sha256": sha_path(path)} for name, path in paths.items()}


def unique_output(requested: str | None) -> Path:
    if requested:
        out = Path(requested)
        require(not out.exists(), "fresh output required")
        return out
    stem = "f3_v2_c0_near_hole_insertion_causal_20260908"
    for i in range(1, 1000):
        out = OUT_ROOT / f"{stem}_{i:03d}"
        if not out.exists(): return out
    raise RuntimeError("no fresh output directory")


def expert_action(row):
    value = np.asarray(row["action"], np.float32)
    require(value.shape == (7,) and np.isfinite(value).all(), "expert action")
    return value


def python_state(env):
    wrapper = env.env if hasattr(env, "level") else env
    raw = wrapper.env
    task = raw.task
    return {
        "adapter_target_pose": np.asarray(wrapper._target_pose).copy(),
        "adapter_workspace_low": np.asarray(wrapper._workspace_low).copy(),
        "adapter_workspace_high": np.asarray(wrapper._workspace_high).copy(),
        "raw_elapsed_steps": int(raw._elapsed_steps),
        "raw_episode_count": int(raw._episode_count),
        "raw_desired_goal": np.asarray(raw._desired_goal).copy(),
        "task_current_step": int(task.current_step),
        "task_is_success": bool(task.is_success),
        "task_insertion_axis": np.asarray(task._insertion_axis).copy(),
        "task_last_metrics": copy.deepcopy(task.last_metrics),
        "controller_q_initial": np.asarray(raw.controller.q_initial).copy(),
    }


def capture(env, observation):
    raw = raw_env(env)
    size = mujoco.mj_stateSize(raw.model, STATE_SPEC)
    vector = np.empty(size, np.float64)
    mujoco.mj_getState(raw.model, raw.data, vector, STATE_SPEC)
    return {"mujoco": vector, "python": python_state(env),
            "observation": np.asarray(observation, np.float32).copy()}


def restore(env, snap):
    env.reset(seed=100)
    wrapper = env.env if hasattr(env, "level") else env
    raw = wrapper.env
    s = snap["python"]
    mujoco.mj_setState(raw.model, raw.data, snap["mujoco"], STATE_SPEC)
    wrapper._target_pose = s["adapter_target_pose"].copy()
    wrapper._workspace_low = s["adapter_workspace_low"].copy()
    wrapper._workspace_high = s["adapter_workspace_high"].copy()
    raw._elapsed_steps = s["raw_elapsed_steps"]
    raw._episode_count = s["raw_episode_count"]
    raw._desired_goal = s["raw_desired_goal"].copy()
    raw.task.current_step = s["task_current_step"]
    raw.task.is_success = s["task_is_success"]
    raw.task._insertion_axis = s["task_insertion_axis"].copy()
    raw.task.last_metrics = copy.deepcopy(s["task_last_metrics"])
    raw.controller.q_initial = s["controller_q_initial"].copy()
    mujoco.mj_forward(raw.model, raw.data)
    obs = wrapper._state(raw._get_obs())
    require(np.array_equal(obs, snap["observation"]),
            f"branch restore obs mismatch {np.max(np.abs(obs-snap['observation']))}")
    return obs


def geom_record(raw, gid):
    model = raw.model; gid = int(gid); bid = int(model.geom_bodyid[gid]); did = int(model.geom_dataid[gid])
    gname = model.geom(gid).name or f"unnamed_geom_{gid}"
    bname = model.body(bid).name or f"unnamed_body_{bid}"
    mesh = model.mesh(did).name if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH) and did >= 0 else None
    return {"id": gid, "geom": gname, "body_id": bid, "body": bname, "mesh": mesh}


def is_descendant(model, child, ancestor):
    node = int(child)
    while node > 0:
        if node == ancestor: return True
        node = int(model.body_parentid[node])
    return node == ancestor


def table_ee_contact(raw):
    model, data = raw.model, raw.data
    table_bodies = {i for i in range(model.nbody) if "table" in (model.body(i).name or "").lower()}
    anchors = {i for i in range(model.nbody) if any(t in (model.body(i).name or "").lower()
               for t in ("ft_sensor_mount", "gripper", "assemble_axle", "end_effector"))}
    ee_bodies = {i for i in range(model.nbody) if any(is_descendant(model, i, a) for a in anchors)}
    records = []
    for i in range(int(data.ncon)):
        c = data.contact[i]; sides = [geom_record(raw, c.geom1), geom_record(raw, c.geom2)]
        bids = {x["body_id"] for x in sides}
        if bids & table_bodies and bids & ee_bodies:
            records.append({"contact_index": i, "sides": sides, "body_pair": [x["body"] for x in sides],
                            "position": np.asarray(c.pos).copy(), "normal": np.asarray(c.frame[:3]).copy(),
                            "distance": float(c.dist), "geom_ids": [int(c.geom1), int(c.geom2)]})
    return records


def all_contacts(raw):
    rows = []
    for i in range(int(raw.data.ncon)):
        c = raw.data.contact[i]; sides = [geom_record(raw, c.geom1), geom_record(raw, c.geom2)]
        rows.append({"body_pair": [x["body"] for x in sides], "geom_pair": [x["geom"] for x in sides],
                     "distance": float(c.dist)})
    return rows


def metric_values(info):
    vals = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
    return vals, float(np.max(vals / LIMITS))


def transition_record(env, step, obs, act, info, terminated, truncated):
    raw = raw_env(env); od = raw._get_obs(); vals, margin = metric_values(info); te = table_ee_contact(raw)
    return {
        "step": step, "lateral_error": vals[0], "axial_error": vals[1],
        "orientation_error": vals[2], "success_margin": margin,
        "near_hole": bool(vals[0] <= .010 and vals[1] <= .020),
        "success": bool(info.get("is_success", terminated and not truncated)),
        "tcp_position": np.asarray(od["tcp_pos"]).copy(), "tcp_orientation": np.asarray(od["tcp_quat"]).copy(),
        "force": float(info.get("force_norm", np.linalg.norm(np.asarray(od["wrench"])[:3]))),
        "contact_pair": [x["body_pair"] for x in te], "all_contacts": all_contacts(raw),
        "table_ee_contact": bool(te), "table_ee_records": te,
        "policy_action": np.asarray(act).copy(), "observation": np.asarray(obs).copy(),
        "terminated": bool(terminated), "truncated": bool(truncated),
    }


def nearest_demo(obs, demo_obs, demo_actions, scale):
    distances = np.sqrt(np.sum(np.square((demo_obs - np.asarray(obs)) / scale), axis=1))
    index = int(np.argmin(distances))
    return demo_actions[index].copy(), index, float(distances[index])


def q_heads(agent, obs, act):
    values = np.asarray(jax.device_get(agent.forward_critic(
        jnp.asarray(obs[None]), jnp.asarray(act[None]), rng=None, train=False)))
    return values[:, 0]


def branch_one(snap, act):
    env = make_train_env("C0")
    try:
        restore(env, snap)
        obs, reward, term, trunc, info = env.step(np.asarray(act, np.float32))
        row = transition_record(env, 1, obs, act, info, term, trunc)
        return row
    finally: env.close()


def geom_distance(raw, gids):
    fromto = np.empty(6, np.float64)
    value = float(mujoco.mj_geomDistance(raw.model, raw.data, int(gids[0]), int(gids[1]), 10.0, fromto))
    return {"distance": value, "fromto": fromto.copy()}


def branch_rollout(snap, first_action, agent, horizon, contact_gids):
    env = make_train_env("C0")
    rows = []
    try:
        obs = restore(env, snap)
        for i in range(1, horizon + 1):
            act = np.asarray(first_action, np.float32) if i == 1 else policy_action(agent, obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(act)
            row = transition_record(env, i, obs, act, info, term, trunc)
            row["table_ee_geometry"] = geom_distance(raw_env(env), contact_gids)
            rows.append(row)
            if term or trunc: break
        return {"first": rows[0], "min_separation": min(x["table_ee_geometry"]["distance"] for x in rows),
                "contact": any(x["table_ee_contact"] for x in rows), "rows": rows}
    finally: env.close()


def compact_metrics(row):
    return {k: row[k] for k in ("tcp_position", "tcp_orientation", "lateral_error", "axial_error",
            "orientation_error", "success_margin", "force", "table_ee_contact", "contact_pair")}


def saturation_summary(rows, center, demo_obs, demo_actions, scale):
    selected = [r for r in rows if max(1, center - 10) <= r["step"] <= center + 10]
    actions = np.asarray([r["policy_action"] for r in selected])
    refs = np.asarray([nearest_demo(r["observation"], demo_obs, demo_actions, scale)[0] for r in selected])
    return {"range": [selected[0]["step"], selected[-1]["step"]], "per_dim_max_abs": np.max(np.abs(actions), axis=0),
            "per_dim_saturation_fraction": np.mean(np.abs(actions) >= .999, axis=0),
            "mean_policy_action": np.mean(actions, axis=0), "mean_demo_nearest_action": np.mean(refs, axis=0)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output"); args = parser.parse_args()
    out = unique_output(args.output); out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(1)
    require(float(cfg["online_admission"]["threshold"]) == THRESHOLD, "threshold")
    before_hashes = protected_hashes(cfg); dump(out / "protected_before.json", before_hashes)
    require(before_hashes["production_demo"]["sha256"] == DEMO_SHA, "production demo sha")
    rows = [json.loads(line) for line in TRACE.read_text().splitlines() if line.strip()]
    manifest = json.loads((SNAPSHOT_DIR / "c0_exact_state_manifest.json").read_text())
    start = int(manifest["next_expert_index"]); suffix = rows[start:]
    demos = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    demo_obs, scale = distance_model(demos); demo_actions = np.asarray([x["actions"] for x in demos], np.float32)
    require(jax.default_backend() == "gpu", "GPU inference required")

    with portable_peg_runtime(ROOT):
        sample_env = make_train_env("C0")
        fresh = make_combined_q_anchor_agent(cfg["seed"], sample_env.observation_space.sample(),
                                             sample_env.action_space.sample(), cfg)
        sample_env.close()
        agent = fresh.replace(state=restore_fresh_state(RUN / "checkpoints", fresh.state, 1000))

        # Frozen expert suffix from the same exact snapshot.
        expert_env = make_train_env("C0"); expert_rows = []; expert_contact = False
        try:
            expert_obs, _ = expert_env.reset(seed=100)
            for local, source_row in enumerate(suffix, 1):
                act = expert_action(source_row)
                expert_obs, _, term, trunc, info = expert_env.step(act)
                rec = transition_record(expert_env, local, expert_obs, act, info, term, trunc)
                expert_rows.append(rec); expert_contact |= rec["table_ee_contact"]
                if term or trunc: break
        finally: expert_env.close()
        require(expert_rows and expert_rows[-1]["success"], "frozen expert suffix must succeed")

        episode_summaries = []; policy_reference = None; state_snaps = None
        for seed in (100, 101, 102):
            env = make_train_env("C0"); sequence = []; snaps = []
            try:
                obs, _ = env.reset(seed=seed)
                for step in range(1, 501):
                    snaps.append(capture(env, obs))
                    act, _ = policy_action(agent, obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(act)
                    rec = transition_record(env, step, obs, act, info, term, trunc); sequence.append(rec)
                    if term or trunc: break
            finally: env.close()
            best = min(sequence, key=lambda x: x["success_margin"])
            near = next((x["step"] for x in sequence if x["near_hole"]), None)
            table = next((x["step"] for x in sequence if x["table_ee_contact"]), None)
            spike = next((x["step"] for x in sequence if x["force"] >= FORCE_SPIKE_N), None)
            divergences = []
            for i in range(min(len(sequence), len(expert_rows))):
                delta = sequence[i]["policy_action"] - expert_rows[i]["policy_action"]
                if np.max(np.abs(delta)) > ACTION_DIVERGENCE_LINF: divergences.append(i + 1)
            divergence = divergences[0] if divergences else None
            episode_summaries.append({"seed": seed, "steps": len(sequence), "success": sequence[-1]["success"],
                "first_near_hole_step": near, "best_success_margin_step": best["step"],
                "best_success_margin": best["success_margin"], "best_lateral": best["lateral_error"],
                "best_axial": best["axial_error"], "best_orientation": best["orientation_error"],
                "first_table_ee_contact_step": table, "first_force_spike_step": spike,
                "first_major_policy_expert_divergence_step": divergence})
            with (out / f"episode_seed{seed}_timeline.jsonl").open("w") as stream:
                for rec in sequence:
                    compact = {k: rec[k] for k in ("step","lateral_error","axial_error","orientation_error",
                        "success_margin","near_hole","success","tcp_position","tcp_orientation","force",
                        "contact_pair","policy_action","terminated","truncated")}
                    stream.write(json.dumps(jvalue(compact), sort_keys=True, allow_nan=False) + "\n")
            if seed == 100: policy_reference, state_snaps = sequence, snaps

        require(all(x == episode_summaries[0] | {"seed": x["seed"]} for x in episode_summaries),
                "deterministic C0 episodes differ")
        best = min(policy_reference, key=lambda x: x["success_margin"])
        mismatch = best["success_margin"] <= 1.0 and not best["success"]
        divergence = episode_summaries[0]["first_major_policy_expert_divergence_step"]
        require(divergence is not None, "no meaningful action divergence")
        p0 = policy_reference[divergence - 1]["policy_action"]
        e0 = expert_rows[divergence - 1]["policy_action"]

        result = {
            "C0_NEAR_HOLE_INSERTION_AUDIT": "ENGINEERING_STOP" if mismatch else "PASS",
            "definitions": {"success_limits": LIMITS, "force_spike_newtons": FORCE_SPIKE_N,
                            "major_action_divergence_linf": ACTION_DIVERGENCE_LINF},
            "episodes": episode_summaries,
            "BEST_SIMULTANEOUS_SUCCESS_MARGIN": best["success_margin"], "BEST_SIMULTANEOUS_STEP": best["step"],
            "BEST_LATERAL": best["lateral_error"], "BEST_AXIAL": best["axial_error"],
            "BEST_ORIENTATION": best["orientation_error"],
            "SUCCESS_PREDICATE_PIPELINE_MISMATCH": "YES" if mismatch else "NO",
            "FIRST_NEAR_HOLE_STEP": episode_summaries[0]["first_near_hole_step"],
            "FIRST_POLICY_EXPERT_ACTION_DIVERGENCE_STEP": divergence,
            "FIRST_TABLE_EE_CONTACT_STEP": episode_summaries[0]["first_table_ee_contact_step"],
            "FIRST_FORCE_SPIKE_STEP": episode_summaries[0]["first_force_spike_step"],
            "POLICY_ACTION_AT_FIRST_DIVERGENCE": p0,
            "EXPERT_ACTION_AT_FIRST_DIVERGENCE": e0, "ACTION_DELTA": p0 - e0,
            "TRAINING_STARTED": False,
        }
        if mismatch:
            result.update(PRIMARY_ROOT_CAUSE="SUCCESS_PREDICATE_PIPELINE_MISMATCH",
                          NEXT_FIX="SUCCESS_PIPELINE_ENGINEERING_REPAIR")
        else:
            # Exact one-step branch at first divergence.
            bp = branch_one(state_snaps[divergence - 1], p0); be = branch_one(state_snaps[divergence - 1], e0)
            damage = bp["success_margin"] > be["success_margin"] + 1e-6 or bp["force"] > be["force"] + 1.0
            result["POLICY_NEXT_METRICS"] = compact_metrics(bp); result["EXPERT_NEXT_METRICS"] = compact_metrics(be)
            result["FIRST_DIVERGENCE_IMMEDIATE_DAMAGE"] = "YES" if damage else "NO"

            hybrids = []
            if damage:
                for dim in range(6):
                    act = p0.copy(); act[dim] = e0[dim]; br = branch_one(state_snaps[divergence - 1], act)
                    hybrids.append({"dimension": f"a{dim}", "action": act,
                        "success_margin_improvement": bp["success_margin"] - br["success_margin"],
                        "lateral_improvement": bp["lateral_error"] - br["lateral_error"],
                        "axial_improvement": bp["axial_error"] - br["axial_error"],
                        "orientation_improvement": bp["orientation_error"] - br["orientation_error"],
                        "force_improvement": bp["force"] - br["force"], "next_metrics": compact_metrics(br)})
            ranked = sorted(hybrids, key=lambda x: x["success_margin_improvement"], reverse=True)
            positive = [x for x in ranked if x["success_margin_improvement"] > 1e-6]
            dominant = positive[0]["dimension"] if positive and (len(positive) == 1 or
                positive[0]["success_margin_improvement"] >= 2 * positive[1]["success_margin_improvement"]) else "NONE"
            result["dimension_branches"] = hybrids
            result["DOMINANT_ACTION_DIMENSION"] = dominant
            result["SECONDARY_ACTION_DIMENSIONS"] = [x["dimension"] for x in positive if x["dimension"] != dominant]
            result["ACTION_FAILURE_STRUCTURE"] = ("SPECIFIC_ACTION_DIMENSION" if dominant != "NONE" else
                                                    "COUPLED_MULTI_AXIS" if damage else "NO_IMMEDIATE_DAMAGE")

            contact_step = episode_summaries[0]["first_table_ee_contact_step"]
            require(contact_step is not None, "expected table-ee contact absent")
            contact_row = policy_reference[contact_step - 1]; collision = contact_row["table_ee_records"][0]
            pre_step = max(1, contact_step - 15); pre_snap = state_snaps[pre_step - 1]
            pre_obs = pre_snap["observation"]; pre_policy = policy_reference[pre_step - 1]["policy_action"]
            pre_expert, pre_demo_index, pre_demo_distance = nearest_demo(pre_obs, demo_obs, demo_actions, scale)
            trans = pre_expert.copy(); trans[:3] = pre_policy[:3]
            rot = pre_expert.copy(); rot[3:6] = pre_policy[3:6]
            horizon = contact_step - pre_step + 1
            branches = {"policy": branch_rollout(pre_snap, pre_policy, agent, horizon, collision["geom_ids"]),
                        "expert_demo_nearest": branch_rollout(pre_snap, pre_expert, agent, horizon, collision["geom_ids"]),
                        "translation_only": branch_rollout(pre_snap, trans, agent, horizon, collision["geom_ids"]),
                        "rotation_only": branch_rollout(pre_snap, rot, agent, horizon, collision["geom_ids"])}
            initial_sep = geom_distance(raw_env(make_train_env("C0")), collision["geom_ids"]) if False else None
            e_sep = branches["expert_demo_nearest"]["min_separation"]
            t_damage = e_sep - branches["translation_only"]["min_separation"]
            r_damage = e_sep - branches["rotation_only"]["min_separation"]
            if branches["policy"]["first"]["table_ee_geometry"]["distance"] <= 0: contact_class = "ALREADY_BAD_STATE"
            elif t_damage > max(1e-6, 2 * r_damage): contact_class = "TRANSLATION_COMMAND"
            elif r_damage > max(1e-6, 2 * t_damage): contact_class = "ROTATION_COMMAND"
            elif t_damage > 1e-6 and r_damage > 1e-6: contact_class = "COUPLED_POSE_COMMAND"
            else: contact_class = "UNKNOWN"
            result["pre_contact"] = {"state_step": pre_step, "horizon": horizon,
                "policy_action": pre_policy, "expert_reference_action": pre_expert,
                "expert_reference": {"method": "production_demo_nearest_normalized_V2_observation",
                                     "index": pre_demo_index, "distance": pre_demo_distance},
                "branches": {k: {"first": compact_metrics(v["first"]), "min_separation": v["min_separation"],
                                  "contact": v["contact"]} for k,v in branches.items()}}
            result["TABLE_CONTACT_CAUSAL_CLASS"] = contact_class
            result["TABLE_GEOM"] = collision["sides"][0] if "table" in collision["sides"][0]["body"].lower() else collision["sides"][1]
            result["END_EFFECTOR_GEOM"] = collision["sides"][1] if "table" in collision["sides"][0]["body"].lower() else collision["sides"][0]
            result["BODY_PAIR"] = collision["body_pair"]
            result["CONTACT_POSITION"] = collision["position"]; result["CONTACT_NORMAL"] = collision["normal"]
            result["CONTACT_DISTANCE"] = collision["distance"]
            result["EXPERT_TABLE_EE_CONTACT"] = "YES" if expert_contact else "NO"
            result["GEOMETRY_ROOT_CAUSE"] = "NO" if not expert_contact else "UNRESOLVED"

            near_step = episode_summaries[0]["first_near_hole_step"]
            sat = {"first_divergence": saturation_summary(policy_reference, divergence, demo_obs, demo_actions, scale),
                   "first_near_hole": saturation_summary(policy_reference, near_step, demo_obs, demo_actions, scale),
                   "first_contact": saturation_summary(policy_reference, contact_step, demo_obs, demo_actions, scale)}
            saturated = any(np.any(np.asarray(x["per_dim_saturation_fraction"]) >= .5) for x in sat.values())
            result["action_saturation_windows"] = sat; result["LOCAL_ACTION_SATURATION"] = "YES" if saturated else "NO"

            critical = {"first_divergence": divergence, "best_success_margin": best["step"], "pre_contact": pre_step}
            q_result = {}
            bad_votes = []
            for name, step in critical.items():
                state_obs = state_snaps[step - 1]["observation"]
                pa = policy_reference[step - 1]["policy_action"]
                if step <= len(expert_rows): ea = expert_rows[step - 1]["policy_action"]; emethod = "time_aligned_expert_suffix"
                else: ea, _, _ = nearest_demo(state_obs, demo_obs, demo_actions, scale); emethod = "demo_nearest"
                da, di, dd = nearest_demo(state_obs, demo_obs, demo_actions, scale)
                qp, qe, qd = q_heads(agent,state_obs,pa), q_heads(agent,state_obs,ea), q_heads(agent,state_obs,da)
                de, ddemo = qp-qe, qp-qd
                q_result[name] = {"step":step,"expert_method":emethod,"demo_index":di,"demo_distance":dd,
                                  "policy":qp,"expert":qe,"demo":qd,"policy_minus_expert":de,"policy_minus_demo":ddemo}
                bad_votes.extend(list(de > 0)); bad_votes.extend(list(ddemo > 0))
            result["critic_local_ranking"] = q_result
            result["Q_POLICY_MINUS_EXPERT"] = {k:v["policy_minus_expert"] for k,v in q_result.items()}
            result["Q_POLICY_MINUS_DEMO"] = {k:v["policy_minus_demo"] for k,v in q_result.items()}
            count = int(sum(bad_votes)); result["CRITIC_PREFERS_POLICY_BAD_ACTION"] = ("YES" if count == len(bad_votes)
                else "NO" if count == 0 else "MIXED")

            # Event ordering and strongest supported root cause.
            events = [(divergence,"divergence"),(near_step,"near_hole"),
                      (episode_summaries[0]["first_force_spike_step"],"force_spike"),(contact_step,"table_ee_contact")]
            result["FAILURE_EVENT_ORDER"] = " -> ".join(f"{name}@{step}" for step,name in sorted(events))
            if result["CRITIC_PREFERS_POLICY_BAD_ACTION"] == "YES":
                root, fix = "LOCAL_CRITIC_MISRANKING_NEAR_INSERTION", "NEAR_INSERTION_VALUE_CALIBRATION_REVIEW"
            elif dominant != "NONE":
                root, fix = "SPECIFIC_ACTION_DIMENSION_FAILURE", "TARGETED_LOCAL_ACTION_CAUSAL_REVIEW"
            elif contact_class in ("TRANSLATION_COMMAND","ROTATION_COMMAND","COUPLED_POSE_COMMAND"):
                root, fix = "TABLE_CONTACT_IS_DOWNSTREAM_CONSEQUENCE", "PRE_CONTACT_POLICY_DIVERGENCE_REPAIR"
            elif branches["expert_demo_nearest"]["contact"] is False and branches["policy"]["contact"] is True:
                root, fix = "TABLE_CONTACT_IS_PRIMARY_LOCAL_BLOCKER", "CONTACT_AVOIDANCE_POLICY_REVIEW"
            elif result["CRITIC_PREFERS_POLICY_BAD_ACTION"] == "NO":
                root, fix = "LOCAL_ACTOR_MULTI_AXIS_CONTROL_FAILURE", "LOCAL_INSERTION_POLICY_SUPERVISION_REVIEW"
            else:
                root, fix = "MIXED_LOCAL_INSERTION_FAILURE", "LOCAL_INSERTION_COMBINED_CAUSAL_REVIEW"
            result["PRIMARY_ROOT_CAUSE"] = root; result["NEXT_FIX"] = fix

        # Formal canonical definitions are independently verified and never used for C0 branching.
        c1 = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        c2 = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        try:
            o1,_=c1.reset(seed=100); o2,_=c2.reset(seed=100)
            canonical_reset = bool(np.array_equal(o1,o2)); canonical_goal = bool(np.array_equal(raw_env(c1)._desired_goal,raw_env(c2)._desired_goal))
        finally: c1.close(); c2.close()
        result.update(PRODUCTION_DEMO_SHA_UNCHANGED="YES", ONLINE_ADMISSION_THRESHOLD=THRESHOLD,
                      ONLINE_ADMISSION_THRESHOLD_CHANGED="NO", EVAL_RESET_CANONICAL="YES" if canonical_reset else "NO",
                      EVAL_GOAL_CANONICAL="YES" if canonical_goal else "NO", EVAL_CURRICULUM_ACTIVE="NO")

    after_hashes = protected_hashes(cfg); dump(out / "protected_after.json", after_hashes)
    unchanged = before_hashes == after_hashes
    result["PROTECTED_ARTIFACTS_CHANGED"] = "NO" if unchanged else "YES"
    require(unchanged, "protected artifact changed during read-only audit")
    dump(out / "expert_suffix_timeline.json", [{k:v for k,v in r.items() if k not in ("observation","table_ee_records","all_contacts")}
                                                for r in expert_rows])
    dump(out / "result.json", result)
    (out / "REPORT.md").write_text("# Peg F3 V2 C0 near-hole to insertion causal audit\n\n```json\n" +
        json.dumps(jvalue(result), indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(jvalue(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
