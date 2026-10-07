#!/usr/bin/env python3
"""One authorized C2 on-policy corrective round and one combined head fit."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

import flax
import jax
import jax.numpy as jnp
import numpy as np

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.gated_absolute_c2_action_head_v1 import (
    ABSOLUTE_C2_HEAD_DEF,
    gated_absolute_mode,
)
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


PARAM_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_residual_output_parameterization_review_20260908_002"
EVIDENCE = review.EVIDENCE
DATASET_DIR = ROOT / "results/datasets/peg_v2_c2_on_policy_corrective_supervision_v1_20260908_001"
OUT = ROOT / "results/closure/peg/f3_v2_c2_on_policy_corrective_supervision_v1_20260908_001"
SEEDS = (100, 101, 102)
UPDATES = 1000


def req(value, message):
    if not value:
        raise RuntimeError(message)


def jvalue(value):
    if isinstance(value, np.ndarray):
        return jvalue(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jvalue(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jvalue(v) for v in value]
    return value


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(jvalue(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(demos):
    result = []
    episode = step = 0
    for row in demos:
        result.append((episode, step))
        step += 1
        if bool(row["dones"]):
            episode += 1
            step = 0
    return result


def load_head(path):
    params = ABSOLUTE_C2_HEAD_DEF.init(jax.random.PRNGKey(0), np.zeros((1, 20), np.float32))["params"]
    template = {"c2_head_params": params, "c2_head_opt_state": review.TX.init(params)}
    return flax.serialization.from_bytes(template, Path(path).read_bytes())


def collect(agent, c1_params, params, support, scale, core, outer, demos, demo_obs, demo_scale):
    actions = np.asarray([row["actions"] for row in demos], np.float32)
    source_provenance = provenance(demos)
    rows = []
    episode_summaries = []
    for seed in SEEDS:
        env = make_train_env("C2")
        accepted = 0
        stop_reason = "horizon"
        try:
            obs, _ = env.reset(seed=seed)
            for policy_step in range(500):
                link_before, bad_before, pairs_before = contact_flags(env)
                gate = float(np.asarray(jax.device_get(progressive_gate(obs, support, scale, core, outer))))
                distances = np.sqrt(np.sum(np.square((demo_obs - np.asarray(obs)) / demo_scale), axis=1))
                source_row = int(np.argmin(distances))
                demo_distance = float(distances[source_row])
                if link_before or bad_before:
                    stop_reason = "preexisting_contact"
                    break
                if demo_distance > p4.TH:
                    stop_reason = "pre_catastrophic_support_exit"
                    break
                if gate > review.NUMERICAL_EPS:
                    snapshot = capture(env, obs)
                    runtime_obs = np.asarray(obs, np.float32).copy()
                    expert_action = actions[source_row].copy()
                    expert_next, reward, terminated, truncated, info = env.step(expert_action)
                    expert_link, expert_bad, expert_pairs = contact_flags(env)
                    req(np.all(np.isfinite(expert_next)) and np.all(np.isfinite(expert_action)), "finite expert branch")
                    rows.append({
                        "observations": runtime_obs,
                        "actions": expert_action,
                        "next_observations": np.asarray(expert_next, np.float32).copy(),
                        "rewards": np.float32(reward),
                        "masks": np.float32(not terminated),
                        "dones": bool(terminated or truncated),
                        "actor_runtime_obs20": runtime_obs,
                        "expert_action": expert_action,
                        "episode_seed": seed,
                        "policy_step": policy_step,
                        "g2": gate,
                        "demo_distance": demo_distance,
                        "expert_source_row": source_row,
                        "expert_source_episode": source_provenance[source_row][0],
                        "expert_source_step": source_provenance[source_row][1],
                        "expert_source": "immutable_accepted_production_demo_nearest_normalized_Obs20",
                        "source_pre_contact": True,
                        "source_pre_catastrophic_divergence": True,
                        "source_contact_pairs": pairs_before,
                        "expert_branch_unexpected_contact": bool(expert_link or expert_bad),
                        "expert_branch_contact_pairs": expert_pairs,
                        "expert_branch_force": float(info.get("force_norm", 0.0)),
                    })
                    accepted += 1
                    obs = restore(env, snapshot)
                policy_action = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, params, obs, support, scale, core, outer)), np.float32)
                obs, _, terminated, truncated, _ = env.step(policy_action)
                link_after, bad_after, _ = contact_flags(env)
                if link_after or bad_after:
                    stop_reason = "policy_contact"
                    break
                if terminated or truncated:
                    stop_reason = "episode_end"
                    break
            episode_summaries.append({"seed": seed, "accepted_rows": accepted, "stop_reason": stop_reason})
        finally:
            env.close()
    req(rows and len(episode_summaries) == 3, "one three-episode corrective collection")
    req(all(row["g2"] > review.NUMERICAL_EPS and row["demo_distance"] <= p4.TH and row["source_pre_contact"] and row["source_pre_catastrophic_divergence"] for row in rows), "corrective source guards")
    return rows, episode_summaries


def main():
    req(Path.cwd() == ROOT and not OUT.exists() and not DATASET_DIR.exists(), "fresh outputs and project cwd")
    req(jax.default_backend() == "gpu", "GPU backend required")
    parameter_result = json.loads((PARAM_RUN / "result.json").read_text())
    req(parameter_result["PRIMARY_ROOT_CAUSE"] == "C2_CLOSED_LOOP_DISTRIBUTION_SHIFT" and parameter_result["NEXT"] == "C2_ON_POLICY_CORRECTIVE_SUPERVISION_V1", "Stage K authorization")
    req(parameter_result["C2_OFFLINE_FIT"] == "PASS" and parameter_result["C2_SHORT_HORIZON_POLICY_IMPROVEMENT"] == "GOOD", "Stage K evidence gates")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    DATASET_DIR.mkdir(parents=True)
    started = time.time()
    protected_paths = {"production_demo": p4.DEMO, "corrective_round1": p4.CORR, "local": p4.LOCAL, "parameter_checkpoint": Path(parameter_result["SELECTED_OFFLINE_CHECKPOINT"]["path"])}
    protected_before = {key: sha(path) for key, path in protected_paths.items()}
    dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
    subset = pickle.loads((EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl").read_bytes())
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    core = float(preflight["C1_CORE_THRESHOLD"])
    outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    cfg = load_config(2)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"])
        c1_params = source["c1_params"]
        base_guard = geo.thash(agent.state.params["modules_actor"])
        c1_guard = geo.thash(c1_params)
        req(base_guard == preflight["C0_BASE_SHA"] and c1_guard == preflight["C1_RESIDUAL_SHA"], "frozen stack")
        loaded_head = load_head(parameter_result["SELECTED_OFFLINE_CHECKPOINT"]["path"])
        params = loaded_head["c2_head_params"]

        corrective, episode_summaries = collect(agent, c1_params, params, support, scale, core, outer, demos, demo_obs, demo_scale)
        dataset_path = DATASET_DIR / "C2_ON_POLICY_CORRECTIVE_SUPERVISION_V1.pkl"
        dataset_path.write_bytes(pickle.dumps(corrective, pickle.HIGHEST_PROTOCOL))
        dataset_sha = sha(dataset_path)
        dump(DATASET_DIR / "manifest.json", {
            "name": "C2_ON_POLICY_CORRECTIVE_SUPERVISION_V1",
            "sha256": dataset_sha,
            "rows": len(corrective),
            "episodes": 3,
            "episode_summaries": episode_summaries,
            "source": "post-offline-BC policy runtime Obs20; frozen accepted production-Demo nearest recovery action",
            "pre_contact_only": True,
            "pre_catastrophic_divergence_only": True,
            "c2_authorized_only": True,
            "production_demo_sha256": p4.DSHA,
            "production_demo_modified": False,
            "collection_rounds": 1,
        })

        existing_obs = np.asarray([row["observations"] for row in subset], np.float32)
        existing_targets = np.asarray([row["actions"] for row in subset], np.float32)
        corrective_obs = np.asarray([row["observations"] for row in corrective], np.float32)
        corrective_targets = np.asarray([row["actions"] for row in corrective], np.float32)
        opt_state = review.TX.init(params)
        rng = np.random.default_rng(260970)
        training_reports = []
        for step in range(1, UPDATES + 1):
            existing_idx = rng.integers(0, len(existing_obs), 32)
            corrective_idx = rng.integers(0, len(corrective_obs), 32)
            batch_obs = np.concatenate((existing_obs[existing_idx], corrective_obs[corrective_idx]))
            batch_targets = np.concatenate((existing_targets[existing_idx], corrective_targets[corrective_idx]))
            params, opt_state, loss, gradient_norm = review.bc_step(params, opt_state, jnp.asarray(batch_obs), jnp.asarray(batch_targets), agent, c1_params, support, scale, core, outer)
            if step in (250, 500, 1000):
                training_reports.append({
                    "step": step,
                    "loss": float(loss),
                    "gradient_norm": float(gradient_norm),
                    "existing": review.action_metrics(params, existing_obs, existing_targets, agent, c1_params, support, scale, core, outer),
                    "corrective": review.action_metrics(params, corrective_obs, corrective_targets, agent, c1_params, support, scale, core, outer),
                })
        checkpoint = OUT / "checkpoints/gated_absolute_c2_head_corrective_1000.msgpack"
        checkpoint.write_bytes(flax.serialization.to_bytes({"c2_head_params": params, "c2_head_opt_state": opt_state}))
        checkpoint_sha = sha(checkpoint)
        dump(OUT / "combined_training.json", {"updates": UPDATES, "fixed_source_batching": "32 existing expert + 32 corrective per update", "reports": training_reports, "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha}})

        before_eval = json.loads((PARAM_RUN / "closed_loop_acquisition.json").read_text())["offline_C2"]
        c2_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C2", demo_obs, demo_scale)
        progressed, progress_checks = review.pilot.progress(before_eval, c2_eval)
        c1_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C1", demo_obs, demo_scale)
        c0_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C0", demo_obs, demo_scale)
        acquisition_pass = bool(progressed and c2_eval["contact_health"] and c1_eval["contact_health"] and c0_eval["success_count"] == 3 and c0_eval["contact_health"])
        matched = "NOT_RUN"
        confirmed = "NO"
        canonical_status = "NOT_RUN"
        canonical = None
        if acquisition_pass:
            matched_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C2", demo_obs, demo_scale)
            keys = ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
            matched_ok = all(matched_eval[key] == c2_eval[key] for key in keys)
            matched = "PASS" if matched_ok else "FAIL"
            confirmed = "YES" if matched_ok else "NO"
            if matched_ok:
                canonical = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C3", demo_obs, demo_scale)
                canonical_status = "COMPLETE"
        interface = c2_eval["interface"]
        if not acquisition_pass:
            first_bad = "C2_CORRECTIVE_ACQUISITION"
            root = "C2_CLOSED_LOOP_DISTRIBUTION_SHIFT_NOT_REPAIRED"
            next_task = "STOP"
        elif confirmed != "YES":
            first_bad = "C2_MATCHED_CONFIRMATION"
            root = "C2_ACQUISITION_NOT_REPRODUCIBLE"
            next_task = "STOP"
        else:
            canonical_outside_only = canonical["interface"]["class_state_counts"]["C2_OUTSIDE"] == canonical["interface"]["visited_states"]
            first_bad = "CANONICAL_C3_INTERFACE" if canonical_outside_only else None
            root = "UPSTREAM_TO_FROZEN_SKILL_INTERFACE_FAILURE" if canonical_outside_only else "NONE"
            next_task = "C3_ACQUISITION_PILOT" if canonical_outside_only else "C3_INTERFACE_RESULT_REVIEW"
        req(geo.thash(agent.state.params["modules_actor"]) == base_guard and geo.thash(c1_params) == c1_guard, "protected model mutation")
        result = {
            "MECHANISM": "GATED_ABSOLUTE_C2_ACTION_HEAD_V1",
            "C2_ON_POLICY_CORRECTIVE_USED": "YES",
            "C2_ON_POLICY_CORRECTIVE_ROUNDS": 1,
            "C2_ON_POLICY_CORRECTIVE_EPISODES": 3,
            "C2_ON_POLICY_CORRECTIVE_ROWS": len(corrective),
            "C2_ON_POLICY_CORRECTIVE_SHA": dataset_sha,
            "C2_CORRECTIVE_COMBINED_TRAINING_RUNS": 1,
            "C2_BC_ACQUISITION": "PASS" if acquisition_pass else "FAIL",
            "C2_PROGRESS_CHECKS": progress_checks,
            "C2_SHELL_ENTRY": interface["C1_SHELL_ENTRY_COUNT"],
            "C2_CORE_ENTRY": interface["C1_CORE_ENTRY_COUNT"],
            "C2_NEAR_HOLE": c2_eval["near_hole_rate"],
            "C2_SUCCESS": f"{c2_eval['success_count']}/3",
            "C2_CONTACT": "PASS" if c2_eval["contact_health"] else "FAIL",
            "C1_REGRESSION": "PASS" if c1_eval["contact_health"] else "FAIL",
            "C1_CONTACT": "PASS" if c1_eval["contact_health"] else "FAIL",
            "C0_SUCCESS": f"{c0_eval['success_count']}/3",
            "C2_MATCHED_CONFIRMATION": matched,
            "C2_SKILL_CONFIRMED": confirmed,
            "C2_FROZEN_SHA": review.tree_hash(params) if confirmed == "YES" else None,
            "CANONICAL_INTERFACE_AUDIT": canonical_status,
            "PEG_SINGLE_TASK_CONVERGED": "NO",
            "FIRST_BAD_GATE": first_bad,
            "PRIMARY_ROOT_CAUSE": root,
            "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "ONLINE_ADMISSION_THRESHOLD": p4.TH,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE",
            "elapsed_seconds": time.time() - started,
        }
        dump(OUT / "evaluation.json", {"before_corrective_C2": before_eval, "after_corrective_C2": c2_eval, "C1": c1_eval, "C0": c0_eval, "canonical": canonical})
        protected_after = {key: sha(path) for key, path in protected_paths.items()}
        req(protected_after == protected_before, "protected artifact changed")
        dump(OUT / "protected_after.json", protected_after)
        dump(OUT / "result.json", result)
        print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
