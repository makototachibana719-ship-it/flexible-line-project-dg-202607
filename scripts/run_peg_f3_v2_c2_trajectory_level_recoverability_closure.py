#!/usr/bin/env python3
"""Frozen-2K C2 full-trajectory expert-recoverability audit; no training."""
from __future__ import annotations

import hashlib
import json
import math
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
import numpy as np

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.audit_peg_f3_v2_c2_closed_loop_recovery_basin as basin
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate, support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_trajectory_level_recoverability_closure_20260909_001"
CHECKPOINT = tail.CONVERGENCE_RUN / "checkpoints/constant_plateau_c2_2000.msgpack"
BOUNDARY_SET = tail.DATASET_PATH
AUDIT_DATA = pc.AUDIT_DATA
DEMO_SHA = tail.DEMO_SHA
SEEDS = (100, 101, 102)
HORIZONS = (1, 4, 8)


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


def obs_hash(obs):
    return hashlib.sha256(np.asarray(obs, np.float32).tobytes()).hexdigest()


def collect_trajectory(agent, c1_params, definition, params, support, scale, core, outer, demo_obs, demo_scale):
    episodes = []
    for seed in SEEDS:
        env = make_train_env("C2")
        rows = []
        try:
            obs, _ = env.reset(seed=seed)
            for step in range(500):
                link, unexpected, pairs = contact_flags(env)
                action = tail.checkpoint_action(definition, params, obs, agent, c1_params, support, scale, core, outer)
                rows.append({
                    "seed": seed,
                    "policy_step": step,
                    "observation": np.asarray(obs, np.float32).copy(),
                    "snapshot": capture(env, obs),
                    "policy_action": np.asarray(action, np.float32),
                    "obs20_sha": obs_hash(obs),
                    "D_DEMO": float(p4.demo_distance(obs, demo_obs, demo_scale)),
                    "D_C1": float(np.asarray(jax.device_get(support_distance(obs, support, scale)))),
                    "g2": float(np.asarray(jax.device_get(progressive_gate(obs, support, scale, core, outer)))),
                    "contact_before": bool(link or unexpected),
                    "contact_pairs_before": pairs,
                })
                if link or unexpected:
                    break
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        episodes.append(rows)
    return episodes


def sequence_then_takeover(oracle, snapshot, actions):
    obs = restore(oracle.env, snapshot)
    healthy = True
    executed = 0
    terminal_success = False
    transition_rows = []
    for action in actions:
        obs, _, terminated, truncated, info = oracle.env.step(np.asarray(action, np.float32))
        executed += 1
        link, unexpected, pairs = contact_flags(oracle.env)
        force = float(info.get("force_norm", 0.0))
        healthy &= bool(not link and not unexpected and np.isfinite(force))
        transition_rows.append({
            "force": force,
            "contact_healthy": bool(not link and not unexpected),
            "contact_pairs": pairs,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated or not healthy:
            terminal_success = bool(terminated and not truncated and info.get("is_success", False))
            break
    if terminal_success:
        takeover = {
            "EXPERT_SHELL_RECOVERABLE": True,
            "EXPERT_CORE_RECOVERABLE": True,
            "EXPERT_SUCCESS_RECOVERABLE": True,
            "EXPERT_CONTACT_HEALTH": True,
            "EXPERT_RECOVERY_STEPS": 0,
        }
    elif healthy:
        takeover = oracle._takeover_current(obs)
    else:
        takeover = {
            "EXPERT_SHELL_RECOVERABLE": False,
            "EXPERT_CORE_RECOVERABLE": False,
            "EXPERT_SUCCESS_RECOVERABLE": False,
            "EXPERT_CONTACT_HEALTH": False,
            "EXPERT_RECOVERY_STEPS": 0,
        }
    return {"policy_steps_executed": executed, "transition_rows": transition_rows, "full_expert_after": takeover}


def nearest_distance(obs, references, normalization):
    delta = (references - np.asarray(obs, np.float32)) / normalization
    distances = np.sqrt(np.sum(np.square(delta), axis=1))
    index = int(np.argmin(distances))
    return float(distances[index]), index


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    OUT.mkdir(parents=True)
    started = time.time()
    protected_paths = {
        "production_demo": p4.DEMO,
        "checkpoint_2k": CHECKPOINT,
        "multi_boundary_set": BOUNDARY_SET,
        "policy_class_audit_data": AUDIT_DATA,
    }
    protected_before = {name: sha(path) for name, path in protected_paths.items()}
    dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == DEMO_SHA, "production Demo SHA")
    req(protected_before["multi_boundary_set"] == "9ea19bc02104475c71614acc55aac28d090ff5b3a2d5bb875279dcfd60aff4c1", "boundary set SHA")

    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
    boundary_rows = pickle.loads(BOUNDARY_SET.read_bytes())
    boundary_obs = np.asarray([row["observation"] for row in boundary_rows], np.float32)
    audit_rows = pickle.loads(AUDIT_DATA.read_bytes())
    audit_obs = np.asarray([row["observations"] for row in audit_rows], np.float32)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    core = float(preflight["C1_CORE_THRESHOLD"])
    outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    cfg = load_config(2)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"])
        c1_params = source["c1_params"]
        definition, params = tail.load_head_checkpoint(CHECKPOINT)
        episodes = collect_trajectory(agent, c1_params, definition, params, support, scale, core, outer, demo_obs, demo_scale)
        req(all(episodes), "three nonempty C2 trajectories")

        unique = {}
        for episode in episodes:
            for row in episode:
                unique.setdefault((row["obs20_sha"], tail.snapshot_hash(row["snapshot"])), row)
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, support, scale, core, outer)
        try:
            takeover_by_key = {}
            for key, row in unique.items():
                takeover_by_key[key] = oracle.takeover(row["snapshot"])

            traces = []
            first_losses = []
            for episode in episodes:
                trace = []
                for row in episode:
                    key = (row["obs20_sha"], tail.snapshot_hash(row["snapshot"]))
                    takeover = takeover_by_key[key]
                    boundary_distance, boundary_index = nearest_distance(row["observation"], boundary_obs, demo_scale)
                    audit_distance, audit_index = nearest_distance(row["observation"], audit_obs, demo_scale)
                    trace.append({
                        **{k: v for k, v in row.items() if k != "snapshot"},
                        "nearest_frozen_boundary_distance": boundary_distance,
                        "nearest_frozen_boundary_index": boundary_index,
                        "exact_frozen_boundary_match": bool(boundary_distance == 0.0),
                        "nearest_training_audit_distance": audit_distance,
                        "nearest_training_audit_index": audit_index,
                        "exact_training_audit_match": bool(audit_distance == 0.0),
                        **takeover,
                    })
                first_bad = next((r["policy_step"] for r in trace if not r["EXPERT_SHELL_RECOVERABLE"]), None)
                first_losses.append(first_bad)
                traces.append(trace)

            causal = []
            for episode, trace, first_bad in zip(episodes, traces, first_losses):
                if first_bad is None:
                    causal.append({"seed": trace[0]["seed"], "first_nonrecoverable_step": None})
                    continue
                last_good = max(r["policy_step"] for r in trace if r["policy_step"] < first_bad and r["EXPERT_SHELL_RECOVERABLE"])
                source = episode[last_good]
                source_takeover = takeover_by_key[(source["obs20_sha"], tail.snapshot_hash(source["snapshot"]))]
                expert_action = np.asarray(demos[source_takeover["expert_source_row"]]["actions"], np.float32)
                policy_one = oracle.action_then_takeover(source["snapshot"], source["policy_action"])
                expert_one = oracle.action_then_takeover(source["snapshot"], expert_action)
                horizon_branches = {}
                for horizon in HORIZONS:
                    start = max(0, first_bad - horizon)
                    actions = [episode[i]["policy_action"] for i in range(start, first_bad)]
                    horizon_branches[horizon] = {
                        "start_step": start,
                        "end_step_exclusive": first_bad,
                        "branch": sequence_then_takeover(oracle, episode[start]["snapshot"], actions),
                    }
                one_step_crossing = bool(
                    not policy_one["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"]
                    and expert_one["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"]
                )
                causal.append({
                    "seed": trace[0]["seed"],
                    "last_recoverable_step": last_good,
                    "first_nonrecoverable_step": first_bad,
                    "source_obs20_sha": source["obs20_sha"],
                    "source_policy_action": source["policy_action"],
                    "source_expert_action": expert_action,
                    "action_abs_error": np.abs(source["policy_action"][:6] - expert_action[:6]),
                    "policy_one_step_then_expert": policy_one,
                    "expert_one_step_then_expert": expert_one,
                    "horizon_branches": horizon_branches,
                    "ONE_STEP_BOUNDARY_CROSSING": "YES" if one_step_crossing else "NO",
                })

            loss_steps = [x for x in first_losses if x is not None]
            exact_boundary_at_loss = []
            exact_training_at_loss = []
            for trace, first_bad in zip(traces, first_losses):
                if first_bad is not None:
                    exact_boundary_at_loss.append(trace[first_bad]["exact_frozen_boundary_match"])
                    exact_training_at_loss.append(trace[first_bad]["exact_training_audit_match"])
            one_step_all = bool(causal and all(x.get("ONE_STEP_BOUNDARY_CROSSING") == "YES" for x in causal))
            unseen_all = bool(exact_boundary_at_loss and not any(exact_boundary_at_loss) and not any(exact_training_at_loss))
            if one_step_all and unseen_all:
                failure_mode = "UNSEEN_TRAJECTORY_BOUNDARY_ONE_STEP_CROSSING"
                next_task = "C2_TRAJECTORY_RECOVERY_HEAD_V1"
                authorized = "YES"
            elif one_step_all:
                failure_mode = "KNOWN_DISTRIBUTION_ONE_STEP_BOUNDARY_CROSSING"
                next_task = "C2_TRAJECTORY_RECOVERY_HEAD_V1"
                authorized = "YES"
            elif loss_steps:
                failure_mode = "MULTI_STEP_COMPOUNDING_TO_RECOVERY_LOSS"
                next_task = "C2_TRAJECTORY_RECOVERY_HEAD_V1"
                authorized = "YES"
            else:
                failure_mode = "NO_EXPERT_RECOVERY_LOSS_BEFORE_ROLLOUT_END"
                next_task = "C2_CLOSED_LOOP_FAILURE_RECLASSIFICATION"
                authorized = "NO"

            dump(OUT / "trajectory_recoverability_trace.json", {"checkpoint": str(CHECKPOINT), "traces": traces})
            dump(OUT / "first_recovery_loss_causal_branches.json", causal)
            result = {
                "C2_TRAJECTORY_LEVEL_RECOVERABILITY_CLOSURE": "PASS",
                "TRAINING_STARTED": False,
                "CHECKPOINT_2K_SHA": protected_before["checkpoint_2k"],
                "C0_BASE_SHA": geo.thash(agent.state.params["modules_actor"]),
                "C1_RESIDUAL_SHA": geo.thash(c1_params),
                "TRAJECTORY_EPISODES": len(episodes),
                "TRAJECTORY_LENGTHS": [len(x) for x in episodes],
                "UNIQUE_EXACT_STATES_AUDITED": len(unique),
                "FIRST_NONRECOVERABLE_STEPS": first_losses,
                "LAST_RECOVERABLE_STEPS": [x.get("last_recoverable_step") for x in causal],
                "ONE_STEP_BOUNDARY_CROSSING_ALL": "YES" if one_step_all else "NO",
                "FIRST_LOSS_EXACT_FROZEN_BOUNDARY_MATCH_ANY": "YES" if any(exact_boundary_at_loss) else "NO",
                "FIRST_LOSS_EXACT_TRAINING_AUDIT_MATCH_ANY": "YES" if any(exact_training_at_loss) else "NO",
                "PRIMARY_ROOT_CAUSE": failure_mode,
                "C2_TRAJECTORY_RECOVERY_HEAD_V1_AUTHORIZED": authorized,
                "NEXT": next_task,
                "PRODUCTION_DEMO_SHA_UNCHANGED": "PENDING_FINAL_CHECK",
                "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
                "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
                "C0_C1_FROZEN": "YES",
                "CANONICAL_EVAL_CHANGED": "NO",
                "PROTECTED_ARTIFACTS_CHANGED": "PENDING_FINAL_CHECK",
                "TRAINING_RUNNING": False,
                "RESIDUAL_PROCESSES": "NONE",
                "elapsed_seconds": time.time() - started,
            }
            protected_after = {name: sha(path) for name, path in protected_paths.items()}
            req(protected_after == protected_before, "protected artifact mutation")
            dump(OUT / "protected_after.json", protected_after)
            result["PRODUCTION_DEMO_SHA_UNCHANGED"] = "YES"
            result["PROTECTED_ARTIFACTS_CHANGED"] = "NO"
            dump(OUT / "result.json", result)
            print(json.dumps(jvalue(result), indent=2, sort_keys=True))
        finally:
            oracle.close()


if __name__ == "__main__":
    main()
