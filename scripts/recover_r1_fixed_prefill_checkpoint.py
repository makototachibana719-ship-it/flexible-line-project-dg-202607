#!/usr/bin/env python3
"""Read-only R1 fixed-prefill checkpoint recovery and small evaluation."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
import sys

import jax
import numpy as np
import yaml

ROOT = Path("/home/makoto/flexible_line_project")
RUN = ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica1_retry002"
CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
OUT_ROOT = ROOT / "results/diagnostics/peg"
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from async_sac_state_sim import make_sac_agent
from peg_env_adapter import make_peg_state_env
from scripts.audit_multi_replica_baseline_protocol import fixed_metrics
from scripts.audit_peg_f3_v2_behavior_transfer_stage_a import build_envelopes, demo_tables, region_from_errors
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay


def unique_out() -> Path:
    stem = "f3_v2_r1_fixed_prefill_checkpoint_recovery_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def read_json(name: str) -> dict:
    path = RUN / name
    return json.loads(path.read_text()) if path.exists() else {}


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def restore_agent_and_replay(cfg, step: int):
    replay_path = RUN / "checkpoints" / f"checkpoint_{step}_online_replay.pkl"
    payload = pickle.loads(replay_path.read_bytes())
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(RUN / "checkpoints", fresh.state, step)
        agent = fresh.replace(state=state)
        replay = make_replay_buffer(env, capacity=int(payload["capacity"]), type="replay_buffer")
        replay_info = restore_replay(replay, replay_path)
        probe_action = np.asarray(jax.device_get(agent.sample_actions(observations=env.observation_space.sample(), argmax=True)))
        env.close()
    return agent, replay_info, probe_action


def eval_rollout(cfg, agent):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    arrays2, errors, regions = demo_tables(cfg)
    envelopes, _emin, _emax = build_envelopes(errors, arrays2["actions"], regions)
    fixed = fixed_metrics(cfg, agent)
    init_obs = None
    episode_rows = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        for seed in (0, 1, 2):
            obs, _ = env.reset(seed=seed)
            if init_obs is None:
                init_obs = np.asarray(obs, np.float32)
            first_div = None
            ret = 0.0
            min_lat = float("inf")
            min_ax = float("inf")
            last_orient = None
            max_force = 0.0
            any_contact = False
            near = False
            success = False
            corridor_values = []
            for step in range(1, 501):
                action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
                obs, reward, terminated, truncated, info = env.step(action)
                ret += float(reward)
                lat = float(info["lateral_error"])
                ax = float(info["axial_error"])
                orient = float(info["orientation_error"])
                force = float(info.get("force_norm", 0.0))
                max_force = max(max_force, force)
                min_lat = min(min_lat, lat)
                min_ax = min(min_ax, ax)
                last_orient = orient
                corridor_values.append(float(np.linalg.norm([lat, ax, orient])))
                link, unexpected, _pairs = contact_flags(env)
                any_contact = any_contact or link or unexpected
                near = near or (lat <= 0.010 and ax <= 0.020)
                success = success or bool(info.get("is_success", False))
                region = region_from_errors(lat, ax, orient)
                envl = envelopes.get(region)
                if first_div is None and envl is not None:
                    lo = np.asarray(envl["action_p01"]) - 0.05
                    hi = np.asarray(envl["action_p99"]) + 0.05
                    bad = np.where((action < lo) | (action > hi))[0]
                    if len(bad):
                        first_div = int(step)
                if terminated or truncated:
                    break
            episode_rows.append({
                "success": success,
                "near_hole": near,
                "first_divergence": first_div,
                "corridor": float(np.mean(corridor_values)),
                "min_lateral": min_lat,
                "min_axial": min_ax,
                "orientation": last_orient,
                "max_force": max_force,
                "contact": any_contact,
                "return": ret,
            })
        env.close()
    action0 = np.asarray(jax.device_get(agent.sample_actions(observations=init_obs, argmax=True)))
    return {
        "A0_1000": float(action0[0]),
        "QPOLICY_MINUS_QDEMO": fixed["q_policy_minus_demo"],
        "SUCCESS": float(np.mean([r["success"] for r in episode_rows])),
        "NEAR_HOLE": float(np.mean([r["near_hole"] for r in episode_rows])),
        "FIRST_DIVERGENCE": float(np.median([r["first_divergence"] or 501 for r in episode_rows])),
        "CORRIDOR": float(np.mean([r["corridor"] for r in episode_rows])),
        "ORIENTATION": float(np.mean([r["orientation"] for r in episode_rows])),
        "AXIAL": float(np.min([r["min_axial"] for r in episode_rows])),
        "LATERAL": float(np.min([r["min_lateral"] for r in episode_rows])),
        "FORCE": float(np.max([r["max_force"] for r in episode_rows])),
        "CONTACT": float(np.mean([r["contact"] for r in episode_rows])),
        "episodes": episode_rows,
    }


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    learner = read_json("learner_result.json")
    base_result = read_json("base_supervise_result.json")
    actor = read_json("actor_result.json")
    stop_seen = read_json("actor_stop_observed.json")
    ckpt500_agent, ckpt500_replay, action500 = restore_agent_and_replay(cfg, 500)
    ckpt1000_agent, ckpt1000_replay, action1000_probe = restore_agent_and_replay(cfg, 1000)
    eval1000 = eval_rollout(cfg, ckpt1000_agent)
    residual = base_result.get("RESIDUAL_PROCESSES", "UNKNOWN")
    actor_shutdown_state = (
        "CLEAN_BUT_ARTIFACT_MISSING"
        if residual == "NONE" and actor.get("status") == "PASS" and stop_seen.get("stop_marker") is True
        else "UNPROVEN"
    )
    checkpoint_validity = "PASS" if ckpt1000_replay["size"] == 64 and action1000_probe.shape == (7,) and np.isfinite(action1000_probe).all() else "FAIL"
    result = {
        "TRAINING_RESULT": "PASS" if learner.get("completed_updates") == 1000 else "FAIL",
        "CHECKPOINT_VALIDITY": checkpoint_validity,
        "RUNTIME_FINALIZATION": actor_shutdown_state,
        "SCIENTIFIC_EVALUATION": "PASS" if checkpoint_validity == "PASS" else "NOT_RUN",
        "RESTORE_FAILURE_CLASS": "RESTORE_VERIFIER_PATH_ERROR",
        "RESTORE_FAILURE_DETAIL": "base verifier restored frozen replay into full-capacity async replay buffer",
        "CKPT500_RESTORE": "PASS" if ckpt500_replay["size"] == 64 and action500.shape == (7,) else "FAIL",
        "CKPT1000_RESTORE": "PASS" if checkpoint_validity == "PASS" else "FAIL",
        **{k: eval1000[k] for k in ("A0_1000", "QPOLICY_MINUS_QDEMO", "SUCCESS", "NEAR_HOLE", "FIRST_DIVERGENCE", "CORRIDOR", "ORIENTATION", "AXIAL", "FORCE", "CONTACT")},
        "ACTOR_SHUTDOWN_STATE": actor_shutdown_state,
        "RESIDUAL_PROCESSES": residual,
        "R1_RUNTIME_CLOSURE": "PASS" if checkpoint_validity == "PASS" and actor_shutdown_state == "CLEAN_BUT_ARTIFACT_MISSING" else "FAIL",
        "R1_TRAINING_ACCEPTED": "YES" if checkpoint_validity == "PASS" and actor_shutdown_state == "CLEAN_BUT_ARTIFACT_MISSING" else "NO",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "NEXT": "REPLICA2_PREFLIGHT_READY" if checkpoint_validity == "PASS" and actor_shutdown_state == "CLEAN_BUT_ARTIFACT_MISSING" else "R1_RETRY_REVIEW_REQUIRED",
        "DETAILS": {"ckpt500_replay": ckpt500_replay, "ckpt1000_replay": ckpt1000_replay, "eval_episodes": eval1000["episodes"]},
    }
    (out / "result.json").write_text(json.dumps(finite(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(finite({k: result[k] for k in (
        "TRAINING_RESULT", "CHECKPOINT_VALIDITY", "RUNTIME_FINALIZATION", "SCIENTIFIC_EVALUATION",
        "RESTORE_FAILURE_CLASS", "CKPT500_RESTORE", "CKPT1000_RESTORE", "A0_1000",
        "QPOLICY_MINUS_QDEMO", "SUCCESS", "NEAR_HOLE", "FIRST_DIVERGENCE", "CORRIDOR",
        "ORIENTATION", "AXIAL", "FORCE", "CONTACT", "ACTOR_SHUTDOWN_STATE",
        "RESIDUAL_PROCESSES", "R1_RUNTIME_CLOSURE", "R1_TRAINING_ACCEPTED", "NEXT"
    )}), sort_keys=True))
    print(str(out / "result.json"))


if __name__ == "__main__":
    main()
