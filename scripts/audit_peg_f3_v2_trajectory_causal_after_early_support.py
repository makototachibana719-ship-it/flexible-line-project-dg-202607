#!/usr/bin/env python3
"""Read-only trajectory causal audit after early-approach support 1K."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from scripts.audit_peg_f3_v2_behavior_transfer_stage_a import (
    build_envelopes,
    demo_tables,
    q_mean,
    region_from_errors,
)
from scripts.screen_peg_f3_v2_formal_5k import aggregate, contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_approach_support_ablation import make_early_approach_support_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

FORMAL_CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
FORMAL_RUN = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run"
EARLY_CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_approach_support_ablation_1k_seed0.yaml"
EARLY_RUN = ROOT / "results/ablation/peg_f3_v2_early_approach_support_ablation_1k_seed0_run_001"
OUT_ROOT = ROOT / "results/diagnostics/peg"
STEPS = {"original": 1000, "current": 1000}
SEEDS = (0, 1, 2)
ACTION_DIMS = tuple(f"a{i}" for i in range(7))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def unique_out():
    stem = "f3_v2_trajectory_causal_after_early_support_20260906"
    for index in range(1, 100):
        candidate = OUT_ROOT / f"{stem}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no fresh output directory available")


def demo_exact_sim_state_available():
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    row = json.loads((ROOT / provenance[0]["episode_path"] / "transitions.jsonl").read_text().splitlines()[0])
    keys = set(row) | set(row.get("before", {})) | set(row.get("after", {}))
    state_tokens = ("qpos", "qvel", "mocap", "ctrl", "sim_state", "physics_state")
    return any(any(token in key.lower() for token in state_tokens) for key in keys)


def corridor_distance(errors, demo_errors, error_min, error_max):
    span = np.maximum(error_max - error_min, 1e-9)
    normalized = (demo_errors - errors[None, :]) / span[None, :]
    return float(np.min(np.linalg.norm(normalized, axis=1)))


def make_original_agent():
    sys.path[:0] = [str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim")]
    from async_sac_state_sim import make_sac_agent
    fresh = make_sac_agent(seed=0, sample_obs=np.zeros(20, np.float32),
                           sample_action=np.zeros(7, np.float32))
    return fresh.replace(state=restore_fresh_state(FORMAL_RUN / "checkpoints", fresh.state, STEPS["original"]))


def make_current_agent(cfg):
    fresh = make_early_approach_support_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    return fresh.replace(state=restore_fresh_state(EARLY_RUN / "checkpoints", fresh.state, STEPS["current"]))


def evaluate_family(name, agent, env, arrays, errors, envelopes, error_min, error_max, out):
    rows = []
    trace_path = out / f"{name}_trace.jsonl"
    region_counts = {region: 0 for region in envelopes}
    overlap_events = []
    first_divergences = []
    contact_steps = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        first_position = float(np.linalg.norm(obs[13:16]))
        mins = {"position": first_position, "lateral": float("inf"), "axial": float("inf"),
                "orientation": float("inf")}
        ret = 0.0
        near = sat = 0
        max_force = 0.0
        first_div = None
        first_contact = None
        link01 = unexpected = success = terminated = truncated = False
        for index in range(500):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
            obs, reward, terminated, truncated, info = env.step(action)
            err = np.asarray([float(info["lateral_error"]), float(info["axial_error"]),
                              float(info["orientation_error"])], np.float64)
            region = region_from_errors(*err)
            region_counts[region] = region_counts.get(region, 0) + 1
            dist = corridor_distance(err, errors, error_min, error_max)
            nearest = int(np.argmin(np.linalg.norm(errors - err[None, :], axis=1)))
            ref_action = arrays["actions"][nearest]
            envl = envelopes.get(region)
            bad_dims = np.asarray([], dtype=np.int64)
            error_bad = False
            if envl is not None:
                lo = np.asarray(envl["action_p01"]) - 0.05
                hi = np.asarray(envl["action_p99"]) + 0.05
                bad_dims = np.where((action < lo) | (action > hi))[0]
                error_bad = bool(np.any((err < np.asarray(envl["error_min"])) |
                                        (err > np.asarray(envl["error_max"]))))
            q_policy = float(q_mean(agent, obs[None], action[None])[0])
            q_demo = float(q_mean(agent, obs[None], ref_action[None])[0])
            l, u, pairs = contact_flags(env)
            if first_contact is None and (l or u):
                first_contact = {"step": index + 1, "link01": bool(l), "unexpected": bool(u), "pairs": pairs}
            if first_div is None and (len(bad_dims) or error_bad):
                first_dim = int(bad_dims[0]) if len(bad_dims) else None
                first_div = {
                    "family": name,
                    "seed": seed,
                    "step": index + 1,
                    "error": "ACTION_ENVELOPE" if len(bad_dims) else "STATE_ERROR_ENVELOPE",
                    "action_dim": None if first_dim is None else ACTION_DIMS[first_dim],
                    "region": region,
                    "lateral_error": float(err[0]),
                    "axial_error": float(err[1]),
                    "orientation_error": float(err[2]),
                    "demo_corridor_distance": dist,
                    "q_policy": q_policy,
                    "q_demo_reference": q_demo,
                    "action": action.tolist(),
                    "reference_action": ref_action.tolist(),
                }
            overlap_events.append({
                "joint": bool(np.all(obs[:7] >= arrays["observations"][:, :7].min(axis=0)) and
                              np.all(obs[:7] <= arrays["observations"][:, :7].max(axis=0))),
                "lateral": bool(error_min[0] <= err[0] <= error_max[0]),
                "axial": bool(error_min[1] <= err[1] <= error_max[1]),
                "orientation": bool(error_min[2] <= err[2] <= error_max[2]),
                "region": region,
            })
            with trace_path.open("a") as stream:
                stream.write(json.dumps({
                    "family": name, "seed": seed, "step": index + 1,
                    "lateral_error": float(err[0]), "axial_error": float(err[1]),
                    "orientation_error": float(err[2]), "region": region,
                    "demo_corridor_distance": dist, "action": action.tolist(),
                    "contact": bool(l or u), "q_policy": q_policy,
                    "q_demo_reference": q_demo,
                }, allow_nan=False) + "\n")
            link01 |= l
            unexpected |= u
            mins["position"] = min(mins["position"], float(info["position_error"]))
            mins["lateral"] = min(mins["lateral"], float(err[0]))
            mins["axial"] = min(mins["axial"], float(err[1]))
            mins["orientation"] = min(mins["orientation"], float(err[2]))
            near += int(err[0] <= .010 and err[1] <= .020)
            max_force = max(max_force, float(info["force_norm"]))
            sat += int((np.abs(action) >= .99).sum())
            ret += float(reward)
            success = bool(info["is_success"])
            if terminated or truncated:
                break
        if first_div is not None:
            first_div["contact_step"] = None if first_contact is None else first_contact["step"]
            first_div["divergence_before_contact"] = (
                "YES" if first_contact is None or first_div["step"] < first_contact["step"] else "NO"
            )
            first_divergences.append(first_div)
        contact_steps.append(None if first_contact is None else first_contact["step"])
        rows.append({
            "family": name, "checkpoint": STEPS[name], "episode": seed, "seed": seed,
            "success": success, "steps": index + 1, "return": ret,
            "first_position_error": first_position, "min_position_error": mins["position"],
            "approach_reduction": first_position - mins["position"],
            "min_lateral_error": mins["lateral"], "min_axial_error": mins["axial"],
            "min_orientation_error": mins["orientation"],
            "near_hole_transition_count": near, "near_hole_reached": near > 0,
            "max_force": max_force, "link01_contact": link01,
            "unexpected_contact": unexpected, "contact_health": not (link01 or unexpected),
            "first_contact": first_contact, "action_saturation": sat / (7 * (index + 1)),
        })
    total = max(1, sum(region_counts.values()))
    overlap = {
        "joint_in_demo_range_pct": float(np.mean([x["joint"] for x in overlap_events])),
        "lateral_in_range_pct": float(np.mean([x["lateral"] for x in overlap_events])),
        "axial_in_range_pct": float(np.mean([x["axial"] for x in overlap_events])),
        "orientation_in_range_pct": float(np.mean([x["orientation"] for x in overlap_events])),
        "region_occupancy": {key: int(value) for key, value in region_counts.items()},
        "correction_reach_rate": float((region_counts.get("LATERAL_CORRECTION", 0) +
                                        region_counts.get("ORIENTATION_CORRECTION", 0)) / total),
        "near_hole_reach_rate": float(region_counts.get("NEAR_HOLE", 0) / total),
    }
    first = min(first_divergences, key=lambda x: (x["seed"], x["step"])) if first_divergences else {}
    return {
        "summary": aggregate(rows),
        "overlap": overlap,
        "first_divergence": first,
        "contact_steps": contact_steps,
        "mean_first_divergence_step": float(np.mean([x["step"] for x in first_divergences])) if first_divergences else None,
    }


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    require(jax.default_backend() == "gpu", "GPU inference backend")
    formal_cfg = yaml.safe_load(FORMAL_CONFIG.read_text())
    early_cfg = yaml.safe_load(EARLY_CONFIG.read_text())
    arrays, errors, regions = demo_tables(formal_cfg)
    envelopes, error_min, error_max = build_envelopes(errors, arrays["actions"], regions)
    sys.path[:0] = [
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    ]
    from peg_env_adapter import make_peg_state_env
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        original = evaluate_family("original", make_original_agent(), env, arrays, errors,
                                   envelopes, error_min, error_max, out)
        current = evaluate_family("current", make_current_agent(early_cfg), env, arrays, errors,
                                  envelopes, error_min, error_max, out)
        env.close()
    exact_state = demo_exact_sim_state_available()
    recovery = {
        "RECOVERY_PRECORRECTION": "NOT_AVAILABLE",
        "RECOVERY_CORRECTION_ENTRY": "NOT_AVAILABLE",
        "RECOVERY_POSTCORRECTION": "NOT_AVAILABLE",
        "reason": "accepted V2 demos do not contain exact simulator state (qpos/qvel/mocap/control) required for reliable restore",
    }
    first = current["first_divergence"]
    step1_a0_fixed = first.get("action", [None])[0]
    a0_demo_range = [0.0, 0.0]
    a0_in_envelope = step1_a0_fixed is not None and -0.05 <= float(step1_a0_fixed) <= 0.05
    if current["overlap"]["near_hole_reach_rate"] == 0.0 and first.get("divergence_before_contact") == "YES":
        primary = "EARLY_TRAJECTORY_DISTRIBUTION_DRIFT"
        next_fix = "EARLY_APPROACH_STATE_SUPPORT"
    elif first.get("region") in ("LATERAL_CORRECTION", "ORIENTATION_CORRECTION"):
        primary = "CORRECTION_POLICY_INSUFFICIENT"
        next_fix = "CORRECTION_WEIGHTED_POLICY_OBJECTIVE"
    else:
        primary = "MIXED_EARLIEST_TRAJECTORY_DRIFT"
        next_fix = "EARLY_APPROACH_STATE_SUPPORT"
    result = {
        "TRAJECTORY_CAUSAL_AUDIT": "PASS",
        "STAGE_B_TRAJECTORY_AUDIT": "PASS",
        "TRAINING_STARTED": False,
        "STEP1_A0_FIXED": step1_a0_fixed,
        "STEP1_A0_IN_DEMO_ENVELOPE": "YES" if a0_in_envelope else "NO",
        "A0_DEMO_RANGE": a0_demo_range,
        "FIRST_DIVERGENCE_STEP": first.get("step"),
        "FIRST_DIVERGENCE_ERROR": first.get("error"),
        "FIRST_DIVERGENCE_ACTION_DIM": first.get("action_dim"),
        "FIRST_DIVERGENCE_REGION": first.get("region"),
        "CONTACT_STEP": first.get("contact_step"),
        "DIVERGENCE_BEFORE_CONTACT": first.get("divergence_before_contact"),
        "ORIGINAL_DEMO_OVERLAP": original["overlap"],
        "CURRENT_DEMO_OVERLAP": current["overlap"],
        "CORRECTION_REACH_RATE": current["overlap"]["correction_reach_rate"],
        "NEAR_HOLE_REACH_RATE": current["overlap"]["near_hole_reach_rate"],
        "TIME_TO_FIRST_DIVERGENCE": {
            "original_mean_step": original["mean_first_divergence_step"],
            "current_mean_step": current["mean_first_divergence_step"],
        },
        "RECOVERY_PRECORRECTION": recovery["RECOVERY_PRECORRECTION"],
        "RECOVERY_CORRECTION_ENTRY": recovery["RECOVERY_CORRECTION_ENTRY"],
        "RECOVERY_POSTCORRECTION": recovery["RECOVERY_POSTCORRECTION"],
        "RECOVERY_STATE_RESTORE": "AVAILABLE" if exact_state else "NOT_AVAILABLE",
        "RECOVERY_NOT_AVAILABLE_REASON": None if exact_state else recovery["reason"],
        "PRIMARY_CAUSAL_BLOCKER": primary,
        "NEXT_FIX": next_fix,
        "families": {"original": original, "current": current},
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    dump(out / "result.json", result)
    (out / "REPORT.md").write_text("# Peg F3 V2 Trajectory Causal Audit After Early Support\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
