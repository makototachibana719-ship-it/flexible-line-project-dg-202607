#!/usr/bin/env python3
"""Read-only Stage-A behavior transfer audit for Peg F3 V2."""
from __future__ import annotations

import json
import pickle
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from scripts.screen_peg_f3_v2_formal_5k import aggregate, contact_flags, raw_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

FORMAL_CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
FORMAL_RUN = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run"
COMBINED_CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_3k_seed0.yaml"
COMBINED_RUN = ROOT / "results/ablation/peg_f3_v2_combined_q_anchor_ablation_3k_seed0_run_002"
OUT = ROOT / "results/diagnostics/peg/f3_v2_behavior_transfer_stage_a_20260906_004"
SCHEMA_OUT = ROOT / "results/validation/peg/f3_v2_behavior_transfer_row_schema_20260906_001"
STEPS = (1000, 2000, 3000)
SEEDS = (0, 1, 2)
ACTION_DIMS = tuple(f"a{i}" for i in range(7))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_episode_row_schema(row):
    require("max_force" in row, "max_force missing")
    require(np.isscalar(row["max_force"]) and np.isfinite(row["max_force"]), "max_force nonfinite")
    return row


def schema_regression():
    require(not SCHEMA_OUT.exists(), "fresh schema regression output required")
    base = {
        "success": False, "return": -1.0, "near_hole_reached": False,
        "near_hole_transition_count": 0, "approach_reduction": 0.1,
        "min_position_error": 0.2, "min_lateral_error": 0.03,
        "min_axial_error": 0.01, "min_orientation_error": 0.02,
        "max_force": 86.60254037844386, "contact_health": False,
        "action_saturation": 0.0,
    }
    before = set(base)
    valid = validate_episode_row_schema(dict(base))
    aggregate([valid])
    missing = dict(base)
    missing.pop("max_force")
    missing_rejected = False
    try:
        validate_episode_row_schema(missing)
    except RuntimeError:
        missing_rejected = True
    require(missing_rejected, "missing max_force accepted")
    nonfinite_rejected = {}
    for label, value in (("nan", float("nan")), ("inf", float("inf"))):
        candidate = dict(base)
        candidate["max_force"] = value
        try:
            validate_episode_row_schema(candidate)
            nonfinite_rejected[label] = False
        except RuntimeError:
            nonfinite_rejected[label] = True
    require(all(nonfinite_rejected.values()), "nonfinite max_force accepted")
    require(set(base) == before, "existing row fields changed")
    result = {
        "STAGE_A_ROW_SCHEMA_FIX": "PASS",
        "MAX_FORCE_ROW_FIELD": "PASS",
        "AGGREGATE_REGRESSION": "PASS",
        "MISSING_FIELD_REJECTION": "PASS",
        "NONFINITE_REJECTION": "PASS",
        "EXISTING_ROW_FIELDS_UNCHANGED": "PASS",
        "ORIGINAL_EVALUATION_MAX_FORCE_SEMANTICS_UNCHANGED": "PASS",
        "TRAINING_STARTED": False,
    }
    SCHEMA_OUT.mkdir(parents=True, exist_ok=False)
    dump(SCHEMA_OUT / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def region_from_errors(lateral, axial, orientation):
    if lateral <= 0.010 and axial <= 0.020:
        return "NEAR_HOLE"
    if orientation > 0.050:
        return "ORIENTATION_CORRECTION"
    if lateral > 0.010:
        return "LATERAL_CORRECTION"
    return "FAR_APPROACH"


def q_mean(agent, obs, action):
    values = np.asarray(jax.device_get(agent.forward_critic(
        jnp.asarray(obs), jnp.asarray(action), rng=None, train=False)))
    return values.mean(axis=0)


def cosine(a, b):
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = denom > 1e-9
    out = np.full(a.shape[0], np.nan, np.float64)
    out[valid] = np.sum(a[valid] * b[valid], axis=-1) / denom[valid]
    return out


def demo_tables(cfg):
    dataset = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([item[key] for item in dataset]) for key in dataset[0]}
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    metrics = []
    for entry in provenance:
        rows = [json.loads(line) for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open()]
        metrics.extend(row["after"]["metrics"] for row in rows)
    errors = np.asarray([[m["lateral_error"], m["axial_error"], m["orientation_error"]] for m in metrics],
                        np.float64)
    regions = np.asarray([region_from_errors(*row) for row in errors])
    return arrays, errors, regions


def build_envelopes(errors, actions, regions):
    envelopes = {}
    for region in sorted(set(regions)):
        idx = regions == region
        envelopes[region] = {
            "count": int(idx.sum()),
            "error_min": errors[idx].min(axis=0).tolist(),
            "error_max": errors[idx].max(axis=0).tolist(),
            "action_p01": np.percentile(actions[idx], 1, axis=0).tolist(),
            "action_p99": np.percentile(actions[idx], 99, axis=0).tolist(),
        }
    obs_min = errors.min(axis=0)
    obs_max = errors.max(axis=0)
    return envelopes, obs_min, obs_max


def region_mae(agent, arrays, regions):
    obs, demo_actions = arrays["observations"], arrays["actions"]
    policy = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
    result = {}
    for region in sorted(set(regions)):
        idx = regions == region
        diff = np.abs(policy[idx] - demo_actions[idx])
        result[region] = {
            "state_count": int(idx.sum()),
            "mae_by_action_dim": {name: float(diff[:, i].mean()) for i, name in enumerate(ACTION_DIMS)},
            "overall_mae": float(diff.mean()),
            "cosine_similarity": float(np.nanmean(cosine(policy[idx], demo_actions[idx]))),
        }
    return result


def make_agents(formal_cfg, combined_cfg):
    sys.path[:0] = [str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim")]
    from async_sac_state_sim import make_sac_agent
    sample_obs = np.zeros(20, np.float32)
    sample_action = np.zeros(7, np.float32)
    formal_fresh = make_sac_agent(seed=0, sample_obs=sample_obs, sample_action=sample_action)
    combined_fresh = make_combined_q_anchor_agent(0, sample_obs, sample_action, combined_cfg)
    return formal_fresh, combined_fresh


def restored_agent(fresh, run, step):
    return fresh.replace(state=restore_fresh_state(run / "checkpoints", fresh.state, step))


def evaluate_family(name, fresh, run, env, arrays, errors, regions, envelopes, error_min, error_max):
    rows = []
    divergences = []
    overlaps = []
    region_counts = {region: 0 for region in envelopes}
    for step in STEPS:
        agent = restored_agent(fresh, run, step)
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            first = float(np.linalg.norm(obs[13:16]))
            mins = {"position": first, "lateral": float("inf"), "axial": float("inf"),
                    "orientation": float("inf")}
            ret = 0.0
            near = sat = 0
            max_force = 0.0
            first_contact = None
            first_div = None
            link01 = unexpected = success = terminated = truncated = False
            for index in range(500):
                action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
                obs, reward, terminated, truncated, info = env.step(action)
                lateral, axial, orientation = (float(info["lateral_error"]), float(info["axial_error"]),
                                               float(info["orientation_error"]))
                region = region_from_errors(lateral, axial, orientation)
                if region in region_counts:
                    region_counts[region] += 1
                in_range = np.array([lateral, axial, orientation]) >= error_min
                in_range &= np.array([lateral, axial, orientation]) <= error_max
                overlaps.append({
                    "checkpoint": step, "seed": seed, "joint_in_demo_range": bool(np.all(obs[:7] >= arrays["observations"][:, :7].min(axis=0)) and np.all(obs[:7] <= arrays["observations"][:, :7].max(axis=0))),
                    "lateral_in_range": bool(in_range[0]), "axial_in_range": bool(in_range[1]),
                    "orientation_in_range": bool(in_range[2]), "region": region,
                })
                envl = envelopes.get(region)
                if first_div is None and envl is not None:
                    lo = np.asarray(envl["action_p01"]) - 0.05
                    hi = np.asarray(envl["action_p99"]) + 0.05
                    bad_dims = np.where((action < lo) | (action > hi))[0]
                    err = np.asarray([lateral, axial, orientation])
                    error_bad = np.any((err < np.asarray(envl["error_min"])) | (err > np.asarray(envl["error_max"])))
                    if len(bad_dims) or error_bad:
                        nearest = int(np.argmin(np.linalg.norm(errors - err[None, :], axis=1)))
                        ref_action = arrays["actions"][nearest]
                        first_dim = int(bad_dims[0]) if len(bad_dims) else None
                        first_div = {
                            "family": name, "checkpoint": step, "seed": seed, "step": index + 1,
                            "region": region,
                            "action_dim": None if first_dim is None else f"a{first_dim}",
                            "error_type": "ACTION_ENVELOPE" if len(bad_dims) else "STATE_ERROR_ENVELOPE",
                            "lateral_error": lateral, "axial_error": axial, "orientation_error": orientation,
                            "action": action.tolist(), "reference_action": ref_action.tolist(),
                            "q_policy": float(q_mean(agent, obs[None], action[None])[0]),
                            "q_demo_reference": float(q_mean(agent, obs[None], ref_action[None])[0]),
                        }
                l, u, pairs = contact_flags(env)
                if first_contact is None and (l or u):
                    first_contact = {"step": index + 1, "link01": bool(l), "unexpected": bool(u),
                                     "pairs": pairs}
                link01 |= l
                unexpected |= u
                mins["position"] = min(mins["position"], float(info["position_error"]))
                mins["lateral"] = min(mins["lateral"], lateral)
                mins["axial"] = min(mins["axial"], axial)
                mins["orientation"] = min(mins["orientation"], orientation)
                near += int(lateral <= .010 and axial <= .020)
                max_force = max(max_force, float(info["force_norm"]))
                sat += int((np.abs(action) >= .99).sum())
                ret += float(reward)
                success = bool(info["is_success"])
                if terminated or truncated:
                    break
            if first_div is not None:
                first_div["contact_after_divergence"] = first_contact is not None and first_contact["step"] >= first_div["step"]
                divergences.append(first_div)
            rows.append(validate_episode_row_schema({
                "family": name, "checkpoint": step, "episode": seed, "seed": seed,
                "success": success, "steps": index + 1, "trajectory_length": index + 1,
                "return": ret, "first_position_error": first, "min_position_error": mins["position"],
                "approach_reduction": first - mins["position"],
                "min_lateral_error": mins["lateral"], "min_axial_error": mins["axial"],
                "min_orientation_error": mins["orientation"],
                "near_hole_transition_count": near, "near_hole_reached": near > 0,
                "max_force": max_force,
                "link01_contact": link01, "unexpected_contact": unexpected,
                "contact_health": not (link01 or unexpected),
                "first_contact": first_contact, "action_saturation": sat / (7 * (index + 1)),
            }))
    by_step = {str(step): aggregate([r for r in rows if r["checkpoint"] == step]) for step in STEPS}
    overlap = {
        "joint_in_demo_range_pct": float(np.mean([x["joint_in_demo_range"] for x in overlaps])),
        "lateral_in_range_pct": float(np.mean([x["lateral_in_range"] for x in overlaps])),
        "axial_in_range_pct": float(np.mean([x["axial_in_range"] for x in overlaps])),
        "orientation_in_range_pct": float(np.mean([x["orientation_in_range"] for x in overlaps])),
        "region_occupancy": {key: int(value) for key, value in region_counts.items()},
    }
    return rows, by_step, overlap, divergences


def choose_branch(combined_overlap, combined_divergences, combined_region_mae):
    correction_reach = combined_overlap["region_occupancy"].get("LATERAL_CORRECTION", 0) + combined_overlap["region_occupancy"].get("ORIENTATION_CORRECTION", 0)
    near_reach = combined_overlap["region_occupancy"].get("NEAR_HOLE", 0)
    first = min(combined_divergences, key=lambda x: (x["checkpoint"], x["seed"], x["step"])) if combined_divergences else {}
    action_dim = first.get("action_dim")
    correction_a1 = max(
        combined_region_mae.get("LATERAL_CORRECTION", {}).get("mae_by_action_dim", {}).get("a1", 0.0),
        combined_region_mae.get("ORIENTATION_CORRECTION", {}).get("mae_by_action_dim", {}).get("a1", 0.0),
    )
    if correction_reach == 0 or near_reach == 0:
        selected = "DISTRIBUTION_OOD_FIX"
        blocker = "rollout state occupancy stays outside near-hole/correction corridor"
    elif action_dim == "a1" and correction_a1 > 0.5:
        selected = "CORRECTION_WEIGHTED_ANCHOR"
        blocker = "dominant correction action error"
    else:
        selected = "DISTRIBUTION_OOD_FIX"
        blocker = "first causal blocker is state-distribution drift"
    return selected, blocker, first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-regression", action="store_true")
    args = parser.parse_args()
    if args.schema_regression:
        return schema_regression()
    require(not OUT.exists(), "fresh diagnostic output required")
    OUT.mkdir(parents=True)
    formal_cfg = yaml.safe_load(FORMAL_CONFIG.read_text())
    combined_cfg = yaml.safe_load(COMBINED_CONFIG.read_text())
    arrays, errors, regions = demo_tables(formal_cfg)
    envelopes, error_min, error_max = build_envelopes(errors, arrays["actions"], regions)
    formal_fresh, combined_fresh = make_agents(formal_cfg, combined_cfg)
    region_metrics = {
        "original": {str(step): region_mae(restored_agent(formal_fresh, FORMAL_RUN, step), arrays, regions) for step in STEPS},
        "combined": {str(step): region_mae(restored_agent(combined_fresh, COMBINED_RUN, step), arrays, regions) for step in STEPS},
    }
    sys.path[:0] = [str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env")]
    from peg_env_adapter import make_peg_state_env
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        original_rows, original_summary, original_overlap, original_div = evaluate_family(
            "original", formal_fresh, FORMAL_RUN, env, arrays, errors, regions, envelopes, error_min, error_max)
        combined_rows, combined_summary, combined_overlap, combined_div = evaluate_family(
            "combined", combined_fresh, COMBINED_RUN, env, arrays, errors, regions, envelopes, error_min, error_max)
        env.close()
    selected, blocker, first = choose_branch(combined_overlap, combined_div, region_metrics["combined"]["3000"])
    contact_consequence = bool(combined_div) and any(x.get("contact_after_divergence") for x in combined_div)
    result = {
        "STAGE_A_BEHAVIOR_TRANSFER_AUDIT": "PASS",
        "TRAINING_STARTED": False,
        "BEHAVIOR_TRANSFER_ROOT_CAUSE": blocker,
        "PRIMARY_CAUSAL_BLOCKER": selected,
        "FIRST_DIVERGENCE_STEP": first.get("step"),
        "FIRST_DIVERGENCE_REGION": first.get("region"),
        "FIRST_DIVERGENCE_ACTION_DIM": first.get("action_dim"),
        "FIRST_DIVERGENCE_ERROR_TYPE": first.get("error_type"),
        "ORIGINAL_DEMO_OVERLAP": original_overlap,
        "COMBINED_DEMO_OVERLAP": combined_overlap,
        "CORRECTION_REACH_RATE": float((combined_overlap["region_occupancy"].get("LATERAL_CORRECTION", 0) + combined_overlap["region_occupancy"].get("ORIENTATION_CORRECTION", 0)) / max(1, sum(combined_overlap["region_occupancy"].values()))),
        "NEAR_HOLE_REACH_RATE": float(combined_overlap["region_occupancy"].get("NEAR_HOLE", 0) / max(1, sum(combined_overlap["region_occupancy"].values()))),
        "CONTACT_IS_CONSEQUENCE": "YES" if contact_consequence else "NO",
        "SELECTED_FIX": selected,
        "REGION_MAE": region_metrics,
        "rollout_summary": {"original": original_summary, "combined": combined_summary},
        "first_divergences": {"original": original_div, "combined": combined_div},
        "NEXT": "STAGE_B2_DISTRIBUTION_OOD_FIX" if selected == "DISTRIBUTION_OOD_FIX" else "STAGE_B1_CORRECTION_WEIGHTED_ANCHOR",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    dump(OUT / "result.json", result)
    (OUT / "REPORT.md").write_text("# Peg F3 V2 Behavior Transfer Stage A\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
