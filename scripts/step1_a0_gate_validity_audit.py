#!/usr/bin/env python3
"""Read-only validity audit for the Peg F3 V2 step1 a0 exact-zero gate."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path[:0] = [
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from peg_env_adapter import make_peg_state_env
from scripts.audit_peg_f3_v2_behavior_transfer_stage_a import region_from_errors
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.a0_action_support_balance_ablation import make_a0_action_support_balance_agent
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_pretanh_a0_anchor_ablation import make_early_pretanh_a0_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

BASELINE_CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_pretanh_a0_anchor_ablation_1k_seed0.yaml"
CANDIDATE_CONFIG = ROOT / "configs/peg_insertion/peg_v2_a0_action_support_balance_ablation_1k_seed0.yaml"
RUNS = {
    "baseline_R1": (BASELINE_CONFIG, ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002/replica1", make_early_pretanh_a0_anchor_agent),
    "baseline_R2": (BASELINE_CONFIG, ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002/replica2", make_early_pretanh_a0_anchor_agent),
    "candidate_R1": (CANDIDATE_CONFIG, ROOT / "results/ablation/peg_f3_v2_a0_action_support_balance_n2_20260906/replica1", make_a0_action_support_balance_agent),
    "candidate_R2": (CANDIDATE_CONFIG, ROOT / "results/ablation/peg_f3_v2_a0_action_support_balance_n2_20260906/replica2_retry001", make_a0_action_support_balance_agent),
}
SEEDS = (0, 1, 2)
WINDOWS = (10, 25, 50)
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out() -> Path:
    stem = "f3_v2_step1_a0_gate_validity_20260906"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def load_demo(cfg: dict):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, metrics = [], []
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            row = json.loads(line)
            phases.append(row["phase"])
            m = row["after"]["metrics"]
            metrics.append([m["lateral_error"], m["axial_error"], m["orientation_error"]])
    errors = np.asarray(metrics, np.float64)
    regions = np.asarray([region_from_errors(*err) for err in errors])
    return arrays, errors, regions, np.asarray(phases)


def build_support(errors, obs, regions):
    support = {}
    for region in sorted(set(regions)):
        idx = regions == region
        support[region] = {
            "error_min": errors[idx].min(axis=0),
            "error_max": errors[idx].max(axis=0),
            "obs_min": obs[idx, :7].min(axis=0),
            "obs_max": obs[idx, :7].max(axis=0),
        }
    return support


def load_agent(cfg: dict, run: Path, factory):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = factory(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def nearest_demo_action(errors, actions, regions, region, err):
    idx = np.where(regions == region)[0]
    if idx.size == 0:
        idx = np.arange(len(errors))
    nearest = idx[int(np.argmin(np.linalg.norm(errors[idx] - err[None, :], axis=1)))]
    return actions[nearest], int(nearest)


def metric_vec(info):
    return np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)


def reset_state(env, seed):
    obs, _ = env.reset(seed=seed)
    inner = getattr(env.unwrapped, "env", env.unwrapped)
    data = getattr(inner, "data", None)
    qpos = np.asarray(data.qpos).copy() if hasattr(data, "qpos") else None
    qvel = np.asarray(data.qvel).copy() if hasattr(data, "qvel") else None
    target = None
    return obs, qpos, qvel, target


def step1_probe(agent, arrays, errors, regions, seed, mode):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs0, qpos0, qvel0, target0 = reset_state(env, seed)
        policy_action = np.asarray(jax.device_get(agent.sample_actions(observations=obs0, argmax=True)), np.float64)
        demo_action, _ = nearest_demo_action(errors, arrays["actions"], regions, "LATERAL_CORRECTION", np.asarray([0.16, 0.035, 0.03]))
        action = demo_action if mode == "demo" else policy_action
        obs1, _reward, _terminated, _truncated, info = env.step(action)
        inner = getattr(env.unwrapped, "env", env.unwrapped)
        data = getattr(inner, "data", None)
        qpos1 = np.asarray(data.qpos).copy() if qpos0 is not None else None
        qvel1 = np.asarray(data.qvel).copy() if qvel0 is not None else None
        target1 = None
        link, unexpected, pairs = contact_flags(env)
        out = {
            "a0": float(action[0]),
            "policy_a0": float(policy_action[0]),
            "delta_lateral": float(info["lateral_error"]),
            "delta_axial": float(info["axial_error"]),
            "delta_orientation": float(info["orientation_error"]),
            "ee_displacement": float(np.linalg.norm(np.asarray(obs1[:3]) - np.asarray(obs0[:3]))),
            "joint_displacement": None if qpos0 is None or qpos1 is None else float(np.linalg.norm(qpos1 - qpos0)),
            "qvel_norm": None if qvel1 is None else float(np.linalg.norm(qvel1)),
            "controller_target_change": None if target0 is None or target1 is None else float(np.linalg.norm(target1 - target0)),
            "contact": bool(link or unexpected),
            "unexpected_contact": bool(unexpected),
            "contact_pairs": pairs,
            "force": float(info.get("force_norm", np.nan)),
        }
        env.close()
        return out


def rollout_window(agent, arrays, errors, regions, support, seed, horizon):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=seed)
        rows = []
        contact_step = None
        for step in range(1, horizon + 1):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            obs, _reward, terminated, truncated, info = env.step(action)
            err = metric_vec(info)
            region = region_from_errors(float(err[0]), float(err[1]), float(err[2]))
            envl = support[region]
            link, unexpected, _pairs = contact_flags(env)
            if contact_step is None and (link or unexpected):
                contact_step = step
            clipped = np.minimum(np.maximum(err, envl["error_min"]), envl["error_max"])
            rows.append({
                "joint": bool(np.all(obs[:7] >= envl["obs_min"]) and np.all(obs[:7] <= envl["obs_max"])),
                "lateral": bool(envl["error_min"][0] <= err[0] <= envl["error_max"][0]),
                "axial": bool(envl["error_min"][1] <= err[1] <= envl["error_max"][1]),
                "orientation": bool(envl["error_min"][2] <= err[2] <= envl["error_max"][2]),
                "corridor_distance": float(np.linalg.norm(err - clipped)),
            })
            if terminated or truncated:
                break
        env.close()
    return {
        "horizon": horizon,
        "joint_overlap": float(np.mean([r["joint"] for r in rows])),
        "lateral_overlap": float(np.mean([r["lateral"] for r in rows])),
        "axial_overlap": float(np.mean([r["axial"] for r in rows])),
        "orientation_overlap": float(np.mean([r["orientation"] for r in rows])),
        "corridor_error": float(np.mean([r["corridor_distance"] for r in rows])),
        "first_contact": contact_step,
    }


def summarize_step1(rows):
    keys = ["delta_lateral", "delta_axial", "delta_orientation", "ee_displacement", "joint_displacement", "controller_target_change", "force"]
    out = {}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = None if not values else float(np.nanmedian(values))
    return out


def summarize_windows(rows):
    out = {}
    for horizon in WINDOWS:
        group = [r for r in rows if r["horizon"] == horizon]
        out[str(horizon)] = {
            "joint_overlap": float(np.mean([r["joint_overlap"] for r in group])),
            "lateral_overlap": float(np.mean([r["lateral_overlap"] for r in group])),
            "axial_overlap": float(np.mean([r["axial_overlap"] for r in group])),
            "orientation_overlap": float(np.mean([r["orientation_overlap"] for r in group])),
            "corridor_error": float(np.mean([r["corridor_error"] for r in group])),
            "first_contact": None if all(r["first_contact"] is None for r in group) else float(np.nanmin([r["first_contact"] if r["first_contact"] is not None else np.nan for r in group])),
        }
    return out


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg_cache, demo_cache = {}, {}
    detail = {}
    for name, (cfg_path, run, factory) in RUNS.items():
        cfg = cfg_cache.setdefault(str(cfg_path), yaml.safe_load(cfg_path.read_text()))
        if str(cfg_path) not in demo_cache:
            arrays, errors, regions, phases = load_demo(cfg)
            demo_cache[str(cfg_path)] = (arrays, errors, regions, phases, build_support(errors, arrays["observations"], regions))
        arrays, errors, regions, phases, support = demo_cache[str(cfg_path)]
        agent = load_agent(cfg, run, factory)
        step_rows = [step1_probe(agent, arrays, errors, regions, seed, "policy") for seed in SEEDS]
        win_rows = [rollout_window(agent, arrays, errors, regions, support, seed, horizon) for seed in SEEDS for horizon in WINDOWS]
        detail[name] = {
            "step1": step_rows,
            "step1_summary": summarize_step1(step_rows),
            "windows": summarize_windows(win_rows),
        }
    cfg = yaml.safe_load(CANDIDATE_CONFIG.read_text())
    arrays, errors, regions, phases = load_demo(cfg)
    support_actions = arrays["actions"][np.isin(phases, cfg["a0_action_support"]["predicate"]["phases"]), 0]
    demo_step = [step1_probe(load_agent(yaml.safe_load(CANDIDATE_CONFIG.read_text()), RUNS["candidate_R1"][1], RUNS["candidate_R1"][2]), arrays, errors, regions, seed, "demo") for seed in SEEDS]
    baseline_dev = np.median([detail["baseline_R1"]["step1_summary"]["ee_displacement"], detail["baseline_R2"]["step1_summary"]["ee_displacement"]])
    candidate_dev = np.median([detail["candidate_R1"]["step1_summary"]["ee_displacement"], detail["candidate_R2"]["step1_summary"]["ee_displacement"]])
    reduction = float((baseline_dev - candidate_dev) / max(abs(baseline_dev), 1e-12))
    def avg_window(prefix, metric):
        vals = []
        for name in detail:
            if name.startswith(prefix):
                vals.append(detail[name]["windows"]["50"][metric])
        return float(np.mean(vals))
    overlap_improvement = {
        "joint_50": avg_window("candidate", "joint_overlap") - avg_window("baseline", "joint_overlap"),
        "lateral_50": avg_window("candidate", "lateral_overlap") - avg_window("baseline", "lateral_overlap"),
        "axial_50": avg_window("candidate", "axial_overlap") - avg_window("baseline", "axial_overlap"),
        "orientation_50": avg_window("candidate", "orientation_overlap") - avg_window("baseline", "orientation_overlap"),
        "corridor_error_50_delta": avg_window("candidate", "corridor_error") - avg_window("baseline", "corridor_error"),
    }
    exact_zero_is_empirical = float(np.min(support_actions)) == 0.0 and float(np.max(support_actions)) == 0.0
    candidate_improves = reduction > 0.20 and overlap_improvement["axial_50"] > 0 and overlap_improvement["corridor_error_50_delta"] <= 0.0
    candidate_harmful = overlap_improvement["joint_50"] <= 0 and overlap_improvement["lateral_50"] <= 0 and overlap_improvement["corridor_error_50_delta"] > 0.0
    if candidate_improves:
        validity = "TOO_STRICT_FOR_BEHAVIOR"
        root = "GATE_OVERSTRICT_EMPIRICAL_ZERO_SUPPORT"
        next_step = "BEHAVIORAL_GATE_REDESIGN_REVIEW"
    elif candidate_harmful:
        validity = "HARD_VALID"
        root = "RESIDUAL_A0_CAUSAL_DRIFT"
        next_step = "BLOCKED_NO_3K"
    else:
        validity = "UNRESOLVED"
        root = "BLOCKED"
        next_step = "BLOCKED"
    result = {
        "STEP1_GATE_VALIDITY_AUDIT": "PASS",
        "STEP1_STATE_DEVIATION_BASELINE": {
            "R1": detail["baseline_R1"]["step1_summary"],
            "R2": detail["baseline_R2"]["step1_summary"],
        },
        "STEP1_STATE_DEVIATION_CANDIDATE": {
            "R1": detail["candidate_R1"]["step1_summary"],
            "R2": detail["candidate_R2"]["step1_summary"],
        },
        "STEP1_STATE_DEVIATION_DEMO": summarize_step1(demo_step),
        "PHYSICAL_DEVIATION_REDUCTION": reduction,
        "SHORT_HORIZON_OVERLAP_IMPROVEMENT": overlap_improvement,
        "SHORT_HORIZON_CORRIDOR_ERROR": {
            "baseline_50": avg_window("baseline", "corridor_error"),
            "candidate_50": avg_window("candidate", "corridor_error"),
        },
        "A0_ZERO_SUPPORT_TYPE": "EMPIRICAL_DEMO_SUPPORT" if exact_zero_is_empirical else "UNRESOLVED",
        "GATE_VALIDITY": validity,
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT": next_step,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": detail,
    }
    result = sanitize(result)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
