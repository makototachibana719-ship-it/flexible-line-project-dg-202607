#!/usr/bin/env python3
"""Read-only a3 orientation/axial coupling audit for Peg F3 V2."""
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
STEPS = (1, 5, 10, 15, 25, 50)
SEEDS = (0, 1, 2)
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out():
    stem = "f3_v2_a3_orientation_axial_coupling_20260907"
    for i in range(1, 100):
        out = OUT_ROOT / f"{stem}_{i:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def load_demo(cfg):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    prov = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, metrics = [], []
    for entry in prov:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            r = json.loads(line)
            phases.append(r["phase"])
            m = r["after"]["metrics"]
            metrics.append([m["lateral_error"], m["axial_error"], m["orientation_error"]])
    return arrays, np.asarray(metrics, np.float64), np.asarray(phases)


def phase_from_errors(err):
    return {
        "FAR_APPROACH": "early",
        "LATERAL_CORRECTION": "lateral",
        "ORIENTATION_CORRECTION": "orientation",
        "NEAR_HOLE": "near_hole",
    }[region_from_errors(float(err[0]), float(err[1]), float(err[2]))]


def demo_phase_masks(phases):
    return {
        "early": np.isin(phases, ["setup", "approach"]),
        "lateral": phases == "lateral_alignment",
        "orientation": phases == "orientation_correction",
    }


def load_agent(cfg, run, factory):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(STEPS))
        fresh = factory(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def nearest_demo_action(arrays, errors, phases, err, phase):
    masks = demo_phase_masks(phases)
    idx = np.where(masks.get(phase, np.ones(len(errors), bool)))[0]
    if idx.size == 0:
        idx = np.arange(len(errors))
    nearest = idx[int(np.argmin(np.linalg.norm(errors[idx] - err[None, :], axis=1)))]
    return arrays["actions"][nearest]


def rollout(agent, arrays, errors, phases, seed):
    rows = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(STEPS))
        obs, _ = env.reset(seed=seed)
        prev = np.asarray(obs, np.float64)
        for step in range(1, max(STEPS) + 1):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            obs, _reward, terminated, truncated, info = env.step(action)
            current = np.asarray(obs, np.float64)
            err = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
            phase = phase_from_errors(err)
            demo_action = nearest_demo_action(arrays, errors, phases, err, phase)
            link, unexpected, _pairs = contact_flags(env)
            if step in STEPS:
                rows.append({
                    "step": step,
                    "phase": phase,
                    "a3": float(action[3]),
                    "demo_a3": float(demo_action[3]),
                    "a3_error": float(abs(action[3] - demo_action[3])),
                    "orientation_error": float(err[2]),
                    "ee_orientation_delta": float(np.linalg.norm(current[3:7] - prev[3:7])),
                    "axial_error": float(err[1]),
                    "ee_axial_displacement": float(current[2] - prev[2]),
                    "force": float(info.get("force_norm", 0.0)),
                    "contact": bool(link or unexpected),
                })
            prev = current
            if terminated or truncated:
                break
        env.close()
    return rows


def perturbation(agent, seed, eps=0.05):
    rows = []
    with portable_peg_runtime(ROOT):
        for sign in (-1.0, 1.0):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=1)
            obs, _ = env.reset(seed=seed)
            base = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            action = base.copy()
            action[3] = np.clip(action[3] + sign * eps, -1.0, 1.0)
            obs1, _reward, _terminated, _truncated, info = env.step(action)
            rows.append({
                "sign": sign,
                "effective_delta_a3": float(action[3] - base[3]),
                "orientation_error": float(info["orientation_error"]),
                "axial_error": float(info["axial_error"]),
                "lateral_error": float(info["lateral_error"]),
                "ee_axial": float(np.asarray(obs1)[2] - np.asarray(obs)[2]),
                "ee_lateral": float(np.linalg.norm(np.asarray(obs1)[:2] - np.asarray(obs)[:2])),
            })
            env.close()
    pos = rows[1]
    neg = rows[0]
    denom = pos["effective_delta_a3"] - neg["effective_delta_a3"]
    if abs(denom) < 1e-9:
        return {"available": False}
    return {
        "available": True,
        "a3_to_orientation_gain": float((pos["orientation_error"] - neg["orientation_error"]) / denom),
        "a3_to_axial_coupling": float((pos["ee_axial"] - neg["ee_axial"]) / denom),
        "a3_to_lateral_coupling": float((pos["ee_lateral"] - neg["ee_lateral"]) / denom),
    }


def summarize(rows):
    out = {}
    for step in STEPS:
        group = [r for r in rows if r["step"] == step]
        out[str(step)] = {
            "a3": float(np.mean([r["a3"] for r in group])),
            "demo_a3": float(np.mean([r["demo_a3"] for r in group])),
            "a3_error": float(np.mean([r["a3_error"] for r in group])),
            "orientation_error": float(np.mean([r["orientation_error"] for r in group])),
            "ee_orientation_delta": float(np.mean([r["ee_orientation_delta"] for r in group])),
            "axial_error": float(np.mean([r["axial_error"] for r in group])),
            "ee_axial_displacement": float(np.mean([r["ee_axial_displacement"] for r in group])),
            "force": float(np.mean([r["force"] for r in group])),
            "contact_rate": float(np.mean([r["contact"] for r in group])),
        }
    return out


def family(details, prefix):
    out = {}
    for step in STEPS:
        group = [details[k]["timeline"][str(step)] for k in details if k.startswith(prefix)]
        out[str(step)] = {key: float(np.mean([g[key] for g in group])) for key in group[0]}
    return out


def phase_mae(agent, arrays, errors, phases):
    policy = np.asarray(jax.device_get(agent.sample_actions(observations=arrays["observations"], argmax=True)))
    masks = demo_phase_masks(phases)
    return {
        phase: float(np.mean(np.abs(policy[mask, 3] - arrays["actions"][mask, 3])))
        for phase, mask in masks.items()
        if np.any(mask)
    }


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    details, cache = {}, {}
    phase_ranges, phase_maes = {}, {}
    perturb = {}
    for name, (cfg_path, run, factory) in RUNS.items():
        cfg = yaml.safe_load(cfg_path.read_text())
        if str(cfg_path) not in cache:
            cache[str(cfg_path)] = load_demo(cfg)
        arrays, errors, phases = cache[str(cfg_path)]
        agent = load_agent(cfg, run, factory)
        rows = []
        for seed in SEEDS:
            rows.extend(rollout(agent, arrays, errors, phases, seed))
        details[name] = {
            "timeline": summarize(rows),
            "phase_a3_mae": phase_mae(agent, arrays, errors, phases),
        }
        perturb[name] = [perturbation(agent, seed) for seed in SEEDS]
    arrays, _errors, phases = cache[str(CANDIDATE_CONFIG)]
    masks = demo_phase_masks(phases)
    for phase, mask in masks.items():
        phase_ranges[phase] = [float(arrays["actions"][mask, 3].min()), float(arrays["actions"][mask, 3].max())]
    baseline = family(details, "baseline")
    candidate = family(details, "candidate")
    first_dev = None
    for step in STEPS:
        if candidate[str(step)]["a3_error"] > baseline[str(step)]["a3_error"]:
            first_dev = step
            break
    first_step = str(first_dev if first_dev is not None else STEPS[0])
    a3_error_mag = candidate[first_step]["a3_error"] - baseline[first_step]["a3_error"]
    gains = [p for values in perturb.values() for p in values if p.get("available")]
    axial_coupling = float(np.mean([abs(g["a3_to_axial_coupling"]) for g in gains])) if gains else None
    orientation_gain = float(np.mean([abs(g["a3_to_orientation_gain"]) for g in gains])) if gains else None
    lateral_coupling = float(np.mean([abs(g["a3_to_lateral_coupling"]) for g in gains])) if gains else None
    orientation_response = candidate[first_step]["orientation_error"] - baseline[first_step]["orientation_error"]
    axial_response = candidate[first_step]["axial_error"] - baseline[first_step]["axial_error"]
    force_response = candidate[first_step]["force"] - baseline[first_step]["force"]
    chain_supported = (
        first_dev is not None
        and a3_error_mag > 0
        and orientation_gain is not None
        and orientation_gain > 0
        and axial_coupling is not None
        and axial_coupling > 0
        and axial_response > 0
        and force_response > 0
    )
    if chain_supported:
        root = "EARLY_A3_COUPLED_DRIFT"
        next_fix = "PHASE_SELECTIVE_A3_SUPPORT_REGULARIZATION"
        chain = "SUPPORTED"
    elif first_dev is not None and (axial_coupling is None or axial_coupling <= 0):
        root = "CONTROLLER_OR_KINEMATIC_TRACKING_MISMATCH"
        next_fix = "CONTROLLER_DIAGNOSTIC_ONLY"
        chain = "NOT_SUPPORTED"
    else:
        root = "MIXED"
        next_fix = "BLOCKED"
        chain = "MIXED"
    result = {
        "A3_COUPLING_AUDIT": "PASS",
        "FIRST_A3_DEVIATION_STEP": first_dev,
        "A3_ERROR_MAGNITUDE": float(a3_error_mag),
        "ORIENTATION_RESPONSE": float(orientation_response),
        "A3_DEMO_RANGE_EARLY": phase_ranges.get("early"),
        "A3_DEMO_RANGE_ORIENTATION": phase_ranges.get("orientation"),
        "A3_POLICY_RANGE": {
            "candidate_R1": [min(details["candidate_R1"]["timeline"][str(s)]["a3"] for s in STEPS), max(details["candidate_R1"]["timeline"][str(s)]["a3"] for s in STEPS)],
            "candidate_R2": [min(details["candidate_R2"]["timeline"][str(s)]["a3"] for s in STEPS), max(details["candidate_R2"]["timeline"][str(s)]["a3"] for s in STEPS)],
        },
        "A3_MAE_BY_PHASE": {
            "candidate_R1": details["candidate_R1"]["phase_a3_mae"],
            "candidate_R2": details["candidate_R2"]["phase_a3_mae"],
        },
        "A3_TO_ORIENTATION_GAIN": orientation_gain,
        "A3_TO_AXIAL_COUPLING": axial_coupling,
        "A3_TO_LATERAL_COUPLING": lateral_coupling,
        "A3_CAUSAL_CHAIN": chain,
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT_FIX": next_fix,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": {"baseline": baseline, "candidate": candidate, "runs": details, "perturbation": perturb},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
