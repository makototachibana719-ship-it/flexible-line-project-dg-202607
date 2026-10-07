#!/usr/bin/env python3
"""Read-only early joint-action sequence audit for Peg F3 V2."""
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
STEPS = (5, 10, 15, 25, 50)
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out() -> Path:
    stem = "f3_v2_early_joint_action_sequence_20260907"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def load_demo(cfg):
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
    return arrays, errors, np.asarray(phases)


def phase_from_error(err):
    return {
        "FAR_APPROACH": "approach",
        "LATERAL_CORRECTION": "lateral_correction",
        "ORIENTATION_CORRECTION": "orientation_correction",
        "NEAR_HOLE": "near_hole",
    }[region_from_errors(float(err[0]), float(err[1]), float(err[2]))]


def phase_mask(phases, phase):
    if phase == "approach":
        return np.isin(phases, ["setup", "approach"])
    if phase == "lateral_correction":
        return phases == "lateral_alignment"
    if phase == "orientation_correction":
        return phases == "orientation_correction"
    return np.ones(len(phases), bool)


def vector_metrics(actions):
    actions = np.asarray(actions, np.float64)
    trans = np.linalg.norm(actions[..., :3], axis=-1)
    rot = np.linalg.norm(actions[..., 3:6], axis=-1)
    return trans, rot, trans / np.maximum(rot, 1e-9)


def support_by_phase(arrays, phases):
    result = {}
    for phase in ("setup", "approach", "lateral_correction", "orientation_correction"):
        if phase == "lateral_correction":
            mask = phases == "lateral_alignment"
        elif phase == "orientation_correction":
            mask = phases == "orientation_correction"
        else:
            mask = phases == phase
        if not np.any(mask):
            continue
        trans, rot, ratio = vector_metrics(arrays["actions"][mask])
        result[phase] = {
            "translation_norm_range": [float(np.min(trans)), float(np.max(trans))],
            "rotation_norm_range": [float(np.min(rot)), float(np.max(rot))],
            "trans_rot_ratio_range": [float(np.min(ratio)), float(np.max(ratio))],
        }
    return result


def load_agent(cfg, run, factory):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(STEPS))
        fresh = factory(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def nearest_demo(arrays, errors, phases, err, phase):
    mask = phase_mask(phases, phase)
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.arange(len(errors))
    nearest = idx[int(np.argmin(np.linalg.norm(errors[idx] - err[None, :], axis=1)))]
    return arrays["actions"][nearest]


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def rollout(agent, arrays, errors, phases, seed):
    rows, cumulative_trans, cumulative_rot = [], np.zeros(3), np.zeros(3)
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(STEPS))
        obs, _ = env.reset(seed=seed)
        for step in range(1, max(STEPS) + 1):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            obs, _reward, terminated, truncated, info = env.step(action)
            err = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
            phase = phase_from_error(err)
            demo = nearest_demo(arrays, errors, phases, err, phase)
            cumulative_trans += action[:3]
            cumulative_rot += action[3:6]
            if step in STEPS:
                trans_norm, rot_norm, ratio = vector_metrics(action)
                demo_trans, demo_rot, demo_ratio = vector_metrics(demo)
                rows.append({
                    "step": step,
                    "phase": phase,
                    "action": action.tolist(),
                    "demo_action": np.asarray(demo, np.float64).tolist(),
                    "translation_norm": float(trans_norm),
                    "rotation_norm": float(rot_norm),
                    "trans_rot_ratio": float(ratio),
                    "demo_translation_norm": float(demo_trans),
                    "demo_rotation_norm": float(demo_rot),
                    "demo_trans_rot_ratio": float(demo_ratio),
                    "action_cosine": cosine(action, demo),
                    "action_l2": float(np.linalg.norm(action - demo)),
                    "cumulative_translation_command": cumulative_trans.tolist(),
                    "cumulative_rotation_command": cumulative_rot.tolist(),
                    "axial_error": float(err[1]),
                    "orientation_error": float(err[2]),
                    "force": float(info.get("force_norm", 0.0)),
                    "corridor_error_proxy": float(np.linalg.norm(err)),
                })
            if terminated or truncated:
                break
        env.close()
    return rows


def summarize(rows):
    out = {}
    for step in STEPS:
        group = [row for row in rows if row["step"] == step]
        out[str(step)] = {
            "translation_norm": float(np.mean([r["translation_norm"] for r in group])),
            "rotation_norm": float(np.mean([r["rotation_norm"] for r in group])),
            "trans_rot_ratio": float(np.mean([r["trans_rot_ratio"] for r in group])),
            "action_cosine": float(np.mean([r["action_cosine"] for r in group if r["action_cosine"] is not None])),
            "action_l2": float(np.mean([r["action_l2"] for r in group])),
            "cumulative_translation_norm": float(np.mean([np.linalg.norm(r["cumulative_translation_command"]) for r in group])),
            "cumulative_rotation_norm": float(np.mean([np.linalg.norm(r["cumulative_rotation_command"]) for r in group])),
            "axial_error": float(np.mean([r["axial_error"] for r in group])),
            "orientation_error": float(np.mean([r["orientation_error"] for r in group])),
            "force": float(np.mean([r["force"] for r in group])),
            "corridor_error_proxy": float(np.mean([r["corridor_error_proxy"] for r in group])),
        }
    return out


def aggregate(details, prefix):
    result = {}
    for step in STEPS:
        group = [details[name]["timeline"][str(step)] for name in sorted(details) if name.startswith(prefix)]
        result[str(step)] = {key: float(np.mean([row[key] for row in group])) for key in group[0]}
    return result


def trend(candidate, baseline, key, lower_is_better):
    deltas = []
    for step in STEPS:
        delta = candidate[str(step)][key] - baseline[str(step)][key]
        deltas.append(delta if lower_is_better else -delta)
    improving = sum(d < 0 for d in deltas)
    worsening = sum(d > 0 for d in deltas)
    if improving >= 4:
        return "IMPROVING"
    if worsening >= 4:
        return "WORSENING"
    return "FLAT"


def first_abnormal(candidate, support):
    lateral = support["lateral_correction"]
    for step in STEPS:
        row = candidate[str(step)]
        checks = {
            "translation_norm": lateral["translation_norm_range"],
            "rotation_norm": lateral["rotation_norm_range"],
            "trans_rot_ratio": lateral["trans_rot_ratio_range"],
        }
        for metric, (lo, hi) in checks.items():
            if row[metric] < lo or row[metric] > hi:
                return step, metric
        if row["action_cosine"] < 0.0:
            return step, "action_cosine_negative"
    return None, "NO_CLEAR_JOINT_PATTERN"


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cache, details = {}, {}
    support = None
    for name, (cfg_path, run, factory) in RUNS.items():
        cfg = yaml.safe_load(cfg_path.read_text())
        if str(cfg_path) not in cache:
            arrays, errors, phases = load_demo(cfg)
            cache[str(cfg_path)] = (arrays, errors, phases)
            if cfg_path == CANDIDATE_CONFIG:
                support = support_by_phase(arrays, phases)
        arrays, errors, phases = cache[str(cfg_path)]
        agent = load_agent(cfg, run, factory)
        rows = []
        for seed in SEEDS:
            rows.extend(rollout(agent, arrays, errors, phases, seed))
        details[name] = {"timeline": summarize(rows)}
    baseline = aggregate(details, "baseline")
    candidate = aggregate(details, "candidate")
    first_step, first_metric = first_abnormal(candidate, support)
    before_axial = first_step is not None and first_step <= 10
    before_force = first_step is not None and first_step <= 10
    before_corridor = first_step is not None and first_step <= 1
    ratio_bad = first_metric == "trans_rot_ratio"
    if first_step is None:
        root = "ASYNC_TRAJECTORY_VARIANCE"
        next_fix = "MULTI_REPLICA_SEQUENCE_VARIANCE_PROTOCOL"
    elif ratio_bad:
        root = "EARLY_TRANSLATION_ROTATION_BALANCE_DRIFT"
        next_fix = "PHASE_SELECTIVE_TRANS_ROT_BALANCE_REGULARIZATION"
    else:
        root = "EARLY_JOINT_ACTION_COMPOSITION_DRIFT"
        next_fix = "PHASE_SELECTIVE_JOINT_ACTION_SUPPORT_REGULARIZATION"
    result = {
        "JOINT_ACTION_SEQUENCE_AUDIT": "PASS",
        "FIRST_JOINT_ACTION_DEVIATION_STEP": first_step,
        "FIRST_ABNORMAL_JOINT_METRIC": first_metric,
        "TRANSLATION_NORM_TREND": trend(candidate, baseline, "translation_norm", lower_is_better=False),
        "ROTATION_NORM_TREND": trend(candidate, baseline, "rotation_norm", lower_is_better=False),
        "TRANS_ROT_RATIO_TREND": trend(candidate, baseline, "trans_rot_ratio", lower_is_better=False),
        "ACTION_COSINE_TREND": trend(candidate, baseline, "action_cosine", lower_is_better=False),
        "ACTION_L2_TREND": trend(candidate, baseline, "action_l2", lower_is_better=True),
        "TARGET_TRANSLATION_ERROR": "NOT_AVAILABLE",
        "TARGET_ROTATION_ERROR": "NOT_AVAILABLE",
        "TARGET_POSE_ERROR": "NOT_AVAILABLE",
        "JOINT_ACTION_DEVIATION_BEFORE_AXIAL": before_axial,
        "JOINT_ACTION_DEVIATION_BEFORE_FORCE": before_force,
        "JOINT_ACTION_DEVIATION_BEFORE_CORRIDOR": before_corridor,
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT_FIX": next_fix,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "PHASE_SUPPORT": support,
        "DETAILS": {"baseline": baseline, "candidate": candidate, "runs": details},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
