#!/usr/bin/env python3
"""Read-only axial/force causal audit for Peg F3 V2 after corridor improvement."""
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
HORIZONS = (1, 5, 10, 25, 50, 100, 150)
OUT_ROOT = ROOT / "results/diagnostics/peg"
ACTION_DIMS = tuple(f"a{i}" for i in range(7))


def unique_out() -> Path:
    stem = "f3_v2_axial_force_causal_audit_20260907"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def load_demo(cfg):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    metrics = []
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            m = json.loads(line)["after"]["metrics"]
            metrics.append([m["lateral_error"], m["axial_error"], m["orientation_error"]])
    errors = np.asarray(metrics, np.float64)
    regions = np.asarray([region_from_errors(*row) for row in errors])
    return arrays, errors, regions


def load_agent(cfg, run, factory):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(HORIZONS))
        fresh = factory(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def nearest_demo_action(arrays, errors, regions, err):
    region = region_from_errors(float(err[0]), float(err[1]), float(err[2]))
    idx = np.where(regions == region)[0]
    if idx.size == 0:
        idx = np.arange(len(errors))
    nearest = idx[int(np.argmin(np.linalg.norm(errors[idx] - err[None, :], axis=1)))]
    return arrays["actions"][nearest]


def rollout(agent, arrays, errors, regions, seed):
    rows = []
    first_contact = None
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(HORIZONS))
        obs, _ = env.reset(seed=seed)
        prev_axial = None
        prev_ee = np.asarray(obs[:3], np.float64)
        for step in range(1, max(HORIZONS) + 1):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            obs, _reward, terminated, truncated, info = env.step(action)
            err = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
            demo_action = nearest_demo_action(arrays, errors, regions, err)
            link, unexpected, pairs = contact_flags(env)
            contact = bool(link or unexpected)
            if first_contact is None and contact:
                first_contact = step
            ee = np.asarray(obs[:3], np.float64)
            axial = float(info["axial_error"])
            rows.append({
                "step": step,
                "axial_error": axial,
                "delta_axial": 0.0 if prev_axial is None else axial - prev_axial,
                "a2": float(action[2]),
                "action": action.tolist(),
                "demo_action": np.asarray(demo_action, np.float64).tolist(),
                "action_error": np.abs(action - demo_action).tolist(),
                "target_axial_displacement": None,
                "ee_axial_displacement": float(ee[2] - prev_ee[2]),
                "force": float(info.get("force_norm", 0.0)),
                "contact": contact,
                "unexpected_contact": bool(unexpected),
                "contact_pairs": pairs,
            })
            prev_axial = axial
            prev_ee = ee
            if terminated or truncated:
                break
        env.close()
    return rows, first_contact


def summarize_run(rollouts):
    result = {}
    for h in HORIZONS:
        samples = [rows[h - 1] for rows, _contact in rollouts if len(rows) >= h]
        result[str(h)] = {
            "axial_error": float(np.mean([s["axial_error"] for s in samples])),
            "delta_axial": float(np.mean([s["delta_axial"] for s in samples])),
            "a2": float(np.mean([s["a2"] for s in samples])),
            "ee_axial_displacement": float(np.mean([s["ee_axial_displacement"] for s in samples])),
            "force": float(np.mean([s["force"] for s in samples])),
            "contact_rate": float(np.mean([s["contact"] for s in samples])),
            "action_error": np.mean([s["action_error"] for s in samples], axis=0).tolist(),
        }
    contacts = [c for _rows, c in rollouts if c is not None]
    result["first_contact"] = None if not contacts else float(np.median(contacts))
    return result


def aggregate(details, prefix):
    out = {}
    for h in HORIZONS:
        group = [details[k][str(h)] for k in sorted(details) if k.startswith(prefix)]
        out[str(h)] = {
            "axial_error": float(np.mean([g["axial_error"] for g in group])),
            "delta_axial": float(np.mean([g["delta_axial"] for g in group])),
            "a2": float(np.mean([g["a2"] for g in group])),
            "ee_axial_displacement": float(np.mean([g["ee_axial_displacement"] for g in group])),
            "force": float(np.mean([g["force"] for g in group])),
            "contact_rate": float(np.mean([g["contact_rate"] for g in group])),
            "action_error": np.mean([g["action_error"] for g in group], axis=0).tolist(),
        }
    contacts = [details[k]["first_contact"] for k in details if k.startswith(prefix) and details[k]["first_contact"] is not None]
    out["first_contact"] = None if not contacts else float(np.median(contacts))
    return out


def first_worsen(candidate, baseline, metric):
    for h in HORIZONS:
        if candidate[str(h)][metric] > baseline[str(h)][metric]:
            return h
    return None


def dominant_action_dim(candidate, baseline, first_step):
    if first_step is None:
        return None, {}
    c = np.asarray(candidate[str(first_step)]["action_error"], np.float64)
    b = np.asarray(baseline[str(first_step)]["action_error"], np.float64)
    diff = c - b
    dim = int(np.argmax(diff))
    return f"a{dim}", {ACTION_DIMS[i]: float(diff[i]) for i in range(7)}


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cache = {}
    details = {}
    for name, (cfg_path, run, factory) in RUNS.items():
        cfg = yaml.safe_load(cfg_path.read_text())
        if str(cfg_path) not in cache:
            cache[str(cfg_path)] = load_demo(cfg)
        arrays, errors, regions = cache[str(cfg_path)]
        agent = load_agent(cfg, run, factory)
        details[name] = summarize_run([rollout(agent, arrays, errors, regions, seed) for seed in SEEDS])
    baseline = aggregate(details, "baseline")
    candidate = aggregate(details, "candidate")
    axial_step = first_worsen(candidate, baseline, "axial_error")
    force_step = first_worsen(candidate, baseline, "force")
    dominant_dim, action_error = dominant_action_dim(candidate, baseline, axial_step)
    force_before_contact = force_step is not None and (
        candidate["first_contact"] is None or force_step < candidate["first_contact"]
    )
    axial_before_force = axial_step is not None and force_step is not None and axial_step <= force_step
    a2_policy_drives = dominant_dim == "a2" or (
        axial_step is not None and abs(candidate[str(axial_step)]["a2"]) > abs(baseline[str(axial_step)]["a2"])
    )
    if a2_policy_drives and axial_before_force:
        root = "EARLY_AXIAL_POLICY_DRIFT"
        next_fix = "AXIAL_ACTION_SUPPORT_REGULARIZATION"
    elif force_step is not None and not axial_before_force:
        root = "CONTACT_FORCE_MECHANISM"
        next_fix = "FORCE_CONTACT_CAUSAL_AUDIT"
    else:
        root = "MIXED_AXIAL_FORCE_EARLY_DRIFT"
        next_fix = "AXIAL_ACTION_SUPPORT_REGULARIZATION_REVIEW"
    result = {
        "AXIAL_FORCE_CAUSAL_AUDIT": "PASS",
        "FIRST_AXIAL_WORSEN_STEP": axial_step,
        "FIRST_AXIAL_WORSEN_ACTION_DIM": dominant_dim,
        "DOMINANT_AXIAL_ACTION_DIM": dominant_dim,
        "ACTION_DIM_ERROR": action_error,
        "FIRST_FORCE_WORSEN_STEP": force_step,
        "FORCE_BEFORE_CONTACT": force_before_contact,
        "AXIAL_DIVERGENCE_BEFORE_FORCE": axial_before_force,
        "POLICY_COMMAND_CAUSES_AXIAL_DRIFT": bool(a2_policy_drives and axial_step is not None),
        "CONTROLLER_TRACKING_FAILURE": "UNAVAILABLE",
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT_FIX": next_fix,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "TIMELINE": {"baseline": baseline, "candidate": candidate},
        "DETAILS": details,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
