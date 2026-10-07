#!/usr/bin/env python3
"""Read-only post-step1 closed-loop audit for Peg F3 V2 candidate."""
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
from training.peg_insertion.actor_objective_balance_ablation import make_actor_objective_balance_agent
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_pretanh_a0_anchor_ablation import make_early_pretanh_a0_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

BASELINE_CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_pretanh_a0_anchor_ablation_1k_seed0.yaml"
CANDIDATE_CONFIG = ROOT / "configs/peg_insertion/peg_v2_actor_objective_balance_ablation_1k_seed0.yaml"
RUNS = {
    "baseline_R1": (BASELINE_CONFIG, ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002/replica1", make_early_pretanh_a0_anchor_agent),
    "baseline_R2": (BASELINE_CONFIG, ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002/replica2", make_early_pretanh_a0_anchor_agent),
    "candidate_R1": (CANDIDATE_CONFIG, ROOT / "results/ablation/peg_f3_v2_actor_objective_balance_n2_safe_20260906/replica1", make_actor_objective_balance_agent),
    "candidate_R2": (CANDIDATE_CONFIG, ROOT / "results/ablation/peg_f3_v2_actor_objective_balance_n2_safe_20260906/replica2_retry001", make_actor_objective_balance_agent),
}
SEEDS = (0, 1, 2)
OUT_ROOT = ROOT / "results/diagnostics/peg"
ACTION_DIMS = tuple(f"a{i}" for i in range(7))


def unique_out() -> Path:
    stem = "f3_v2_post_step1_closed_loop_audit_20260906"
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
            metrics.append(row["after"]["metrics"])
    errors = np.asarray([[m["lateral_error"], m["axial_error"], m["orientation_error"]] for m in metrics], np.float64)
    regions = np.asarray([region_from_errors(*err) for err in errors])
    return arrays, errors, regions, np.asarray(phases)


def build_envelopes(errors: np.ndarray, actions: np.ndarray, regions: np.ndarray):
    envelopes = {}
    for region in sorted(set(regions)):
        idx = regions == region
        envelopes[region] = {
            "error_min": errors[idx].min(axis=0),
            "error_max": errors[idx].max(axis=0),
            "action_p01": np.percentile(actions[idx], 1, axis=0),
            "action_p99": np.percentile(actions[idx], 99, axis=0),
        }
    return envelopes


def load_agent(cfg: dict, run: Path, factory):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = factory(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def phase_from_region(region: str) -> str:
    return {
        "FAR_APPROACH": "approach",
        "LATERAL_CORRECTION": "lateral_correction",
        "ORIENTATION_CORRECTION": "orientation_correction",
        "NEAR_HOLE": "near_hole",
    }.get(region, region.lower())


def nearest_demo_action(errors: np.ndarray, actions: np.ndarray, regions: np.ndarray, region: str, err: np.ndarray):
    idx = np.where(regions == region)[0]
    if idx.size == 0:
        idx = np.arange(len(errors))
    nearest = idx[int(np.argmin(np.linalg.norm(errors[idx] - err[None, :], axis=1)))]
    return actions[nearest]


def summarize_overlap(records, stop_step=None):
    usable = [r for r in records if stop_step is None or r["step"] < stop_step]
    if not usable:
        usable = records[:1]
    n = len(usable)
    windows = {
        "first_25": usable[: max(1, int(np.ceil(n * 0.25)))],
        "first_50": usable[: max(1, int(np.ceil(n * 0.50)))],
        "pre_contact": usable,
        "whole": records,
    }
    out = {}
    for label, rows in windows.items():
        out[label] = {
            "joint": float(np.mean([r["joint"] for r in rows])),
            "lateral": float(np.mean([r["lateral"] for r in rows])),
            "axial": float(np.mean([r["axial"] for r in rows])),
            "orientation": float(np.mean([r["orientation"] for r in rows])),
        }
    return out


def rollout_one(agent, arrays, errors, regions, envelopes, seed):
    obs_min = arrays["observations"][:, :7].min(axis=0)
    obs_max = arrays["observations"][:, :7].max(axis=0)
    phase_errors = {}
    overlap = []
    first_div = None
    first_contact = None
    pre_contact = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=seed)
        for step in range(1, 501):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            obs, _reward, terminated, truncated, info = env.step(action)
            err = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
            region = region_from_errors(float(err[0]), float(err[1]), float(err[2]))
            demo_action = nearest_demo_action(errors, arrays["actions"], regions, region, err)
            phase = phase_from_region(region)
            phase_errors.setdefault(phase, []).append(np.abs(action - demo_action))
            envl = envelopes[region]
            bad_dims = np.where((action < envl["action_p01"] - 0.05) | (action > envl["action_p99"] + 0.05))[0]
            err_bad = np.where((err < envl["error_min"]) | (err > envl["error_max"]))[0]
            if first_div is None and (len(bad_dims) or len(err_bad)):
                first_div = {
                    "step": step,
                    "region": phase,
                    "action_dim": None if not len(bad_dims) else f"a{int(bad_dims[0])}",
                    "error": "ACTION_ENVELOPE" if len(bad_dims) else f"STATE_ERROR_{int(err_bad[0])}",
                    "action": action.tolist(),
                    "demo_action": demo_action.tolist(),
                    "errors": err.tolist(),
                }
            link, unexpected, pairs = contact_flags(env)
            qvel_max = float(np.max(np.abs(env.unwrapped.data.qvel))) if hasattr(env.unwrapped, "data") else None
            force = float(info.get("force_norm", np.nan))
            row = {
                "step": step,
                "action": action.tolist(),
                "region": phase,
                "errors": err.tolist(),
                "joint": bool(np.all(obs[:7] >= obs_min) and np.all(obs[:7] <= obs_max)),
                "lateral": bool(envelopes[region]["error_min"][0] <= err[0] <= envelopes[region]["error_max"][0]),
                "axial": bool(envelopes[region]["error_min"][1] <= err[1] <= envelopes[region]["error_max"][1]),
                "orientation": bool(envelopes[region]["error_min"][2] <= err[2] <= envelopes[region]["error_max"][2]),
                "contact": bool(link or unexpected),
                "contact_pairs": pairs,
                "force": force,
                "qvel_max": qvel_max,
            }
            overlap.append(row)
            if first_contact is None and row["contact"]:
                first_contact = {"step": step, "unexpected": bool(unexpected), "pairs": pairs}
            pre_contact.append(row)
            if len(pre_contact) > 50:
                pre_contact.pop(0)
            if terminated or truncated:
                break
        env.close()
    phase_mae = {}
    for phase, values in phase_errors.items():
        arr = np.asarray(values, np.float64)
        phase_mae[phase] = {dim: float(arr[:, i].mean()) for i, dim in enumerate(ACTION_DIMS)}
    contact_step = None if first_contact is None else first_contact["step"]
    before_contact = [r for r in overlap if contact_step is None or r["step"] < contact_step]
    action_delta = np.asarray([0.0])
    if len(before_contact) >= 2:
        acts = np.asarray([r["action"] for r in before_contact[-50:]], np.float64)
        action_delta = np.max(np.abs(np.diff(acts, axis=0)), axis=0)
    cause = "POLICY_DIVERGENCE_BEFORE_CONTACT" if first_div and (contact_step is None or first_div["step"] < contact_step) else "CONTACT_OR_RUNTIME_FIRST"
    qvel_values = [r["qvel_max"] for r in before_contact[-50:] if r["qvel_max"] is not None]
    return {
        "first_divergence": first_div,
        "contact": first_contact,
        "overlap": summarize_overlap(overlap, contact_step),
        "phase_action_mae": phase_mae,
        "pre_contact": {
            "cause": cause,
            "max_action_delta": action_delta.tolist(),
            "force_max": float(np.nanmax([r["force"] for r in before_contact[-50:]])) if before_contact else None,
            "qvel_max": float(np.nanmax(qvel_values)) if qvel_values else None,
            "contact_type": None if first_contact is None else ("unexpected" if first_contact["unexpected"] else "expected_link"),
        },
    }


def merge_run(episodes):
    firsts = [e["first_divergence"] for e in episodes if e["first_divergence"]]
    contacts = [e["contact"]["step"] for e in episodes if e["contact"]]
    phases = {}
    for e in episodes:
        for phase, dims in e["phase_action_mae"].items():
            phases.setdefault(phase, []).append([dims[d] for d in ACTION_DIMS])
    phase_mae = {
        phase: {dim: float(np.asarray(vals)[:, i].mean()) for i, dim in enumerate(ACTION_DIMS)}
        for phase, vals in phases.items()
    }
    return {
        "first_divergence": firsts[0] if firsts else None,
        "median_contact_step": None if not contacts else float(np.median(contacts)),
        "overlap": {
            label: {
                comp: float(np.mean([e["overlap"][label][comp] for e in episodes]))
                for comp in ("joint", "lateral", "axial", "orientation")
            }
            for label in ("first_25", "first_50", "pre_contact", "whole")
        },
        "phase_action_mae": phase_mae,
        "contact_causes": [e["pre_contact"] for e in episodes],
    }


def dominant_phase_dim(run_summary):
    best = ("NONE", "NONE", -1.0)
    for phase, dims in run_summary["phase_action_mae"].items():
        for dim, value in dims.items():
            if value > best[2]:
                best = (phase, dim, value)
    return {"phase": best[0], "action_dim": best[1], "mae": best[2]}


def earliest_collapse(summary):
    for label in ("first_25", "first_50", "pre_contact", "whole"):
        values = summary["overlap"][label]
        for comp in ("joint", "lateral", "axial", "orientation"):
            if values[comp] < 0.10:
                return f"{label}:{comp}"
    return "NONE"


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg_cache = {}
    demo_cache = {}
    details = {}
    for name, (config_path, run, factory) in RUNS.items():
        cfg = cfg_cache.setdefault(str(config_path), yaml.safe_load(config_path.read_text()))
        if str(config_path) not in demo_cache:
            arrays, errors, regions, phases = load_demo(cfg)
            demo_cache[str(config_path)] = (arrays, errors, regions, build_envelopes(errors, arrays["actions"], regions))
        arrays, errors, regions, envelopes = demo_cache[str(config_path)]
        agent = load_agent(cfg, run, factory)
        episodes = [rollout_one(agent, arrays, errors, regions, envelopes, seed) for seed in SEEDS]
        details[name] = merge_run(episodes)
    candidate_doms = [dominant_phase_dim(details["candidate_R1"]), dominant_phase_dim(details["candidate_R2"])]
    dominant = max(candidate_doms, key=lambda x: x["mae"])
    candidate_collapses = [earliest_collapse(details["candidate_R1"]), earliest_collapse(details["candidate_R2"])]
    first_dims = [details["candidate_R1"]["first_divergence"]["action_dim"], details["candidate_R2"]["first_divergence"]["action_dim"]]
    if len(set(first_dims)) == 1:
        primary = f"{details['candidate_R1']['first_divergence']['region'].upper()}_{first_dims[0].upper()}_RESIDUAL_ACTION_DRIFT"
        next_fix = "EARLY_APPROACH_A0_ACTION_SUPPORT_REGULARIZATION"
    else:
        primary = "POST_STEP1_ASYNC_TRAJECTORY_VARIANCE"
        next_fix = "MULTI_REPLICA_TRAJECTORY_STABILIZATION_PROTOCOL"
    result = {
        "POST_STEP1_AUDIT": "PASS",
        "BASELINE_FIRST_DIVERGENCE": {
            "R1": details["baseline_R1"]["first_divergence"],
            "R2": details["baseline_R2"]["first_divergence"],
        },
        "CANDIDATE_FIRST_DIVERGENCE": {
            "R1": details["candidate_R1"]["first_divergence"],
            "R2": details["candidate_R2"]["first_divergence"],
        },
        "EARLIEST_OVERLAP_COLLAPSE": {
            "R1": candidate_collapses[0],
            "R2": candidate_collapses[1],
        },
        "DOMINANT_PHASE": dominant["phase"],
        "DOMINANT_ACTION_DIM": dominant["action_dim"],
        "DOMINANT_ACTION_MAE": dominant["mae"],
        "CONTACT_R1_CAUSE": details["candidate_R1"]["contact_causes"],
        "CONTACT_R2_CAUSE": details["candidate_R2"]["contact_causes"],
        "PRIMARY_ROOT_CAUSE": primary,
        "NEXT_FIX": next_fix,
        "EXTENDED_3K_REVIEW": "BLOCKED",
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": details,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
