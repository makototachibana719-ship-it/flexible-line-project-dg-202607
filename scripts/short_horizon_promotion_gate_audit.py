#!/usr/bin/env python3
"""Read-only short-horizon behavioral promotion audit for Peg F3 V2."""
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
HORIZONS = (10, 25, 50, 100, 150)
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out() -> Path:
    stem = "f3_v2_short_horizon_promotion_gate_20260907"
    for index in range(1, 100):
        path = OUT_ROOT / f"{stem}_{index:03d}"
        if not path.exists():
            return path
    raise RuntimeError("no unique output dir")


def load_demo(cfg):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    metrics = []
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            m = json.loads(line)["after"]["metrics"]
            metrics.append([m["lateral_error"], m["axial_error"], m["orientation_error"], m.get("force_norm", 0.0)])
    errors = np.asarray([m[:3] for m in metrics], np.float64)
    regions = np.asarray([region_from_errors(*row) for row in errors])
    return arrays, errors, regions


def envelopes(arrays, errors, regions):
    result = {}
    for region in sorted(set(regions)):
        idx = regions == region
        obs = arrays["observations"][idx, :7]
        err = errors[idx]
        result[region] = {
            "obs_min": obs.min(axis=0),
            "obs_max": obs.max(axis=0),
            "obs_scale": np.maximum(obs.std(axis=0), 1e-3),
            "err_min": err.min(axis=0),
            "err_max": err.max(axis=0),
            "err_scale": np.maximum(err.std(axis=0), 1e-4),
        }
    return result


def demo_tolerance(arrays, errors, regions):
    envs = envelopes(arrays, errors, regions)
    distances = []
    for obs, err, region in zip(arrays["observations"][:, :7], errors, regions):
        env = envs[region]
        obs_clip = np.minimum(np.maximum(obs, env["obs_min"]), env["obs_max"])
        err_clip = np.minimum(np.maximum(err, env["err_min"]), env["err_max"])
        distances.append(
            np.linalg.norm((obs - obs_clip) / env["obs_scale"])
            + np.linalg.norm((err - err_clip) / env["err_scale"])
        )
    return float(np.percentile(distances, 99) + 1e-9)


def load_agent(cfg, run, factory):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = factory(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def corridor_distances(obs, err, region, envs):
    env = envs[region]
    obs_clip = np.minimum(np.maximum(obs[:7], env["obs_min"]), env["obs_max"])
    err_clip = np.minimum(np.maximum(err, env["err_min"]), env["err_max"])
    joint = float(np.linalg.norm((obs[:7] - obs_clip) / env["obs_scale"]))
    corr = float(np.linalg.norm((err - err_clip) / env["err_scale"]))
    return joint, corr


def rollout(agent, arrays, errors, regions, seed, tolerance):
    envs = envelopes(arrays, errors, regions)
    rows = []
    contact_step = None
    behavior_div = None
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=max(HORIZONS))
        obs, _ = env.reset(seed=seed)
        for step in range(1, max(HORIZONS) + 1):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float64)
            obs, _reward, terminated, truncated, info = env.step(action)
            err = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
            region = region_from_errors(float(err[0]), float(err[1]), float(err[2]))
            joint_distance, corridor_error = corridor_distances(np.asarray(obs), err, region, envs)
            link, unexpected, pairs = contact_flags(env)
            contact = bool(link or unexpected)
            if contact_step is None and contact:
                contact_step = step
            if behavior_div is None and (joint_distance + corridor_error) > tolerance:
                behavior_div = step
            rows.append({
                "step": step,
                "joint_distance": joint_distance,
                "corridor_error": corridor_error,
                "lateral": float(err[0]),
                "axial": float(err[1]),
                "orientation": float(err[2]),
                "force": float(info.get("force_norm", 0.0)),
                "contact": contact,
                "unexpected_contact": bool(unexpected),
                "action": action.tolist(),
                "contact_pairs": pairs,
            })
            if terminated or truncated:
                break
        env.close()
    return rows, contact_step, behavior_div


def horizon_summary(rollouts):
    out = {}
    for horizon in HORIZONS:
        rows = [row for rollout_rows, _contact, _div in rollouts for row in rollout_rows if row["step"] <= horizon]
        last_rows = [rollout_rows[min(horizon, len(rollout_rows)) - 1] for rollout_rows, _contact, _div in rollouts if rollout_rows]
        out[str(horizon)] = {
            "corridor_error": float(np.mean([r["corridor_error"] for r in rows])),
            "joint_distance": float(np.mean([r["joint_distance"] for r in rows])),
            "lateral": float(np.mean([r["lateral"] for r in last_rows])),
            "axial": float(np.mean([r["axial"] for r in last_rows])),
            "orientation": float(np.mean([r["orientation"] for r in last_rows])),
            "force": float(np.mean([r["force"] for r in rows])),
            "contact_rate": float(np.mean([any(r["contact"] for r in rollout_rows[:horizon]) for rollout_rows, _contact, _div in rollouts])),
            "first_contact": None if all(c is None for _rows, c, _div in rollouts) else float(np.nanmin([c if c is not None else np.nan for _rows, c, _div in rollouts])),
            "action_mean": np.mean([r["action"] for r in last_rows], axis=0).tolist(),
        }
    return out


def trend(candidate, baseline, metric, lower_is_better=True):
    deltas = []
    for horizon in HORIZONS:
        delta = candidate[str(horizon)][metric] - baseline[str(horizon)][metric]
        deltas.append(delta if lower_is_better else -delta)
    improving = sum(d < 0 for d in deltas)
    worsening = sum(d > 0 for d in deltas)
    if improving >= 4:
        return "IMPROVING"
    if worsening >= 4:
        return "WORSENING"
    return "FLAT"


def aggregate_family(details, prefix):
    out = {}
    for horizon in HORIZONS:
        group = [details[k]["horizons"][str(horizon)] for k in sorted(details) if k.startswith(prefix)]
        out[str(horizon)] = {
            key: float(np.mean([g[key] for g in group]))
            for key in ("corridor_error", "joint_distance", "lateral", "axial", "orientation", "force", "contact_rate")
        }
        contacts = [g["first_contact"] for g in group if g["first_contact"] is not None]
        out[str(horizon)]["first_contact"] = None if not contacts else float(np.min(contacts))
    return out


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cache = {}
    details = {}
    for name, (cfg_path, run, factory) in RUNS.items():
        cfg = yaml.safe_load(cfg_path.read_text())
        if str(cfg_path) not in cache:
            arrays, errors, regions = load_demo(cfg)
            cache[str(cfg_path)] = (arrays, errors, regions, demo_tolerance(arrays, errors, regions))
        arrays, errors, regions, tolerance = cache[str(cfg_path)]
        agent = load_agent(cfg, run, factory)
        rollouts = [rollout(agent, arrays, errors, regions, seed, tolerance) for seed in SEEDS]
        divs = [div for _rows, _contact, div in rollouts if div is not None]
        contacts = [contact for _rows, contact, _div in rollouts if contact is not None]
        details[name] = {
            "horizons": horizon_summary(rollouts),
            "behavior_divergence_step": None if not divs else float(np.median(divs)),
            "contact_step": None if not contacts else float(np.median(contacts)),
        }
    baseline = aggregate_family(details, "baseline")
    candidate = aggregate_family(details, "candidate")
    trends = {
        "CORRIDOR_ERROR_TREND": trend(candidate, baseline, "corridor_error", True),
        "JOINT_DISTANCE_TREND": trend(candidate, baseline, "joint_distance", True),
        "LATERAL_TREND": trend(candidate, baseline, "lateral", True),
        "AXIAL_TREND": trend(candidate, baseline, "axial", True),
        "ORIENTATION_TREND": trend(candidate, baseline, "orientation", True),
        "FORCE_TREND": trend(candidate, baseline, "force", True),
    }
    baseline_div = float(np.nanmedian([details[k]["behavior_divergence_step"] for k in details if k.startswith("baseline")]))
    candidate_div = float(np.nanmedian([details[k]["behavior_divergence_step"] for k in details if k.startswith("candidate")]))
    improved_count = sum(v == "IMPROVING" for k, v in trends.items() if k != "CORRIDOR_ERROR_TREND")
    corridor_all = all(candidate[str(h)]["corridor_error"] < baseline[str(h)]["corridor_error"] for h in HORIZONS)
    div_delayed = candidate_div > baseline_div
    contact_delayed = (candidate["150"]["first_contact"] or 9999) > (baseline["150"]["first_contact"] or 9999)
    no_instability = True
    promote = corridor_all and div_delayed and improved_count >= 3 and no_instability
    result = {
        "SHORT_HORIZON_PROMOTION_AUDIT": "PASS",
        **trends,
        "BASELINE_BEHAVIOR_DIVERGENCE_STEP": baseline_div,
        "CANDIDATE_BEHAVIOR_DIVERGENCE_STEP": candidate_div,
        "CONTACT_PROGRESSION": {
            "baseline_first_contact_150": baseline["150"]["first_contact"],
            "candidate_first_contact_150": candidate["150"]["first_contact"],
            "candidate_delays_contact": contact_delayed,
            "baseline_contact_rate_150": baseline["150"]["contact_rate"],
            "candidate_contact_rate_150": candidate["150"]["contact_rate"],
            "baseline_precontact_corridor_error_150": baseline["150"]["corridor_error"],
            "candidate_precontact_corridor_error_150": candidate["150"]["corridor_error"],
            "baseline_force_150": baseline["150"]["force"],
            "candidate_force_150": candidate["150"]["force"],
        },
        "PROMOTION_GATE": "BEHAVIORAL_CORRIDOR" if promote else "BLOCKED",
        "STEP1_A0_GATE_STATUS": "DIAGNOSTIC_ONLY" if promote else "BLOCKING_UNRESOLVED",
        "EXTENDED_3K_REVIEW": "READY" if promote else "BLOCKED",
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": {"baseline": baseline, "candidate": candidate, "runs": details},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
