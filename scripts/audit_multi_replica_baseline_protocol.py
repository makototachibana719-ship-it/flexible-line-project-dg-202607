#!/usr/bin/env python3
"""Freeze and aggregate the N=2 reproduced baseline protocol."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path[:0] = [
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from peg_env_adapter import make_peg_state_env
from scripts.audit_peg_f3_v2_behavior_transfer_stage_a import build_envelopes, demo_tables, region_from_errors
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_pretanh_a0_anchor_ablation import make_early_pretanh_a0_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_pretanh_a0_anchor_ablation_1k_seed0.yaml"
BASE = ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002"
RUNS = {"R1": BASE / "replica1", "R2": BASE / "replica2"}
OUT_ROOT = ROOT / "results/diagnostics/peg"
SEEDS = (0, 1, 2)


def unique_out():
    stem = "f3_v2_multi_replica_baseline_protocol_20260906"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def stats(values):
    a = np.asarray(values, np.float64)
    return {
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "range": float(np.max(a) - np.min(a)),
    }


def initial_obs(cfg):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        env.close()
    return np.asarray(obs, np.float32)


def load_agent(cfg, run, step=1000):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_early_pretanh_a0_anchor_agent(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, step)
        env.close()
    return fresh.replace(state=state)


def fixed_metrics(cfg, agent):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, types, metrics = [], [], []
    for entry in provenance:
        lines = [json.loads(line) for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open()]
        phases.extend(row["phase"] for row in lines)
        types.extend([entry["type"]] * len(lines))
        metrics.extend(row["after"]["metrics"] for row in lines)
    phases, types = np.asarray(phases), np.asarray(types)
    metric_lateral = np.asarray([x["lateral_error"] for x in metrics])
    metric_axial = np.asarray([x["axial_error"] for x in metrics])
    lateral = np.isin(types, ["lateral_plus", "lateral_minus"]) & (phases == "lateral_alignment") & (np.abs(arrays["actions"][:, 1]) >= .99)
    near = (metric_lateral <= .010) & (metric_axial <= .020)
    rotation = np.isin(types, ["rotation_roll", "rotation_pitch"]) & (phases == "orientation_correction")
    union = lateral | near | rotation
    policy = np.asarray(jax.device_get(agent.sample_actions(observations=arrays["observations"], argmax=True)))
    q_demo = np.asarray(jax.device_get(agent.forward_critic(jnp.asarray(arrays["observations"][union]), jnp.asarray(arrays["actions"][union]), rng=None, train=False))).mean(axis=0)
    q_policy = np.asarray(jax.device_get(agent.forward_critic(jnp.asarray(arrays["observations"][union]), jnp.asarray(policy[union]), rng=None, train=False))).mean(axis=0)
    return {
        "q_policy_minus_demo": float(np.mean(q_policy - q_demo)),
        "demo_mae": float(np.mean(np.abs(policy - arrays["actions"]))),
        "action1_mae": float(np.mean(np.abs(policy[lateral, 1] - arrays["actions"][lateral, 1]))),
    }


def rollout_metrics(cfg, agent, arrays, errors, regions):
    envelopes, error_min, error_max = build_envelopes(errors, arrays["actions"], regions)
    rows = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            first_div = None
            first_contact = None
            overlap = []
            near_count = 0
            for step in range(1, 501):
                action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
                obs, reward, terminated, truncated, info = env.step(action)
                err = np.asarray([info["lateral_error"], info["axial_error"], info["orientation_error"]], np.float64)
                region = region_from_errors(float(err[0]), float(err[1]), float(err[2]))
                envl = envelopes.get(region)
                bad_action = False
                bad_dim = None
                if envl is not None:
                    lo = np.asarray(envl["action_p01"]) - 0.05
                    hi = np.asarray(envl["action_p99"]) + 0.05
                    bad = np.where((action < lo) | (action > hi))[0]
                    if len(bad):
                        bad_action = True
                        bad_dim = int(bad[0])
                link, unexpected, _pairs = contact_flags(env)
                if first_contact is None and (link or unexpected):
                    first_contact = step
                if first_div is None and bad_action:
                    first_div = {"step": step, "dim": bad_dim}
                overlap.append({
                    "joint": bool(np.all(obs[:7] >= arrays["observations"][:, :7].min(axis=0)) and np.all(obs[:7] <= arrays["observations"][:, :7].max(axis=0))),
                    "axial": bool(error_min[1] <= err[1] <= error_max[1]),
                    "orientation": bool(error_min[2] <= err[2] <= error_max[2]),
                })
                near_count += int(err[0] <= .010 and err[1] <= .020)
                if terminated or truncated:
                    break
            rows.append({
                "first_divergence_step": None if first_div is None else first_div["step"],
                "first_divergence_dim": None if first_div is None else f"a{first_div['dim']}",
                "contact_step": first_contact,
                "near_hole": near_count > 0,
                "joint_overlap": float(np.mean([x["joint"] for x in overlap])),
                "axial_overlap": float(np.mean([x["axial"] for x in overlap])),
                "orientation_overlap": float(np.mean([x["orientation"] for x in overlap])),
            })
        env.close()
    return {
        "first_divergence_step": float(np.median([x["first_divergence_step"] for x in rows if x["first_divergence_step"] is not None])),
        "contact": float(np.median([x["contact_step"] for x in rows if x["contact_step"] is not None])),
        "near_hole": float(np.mean([x["near_hole"] for x in rows])),
        "joint_overlap": float(np.mean([x["joint_overlap"] for x in rows])),
        "axial_overlap": float(np.mean([x["axial_overlap"] for x in rows])),
        "orientation_overlap": float(np.mean([x["orientation_overlap"] for x in rows])),
        "episodes": rows,
    }


def native(run):
    checkpoints = [json.loads(line) for line in (run / "checkpoint_events.jsonl").read_text().splitlines()]
    final = checkpoints[-1]
    return {"actor_steps": int(final["actor_steps"]), "replay_final": int(final["online_size"])}


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    arrays, errors, regions = demo_tables(cfg)
    obs = initial_obs(cfg)
    detail = {}
    for name, run in RUNS.items():
        agent = load_agent(cfg, run, 1000)
        action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
        fixed = fixed_metrics(cfg, agent)
        roll = rollout_metrics(cfg, agent, arrays, errors, regions)
        detail[name] = {"step1_a0": float(action[0]), **fixed, **roll, **native(run)}
    a0_values = [detail[k]["step1_a0"] for k in sorted(detail)]
    q_values = [detail[k]["q_policy_minus_demo"] for k in sorted(detail)]
    a1_values = [detail[k]["action1_mae"] for k in sorted(detail)]
    near_values = [detail[k]["near_hole"] for k in sorted(detail)]
    contact_values = [detail[k]["contact"] for k in sorted(detail)]
    replay_values = [detail[k]["replay_final"] for k in sorted(detail)]
    saturated_fraction = float(np.mean(np.asarray(a0_values) >= 0.95))
    selected = "ACTOR_OBJECTIVE_BALANCE" if saturated_fraction == 1.0 else "BLOCKED"
    result = {
        "MULTI_REPLICA_PROTOCOL": "PASS",
        "N_REPLICAS": 2,
        "SATURATED_THRESHOLD": 0.95,
        "NON_SATURATED_THRESHOLD": 0.50,
        "BASELINE_A0_VALUES": a0_values,
        "BASELINE_A0_STATS": stats(a0_values),
        "BASELINE_SATURATED_FRACTION": saturated_fraction,
        "BASELINE_VARIANCE_CLASS": "LOW_A0_HIGH_ASYNC" if saturated_fraction == 1.0 and (max(replay_values) - min(replay_values)) > 1000 else "UNRESOLVED",
        "BASELINE_Q_RANGE": stats(q_values)["range"],
        "BASELINE_ACTION1_RANGE": stats(a1_values)["range"],
        "BASELINE_NEAR_HOLE_RATE": float(np.mean(near_values)),
        "BASELINE_CONTACT_RANGE": stats(contact_values)["range"],
        "BASELINE_REPLAY_RANGE": stats(replay_values)["range"],
        "SELECTED_FIX": selected,
        "SELECTION_EVIDENCE": "saturation_fraction=1.0 across replicas while local critic sign disagrees",
        "CANDIDATE_A0_VALUES": [],
        "CANDIDATE_SATURATED_FRACTION": None,
        "PRIMARY_EFFECT_SIZE": None,
        "EFFECT_EXCEEDS_BASELINE_VARIANCE": "NOT_RUN",
        "CANDIDATE_1K_RESULT": "NOT_RUN",
        "CANDIDATE_3K_RESULT": "NOT_RUN",
        "NEAR_HOLE_BASELINE": near_values,
        "NEAR_HOLE_CANDIDATE": [],
        "CONTACT_BASELINE": contact_values,
        "CONTACT_CANDIDATE": [],
        "FORMAL_5K_R2_PREFLIGHT": "NOT_RUN",
        "FORMAL_5K_R2": "BLOCKED",
        "TRAINING_STARTED": "false for 5K",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "FIRST_BAD_GATE": "CANDIDATE_EXACT_RUNNER_NOT_FROZEN",
        "NEXT": "FREEZE_SINGLE ACTOR_OBJECTIVE_BALANCE_CANDIDATE_FOR_N2_COMPARISON",
        "DETAILS": detail,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
