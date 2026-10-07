#!/usr/bin/env python3
"""Read-only N=2 geometry gate for GLOBAL_ACTION_SUPPORT_CQ."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from peg_env_adapter import make_peg_state_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.global_action_support_cq_candidate import make_global_action_support_cq_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_global_action_support_cq_1k_seed0.yaml"
RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_global_action_support_cq_n2_20260907/replica1",
    "R2": ROOT / "results/ablation/peg_f3_v2_global_action_support_cq_n2_20260907/replica2",
}
OUT_ROOT = ROOT / "results/diagnostics/peg"
GRID = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
BOUNDARY_THRESHOLD = 0.95
OPTIMUM_TOLERANCE = 0.05
DIMS = range(6)


def unique_out() -> Path:
    stem = "f3_v2_global_action_support_cq_geometry_n2_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def load_demo_state_table(cfg):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    obs = np.asarray([row["observations"] for row in rows], np.float32)
    actions = np.asarray([row["actions"] for row in rows], np.float32)
    phases = []
    metrics = []
    prov = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    for entry in prov:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            row = json.loads(line)
            phases.append(row["phase"])
            metrics.append(row["after"]["metrics"])
    return obs, actions, np.asarray(phases), metrics


def initial_observation(cfg):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        env.close()
    return np.asarray(obs, np.float32)


def selected_states(cfg):
    obs_table, actions, phases, metrics = load_demo_state_table(cfg)
    init = initial_observation(cfg)
    init_idx = int(np.argmin(np.linalg.norm(obs_table - init[None], axis=1)))
    approach_idx = int(np.where(phases == "approach")[0][0])
    lateral_idx = int(np.where(phases == "lateral_alignment")[0][0])
    orient_idx = int(np.argmax([m["orientation_error"] for m in metrics]))
    return {
        "initial": (init, actions[init_idx], init_idx),
        "approach": (obs_table[approach_idx], actions[approach_idx], approach_idx),
        "lateral": (obs_table[lateral_idx], actions[lateral_idx], lateral_idx),
        "orientation": (obs_table[orient_idx], actions[orient_idx], orient_idx),
    }


def load_agent(cfg, run):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_global_action_support_cq_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def q_eval(agent, obs, actions):
    obs_batch = jnp.asarray(np.repeat(obs[None], len(actions), axis=0))
    q = np.asarray(jax.device_get(agent.forward_critic(obs_batch, jnp.asarray(actions), rng=None, train=False)))
    return q


def sweep_dim(agent, obs, demo_action, policy_action, dim, reference):
    base = demo_action if reference == "demo" else policy_action
    actions = np.repeat(base[None], len(GRID), axis=0)
    actions[:, dim] = GRID
    q_heads = q_eval(agent, obs, actions)
    q_mean = q_heads.mean(axis=0)
    arg = float(GRID[int(np.argmax(q_mean))])
    demo_i = int(np.argmin(np.abs(GRID - demo_action[dim])))
    policy_i = int(np.argmin(np.abs(GRID - policy_action[dim])))
    return {
        "argmax": arg,
        "boundary": abs(arg) >= BOUNDARY_THRESHOLD,
        "demo_optimum": abs(arg - float(demo_action[dim])) <= OPTIMUM_TOLERANCE,
        "policy_optimum": abs(arg - float(policy_action[dim])) <= OPTIMUM_TOLERANCE,
        "q_at_demo": float(q_mean[demo_i]),
        "q_at_policy": float(q_mean[policy_i]),
        "q_max": float(np.max(q_mean)),
        "local_grad_at_demo": float(np.gradient(q_mean, GRID)[demo_i]),
        "local_grad_at_policy": float(np.gradient(q_mean, GRID)[policy_i]),
        "head_argmax": [float(GRID[int(np.argmax(q_heads[h]))]) for h in range(q_heads.shape[0])],
        "head_boundary": [bool(abs(float(GRID[int(np.argmax(q_heads[h]))])) >= BOUNDARY_THRESHOLD) for h in range(q_heads.shape[0])],
    }


def analyze_state(agent, obs, demo_action):
    dist = agent.forward_policy(jnp.asarray(obs[None]), rng=None, train=False)
    policy_action = np.asarray(jax.device_get(dist.mode()))[0]
    dims = {}
    for dim in DIMS:
        dims[f"a{dim}"] = {
            "demo_ref": sweep_dim(agent, obs, demo_action, policy_action, dim, "demo"),
            "policy_ref": sweep_dim(agent, obs, demo_action, policy_action, dim, "policy"),
        }
    return {"policy_action": policy_action.tolist(), "dims": dims}


def count_for_state(state):
    vals = [state["dims"][f"a{d}"]["policy_ref"] for d in DIMS]
    return {
        "boundary": sum(v["boundary"] for v in vals),
        "demo_optimum": sum(v["demo_optimum"] for v in vals),
        "policy_optimum": sum(v["policy_optimum"] for v in vals),
        "interior": sum(not v["boundary"] for v in vals),
        "head_consensus_boundary": sum(all(v["head_boundary"]) for v in vals),
        "one_head_boundary": sum(any(v["head_boundary"]) and not all(v["head_boundary"]) for v in vals),
    }


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    states = selected_states(cfg)
    detail = {}
    phase_summary = {phase: [] for phase in states}
    head_votes = []
    for run_name, run in RUNS.items():
        agent = load_agent(cfg, run)
        detail[run_name] = {}
        for phase, (obs, demo_action, demo_index) in states.items():
            state = analyze_state(agent, obs, demo_action)
            counts = count_for_state(state)
            detail[run_name][phase] = {"demo_index": demo_index, "counts": counts, **state}
            phase_summary[phase].append(counts)
            head_votes.append(counts)
    initial_counts = {r: detail[r]["initial"]["counts"] for r in detail}
    phase_fraction = {
        phase: float(np.mean([c["boundary"] / len(DIMS) for c in counts]))
        for phase, counts in phase_summary.items()
    }
    demo_opt_fraction = {
        phase: float(np.mean([c["demo_optimum"] / len(DIMS) for c in counts]))
        for phase, counts in phase_summary.items()
    }
    q_adv = {}
    for phase in states:
        vals = []
        for run_name in detail:
            for dim in DIMS:
                v = detail[run_name][phase]["dims"][f"a{dim}"]["policy_ref"]
                vals.append(v["q_at_policy"] - v["q_at_demo"])
        q_adv[phase] = float(np.mean(vals))
    geometry_pass = (
        all(initial_counts[r]["boundary"] <= 4 for r in initial_counts)
        and all(initial_counts[r]["demo_optimum"] > 0 for r in initial_counts)
        and phase_fraction["initial"] < 1.0
        and phase_fraction["approach"] < 1.0
    )
    consensus = sum(c["head_consensus_boundary"] for c in head_votes)
    one_head = sum(c["one_head_boundary"] for c in head_votes)
    if consensus > one_head and consensus > 0:
        head_agreement = "CONSENSUS"
    elif one_head > consensus:
        head_agreement = "ONE_HEAD_DOMINATED"
    else:
        head_agreement = "MIXED"
    root = "NONE_AT_GEOMETRY_GATE" if geometry_pass else "GLOBAL_ACTION_SUPPORT_CQ_GEOMETRY_SCIENTIFIC_FAIL"
    next_fix = "BEHAVIOR_AUDIT" if geometry_pass else "STOP_NO_CANDIDATE2"
    result = {
        "GLOBAL_CQ_GEOMETRY_1K": "PASS" if geometry_pass else "FAIL",
        "R1_BOUNDARY_ARGMAX_COUNT": initial_counts["R1"]["boundary"],
        "R2_BOUNDARY_ARGMAX_COUNT": initial_counts["R2"]["boundary"],
        "R1_DEMO_OPTIMUM_COUNT": initial_counts["R1"]["demo_optimum"],
        "R2_DEMO_OPTIMUM_COUNT": initial_counts["R2"]["demo_optimum"],
        "INITIAL_PHASE_BOUNDARY_FRACTION": phase_fraction["initial"],
        "APPROACH_BOUNDARY_FRACTION": phase_fraction["approach"],
        "LATERAL_BOUNDARY_FRACTION": phase_fraction["lateral"],
        "ORIENTATION_BOUNDARY_FRACTION": phase_fraction["orientation"],
        "PHASE_DEMO_OPTIMUM_FRACTION": demo_opt_fraction,
        "PHASE_POLICY_Q_ADVANTAGE_OVER_DEMO": q_adv,
        "CRITIC_HEAD_AGREEMENT": head_agreement,
        "TARGET_VS_CURRENT_GEOMETRY": "UNAVAILABLE",
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT_FIX": next_fix,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": detail,
    }
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(finite(result), indent=2, sort_keys=True) + "\n")
    summary_keys = [k for k in result if k != "DETAILS"]
    print(json.dumps(finite({k: result[k] for k in summary_keys}), sort_keys=True))
    print(str(out / "result.json"))


if __name__ == "__main__":
    main()
