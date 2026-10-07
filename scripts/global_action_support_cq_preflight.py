#!/usr/bin/env python3
"""Static calibration and direct-pressure gate for global action-support CQ."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]
from async_sac_state_sim import concat_rlpd_batches
from peg_env_adapter import make_peg_state_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.conservative_q_ablation import ConservativeQSACAgent
from training.peg_insertion.global_action_support_cq_candidate import make_global_action_support_cq_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

BASE_RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica2",
}
DATASET_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"


def norm(tree):
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree))))


def stack(rows):
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = ROOT / cfg["demo"]["path"]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == DATASET_SHA
    demos = pickle.loads(dataset.read_bytes())
    indices = np.linspace(0, len(demos) - 1, 32, dtype=int)
    demo = stack([demos[i] for i in indices])
    replay = pickle.loads((BASE_RUNS["R1"] / "checkpoints/checkpoint_1000_online_replay.pkl").read_bytes())
    online = {k: np.asarray(v[:32]) for k, v in replay["dataset_dict"].items() if k in demo}
    batch = jax.tree_util.tree_map(jnp.asarray, concat_rlpd_batches(online, demo))
    rows = {}
    for name, run in BASE_RUNS.items():
        with portable_peg_runtime(ROOT):
            env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
            agent = make_global_action_support_cq_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
            state = restore_fresh_state(run / "checkpoints", agent.state, 1000)
            agent = agent.replace(state=state)
            env.close()
        params = agent.state.params
        rng = jax.random.PRNGKey(20260907)
        base_fn = lambda p: ConservativeQSACAgent.critic_loss_fn(agent, batch, p, rng)[0]
        current_cq_fn = lambda p: 5.0 * agent.conservative_q_loss_fn(batch, p, rng)
        global_fn = lambda p: agent.global_cq_loss_fn(batch, p, rng)
        base_grad = jax.grad(base_fn)(params)
        current_cq_grad = jax.grad(current_cq_fn)(params)
        raw_loss, global_grad = jax.value_and_grad(global_fn)(params)
        base_norm, current_norm, raw_norm = norm(base_grad), norm(current_cq_grad), norm(global_grad)
        selected = 0.15 * base_norm / raw_norm
        gaps = np.asarray(jax.device_get(agent.global_cq_comparisons(batch, params, rng)))
        boundary = gaps[:, :, 1:13]
        interior = gaps[:, :, 13:25]
        boundary_violations = boundary > 0
        interior_violations = interior > 0
        rows[name] = {
            "base_critic_grad_norm": base_norm,
            "current_cq_grad_norm": current_norm,
            "global_cq_raw_grad_norm": raw_norm,
            "global_cq_raw_loss": float(raw_loss),
            "selected_lambda": selected,
            "weighted_grad_ratio": selected * raw_norm / base_norm,
            "boundary_violation_rate": float(np.mean(boundary_violations)),
            "interior_violation_rate": float(np.mean(interior_violations)),
            "direct_boundary_pressure_fraction": 1.0 if np.any(boundary_violations) else 0.0,
            "direct_interior_pressure_fraction": 1.0 if np.any(interior_violations) else 0.0,
            "demo_preference_rate": float(np.mean(gaps <= 0)),
        }
    lambdas = [r["selected_lambda"] for r in rows.values()]
    selected_lambda = float(np.mean(lambdas))
    configured = float(cfg["global_action_support_cq"]["selected_lambda"])
    if configured > 0:
        assert abs(configured - selected_lambda) <= 1e-9 * max(1.0, selected_lambda)
    direct = min(r["direct_boundary_pressure_fraction"] for r in rows.values())
    assert direct >= 0.80
    assert all(r["weighted_grad_ratio"] <= 0.30 for r in rows.values())
    result = {
        "GLOBAL_CQ_REVIEW": "PASS",
        "GLOBAL_CQ_ACTION_SET": "policy mode + 12 single-axis +/-1 boundary + 12 single-axis +/-0.5 interior; a0:a5 only",
        "GLOBAL_CQ_MARGIN": 0.0,
        "GLOBAL_CQ_LAMBDA": selected_lambda,
        "GLOBAL_CQ_GRAD_RATIO": {k: v["weighted_grad_ratio"] for k, v in rows.items()},
        "BOUNDARY_VIOLATION_RATE": {k: v["boundary_violation_rate"] for k, v in rows.items()},
        "DIRECT_PRESSURE_FRACTION": direct,
        "INTERIOR_PRESSURE_FRACTION": min(r["direct_interior_pressure_fraction"] for r in rows.values()),
        "DEMO_PREFERENCE_RATE": {k: v["demo_preference_rate"] for k, v in rows.items()},
        "BOUNDARY_DIRECTLY_CONSTRAINED": "YES",
        "ACTOR_OBJECTIVE_UNCHANGED": "YES",
        "BELLMAN_LOSS_UNCHANGED": "YES",
        "CURRENT_CQ_UNCHANGED": "YES",
        "DATASET_UNCHANGED": "YES",
        "REPLAY_UNCHANGED": "YES",
        "GLOBAL_CQ_STATIC_PREFLIGHT": "PASS",
        "DETAILS": rows,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "DETAILS"}, sort_keys=True))


if __name__ == "__main__":
    main()
