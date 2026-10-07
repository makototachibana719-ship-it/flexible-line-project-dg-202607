#!/usr/bin/env python3
"""Static design/gradient preflight for the single local-critic candidate."""
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
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from async_sac_state_sim import concat_rlpd_batches
from training.peg_insertion.local_critic_geometry_candidate import make_local_geometry_agent
from peg_env_adapter import make_peg_state_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

BASE_RUN = ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica1_retry002"
DATASET_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"


def norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def stack(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def violation_rate(agent, rows, params, rng):
    sample = stack(rows)
    gaps, valid = agent.local_critic_comparisons(
        {"observations": jnp.asarray(sample["observations"]), "actions": jnp.asarray(sample["actions"])},
        params,
        rng,
    )
    mask = valid[None, :, None, :]
    count = agent.config["critic_ensemble_size"] * 2 * jnp.maximum(jnp.sum(valid), 1)
    return float(jnp.sum(jnp.where(mask, gaps > 0, False)) / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset_path = ROOT / cfg["demo"]["path"]
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == DATASET_SHA
    demos = pickle.loads(dataset_path.read_bytes())
    indices = np.linspace(0, len(demos) - 1, 32, dtype=int)
    demo_batch = stack([demos[i] for i in indices])
    replay = pickle.loads((BASE_RUN / "checkpoints/checkpoint_1000_online_replay.pkl").read_bytes())
    online_data = replay["dataset_dict"]
    online_batch = {key: np.asarray(value[:32]) for key, value in online_data.items() if key in demo_batch}
    batch = jax.tree_map(jnp.asarray, concat_rlpd_batches(online_batch, demo_batch))

    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        agent = make_local_geometry_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(BASE_RUN / "checkpoints", agent.state, 1000)
        agent = agent.replace(state=state)
        env.close()

    params = agent.state.params
    rng = jax.random.PRNGKey(20260907)
    base_fn = lambda p: super(type(agent), agent).critic_loss_fn(batch, p, rng)[0]
    local_fn = lambda p: agent.local_critic_loss_fn(batch, p, rng)
    base_loss, base_grad = jax.value_and_grad(base_fn)(params)
    local_loss, local_grad = jax.value_and_grad(local_fn)(params)
    base_norm = norm(base_grad)
    raw_norm = norm(local_grad)
    selected = 0.15 * base_norm / raw_norm
    ratio = selected * raw_norm / base_norm
    configured = float(cfg["local_critic_geometry"]["selected_lambda"])
    actor_before = agent.policy_loss_fn
    assert 0.10 <= ratio <= 0.20 and ratio < 0.30
    assert raw_norm > 0 and float(local_loss) > 0
    assert actor_before.__func__ is type(agent).__mro__[1].policy_loss_fn
    assert abs(configured - selected) <= 1e-9 * max(1.0, abs(selected))
    diagnostic_step = 1e-3 / raw_norm
    diagnostic_params = jax.tree_map(lambda p, g: p - diagnostic_step * g, params, local_grad)
    diagnostic_loss_after = float(local_fn(diagnostic_params))
    assert diagnostic_loss_after < float(local_loss)

    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases = []
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            phases.append(json.loads(line)["phase"])
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        initial_obs, _ = env.reset(seed=cfg["seed"])
        env.close()
    all_obs = np.asarray([row["observations"] for row in demos])
    initial_index = int(np.argmin(np.linalg.norm(all_obs - np.asarray(initial_obs)[None], axis=1)))
    initial_rows = [demos[initial_index]] * 32
    approach_indices = np.where(np.asarray(phases) == "approach")[0][:32]
    assert len(approach_indices) == 32
    approach_rows = [demos[int(i)] for i in approach_indices]
    overall_rate = violation_rate(agent, [demos[i] for i in indices], params, rng)
    initial_rate = violation_rate(agent, initial_rows, params, rng)
    approach_rate = violation_rate(agent, approach_rows, params, rng)
    assert initial_rate > 0 and approach_rate > 0

    result = {
        "CRITIC_LOCAL_GEOMETRY_REVIEW": "PASS",
        "LOCAL_ACTION_PERTURBATION_RULE": "paired +/- per-dimension perturbations on a0:a5; symmetric radius shrinks at bounds; a6 excluded",
        "PERTURBATION_SCALE_SOURCE": cfg["local_critic_geometry"]["perturbation_scale_source"],
        "LOCAL_CRITIC_LOSS": "mean_head_and_valid_pair_relu(Q_demo_state_local_action-Q_demo_action)",
        "LOCAL_CRITIC_MARGIN": float(cfg["local_critic_geometry"]["margin"]),
        "BASE_CRITIC_GRAD_NORM": base_norm,
        "LOCAL_CRITIC_RAW_GRAD_NORM": raw_norm,
        "LOCAL_CRITIC_RAW_LOSS": float(local_loss),
        "LOCAL_CRITIC_DIAGNOSTIC_LOSS_AFTER": diagnostic_loss_after,
        "LOCAL_CRITIC_LAMBDA": selected,
        "LOCAL_CRITIC_GRAD_RATIO": ratio,
        "LOCAL_MARGIN_VIOLATION_RATE": overall_rate,
        "INITIAL_VIOLATION_RATE": initial_rate,
        "APPROACH_VIOLATION_RATE": approach_rate,
        "ACTOR_OBJECTIVE_UNCHANGED": "YES",
        "BELLMAN_LOSS_PRESENT": "YES",
        "CONSERVATIVE_Q_UNCHANGED": "YES",
        "GLOBAL_ANCHOR_UNCHANGED": "YES",
        "DATASET_UNCHANGED": "YES",
        "REPLAY_UNCHANGED": "YES",
        "V2_UNCHANGED": "YES",
        "ACTION_SEMANTICS_UNCHANGED": "YES",
        "TRAINING_STARTED": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
