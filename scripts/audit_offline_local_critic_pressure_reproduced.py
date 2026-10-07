#!/usr/bin/env python3
"""Offline local critic-pressure audit for reproduced saturated pre-tanh lineage."""
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
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_pretanh_a0_anchor_ablation import make_early_pretanh_a0_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_pretanh_a0_anchor_ablation_1k_seed0.yaml"
BASE = ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002"
RUNS = {
    "R1": BASE / "replica1",
    "R2": BASE / "replica2",
}
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out():
    stem = "f3_v2_offline_local_critic_pressure_reproduced_20260906"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def initial_observation(cfg):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        env.close()
    obs = np.asarray(obs, np.float32)
    require(obs.shape == (20,), "initial observation shape")
    return obs


def demo_initial_action(cfg, obs):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    demo_obs = np.asarray([row["observations"] for row in rows], np.float32)
    demo_actions = np.asarray([row["actions"] for row in rows], np.float32)
    distances = np.linalg.norm(demo_obs - obs[None], axis=1)
    idx = int(np.argmin(distances))
    require(float(distances[idx]) == 0.0, "initial state not exact demo support")
    return demo_actions[idx], idx


def scalar(x):
    return float(np.asarray(jax.device_get(x)).reshape(()))


def audit_checkpoint(cfg, run_path: Path, step: int, obs: np.ndarray, demo_action: np.ndarray):
    require((run_path / "checkpoints" / f"checkpoint_{step}").exists(), f"checkpoint {step} missing")
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_early_pretanh_a0_anchor_agent(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run_path / "checkpoints", fresh.state, step)
        agent = fresh.replace(state=state)
        env.close()

    obs_batch = jnp.asarray(obs[None])
    dist = agent.forward_policy(obs_batch, rng=None, train=False)
    mode = np.asarray(jax.device_get(dist.mode()))[0]
    loc = np.asarray(jax.device_get(dist.distribution.loc))[0]
    std = np.asarray(jax.device_get(dist.distribution.stddev()))[0]
    temperature = scalar(agent.forward_temperature())

    def q_heads(action):
        values = np.asarray(jax.device_get(agent.forward_critic(
            obs_batch, jnp.asarray(action[None], dtype=jnp.float32), rng=None, train=False
        )))
        require(values.shape[0] == 10 and np.isfinite(values).all(), "critic heads")
        return values[:, 0]

    grid = np.round(np.arange(0.80, 1.0001, 0.005), 6).astype(np.float32)
    actions = np.repeat(mode[None], len(grid), axis=0)
    actions[:, 0] = grid
    q_grid_heads = np.asarray(jax.device_get(agent.forward_critic(
        jnp.asarray(np.repeat(obs[None], len(grid), axis=0)),
        jnp.asarray(actions),
        rng=None,
        train=False,
    )))
    q_grid = q_grid_heads.mean(axis=0)
    mode_i = int(np.argmin(np.abs(grid - mode[0])))
    local_dq = float(np.gradient(q_grid, grid)[mode_i])
    argmax = float(grid[int(np.argmax(q_grid))])
    monotonic_nondec = bool(np.all(np.diff(q_grid) >= -1e-5))

    action_zero = np.array(mode, copy=True)
    action_zero[0] = 0.0
    q_zero = float(q_heads(action_zero).mean())
    q_policy = float(q_heads(mode).mean())
    q_demo = float(q_heads(demo_action).mean())

    action0 = jnp.asarray(mode)
    z0 = jnp.asarray(loc)

    def q_mean(action):
        return jnp.mean(agent.forward_critic(obs_batch, action[None], rng=None, train=False))

    def action_q_loss(action):
        return -q_mean(action)

    def action_entropy_loss(action):
        return agent.forward_temperature() * dist.log_prob(action[None])[0]

    def action_sac_loss(action):
        return action_q_loss(action) + action_entropy_loss(action)

    def z_q_loss(z):
        return action_q_loss(jnp.tanh(z))

    def z_entropy_loss(z):
        return action_entropy_loss(jnp.tanh(z))

    def z_sac_loss(z):
        return z_q_loss(z) + z_entropy_loss(z)

    dl_q_da = np.asarray(jax.device_get(jax.grad(action_q_loss)(action0)))
    dl_entropy_da = np.asarray(jax.device_get(jax.grad(action_entropy_loss)(action0)))
    dl_sac_da = np.asarray(jax.device_get(jax.grad(action_sac_loss)(action0)))
    dl_q_dz = np.asarray(jax.device_get(jax.grad(z_q_loss)(z0)))
    dl_entropy_dz = np.asarray(jax.device_get(jax.grad(z_entropy_loss)(z0)))
    dl_sac_dz = np.asarray(jax.device_get(jax.grad(z_sac_loss)(z0)))

    return {
        "checkpoint": step,
        "A0_MODE": float(mode[0]),
        "A0_PRETANH": float(loc[0]),
        "A0_STD": float(std[0]),
        "TEMPERATURE": temperature,
        "LOCAL_DQ_DA0_NEAR_MODE": local_dq,
        "LOCAL_Q_ARGMAX_A0": argmax,
        "LOCAL_Q_MONOTONIC_NONDECREASING": monotonic_nondec,
        "Q_A0_ZERO": q_zero,
        "Q_A0_POLICY": q_policy,
        "Q_DEMO_FULL_ACTION": q_demo,
        "Q_POLICY_MINUS_ZERO": q_policy - q_zero,
        "DL_Q_DA0": float(dl_q_da[0]),
        "DL_ENTROPY_DA0": float(dl_entropy_da[0]),
        "DL_SAC_DA0": float(dl_sac_da[0]),
        "DL_Q_DZ0": float(dl_q_dz[0]),
        "DL_ENTROPY_DZ0": float(dl_entropy_dz[0]),
        "DL_SAC_DZ0": float(dl_sac_dz[0]),
        "Q_GRID": [
            {"a0": float(a0), "q_mean": float(q), "q_heads": [float(x) for x in q_grid_heads[:, i]]}
            for i, (a0, q) in enumerate(zip(grid, q_grid))
        ],
    }


def trend(a, b, eps=1e-3):
    if b > a + eps:
        return "INCREASING"
    if b < a - eps:
        return "DECREASING"
    return "STABLE"


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    require(jax.default_backend() == "gpu", "gpu backend")
    obs = initial_observation(cfg)
    demo_action, demo_idx = demo_initial_action(cfg, obs)
    require(float(demo_action[0]) == 0.0, "demo target a0")

    rows = {}
    for replica, run in RUNS.items():
        rows[replica] = {
            step: audit_checkpoint(cfg, run, step, obs, demo_action)
            for step in (500, 1000)
        }

    r1_slope = trend(rows["R1"][500]["LOCAL_DQ_DA0_NEAR_MODE"], rows["R1"][1000]["LOCAL_DQ_DA0_NEAR_MODE"])
    r2_slope = trend(rows["R2"][500]["LOCAL_DQ_DA0_NEAR_MODE"], rows["R2"][1000]["LOCAL_DQ_DA0_NEAR_MODE"])
    slopes_1000 = [rows["R1"][1000]["LOCAL_DQ_DA0_NEAR_MODE"], rows["R2"][1000]["LOCAL_DQ_DA0_NEAR_MODE"]]
    q_pressure = [rows["R1"][1000]["DL_Q_DA0"], rows["R2"][1000]["DL_Q_DA0"]]

    if all(s > 0.05 for s in slopes_1000) and all(g < -0.01 for g in q_pressure):
        primary = "LOCAL_CRITIC_PRESSURE_AT_SATURATION"
        next_fix = "LOCAL_ACTION_SUPPORT_CRITIC_REGULARIZATION"
    elif all(abs(s) <= 0.05 for s in slopes_1000) and any(rows[r][1000]["DL_SAC_DA0"] < -0.01 for r in rows):
        primary = "ACTOR_ENTROPY_OR_OBJECTIVE_PRESSURE"
        next_fix = "ACTOR_OBJECTIVE_BALANCE"
    elif np.sign(slopes_1000[0]) != np.sign(slopes_1000[1]):
        primary = "ASYNC_CRITIC_GEOMETRY_VARIANCE"
        next_fix = "REPRODUCIBILITY_AWARE_MULTI_REPLICA_PROTOCOL"
    else:
        primary = "BLOCKED"
        next_fix = "BLOCKED"

    result = {
        "LOCAL_CRITIC_PRESSURE_AUDIT": "PASS" if primary != "BLOCKED" else "FAIL",
        "INITIAL_STATE_IN_DEMO_SUPPORT": "YES",
        "FULL_OBS_DISTANCE": 0.0,
        "DEMO_TARGET_A0": float(demo_action[0]),
        "DEMO_NEAREST_INDEX": demo_idx,
        "R1_A0_500": rows["R1"][500]["A0_MODE"],
        "R1_A0_1000": rows["R1"][1000]["A0_MODE"],
        "R2_A0_500": rows["R2"][500]["A0_MODE"],
        "R2_A0_1000": rows["R2"][1000]["A0_MODE"],
        "LOCAL_DQ_DA0_R1_500": rows["R1"][500]["LOCAL_DQ_DA0_NEAR_MODE"],
        "LOCAL_DQ_DA0_R1_1000": rows["R1"][1000]["LOCAL_DQ_DA0_NEAR_MODE"],
        "LOCAL_DQ_DA0_R2_500": rows["R2"][500]["LOCAL_DQ_DA0_NEAR_MODE"],
        "LOCAL_DQ_DA0_R2_1000": rows["R2"][1000]["LOCAL_DQ_DA0_NEAR_MODE"],
        "Q_POLICY_MINUS_ZERO_R1_1000": rows["R1"][1000]["Q_POLICY_MINUS_ZERO"],
        "Q_POLICY_MINUS_ZERO_R2_1000": rows["R2"][1000]["Q_POLICY_MINUS_ZERO"],
        "DL_Q_DA0": {"R1_1000": rows["R1"][1000]["DL_Q_DA0"], "R2_1000": rows["R2"][1000]["DL_Q_DA0"]},
        "DL_ENTROPY_DA0": {"R1_1000": rows["R1"][1000]["DL_ENTROPY_DA0"], "R2_1000": rows["R2"][1000]["DL_ENTROPY_DA0"]},
        "DL_SAC_DA0": {"R1_1000": rows["R1"][1000]["DL_SAC_DA0"], "R2_1000": rows["R2"][1000]["DL_SAC_DA0"]},
        "DL_Q_DZ0": {"R1_1000": rows["R1"][1000]["DL_Q_DZ0"], "R2_1000": rows["R2"][1000]["DL_Q_DZ0"]},
        "DL_ENTROPY_DZ0": {"R1_1000": rows["R1"][1000]["DL_ENTROPY_DZ0"], "R2_1000": rows["R2"][1000]["DL_ENTROPY_DZ0"]},
        "DL_SAC_DZ0": {"R1_1000": rows["R1"][1000]["DL_SAC_DZ0"], "R2_1000": rows["R2"][1000]["DL_SAC_DZ0"]},
        "CRITIC_PRESSURE_TREND": {"R1": r1_slope, "R2": r2_slope},
        "PRIMARY_ROOT_CAUSE": primary,
        "NEXT_FIX": next_fix,
        "NEW_TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": rows,
    }
    dump(out / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
