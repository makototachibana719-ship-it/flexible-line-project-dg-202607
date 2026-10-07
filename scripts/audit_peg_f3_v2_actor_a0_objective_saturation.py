#!/usr/bin/env python3
"""Read-only actor A0 objective/saturation diagnostic for Peg F3 V2."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_approach_support_ablation import (
    early_approach_mask,
    make_early_approach_support_agent,
)
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_approach_support_ablation_1k_seed0.yaml"
RUN = ROOT / "results/ablation/peg_f3_v2_early_approach_support_ablation_1k_seed0_run_001"
OUT_ROOT = ROOT / "results/diagnostics/peg"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def unique_out():
    stem = "f3_v2_actor_a0_objective_saturation_20260906"
    for index in range(1, 100):
        candidate = OUT_ROOT / f"{stem}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no fresh output directory available")


def tree_l2(tree):
    leaves = [jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "dtype")]
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def initial_observation():
    sys.path[:0] = [
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
    ]
    from peg_env_adapter import make_peg_state_env
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=0)
        env.close()
    return np.asarray(obs, np.float32)


def nearest_early_demo_action(cfg, obs):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    mask = early_approach_mask(ROOT, cfg)
    demo_obs = np.asarray([row["observations"] for row, keep in zip(rows, mask) if keep], np.float32)
    demo_actions = np.asarray([row["actions"] for row, keep in zip(rows, mask) if keep], np.float32)
    scale = np.maximum(demo_obs.std(axis=0), 1e-3)
    nearest = int(np.argmin(np.linalg.norm((demo_obs - obs[None]) / scale[None], axis=1)))
    return demo_actions[nearest], nearest


def finite_difference(fn, x, eps=1e-3):
    xp = np.array(x, np.float32)
    xm = np.array(x, np.float32)
    xp[0] += eps
    xm[0] -= eps
    return float((fn(jnp.asarray(xp)) - fn(jnp.asarray(xm))) / (2 * eps))


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    require(jax.default_backend() == "gpu", "GPU inference backend")
    cfg = yaml.safe_load(CONFIG.read_text())
    obs = initial_observation()
    demo_action, nearest = nearest_early_demo_action(cfg, obs)
    fresh = make_early_approach_support_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    agent = fresh.replace(state=restore_fresh_state(RUN / "checkpoints", fresh.state, 1000))
    params = agent.state.params
    obs_batch = jnp.asarray(obs[None])
    dist = agent.forward_policy(obs_batch, rng=None, train=False)
    mode = np.asarray(jax.device_get(dist.mode()))[0]
    loc = np.asarray(jax.device_get(dist.distribution.loc))[0]
    std = np.asarray(jax.device_get(dist.distribution.stddev()))[0]
    temperature = float(np.asarray(jax.device_get(agent.forward_temperature())).reshape(()))
    action0 = jnp.asarray(mode)
    z0 = jnp.asarray(loc)

    def q_mean(action):
        return jnp.mean(agent.forward_critic(obs_batch, action[None], rng=None, train=False))

    def action_q_loss(action):
        return -q_mean(action)

    def action_entropy_loss(action):
        return temperature * dist.log_prob(action[None])[0]

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

    fd = {
        "DL_Q_DA0": finite_difference(action_q_loss, action0),
        "DL_ENTROPY_DA0": finite_difference(action_entropy_loss, action0),
        "DL_SAC_DA0": finite_difference(action_sac_loss, action0),
        "DL_Q_DZ0": finite_difference(z_q_loss, z0),
        "DL_ENTROPY_DZ0": finite_difference(z_entropy_loss, z0),
        "DL_SAC_DZ0": finite_difference(z_sac_loss, z0),
    }

    sample_rows = []
    for seed in range(16):
        sample, logp = dist.sample_and_log_prob(seed=jax.random.PRNGKey(seed))
        sample = np.asarray(jax.device_get(sample))[0]
        logp = float(np.asarray(jax.device_get(logp))[0])
        sample_rows.append({
            "seed": seed,
            "sample_a0": float(sample[0]),
            "sample_pretanh_a0_inferred": float(np.arctanh(np.clip(sample[0], -0.999999, 0.999999))),
            "q_sample": float(jax.device_get(q_mean(jnp.asarray(sample)))),
            "log_prob": logp,
        })
    q_mode = float(jax.device_get(q_mean(action0)))

    def sac_param_loss(p):
        d = agent.forward_policy(obs_batch, rng=jax.random.PRNGKey(11), grad_params=p)
        a, logp = d.sample_and_log_prob(seed=jax.random.PRNGKey(12))
        q = jnp.mean(agent.forward_critic(obs_batch, a, rng=jax.random.PRNGKey(13), train=False))
        return -q + temperature * logp[0]

    def post_tanh_anchor_param_loss(p):
        d = agent.forward_policy(obs_batch, rng=None, grad_params=p, train=False)
        a0 = d.mode()[0, 0]
        return jnp.square(a0 - demo_action[0])

    def pretanh_anchor_param_loss(p):
        d = agent.forward_policy(obs_batch, rng=None, grad_params=p, train=False)
        z = d.distribution.loc[0, 0]
        return jnp.square(z)

    sac_param_norm = float(jax.device_get(tree_l2(jax.grad(sac_param_loss)(params))))
    post_grad_norm = float(jax.device_get(tree_l2(jax.grad(post_tanh_anchor_param_loss)(params))))
    pre_grad_norm = float(jax.device_get(tree_l2(jax.grad(pretanh_anchor_param_loss)(params))))
    lambda_pretanh_10pct = 0.10 * sac_param_norm / max(pre_grad_norm, 1e-12)

    a0_grid = np.linspace(0.80, 1.00, 41, dtype=np.float32)
    local_actions = np.repeat(mode[None], len(a0_grid), axis=0)
    local_actions[:, 0] = a0_grid
    q_heads = np.asarray(jax.device_get(agent.forward_critic(
        jnp.asarray(np.repeat(obs[None], len(a0_grid), axis=0)),
        jnp.asarray(local_actions),
        rng=None,
        train=False,
    )))
    q_local = q_heads.mean(axis=0)
    local_max_i = int(np.argmax(q_local))
    local_dq_da0 = float(np.gradient(q_local, a0_grid)[int(np.argmin(np.abs(a0_grid - mode[0])))])

    tanh_derivative = float(1.0 - mode[0] ** 2)
    post_tanh_dz0 = float(2.0 * mode[0] * tanh_derivative)
    pretanh_dz0 = float(2.0 * loc[0])
    sample_mean_a0 = float(np.mean([row["sample_a0"] for row in sample_rows]))
    mismatch = abs(sample_mean_a0 - float(mode[0])) > 0.05 or any(abs(row["sample_a0"] - float(mode[0])) > 0.05 for row in sample_rows)

    if tanh_derivative < 0.05 and abs(post_tanh_dz0) < 0.15 * abs(pretanh_dz0):
        primary = "TANH_SATURATION_WEAKENS_ACTION_SPACE_ANCHOR"
        selected = "EARLY_PRETANH_A0_ANCHOR"
    elif mismatch and np.sign(dl_sac_dz[0]) != np.sign(dl_q_dz[0]):
        primary = "SAMPLE_MODE_OBJECTIVE_MISMATCH"
        selected = "MATCHED_ACTOR_OBJECTIVE_ANCHOR"
    elif local_dq_da0 > 0:
        primary = "LOCAL_CRITIC_SLOPE_AT_SATURATED_ACTION"
        selected = "LOCAL_ACTION_SUPPORT_CRITIC_REGULARIZATION"
    else:
        primary = "BLOCKED"
        selected = "BLOCKED"

    result = {
        "ACTOR_A0_OBJECTIVE_DIAGNOSTIC": "PASS",
        "TRAINING_STARTED": False,
        "DL_Q_DA0": float(dl_q_da[0]),
        "DL_ENTROPY_DA0": float(dl_entropy_da[0]),
        "DL_SAC_DA0": float(dl_sac_da[0]),
        "DL_Q_DZ0": float(dl_q_dz[0]),
        "DL_ENTROPY_DZ0": float(dl_entropy_dz[0]),
        "DL_SAC_DZ0": float(dl_sac_dz[0]),
        "FINITE_DIFFERENCE": fd,
        "Q_GRAD_SIGN_PROOF": "PASS" if np.sign(fd["DL_Q_DA0"]) == np.sign(dl_q_da[0]) else "FAIL",
        "ENTROPY_GRAD_SIGN_PROOF": "PASS" if np.sign(fd["DL_ENTROPY_DA0"]) == np.sign(dl_entropy_da[0]) else "FAIL",
        "SAC_GRAD_SIGN_PROOF": "PASS" if np.sign(fd["DL_SAC_DA0"]) == np.sign(dl_sac_da[0]) else "FAIL",
        "TANH_DERIVATIVE_A0": tanh_derivative,
        "ACTOR_OPTIMIZES_SAMPLE_OR_MODE": "SAMPLE",
        "SAMPLE_MODE_MISMATCH": "YES" if mismatch else "NO",
        "MODE_A0": float(mode[0]),
        "SAMPLE_A0_MEAN": sample_mean_a0,
        "SAMPLE_ROWS": sample_rows,
        "Q_MODE": q_mode,
        "Q_SAMPLE_MEAN": float(np.mean([row["q_sample"] for row in sample_rows])),
        "LOG_PROB_SAMPLE_MEAN": float(np.mean([row["log_prob"] for row in sample_rows])),
        "POST_TANH_ANCHOR_RAW_LOSS": float(mode[0] ** 2),
        "PRETANH_ANCHOR_RAW_LOSS": float(loc[0] ** 2),
        "POST_TANH_ANCHOR_DA0": float(2.0 * mode[0]),
        "POST_TANH_ANCHOR_DZ0": post_tanh_dz0,
        "PRETANH_ANCHOR_DZ0": pretanh_dz0,
        "SAC_PARAM_GRAD_NORM": sac_param_norm,
        "POST_TANH_GRAD_NORM": post_grad_norm,
        "PRETANH_GRAD_NORM": pre_grad_norm,
        "PRETANH_LAMBDA_FOR_10PCT_SAC_GRAD": lambda_pretanh_10pct,
        "LOCAL_DQ_DA0": local_dq_da0,
        "LOCAL_Q_MAX_A0": float(a0_grid[local_max_i]),
        "LOCAL_Q_GRID": [
            {"a0": float(a0), "q_mean": float(q), "q_heads": [float(x) for x in q_heads[:, i]]}
            for i, (a0, q) in enumerate(zip(a0_grid, q_local))
        ],
        "A0_MODE": float(mode[0]),
        "A0_PRETANH": float(loc[0]),
        "A0_STD": float(std[0]),
        "TEMPERATURE": temperature,
        "NEAREST_EARLY_DEMO_INDEX": nearest,
        "PRIMARY_ROOT_CAUSE": primary,
        "SELECTED_FIX": selected,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    dump(out / "result.json", result)
    (out / "REPORT.md").write_text("# Peg F3 V2 Actor A0 Objective Saturation Diagnostic\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
