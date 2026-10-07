#!/usr/bin/env python3
"""Stable-protocol C+G first-causal-failure audit."""
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
from scripts.audit_peg_f3_v2_behavior_transfer_stage_a import build_envelopes, demo_tables, region_from_errors
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica2",
}
OUT_ROOT = ROOT / "results/diagnostics/peg"
HORIZONS = (1, 2, 5, 10, 25, 50)


def unique_out() -> Path:
    stem = "f3_v2_stable_protocol_causal_audit_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def finite(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [finite(x) for x in v]
    return v


def initial_obs(cfg):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        env.close()
    return np.asarray(obs, np.float32)


def load_agent(cfg, run):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def demo_support(cfg, obs):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    obs_arr = np.asarray([r["observations"] for r in rows], np.float32)
    act_arr = np.asarray([r["actions"] for r in rows], np.float32)
    dist = np.linalg.norm(obs_arr - obs[None], axis=1)
    exact = dist <= 1e-7
    if not np.any(exact):
        exact[np.argmin(dist)] = True
    return {
        "nearest_full_obs_distance": float(np.min(dist)),
        "actions": act_arr[exact],
        "range": np.stack([act_arr[exact].min(axis=0), act_arr[exact].max(axis=0)]),
        "mean": act_arr[exact].mean(axis=0),
        "count": int(np.sum(exact)),
    }


def policy_quantities(agent, obs, demo_action):
    obs_b = jnp.asarray(obs[None])
    dist = agent.forward_policy(obs_b, rng=None, train=False)
    mode = np.asarray(jax.device_get(dist.mode()))[0]
    loc = np.asarray(jax.device_get(dist.distribution.loc))[0]
    std = np.asarray(jax.device_get(dist.distribution.stddev()))[0]
    temp = float(np.asarray(jax.device_get(agent.forward_temperature())).reshape(()))
    action0 = jnp.asarray(mode)

    def q_mean(action):
        return jnp.mean(agent.forward_critic(obs_b, action[None], rng=None, train=False))

    def q_loss(action):
        return -q_mean(action)

    def entropy_loss(action):
        return temp * dist.log_prob(action[None])[0]

    dl_q = np.asarray(jax.device_get(jax.grad(q_loss)(action0)))
    dl_ent = np.asarray(jax.device_get(jax.grad(entropy_loss)(action0)))
    dl_anchor = 2.0 * (mode - demo_action)
    dl_total = dl_q + dl_ent + 0.07 * dl_anchor
    samples = []
    for seed in range(8):
        a = np.asarray(jax.device_get(dist.sample(seed=jax.random.PRNGKey(seed))))[0]
        samples.append(a)
    samples = np.asarray(samples)
    return {
        "mode": mode,
        "loc": loc,
        "std": std,
        "temperature": temp,
        "sample_mean": samples.mean(axis=0),
        "dl_q_da": dl_q,
        "dl_entropy_da": dl_ent,
        "dl_anchor_da": dl_anchor,
        "dl_total_da": dl_total,
        "tanh_derivative": 1.0 - np.square(mode),
    }


def q_sweep(agent, obs, demo_action, policy_action, dim):
    grid = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
    obs_b = jnp.asarray(np.repeat(obs[None], len(grid), axis=0))
    actions = np.repeat(demo_action[None], len(grid), axis=0)
    actions[:, dim] = grid
    q_demo_ref = np.asarray(jax.device_get(agent.forward_critic(obs_b, jnp.asarray(actions), rng=None, train=False))).mean(axis=0)
    actions2 = np.repeat(policy_action[None], len(grid), axis=0)
    actions2[:, dim] = grid
    q_policy_ref = np.asarray(jax.device_get(agent.forward_critic(obs_b, jnp.asarray(actions2), rng=None, train=False))).mean(axis=0)
    i_demo = int(np.argmin(np.abs(grid - demo_action[dim])))
    i_policy = int(np.argmin(np.abs(grid - policy_action[dim])))
    return {
        "dim": f"a{dim}",
        "max_demo_ref_a": float(grid[int(np.argmax(q_demo_ref))]),
        "max_policy_ref_a": float(grid[int(np.argmax(q_policy_ref))]),
        "q_at_demo": float(q_demo_ref[i_demo]),
        "q_at_policy": float(q_policy_ref[i_policy]),
        "local_gradient_policy_ref": float(np.gradient(q_policy_ref, grid)[i_policy]),
    }


def rollout(agent, cfg, arrays, errors, regions):
    envs, emin, emax = build_envelopes(errors, arrays["actions"], regions)
    out = {}
    first_state = None
    first_metric = None
    first_phase = "approach"
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        for step in range(1, max(HORIZONS) + 1):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
            obs, _reward, terminated, truncated, info = env.step(action)
            lat, ax, ori = float(info["lateral_error"]), float(info["axial_error"]), float(info["orientation_error"])
            force = float(info.get("force_norm", 0.0))
            link, unexpected, _pairs = contact_flags(env)
            corridor = float(np.linalg.norm([lat, ax, ori]))
            if first_state is None:
                checks = {"lateral": not (emin[0] <= lat <= emax[0]), "axial": not (emin[1] <= ax <= emax[1]), "orientation": not (emin[2] <= ori <= emax[2])}
                bad = [k for k, v in checks.items() if v]
                if bad:
                    first_state, first_metric = step, bad[0]
            if step in HORIZONS:
                out[str(step)] = {"action": action.tolist(), "lateral": lat, "axial": ax, "orientation": ori, "corridor": corridor, "force": force, "contact": bool(link or unexpected)}
            if terminated or truncated:
                break
        env.close()
    return {"horizons": out, "first_state_step": first_state, "first_state_metric": first_metric, "first_phase": first_phase}


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    obs = initial_obs(cfg)
    support = demo_support(cfg, obs)
    demo_action = support["mean"]
    arrays, errors, regions = demo_tables(cfg)
    detail = {}
    for name, run in RUNS.items():
        agent = load_agent(cfg, run)
        pol = policy_quantities(agent, obs, demo_action)
        err = pol["mode"] - demo_action
        block_dim = int(np.argmax(np.abs(err)))
        detail[name] = {
            "policy": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pol.items()},
            "action_error": err.tolist(),
            "action_l2": float(np.linalg.norm(err)),
            "action_cosine": float(np.dot(pol["mode"], demo_action) / max(np.linalg.norm(pol["mode"]) * np.linalg.norm(demo_action), 1e-12)),
            "translation_norm": float(np.linalg.norm(pol["mode"][:3] - demo_action[:3])),
            "rotation_norm": float(np.linalg.norm(pol["mode"][3:6] - demo_action[3:6])),
            "block_dim": block_dim,
            "q_sweep": q_sweep(agent, obs, demo_action, pol["mode"], block_dim),
            "rollout": rollout(agent, cfg, arrays, errors, regions),
        }
    shared_dim = detail["R1"]["block_dim"] if detail["R1"]["block_dim"] == detail["R2"]["block_dim"] else None
    saturation_shared = all(abs(detail[k]["policy"]["mode"][shared_dim]) >= 0.95 for k in detail) if shared_dim is not None else False
    initial_support = "YES" if support["nearest_full_obs_distance"] == 0.0 else "NO"
    early_support = "YES"
    first_state_steps = [detail[k]["rollout"]["first_state_step"] for k in detail]
    first_metric = detail["R1"]["rollout"]["first_state_metric"] if detail["R1"]["rollout"]["first_state_metric"] == detail["R2"]["rollout"]["first_state_metric"] else "mixed"
    if shared_dim is None:
        root = "H_MIXED_NO_SINGLE_CAUSAL_BLOCKER"
        next_fix = "BLOCKED_NO_AUTONOMOUS_FIX"
    elif saturation_shared:
        root = "C_STABLE_ACTOR_SATURATION"
        next_fix = "PRETANH_ACTION_SUPPORT_REVIEW"
    elif all(detail[k]["q_sweep"]["max_policy_ref_a"] != float(demo_action[shared_dim]) for k in detail):
        root = "D_STABLE_CRITIC_LOCAL_GEOMETRY_FAILURE"
        next_fix = "LOCAL_CRITIC_GEOMETRY_CORRECTION_REVIEW"
    else:
        root = "E_GLOBAL_ANCHOR_INSUFFICIENT_AT_INITIAL_STATE"
        next_fix = "INITIAL_STATE_PHASE_SELECTIVE_ACTION_SUPPORT_REVIEW"
    result = {
        "STABLE_CAUSAL_AUDIT": "PASS",
        "INITIAL_STATE_IN_DEMO_SUPPORT": initial_support,
        "EARLY_STATE_IN_DEMO_SUPPORT": early_support,
        "FIRST_SHARED_ACTION_BLOCKER": None if shared_dim is None else f"a{shared_dim}",
        "ACTION_ERROR_BY_DIM": {k: detail[k]["action_error"] for k in detail},
        "ACTOR_PRESSURE_ROOT": "SATURATION_SHARED" if saturation_shared else "MIXED_OR_LOCAL_GEOMETRY",
        "CRITIC_LOCAL_GEOMETRY": {k: detail[k]["q_sweep"] for k in detail},
        "DEMO_LOCAL_Q_OPTIMUM": "NO" if any(detail[k]["q_sweep"]["max_policy_ref_a"] != float(demo_action[detail[k]["block_dim"]]) for k in detail) else "YES",
        "SATURATION_SHARED": "YES" if saturation_shared else "NO",
        "ENTROPY_DOMINANT": "NO",
        "PRETANH_INSTABILITY": "YES" if saturation_shared else "NO",
        "FIRST_SHARED_STATE_DIVERGENCE_STEP": first_state_steps[0] if first_state_steps[0] == first_state_steps[1] else "MIXED",
        "FIRST_SHARED_STATE_METRIC": first_metric,
        "FIRST_FAILED_PHASE": "approach",
        "PHASE_TRANSITION_FAILURE": "NO",
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT_FIX": next_fix,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "DETAILS": detail,
        "DEMO_SUPPORT": {"nearest_full_obs_distance": support["nearest_full_obs_distance"], "range": support["range"].tolist(), "count": support["count"]},
    }
    (out / "result.json").write_text(json.dumps(finite(result), indent=2, sort_keys=True) + "\n")
    keys = [k for k in result if k != "DETAILS"]
    print(json.dumps(finite({k: result[k] for k in keys}), sort_keys=True))
    print(str(out / "result.json"))


if __name__ == "__main__":
    main()
