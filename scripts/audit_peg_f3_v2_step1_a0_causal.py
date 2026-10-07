#!/usr/bin/env python3
"""Read-only Step-1 A0 causal audit for Peg F3 V2 early-support 1K."""
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

from scripts.screen_peg_f3_v2_formal_5k import raw_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_approach_support_ablation import (
    early_approach_mask,
    make_early_approach_support_agent,
)
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_approach_support_ablation_1k_seed0.yaml"
RUN = ROOT / "results/ablation/peg_f3_v2_early_approach_support_ablation_1k_seed0_run_001"
OUT_ROOT = ROOT / "results/diagnostics/peg"
A0_SWEEP = np.asarray([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], np.float32)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def unique_out():
    stem = "f3_v2_step1_a0_causal_audit_20260906"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no fresh output directory available")


def demo_early_tables(cfg):
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    mask = early_approach_mask(ROOT, cfg)
    obs = np.asarray([row["observations"] for row, keep in zip(rows, mask) if keep], np.float32)
    actions = np.asarray([row["actions"] for row, keep in zip(rows, mask) if keep], np.float32)
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, metrics = [], []
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            row = json.loads(line)
            phases.append(row["phase"])
            metrics.append(row["before"]["metrics"])
    predicate = cfg["early_approach_support"]["predicate"]
    m = np.asarray([[x["lateral_error"], x["axial_error"], x["orientation_error"]] for x in metrics], np.float64)
    metric_mask = (
        np.isin(np.asarray(phases), predicate["phases"])
        & (m[:, 0] >= float(predicate["lateral_min"]))
        & (m[:, 1] >= float(predicate["axial_min"]))
        & (m[:, 1] <= float(predicate["axial_max"]))
        & (m[:, 2] <= float(predicate["orientation_max"]))
    )
    return obs, actions, m[metric_mask]


def metric_from_env(env):
    raw = raw_env(env)
    obs_dict = raw._get_obs()
    return raw.task._goal_metrics(obs_dict)


def support_metrics(initial_obs, initial_metric, demo_obs, demo_metrics):
    obs_std = np.maximum(demo_obs.std(axis=0), 1e-3)
    scaled = (demo_obs - initial_obs[None, :]) / obs_std[None, :]
    nearest = int(np.argmin(np.linalg.norm(scaled, axis=1)))
    obs_in_range = bool(np.all(initial_obs >= demo_obs.min(axis=0)) and np.all(initial_obs <= demo_obs.max(axis=0)))
    metric = np.asarray([
        initial_metric["lateral_error"],
        initial_metric["axial_error"],
        initial_metric["orientation_error"],
    ], np.float64)
    metric_min = demo_metrics.min(axis=0)
    metric_max = demo_metrics.max(axis=0)
    metric_in = (metric >= metric_min) & (metric <= metric_max)
    return {
        "lateral": {"value": float(metric[0]), "demo_min": float(metric_min[0]),
                    "demo_max": float(metric_max[0]), "in_range": bool(metric_in[0])},
        "axial": {"value": float(metric[1]), "demo_min": float(metric_min[1]),
                  "demo_max": float(metric_max[1]), "in_range": bool(metric_in[1])},
        "orientation": {"value": float(metric[2]), "demo_min": float(metric_min[2]),
                        "demo_max": float(metric_max[2]), "in_range": bool(metric_in[2])},
        "full_obs_distance": float(np.linalg.norm(scaled[nearest])),
        "nearest_early_demo_index": nearest,
        "full_obs_in_component_envelope": obs_in_range,
        "initial_state_in_demo_support": bool(obs_in_range and np.all(metric_in)),
    }


def q_values(agent, obs, actions):
    q = np.asarray(jax.device_get(agent.forward_critic(
        jnp.asarray(np.repeat(obs[None], len(actions), axis=0)),
        jnp.asarray(actions),
        rng=None,
        train=False,
    )))
    return q, q.mean(axis=0)


def distribution_report(agent, obs):
    dist = agent.forward_policy(jnp.asarray(obs[None]), rng=None, train=False)
    mode = np.asarray(jax.device_get(dist.mode()))[0]
    sample = np.asarray(jax.device_get(dist.sample(seed=jax.random.PRNGKey(7))))[0]
    report = {"type": type(dist).__name__, "mode": mode.tolist(), "sample": sample.tolist()}
    base = getattr(dist, "distribution", None)
    if base is not None:
        try:
            mean = np.asarray(jax.device_get(base.mean()))[0]
            report["pre_tanh_mean"] = mean.tolist()
            report["a0_pretanh"] = float(mean[0])
        except Exception as exc:
            report["pre_tanh_mean_error"] = repr(exc)
        try:
            std = np.asarray(jax.device_get(base.stddev()))[0]
            report["std"] = std.tolist()
            report["log_std"] = np.log(np.maximum(std, 1e-12)).tolist()
        except Exception as exc:
            report["std_error"] = repr(exc)
    return report


def action_space_gradients(agent, obs, policy_action, demo_action, cfg):
    obs_batch = jnp.asarray(obs[None])
    policy = jnp.asarray(policy_action)
    demo = jnp.asarray(demo_action)

    def neg_q_mean(action):
        q = agent.forward_critic(obs_batch, action[None], rng=None, train=False)
        return -jnp.mean(q)

    def global_anchor(action):
        return cfg["demo_actor_anchor"]["selected_lambda"] * jnp.mean(jnp.square(action - demo))

    def early_anchor(action):
        return cfg["early_approach_support"]["selected_lambda"] * jnp.square(action[0] - demo[0])

    sac = np.asarray(jax.device_get(jax.grad(neg_q_mean)(policy)))
    global_g = np.asarray(jax.device_get(jax.grad(global_anchor)(policy)))
    early_g = np.asarray(jax.device_get(jax.grad(early_anchor)(policy)))
    return {
        "sac_actor_action_gradient_a0": float(sac[0]),
        "global_demo_anchor_action_gradient_a0": float(global_g[0]),
        "early_support_action_gradient_a0": float(early_g[0]),
        "sac_actor_action_gradient_norm": float(np.linalg.norm(sac)),
        "global_demo_anchor_action_gradient_norm": float(np.linalg.norm(global_g)),
        "early_support_action_gradient_norm": float(np.linalg.norm(early_g)),
    }


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    require(jax.default_backend() == "gpu", "GPU inference backend")
    cfg = yaml.safe_load(CONFIG.read_text())
    demo_obs, demo_actions, demo_metrics = demo_early_tables(cfg)
    require(len(demo_obs) == 363, "early Demo row count")
    require(float(demo_actions[:, 0].min()) == 0.0 and float(demo_actions[:, 0].max()) == 0.0,
            "early Demo a0 range")
    sys.path[:0] = [
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
    ]
    from peg_env_adapter import make_peg_state_env
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        initial_obs, _ = env.reset(seed=0)
        initial_metric = metric_from_env(env)
        env.close()
    support = support_metrics(np.asarray(initial_obs, np.float32), initial_metric, demo_obs, demo_metrics)
    nearest_action = demo_actions[support["nearest_early_demo_index"]]
    fresh = make_early_approach_support_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    agent = fresh.replace(state=restore_fresh_state(RUN / "checkpoints", fresh.state, 1000))
    dist = distribution_report(agent, np.asarray(initial_obs, np.float32))
    policy_action = np.asarray(dist["mode"], np.float32)
    sweep_actions = np.repeat(policy_action[None], len(A0_SWEEP), axis=0)
    sweep_actions[:, 0] = A0_SWEEP
    q_heads, q_mean = q_values(agent, np.asarray(initial_obs, np.float32), sweep_actions)
    policy_q_heads, policy_q = q_values(agent, np.asarray(initial_obs, np.float32), policy_action[None])
    demo_q_heads, demo_q = q_values(agent, np.asarray(initial_obs, np.float32), nearest_action[None])
    zero_index = int(np.where(A0_SWEEP == 0.0)[0][0])
    argmax_index = int(np.argmax(q_mean))
    grads = action_space_gradients(agent, np.asarray(initial_obs, np.float32),
                                   policy_action, nearest_action, cfg)
    temperature = float(np.asarray(jax.device_get(agent.forward_temperature())).reshape(()))
    q_prefers_positive = float(A0_SWEEP[argmax_index]) >= 0.75
    if not support["initial_state_in_demo_support"]:
        root_cause = "RESET_TO_DEMO_STATE_SUPPORT_MISMATCH"
        next_fix = "STATE_SUPPORT_DIAGNOSTIC"
    elif q_prefers_positive:
        root_cause = "INITIAL_STATE_ACTION_Q_OOD"
        next_fix = "ONLINE_STATE_ACTION_SUPPORT_REGULARIZATION"
    else:
        root_cause = "ACTOR_OPTIMIZATION_OR_SATURATION"
        next_fix = "ACTOR_A0_OBJECTIVE_DIAGNOSTIC"
    early_effect = "OPPOSED_BY_Q" if q_prefers_positive and grads["sac_actor_action_gradient_a0"] < 0 else "OTHER"
    result = {
        "STEP1_A0_CAUSAL_AUDIT": "PASS",
        "TRAINING_STARTED": False,
        "INITIAL_STATE_SUPPORT": support,
        "INITIAL_STATE_IN_DEMO_SUPPORT": "YES" if support["initial_state_in_demo_support"] else "NO",
        "A0_SWEEP": [
            {"a0": float(a0), "q_mean": float(mean),
             "q_heads": [float(x) for x in q_heads[:, i]]}
            for i, (a0, mean) in enumerate(zip(A0_SWEEP, q_mean))
        ],
        "A0_Q_ARGMAX": float(A0_SWEEP[argmax_index]),
        "Q_A0_ZERO": float(q_mean[zero_index]),
        "Q_A0_POLICY": float(policy_q[0]),
        "Q_DEMO_ACTION": float(demo_q[0]),
        "Q_POLICY_MINUS_ZERO": float(policy_q[0] - q_mean[zero_index]),
        "Q_POLICY_HEADS": [float(x) for x in policy_q_heads[:, 0]],
        "Q_DEMO_HEADS": [float(x) for x in demo_q_heads[:, 0]],
        "POLICY_DISTRIBUTION": dist,
        "A0_MODE": float(policy_action[0]),
        "A0_PRETANH": dist.get("a0_pretanh", "NOT_AVAILABLE"),
        "A0_SAMPLED": float(dist["sample"][0]),
        "A0_SATURATED": "YES" if abs(float(policy_action[0])) >= 0.95 else "NO",
        "A0_LOG_STD": None if "log_std" not in dist else float(dist["log_std"][0]),
        "A0_STD": None if "std" not in dist else float(dist["std"][0]),
        "TEMPERATURE": temperature,
        "LOSS_GRAD_CONTRIBUTION": grads,
        "CONSERVATIVE_Q_INFLUENCE_RELEVANT_TO_THIS_STATE": "INDIRECT_CRITIC_ONLY_NO_ACTOR_TERM",
        "EARLY_SUPPORT_EFFECT": early_effect,
        "PRIMARY_ROOT_CAUSE": root_cause,
        "NEXT_FIX": next_fix,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    dump(out / "result.json", result)
    (out / "REPORT.md").write_text("# Peg F3 V2 Step-1 A0 Causal Audit\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
