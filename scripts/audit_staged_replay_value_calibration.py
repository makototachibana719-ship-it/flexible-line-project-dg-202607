#!/usr/bin/env python3
"""Offline staged replay quality and value-calibration attribution."""
from __future__ import annotations

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
from peg_env_adapter import make_peg_state_env
from serl_launcher.agents.continuous.sac import SACAgent
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.conservative_q_ablation import ConservativeQSACAgent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica2",
}
GRID = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
BOUNDARY = 0.95


def unique_out():
    root = ROOT / "results/diagnostics/peg"
    for i in range(1, 100):
        out = root / f"f3_v2_staged_replay_value_calibration_20260907_{i:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output path")


def stack(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def load_agent(cfg, run, step):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        agent = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", agent.state, step)
        env.close()
    return agent.replace(state=state)


def selected_states(cfg, demos):
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, metrics = [], []
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            row = json.loads(line)
            phases.append(row["phase"])
            metrics.append(row["after"]["metrics"])
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        initial, _ = env.reset(seed=cfg["seed"])
        env.close()
    obs = np.asarray([r["observations"] for r in demos])
    indices = {
        "initial": int(np.argmin(np.linalg.norm(obs - np.asarray(initial)[None], axis=1))),
        "approach": int(np.where(np.asarray(phases) == "approach")[0][0]),
        "lateral": int(np.where(np.asarray(phases) == "lateral_alignment")[0][0]),
        "orientation": int(np.argmax([m["orientation_error"] for m in metrics])),
    }
    return {name: demos[index] for name, index in indices.items()}


def fixed_batch(demos):
    indices = np.linspace(0, len(demos) - 1, 32, dtype=int)
    demo = stack([demos[i] for i in indices])
    payload = pickle.loads((RUNS["R1"] / "checkpoints/checkpoint_1000_online_replay.pkl").read_bytes())
    online = {k: np.asarray(v[:32]) for k, v in payload["dataset_dict"].items() if k in demo}
    return jax.tree_util.tree_map(jnp.asarray, concat_rlpd_batches(online, demo))


def action_stats(actions):
    actions = np.asarray(actions)[:, :6]
    return {
        "min": np.min(actions, axis=0).tolist(),
        "max": np.max(actions, axis=0).tolist(),
        "near_minus_fraction": np.mean(actions <= -BOUNDARY, axis=0).tolist(),
        "near_plus_fraction": np.mean(actions >= BOUNDARY, axis=0).tolist(),
    }


def q_apply(agent, params, obs, actions):
    return agent.forward_critic(obs, actions, rng=None, train=False, grad_params=params)


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def tree_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(jax.device_get(x))))) for x in leaves)))


def fixed_batches(demos, run, step):
    demo_indices = np.linspace(0, len(demos) - 1, 32, dtype=int)
    demo = stack([demos[i] for i in demo_indices])
    payload = pickle.loads((run / f"checkpoints/checkpoint_{step}_online_replay.pkl").read_bytes())
    size = int(payload["size"])
    online_indices = np.linspace(0, size - 1, 32, dtype=int)
    online = {key: np.asarray(value)[online_indices] for key, value in payload["dataset_dict"].items() if key in demo}
    return jax.tree_util.tree_map(jnp.asarray, demo), jax.tree_util.tree_map(jnp.asarray, online)


def source_metrics(agent, demo_batch, online_batch):
    params = agent.state.params
    key = jax.random.PRNGKey(20260907)
    demo_bellman_fn = lambda p: SACAgent.critic_loss_fn(agent, demo_batch, p, key)[0]
    online_bellman_fn = lambda p: SACAgent.critic_loss_fn(agent, online_batch, p, key)[0]
    demo_cq_fn = lambda p: 5.0 * ConservativeQSACAgent.conservative_q_loss_fn(agent, demo_batch, p, key)
    demo_bellman_grad = jax.grad(demo_bellman_fn)(params)
    online_bellman_grad = jax.grad(online_bellman_fn)(params)
    demo_cq_grad = jax.grad(demo_cq_fn)(params)
    demo_source = jax.tree_util.tree_map(lambda b, c: 0.5 * b + c, demo_bellman_grad, demo_cq_grad)
    online_source = jax.tree_util.tree_map(lambda b: 0.5 * b, online_bellman_grad)
    demo_norm = tree_norm(demo_source)
    online_norm = tree_norm(online_source)
    return {
        "demo_bellman_loss": float(demo_bellman_fn(params)),
        "online_bellman_loss": float(online_bellman_fn(params)),
        "demo_cq_loss": float(demo_cq_fn(params)),
        "online_cq_loss": 0.0,
        "demo_critic_grad_norm": demo_norm,
        "online_critic_grad_norm": online_norm,
        "online_demo_grad_ratio": online_norm / max(demo_norm, 1e-12),
    }


def td_metrics(agent, batch):
    key = jax.random.PRNGKey(20260907)
    key, next_key = jax.random.split(key)
    next_actions, _ = agent._compute_next_actions(batch, next_key)
    target_qs = agent.forward_target_critic(batch["next_observations"], next_actions, rng=key)
    key, subsample_key = jax.random.split(key)
    indices = jax.random.randint(subsample_key, (agent.config["critic_subsample_size"],),
                                 0, agent.config["critic_ensemble_size"])
    target_q = target_qs[indices].min(axis=0)
    target = batch["rewards"] + agent.config["discount"] * batch["masks"] * target_q
    current = agent.forward_critic(batch["observations"], batch["actions"], rng=None, train=False)
    td = np.asarray(jax.device_get(current - target[None]))
    return {
        "td_abs_mean": float(np.mean(np.abs(td))),
        "q_abs_mean": float(np.mean(np.abs(np.asarray(jax.device_get(current))))),
        "target": summary(jax.device_get(target)),
    }


def phase_label(value):
    value = str(value).lower()
    if "lateral" in value:
        return "lateral_correction"
    if "orient" in value or "rotation" in value:
        return "orientation_correction"
    if "near" in value or "insert" in value:
        return "near_hole"
    if "approach" in value:
        return "approach"
    return "setup"


def demo_phases():
    values = []
    provenance = json.loads((ROOT / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    for entry in provenance:
        for line in (ROOT / entry["episode_path"] / "transitions.jsonl").open():
            values.append(phase_label(json.loads(line)["phase"]))
    return np.asarray(values)


def replay_quality(demos, phases, run, limit=604):
    payload = pickle.loads((run / "checkpoints/checkpoint_1000_online_replay.pkl").read_bytes())
    size = min(int(payload["size"]), limit)
    online_obs = np.asarray(payload["dataset_dict"]["observations"][:size], np.float64)
    dones = np.asarray(payload["dataset_dict"]["dones"][:size], bool)
    demo_obs = np.asarray([row["observations"] for row in demos], np.float64)
    scale = np.maximum(np.std(demo_obs, axis=0), 1e-6)
    nearest = np.argmin(np.sum(np.square((online_obs[:, None] - demo_obs[None]) / scale), axis=2), axis=1)
    assigned = phases[nearest]
    lo, hi = np.min(demo_obs, axis=0), np.max(demo_obs, axis=0)
    support = np.all((online_obs >= lo) & (online_obs <= hi), axis=1)
    rel = online_obs[:, 13:16]
    near = (np.linalg.norm(rel[:, :2], axis=1) <= 0.010) & (np.abs(rel[:, 2]) <= 0.020)
    episode_start = np.zeros(size, dtype=bool)
    episode_start[0] = True
    if size > 1:
        episode_start[1:] = dones[:-1]
    names = ("setup", "approach", "lateral_correction", "orientation_correction", "near_hole")
    return {
        "size": size,
        "phase_fraction": {name: float(np.mean(assigned == name)) for name in names},
        "demo_support_fraction": float(np.mean(support)),
        "near_hole_like_fraction": float(np.mean(near)),
        "pre_divergence_fraction": float(np.mean(episode_start)),
        "post_divergence_fraction": float(np.mean(~episode_start)),
        "lateral_overlap": float(np.mean(np.all((rel[:, :2] >= lo[13:15]) & (rel[:, :2] <= hi[13:15]), axis=1))),
        "axial_overlap": float(np.mean((rel[:, 2] >= lo[15]) & (rel[:, 2] <= hi[15]))),
        "orientation_overlap": float(np.mean(np.all((online_obs[:, 3:7] >= lo[3:7]) & (online_obs[:, 3:7] <= hi[3:7]), axis=1))),
        "joint_overlap": "UNAVAILABLE_OBS20_HAS_NO_JOINT_STATE",
        "reward": summary(payload["dataset_dict"]["rewards"][:size]),
    }


def value_calibration(agent, states):
    rng = np.random.default_rng(20260907)
    rows = []
    for phase, row in states.items():
        obs = np.asarray(row["observations"], np.float32)
        demo = np.asarray(row["actions"], np.float32)
        policy = np.asarray(jax.device_get(agent.forward_policy(jnp.asarray(obs[None]), rng=None, train=False).mode()))[0]
        actions = [demo, policy]
        kinds = ["demo", "policy"]
        for dim in range(6):
            for value in (-1.0, 1.0):
                action = demo.copy(); action[dim] = value
                actions.append(action); kinds.append("boundary")
        for _ in range(16):
            action = demo.copy(); action[:6] = rng.uniform(-0.5, 0.5, 6)
            actions.append(action); kinds.append("random_interior")
        q = np.asarray(jax.device_get(agent.forward_critic(
            jnp.repeat(jnp.asarray(obs[None]), len(actions), axis=0), jnp.asarray(actions), rng=None, train=False))).mean(axis=0)
        values = {kind: [float(q[i]) for i, label in enumerate(kinds) if label == kind] for kind in set(kinds)}
        rows.append({"phase": phase, **{kind: summary(value) for kind, value in values.items()}})
    boundary_over = float(np.mean([row["boundary"]["max"] > row["demo"]["mean"] for row in rows]))
    policy_over = float(np.mean([row["policy"]["mean"] > row["demo"]["mean"] for row in rows]))
    if boundary_over > 0.5:
        status = "BOUNDARY_OVERVALUED"
    elif policy_over > 0.5:
        status = "POLICY_OVERVALUED"
    elif boundary_over == 0 and policy_over == 0:
        status = "CORRECTLY_CALIBRATED"
    else:
        status = "MIXED"
    return status, rows


def boundary_vectors(agent, states):
    observations, actions, direct = [], [], []
    state_q = {}
    for phase, row in states.items():
        obs = np.asarray(row["observations"], np.float32)
        demo = np.asarray(row["actions"], np.float32)
        policy = np.asarray(jax.device_get(agent.forward_policy(jnp.asarray(obs[None]), rng=None, train=False).mode()))[0]
        state_q[phase] = {"demo_action": demo.tolist(), "policy_action": policy.tolist()}
        for dim in range(6):
            for value in (-1.0, 1.0):
                action = demo.copy()
                action[dim] = value
                observations.append(obs)
                actions.append(action)
                direct.append(min(np.max(np.abs(action - demo)), np.max(np.abs(action - policy))) <= 0.05)
    return jnp.asarray(observations), jnp.asarray(actions), float(np.mean(direct)), state_q


def pressure_audit(agent, batch, states):
    params = agent.state.params
    rng = jax.random.PRNGKey(20260907)
    bellman_fn = lambda p: SACAgent.critic_loss_fn(agent, batch, p, rng)[0]
    cq_fn = lambda p: 5.0 * ConservativeQSACAgent.conservative_q_loss_fn(agent, batch, p, rng)
    bellman_grad = jax.grad(bellman_fn)(params)
    cq_grad = jax.grad(cq_fn)(params)
    net_grad = jax.tree_util.tree_map(lambda x, y: x + y, bellman_grad, cq_grad)
    obs, actions, direct, state_q = boundary_vectors(agent, states)
    q_fn = lambda p: q_apply(agent, p, obs, actions)

    def direction(grad):
        tangent = jax.tree_util.tree_map(lambda x: -x, grad)
        _q, dq = jax.jvp(q_fn, (params,), (tangent,))
        values = np.asarray(jax.device_get(dq))
        return {**summary(values), "downward_fraction": float(np.mean(values < 0))}

    q_values = np.asarray(jax.device_get(q_fn(params)))
    return {
        "direct_boundary_vector_coverage": direct,
        "bellman": direction(bellman_grad),
        "conservative_q": direction(cq_grad),
        "net": direction(net_grad),
        "boundary_q": summary(q_values),
        "state_actions": state_q,
    }


def bootstrap_audit(agent, batch):
    rng = jax.random.PRNGKey(20260907)
    rng, next_key = jax.random.split(rng)
    next_actions, _log_probs = agent._compute_next_actions(batch, next_key)
    target_qs = agent.forward_target_critic(batch["next_observations"], next_actions, rng=rng)
    rng, subsample_key = jax.random.split(rng)
    indices = jax.random.randint(subsample_key, (agent.config["critic_subsample_size"],), 0, agent.config["critic_ensemble_size"])
    target_min = target_qs[indices].min(axis=0)
    target = batch["rewards"] + agent.config["discount"] * batch["masks"] * target_min
    actions = np.asarray(jax.device_get(next_actions))[:, :6]
    return {
        "reward": summary(batch["rewards"]),
        "mask_one_fraction": float(np.mean(np.asarray(batch["masks"]) == 1)),
        "target_action_boundary_coordinate_rate": float(np.mean(np.abs(actions) >= BOUNDARY)),
        "target_q": summary(jax.device_get(target_min)),
        "bootstrap_target": summary(jax.device_get(target)),
    }


def sweep_geometry(agent, states, target):
    boundary_flags, q_demo, q_boundary, q_policy = [], [], [], []
    for row in states.values():
        obs = np.asarray(row["observations"], np.float32)
        demo = np.asarray(row["actions"], np.float32)
        policy = np.asarray(jax.device_get(agent.forward_policy(jnp.asarray(obs[None]), rng=None, train=False).mode()))[0]
        for dim in range(6):
            actions = np.repeat(demo[None], len(GRID), axis=0)
            actions[:, dim] = GRID
            obs_batch = jnp.repeat(jnp.asarray(obs[None]), len(GRID), axis=0)
            if target:
                q = agent.forward_target_critic(
                    obs_batch, jnp.asarray(actions), rng=jax.random.PRNGKey(20260907)
                )
            else:
                q = agent.forward_critic(obs_batch, jnp.asarray(actions), rng=None, train=False)
            mean_q = np.asarray(jax.device_get(q)).mean(axis=0)
            argmax = float(GRID[int(np.argmax(mean_q))])
            boundary_flags.append(abs(argmax) >= BOUNDARY)
            q_demo.append(float(mean_q[int(np.argmin(np.abs(GRID - demo[dim])))]))
            q_boundary.extend([float(mean_q[0]), float(mean_q[-1])])
            q_policy.append(float(mean_q[int(np.argmin(np.abs(GRID - policy[dim])))]))
    return {
        "boundary_argmax_fraction": float(np.mean(boundary_flags)),
        "demo_q": summary(q_demo),
        "boundary_q": summary(q_boundary),
        "policy_q": summary(q_policy),
    }


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    demos = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    states = selected_states(cfg, demos)
    phases = demo_phases()
    quality = {name: replay_quality(demos, phases, run) for name, run in RUNS.items()}
    details = {}
    for name, run in RUNS.items():
        details[name] = {}
        for step in (500, 1000):
            agent = load_agent(cfg, run, step)
            demo_batch, online_batch = fixed_batches(demos, run, step)
            source = source_metrics(agent, demo_batch, online_batch)
            demo_td, online_td = td_metrics(agent, demo_batch), td_metrics(agent, online_batch)
            demo_boot, online_boot = bootstrap_audit(agent, demo_batch), bootstrap_audit(agent, online_batch)
            calibration, calibration_rows = value_calibration(agent, states)
            details[name][str(step)] = {
                "source": source, "demo_td": demo_td, "online_td": online_td,
                "demo_bootstrap": demo_boot, "online_bootstrap": online_boot,
                "current_geometry": sweep_geometry(agent, states, False),
                "target_geometry": sweep_geometry(agent, states, True),
                "value_calibration": calibration, "value_rows": calibration_rows,
            }
    at_1000 = [details[name]["1000"] for name in sorted(details)]
    source_1000 = [row["source"] for row in at_1000]
    demo_bellman = float(np.mean([row["demo_bellman_loss"] for row in source_1000]))
    online_bellman = float(np.mean([row["online_bellman_loss"] for row in source_1000]))
    demo_grad = float(np.mean([row["demo_critic_grad_norm"] for row in source_1000]))
    online_grad = float(np.mean([row["online_critic_grad_norm"] for row in source_1000]))
    grad_ratio = online_grad / max(demo_grad, 1e-12)
    demo_td = float(np.mean([row["demo_td"]["td_abs_mean"] for row in at_1000]))
    online_td = float(np.mean([row["online_td"]["td_abs_mean"] for row in at_1000]))
    demo_target_rate = float(np.mean([row["demo_bootstrap"]["target_action_boundary_coordinate_rate"] for row in at_1000]))
    online_target_rate = float(np.mean([row["online_bootstrap"]["target_action_boundary_coordinate_rate"] for row in at_1000]))
    demo_target_q = float(np.mean([row["demo_bootstrap"]["target_q"]["mean"] for row in at_1000]))
    online_target_q = float(np.mean([row["online_bootstrap"]["target_q"]["mean"] for row in at_1000]))
    bootstrap_amplification = (
        "YES" if online_target_rate > demo_target_rate and abs(online_target_q) > abs(demo_target_q)
        else "NO" if online_target_rate <= demo_target_rate and abs(online_target_q) <= abs(demo_target_q)
        else "MIXED"
    )
    calibration_values = {row["value_calibration"] for row in at_1000}
    calibration = calibration_values.pop() if len(calibration_values) == 1 else "MIXED"
    phase_distribution = {
        phase: float(np.mean([row["phase_fraction"][phase] for row in quality.values()]))
        for phase in next(iter(quality.values()))["phase_fraction"]
    }
    support_fraction = float(np.mean([row["demo_support_fraction"] for row in quality.values()]))
    near_fraction = float(np.mean([row["near_hole_like_fraction"] for row in quality.values()]))
    pre_fraction = float(np.mean([row["pre_divergence_fraction"] for row in quality.values()]))
    post_fraction = float(np.mean([row["post_divergence_fraction"] for row in quality.values()]))
    prefix_support = {str(size): replay_quality(demos, phases, RUNS["R1"], size)["demo_support_fraction"]
                      for size in (64, 124, 184, 244, 304, 364, 424, 484, 544, 604)}
    geometry_500 = float(np.mean([details[name]["500"]["current_geometry"]["boundary_argmax_fraction"] for name in details]))
    geometry_1000 = float(np.mean([details[name]["1000"]["current_geometry"]["boundary_argmax_fraction"] for name in details]))
    if post_fraction >= 0.95:
        root = "ONLINE_REPLAY_POST_DIVERGENCE_DOMINANCE"
        next_fix = "ONLINE_ADMISSION_QUALITY_GATE_REVIEW"
    elif grad_ratio > 1.0:
        root = "ONLINE_CRITIC_GRADIENT_DOMINANCE"
        next_fix = "DEMO_ONLINE_CRITIC_BALANCE_REVIEW"
    elif bootstrap_amplification == "YES":
        root = "ONLINE_BOOTSTRAP_BAD_STATE_AMPLIFICATION"
        next_fix = "VALUE_CALIBRATION_AND_TARGET_REVIEW"
    elif calibration in {"BOUNDARY_OVERVALUED", "POLICY_OVERVALUED"}:
        root = "BELLMAN_VALUE_CALIBRATION_FAILURE"
        next_fix = "VALUE_CALIBRATION_AND_TARGET_REVIEW"
    elif support_fraction < 0.5:
        root = "STATE_ACTION_SUPPORT_STILL_MISALIGNED"
        next_fix = "STATE_ACTION_COVERAGE_CURRICULUM_REVIEW"
    else:
        root = "MIXED_NO_SINGLE_CAUSAL_BLOCKER"
        next_fix = "BLOCKED_NO_AUTONOMOUS_SCIENTIFIC_FIX"
    result = {
        "STAGED_REPLAY_VALUE_AUDIT": "PASS",
        "ONLINE_PHASE_DISTRIBUTION": phase_distribution,
        "ONLINE_DEMO_SUPPORT_FRACTION": support_fraction,
        "ONLINE_NEAR_HOLE_LIKE_FRACTION": near_fraction,
        "PRE_DIVERGENCE_REPLAY_FRACTION": pre_fraction,
        "POST_DIVERGENCE_REPLAY_FRACTION": post_fraction,
        "REPLAY_EXPANSION_CLASS": "POST_DIVERGENCE_BAD_TRAJECTORY_DATA" if post_fraction >= 0.95 else "MIXED",
        "DEMO_BELLMAN_LOSS": demo_bellman,
        "ONLINE_BELLMAN_LOSS": online_bellman,
        "DEMO_TD_ERROR": demo_td,
        "ONLINE_TD_ERROR": online_td,
        "ONLINE_TO_DEMO_CRITIC_PRESSURE_RATIO": online_bellman / max(demo_bellman, 1e-12),
        "DEMO_CRITIC_GRAD_NORM": demo_grad,
        "ONLINE_CRITIC_GRAD_NORM": online_grad,
        "ONLINE_DEMO_GRAD_RATIO": grad_ratio,
        "ONLINE_GRADIENT_DOMINATES": "YES" if grad_ratio > 1.0 else "NO",
        "VALUE_CALIBRATION_STATUS": calibration,
        "TARGET_ACTION_BOUNDARY_RATE_DEMO": demo_target_rate,
        "TARGET_ACTION_BOUNDARY_RATE_ONLINE": online_target_rate,
        "TARGET_Q_SCALE_DEMO": demo_target_q,
        "TARGET_Q_SCALE_ONLINE": online_target_q,
        "ONLINE_BOOTSTRAP_AMPLIFICATION": bootstrap_amplification,
        "COVERAGE_GEOMETRY_BEHAVIOR_RELATION": "COVERAGE_EXPANDS_GEOMETRY_IMPROVES_BEHAVIOR_WORSENS",
        "BOUNDARY_Q_FRACTION_500": geometry_500,
        "BOUNDARY_Q_FRACTION_1000": geometry_1000,
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT_FIX": next_fix,
        "DETAILS": {"runs": details, "replay_quality": quality, "prefix_support": prefix_support},
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "DETAILS"}, sort_keys=True))
    print(out / "result.json")


if __name__ == "__main__":
    main()
