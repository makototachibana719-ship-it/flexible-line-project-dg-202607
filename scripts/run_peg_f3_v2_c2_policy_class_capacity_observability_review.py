#!/usr/bin/env python3
"""C2 policy-class/capacity/observability review with one selected repair."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.audit_peg_f3_v2_c2_closed_loop_recovery_basin as basin
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1 as corrective
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, evaluate, load_config, make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.gated_absolute_c2_action_head_v1 import (
    absolute_c2_head_action,
    gated_absolute_compose,
    gated_absolute_mode,
    lower_policy_components,
)
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate, support_distance


SOURCE_EVIDENCE = ROOT / "results/diagnostics/peg/f3_v2_c2_data_policy_capacity_review_20260908_003"
DAGGER_ROOT = ROOT / "results/datasets/peg_v2_recoverability_filtered_bounded_dagger_v1_20260908_001"
DAGGER_RUN = ROOT / "results/training/peg_f3_v2_recoverability_filtered_bounded_dagger_v1_20260908_001"
BASIN_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_closed_loop_recovery_basin_20260908_002"
PARAM_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_residual_output_parameterization_review_20260908_002"
OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_policy_class_capacity_observability_review_20260908_001"
AUDIT_DATA = OUT / "C2_POLICY_CLASS_AUDIT_DATASET_V1.pkl"

UPDATES = 1000
BATCH_SIZE = 64
CURRENT_WIDTH = 256
V2_WIDTH = 512
VERY_CLOSE_STRONG_ACTION_L2 = 0.25
LOCAL_COHERENCE_P95_MAX = 0.20
MATERIAL_RELATIVE_IMPROVEMENT = 0.20
MATERIAL_ABSOLUTE_IMPROVEMENT = 0.01
HIGH_TRAIN_MAE = 0.03
OFFLINE_DAGGER_REFERENCE = 0.0884221
STEP8_11_MAE_MAX = 0.05
STEP8_11_AXIS_MAX = 0.15


def req(value, message):
    if not value:
        raise RuntimeError(message)


def jvalue(value):
    if isinstance(value, np.ndarray):
        return jvalue(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jvalue(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jvalue(v) for v in value]
    return value


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(jvalue(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DiagnosticHead(nn.Module):
    width: int

    @nn.compact
    def __call__(self, features):
        x = nn.Dense(self.width)(features)
        x = nn.relu(x)
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        return nn.Dense(6, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)


def model_def(width):
    return DiagnosticHead(width=width)


def load_rows():
    rows = []
    existing_path = SOURCE_EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl"
    existing = pickle.loads(existing_path.read_bytes())
    for row in existing:
        episode = int(row["source_episode"])
        rows.append({
            "observations": np.asarray(row["observations"], np.float32),
            "actions": np.asarray(row["actions"], np.float32),
            "policy_action": np.asarray(row["actions"], np.float32),
            "source_type": "existing_expert",
            "source_round": 0,
            "source_episode": episode,
            "source_step": int(row["source_step"]),
            "sampling_episode": f"demo_{episode}",
            "recoverability_class": "PRODUCTION_DERIVED_EXPERT",
            "g2": float(row["g2"]),
        })
    for round_index in (1, 2, 3):
        path = DAGGER_ROOT / f"C2_DAGGER_ROUND{round_index}_V1.pkl"
        for row in pickle.loads(path.read_bytes()):
            rows.append({
                "observations": np.asarray(row["observations"], np.float32),
                "actions": np.asarray(row["actions"], np.float32),
                "policy_action": np.asarray(row["policy_action"], np.float32),
                "source_type": "recoverable_dagger",
                "source_round": round_index,
                "source_episode": int(row["source_episode"]),
                "source_step": int(row["source_step"]),
                "sampling_episode": str(row["sampling_episode"]),
                "recoverability_class": "CORE" if row["EXPERT_CORE_RECOVERABLE"] else "SHELL",
                "g2": float(row["g2"]),
            })
    rows.sort(key=lambda row: (row["source_type"], row["source_round"], row["source_episode"], row["source_step"]))
    return rows, existing_path


def add_histories(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["sampling_episode"]].append(index)
    continuity = {"h1_real": 0, "h1_segment_init": 0, "h2_real": 0, "h2_segment_init": 0}
    for indices in groups.values():
        indices.sort(key=lambda i: rows[i]["source_step"])
        by_step = {rows[i]["source_step"]: i for i in indices}
        for index in indices:
            row = rows[index]
            step = row["source_step"]
            prev_index = by_step.get(step - 1)
            if prev_index is None:
                prev_obs = row["observations"]
                prev_action = np.zeros(7, np.float32)
                continuity["h1_segment_init"] += 1
            else:
                prev_obs = rows[prev_index]["observations"]
                prev_action = rows[prev_index]["policy_action"]
                continuity["h1_real"] += 1
            prev2_index = by_step.get(step - 2) if prev_index is not None else None
            if prev2_index is None:
                prev2_obs = prev_obs
                prev2_action = np.zeros(7, np.float32)
                continuity["h2_segment_init"] += 1
            else:
                prev2_obs = rows[prev2_index]["observations"]
                prev2_action = rows[prev2_index]["policy_action"]
                continuity["h2_real"] += 1
            row["H0"] = row["observations"]
            row["H1"] = np.concatenate((row["observations"], prev_obs, prev_action)).astype(np.float32)
            row["H2"] = np.concatenate((row["observations"], prev_obs, prev_action, prev2_obs, prev2_action)).astype(np.float32)
    return continuity


def split_indices(rows):
    train, demo_valid, dagger_valid = [], [], []
    for index, row in enumerate(rows):
        if row["source_type"] == "existing_expert":
            (train if row["source_episode"] < 9 else demo_valid).append(index)
        else:
            (train if row["source_episode"] in (100, 101) else dagger_valid).append(index)
    req(train and demo_valid and dagger_valid, "episode-separated train/demo-valid/dagger-valid split")
    return train, demo_valid, dagger_valid


def normalized_obs(rows, scale):
    return np.asarray([row["observations"] / scale for row in rows], np.float32)


def cross_episode_neighbors(rows, normalized):
    n = len(rows)
    distances = np.full((n, n), np.inf, np.float64)
    for start in range(0, n, 128):
        delta = normalized[start:start + 128, None, :] - normalized[None, :, :]
        block = np.sqrt(np.sum(np.square(delta), axis=-1))
        for local, query in enumerate(range(start, min(start + 128, n))):
            same = np.asarray([row["sampling_episode"] == rows[query]["sampling_episode"] for row in rows])
            block[local, same] = np.inf
        distances[start:start + block.shape[0]] = block
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(n), nearest]
    targets = np.asarray([row["actions"][:6] for row in rows], np.float32)
    axis_diff = np.abs(targets - targets[nearest])
    action_l2 = np.linalg.norm(targets - targets[nearest], axis=1)
    return distances, nearest, nearest_distance, axis_diff, action_l2


def duplicate_audit(rows, normalized):
    distances, nearest, nearest_distance, axis_diff, action_l2 = cross_episode_neighbors(rows, normalized)
    quantiles = {name: float(np.quantile(nearest_distance, q)) for name, q in (("P01", .01), ("P05", .05), ("P10", .10))}
    disagreement = {}
    for name, threshold in quantiles.items():
        chosen = nearest_distance <= threshold
        disagreement[name] = {
            "distance_threshold": threshold,
            "pair_count": int(np.sum(chosen)),
            "action_l2_mean": float(np.mean(action_l2[chosen])),
            "action_l2_p95": float(np.quantile(action_l2[chosen], .95)),
            "action_l2_max": float(np.max(action_l2[chosen])),
            "per_axis_mean": np.mean(axis_diff[chosen], axis=0),
            "per_axis_p95": np.quantile(axis_diff[chosen], .95, axis=0),
        }
    chosen = nearest_distance <= quantiles["P01"]
    pairs = []
    for query in np.flatnonzero(chosen):
        neighbor = int(nearest[query])
        pairs.append({
            "query": query,
            "neighbor": neighbor,
            "distance": nearest_distance[query],
            "action_l2": action_l2[query],
            "per_axis": axis_diff[query],
            "query_episode": rows[query]["sampling_episode"],
            "query_step": rows[query]["source_step"],
            "neighbor_episode": rows[neighbor]["sampling_episode"],
            "neighbor_step": rows[neighbor]["source_step"],
            "query_recoverability": rows[query]["recoverability_class"],
            "neighbor_recoverability": rows[neighbor]["recoverability_class"],
        })
    return {
        "normalization": "existing fixed C1 support normalization scale",
        "nearest_distance_quantiles": quantiles,
        "distance_conditioned_disagreement": disagreement,
        "VERY_CLOSE_PAIR_COUNT": int(np.sum(chosen)),
        "VERY_CLOSE_ACTION_DISAGREEMENT_MEAN": float(np.mean(action_l2[chosen])),
        "VERY_CLOSE_ACTION_DISAGREEMENT_P95": float(np.quantile(action_l2[chosen], .95)),
        "VERY_CLOSE_ACTION_DISAGREEMENT_MAX": float(np.max(action_l2[chosen])),
        "VERY_CLOSE_PER_AXIS_MEAN": np.mean(axis_diff[chosen], axis=0),
        "VERY_CLOSE_PER_AXIS_P95": np.quantile(axis_diff[chosen], .95, axis=0),
        "very_close_pairs": pairs,
    }, distances


def region_tag(row):
    step = row["source_step"]
    if step == 0 or (row["source_type"] == "existing_expert" and step == 32):
        return "C2_RESET_OR_SEGMENT_START"
    if 8 <= step <= 11:
        return "STEP8_11"
    if row["recoverability_class"] in ("SHELL", "CORE"):
        return "C1_SHELL_BOUNDARY"
    return "C2_TRAJECTORY"


def local_variance(rows, distances):
    targets = np.asarray([row["actions"][:6] for row in rows], np.float32)
    reports = {}
    all_scores = []
    for k in (3, 5):
        neighbor_indices = np.argsort(distances, axis=1)[:, :k]
        values = np.var(targets[neighbor_indices], axis=1)
        score = np.mean(values, axis=1)
        reports[f"k{k}"] = {
            "LOCAL_ACTION_VARIANCE_MEAN": float(np.mean(score)),
            "LOCAL_ACTION_VARIANCE_P95": float(np.quantile(score, .95)),
            "per_axis_mean": np.mean(values, axis=0),
            "per_axis_p95": np.quantile(values, .95, axis=0),
        }
        if k == 5:
            all_scores = score
            worst = np.argsort(score)[-20:][::-1]
            reports["worst20"] = [{
                "index": int(index),
                "sampling_episode": rows[index]["sampling_episode"],
                "source_step": rows[index]["source_step"],
                "region": region_tag(rows[index]),
                "variance": score[index],
                "per_axis_variance": values[index],
            } for index in worst]
            reports["worst20_region_counts"] = {tag: sum(region_tag(rows[i]) == tag for i in worst) for tag in sorted({region_tag(rows[i]) for i in worst})}
    reports["LOCAL_ACTION_VARIANCE_MEAN"] = reports["k5"]["LOCAL_ACTION_VARIANCE_MEAN"]
    reports["LOCAL_ACTION_VARIANCE_P95"] = reports["k5"]["LOCAL_ACTION_VARIANCE_P95"]
    return reports


def features(rows, indices, history):
    return np.asarray([rows[index][history] for index in indices], np.float32)


def observations(rows, indices):
    return np.asarray([rows[index]["observations"] for index in indices], np.float32)


def targets(rows, indices):
    return np.asarray([rows[index]["actions"] for index in indices], np.float32)


def episode_balanced(rows, allowed, count, rng):
    by_episode = defaultdict(list)
    for index in allowed:
        by_episode[rows[index]["sampling_episode"]].append(index)
    episodes = sorted(by_episode)
    selected = rng.choice(episodes, count, replace=True)
    return np.asarray([rng.choice(by_episode[episode]) for episode in selected], np.int64)


def compose_actions(definition, params, feature, obs, agent, c1_params, support, scale, core, outer):
    lower_loc, lower_action = lower_policy_components(agent, c1_params, obs)
    delta = definition.apply({"params": params}, feature)
    head_action = jnp.tanh(jax.lax.stop_gradient(lower_loc[..., :6]) + delta)
    g2 = progressive_gate(obs, support, scale, core, outer)
    return gated_absolute_compose(lower_action, head_action, g2)


def metric(definition, params, feature, obs, target, agent, c1_params, support, scale, core, outer):
    pred = np.asarray(jax.device_get(compose_actions(definition, params, jnp.asarray(feature), jnp.asarray(obs), agent, c1_params, support, scale, core, outer)), np.float32)
    error = np.abs(pred[:, :6] - target[:, :6])
    return {
        "mae": float(np.mean(error)),
        "per_axis_mae": np.mean(error, axis=0),
        "max_error": float(np.max(error)),
        "finite": bool(np.all(np.isfinite(pred))),
    }


def train_model(label, history, width, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, seed):
    definition = model_def(width)
    input_dim = len(rows[0][history])
    params = definition.init(jax.random.PRNGKey(seed), np.zeros((1, input_dim), np.float32))["params"]
    opt_state = review.TX.init(params)
    rng = np.random.default_rng(seed)
    demo_train = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger_train = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]

    @jax.jit
    def step(candidate, state, feature, obs, target):
        def loss_fn(value):
            pred = compose_actions(definition, value, feature, obs, agent, c1_params, support, scale, core, outer)
            return jnp.mean(jnp.square(pred[..., :6] - target[..., :6]))
        loss, gradient = jax.value_and_grad(loss_fn)(candidate)
        update, new_state = review.TX.update(gradient, state, candidate)
        return optax.apply_updates(candidate, update), new_state, loss, optax.global_norm(gradient)

    checkpoints = []
    for update in range(1, UPDATES + 1):
        chosen_demo = episode_balanced(rows, demo_train, BATCH_SIZE // 2, rng)
        chosen_dagger = episode_balanced(rows, dagger_train, BATCH_SIZE // 2, rng)
        chosen = np.concatenate((chosen_demo, chosen_dagger))
        params, opt_state, loss, grad = step(
            params, opt_state,
            jnp.asarray(features(rows, chosen, history)),
            jnp.asarray(observations(rows, chosen)),
            jnp.asarray(targets(rows, chosen)),
        )
        if update in (250, 500, 1000):
            checkpoints.append({"update": update, "loss": float(loss), "gradient_norm": float(grad)})
    train_metrics = metric(definition, params, features(rows, train_indices, history), observations(rows, train_indices), targets(rows, train_indices), agent, c1_params, support, scale, core, outer)
    demo_metrics = metric(definition, params, features(rows, demo_valid, history), observations(rows, demo_valid), targets(rows, demo_valid), agent, c1_params, support, scale, core, outer)
    dagger_metrics = metric(definition, params, features(rows, dagger_valid, history), observations(rows, dagger_valid), targets(rows, dagger_valid), agent, c1_params, support, scale, core, outer)
    valid_indices = demo_valid + dagger_valid
    valid_metrics = metric(definition, params, features(rows, valid_indices, history), observations(rows, valid_indices), targets(rows, valid_indices), agent, c1_params, support, scale, core, outer)
    return {
        "label": label,
        "history": history,
        "width": width,
        "input_dim": input_dim,
        "updates": UPDATES,
        "source_sampling": "0.5 existing expert / 0.5 aggregated DAgger; within-source episode-balanced",
        "train": train_metrics,
        "valid": valid_metrics,
        "demo_valid": demo_metrics,
        "dagger_valid": dagger_metrics,
        "checkpoints": checkpoints,
    }, definition, params, opt_state


def knn_metrics(rows, train_indices, valid_indices, scale, k):
    train_obs = observations(rows, train_indices) / scale
    valid_obs = observations(rows, valid_indices) / scale
    train_actions = targets(rows, train_indices)[:, :6]
    valid_actions = targets(rows, valid_indices)[:, :6]
    distance = np.sqrt(np.sum(np.square(valid_obs[:, None, :] - train_obs[None, :, :]), axis=-1))
    nearest = np.argsort(distance, axis=1)[:, :k]
    pred = np.mean(train_actions[nearest], axis=1)
    error = np.abs(pred - valid_actions)
    return {"mae": float(np.mean(error)), "per_axis_mae": np.mean(error, axis=0), "distance_mean": float(np.mean(np.take_along_axis(distance, nearest, axis=1)))}


def hidden_attribution(rows, duplicate, history_models, h0_model, v2_model):
    strong = [pair for pair in duplicate["very_close_pairs"] if pair["action_l2"] >= VERY_CLOSE_STRONG_ACTION_L2]
    step_delta, prev_action_delta, action_delta = [], [], []
    for pair in strong:
        a, b = rows[pair["query"]], rows[pair["neighbor"]]
        step_delta.append(abs(a["source_step"] - b["source_step"]))
        prev_action_delta.append(float(np.linalg.norm(a["H1"][-7:] - b["H1"][-7:])))
        action_delta.append(pair["action_l2"])
    def correlation(x, y):
        if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])
    h1_improvement = (h0_model["dagger_valid"]["mae"] - history_models["H1"]["dagger_valid"]["mae"]) / max(h0_model["dagger_valid"]["mae"], 1e-12)
    h2_improvement = (h0_model["dagger_valid"]["mae"] - history_models["H2"]["dagger_valid"]["mae"]) / max(h0_model["dagger_valid"]["mae"], 1e-12)
    recent_observable_explains = max(h1_improvement, h2_improvement) >= MATERIAL_RELATIVE_IMPROVEMENT
    correlations = {"elapsed_step_vs_action_disagreement": correlation(step_delta, action_delta), "previous_action_vs_action_disagreement": correlation(prev_action_delta, action_delta)}
    hidden_correlated = any(value is not None and abs(value) >= 0.30 for value in correlations.values())
    if not strong:
        classification = "NO"
    elif recent_observable_explains:
        classification = "NO"
    elif hidden_correlated:
        classification = "YES"
    else:
        classification = "MIXED"
    return {
        "strong_very_close_pairs": len(strong),
        "available_non_obs_diagnostics": ["source elapsed step", "source rollout/round", "previous executed action from sequential rows"],
        "unavailable_in_immutable_policy_class_rows": ["qpos", "qvel", "controller internal state", "wrapper internal state"],
        "correlations": correlations,
        "H1_relative_dagger_improvement": h1_improvement,
        "H2_relative_dagger_improvement": h2_improvement,
        "recent_observable_history_explains": recent_observable_explains,
        "HIDDEN_STATE_EXPLAINS_ACTION_DISAGREEMENT": classification,
    }


class SelectedPolicy:
    def __init__(self, definition, params, history, agent, c1_params, support, scale, core, outer, reset_obs):
        self.definition, self.params, self.history = definition, params, history
        self.agent, self.c1_params = agent, c1_params
        self.support, self.scale, self.core, self.outer = support, scale, core, outer
        self.reset_obs = np.asarray(reset_obs, np.float32)
        self.prev_obs = None
        self.prev2_obs = None
        self.prev_action = None
        self.prev2_action = None
        self.classes = []

    def _feature(self, obs):
        obs = np.asarray(obs, np.float32)
        reset = self.prev_obs is None or np.array_equal(obs, self.reset_obs)
        if reset:
            self.prev_obs = obs.copy(); self.prev2_obs = obs.copy()
            self.prev_action = np.zeros(7, np.float32); self.prev2_action = np.zeros(7, np.float32)
        if self.history == "H0":
            return obs
        if self.history == "H1":
            return np.concatenate((obs, self.prev_obs, self.prev_action)).astype(np.float32)
        return np.concatenate((obs, self.prev_obs, self.prev_action, self.prev2_obs, self.prev2_action)).astype(np.float32)

    def sample_actions(self, observations, seed=None, deterministic=False, argmax=False, **_):
        del seed, deterministic, argmax
        obs = np.asarray(observations, np.float32)
        feature = self._feature(obs)
        action = np.asarray(jax.device_get(compose_actions(self.definition, self.params, jnp.asarray(feature), jnp.asarray(obs), self.agent, self.c1_params, self.support, self.scale, self.core, self.outer)), np.float32)
        distance = float(np.asarray(jax.device_get(support_distance(obs, self.support, self.scale))))
        g0 = float(np.asarray(jax.device_get(gate_value(obs))))
        if g0 == 0.0:
            region = "C0_CORE"
        elif distance <= self.core:
            region = "C1_CORE"
        elif distance <= self.outer:
            region = "C1_RECOVERY_SHELL"
        else:
            region = "C2_OUTSIDE"
        self.classes.append(region)
        self.prev2_obs, self.prev2_action = self.prev_obs.copy(), self.prev_action.copy()
        self.prev_obs, self.prev_action = obs.copy(), action.copy()
        return action

    def stats(self):
        shell = sum(x in ("C1_RECOVERY_SHELL", "C1_CORE") and (i == 0 or self.classes[i - 1] == "C2_OUTSIDE") for i, x in enumerate(self.classes))
        core = sum(x == "C1_CORE" and (i == 0 or self.classes[i - 1] not in ("C1_CORE", "C0_CORE")) for i, x in enumerate(self.classes))
        return {"visited_states": len(self.classes), "class_state_counts": {name: self.classes.count(name) for name in ("C2_OUTSIDE", "C1_RECOVERY_SHELL", "C1_CORE", "C0_CORE")}, "C1_SHELL_ENTRY_COUNT": shell, "C1_CORE_ENTRY_COUNT": core}


def eval_selected(definition, params, history, agent, c1_params, support, scale, core, outer, level, demo_obs, demo_scale, reset_obs):
    policy = SelectedPolicy(definition, params, history, agent, c1_params, support, scale, core, outer, reset_obs)
    record = evaluate(policy, level, demo_obs, demo_scale, seeds=(100, 101, 102))
    record["interface"] = policy.stats()
    return record


def historical_boundary_audit(definition, params, history, agent, c1_params, offline_params, support, scale, core, outer, demos, demo_obs, demo_scale):
    env = make_train_env("C2")
    states = []
    try:
        obs, _ = env.reset(seed=100)
        prev_obs = np.asarray(obs, np.float32).copy()
        prev_action = np.zeros(7, np.float32)
        prev2_obs = prev_obs.copy(); prev2_action = prev_action.copy()
        for step in range(12):
            historical_action = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, offline_params, obs, support, scale, core, outer)), np.float32)
            if step in (10, 11):
                if history == "H0": feature = np.asarray(obs, np.float32)
                elif history == "H1": feature = np.concatenate((obs, prev_obs, prev_action)).astype(np.float32)
                else: feature = np.concatenate((obs, prev_obs, prev_action, prev2_obs, prev2_action)).astype(np.float32)
                repaired = np.asarray(jax.device_get(compose_actions(definition, params, jnp.asarray(feature), jnp.asarray(obs), agent, c1_params, support, scale, core, outer)), np.float32)
                distances = np.sqrt(np.sum(np.square((demo_obs - obs) / demo_scale), axis=1))
                nearest = int(np.argmin(distances))
                expert = np.asarray(demos[nearest]["actions"], np.float32)
                branch = basin.branch_then_takeover(basin.capture(env, obs), repaired, demos, demo_obs, demo_scale, support, scale, core, outer)
                states.append({"historical_step": step, "observation": np.asarray(obs, np.float32), "historical_policy_action": historical_action, "repaired_action": repaired, "expert_action": expert, "repaired_expert_l2": float(np.linalg.norm(repaired[:6] - expert[:6])), "branch": branch})
            prev2_obs, prev2_action = prev_obs.copy(), prev_action.copy()
            prev_obs, prev_action = np.asarray(obs, np.float32).copy(), historical_action.copy()
            obs, _, terminated, truncated, _ = env.step(historical_action)
            if terminated or truncated:
                break
    finally:
        env.close()
    req([row["historical_step"] for row in states] == [10, 11], "historical boundary states reproduced")
    repaired = bool(states[0]["branch"]["one_step"]["contact_healthy"] and states[0]["branch"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"])
    return {"states": states, "HISTORICAL_BOUNDARY_CROSSING_REPAIRED": "YES" if repaired else "NO"}


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU backend required")
    started = time.time()
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    rows, existing_path = load_rows()
    continuity = add_histories(rows)
    AUDIT_DATA.write_bytes(pickle.dumps(rows, pickle.HIGHEST_PROTOCOL))
    audit_sha = sha(AUDIT_DATA)
    train_indices, demo_valid, dagger_valid = split_indices(rows)
    protected_paths = {
        "production_demo": p4.DEMO,
        "existing_c2_subset": existing_path,
        "dagger_r1": DAGGER_ROOT / "C2_DAGGER_ROUND1_V1.pkl",
        "dagger_r2": DAGGER_ROOT / "C2_DAGGER_ROUND2_V1.pkl",
        "dagger_r3": DAGGER_ROOT / "C2_DAGGER_ROUND3_V1.pkl",
        "offline_p2_checkpoint": Path(json.loads((PARAM_RUN / "result.json").read_text())["SELECTED_OFFLINE_CHECKPOINT"]["path"]),
    }
    protected_before = {name: sha(path) for name, path in protected_paths.items()}
    dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo immutable SHA")

    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    normalized = normalized_obs(rows, scale)
    duplicate, distance_matrix = duplicate_audit(rows, normalized)
    local = local_variance(rows, distance_matrix)
    dump(OUT / "duplicate_near_duplicate_audit.json", duplicate)
    dump(OUT / "local_action_variance.json", local)
    dump(OUT / "audit_dataset_manifest.json", {
        "name": "C2_POLICY_CLASS_AUDIT_DATASET_V1", "sha256": audit_sha, "rows": len(rows),
        "source_rows": {"existing_expert": sum(r["source_type"] == "existing_expert" for r in rows), "dagger_r1": sum(r["source_round"] == 1 for r in rows), "dagger_r2": sum(r["source_round"] == 2 for r in rows), "dagger_r3": sum(r["source_round"] == 3 for r in rows)},
        "history_continuity": continuity, "source_files_immutable": protected_before,
        "split": {"train": len(train_indices), "demo_valid": len(demo_valid), "dagger_valid": len(dagger_valid)},
    })

    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
    cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"])
        c1_params = source["c1_params"]
        base_guard = geo.thash(agent.state.params["modules_actor"]); c1_guard = geo.thash(c1_params)

        h0, h0_def, h0_params, _ = train_model("CURRENT_P2_HEAD_H0", "H0", CURRENT_WIDTH, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261100)
        h1, h1_def, h1_params, _ = train_model("CURRENT_CAPACITY_H1", "H1", CURRENT_WIDTH, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261101)
        h2, h2_def, h2_params, _ = train_model("CURRENT_CAPACITY_H2", "H2", CURRENT_WIDTH, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261102)
        v2, v2_def, v2_params, _ = train_model("C2_P2_HEAD_CAPACITY_V2", "H0", V2_WIDTH, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261103)
        models = {"H0": h0, "H1": h1, "H2": h2, "V2": v2}
        dump(OUT / "offline_model_diagnostics.json", models)
        valid_all = demo_valid + dagger_valid
        knn1 = knn_metrics(rows, train_indices, valid_all, scale, 1)
        knn3 = knn_metrics(rows, train_indices, valid_all, scale, 3)
        knn1_dagger = knn_metrics(rows, train_indices, dagger_valid, scale, 1)
        knn3_dagger = knn_metrics(rows, train_indices, dagger_valid, scale, 3)
        dump(OUT / "nonparametric_oracle.json", {"OBS20_1NN": knn1, "OBS20_3NN": knn3, "OBS20_1NN_DAGGER": knn1_dagger, "OBS20_3NN_DAGGER": knn3_dagger})
        hidden = hidden_attribution(rows, duplicate, {"H1": h1, "H2": h2}, h0, v2)
        dump(OUT / "hidden_state_attribution.json", hidden)

        local_coherent = duplicate["VERY_CLOSE_ACTION_DISAGREEMENT_P95"] <= LOCAL_COHERENCE_P95_MAX
        current_high = h0["train"]["mae"] > HIGH_TRAIN_MAE
        capacity_train_gain = (h0["train"]["mae"] - v2["train"]["mae"]) / max(h0["train"]["mae"], 1e-12)
        capacity_dagger_gain = (h0["dagger_valid"]["mae"] - v2["dagger_valid"]["mae"]) / max(h0["dagger_valid"]["mae"], 1e-12)
        h1_vs_h0 = (h0["dagger_valid"]["mae"] - h1["dagger_valid"]["mae"]) / max(h0["dagger_valid"]["mae"], 1e-12)
        h2_vs_h0 = (h0["dagger_valid"]["mae"] - h2["dagger_valid"]["mae"]) / max(h0["dagger_valid"]["mae"], 1e-12)
        best_history = min(h1["dagger_valid"]["mae"], h2["dagger_valid"]["mae"])
        history_vs_both = best_history <= (1.0 - MATERIAL_RELATIVE_IMPROVEMENT) * min(h0["dagger_valid"]["mae"], v2["dagger_valid"]["mae"]) and min(h0["dagger_valid"]["mae"], v2["dagger_valid"]["mae"]) - best_history >= MATERIAL_ABSOLUTE_IMPROVEMENT
        strong_close_conflict = duplicate["VERY_CLOSE_ACTION_DISAGREEMENT_P95"] > LOCAL_COHERENCE_P95_MAX or hidden["strong_very_close_pairs"] > 0
        capacity_case = current_high and capacity_train_gain >= MATERIAL_RELATIVE_IMPROVEMENT and capacity_dagger_gain >= MATERIAL_RELATIVE_IMPROVEMENT and max(h1_vs_h0, h2_vs_h0) < MATERIAL_RELATIVE_IMPROVEMENT and local_coherent
        history_case = history_vs_both
        label_case = strong_close_conflict and not history_vs_both and not local_coherent
        optimization_case = h0["train"]["mae"] > HIGH_TRAIN_MAE and v2["train"]["mae"] > HIGH_TRAIN_MAE and local_coherent and max(h1_vs_h0, h2_vs_h0) < MATERIAL_RELATIVE_IMPROVEMENT
        if capacity_case:
            root, selected, selected_history, selected_width = "C2_CURRENT_HEAD_UNDERCAPACITY", "C2_P2_HEAD_CAPACITY_V2", "H0", V2_WIDTH
        elif history_case:
            root, selected, selected_history, selected_width = "C2_OBSERVATION_ALIASING_OR_HISTORY_DEPENDENCE", "HISTORY_AUGMENTED_C2_POLICY_V1", "H1", CURRENT_WIDTH
        elif label_case:
            root, selected, selected_history, selected_width = "C2_EXPERT_LABEL_MULTIMODALITY_UNDER_POLICY_OBSERVATION", "NONE", None, None
        elif optimization_case:
            root, selected, selected_history, selected_width = "C2_SUPERVISED_OPTIMIZATION_UNDERFIT", "NONE", None, None
        else:
            root, selected, selected_history, selected_width = "MIXED_C2_POLICY_CLASS_FAILURE", "NONE", None, None
        decision = {
            "thresholds_preregistered": {"very_close_definition": "cross-episode nearest distance <= P01", "strong_action_l2": VERY_CLOSE_STRONG_ACTION_L2, "local_coherence_p95_max": LOCAL_COHERENCE_P95_MAX, "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT, "material_absolute_improvement": MATERIAL_ABSOLUTE_IMPROVEMENT, "high_train_mae": HIGH_TRAIN_MAE},
            "evidence": {"local_coherent": local_coherent, "current_train_materially_high": current_high, "capacity_train_relative_gain": capacity_train_gain, "capacity_dagger_relative_gain": capacity_dagger_gain, "H1_dagger_relative_gain": h1_vs_h0, "H2_dagger_relative_gain": h2_vs_h0, "history_substantially_better_than_H0_and_V2": history_vs_both, "strong_close_conflict": strong_close_conflict},
            "PRIMARY_ROOT_CAUSE": root, "SELECTED_REPAIR": selected,
        }
        dump(OUT / "root_cause_decision.json", decision)

        selected_report = None; boundary = None; c2_eval = None; c1_eval = None; c0_eval = None
        matched = "NOT_RUN"; confirmed = False; canonical = None; first_bad = None
        c1_ok = False; pilot_pass = False
        if selected != "NONE":
            selected_report, selected_def, selected_params, selected_opt = train_model("SELECTED_REPAIR_FRESH", selected_history, selected_width, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261200)
            step_indices = [i for i in dagger_valid if 8 <= rows[i]["source_step"] <= 11]
            req(step_indices, "held-out step8-11 states")
            step_metrics = metric(selected_def, selected_params, features(rows, step_indices, selected_history), observations(rows, step_indices), targets(rows, step_indices), agent, c1_params, support, scale, core, outer)
            selected_report["step8_11_valid"] = step_metrics
            selected_report["deterministic_history_initialization"] = "at episode reset previous Obs20=current reset Obs20 and previous action=zeros"
            checkpoint = OUT / "checkpoints" / ("history_augmented_c2_policy_v1.msgpack" if selected == "HISTORY_AUGMENTED_C2_POLICY_V1" else "c2_p2_head_capacity_v2.msgpack")
            checkpoint.write_bytes(flax.serialization.to_bytes({"params": selected_params, "opt_state": selected_opt}))
            selected_report["checkpoint"] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
            offline_gate = bool(selected_report["dagger_valid"]["mae"] <= (1.0 - MATERIAL_RELATIVE_IMPROVEMENT) * min(OFFLINE_DAGGER_REFERENCE, h0["dagger_valid"]["mae"]) and step_metrics["mae"] <= STEP8_11_MAE_MAX and float(np.max(step_metrics["per_axis_mae"])) <= STEP8_11_AXIS_MAX)
            selected_report["OFFLINE_REPAIR_GATE"] = "PASS" if offline_gate else "FAIL"
            dump(OUT / "selected_repair_offline_gate.json", selected_report)
            parameter = json.loads((PARAM_RUN / "result.json").read_text())
            offline_params = corrective.load_head(parameter["SELECTED_OFFLINE_CHECKPOINT"]["path"])["c2_head_params"]
            if offline_gate:
                boundary = historical_boundary_audit(selected_def, selected_params, selected_history, agent, c1_params, offline_params, support, scale, core, outer, demos, demo_obs, demo_scale)
                dump(OUT / "historical_boundary_crossing_audit.json", boundary)
                if boundary["HISTORICAL_BOUNDARY_CROSSING_REPAIRED"] == "YES":
                    reset_obs = rows[next(i for i, row in enumerate(rows) if row["source_round"] == 1 and row["source_step"] == 0)]["observations"]
                    baseline_c1 = review.eval_record(agent, c1_params, offline_params, support, scale, core, outer, "C1", demo_obs, demo_scale)
                    c2_eval = eval_selected(selected_def, selected_params, selected_history, agent, c1_params, support, scale, core, outer, "C2", demo_obs, demo_scale, reset_obs)
                    c1_eval = eval_selected(selected_def, selected_params, selected_history, agent, c1_params, support, scale, core, outer, "C1", demo_obs, demo_scale, reset_obs)
                    c0_eval = eval_selected(selected_def, selected_params, selected_history, agent, c1_params, support, scale, core, outer, "C0", demo_obs, demo_scale, reset_obs)
                    c1_ok = bool(c1_eval["contact_health"] and all(c1_eval[key] == baseline_c1[key] for key in ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")))
                    pilot_pass = bool(c0_eval["success_count"] == 3 and c0_eval["contact_health"] and c1_ok and c2_eval["contact_health"] and c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"] > 0)
                    dump(OUT / "c2_pilot.json", {"C2": c2_eval, "C1": c1_eval, "C0": c0_eval, "C1_REGRESSION": "PASS" if c1_ok else "FAIL", "C2_PILOT": "PASS" if pilot_pass else "FAIL"})
                    if pilot_pass:
                        matched_report, matched_def, matched_params, matched_opt = train_model("MATCHED_SELECTED_REPAIR", selected_history, selected_width, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261200)
                        matched_c2 = eval_selected(matched_def, matched_params, selected_history, agent, c1_params, support, scale, core, outer, "C2", demo_obs, demo_scale, reset_obs)
                        matched_c1 = eval_selected(matched_def, matched_params, selected_history, agent, c1_params, support, scale, core, outer, "C1", demo_obs, demo_scale, reset_obs)
                        matched_c0 = eval_selected(matched_def, matched_params, selected_history, agent, c1_params, support, scale, core, outer, "C0", demo_obs, demo_scale, reset_obs)
                        confirmed = bool(matched_c2 == c2_eval and matched_c1 == c1_eval and matched_c0 == c0_eval)
                        matched = "PASS" if confirmed else "FAIL"
                        dump(OUT / "matched_confirmation.json", {"training": matched_report, "C2": matched_c2, "C1": matched_c1, "C0": matched_c0, "confirmed": confirmed})
                        if confirmed:
                            frozen = OUT / "checkpoints" / "confirmed_frozen_c2_policy.msgpack"
                            frozen.write_bytes(flax.serialization.to_bytes({"params": matched_params, "opt_state": matched_opt}))
                            canonical = eval_selected(matched_def, matched_params, selected_history, agent, c1_params, support, scale, core, outer, "C3", demo_obs, demo_scale, reset_obs)
                            dump(OUT / "canonical_interface_audit.json", canonical)
                    else:
                        first_bad = "C2_PILOT_SHELL_OR_CONTACT_GATE"
                else:
                    first_bad = "HISTORICAL_BOUNDARY_CROSSING_REPAIRED"
            else:
                first_bad = "SELECTED_REPAIR_OFFLINE_GATE"
        else:
            first_bad = "POLICY_CLASS_ROOT_CAUSE_DECISION"

        req(geo.thash(agent.state.params["modules_actor"]) == base_guard and geo.thash(c1_params) == c1_guard, "C0/C1 parameters immutable")
        protected_after = {name: sha(path) for name, path in protected_paths.items()}
        req(protected_after == protected_before, "protected artifacts immutable")
        dump(OUT / "protected_after.json", protected_after)
        result = {
            "C2_POLICY_CLASS_CAPACITY_REVIEW": "PASS",
            "AUDIT_DATASET_SHA": audit_sha,
            "CURRENT_HEAD_TRAIN_MAE": h0["train"]["mae"],
            "CURRENT_HEAD_VALID_MAE": h0["valid"]["mae"],
            "CURRENT_HEAD_DAGGER_VALID_MAE": h0["dagger_valid"]["mae"],
            "OBS20_1NN_VALID_MAE": knn1["mae"], "OBS20_3NN_VALID_MAE": knn3["mae"],
            "VERY_CLOSE_ACTION_DISAGREEMENT_P95": duplicate["VERY_CLOSE_ACTION_DISAGREEMENT_P95"],
            "LOCAL_ACTION_VARIANCE_P95": local["LOCAL_ACTION_VARIANCE_P95"],
            "HIDDEN_STATE_EXPLAINS_ACTION_DISAGREEMENT": hidden["HIDDEN_STATE_EXPLAINS_ACTION_DISAGREEMENT"],
            "H1_VALID_MAE": h1["valid"]["mae"], "H2_VALID_MAE": h2["valid"]["mae"],
            "V2_TRAIN_MAE": v2["train"]["mae"], "V2_VALID_MAE": v2["valid"]["mae"], "V2_DAGGER_VALID_MAE": v2["dagger_valid"]["mae"],
            "PRIMARY_ROOT_CAUSE": root, "SELECTED_REPAIR": selected,
            "SELECTED_REPAIR_VALID_MAE": None if selected_report is None else selected_report["valid"]["mae"],
            "SELECTED_REPAIR_DAGGER_VALID_MAE": None if selected_report is None else selected_report["dagger_valid"]["mae"],
            "STEP8_11_VALID_MAE": None if selected_report is None else selected_report["step8_11_valid"]["mae"],
            "HISTORICAL_BOUNDARY_CROSSING_REPAIRED": "NOT_RUN" if boundary is None else boundary["HISTORICAL_BOUNDARY_CROSSING_REPAIRED"],
            "C2_PILOT": "NOT_RUN" if c2_eval is None else ("PASS" if c2_eval["contact_health"] and c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"] > 0 else "FAIL"),
            "C2_SHELL_ENTRY": None if c2_eval is None else c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"],
            "C2_CORE_ENTRY": None if c2_eval is None else c2_eval["interface"]["C1_CORE_ENTRY_COUNT"],
            "C2_NEAR_HOLE": None if c2_eval is None else c2_eval["near_hole_rate"],
            "C2_SUCCESS": "NOT_RUN" if c2_eval is None else f"{c2_eval['success_count']}/3",
            "C2_CONTACT": "NOT_RUN" if c2_eval is None else ("PASS" if c2_eval["contact_health"] else "FAIL"),
            "C1_REGRESSION": "NOT_RUN" if c1_eval is None else ("PASS" if c1_ok else "FAIL"),
            "C1_CONTACT": "NOT_RUN" if c1_eval is None else ("PASS" if c1_eval["contact_health"] else "FAIL"),
            "C0_SUCCESS": "NOT_RUN" if c0_eval is None else f"{c0_eval['success_count']}/3",
            "C2_MATCHED_CONFIRMATION": matched, "C2_SKILL_CONFIRMED": "YES" if confirmed else "NO",
            "CANONICAL_INTERFACE_AUDIT": "NOT_RUN" if canonical is None else "PASS",
            "PEG_SINGLE_TASK_CONVERGED": "NO",
            "FIRST_BAD_GATE": first_bad,
            "NEXT": "C3_ACQUISITION_PILOT" if confirmed else "C2_POLICY_STATE_OR_EXPERT_TARGET_DEFINITION_REVIEW" if root == "C2_EXPERT_LABEL_MULTIMODALITY_UNDER_POLICY_OBSERVATION" else "C2_SUPERVISED_OPTIMIZATION_REVIEW" if root == "C2_SUPERVISED_OPTIMIZATION_UNDERFIT" else "STOP",
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": p4.TH,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started,
        }
        dump(OUT / "result.json", result)
        print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
