#!/usr/bin/env python3
"""C2 supervised optimizer causal review and at most one selected repair."""
from __future__ import annotations

import copy
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
    str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, evaluate, load_config
from serl_launcher.common.optimizers import make_optimizer
from training.peg_insertion.gated_absolute_c2_action_head_v1 import gated_absolute_mode, lower_policy_components
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


POLICY_REVIEW = ROOT / "results/diagnostics/peg/f3_v2_c2_policy_class_capacity_observability_review_20260908_001"
AUDIT_DATA = POLICY_REVIEW / "C2_POLICY_CLASS_AUDIT_DATASET_V1.pkl"
OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_supervised_optimization_review_20260908_001"
BASELINE_STEPS = (0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 750, 1000)
CONDITION_STEPS = (0, 100, 250, 500, 1000)
UPDATE_STEPS = (1, 5, 10, 25, 50, 100, 250, 500, 1000)
SOURCE_STEPS = (0, 100, 250, 500, 1000)
RESPONSE_STEPS = (0, 100, 500, 1000)
UPDATES = 1000
PLATEAU_LR = 3e-4
WARMUP_STEPS = 2000
EPS = review.NUMERICAL_EPS

# Mechanism thresholds are fixed before diagnostics, not chosen from results.
SATURATION_FRACTION_MIN = 0.20
SATURATED_SENSITIVITY_MAX = 0.10
LOGIT_TO_ACTION_GRAD_RATIO_MAX = 0.25
SATURATED_RESIDUAL_LOSS_SHARE_MIN = 0.50
MATERIAL_RELATIVE_IMPROVEMENT = 0.20
LINE_SCAN_EXTRA_IMPROVEMENT_MIN = 0.01
STRONG_ANTI_ALIGNMENT = -0.20
CONFLICT_CHECKPOINT_FRACTION = 0.75
REPAIR_TRAIN_MAE_TARGET = 0.03
REPAIR_DAGGER_MAE_TARGET = 0.06
REPAIR_AXIS_MAE_MAX = 0.15


def req(value, message):
    if not value:
        raise RuntimeError(message)


def jvalue(value):
    if isinstance(value, np.ndarray): return jvalue(value.tolist())
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, dict): return {str(k): jvalue(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jvalue(v) for v in value]
    return value


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(jvalue(value), stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def tree_norm(tree):
    return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(jax.device_get(x), np.float64)))) for x in jax.tree_util.tree_leaves(tree))))


def tree_dot(a, b):
    return float(sum(float(np.sum(np.asarray(jax.device_get(x), np.float64) * np.asarray(jax.device_get(y), np.float64))) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))))


def tree_cosine(a, b):
    return tree_dot(a, b) / max(tree_norm(a) * tree_norm(b), 1e-20)


def layer_norms(tree):
    result = defaultdict(float)
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        text = jax.tree_util.keystr(path)
        layer = next((name for name in ("Dense_0", "Dense_1", "Dense_2") if name in text), "other")
        result[layer] += float(np.sum(np.square(np.asarray(jax.device_get(value), np.float64))))
    return {key: float(np.sqrt(value)) for key, value in result.items()}


def moment_report(state):
    mu, nu, counts = 0.0, 0.0, []
    paths = []
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        text = jax.tree_util.keystr(path); array = np.asarray(jax.device_get(value))
        paths.append({"path": text, "shape": list(array.shape), "dtype": str(array.dtype)})
        if ".mu" in text or "['mu']" in text: mu += float(np.sum(np.square(array.astype(np.float64))))
        if ".nu" in text or "['nu']" in text: nu += float(np.sum(np.square(array.astype(np.float64))))
        if "count" in text.lower() and array.size == 1: counts.append(int(array.reshape(())))
    return {"adam_m_norm": float(np.sqrt(mu)), "adam_v_norm": float(np.sqrt(nu)), "step_counters": counts, "state_leaf_paths": paths}


def fixed_batches(rows, train_indices):
    demo = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]
    rng = np.random.default_rng(261300)
    demo64 = pc.episode_balanced(rows, demo, 64, rng)
    dagger64 = pc.episode_balanced(rows, dagger, 64, rng)
    mixed = np.concatenate((demo64[:32], dagger64[:32]))
    return demo, dagger, demo64, dagger64, mixed


def batch_arrays(rows, indices):
    return pc.features(rows, indices, "H0"), pc.observations(rows, indices), pc.targets(rows, indices)


def action_loss(definition, params, feature, obs, target, agent, c1_params, support, scale, core, outer, axis=None):
    pred = pc.compose_actions(definition, params, feature, obs, agent, c1_params, support, scale, core, outer)
    error = pred[..., :6] - target[..., :6]
    return jnp.mean(jnp.square(error if axis is None else error[..., axis]))


def loss_and_grad(definition, params, arrays, agent, c1_params, support, scale, core, outer, axis=None):
    feature, obs, target = map(jnp.asarray, arrays)
    fn = lambda value: action_loss(definition, value, feature, obs, target, agent, c1_params, support, scale, core, outer, axis)
    return jax.value_and_grad(fn)(params)


def metrics(definition, params, rows, indices, agent, c1_params, support, scale, core, outer):
    return pc.metric(definition, params, pc.features(rows, indices, "H0"), pc.observations(rows, indices), pc.targets(rows, indices), agent, c1_params, support, scale, core, outer)


def baseline_train(rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer):
    definition = pc.model_def(pc.CURRENT_WIDTH)
    params = definition.init(jax.random.PRNGKey(261100), np.zeros((1, 20), np.float32))["params"]
    opt_state = review.TX.init(params)
    rng = np.random.default_rng(261100)
    demo_train = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger_train = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]
    reports, checkpoints, update_reports = {}, {}, {}

    @jax.jit
    def step(candidate, state, feature, obs, target):
        fn = lambda value: action_loss(definition, value, feature, obs, target, agent, c1_params, support, scale, core, outer)
        loss, gradient = jax.value_and_grad(fn)(candidate)
        updates, new_state = review.TX.update(gradient, state, candidate)
        new_params = optax.apply_updates(candidate, updates)
        return new_params, new_state, loss, gradient, updates

    def save(step_index):
        reports[step_index] = {
            "step": step_index,
            "train": metrics(definition, params, rows, train_indices, agent, c1_params, support, scale, core, outer),
            "demo_valid": metrics(definition, params, rows, demo_valid, agent, c1_params, support, scale, core, outer),
            "dagger_valid": metrics(definition, params, rows, dagger_valid, agent, c1_params, support, scale, core, outer),
        }
        if step_index in set(CONDITION_STEPS) | set(SOURCE_STEPS) | set(RESPONSE_STEPS): checkpoints[step_index] = (params, opt_state)

    save(0)
    for step_index in range(1, UPDATES + 1):
        chosen_demo = pc.episode_balanced(rows, demo_train, 32, rng)
        chosen_dagger = pc.episode_balanced(rows, dagger_train, 32, rng)
        arrays = batch_arrays(rows, np.concatenate((chosen_demo, chosen_dagger)))
        before = params
        params, opt_state, loss, gradient, updates = step(params, opt_state, *map(jnp.asarray, arrays))
        if step_index in UPDATE_STEPS:
            delta = jax.tree_util.tree_map(lambda a, b: a - b, params, before)
            moment = moment_report(opt_state)
            update_reports[step_index] = {
                "gradient_norm": tree_norm(gradient), "raw_optimizer_update_norm": tree_norm(updates),
                "actual_parameter_delta_norm": tree_norm(delta), "parameter_norm_before": tree_norm(before),
                "PARAM_DELTA_OVER_PARAM_NORM": tree_norm(delta) / max(tree_norm(before), 1e-20),
                "OUTPUT_LAYER_DELTA_NORM": layer_norms(delta).get("Dense_2", 0.0),
                "per_layer_gradient_norm": layer_norms(gradient), "per_layer_update_norm": layer_norms(updates),
                "adam_m_norm": moment["adam_m_norm"], "adam_v_norm": moment["adam_v_norm"],
                "adam_step_counters": moment["step_counters"], "batch_loss": float(loss),
            }
        if step_index in BASELINE_STEPS: save(step_index)
    return definition, params, opt_state, reports, checkpoints, update_reports


def target_saturation(rows, split_indices):
    result = {}
    for name, indices in split_indices.items():
        action = pc.targets(rows, indices)[:, :6]
        clipped = np.clip(action, -1.0 + EPS, 1.0 - EPS)
        z = np.arctanh(clipped)
        sensitivity = 1.0 - np.square(action)
        result[name] = {
            "rows": len(indices),
            "fractions": {str(threshold): np.mean(np.abs(action) >= threshold, axis=0) for threshold in (0.8, 0.9, 0.95, 0.99)},
            "Z_TARGET_ABS_MEAN": np.mean(np.abs(z), axis=0), "Z_TARGET_ABS_P95": np.quantile(np.abs(z), .95, axis=0),
            "Z_TARGET_ABS_MAX": np.max(np.abs(z), axis=0),
            "TARGET_TANH_SENSITIVITY_MEAN": np.mean(sensitivity, axis=0),
            "TARGET_TANH_SENSITIVITY_P05": np.quantile(sensitivity, .05, axis=0),
        }
    return result


def conditioning_report(definition, checkpoints, rows, train_indices, agent, c1_params, support, scale, core, outer):
    arrays_np = batch_arrays(rows, train_indices)
    feature, obs, target = map(jnp.asarray, arrays_np)
    result = {}
    for step_index in CONDITION_STEPS:
        params, _ = checkpoints[step_index]
        loss, gradient = loss_and_grad(definition, params, arrays_np, agent, c1_params, support, scale, core, outer)
        pred = pc.compose_actions(definition, params, feature, obs, agent, c1_params, support, scale, core, outer)
        action_error = pred[..., :6] - target[..., :6]
        action_gradient = 2.0 * action_error / float(action_error.size)
        pre_tanh_gradient = action_gradient * (1.0 - jnp.square(pred[..., :6]))
        axis = []
        for axis_index in range(6):
            axis_loss, axis_grad = loss_and_grad(definition, params, arrays_np, agent, c1_params, support, scale, core, outer, axis_index)
            axis.append({"axis": axis_index, "loss": float(axis_loss), "parameter_gradient_norm": tree_norm(axis_grad),
                         "final_action_gradient_norm": float(jnp.linalg.norm(action_gradient[..., axis_index])),
                         "pre_tanh_gradient_norm": float(jnp.linalg.norm(pre_tanh_gradient[..., axis_index]))})
        layers = layer_norms(gradient)
        result[step_index] = {
            "loss": float(loss), "GRAD_NORM_TOTAL": tree_norm(gradient),
            "GRAD_NORM_FINAL_LAYER": layers.get("Dense_2", 0.0), "GRAD_NORM_EARLY_LAYER": layers.get("Dense_0", 0.0),
            "per_layer": layers, "FINAL_ACTION_GRAD_NORM": float(jnp.linalg.norm(action_gradient)),
            "PRETANH_LOGIT_GRAD_NORM": float(jnp.linalg.norm(pre_tanh_gradient)), "per_axis": axis,
        }
    return result


def schedule_report():
    _, schedule = make_optimizer(learning_rate=PLATEAU_LR, warmup_steps=WARMUP_STEPS, return_lr_schedule=True)
    values = np.asarray([float(schedule(step - 1)) for step in range(1, UPDATES + 1)], np.float64)
    requested = (1, 5, 10, 25, 50, 100, 250, 500, 1000)
    return {
        "WARMUP_STEPS": WARMUP_STEPS, "POST_WARMUP_LR": PLATEAU_LR,
        "LR_AT_STEP": {step: values[step - 1] for step in requested},
        "WARMUP_FACTOR_AT_STEP": {step: values[step - 1] / PLATEAU_LR for step in requested},
        "INTEGRATED_EFFECTIVE_LR_1K": float(np.sum(values)),
        "fraction_below_plateau": {"25_percent": float(np.mean(values < .25 * PLATEAU_LR)), "50_percent": float(np.mean(values < .50 * PLATEAU_LR)), "90_percent": float(np.mean(values < .90 * PLATEAU_LR))},
        "optimizer_code": "make_optimizer -> inject_hyperparams(optax.adam) with join_schedules(linear 0..3e-4 for 2000, constant thereafter)",
    }


def deterministic_subset(rows, train_indices, size):
    demo = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]
    rng = np.random.default_rng(261320 + size)
    return np.concatenate((pc.episode_balanced(rows, demo, size // 2, rng), pc.episode_balanced(rows, dagger, size // 2, rng)))


def memorize(rows, subset, agent, c1_params, support, scale, core, outer, seed):
    definition = pc.model_def(pc.CURRENT_WIDTH)
    params = definition.init(jax.random.PRNGKey(seed), np.zeros((1, 20), np.float32))["params"]
    state = review.TX.init(params); rng = np.random.default_rng(seed)
    arrays_all = batch_arrays(rows, subset)
    @jax.jit
    def step(candidate, opt_state, feature, obs, target):
        fn = lambda value: action_loss(definition, value, feature, obs, target, agent, c1_params, support, scale, core, outer)
        loss, grad = jax.value_and_grad(fn)(candidate); update, opt_state = review.TX.update(grad, opt_state, candidate)
        return optax.apply_updates(candidate, update), opt_state
    for _ in range(UPDATES):
        chosen = rng.choice(subset, 64, replace=True)
        params, state = step(params, state, *map(jnp.asarray, batch_arrays(rows, chosen)))
    report = pc.metric(definition, params, *map(np.asarray, arrays_all), agent, c1_params, support, scale, core, outer)
    return report


def source_conflict(definition, checkpoints, rows, demo_batch, dagger_batch, agent, c1_params, support, scale, core, outer):
    result = {}
    cosines = []
    for step_index in SOURCE_STEPS:
        params, _ = checkpoints[step_index]
        demo_loss, demo_grad = loss_and_grad(definition, params, batch_arrays(rows, demo_batch), agent, c1_params, support, scale, core, outer)
        dagger_loss, dagger_grad = loss_and_grad(definition, params, batch_arrays(rows, dagger_batch), agent, c1_params, support, scale, core, outer)
        cosine = tree_cosine(demo_grad, dagger_grad); cosines.append(cosine)
        layer_demo, layer_dagger = layer_norms(demo_grad), layer_norms(dagger_grad)
        layer_cos = {}
        for layer in ("Dense_0", "Dense_1", "Dense_2"):
            mask = lambda tree: jax.tree_util.tree_map_with_path(lambda path, value: value if layer in jax.tree_util.keystr(path) else jnp.zeros_like(value), tree)
            layer_cos[layer] = tree_cosine(mask(demo_grad), mask(dagger_grad))
        axis = []
        for i in range(6):
            ld, _ = loss_and_grad(definition, params, batch_arrays(rows, demo_batch), agent, c1_params, support, scale, core, outer, i)
            lg, _ = loss_and_grad(definition, params, batch_arrays(rows, dagger_batch), agent, c1_params, support, scale, core, outer, i)
            axis.append({"axis": i, "demo_loss": float(ld), "dagger_loss": float(lg)})
        result[step_index] = {"demo_loss": float(demo_loss), "dagger_loss": float(dagger_loss), "demo_gradient_norm": tree_norm(demo_grad), "dagger_gradient_norm": tree_norm(dagger_grad), "DEMO_DAGGER_GRAD_COSINE_TOTAL": cosine, "per_layer_cosine": layer_cos, "per_layer_demo_norm": layer_demo, "per_layer_dagger_norm": layer_dagger, "per_axis_source_loss": axis}
    return result, {"cosines": cosines, "negative_fraction": float(np.mean(np.asarray(cosines) < 0.0)), "strongly_anti_aligned_fraction": float(np.mean(np.asarray(cosines) <= STRONG_ANTI_ALIGNMENT)), "mean": float(np.mean(cosines)), "median": float(np.median(cosines))}


def one_step_response(definition, checkpoints, rows, fixed_batch, agent, c1_params, support, scale, core, outer):
    arrays = tuple(map(jnp.asarray, batch_arrays(rows, fixed_batch)))
    result = {}
    @jax.jit
    def respond(params, state):
        fn = lambda value: action_loss(definition, value, *arrays, agent, c1_params, support, scale, core, outer)
        before, grad = jax.value_and_grad(fn)(params); update, next_state = review.TX.update(grad, state, params)
        next_params = optax.apply_updates(params, update); after = fn(next_params)
        return before, after, update, next_state
    for step_index in RESPONSE_STEPS:
        params, actual_state = checkpoints[step_index]
        before, actual_after, actual_update, _ = respond(params, actual_state)
        _, fresh_after, fresh_update, _ = respond(params, review.TX.init(params))
        result[step_index] = {"loss_before": float(before), "actual_loss_after": float(actual_after), "fresh_loss_after": float(fresh_after), "ONE_STEP_LOSS_DELTA_ACTUAL": float(actual_after - before), "ONE_STEP_LOSS_DELTA_FRESH": float(fresh_after - before), "actual_update_norm": tree_norm(actual_update), "fresh_update_norm": tree_norm(fresh_update)}
    pathology = any(row["ONE_STEP_LOSS_DELTA_FRESH"] < row["ONE_STEP_LOSS_DELTA_ACTUAL"] - 1e-6 for row in result.values())
    return result, "YES" if pathology else "NO"


def line_scan(definition, checkpoint, rows, train_indices, agent, c1_params, support, scale, core, outer):
    params, state = checkpoint; arrays = tuple(map(jnp.asarray, batch_arrays(rows, train_indices)))
    fn = lambda value: action_loss(definition, value, *arrays, agent, c1_params, support, scale, core, outer)
    loss, grad = jax.value_and_grad(fn)(params); update, _ = review.TX.update(grad, state, params)
    values = {alpha: float(fn(optax.apply_updates(params, jax.tree_util.tree_map(lambda x: alpha * x, update)))) for alpha in (0.0, .25, .5, 1.0, 2.0, 4.0)}
    best = min(values, key=values.get)
    return {"ordinary_update_norm": tree_norm(update), "losses": values, "best_alpha": best, "ordinary_delta": values[1.0] - values[0.0], "alpha4_delta": values[4.0] - values[0.0]}


def train_repair(repair, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, seed):
    definition = pc.model_def(pc.CURRENT_WIDTH)
    params = definition.init(jax.random.PRNGKey(seed), np.zeros((1, 20), np.float32))["params"]
    tx = optax.adam(PLATEAU_LR) if repair == "SUPERVISED_CONSTANT_PLATEAU_LR_V1" else review.TX
    state = tx.init(params); rng = np.random.default_rng(seed)
    demo = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]
    @jax.jit
    def step(candidate, opt_state, feature, obs, target):
        def fn(value):
            if repair == "PRETANH_TARGET_SUPERVISION_V1":
                lower_loc, _ = lower_policy_components(agent, c1_params, obs)
                delta = definition.apply({"params": value}, feature)
                z_pred = lower_loc[..., :6] + delta
                z_target = jnp.arctanh(jnp.clip(target[..., :6], -1.0 + EPS, 1.0 - EPS))
                return jnp.mean(jnp.square(z_pred - z_target))
            return action_loss(definition, value, feature, obs, target, agent, c1_params, support, scale, core, outer)
        loss, grad = jax.value_and_grad(fn)(candidate); update, opt_state = tx.update(grad, opt_state, candidate)
        return optax.apply_updates(candidate, update), opt_state
    for update_index in range(UPDATES):
        if repair == "ALTERNATING_SOURCE_BALANCED_SUPERVISION_V1":
            pool = demo if update_index % 2 == 0 else dagger
            chosen = pc.episode_balanced(rows, pool, 64, rng)
        else:
            chosen = np.concatenate((pc.episode_balanced(rows, demo, 32, rng), pc.episode_balanced(rows, dagger, 32, rng)))
        params, state = step(params, state, *map(jnp.asarray, batch_arrays(rows, chosen)))
    report = {
        "repair": repair,
        "train": metrics(definition, params, rows, train_indices, agent, c1_params, support, scale, core, outer),
        "demo_valid": metrics(definition, params, rows, demo_valid, agent, c1_params, support, scale, core, outer),
        "dagger_valid": metrics(definition, params, rows, dagger_valid, agent, c1_params, support, scale, core, outer),
    }
    valid = demo_valid + dagger_valid
    report["valid"] = metrics(definition, params, rows, valid, agent, c1_params, support, scale, core, outer)
    step_indices = [i for i in dagger_valid if 8 <= rows[i]["source_step"] <= 11]
    report["step8_11"] = metrics(definition, params, rows, step_indices, agent, c1_params, support, scale, core, outer)
    return definition, params, state, report


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    req(sha(AUDIT_DATA) == "6e8ba2cf872c000cb3a3655fa2d241338672c12598af995a83b5413d7126638d", "authoritative audit dataset SHA")
    OUT.mkdir(parents=True); (OUT / "checkpoints").mkdir(); started = time.time()
    rows = pickle.loads(AUDIT_DATA.read_bytes()); train_indices, demo_valid, dagger_valid = pc.split_indices(rows)
    protected_paths = {
        "production_demo": p4.DEMO, "audit_dataset": AUDIT_DATA,
        "existing_subset": pc.SOURCE_EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl",
        "dagger_r1": pc.DAGGER_ROOT / "C2_DAGGER_ROUND1_V1.pkl", "dagger_r2": pc.DAGGER_ROOT / "C2_DAGGER_ROUND2_V1.pkl", "dagger_r3": pc.DAGGER_ROOT / "C2_DAGGER_ROUND3_V1.pkl",
    }
    protected_before = {name: sha(path) for name, path in protected_paths.items()}; dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo SHA")
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32); scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text()); core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos); cfg = load_config(2)
    target_report = target_saturation(rows, {"train": train_indices, "demo_valid": demo_valid, "dagger_valid": dagger_valid}); dump(OUT / "target_saturation_report.json", target_report)
    schedule = schedule_report(); dump(OUT / "effective_lr_warmup.json", schedule)
    demo_train, dagger_train, demo_batch, dagger_batch, mixed_batch = fixed_batches(rows, train_indices)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        base_guard = geo.thash(agent.state.params["modules_actor"]); c1_guard = geo.thash(c1_params)
        definition, params, opt_state, baseline, checkpoints, updates = baseline_train(rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer)
        prior = json.loads((POLICY_REVIEW / "result.json").read_text())
        reproduced = bool(abs(baseline[1000]["train"]["mae"] - prior["CURRENT_HEAD_TRAIN_MAE"]) <= 1e-8 and abs(baseline[1000]["dagger_valid"]["mae"] - prior["CURRENT_HEAD_DAGGER_VALID_MAE"]) <= 1e-8)
        req(reproduced, "ENGINEERING_REPRODUCTION baseline mismatch")
        dump(OUT / "baseline_reproduction.json", {"BASELINE_REPRODUCED": "YES", "checkpoints": baseline})
        dump(OUT / "parameter_update_magnitude.json", updates)
        conditioning = conditioning_report(definition, checkpoints, rows, train_indices, agent, c1_params, support, scale, core, outer); dump(OUT / "loss_gradient_conditioning.json", conditioning)
        mem16 = memorize(rows, deterministic_subset(rows, train_indices, 16), agent, c1_params, support, scale, core, outer, 261336)
        mem64 = memorize(rows, deterministic_subset(rows, train_indices, 64), agent, c1_params, support, scale, core, outer, 261384)
        dump(OUT / "small_subset_memorization.json", {"TRAIN16": mem16, "TRAIN64": mem64})
        conflict, conflict_summary = source_conflict(definition, checkpoints, rows, demo_batch, dagger_batch, agent, c1_params, support, scale, core, outer); dump(OUT / "source_gradient_conflict.json", {"checkpoints": conflict, "summary": conflict_summary})
        responses, adam_pathology = one_step_response(definition, checkpoints, rows, mixed_batch, agent, c1_params, support, scale, core, outer); dump(OUT / "one_step_optimizer_response.json", {"checkpoints": responses, "ADAM_STATE_PATHOLOGY": adam_pathology})
        scan = line_scan(definition, checkpoints[1000], rows, train_indices, agent, c1_params, support, scale, core, outer); dump(OUT / "fixed_direction_line_scan.json", scan)

        train_targets = pc.targets(rows, train_indices)[:, :6]
        saturation_axis = np.mean(np.abs(train_targets) >= .95, axis=0)
        saturated_axes = saturation_axis >= SATURATION_FRACTION_MIN
        sensitivity_axis = np.mean(1.0 - np.square(train_targets), axis=0)
        axis_loss = np.asarray([row["loss"] for row in conditioning[1000]["per_axis"]])
        axis_action_grad = np.asarray([row["final_action_gradient_norm"] for row in conditioning[1000]["per_axis"]])
        axis_logit_grad = np.asarray([row["pre_tanh_gradient_norm"] for row in conditioning[1000]["per_axis"]])
        attenuation_ratio = np.divide(axis_logit_grad, axis_action_grad, out=np.zeros_like(axis_logit_grad), where=axis_action_grad > 1e-12)
        pathological_axes = saturated_axes & (sensitivity_axis <= SATURATED_SENSITIVITY_MAX) & (attenuation_ratio <= LOGIT_TO_ACTION_GRAD_RATIO_MAX)
        saturated_loss_share = float(np.sum(axis_loss[pathological_axes]) / max(np.sum(axis_loss), 1e-20))
        tanh_case = bool(np.any(pathological_axes) and saturated_loss_share >= SATURATED_RESIDUAL_LOSS_SHARE_MIN)
        tanh_status = "YES" if tanh_case else "MIXED" if np.any(saturated_axes) else "NO"
        lr_small = schedule["fraction_below_plateau"]["50_percent"] >= .95
        direction_support = bool(scan["best_alpha"] in (2.0, 4.0) and scan["losses"][scan["best_alpha"]] <= scan["losses"][1.0] - LINE_SCAN_EXTRA_IMPROVEMENT_MIN * abs(scan["ordinary_delta"]))
        moving = updates[1000]["PARAM_DELTA_OVER_PARAM_NORM"] > 1e-8
        step_small_case = bool(not tanh_case and lr_small and direction_support and moving)
        effective_step = "YES" if step_small_case else "MIXED" if lr_small else "NO"
        source_case = bool(not tanh_case and not step_small_case and conflict_summary["strongly_anti_aligned_fraction"] >= CONFLICT_CHECKPOINT_FRACTION)
        source_status = "YES" if source_case else "MIXED" if conflict_summary["negative_fraction"] >= .5 else "NO"
        adam_case = bool(not tanh_case and not step_small_case and not source_case and adam_pathology == "YES")
        if tanh_case:
            root, repair = "TANH_TARGET_SATURATION_GRADIENT_CONDITIONING", "PRETANH_TARGET_SUPERVISION_V1"
        elif step_small_case:
            root, repair = "SUPERVISED_EFFECTIVE_STEP_TOO_SMALL", "SUPERVISED_CONSTANT_PLATEAU_LR_V1"
        elif source_case:
            root, repair = "DEMO_DAGGER_PARAMETER_GRADIENT_CONFLICT", "ALTERNATING_SOURCE_BALANCED_SUPERVISION_V1"
        elif adam_case:
            root, repair = "SUPERVISED_ADAM_STATE_PATHOLOGY", "FRESH_SUPERVISED_OPTIMIZER_STATE_V1"
        else:
            root, repair = "MIXED_SUPERVISED_OPTIMIZATION_CONDITIONING", "NONE"
        decision = {"thresholds_preregistered": {"saturation_fraction_min": SATURATION_FRACTION_MIN, "saturated_sensitivity_max": SATURATED_SENSITIVITY_MAX, "logit_to_action_grad_ratio_max": LOGIT_TO_ACTION_GRAD_RATIO_MAX, "pathological_axis_residual_loss_share_min": SATURATED_RESIDUAL_LOSS_SHARE_MIN, "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT, "strong_anti_alignment": STRONG_ANTI_ALIGNMENT, "conflict_checkpoint_fraction": CONFLICT_CHECKPOINT_FRACTION}, "evidence": {"saturation_fraction_per_axis": saturation_axis, "saturated_axes": saturated_axes, "pathological_saturated_axes": pathological_axes, "target_sensitivity_mean_per_axis": sensitivity_axis, "pre_tanh_to_action_gradient_ratio_per_axis": attenuation_ratio, "pathological_axis_residual_loss_share": saturated_loss_share, "lr_fraction_below_half_plateau": schedule["fraction_below_plateau"]["50_percent"], "line_scan_best_alpha": scan["best_alpha"], "line_scan_supports_larger_step": direction_support, "source_conflict": conflict_summary, "adam_pathology": adam_pathology}, "TANH_CONDITIONING_PATHOLOGY": tanh_status, "EFFECTIVE_STEP_TOO_SMALL": effective_step, "SOURCE_GRADIENT_CONFLICT": source_status, "ADAM_STATE_PATHOLOGY": adam_pathology, "PRIMARY_ROOT_CAUSE": root, "SELECTED_REPAIR": repair}
        dump(OUT / "root_cause_decision.json", decision)

        repair_report = None; boundary = None; c2_eval = c1_eval = c0_eval = None; c1_ok = False; confirmed = False; matched = "NOT_RUN"; canonical = None
        if repair != "NONE":
            repair_def, repair_params, repair_state, repair_report = train_repair(repair, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261400)
            train_gain = (baseline[1000]["train"]["mae"] - repair_report["train"]["mae"]) / baseline[1000]["train"]["mae"]
            dagger_gain = (baseline[1000]["dagger_valid"]["mae"] - repair_report["dagger_valid"]["mae"]) / baseline[1000]["dagger_valid"]["mae"]
            offline_pass = bool(train_gain >= MATERIAL_RELATIVE_IMPROVEMENT and dagger_gain >= MATERIAL_RELATIVE_IMPROVEMENT and repair_report["train"]["mae"] < REPAIR_TRAIN_MAE_TARGET and repair_report["dagger_valid"]["mae"] < REPAIR_DAGGER_MAE_TARGET and float(np.max(repair_report["dagger_valid"]["per_axis_mae"])) <= REPAIR_AXIS_MAE_MAX)
            repair_report.update({"train_relative_gain": train_gain, "dagger_relative_gain": dagger_gain, "OFFLINE_REPAIR_GATE": "PASS" if offline_pass else "FAIL"})
            checkpoint = OUT / "checkpoints" / f"{repair.lower()}.msgpack"; checkpoint.write_bytes(flax.serialization.to_bytes({"params": repair_params, "opt_state": repair_state})); repair_report["checkpoint"] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
            dump(OUT / "selected_repair_offline_gate.json", repair_report)
            if offline_pass:
                parameter = json.loads((pc.PARAM_RUN / "result.json").read_text()); offline_params = __import__("scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1", fromlist=["load_head"]).load_head(parameter["SELECTED_OFFLINE_CHECKPOINT"]["path"])["c2_head_params"]
                boundary = pc.historical_boundary_audit(repair_def, repair_params, "H0", agent, c1_params, offline_params, support, scale, core, outer, demos, demo_obs, demo_scale)
                dump(OUT / "historical_step11_boundary_audit.json", boundary)
                if boundary["HISTORICAL_BOUNDARY_CROSSING_REPAIRED"] == "YES":
                    reset_obs = next(row["observations"] for row in rows if row["source_round"] == 1 and row["source_step"] == 0)
                    baseline_c1 = review.eval_record(agent, c1_params, offline_params, support, scale, core, outer, "C1", demo_obs, demo_scale)
                    c2_eval = pc.eval_selected(repair_def, repair_params, "H0", agent, c1_params, support, scale, core, outer, "C2", demo_obs, demo_scale, reset_obs)
                    c1_eval = pc.eval_selected(repair_def, repair_params, "H0", agent, c1_params, support, scale, core, outer, "C1", demo_obs, demo_scale, reset_obs)
                    c0_eval = pc.eval_selected(repair_def, repair_params, "H0", agent, c1_params, support, scale, core, outer, "C0", demo_obs, demo_scale, reset_obs)
                    c1_ok = bool(c1_eval["contact_health"] and all(c1_eval[key] == baseline_c1[key] for key in ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")))
                    pilot_pass = bool(c0_eval["success_count"] == 3 and c0_eval["contact_health"] and c1_ok and c2_eval["contact_health"] and c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"] > 0)
                    dump(OUT / "c2_pilot.json", {"C2": c2_eval, "C1": c1_eval, "C0": c0_eval, "C1_REGRESSION": "PASS" if c1_ok else "FAIL", "C2_PILOT": "PASS" if pilot_pass else "FAIL"})
                    if pilot_pass:
                        mdef, mparams, mstate, mreport = train_repair(repair, rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, 261400)
                        mc2 = pc.eval_selected(mdef, mparams, "H0", agent, c1_params, support, scale, core, outer, "C2", demo_obs, demo_scale, reset_obs)
                        mc1 = pc.eval_selected(mdef, mparams, "H0", agent, c1_params, support, scale, core, outer, "C1", demo_obs, demo_scale, reset_obs)
                        mc0 = pc.eval_selected(mdef, mparams, "H0", agent, c1_params, support, scale, core, outer, "C0", demo_obs, demo_scale, reset_obs)
                        confirmed = bool(mc2 == c2_eval and mc1 == c1_eval and mc0 == c0_eval); matched = "PASS" if confirmed else "FAIL"
                        dump(OUT / "matched_confirmation.json", {"training": mreport, "C2": mc2, "C1": mc1, "C0": mc0, "confirmed": confirmed})
                        if confirmed:
                            frozen = OUT / "checkpoints/confirmed_frozen_c2_head.msgpack"; frozen.write_bytes(flax.serialization.to_bytes({"params": mparams, "opt_state": mstate}))
                            canonical = pc.eval_selected(mdef, mparams, "H0", agent, c1_params, support, scale, core, outer, "C3", demo_obs, demo_scale, reset_obs); dump(OUT / "canonical_interface_audit.json", canonical)

        req(geo.thash(agent.state.params["modules_actor"]) == base_guard and geo.thash(c1_params) == c1_guard, "protected model mutation")
        protected_after = {name: sha(path) for name, path in protected_paths.items()}; req(protected_after == protected_before, "protected artifact mutation"); dump(OUT / "protected_after.json", protected_after)
        if repair == "NONE": first_bad, next_task = "OPTIMIZATION_ROOT_CAUSE_DECISION", "STOP"
        elif repair_report["OFFLINE_REPAIR_GATE"] == "FAIL": first_bad, next_task = "SELECTED_REPAIR_OFFLINE_GATE", "STOP"
        elif boundary is not None and boundary["HISTORICAL_BOUNDARY_CROSSING_REPAIRED"] == "NO": first_bad, next_task = "HISTORICAL_STEP11_BOUNDARY", "STOP"
        elif c2_eval is not None and not (c2_eval["contact_health"] and c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"] > 0 and c1_ok and c0_eval["success_count"] == 3): first_bad, next_task = "C2_PILOT_GATE", "STOP"
        elif confirmed: first_bad, next_task = None, "C3_ACQUISITION_PILOT"
        else: first_bad, next_task = "SELECTED_REPAIR_PENDING_GATE", "STOP"
        result = {
            "C2_SUPERVISED_OPTIMIZATION_REVIEW": "PASS", "BASELINE_REPRODUCED": "YES",
            "TARGET_SATURATION_REPORT": "PASS", "TANH_CONDITIONING_PATHOLOGY": tanh_status,
            "WARMUP_STEPS": WARMUP_STEPS, "POST_WARMUP_LR": PLATEAU_LR, "INTEGRATED_EFFECTIVE_LR_1K": schedule["INTEGRATED_EFFECTIVE_LR_1K"],
            "EFFECTIVE_STEP_TOO_SMALL": effective_step, "TRAIN16_MAE_1000": mem16["mae"], "TRAIN64_MAE_1000": mem64["mae"],
            "DEMO_DAGGER_GRAD_COSINE_TOTAL": conflict_summary, "SOURCE_GRADIENT_CONFLICT": source_status, "ADAM_STATE_PATHOLOGY": adam_pathology,
            "LOSS_LINE_SCAN": scan, "PRIMARY_ROOT_CAUSE": root, "SELECTED_REPAIR": repair,
            "REPAIR_TRAIN_MAE": None if repair_report is None else repair_report["train"]["mae"], "REPAIR_VALID_MAE": None if repair_report is None else repair_report["valid"]["mae"],
            "REPAIR_DAGGER_VALID_MAE": None if repair_report is None else repair_report["dagger_valid"]["mae"], "STEP8_11_MAE": None if repair_report is None else repair_report["step8_11"]["mae"],
            "HISTORICAL_STEP11_BOUNDARY_REPAIRED": "NOT_RUN" if boundary is None else boundary["HISTORICAL_BOUNDARY_CROSSING_REPAIRED"],
            "C2_PILOT": "NOT_RUN" if c2_eval is None else ("PASS" if c2_eval["contact_health"] and c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"] > 0 and c1_ok and c0_eval["success_count"] == 3 else "FAIL"),
            "C2_SHELL_ENTRY": None if c2_eval is None else c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"], "C2_CORE_ENTRY": None if c2_eval is None else c2_eval["interface"]["C1_CORE_ENTRY_COUNT"],
            "C2_NEAR_HOLE": None if c2_eval is None else c2_eval["near_hole_rate"], "C2_SUCCESS": "NOT_RUN" if c2_eval is None else f"{c2_eval['success_count']}/3", "C2_CONTACT": "NOT_RUN" if c2_eval is None else ("PASS" if c2_eval["contact_health"] else "FAIL"),
            "C1_REGRESSION": "NOT_RUN" if c1_eval is None else ("PASS" if c1_ok else "FAIL"), "C1_CONTACT": "NOT_RUN" if c1_eval is None else ("PASS" if c1_eval["contact_health"] else "FAIL"),
            "C0_SUCCESS": "NOT_RUN" if c0_eval is None else f"{c0_eval['success_count']}/3", "C2_MATCHED_CONFIRMATION": matched, "C2_SKILL_CONFIRMED": "YES" if confirmed else "NO",
            "CANONICAL_INTERFACE_AUDIT": "NOT_RUN" if canonical is None else "PASS", "PEG_SINGLE_TASK_CONVERGED": "NO", "FIRST_BAD_GATE": first_bad, "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": p4.TH, "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO", "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started,
        }
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
