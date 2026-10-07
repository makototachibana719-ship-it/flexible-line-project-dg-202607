#!/usr/bin/env python3
"""Static P1/P2 review, selected P2 offline fit, and bounded C2 gates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import time
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
import jax
import jax.numpy as jnp
import numpy as np
import optax

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_data_policy_capacity_review as prior
import scripts.run_peg_f3_v2_observation_gated_residual_c1_pilot as c1
import scripts.run_peg_f3_v2_progressive_frozen_residual_v2_c2_pilot as pilot
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    distance_model,
    evaluate,
    load_config,
    make_train_env,
)
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.common.optimizers import make_optimizer
from training.peg_insertion.gated_absolute_c2_action_head_v1 import (
    ABSOLUTE_C2_HEAD_DEF,
    absolute_c2_head_action,
    gated_absolute_compose,
    gated_absolute_mode,
    lower_policy_components,
)
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import (
    progressive_gate,
    progressive_mode,
    support_distance,
)
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


EVIDENCE = ROOT / "results/diagnostics/peg/f3_v2_c2_data_policy_capacity_review_20260908_003"
PREFLIGHT = ROOT / "results/preflight/peg/f3_v2_progressive_frozen_residual_v2_20260908_002"
PILOT = ROOT / "results/training/peg_f3_v2_progressive_frozen_residual_v2_c2_pilot_20260908_002"
SOURCE_CP = PILOT / "checkpoints/c2_learner_0250_residual_062.msgpack"
OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_residual_output_parameterization_review_20260908_002"
TX = make_optimizer(learning_rate=3e-4, warmup_steps=2000)
NUMERICAL_EPS = 1e-6
CHECKPOINTS = (250, 500, 1000)
VALID_MAE_RATIO_MAX = 0.50
RESET_ACTION_ERROR_MAX = 0.05
SHORT_HORIZON = 12


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


def tree_hash(tree):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(jax.device_get(value))
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def init_loaded(common):
    old = prior.init_c1(common)
    residual = prior.RESIDUAL_DEF.init(jax.random.PRNGKey(260920), np.zeros((1, 20), np.float32))["params"]
    return {
        "agent_state": old["agent_state"],
        "c1_params": old["residual_params"],
        "c1_opt_state": old["residual_opt_state"],
        "c2_params": residual,
        "c2_opt_state": prior.TX.init(residual),
    }


def region(obs, d_c1, core, outer):
    if float(np.asarray(jax.device_get(gate_value(obs)))) == 0.0:
        return "C0_CORE_REGION"
    if d_c1 <= core:
        return "C1_CORE_REGION"
    if d_c1 <= outer:
        return "C1_SHELL_REGION"
    return "C2_ACTIVE_REGION"


def cosine_rows(pred, target):
    numerator = np.sum(pred * target, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def action_metrics(params, obs, targets, agent, c1_params, support, scale, core, outer):
    actions = np.asarray(
        jax.device_get(gated_absolute_mode(agent, c1_params, params, jnp.asarray(obs), support, scale, core, outer)),
        np.float32,
    )
    error = np.abs(actions[:, :6] - targets[:, :6])
    cosines = cosine_rows(actions[:, :6], targets[:, :6])
    _, lower = lower_policy_components(agent, c1_params, jnp.asarray(obs))
    head = np.asarray(jax.device_get(absolute_c2_head_action(params, jnp.asarray(obs), lower)), np.float32)
    return {
        "action_mae": float(np.mean(error)),
        "action_cosine": float(np.mean(cosines)),
        "per_axis_action_mae": np.mean(error, axis=0),
        "max_action_error": float(np.max(error)),
        "head_saturation_fraction": float(np.mean(np.abs(head) >= 0.999)),
        "finite": bool(np.all(np.isfinite(actions)) and np.all(np.isfinite(head))),
    }


@jax.jit
def bc_step(params, opt_state, observations, targets, agent, c1_params, support, scale, core, outer):
    def loss_fn(candidate):
        actions = gated_absolute_mode(
            agent, c1_params, candidate, observations, support, scale, core, outer
        )
        return jnp.mean(jnp.square(actions[..., :6] - targets[..., :6]))

    loss, gradients = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = TX.update(gradients, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state, loss, optax.global_norm(gradients)


class AbsolutePolicy:
    def __init__(self, agent, c1_params, c2_params, support, scale, core, outer, record=False):
        self.agent = agent
        self.c1_params = c1_params
        self.c2_params = c2_params
        self.support = support
        self.scale = scale
        self.core = core
        self.outer = outer
        self.record = record
        self.classes = []

    def sample_actions(self, observations, seed=None, deterministic=False, argmax=False, **_):
        del seed, deterministic, argmax
        obs = jnp.asarray(observations)
        action = gated_absolute_mode(
            self.agent,
            self.c1_params,
            self.c2_params,
            obs,
            self.support,
            self.scale,
            self.core,
            self.outer,
        )
        if self.record:
            flat = np.asarray(obs).reshape((-1, 20))
            distances = np.asarray(jax.device_get(support_distance(flat, self.support, self.scale))).reshape(-1)
            g0 = np.asarray(jax.device_get(gate_value(flat))).reshape(-1)
            for d, z in zip(distances, g0):
                self.classes.append(
                    "C0_CORE" if z == 0 else "C1_CORE" if d <= self.core else "C1_RECOVERY_SHELL" if d <= self.outer else "C2_OUTSIDE"
                )
        return action

    def stats(self):
        shell_entries = sum(
            x in ("C1_RECOVERY_SHELL", "C1_CORE") and (i == 0 or self.classes[i - 1] == "C2_OUTSIDE")
            for i, x in enumerate(self.classes)
        )
        core_entries = sum(
            x == "C1_CORE" and (i == 0 or self.classes[i - 1] not in ("C1_CORE", "C0_CORE"))
            for i, x in enumerate(self.classes)
        )
        return {
            "visited_states": len(self.classes),
            "class_state_counts": {name: self.classes.count(name) for name in ("C2_OUTSIDE", "C1_RECOVERY_SHELL", "C1_CORE", "C0_CORE")},
            "C1_SHELL_ENTRY_COUNT": shell_entries,
            "C1_CORE_ENTRY_COUNT": core_entries,
        }


def eval_record(agent, c1_params, c2_params, support, scale, core, outer, level, demo_obs, demo_scale, seeds=(100, 101, 102)):
    policy = AbsolutePolicy(agent, c1_params, c2_params, support, scale, core, outer, True)
    result = evaluate(policy, level, demo_obs, demo_scale, seeds=seeds)
    result["interface"] = policy.stats()
    return result


def reproduce_probe_snapshots(agent, c1_params, old_c2_params, support, scale, core, outer, probes):
    wanted = {int(row["source_step"]): row for row in probes}
    snapshots = {}
    env = make_train_env("C2")
    try:
        obs, _ = env.reset(seed=100)
        for step in range(max(wanted) + 1):
            if step in wanted:
                req(np.array_equal(np.asarray(obs, np.float32), wanted[step]["observation"]), f"probe replay mismatch step {step}")
                snapshots[step] = capture(env, obs)
            action = np.asarray(
                jax.device_get(progressive_mode(agent, c1_params, old_c2_params, obs, support, scale, core, outer)),
                np.float32,
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    req(set(snapshots) == set(wanted), "all probe snapshots reproduced")
    return snapshots


def short_rollout(snapshot, mode, params, agent, c1_params, support, scale, core, outer, demo_obs, demo_actions, demo_scale):
    env = make_train_env("C2")
    rows = []
    try:
        obs = restore(env, snapshot)
        first_position = float(np.linalg.norm(obs[13:16]))
        first_support = float(np.asarray(jax.device_get(support_distance(obs, support, scale))))
        for step in range(SHORT_HORIZON):
            if mode == "lower":
                _, action_jax = lower_policy_components(agent, c1_params, jnp.asarray(obs))
                action = np.asarray(jax.device_get(action_jax), np.float32)
            elif mode == "offline_c2":
                action = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, params, obs, support, scale, core, outer)), np.float32)
            else:
                distances = np.sqrt(np.sum(np.square((demo_obs - obs) / demo_scale), axis=1))
                nearest = int(np.argmin(distances))
                if float(distances[nearest]) > p4.TH:
                    break
                action = np.asarray(demo_actions[nearest], np.float32)
            next_obs, _, terminated, truncated, info = env.step(action)
            link, bad, _ = contact_flags(env)
            rows.append({
                "step": step + 1,
                "position": float(np.linalg.norm(next_obs[13:16])),
                "D_C1": float(np.asarray(jax.device_get(support_distance(next_obs, support, scale)))),
                "demo_distance": p4.demo_distance(next_obs, demo_obs, demo_scale),
                "force": float(info.get("force_norm", 0.0)),
                "unexpected_contact": bool(link or bad),
            })
            obs = next_obs
            if terminated or truncated:
                break
        return {
            "valid_steps": len(rows),
            "initial_position": first_position,
            "initial_D_C1": first_support,
            "min_position": min([first_position] + [x["position"] for x in rows]),
            "min_D_C1": min([first_support] + [x["D_C1"] for x in rows]),
            "final_position": rows[-1]["position"] if rows else first_position,
            "final_D_C1": rows[-1]["D_C1"] if rows else first_support,
            "max_force": max([0.0] + [x["force"] for x in rows]),
            "unexpected_contact": any(x["unexpected_contact"] for x in rows),
            "rows": rows,
        }
    finally:
        env.close()


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd required")
    req(jax.default_backend() == "gpu", "GPU backend required")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    started = time.time()
    protected_paths = {"production_demo": p4.DEMO, "corrective": p4.CORR, "local": p4.LOCAL, "source_checkpoint": SOURCE_CP}
    protected_before = {key: sha(path) for key, path in protected_paths.items()}
    dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo SHA")
    req(sha(EVIDENCE / "C2_BEHAVIOR_CRITICAL_PROBE_V1.pkl") == "386c1e56a0ad5d47ce577d5dcc130830a6a5caa565e852f52150aa1b6f6344c4", "probe SHA")
    req(sha(EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl") == "733bd017d8da9bf3478b6a196b6193ea079f7adc85c770b2ffedc075826fcd02", "subset SHA")

    cfg = load_config(2)
    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
    demo_actions = np.asarray([row["actions"] for row in demos], np.float32)
    probes = pickle.loads((EVIDENCE / "C2_BEHAVIOR_CRITICAL_PROBE_V1.pkl").read_bytes())
    subset = pickle.loads((EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl").read_bytes())
    preflight = json.loads((PREFLIGHT / "result.json").read_text())
    support_bundle = pickle.loads((PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    core = float(preflight["C1_CORE_THRESHOLD"])
    outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        loaded = flax.serialization.from_bytes(init_loaded(common), SOURCE_CP.read_bytes())
        agent = common.replace(state=loaded["agent_state"])
        c1_params = loaded["c1_params"]
        old_c2_params = loaded["c2_params"]
        base_guard = geo.thash(agent.state.params["modules_actor"])
        c1_guard = geo.thash(c1_params)
        req(base_guard == preflight["C0_BASE_SHA"] and c1_guard == preflight["C1_RESIDUAL_SHA"], "frozen stack binding")

        all_rows = []
        for source, rows in (("failure_probe", probes), ("existing_expert_subset", subset)):
            observations = np.asarray([row["observation"] if source == "failure_probe" else row["observations"] for row in rows], np.float32)
            targets = np.asarray([row["expert_action"] if source == "failure_probe" else row["actions"] for row in rows], np.float32)
            lower_loc_jax, lower_action_jax = lower_policy_components(agent, c1_params, jnp.asarray(observations))
            lower_loc = np.asarray(jax.device_get(lower_loc_jax), np.float64)
            lower_action = np.asarray(jax.device_get(lower_action_jax), np.float64)
            gates = np.asarray(jax.device_get(progressive_gate(observations, support, scale, core, outer)), np.float64)
            distances = np.asarray(jax.device_get(support_distance(observations, support, scale)), np.float64)
            for index in range(len(rows)):
                target = targets[index, :6].astype(np.float64)
                g2 = float(gates[index])
                label = region(observations[index], float(distances[index]), core, outer)
                if g2 <= NUMERICAL_EPS:
                    all_rows.append({"source": source, "index": index, "region": label, "authorized": False})
                    continue
                target_logits = np.arctanh(np.clip(target, -0.999999, 0.999999))
                p1 = (target_logits - lower_loc[index, :6]) / g2
                p1_reconstruction = np.tanh(lower_loc[index, :6] + g2 * p1)
                p2 = (target - (1.0 - g2) * lower_action[index, :6]) / g2
                p2_reconstruction = (1.0 - g2) * lower_action[index, :6] + g2 * p2
                all_rows.append({
                    "source": source,
                    "index": index,
                    "source_episode": int(rows[index].get("expert_source_episode", rows[index].get("source_episode", -1))),
                    "source_step": int(rows[index].get("expert_source_step", rows[index].get("source_step", -1))),
                    "region": label,
                    "authorized": True,
                    "g2": g2,
                    "p1_required_delta_z": p1,
                    "p1_finite": bool(np.all(np.isfinite(p1))),
                    "p1_reconstruction_error": float(np.max(np.abs(p1_reconstruction - target))),
                    "p2_required_action_head": p2,
                    "p2_representable": bool(np.all(np.abs(p2) <= 1.0 + NUMERICAL_EPS)),
                    "p2_reconstruction_error": float(np.max(np.abs(p2_reconstruction - target))),
                })

        controlled = [row for row in all_rows if row["authorized"]]
        p1_values = np.abs(np.asarray([row["p1_required_delta_z"] for row in controlled]))
        p2_values = np.abs(np.asarray([row["p2_required_action_head"] for row in controlled]))
        p2_failures = [row for row in controlled if not row["p2_representable"]]
        static = {
            "CURRENT_C2_POLICY_EQUATION": "z_lower=base_loc + g1*tanh(r_C1); z_final[:6]=z_lower[:6] + g2*tanh(r_C2); a_final=tanh(z_final), with base scale_diag and a6 unchanged",
            "ENVIRONMENT_ACTION_PATH": "env clips normalized action to its unchanged action_space then passes it to the unchanged controller; candidate changes neither action_space nor controller/physical scaling",
            "region_counts": {name: sum(row["region"] == name for row in all_rows) for name in ("C2_ACTIVE_REGION", "C1_SHELL_REGION", "C1_CORE_REGION", "C0_CORE_REGION")},
            "unauthorized_g2_zero_count": sum(not row["authorized"] for row in all_rows),
            "P1_REPRESENTABLE_FRACTION": float(np.mean([row["p1_finite"] for row in controlled])),
            "P1_REQUIRED_DELTA_Z": {"mean": float(np.mean(p1_values)), "p95": float(np.quantile(p1_values, 0.95)), "max": float(np.max(p1_values)), "per_axis_max": np.max(p1_values, axis=0)},
            "P1_FINITE_FRACTION": float(np.mean([row["p1_finite"] for row in controlled])),
            "P1_RECONSTRUCTION_ERROR_MAX": float(max(row["p1_reconstruction_error"] for row in controlled)),
            "P2_REPRESENTABLE_FRACTION": float(np.mean([row["p2_representable"] for row in controlled])),
            "P2_REQUIRED_ACTION_HEAD": {"mean_magnitude": float(np.mean(p2_values)), "p95": float(np.quantile(p2_values, 0.95)), "max": float(np.max(p2_values))},
            "P2_UNREPRESENTABLE_STATE_COUNT": len(p2_failures),
            "P2_FAILURES_BY_REGION": {name: sum(row["region"] == name for row in p2_failures) for name in ("C2_ACTIVE_REGION", "C1_SHELL_REGION", "C1_CORE_REGION")},
            "P2_RECONSTRUCTION_ERROR_MAX": float(max(row["p2_reconstruction_error"] for row in controlled)),
        }
        dump(OUT / "static_representability.json", static)
        dump(OUT / "static_representability_rows.json", all_rows)
        selected = "GATED_ABSOLUTE_C2_ACTION_HEAD_V1" if static["P2_REPRESENTABLE_FRACTION"] == 1.0 else "PRETANH_ADDITIVE_RESIDUAL_V1" if static["P1_REPRESENTABLE_FRACTION"] == 1.0 else "NONE"
        req(selected == "GATED_ABSOLUTE_C2_ACTION_HEAD_V1", "P2 must be selected solely by complete static representability")

        params = ABSOLUTE_C2_HEAD_DEF.init(jax.random.PRNGKey(260960), np.zeros((1, 20), np.float32))["params"]
        c0_bundle = pickle.loads((PREFLIGHT / "C0_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
        c0_obs = np.asarray(c0_bundle["observations"], np.float32)
        c1_probe = pickle.loads((PREFLIGHT / "C1_BEHAVIOR_CRITICAL_PROBE_V1.pkl").read_bytes())
        c1_obs = np.asarray([row["observation"] for row in c1_probe], np.float32)
        _, c0_lower_jax = lower_policy_components(agent, c1_params, c0_obs)
        _, c1_lower_jax = lower_policy_components(agent, c1_params, c1_obs)
        c0_lower = np.asarray(jax.device_get(c0_lower_jax))
        c1_lower = np.asarray(jax.device_get(c1_lower_jax))
        c0_initial = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, params, c0_obs, support, scale, core, outer)))
        c1_initial = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, params, c1_obs, support, scale, core, outer)))
        initial_c0_delta = float(np.max(np.abs(c0_initial - c0_lower)))
        initial_c1_delta = float(np.max(np.abs(c1_initial - c1_lower)))
        leaves = jax.tree_util.tree_leaves(params)
        keys = iter(jax.random.split(jax.random.PRNGKey(260961), len(leaves)))
        perturbed = jax.tree_util.tree_map(lambda value: value + 0.5 * jax.random.normal(next(keys), value.shape), params)
        c0_perturbed = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, perturbed, c0_obs, support, scale, core, outer)))
        c1_perturbed = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, perturbed, c1_obs, support, scale, core, outer)))
        max_c0_delta = float(np.max(np.abs(c0_perturbed - c0_lower)))
        max_c1_delta = float(np.max(np.abs(c1_perturbed - c1_lower)))
        req(initial_c0_delta <= 1e-7 and initial_c1_delta <= 1e-7 and max_c0_delta <= 1e-7 and max_c1_delta <= 1e-7, "lower-skill invariance")

        probe_obs = np.asarray([row["observation"] for row in probes], np.float32)
        probe_targets = np.asarray([row["expert_action"] for row in probes], np.float32)
        subset_obs = np.asarray([row["observations"] for row in subset], np.float32)
        subset_targets = np.asarray([row["actions"] for row in subset], np.float32)
        _, probe_lower_jax = lower_policy_components(agent, c1_params, probe_obs)
        probe_lower = np.asarray(jax.device_get(probe_lower_jax))
        probe_g2 = np.asarray(jax.device_get(progressive_gate(probe_obs, support, scale, core, outer)))
        required_probe_head = (probe_targets[:, :6] - (1.0 - probe_g2[:, None]) * probe_lower[:, :6]) / probe_g2[:, None]
        implemented_probe = np.asarray(jax.device_get(gated_absolute_compose(jnp.asarray(probe_lower), jnp.asarray(required_probe_head), jnp.asarray(probe_g2))))
        reset_reconstruction = float(np.max(np.abs(implemented_probe[0, :6] - probe_targets[0, :6])))
        actual_fraction = float(np.mean(np.all(np.abs(np.asarray([row["p2_required_action_head"] for row in controlled if row["source"] == "existing_expert_subset"])) <= 1.0 + NUMERICAL_EPS, axis=1)))
        req(reset_reconstruction <= 1e-6 and actual_fraction == 1.0, "implemented P2 representation gate")

        train_indices = [i for i, row in enumerate(subset) if int(row["source_episode"]) < 9]
        valid_indices = [i for i, row in enumerate(subset) if int(row["source_episode"]) >= 9]
        req(train_indices and valid_indices and not set(train_indices).intersection(valid_indices), "episode-separated split")
        baseline_valid = action_metrics(params, subset_obs[valid_indices], subset_targets[valid_indices], agent, c1_params, support, scale, core, outer)
        baseline_probe = action_metrics(params, probe_obs, probe_targets, agent, c1_params, support, scale, core, outer)
        opt_state = TX.init(params)
        rng = np.random.default_rng(260962)
        reports = []
        saved = {}
        for step in range(1, 1001):
            indices = rng.choice(train_indices, size=64, replace=True)
            params, opt_state, loss, gradient_norm = bc_step(params, opt_state, jnp.asarray(subset_obs[indices]), jnp.asarray(subset_targets[indices]), agent, c1_params, support, scale, core, outer)
            if step in CHECKPOINTS:
                validation = action_metrics(params, subset_obs[valid_indices], subset_targets[valid_indices], agent, c1_params, support, scale, core, outer)
                probe_metrics = action_metrics(params, probe_obs, probe_targets, agent, c1_params, support, scale, core, outer)
                checkpoint = OUT / "checkpoints" / f"gated_absolute_c2_head_bc_{step:04d}.msgpack"
                checkpoint.write_bytes(flax.serialization.to_bytes({"c2_head_params": params, "c2_head_opt_state": opt_state}))
                saved[step] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
                reports.append({"step": step, "loss": float(loss), "gradient_norm": float(gradient_norm), "validation": validation, "failure_probe": probe_metrics, "checkpoint": saved[step]})
        best = min(reports, key=lambda row: row["validation"]["action_mae"])
        best_step = int(best["step"])
        template = {"c2_head_params": ABSOLUTE_C2_HEAD_DEF.init(jax.random.PRNGKey(0), np.zeros((1, 20), np.float32))["params"], "c2_head_opt_state": TX.init(ABSOLUTE_C2_HEAD_DEF.init(jax.random.PRNGKey(0), np.zeros((1, 20), np.float32))["params"])}
        best_loaded = flax.serialization.from_bytes(template, Path(best["checkpoint"]["path"]).read_bytes())
        best_params = best_loaded["c2_head_params"]
        reset_action = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, best_params, probe_obs[0], support, scale, core, outer)))
        reset_action_error = float(np.mean(np.abs(reset_action[:6] - probe_targets[0, :6])))
        target_saturation = float(np.mean(np.abs(subset_targets[valid_indices, :6]) >= 0.999))
        offline_fit = bool(
            best["validation"]["action_mae"] <= VALID_MAE_RATIO_MAX * baseline_valid["action_mae"]
            and reset_action_error <= RESET_ACTION_ERROR_MAX
            and best["validation"]["finite"]
            and best["validation"]["head_saturation_fraction"] <= target_saturation + 0.05
        )
        dump(OUT / "offline_fit.json", {"split": {"train_episodes": list(range(9)), "validation_episodes": [9, 10], "train_rows": len(train_indices), "validation_rows": len(valid_indices)}, "thresholds_preregistered_before_fit": {"validation_mae_ratio_max": VALID_MAE_RATIO_MAX, "reset_action_error_max": RESET_ACTION_ERROR_MAX, "head_saturation_excess_max": 0.05}, "zero_init_validation": baseline_valid, "zero_init_failure_probe": baseline_probe, "reports": reports, "selected_step_by_validation_action_error_only": best_step, "selected": best, "C2_RESET_ACTION_ERROR": reset_action_error, "C2_OFFLINE_FIT": "PASS" if offline_fit else "FAIL"})

        short_result = "NOT_RUN"
        short_reports = []
        short_useful = False
        c2_eval = c1_eval = c0_eval = matched_eval = canonical_eval = None
        if offline_fit:
            snapshots = reproduce_probe_snapshots(agent, c1_params, old_c2_params, support, scale, core, outer, probes)
            for probe in probes:
                snap = snapshots[int(probe["source_step"])]
                lower_roll = short_rollout(snap, "lower", best_params, agent, c1_params, support, scale, core, outer, demo_obs, demo_actions, demo_scale)
                c2_roll = short_rollout(snap, "offline_c2", best_params, agent, c1_params, support, scale, core, outer, demo_obs, demo_actions, demo_scale)
                expert_roll = short_rollout(snap, "expert", best_params, agent, c1_params, support, scale, core, outer, demo_obs, demo_actions, demo_scale)
                useful = bool(
                    (c2_roll["min_D_C1"] < lower_roll["min_D_C1"] - 1e-6 or c2_roll["min_position"] < lower_roll["min_position"] - 1e-6)
                    and c2_roll["max_force"] <= lower_roll["max_force"] + 5.0
                    and not c2_roll["unexpected_contact"]
                )
                short_reports.append({"source_step": int(probe["source_step"]), "tags": probe["tags"], "lower": lower_roll, "offline_c2": c2_roll, "expert_where_valid": expert_roll, "useful": useful})
            short_useful = sum(row["useful"] for row in short_reports) >= 2
            short_result = "GOOD" if short_useful else "POOR"
            dump(OUT / "short_horizon_audit.json", {"horizon": SHORT_HORIZON, "useful_probe_count": sum(row["useful"] for row in short_reports), "probe_count": len(short_reports), "C2_SHORT_HORIZON_POLICY_IMPROVEMENT": short_result, "reports": short_reports})

        acquisition = "NOT_RUN"
        matched = "NOT_RUN"
        confirmed = "NO"
        canonical_status = "NOT_RUN"
        if offline_fit and short_useful:
            zero_params = ABSOLUTE_C2_HEAD_DEF.init(jax.random.PRNGKey(260963), np.zeros((1, 20), np.float32))["params"]
            baseline_c2 = eval_record(agent, c1_params, zero_params, support, scale, core, outer, "C2", demo_obs, demo_scale)
            c2_eval = eval_record(agent, c1_params, best_params, support, scale, core, outer, "C2", demo_obs, demo_scale)
            c1_eval = eval_record(agent, c1_params, best_params, support, scale, core, outer, "C1", demo_obs, demo_scale)
            c0_eval = eval_record(agent, c1_params, best_params, support, scale, core, outer, "C0", demo_obs, demo_scale)
            progressed, progress_checks = pilot.progress(baseline_c2, c2_eval)
            c1_retained = bool(c1_eval["contact_health"])
            c0_retained = bool(c0_eval["success_count"] == 3 and c0_eval["contact_health"])
            acquisition_pass = bool(progressed and c1_retained and c0_retained and c2_eval["contact_health"])
            acquisition = "PASS" if acquisition_pass else "FAIL"
            dump(OUT / "closed_loop_acquisition.json", {"baseline_C2": baseline_c2, "offline_C2": c2_eval, "C2_progress_checks": progress_checks, "C1_regression": c1_eval, "C0_regression": c0_eval, "C2_BC_ACQUISITION": acquisition})
            if acquisition_pass:
                matched_eval = eval_record(agent, c1_params, best_params, support, scale, core, outer, "C2", demo_obs, demo_scale)
                match_keys = ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
                matched_ok = all(matched_eval[key] == c2_eval[key] for key in match_keys)
                matched = "PASS" if matched_ok else "FAIL"
                confirmed = "YES" if matched_ok else "NO"
                if matched_ok:
                    canonical_eval = eval_record(agent, c1_params, best_params, support, scale, core, outer, "C3", demo_obs, demo_scale)
                    canonical_status = "COMPLETE"
                dump(OUT / "matched_confirmation_and_canonical_interface.json", {"C2_MATCHED_CONFIRMATION": matched, "C2_SKILL_CONFIRMED": confirmed, "matched_C2": matched_eval, "C2_FROZEN_SHA": tree_hash(best_params) if matched_ok else None, "CANONICAL_INTERFACE_AUDIT": canonical_status, "canonical": canonical_eval})

        c2_interface = c2_eval["interface"] if c2_eval else {"C1_SHELL_ENTRY_COUNT": 0, "C1_CORE_ENTRY_COUNT": 0}
        if not offline_fit:
            first_bad = "C2_OFFLINE_FIT"
            root = "C2_POLICY_HEAD_OFFLINE_CAPACITY_FAILURE"
            next_task = "C2_POLICY_CAPACITY_REVIEW"
        elif not short_useful:
            first_bad = "C2_SHORT_HORIZON_POLICY_IMPROVEMENT"
            root = "C2_POLICY_ACTION_SEQUENCE_NOT_USEFUL"
            next_task = "C2_EXPERT_ACTION_SEQUENCE_CAUSAL_REVIEW"
        elif acquisition != "PASS":
            first_bad = "C2_BC_ACQUISITION"
            root = "C2_CLOSED_LOOP_DISTRIBUTION_SHIFT"
            next_task = "C2_ON_POLICY_CORRECTIVE_SUPERVISION_V1"
        elif confirmed != "YES":
            first_bad = "C2_MATCHED_CONFIRMATION"
            root = "C2_ACQUISITION_NOT_REPRODUCIBLE"
            next_task = "STOP"
        else:
            first_bad = "CANONICAL_C3_INTERFACE" if canonical_eval and canonical_eval["interface"]["class_state_counts"]["C2_OUTSIDE"] == canonical_eval["interface"]["visited_states"] else None
            root = "UPSTREAM_TO_FROZEN_SKILL_INTERFACE_FAILURE" if first_bad else "NONE"
            next_task = "C3_ACQUISITION_PILOT" if first_bad else "C3_INTERFACE_RESULT_REVIEW"

        req(geo.thash(agent.state.params["modules_actor"]) == base_guard and geo.thash(c1_params) == c1_guard, "protected model mutation")
        result = {
            "C2_RESIDUAL_OUTPUT_PARAMETERIZATION_REVIEW": "PASS",
            "CURRENT_C2_POLICY_EQUATION": static["CURRENT_C2_POLICY_EQUATION"],
            "P1_REPRESENTABLE_FRACTION": static["P1_REPRESENTABLE_FRACTION"],
            "P1_REQUIRED_DELTA_Z_P95": static["P1_REQUIRED_DELTA_Z"]["p95"],
            "P1_REQUIRED_DELTA_Z_MAX": static["P1_REQUIRED_DELTA_Z"]["max"],
            "P2_REPRESENTABLE_FRACTION": static["P2_REPRESENTABLE_FRACTION"],
            "P2_UNREPRESENTABLE_STATE_COUNT": static["P2_UNREPRESENTABLE_STATE_COUNT"],
            "SELECTED_PARAMETERIZATION": selected,
            "C2_RESET_EXPERT_RECONSTRUCTION_ERROR": reset_reconstruction,
            "C2_EXPERT_REPRESENTABLE_FRACTION": actual_fraction,
            "INITIAL_ACTION_MAX_DELTA_C0": initial_c0_delta,
            "INITIAL_ACTION_MAX_DELTA_C1": initial_c1_delta,
            "C0_INVARIANCE": "PASS" if max_c0_delta <= 1e-7 else "FAIL",
            "MAX_C0_ACTION_DELTA": max_c0_delta,
            "C1_CORE_INVARIANCE": "PASS" if max_c1_delta <= 1e-7 else "FAIL",
            "MAX_C1_CORE_ACTION_DELTA": max_c1_delta,
            "C2_OFFLINE_FIT": "PASS" if offline_fit else "FAIL",
            "SELECTED_OFFLINE_CHECKPOINT": saved[best_step],
            "VALID_MAE": best["validation"]["action_mae"],
            "VALID_COSINE": best["validation"]["action_cosine"],
            "PER_AXIS_MAE": best["validation"]["per_axis_action_mae"],
            "C2_RESET_ACTION_ERROR": reset_action_error,
            "FAILURE_PROBE_ACTION_ERROR": best["failure_probe"]["action_mae"],
            "C2_SHORT_HORIZON_POLICY_IMPROVEMENT": short_result,
            "C2_BC_ACQUISITION": acquisition,
            "C2_SHELL_ENTRY": c2_interface["C1_SHELL_ENTRY_COUNT"],
            "C2_CORE_ENTRY": c2_interface["C1_CORE_ENTRY_COUNT"],
            "C2_NEAR_HOLE": c2_eval["near_hole_rate"] if c2_eval else 0.0,
            "C2_SUCCESS": f"{c2_eval['success_count']}/3" if c2_eval else "NOT_RUN",
            "C2_CONTACT": "PASS" if c2_eval and c2_eval["contact_health"] else "NOT_RUN" if not c2_eval else "FAIL",
            "C1_REGRESSION": "PASS" if c1_eval and c1_eval["contact_health"] else "NOT_RUN" if not c1_eval else "FAIL",
            "C1_CONTACT": "PASS" if c1_eval and c1_eval["contact_health"] else "NOT_RUN" if not c1_eval else "FAIL",
            "C0_SUCCESS": f"{c0_eval['success_count']}/3" if c0_eval else "NOT_RUN",
            "C2_ON_POLICY_CORRECTIVE_USED": "NO",
            "C2_MATCHED_CONFIRMATION": matched,
            "C2_SKILL_CONFIRMED": confirmed,
            "C2_FROZEN_SHA": tree_hash(best_params) if confirmed == "YES" else None,
            "CANONICAL_INTERFACE_AUDIT": canonical_status,
            "PEG_SINGLE_TASK_CONVERGED": "NO",
            "FIRST_BAD_GATE": first_bad,
            "PRIMARY_ROOT_CAUSE": root,
            "NEXT": next_task,
            "C0_BASE_SHA": base_guard,
            "C1_RESIDUAL_SHA": c1_guard,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "ONLINE_ADMISSION_THRESHOLD": p4.TH,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE",
            "elapsed_seconds": time.time() - started,
        }
        protected_after = {key: sha(path) for key, path in protected_paths.items()}
        req(protected_after == protected_before, "protected artifact changed")
        dump(OUT / "protected_after.json", protected_after)
        dump(OUT / "result.json", result)
        print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
