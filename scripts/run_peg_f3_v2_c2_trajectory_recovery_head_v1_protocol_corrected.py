#!/usr/bin/env python3
"""Bounded CEGIS acquisition for C2_TRAJECTORY_RECOVERY_HEAD_V1."""
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
import scripts.audit_peg_f3_v2_c2_closed_loop_recovery_basin as basin
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, evaluate, load_config, make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.c2_trajectory_recovery_head_v1 import (
    TRAJECTORY_RECOVERY_HEAD_DEF, compose_recovery_action, recovery_distance,
)
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/training/peg_f3_v2_c2_trajectory_recovery_head_v1_20260909_002"
DATA_OUT = ROOT / "results/datasets/peg_v2_c2_trajectory_recovery_head_v1_20260909_002"
AUDIT_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_trajectory_level_recoverability_closure_20260909_001"
CHECKPOINT = tail.CONVERGENCE_RUN / "checkpoints/constant_plateau_c2_2000.msgpack"
MAX_ROUNDS = 3
UPDATES = 3000
LR = 3e-4
SEEDS = (100, 101, 102)
HORIZONS = (1, 2, 4, 8)
MONITOR_STEPS = (500, 1000, 1500, 2000, 2500, 3000)


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
        json.dump(jvalue(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def tree_hash(tree):
    return hashlib.sha256(flax.serialization.to_bytes(jax.device_get(tree))).hexdigest()


class CompositePolicy:
    def __init__(self, action_fn, support, scale, core, outer):
        self.action_fn = action_fn
        self.support, self.scale, self.core, self.outer = support, scale, core, outer
        self.classes = []

    def sample_actions(self, observations, seed=None, deterministic=False, argmax=False, **_):
        del seed, deterministic, argmax
        obs = np.asarray(observations, np.float32)
        action = self.action_fn(obs)
        d = float(np.asarray(jax.device_get(support_distance(obs, self.support, self.scale))))
        g0 = float(np.asarray(jax.device_get(gate_value(obs))))
        self.classes.append("C0_CORE" if g0 == 0 else "C1_CORE" if d <= self.core else "C1_RECOVERY_SHELL" if d <= self.outer else "C2_OUTSIDE")
        return action

    def stats(self):
        shell = sum(x in ("C1_RECOVERY_SHELL", "C1_CORE") and (i == 0 or self.classes[i-1] == "C2_OUTSIDE") for i, x in enumerate(self.classes))
        core = sum(x == "C1_CORE" and (i == 0 or self.classes[i-1] not in ("C1_CORE", "C0_CORE")) for i, x in enumerate(self.classes))
        return {"visited_states": len(self.classes), "C1_SHELL_ENTRY_COUNT": shell, "C1_CORE_ENTRY_COUNT": core,
                "class_state_counts": {x: self.classes.count(x) for x in ("C2_OUTSIDE", "C1_RECOVERY_SHELL", "C1_CORE", "C0_CORE")}}


def make_action_fn(global_def, global_params, local_params, recovery_support, recovery_scale, recovery_core, recovery_outer,
                   agent, c1_params, c1_support, c1_scale, c1_core, c1_outer):
    def action_fn(obs):
        global_action = tail.checkpoint_action(global_def, global_params, obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
        action, _ = compose_recovery_action(TRAJECTORY_RECOVERY_HEAD_DEF, local_params, jnp.asarray(obs), jnp.asarray(global_action),
            recovery_support, recovery_scale, recovery_core, recovery_outer, c1_support, c1_scale, c1_core, c1_outer)
        return np.asarray(jax.device_get(action), np.float32)
    return action_fn


def collect_policy_rollout(action_fn):
    episodes = []
    for seed in SEEDS:
        env = make_train_env("C2"); rows = []
        try:
            obs, _ = env.reset(seed=seed)
            for step in range(500):
                link, unexpected, pairs = contact_flags(env)
                action = action_fn(obs)
                rows.append({"seed": seed, "policy_step": step, "observation": np.asarray(obs, np.float32).copy(),
                    "snapshot": capture(env, obs), "policy_action": action, "contact_before": bool(link or unexpected), "contact_pairs": pairs})
                if link or unexpected: break
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated: break
        finally: env.close()
        episodes.append(rows)
    return episodes


def first_boundary(episode, oracle):
    audited = []
    for row in episode:
        takeover = oracle.takeover(row["snapshot"])
        audited.append({"step": row["policy_step"], "recoverable": takeover["EXPERT_SHELL_RECOVERABLE"], "takeover": takeover})
        if not takeover["EXPERT_SHELL_RECOVERABLE"]: break
    bad = next((x for x in audited if not x["recoverable"]), None)
    if bad is None: return None, audited
    good = max((x for x in audited if x["step"] < bad["step"] and x["recoverable"]), key=lambda x: x["step"])
    return (good["step"], bad["step"], good["takeover"]), audited


def collect_expert_recovery(source, source_takeover, demos, support, scale, outer):
    env = make_train_env("C2"); rows = []
    try:
        obs = restore(env, source["snapshot"])
        start = int(source_takeover["expert_source_row"])
        ranges = basin.episode_ranges(demos)
        _, _, _, end = basin.row_provenance(ranges, start)
        for index in range(start, end):
            d = float(np.asarray(jax.device_get(support_distance(obs, support, scale))))
            g0 = float(np.asarray(jax.device_get(gate_value(obs))))
            if d <= outer or g0 == 0.0: break
            action = np.asarray(demos[index]["actions"], np.float32)
            rows.append({"observations": np.asarray(obs, np.float32).copy(), "actions": action,
                         "expert_source_row": index, "expert_source_episode": source_takeover["expert_source_episode"],
                         "expert_source_step": int(source_takeover["expert_source_step"] + len(rows))})
            obs, _, terminated, truncated, _ = env.step(action)
            link, unexpected, _ = contact_flags(env)
            req(not link and not unexpected, "expert recovery contact health")
            if terminated or truncated: break
        final_d = float(np.asarray(jax.device_get(support_distance(obs, support, scale))))
        final_g0 = float(np.asarray(jax.device_get(gate_value(obs))))
        req(final_d <= outer or final_g0 == 0.0, "expert sequence reaches C1 shell/core")
        return rows, {"rows": len(rows), "final_D_C1": final_d, "final_g0": final_g0}
    finally: env.close()


def derive_gate(recovery_rows, negative_obs, scale):
    positive = np.asarray([row["observations"] for row in recovery_rows], np.float32)
    negative = np.asarray(negative_obs, np.float32)
    episodes = np.asarray([row["source_seed"] for row in recovery_rows])
    leave_one_out = []
    for index in range(len(positive)):
        candidates = positive[episodes != episodes[index]]
        req(len(candidates), "episode-separated recovery support")
        delta = (candidates - positive[index]) / scale
        leave_one_out.append(float(np.min(np.sqrt(np.sum(np.square(delta), axis=1)))))
    core = float(max(leave_one_out) + np.finfo(np.float32).eps)
    delta = (negative[:, None, :] - positive[None, :, :]) / scale[None, None, :]
    distances = np.min(np.sqrt(np.sum(np.square(delta), axis=-1)), axis=1)
    strictly_outside = distances[distances > core]
    req(len(strictly_outside), "RECOVERY_GATE_GEOMETRY_NOT_SEPARABLE")
    outer = float(np.min(strictly_outside))
    req(outer > core > 0.0, "derived local recovery thresholds")
    return core, outer, {"leave_one_episode_out_max": max(leave_one_out), "leave_one_episode_out_distances": leave_one_out,
                         "minimum_nonrecovery_distance_strictly_above_core": outer,
                         "derivation": "core=max leave-one-episode-out NN + float32 eps; outer=smallest observed non-recovery distance strictly above core"}


def train_local(rows, global_def, global_params, recovery_support, recovery_scale, recovery_core, recovery_outer,
                agent, c1_params, c1_support, c1_scale, c1_core, c1_outer, seed=381700):
    params = TRAJECTORY_RECOVERY_HEAD_DEF.init(jax.random.PRNGKey(seed), np.zeros((1, 20), np.float32))["params"]
    tx = optax.adam(LR); state = tx.init(params); rng = np.random.default_rng(seed)
    observations = np.asarray([x["observations"] for x in rows], np.float32)
    targets = np.asarray([x["actions"] for x in rows], np.float32)
    global_actions = np.asarray([tail.checkpoint_action(global_def, global_params, x, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer) for x in observations], np.float32)
    @jax.jit
    def step(candidate, opt_state, obs, frozen, target):
        def loss_fn(value):
            action, _ = compose_recovery_action(TRAJECTORY_RECOVERY_HEAD_DEF, value, obs, frozen, recovery_support,
                recovery_scale, recovery_core, recovery_outer, c1_support, c1_scale, c1_core, c1_outer)
            return jnp.mean(jnp.square(action[..., :6] - target[..., :6]))
        loss, grad = jax.value_and_grad(loss_fn)(candidate)
        updates, new_state = tx.update(grad, opt_state, candidate)
        return optax.apply_updates(candidate, updates), new_state, loss
    reports, bundles = {}, {}
    train_indices = np.asarray([i for i, row in enumerate(rows) if row["source_seed"] in (100, 101)], np.int64)
    heldout_indices = np.asarray([i for i, row in enumerate(rows) if row["source_seed"] == 102], np.int64)
    req(len(train_indices) and len(heldout_indices), "episode-separated train/heldout recovery rows")
    for update in range(1, UPDATES + 1):
        chosen = rng.choice(train_indices, 64, replace=True)
        params, state, loss = step(params, state, jnp.asarray(observations[chosen]), jnp.asarray(global_actions[chosen]), jnp.asarray(targets[chosen]))
        if update in MONITOR_STEPS:
            pred, gates = compose_recovery_action(TRAJECTORY_RECOVERY_HEAD_DEF, params, jnp.asarray(observations), jnp.asarray(global_actions),
                recovery_support, recovery_scale, recovery_core, recovery_outer, c1_support, c1_scale, c1_core, c1_outer)
            error = np.abs(np.asarray(jax.device_get(pred))[:, :6] - targets[:, :6])
            reports[update] = {"batch_loss": float(loss), "train_mae": float(np.mean(error[train_indices])),
                               "heldout_recovery_mae": float(np.mean(error[heldout_indices])),
                               "heldout_per_axis_mae": np.mean(error[heldout_indices], axis=0), "p95": float(np.quantile(error[heldout_indices], .95)),
                               "max": float(np.max(error[heldout_indices])), "min_gate": float(np.min(np.asarray(jax.device_get(gates))))}
            bundles[update] = (params, state)
    return reports, bundles


def horizon_validation(source, action_fn, oracle):
    reports = {}
    for horizon in HORIZONS:
        obs = restore(oracle.env, source["snapshot"]); transitions = []
        healthy = True
        for _ in range(horizon):
            action = action_fn(obs)
            obs, _, terminated, truncated, info = oracle.env.step(action)
            link, unexpected, pairs = contact_flags(oracle.env)
            force = float(info.get("force_norm", 0.0)); healthy &= bool(not link and not unexpected and np.isfinite(force))
            transitions.append({"force": force, "contact_healthy": not link and not unexpected, "pairs": pairs})
            if terminated or truncated or not healthy: break
        takeover = oracle._takeover_current(obs) if healthy else {"EXPERT_SHELL_RECOVERABLE": False, "EXPERT_CORE_RECOVERABLE": False}
        reports[horizon] = {"policy_steps": len(transitions), "contact_healthy": healthy, "takeover": takeover,
                            "passed": bool(healthy and takeover["EXPERT_SHELL_RECOVERABLE"])}
    return reports


def static_invariance(global_def, global_params, recovery_support, recovery_scale, recovery_core, recovery_outer,
                      negative_obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer):
    template = TRAJECTORY_RECOVERY_HEAD_DEF.init(jax.random.PRNGKey(991), np.zeros((1, 20), np.float32))["params"]
    perturbed = jax.tree_util.tree_map(lambda x: x + jnp.full_like(x, 0.123), template)
    core_probes = np.asarray(c1_support[:min(64, len(c1_support))], np.float32)
    outside = [obs for obs in negative_obs if float(np.asarray(jax.device_get(recovery_distance(obs, recovery_support, recovery_scale)))) >= recovery_outer]
    req(outside, "outside recovery support invariance probes")
    outside_probes = np.asarray(outside[:min(64, len(outside))], np.float32)
    def delta(probes):
        global_actions = np.asarray([tail.checkpoint_action(global_def, global_params, obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer) for obs in probes], np.float32)
        composed, _ = compose_recovery_action(TRAJECTORY_RECOVERY_HEAD_DEF, perturbed, jnp.asarray(probes), jnp.asarray(global_actions),
            recovery_support, recovery_scale, recovery_core, recovery_outer, c1_support, c1_scale, c1_core, c1_outer)
        return float(np.max(np.abs(np.asarray(jax.device_get(composed)) - global_actions)))
    core_delta, outside_delta = delta(core_probes), delta(outside_probes)
    req(core_delta == 0.0 and outside_delta == 0.0, "exact recovery-head invariance")
    return {"C0_C1_CORE_MAX_ACTION_DELTA": core_delta, "OUTSIDE_RECOVERY_OUTER_MAX_ACTION_DELTA": outside_delta, "PASS": True}


def pilot(action_fn, support, scale, core, outer, demo_obs, demo_scale, baseline_c1):
    records = {}
    for level in ("C0", "C1", "C2"):
        policy = CompositePolicy(action_fn, support, scale, core, outer)
        record = evaluate(policy, level, demo_obs, demo_scale, seeds=SEEDS); record["interface"] = policy.stats(); records[level] = record
    c2 = records["C2"]
    c1_keys = ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
    c1_regression = bool(all(records["C1"][key] == baseline_c1[key] for key in c1_keys))
    passed = bool(records["C0"]["success_count"] == 3 and records["C0"]["contact_health"] and c1_regression
                  and c2["contact_health"] and c2["interface"]["C1_SHELL_ENTRY_COUNT"] >= 2)
    return {**records, "C1_REGRESSION": "PASS" if c1_regression else "FAIL", "passed": passed, "C2_PILOT": "PASS" if passed else "FAIL"}


def main():
    req(Path.cwd() == ROOT and not OUT.exists() and not DATA_OUT.exists(), "fresh output paths and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    audit = json.loads((AUDIT_RUN / "result.json").read_text())
    req(audit["C2_TRAJECTORY_RECOVERY_HEAD_V1_AUTHORIZED"] == "YES", "trajectory evidence authorization")
    OUT.mkdir(parents=True); (OUT / "checkpoints").mkdir(); DATA_OUT.mkdir(parents=True); started = time.time()
    protected_paths = {"production_demo": p4.DEMO, "global_c2_2k": CHECKPOINT, "multi_boundary_set": tail.DATASET_PATH, "trajectory_audit": AUDIT_RUN / "result.json"}
    protected_before = {k: sha(v) for k, v in protected_paths.items()}; dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == tail.DEMO_SHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    c1_support = np.asarray(support_bundle["observations"], np.float32); c1_scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text()); c1_core = float(preflight["C1_CORE_THRESHOLD"]); c1_outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos); source_bundle = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source_bundle["agent_state"]); c1_params = source_bundle["c1_params"]
        base_sha, c1_sha = geo.thash(agent.state.params["modules_actor"]), geo.thash(c1_params)
        global_def, global_params = tail.load_head_checkpoint(CHECKPOINT)
        global_sha = tree_hash(global_params)
        baseline_c1 = pc.eval_selected(global_def, global_params, "H0", agent, c1_params, c1_support, c1_scale, c1_core, c1_outer, "C1", demo_obs, demo_scale, np.zeros(20, np.float32))
        cumulative, negative_obs, round_reports, heldout_starts = [], [], [], []
        local_params = TRAJECTORY_RECOVERY_HEAD_DEF.init(jax.random.PRNGKey(381700), np.zeros((1,20), np.float32))["params"]
        recovery_support = np.zeros((1,20), np.float32); recovery_core, recovery_outer = 1e-6, 2e-6
        confirmed = False; selected_round = None; final_pilot = None; matched = "NOT_RUN"; no_new_counterexample = False
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, c1_support, c1_scale, c1_core, c1_outer)
        try:
            for round_index in range(1, MAX_ROUNDS + 1):
                if cumulative:
                    action_fn = make_action_fn(global_def, global_params, local_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                        agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                else:
                    action_fn = lambda obs: tail.checkpoint_action(global_def, global_params, obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                episodes = collect_policy_rollout(action_fn)
                boundaries = [first_boundary(ep, oracle)[0] for ep in episodes]
                req(all(x is not None for x in boundaries), "each failed rollout exposes a recoverable boundary")
                if cumulative:
                    boundary_distances = [float(np.asarray(jax.device_get(recovery_distance(ep[b[0]]["observation"], recovery_support, c1_scale)))) for ep, b in zip(episodes, boundaries)]
                    if not any(distance > recovery_outer for distance in boundary_distances):
                        no_new_counterexample = True
                        dump(OUT / f"cegis_round{round_index}_not_run.json", {"reason": "no genuinely new first-boundary counterexample outside frozen recovery support", "boundary_distances": boundary_distances, "recovery_outer": recovery_outer})
                        break
                new_rows = []
                for episode, boundary in zip(episodes, boundaries):
                    good, bad, takeover = boundary
                    rows, sequence_info = collect_expert_recovery(episode[good], takeover, demos, c1_support, c1_scale, c1_outer)
                    for row in rows: row.update({"cegis_round": round_index, "source_seed": episode[0]["seed"], "boundary_step": good})
                    new_rows.extend(rows)
                    if episode[0]["seed"] == 102: heldout_starts.append(episode[good])
                cumulative.extend(new_rows)
                recovery_hashes = {hashlib.sha256(row["observations"].tobytes()).hexdigest() for row in cumulative}
                for episode in episodes:
                    negative_obs.extend(row["observation"] for row in episode if hashlib.sha256(row["observation"].tobytes()).hexdigest() not in recovery_hashes)
                recovery_support = np.asarray([x["observations"] for x in cumulative], np.float32)
                recovery_core, recovery_outer, gate_info = derive_gate(cumulative, negative_obs, c1_scale)
                invariance = static_invariance(global_def, global_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                    negative_obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                dataset_path = DATA_OUT / f"C2_TRAJECTORY_RECOVERY_CEGIS_R{round_index}_V1.pkl"; dataset_path.write_bytes(pickle.dumps(cumulative, pickle.HIGHEST_PROTOCOL))
                dump(DATA_OUT / f"C2_TRAJECTORY_RECOVERY_CEGIS_R{round_index}_V1_manifest.json", {"rows": len(cumulative), "sha256": sha(dataset_path), "round": round_index,
                    "train_episodes": [100, 101], "heldout_episode": 102, "authoritative": "last expert-recoverable policy state plus frozen expert sequence to C1 shell/core", "production_demo_modified": False})
                training, bundles = train_local(cumulative, global_def, global_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                    agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                checkpoint_reports = {}
                for step_index in MONITOR_STEPS:
                    candidate_params, candidate_state = bundles[step_index]
                    candidate_fn = make_action_fn(global_def, global_params, candidate_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                        agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                    validations = [horizon_validation(start, candidate_fn, oracle) for start in heldout_starts]
                    checkpoint_reports[step_index] = {
                        "H1_SHELL": float(np.mean([x[1]["passed"] for x in validations])),
                        "H2_SHELL": float(np.mean([x[2]["passed"] for x in validations])),
                        "H4_SHELL": float(np.mean([x[4]["passed"] for x in validations])),
                        "H8_SHELL": float(np.mean([x[8]["passed"] for x in validations])),
                        "H8_CORE": float(np.mean([x[8]["contact_healthy"] and x[8]["takeover"]["EXPERT_CORE_RECOVERABLE"] for x in validations])),
                        "validations": validations,
                    }
                    checkpoint_path = OUT / "checkpoints" / f"trajectory_recovery_head_r{round_index}_{step_index}.msgpack"
                    checkpoint_path.write_bytes(flax.serialization.to_bytes({"params": candidate_params, "opt_state": candidate_state, "support": recovery_support,
                        "recovery_core": recovery_core, "recovery_outer": recovery_outer}))
                    checkpoint_reports[step_index]["checkpoint"] = str(checkpoint_path); checkpoint_reports[step_index]["checkpoint_sha"] = sha(checkpoint_path)
                selected_step = min(MONITOR_STEPS, key=lambda step: (-checkpoint_reports[step]["H8_SHELL"], -checkpoint_reports[step]["H4_SHELL"],
                    -checkpoint_reports[step]["H8_CORE"], training[step]["heldout_recovery_mae"]))
                local_params, local_state = bundles[selected_step]
                selected_offline = checkpoint_reports[selected_step]
                offline_pass = bool(selected_offline["H1_SHELL"] >= .90 and selected_offline["H4_SHELL"] >= .85 and selected_offline["H8_SHELL"] >= .75)
                action_fn = make_action_fn(global_def, global_params, local_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                    agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                final_pilot = pilot(action_fn, c1_support, c1_scale, c1_core, c1_outer, demo_obs, demo_scale, baseline_c1) if offline_pass else None
                if final_pilot is not None:
                    post_pilot_episodes = collect_policy_rollout(action_fn)
                    post_pilot_boundaries = [first_boundary(ep, oracle)[0] for ep in post_pilot_episodes]
                    first_bad_steps = [None if boundary is None else boundary[1] for boundary in post_pilot_boundaries]
                    materially_later = all(step is None or step > 10 for step in first_bad_steps)
                    final_pilot["FIRST_NONRECOVERABLE_STEPS"] = first_bad_steps
                    final_pilot["FIRST_NONRECOVERABLE_MATERIALLY_LATER_THAN_BASELINE"] = "YES" if materially_later else "NO"
                    final_pilot["passed"] = bool(final_pilot["passed"] and materially_later)
                    final_pilot["C2_PILOT"] = "PASS" if final_pilot["passed"] else "FAIL"
                report = {"round": round_index, "boundary_steps": [[x[0], x[1]] for x in boundaries], "new_rows": len(new_rows), "cumulative_rows": len(cumulative),
                          "gate": gate_info, "recovery_core": recovery_core, "recovery_outer": recovery_outer, "training": training, "checkpoint_reports": checkpoint_reports,
                          "selected_step": selected_step, "offline_gate": "PASS" if offline_pass else "FAIL", "static_invariance": invariance, "gate_transition_continuity": "PASS_BY_SMOOTHSTEP_CONSTRUCTION",
                          "pilot": final_pilot, "checkpoint": selected_offline["checkpoint"], "checkpoint_sha": selected_offline["checkpoint_sha"]}
                dump(OUT / f"cegis_round{round_index}.json", report); round_reports.append(report)
                if not offline_pass:
                    break
                if final_pilot is not None and final_pilot["passed"]:
                    selected_round = round_index
                    matched_training, matched_bundles = train_local(cumulative, global_def, global_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                        agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                    matched_params, matched_state = matched_bundles[selected_step]
                    matched_fn = make_action_fn(global_def, global_params, matched_params, recovery_support, c1_scale, recovery_core, recovery_outer,
                        agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                    matched_pilot = pilot(matched_fn, c1_support, c1_scale, c1_core, c1_outer, demo_obs, demo_scale, baseline_c1)
                    confirmed = bool(tree_hash(matched_params) == tree_hash(local_params) and matched_pilot == final_pilot)
                    matched = "PASS" if confirmed else "FAIL"
                    dump(OUT / "matched_confirmation.json", {"selected_step": selected_step, "training": matched_training, "params_sha": tree_hash(matched_params), "pilot": matched_pilot, "confirmed": confirmed})
                    if confirmed:
                        frozen = OUT / "checkpoints" / "C2_CONFIRMED_FROZEN_COMPOSITE_V1.msgpack"
                        frozen.write_bytes(flax.serialization.to_bytes({"global_c2_params": global_params, "local_recovery_params": matched_params,
                            "recovery_support": recovery_support, "recovery_scale": c1_scale, "recovery_core": recovery_core, "recovery_outer": recovery_outer}))
                    break
        finally: oracle.close()

        req(geo.thash(agent.state.params["modules_actor"]) == base_sha and geo.thash(c1_params) == c1_sha and tree_hash(global_params) == global_sha, "frozen model mutation")
        if confirmed:
            root, next_task, first_bad = "NONE", "CANONICAL_INTERFACE_REACHABILITY_AUDIT", None
        elif no_new_counterexample:
            root, next_task, first_bad = "C2_RECOVERY_SUPPORT_GENERALIZATION_FAILURE", "HISTORY_AUGMENTED_LOCAL_RECOVERY_POLICY_REVIEW", "NEW_COUNTEREXAMPLE_GATE"
        elif round_reports and round_reports[-1]["offline_gate"] == "FAIL":
            root, next_task, first_bad = "LOCAL_RECOVERY_POLICY_MULTI_STEP_GATE_FAILURE", "HISTORY_AUGMENTED_LOCAL_RECOVERY_POLICY_REVIEW", "RECOVERY_HEAD_OFFLINE_H8_GATE"
        else:
            root, next_task, first_bad = "C2_RECOVERY_BOUNDARY_HAS_HIGH_TRAJECTORY_COMPLEXITY", "HISTORY_AUGMENTED_LOCAL_RECOVERY_POLICY_REVIEW", "C2_PILOT_GATE"
        protected_after = {k: sha(v) for k, v in protected_paths.items()}; req(protected_after == protected_before, "protected artifact mutation"); dump(OUT / "protected_after.json", protected_after)
        c2 = None if final_pilot is None else final_pilot["C2"]
        selected_report = None if not round_reports else round_reports[-1]
        result = {"AUTHORITATIVE": "YES", "C2_TRAJECTORY_RECOVERY_HEAD_V1": "PASS" if confirmed else "FAIL", "CEGIS_ROUNDS_USED": len(round_reports), "CEGIS_MAX_ROUNDS": MAX_ROUNDS,
            "SELECTED_ROUND": selected_round, "C2_SHELL_ENTRY": None if c2 is None else c2["interface"]["C1_SHELL_ENTRY_COUNT"],
            "C2_CORE_ENTRY": None if c2 is None else c2["interface"]["C1_CORE_ENTRY_COUNT"], "C2_SUCCESS": "NOT_RUN" if c2 is None else f"{c2['success_count']}/3",
            "C2_CONTACT": "NOT_RUN" if c2 is None else "PASS" if c2["contact_health"] else "FAIL", "C0_SUCCESS": "NOT_RUN" if final_pilot is None else f"{final_pilot['C0']['success_count']}/3",
            "C1_CONTACT": "NOT_RUN" if final_pilot is None else "PASS" if final_pilot["C1"]["contact_health"] else "FAIL", "C2_MATCHED_CONFIRMATION": matched,
            "C2_SKILL_CONFIRMED": "YES" if confirmed else "NO", "PRIMARY_ROOT_CAUSE": root, "FIRST_BAD_GATE": first_bad, "NEXT": next_task,
            "RECOVERY_CORE_THRESHOLD": recovery_core, "RECOVERY_OUTER_THRESHOLD": recovery_outer,
            "RECOVERY_HEAD_H1_PRESERVATION": None if selected_report is None else selected_report["checkpoint_reports"][selected_report["selected_step"]]["H1_SHELL"],
            "RECOVERY_HEAD_H2_PRESERVATION": None if selected_report is None else selected_report["checkpoint_reports"][selected_report["selected_step"]]["H2_SHELL"],
            "RECOVERY_HEAD_H4_PRESERVATION": None if selected_report is None else selected_report["checkpoint_reports"][selected_report["selected_step"]]["H4_SHELL"],
            "RECOVERY_HEAD_H8_PRESERVATION": None if selected_report is None else selected_report["checkpoint_reports"][selected_report["selected_step"]]["H8_SHELL"],
            "EXPERIMENT_NEXT": "CONTINUE" if confirmed else "STOP_THIS_BRANCH", "PROJECT_STATUS": "ACTIVE", "PROJECT_NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814, "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "C0_C1_GLOBAL_C2_FROZEN": "YES", "PROTECTED_ARTIFACTS_CHANGED": "NO", "PEG_SINGLE_TASK_CONVERGED": "NO",
            "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started}
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
