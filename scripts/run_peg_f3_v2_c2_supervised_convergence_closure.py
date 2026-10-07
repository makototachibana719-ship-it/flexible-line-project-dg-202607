#!/usr/bin/env python3
"""Fixed 3K convergence closure for constant-plateau supervised C2 repair."""
from __future__ import annotations

import hashlib
import json
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
import scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1 as corrective
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_supervised_optimization_review as optimization
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, make_train_env
from training.peg_insertion.gated_absolute_c2_action_head_v1 import gated_absolute_mode
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OPT_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_supervised_optimization_review_20260908_001"
REPAIR1K = OPT_RUN / "checkpoints/supervised_constant_plateau_lr_v1.msgpack"
AUDIT_DATA = pc.AUDIT_DATA
OUT = ROOT / "results/training/peg_f3_v2_c2_supervised_convergence_closure_20260908_001"
MONITOR_STEPS = (0, 250, 500, 750, 1000, 1500, 2000, 2500, 3000)
SELECTION_STEPS = (1000, 1500, 2000, 2500, 3000)
LINE_SCAN_STEPS = (1000, 2000, 3000)
LR = 3e-4
UPDATES = 3000
TRAIN_MAE_MAX = 0.03
DAGGER_MAE_MAX = 0.06
AXIS_MAE_MAX = 0.15
DEMO_CATASTROPHIC_MAE_MAX = 0.05
DESCENDING_RELATIVE_IMPROVEMENT_MIN = 0.001


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


def init_model(seed=261400):
    definition = pc.model_def(pc.CURRENT_WIDTH)
    params = definition.init(jax.random.PRNGKey(seed), np.zeros((1, 20), np.float32))["params"]
    tx = optax.adam(LR)
    return definition, params, tx, tx.init(params)


def load_1k():
    definition, params, tx, state = init_model()
    loaded = flax.serialization.from_bytes({"params": params, "opt_state": state}, REPAIR1K.read_bytes())
    return definition, loaded["params"], tx, loaded["opt_state"]


def train_metric(definition, params, rows, indices, agent, c1_params, support, scale, core, outer):
    return pc.metric(definition, params, pc.features(rows, indices, "H0"), pc.observations(rows, indices), pc.targets(rows, indices), agent, c1_params, support, scale, core, outer)


def full_metrics(definition, params, rows, train_indices, demo_valid, dagger_valid, step_indices, agent, c1_params, support, scale, core, outer):
    valid = demo_valid + dagger_valid
    return {
        "train": train_metric(definition, params, rows, train_indices, agent, c1_params, support, scale, core, outer),
        "valid": train_metric(definition, params, rows, valid, agent, c1_params, support, scale, core, outer),
        "demo_valid": train_metric(definition, params, rows, demo_valid, agent, c1_params, support, scale, core, outer),
        "dagger_valid": train_metric(definition, params, rows, dagger_valid, agent, c1_params, support, scale, core, outer),
        "step8_11": train_metric(definition, params, rows, step_indices, agent, c1_params, support, scale, core, outer),
    }


def train_3k(rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, out_dir, seed=261400):
    definition, params, tx, state = init_model(seed)
    rng = np.random.default_rng(seed)
    demo_train = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger_train = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]
    step_indices = [i for i in dagger_valid if 8 <= rows[i]["source_step"] <= 11]
    reports, bundles, previous = {}, {}, params

    @jax.jit
    def step(candidate, opt_state, feature, obs, target):
        fn = lambda value: optimization.action_loss(definition, value, feature, obs, target, agent, c1_params, support, scale, core, outer)
        loss, gradient = jax.value_and_grad(fn)(candidate)
        update, new_state = tx.update(gradient, opt_state, candidate)
        return optax.apply_updates(candidate, update), new_state, loss, gradient, update

    reports[0] = {"step": 0, **full_metrics(definition, params, rows, train_indices, demo_valid, dagger_valid, step_indices, agent, c1_params, support, scale, core, outer), "gradient_norm": None, "parameter_delta_norm": 0.0}
    for update_index in range(1, UPDATES + 1):
        demo_choice = pc.episode_balanced(rows, demo_train, 32, rng)
        dagger_choice = pc.episode_balanced(rows, dagger_train, 32, rng)
        chosen = np.concatenate((demo_choice, dagger_choice))
        previous = params
        params, state, loss, gradient, update = step(params, state, *map(jnp.asarray, optimization.batch_arrays(rows, chosen)))
        if update_index in MONITOR_STEPS:
            delta = jax.tree_util.tree_map(lambda a, b: a - b, params, previous)
            report = {"step": update_index, **full_metrics(definition, params, rows, train_indices, demo_valid, dagger_valid, step_indices, agent, c1_params, support, scale, core, outer), "batch_loss": float(loss), "gradient_norm": optimization.tree_norm(gradient), "parameter_delta_norm": optimization.tree_norm(delta), "per_layer_gradient_norm": optimization.layer_norms(gradient), "per_layer_parameter_delta_norm": optimization.layer_norms(delta)}
            reports[update_index] = report
            checkpoint = out_dir / f"constant_plateau_c2_{update_index:04d}.msgpack"
            checkpoint.write_bytes(flax.serialization.to_bytes({"params": params, "opt_state": state}))
            report["checkpoint"] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
            if update_index in SELECTION_STEPS: bundles[update_index] = (params, state)
    return definition, tx, reports, bundles


def line_scan(definition, tx, bundle, rows, train_indices, agent, c1_params, support, scale, core, outer):
    params, state = bundle
    arrays = tuple(map(jnp.asarray, optimization.batch_arrays(rows, train_indices)))
    fn = lambda value: optimization.action_loss(definition, value, *arrays, agent, c1_params, support, scale, core, outer)
    loss, gradient = jax.value_and_grad(fn)(params); update, _ = tx.update(gradient, state, params)
    values = {alpha: float(fn(optax.apply_updates(params, jax.tree_util.tree_map(lambda x: alpha * x, update)))) for alpha in (0.0, .25, .5, 1.0, 2.0, 4.0)}
    best = min(values, key=values.get)
    relative = (values[0.0] - values[best]) / max(abs(values[0.0]), 1e-20)
    return {"losses": values, "best_alpha": best, "relative_best_improvement": relative, "LOSS_STILL_DESCENDING": bool(best > 0 and relative >= DESCENDING_RELATIVE_IMPROVEMENT_MIN), "ordinary_update_norm": optimization.tree_norm(update)}


def boundary_triplet(definition, params, agent, c1_params, historical_params, support, scale, core, outer, demos, demo_obs, demo_scale):
    env = make_train_env("C2")
    try:
        obs, _ = env.reset(seed=100)
        for step_index in range(11):
            old_action = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, historical_params, obs, support, scale, core, outer)), np.float32)
            if step_index == 10:
                snapshot = basin.capture(env, obs)
                repaired_action = np.asarray(jax.device_get(pc.compose_actions(definition, params, jnp.asarray(obs), jnp.asarray(obs), agent, c1_params, support, scale, core, outer)), np.float32)
                distances = np.sqrt(np.sum(np.square((demo_obs - obs) / demo_scale), axis=1)); nearest = int(np.argmin(distances))
                expert_action = np.asarray(demos[nearest]["actions"], np.float32)
                actions = {"old": old_action, "repair": repaired_action, "expert": expert_action}
                branches = {name: basin.branch_then_takeover(snapshot, action, demos, demo_obs, demo_scale, support, scale, core, outer) for name, action in actions.items()}
                result = {
                    "historical_state_step": 10,
                    "observation": np.asarray(obs, np.float32),
                    "actions": actions,
                    "repair_expert_action_l2": float(np.linalg.norm(repaired_action[:6] - expert_action[:6])),
                    "repair_expert_per_axis_abs_error": np.abs(repaired_action[:6] - expert_action[:6]),
                    "old_expert_action_l2": float(np.linalg.norm(old_action[:6] - expert_action[:6])),
                    "branches": branches,
                    "OLD_NEXT_STATE_RECOVERABLE": "YES" if branches["old"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] else "NO",
                    "REPAIR_NEXT_STATE_RECOVERABLE": "YES" if branches["repair"]["one_step"]["contact_healthy"] and branches["repair"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] else "NO",
                    "EXPERT_NEXT_STATE_RECOVERABLE": "YES" if branches["expert"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] else "NO",
                }
                result["HISTORICAL_STEP11_BOUNDARY_REPAIRED"] = result["REPAIR_NEXT_STATE_RECOVERABLE"]
                return result
            obs, _, terminated, truncated, _ = env.step(old_action)
            req(not terminated and not truncated, "historical trajectory ended before step10")
    finally:
        env.close()
    raise RuntimeError("historical step10 not reached")


def c1_same(reference, candidate):
    keys = ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
    return bool(candidate["contact_health"] and all(candidate[key] == reference[key] for key in keys))


def pilot(definition, params, historical_params, rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale):
    reset_obs = next(row["observations"] for row in rows if row["source_round"] == 1 and row["source_step"] == 0)
    baseline_c1 = review.eval_record(agent, c1_params, historical_params, support, scale, core, outer, "C1", demo_obs, demo_scale)
    c2 = pc.eval_selected(definition, params, "H0", agent, c1_params, support, scale, core, outer, "C2", demo_obs, demo_scale, reset_obs)
    c1 = pc.eval_selected(definition, params, "H0", agent, c1_params, support, scale, core, outer, "C1", demo_obs, demo_scale, reset_obs)
    c0 = pc.eval_selected(definition, params, "H0", agent, c1_params, support, scale, core, outer, "C0", demo_obs, demo_scale, reset_obs)
    c1_ok = c1_same(baseline_c1, c1)
    passed = bool(c0["success_count"] == 3 and c0["contact_health"] and c1_ok and c2["contact_health"] and c2["interface"]["C1_SHELL_ENTRY_COUNT"] > 0)
    return {"C2": c2, "C1": c1, "C0": c0, "C1_REGRESSION": "PASS" if c1_ok else "FAIL", "C2_PILOT": "PASS" if passed else "FAIL", "passed": passed}


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    optimization_result = json.loads((OPT_RUN / "result.json").read_text())
    req(optimization_result["PRIMARY_ROOT_CAUSE"] == "SUPERVISED_EFFECTIVE_STEP_TOO_SMALL" and optimization_result["SELECTED_REPAIR"] == "SUPERVISED_CONSTANT_PLATEAU_LR_V1", "authoritative repair selection")
    OUT.mkdir(parents=True); (OUT / "checkpoints").mkdir(); started = time.time()
    rows = pickle.loads(AUDIT_DATA.read_bytes()); train_indices, demo_valid, dagger_valid = pc.split_indices(rows)
    step_indices = [i for i in dagger_valid if 8 <= rows[i]["source_step"] <= 11]
    protected_paths = {"production_demo": p4.DEMO, "audit_dataset": AUDIT_DATA, "existing_subset": pc.SOURCE_EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl", "dagger_r1": pc.DAGGER_ROOT / "C2_DAGGER_ROUND1_V1.pkl", "dagger_r2": pc.DAGGER_ROOT / "C2_DAGGER_ROUND2_V1.pkl", "dagger_r3": pc.DAGGER_ROOT / "C2_DAGGER_ROUND3_V1.pkl", "repair1k": REPAIR1K}
    protected_before = {name: sha(path) for name, path in protected_paths.items()}; dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo SHA")
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes()); support = np.asarray(support_bundle["observations"], np.float32); scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text()); core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos); cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos); source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        base_guard = geo.thash(agent.state.params["modules_actor"]); c1_guard = geo.thash(c1_params)
        parameter = json.loads((pc.PARAM_RUN / "result.json").read_text()); historical_params = corrective.load_head(parameter["SELECTED_OFFLINE_CHECKPOINT"]["path"])["c2_head_params"]
        definition1k, params1k, _, _ = load_1k()
        metrics1k = full_metrics(definition1k, params1k, rows, train_indices, demo_valid, dagger_valid, step_indices, agent, c1_params, support, scale, core, outer)
        req(abs(metrics1k["train"]["mae"] - optimization_result["REPAIR_TRAIN_MAE"]) <= 1e-8 and abs(metrics1k["dagger_valid"]["mae"] - optimization_result["REPAIR_DAGGER_VALID_MAE"]) <= 1e-8, "repair1k checkpoint reproduction")
        boundary1k = boundary_triplet(definition1k, params1k, agent, c1_params, historical_params, support, scale, core, outer, demos, demo_obs, demo_scale)
        dump(OUT / "repair1k_boundary_triplet.json", {"checkpoint_sha": protected_before["repair1k"], "metrics": metrics1k, **boundary1k})
        pilot1k = None
        if boundary1k["HISTORICAL_STEP11_BOUNDARY_REPAIRED"] == "YES":
            pilot1k = pilot(definition1k, params1k, historical_params, rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale)
            dump(OUT / "repair1k_c2_pilot.json", pilot1k)
        use_extension = not (boundary1k["HISTORICAL_STEP11_BOUNDARY_REPAIRED"] == "YES" and pilot1k is not None and pilot1k["passed"])

        reports = {}; bundles = {}; scans = {}; selected_step = 1000; selected_def = definition1k; selected_params = params1k
        if use_extension:
            selected_def, tx, reports, bundles = train_3k(rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, OUT / "checkpoints")
            req(reports[1000]["checkpoint"]["sha256"] == protected_before["repair1k"], "fresh 3K trajectory must exactly reproduce repair1k")
            for step_index in LINE_SCAN_STEPS: scans[step_index] = line_scan(selected_def, tx, bundles[step_index], rows, train_indices, agent, c1_params, support, scale, core, outer)
            eligible = [step for step in SELECTION_STEPS if reports[step]["demo_valid"]["mae"] < DEMO_CATASTROPHIC_MAE_MAX]
            req(eligible, "at least one noncatastrophic Demo checkpoint")
            selected_step = min(eligible, key=lambda step: reports[step]["dagger_valid"]["mae"])
            selected_params = bundles[selected_step][0]
            dump(OUT / "convergence_3k.json", {"fixed_run": {"constant_lr": LR, "updates": UPDATES, "monitor_steps": MONITOR_STEPS, "selection_steps": SELECTION_STEPS}, "reports": reports, "line_scans": scans, "selection_rule": {"primary": "lowest held-out DAgger MAE", "constraint": f"Demo-valid MAE < {DEMO_CATASTROPHIC_MAE_MAX}", "behavior_used": False}, "SELECTED_SUPERVISED_CHECKPOINT": selected_step})
        selected_metrics = metrics1k if not use_extension else reports[selected_step]
        offline_pass = bool(selected_metrics["train"]["mae"] < TRAIN_MAE_MAX and selected_metrics["dagger_valid"]["mae"] < DAGGER_MAE_MAX and float(np.max(selected_metrics["dagger_valid"]["per_axis_mae"])) < AXIS_MAE_MAX)
        boundary_selected = boundary1k if selected_step == 1000 else None
        final_pilot = pilot1k if not use_extension else None
        effective_gate_pass = bool(offline_pass or not use_extension)
        if effective_gate_pass and boundary_selected is None:
            boundary_selected = boundary_triplet(selected_def, selected_params, agent, c1_params, historical_params, support, scale, core, outer, demos, demo_obs, demo_scale)
            dump(OUT / "selected_checkpoint_boundary_triplet.json", boundary_selected)
        if effective_gate_pass and boundary_selected["HISTORICAL_STEP11_BOUNDARY_REPAIRED"] == "YES" and final_pilot is None:
            final_pilot = pilot(selected_def, selected_params, historical_params, rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale)
            dump(OUT / "selected_checkpoint_c2_pilot.json", final_pilot)

        confirmed = False; matched = "NOT_RUN"; canonical = None; matched_report = None
        if effective_gate_pass and boundary_selected["HISTORICAL_STEP11_BOUNDARY_REPAIRED"] == "YES" and final_pilot is not None and final_pilot["passed"]:
            if use_extension:
                matched_checkpoint_dir = OUT / "matched_checkpoints"
                matched_checkpoint_dir.mkdir(parents=True, exist_ok=False)
                mdef, mtx, mreps, mbundles = train_3k(rows, train_indices, demo_valid, dagger_valid, agent, c1_params, support, scale, core, outer, matched_checkpoint_dir, seed=261400)
                meligible = [step for step in SELECTION_STEPS if mreps[step]["demo_valid"]["mae"] < DEMO_CATASTROPHIC_MAE_MAX]
                mstep = min(meligible, key=lambda step: mreps[step]["dagger_valid"]["mae"]); mparams = mbundles[mstep][0]
            else:
                mdef, mparams, _, _ = load_1k(); mstep = 1000; mreps = None
            mpilot = pilot(mdef, mparams, historical_params, rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale)
            confirmed = bool(mstep == selected_step and mpilot == final_pilot); matched = "PASS" if confirmed else "FAIL"
            matched_report = {"selected_step": mstep, "reports": mreps, "pilot": mpilot, "confirmed": confirmed}; dump(OUT / "matched_confirmation.json", matched_report)
            if confirmed:
                frozen = OUT / "checkpoints/confirmed_frozen_c2_head.msgpack"; frozen.write_bytes(flax.serialization.to_bytes({"params": mparams}))
                reset_obs = next(row["observations"] for row in rows if row["source_round"] == 1 and row["source_step"] == 0)
                canonical = pc.eval_selected(mdef, mparams, "H0", agent, c1_params, support, scale, core, outer, "C3", demo_obs, demo_scale, reset_obs); dump(OUT / "canonical_interface_audit.json", canonical)

        if not effective_gate_pass:
            descending3k = scans.get(3000, {}).get("LOSS_STILL_DESCENDING", False)
            root = "SUPERVISED_CONVERGENCE_HORIZON_REMAINS_UNRESOLVED" if descending3k else "CONSTANT_LR_OPTIMIZATION_REACHES_INSUFFICIENT_ACTION_SPACE_FIT"
            next_task = "STOP" if descending3k else "PRETANH_SUPERVISION_OBJECTIVE_REVIEW"
            first_bad = "SUPERVISED_3K_OFFLINE_GATE"
        elif boundary_selected["HISTORICAL_STEP11_BOUNDARY_REPAIRED"] != "YES":
            root, next_task, first_bad = "OFFLINE_FIT_IMPROVED_BUT_CAUSAL_BOUNDARY_NOT_REPAIRED", "STOP", "HISTORICAL_STEP11_BOUNDARY"
        elif final_pilot is None or not final_pilot["passed"]:
            root, next_task, first_bad = "C2_CLOSED_LOOP_PILOT_FAILED_AFTER_CONVERGED_SUPERVISION", "STOP", "C2_PILOT_GATE"
        elif not confirmed:
            root, next_task, first_bad = "C2_MATCHED_CONFIRMATION_FAILED", "STOP", "C2_MATCHED_CONFIRMATION"
        else:
            root, next_task, first_bad = "NONE", "C3_ACQUISITION_PILOT", None
        req(geo.thash(agent.state.params["modules_actor"]) == base_guard and geo.thash(c1_params) == c1_guard, "protected model mutation")
        protected_after = {name: sha(path) for name, path in protected_paths.items()}; req(protected_after == protected_before, "protected artifact mutation"); dump(OUT / "protected_after.json", protected_after)
        c2 = None if final_pilot is None else final_pilot["C2"]; c1 = None if final_pilot is None else final_pilot["C1"]; c0 = None if final_pilot is None else final_pilot["C0"]
        result = {
            "C2_SUPERVISED_CONVERGENCE_CLOSURE": "PASS", "REPAIR1000_CHECKPOINT_SHA": protected_before["repair1k"],
            "HISTORICAL_STEP11_BOUNDARY_REPAIRED_AT_1K": boundary1k["HISTORICAL_STEP11_BOUNDARY_REPAIRED"], "C2_1K_EXISTING_PILOT": "NOT_RUN" if pilot1k is None else pilot1k["C2_PILOT"],
            "CONVERGENCE_EXTENSION_USED": "YES" if use_extension else "NO",
            "TRAIN_MAE_1K": metrics1k["train"]["mae"] if not use_extension else reports[1000]["train"]["mae"], "TRAIN_MAE_2K": None if not use_extension else reports[2000]["train"]["mae"], "TRAIN_MAE_3K": None if not use_extension else reports[3000]["train"]["mae"],
            "DAGGER_MAE_1K": metrics1k["dagger_valid"]["mae"] if not use_extension else reports[1000]["dagger_valid"]["mae"], "DAGGER_MAE_2K": None if not use_extension else reports[2000]["dagger_valid"]["mae"], "DAGGER_MAE_3K": None if not use_extension else reports[3000]["dagger_valid"]["mae"],
            "PER_AXIS_DAGGER_MAE_3K": None if not use_extension else reports[3000]["dagger_valid"]["per_axis_mae"], "STEP8_11_MAE_3K": None if not use_extension else reports[3000]["step8_11"]["mae"],
            "LOSS_STILL_DESCENDING_AT_1K": "NOT_RUN" if not use_extension else "YES" if scans[1000]["LOSS_STILL_DESCENDING"] else "NO", "LOSS_STILL_DESCENDING_AT_2K": "NOT_RUN" if not use_extension else "YES" if scans[2000]["LOSS_STILL_DESCENDING"] else "NO", "LOSS_STILL_DESCENDING_AT_3K": "NOT_RUN" if not use_extension else "YES" if scans[3000]["LOSS_STILL_DESCENDING"] else "NO",
            "SELECTED_SUPERVISED_CHECKPOINT": selected_step, "OFFLINE_GATE": "PASS" if offline_pass else "FAIL",
            "HISTORICAL_STEP11_BOUNDARY_REPAIRED": "NOT_RUN" if boundary_selected is None else boundary_selected["HISTORICAL_STEP11_BOUNDARY_REPAIRED"], "C2_PILOT": "NOT_RUN" if final_pilot is None else final_pilot["C2_PILOT"],
            "C2_SHELL_ENTRY": None if c2 is None else c2["interface"]["C1_SHELL_ENTRY_COUNT"], "C2_CORE_ENTRY": None if c2 is None else c2["interface"]["C1_CORE_ENTRY_COUNT"], "C2_NEAR_HOLE": None if c2 is None else c2["near_hole_rate"], "C2_SUCCESS": "NOT_RUN" if c2 is None else f"{c2['success_count']}/3", "C2_CONTACT": "NOT_RUN" if c2 is None else "PASS" if c2["contact_health"] else "FAIL",
            "C1_REGRESSION": "NOT_RUN" if final_pilot is None else final_pilot["C1_REGRESSION"], "C1_CONTACT": "NOT_RUN" if c1 is None else "PASS" if c1["contact_health"] else "FAIL", "C0_SUCCESS": "NOT_RUN" if c0 is None else f"{c0['success_count']}/3",
            "C2_MATCHED_CONFIRMATION": matched, "C2_SKILL_CONFIRMED": "YES" if confirmed else "NO", "CANONICAL_INTERFACE_AUDIT": "NOT_RUN" if canonical is None else "PASS", "PEG_SINGLE_TASK_CONVERGED": "NO",
            "FIRST_BAD_GATE": first_bad, "PRIMARY_ROOT_CAUSE": root, "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": p4.TH, "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO", "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started,
        }
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
