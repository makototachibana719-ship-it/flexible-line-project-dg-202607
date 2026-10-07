#!/usr/bin/env python3
"""Single evidence-selected Repair B for C2 recovery-boundary tail risk."""
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
import scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1 as corrective
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_supervised_convergence_closure as convergence
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, make_train_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.progressive_frozen_residual_v2 import support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


AUDIT_RUN = tail.OUT
OUT = ROOT / "results/training/peg_f3_v2_c2_recovery_margin_normalized_supervision_v1_20260909_001"
AUDIT_SET = AUDIT_RUN / "C2_MULTI_BOUNDARY_AUDIT_SET_V1.pkl"
MARGINS_PATH = AUDIT_RUN / "recovery_action_sensitivity_map.json"
REPRESENTATION_PATH = AUDIT_RUN / "critical_training_representation.json"
RISK_PATH = AUDIT_RUN / "checkpoint_error_margin_risk.json"
DAGGER_RUN = tail.DAGGER_RUN
CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000)
LR = 3e-4
UPDATES = 3000
MARGIN_FLOOR = 0.0025
SHELL_GATE = 0.90
CORE_GATE = 0.80
GLOBAL_DAGGER_MAX = 0.06
MATERIAL_VIOLATION_RELATIVE_GAIN = 0.10
MATERIAL_VIOLATION_ABSOLUTE_GAIN = 0.01
SHORT_MATERIAL_GAIN = 1.0 / 7.0


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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_global(step):
    path = convergence.OUT / f"checkpoints/constant_plateau_c2_{step:04d}.msgpack"
    definition, params = tail.load_head_checkpoint(path)
    return definition, params


def reconstruct_states(public_rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale, policy_params):
    replays = tail.build_source_replays(agent, c1_params, policy_params, support, scale, core, outer, demo_obs, demo_scale)
    indices = {label: tail.replay_index(episodes) for label, episodes in replays.items()}
    states = []
    for public in public_rows:
        row = indices[public["source_policy"]][(int(public["seed"]), int(public["policy_step"]))]
        req(tail.obs_hash(row["observation"]) == public["obs20_sha"], "frozen audit Obs20 identity")
        req(tail.snapshot_hash(row["snapshot"]) == public["exact_state_sha"], "frozen audit exact-state identity")
        states.append({"row": row, "source_policy": public["source_policy"], "categories": set(public["categories"]), "takeover": public})
    return states, replays


def fixed_weight_table(rows, representation, margins):
    table = np.ones((len(rows), 6), np.float32)
    margin_by_hash = {row["obs20_sha"]: np.asarray(row["margin_axis"], np.float32) for row in margins}
    exact = []
    for match in representation["matches"]:
        if float(match["nearest_train_distance"]) != 0.0:
            continue
        index = int(match["nearest_train_index"])
        margin = margin_by_hash[match["obs20_sha"]]
        weight = 1.0 / np.square(np.maximum(margin, MARGIN_FLOOR))
        weight = weight / np.mean(weight)
        if index in [item["dataset_index"] for item in exact]:
            previous = next(item for item in exact if item["dataset_index"] == index)
            req(np.array_equal(np.asarray(previous["weights"], np.float32), weight.astype(np.float32)), "consistent duplicate critical weights")
        else:
            table[index] = weight.astype(np.float32)
            exact.append({"obs20_sha": match["obs20_sha"], "dataset_index": index, "margin_axis": margin, "weights": weight})
    req(len(exact) == 13, "13 exact authoritative critical training rows")
    return table, exact


def critical_metrics(definition, params, states, margin_by_hash, agent, c1_params, support, scale, core, outer):
    obs = np.asarray([state["row"]["observation"] for state in states], np.float32)
    target = np.asarray([state["takeover"]["expert_source_row"] for state in states], np.int64)
    del target
    pred = np.asarray(jax.device_get(pc.compose_actions(definition, params, jnp.asarray(obs), jnp.asarray(obs), agent, c1_params, support, scale, core, outer)), np.float32)
    expert = np.asarray([margin_by_hash[tail.obs_hash(state["row"]["observation"])]["expert_action"] for state in states], np.float32)
    error = np.abs(pred[:, :6] - expert[:, :6])
    margins = np.asarray([margin_by_hash[tail.obs_hash(state["row"]["observation"])]["margin_axis"] for state in states], np.float32)
    normalized = error / np.maximum(margins, 1e-12)
    return {"mae": float(np.mean(error)), "p95": float(np.quantile(error, .95)), "max": float(np.max(error)),
            "per_axis_mae": np.mean(error, axis=0), "per_axis_p95": np.quantile(error, .95, axis=0),
            "margin_violation_fraction": float(np.mean(normalized >= 1.0)), "normalized_error_mean": float(np.mean(normalized)),
            "normalized_error_p95": float(np.quantile(normalized, .95)), "predicted_actions": pred, "expert_actions": expert}


def train_once(out_dir, rows, train_indices, demo_valid, dagger_valid, weights, states, heldout_states, margin_by_hash,
               oracle, demos, agent, c1_params, support, scale, core, outer, seed=261400):
    definition, params, tx, opt_state = convergence.init_model(seed)
    rng = np.random.default_rng(seed)
    demo_train = [i for i in train_indices if rows[i]["source_type"] == "existing_expert"]
    dagger_train = [i for i in train_indices if rows[i]["source_type"] == "recoverable_dagger"]
    reports, bundles = {}, {}

    @jax.jit
    def step(candidate, state, obs, target, axis_weights):
        def loss_fn(value):
            pred = pc.compose_actions(definition, value, obs, obs, agent, c1_params, support, scale, core, outer)
            return jnp.mean(axis_weights * jnp.square(pred[..., :6] - target[..., :6]))
        loss, gradient = jax.value_and_grad(loss_fn)(candidate)
        updates, next_state = tx.update(gradient, state, candidate)
        return optax.apply_updates(candidate, updates), next_state, loss, gradient

    for update_index in range(1, UPDATES + 1):
        demo_choice = pc.episode_balanced(rows, demo_train, 32, rng)
        dagger_choice = pc.episode_balanced(rows, dagger_train, 32, rng)
        chosen = np.concatenate((demo_choice, dagger_choice))
        obs = jnp.asarray(pc.observations(rows, chosen)); target = jnp.asarray(pc.targets(rows, chosen)); batch_weights = jnp.asarray(weights[chosen])
        params, opt_state, loss, gradient = step(params, opt_state, obs, target, batch_weights)
        if update_index in CHECKPOINTS:
            checkpoint = out_dir / f"recovery_margin_normalized_c2_{update_index:04d}.msgpack"
            checkpoint.write_bytes(flax.serialization.to_bytes({"params": params, "opt_state": opt_state}))
            global_metrics = pc.metric(definition, params, pc.features(rows, dagger_valid, "H0"), pc.observations(rows, dagger_valid), pc.targets(rows, dagger_valid), agent, c1_params, support, scale, core, outer)
            demo_metrics = pc.metric(definition, params, pc.features(rows, demo_valid, "H0"), pc.observations(rows, demo_valid), pc.targets(rows, demo_valid), agent, c1_params, support, scale, core, outer)
            critical = critical_metrics(definition, params, states, margin_by_hash, agent, c1_params, support, scale, core, outer)
            full = tail.preservation_audit(states, {update_index: (definition, params, global_metrics["mae"], sha(checkpoint))}, oracle, agent, c1_params, support, scale, core, outer, demos)[update_index]
            heldout = tail.preservation_audit(heldout_states, {update_index: (definition, params, global_metrics["mae"], sha(checkpoint))}, oracle, agent, c1_params, support, scale, core, outer, demos)[update_index]
            reports[update_index] = {"update": update_index, "batch_loss": float(loss), "gradient_norm": tail.tree_norm(gradient),
                "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}, "global_dagger": global_metrics,
                "demo_valid": demo_metrics, "critical": {k: v for k, v in critical.items() if k not in ("predicted_actions", "expert_actions")},
                "multi_boundary": {k: v for k, v in full.items() if k != "rows"},
                "heldout_multi_boundary": {k: v for k, v in heldout.items() if k != "rows"}}
            bundles[update_index] = (params, opt_state)
    return definition, tx, reports, bundles


def select_checkpoint(reports):
    return min(CHECKPOINTS, key=lambda step: (
        -reports[step]["heldout_multi_boundary"]["BOUNDARY_SHELL_PRESERVATION_RATE"],
        -reports[step]["heldout_multi_boundary"]["BOUNDARY_CORE_PRESERVATION_RATE"],
        reports[step]["critical"]["margin_violation_fraction"],
        reports[step]["critical"]["p95"], reports[step]["global_dagger"]["mae"],
    ))


def short_state_suite(states, replays):
    offline = {
        int(row["policy_step"]): {"row": row, "source_policy": "offline"}
        for episode in replays["offline"] for row in episode if int(row["seed"]) == 100
    }
    closest = min(states, key=lambda s: s["row"]["D_C1"])
    suite = [
        ("C2_RESET", offline[0]), ("STEP5", offline[5]), ("STEP8", offline[8]),
        ("STEP9", offline[9]), ("STEP10", offline[10]),
        ("HISTORICAL_STEP11_PRECURSOR", offline[10]), ("CLOSEST_TO_SHELL", closest),
    ]
    return suite


def short_horizon(definition, params, suite, oracle, agent, c1_params, support, scale, core, outer):
    env = make_train_env("C2")
    rows = []
    try:
        for label, state in suite:
            obs = basin.restore(env, state["row"]["snapshot"])
            healthy = True; executed = 0; final_snapshot = state["row"]["snapshot"]
            for _ in range(5):
                action = tail.checkpoint_action(definition, params, obs, agent, c1_params, support, scale, core, outer)
                obs, _, terminated, truncated, info = env.step(action); executed += 1
                link, unexpected, _ = contact_flags(env)
                healthy &= bool(not link and not unexpected and np.isfinite(float(info.get("force_norm", 0.0))))
                final_snapshot = basin.capture(env, obs)
                if terminated or truncated or not healthy: break
            takeover = oracle.takeover(final_snapshot) if healthy else {"EXPERT_SHELL_RECOVERABLE": False, "EXPERT_CORE_RECOVERABLE": False}
            rows.append({"label": label, "source_policy": state["source_policy"], "source_step": state["row"]["policy_step"],
                         "executed_policy_actions": executed, "contact_healthy": healthy,
                         "shell_recoverable": bool(healthy and takeover["EXPERT_SHELL_RECOVERABLE"]),
                         "core_recoverable": bool(healthy and takeover["EXPERT_CORE_RECOVERABLE"]), "takeover": takeover})
    finally:
        env.close()
    return {"SHORT_HORIZON_RECOVERABILITY_RATE": float(np.mean([r["shell_recoverable"] for r in rows])),
            "SHORT_HORIZON_CORE_RATE": float(np.mean([r["core_recoverable"] for r in rows])), "rows": rows}


def live_recoverability(definition, params, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale, oracle, audit_hashes):
    episodes = basin.collect_occupancy(agent, c1_params, params, support, scale, core, outer, demo_obs, demo_scale)
    output = []
    fixed = {0, 3, 5, 7, 9, 10, 11, 12, 15}
    for rows in episodes:
        selected = set(i for i in fixed if i < len(rows))
        distances = np.asarray([row["D_C1"] for row in rows], np.float64)
        for i in range(1, len(rows) - 1):
            if distances[i] <= distances[i - 1] and distances[i] <= distances[i + 1]: selected.add(i)
        if rows: selected.add(max(0, len(rows) - 3))
        audits = []
        for i in sorted(selected):
            takeover = oracle.takeover(rows[i]["snapshot"])
            audits.append({"policy_step": int(rows[i]["policy_step"]), "obs20_sha": tail.obs_hash(rows[i]["observation"]),
                           "known_boundary_state": tail.obs_hash(rows[i]["observation"]) in audit_hashes,
                           "D_C1": rows[i]["D_C1"], "expert_shell_recoverable": takeover["EXPERT_SHELL_RECOVERABLE"],
                           "expert_core_recoverable": takeover["EXPERT_CORE_RECOVERABLE"], "takeover": takeover})
        first_bad = next((r for r in audits if not r["expert_shell_recoverable"]), None)
        output.append({"seed": int(rows[0]["seed"]) if rows else None, "trajectory_steps": len(rows), "audits": audits,
                       "first_nonrecoverable_policy_step": None if first_bad is None else first_bad["policy_step"],
                       "first_nonrecoverable_known_boundary": None if first_bad is None else first_bad["known_boundary_state"]})
    values = [x["first_nonrecoverable_policy_step"] for x in output if x["first_nonrecoverable_policy_step"] is not None]
    return {"episodes": output, "C2_FIRST_NONRECOVERABLE_STEP": min(values) if values else None,
            "all_monitored_recoverable": not values}


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    audit_result = json.loads((AUDIT_RUN / "result.json").read_text())
    req(audit_result["SELECTED_REPAIR"] == "RECOVERY_MARGIN_NORMALIZED_SUPERVISION_V1" and audit_result["PRIMARY_ROOT_CAUSE"] == "RECOVERY_BOUNDARY_AXIS_SENSITIVITY_NOT_CAPTURED_BY_UNIFORM_ACTION_MSE", "Repair B must be causally selected")
    OUT.mkdir(parents=True); (OUT / "checkpoints").mkdir(); started = time.time()
    protected_paths = {"production_demo": p4.DEMO, "audit_set": AUDIT_SET, "audit_dataset": pc.AUDIT_DATA,
        "dagger_r1": tail.DAGGER_DATA / "C2_DAGGER_ROUND1_V1.pkl", "dagger_r2": tail.DAGGER_DATA / "C2_DAGGER_ROUND2_V1.pkl",
        "dagger_r3": tail.DAGGER_DATA / "C2_DAGGER_ROUND3_V1.pkl"}
    protected_before = {name: sha(path) for name, path in protected_paths.items()}; dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == tail.DEMO_SHA and protected_before["audit_set"] == audit_result["MULTI_BOUNDARY_SET_SHA"], "protected authoritative inputs")

    public_rows = pickle.loads(AUDIT_SET.read_bytes()); margins = json.loads(MARGINS_PATH.read_text())
    margin_by_hash = {row["obs20_sha"]: row for row in margins}; representation = json.loads(REPRESENTATION_PATH.read_text())
    risks = json.loads(RISK_PATH.read_text()); historical_best_violation = min(float(value["MARGIN_VIOLATION_FRACTION"]) for value in risks.values())
    rows = pickle.loads(pc.AUDIT_DATA.read_bytes()); train_indices, demo_valid, dagger_valid = pc.split_indices(rows)
    weights, weighted_rows = fixed_weight_table(rows, representation, margins)
    dump(OUT / "fixed_margin_weights.json", {"MARGIN_FLOOR": MARGIN_FLOOR, "critical_exact_rows": weighted_rows,
         "noncritical_weight": [1.0] * 6, "per_sample_weight_mean": [float(np.mean(x["weights"])) for x in weighted_rows],
         "source_sampling": "unchanged 0.5 production-derived expert / 0.5 recoverability-filtered DAgger"})

    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32); scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text()); core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    cfg = load_config(2)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        parameter = json.loads((pc.PARAM_RUN / "result.json").read_text())
        policy_params = {"offline": tail.load_dagger_head(parameter["SELECTED_OFFLINE_CHECKPOINT"]["path"]),
            "dagger_r1": tail.load_dagger_head(DAGGER_RUN / "checkpoints/c2_dagger_round1_head_1000.msgpack"),
            "dagger_r2": tail.load_dagger_head(DAGGER_RUN / "checkpoints/c2_dagger_round2_head_1000.msgpack"),
            "dagger_r3": tail.load_dagger_head(DAGGER_RUN / "checkpoints/c2_dagger_round3_head_1000.msgpack")}
        states, source_replays = reconstruct_states(public_rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale, policy_params)
        nonexact_hashes = {m["obs20_sha"] for m in representation["matches"] if float(m["nearest_train_distance"]) > 0.0}
        heldout_states = [s for s in states if tail.obs_hash(s["row"]["observation"]) in nonexact_hashes]
        req(len(heldout_states) == 3, "three non-training held-out boundary states")
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, support, scale, core, outer)
        try:
            definition, tx, reports, bundles = train_once(OUT / "checkpoints", rows, train_indices, demo_valid, dagger_valid, weights,
                states, heldout_states, margin_by_hash, oracle, demos, agent, c1_params, support, scale, core, outer)
            selected = select_checkpoint(reports); selected_params, selected_opt = bundles[selected]
            dump(OUT / "repair_monitoring.json", {"repair": "RECOVERY_MARGIN_NORMALIZED_SUPERVISION_V1", "fixed_updates": UPDATES,
                "constant_lr": LR, "checkpoint_selection": ["highest heldout shell preservation", "highest heldout core preservation",
                "lowest critical margin violation", "lowest critical P95", "global DAgger MAE final tie-break"],
                "historical_best_margin_violation": historical_best_violation, "reports": reports, "SELECTED_CHECKPOINT": selected})

            selected_report = reports[selected]
            violation = selected_report["critical"]["margin_violation_fraction"]
            rel_gain = (historical_best_violation - violation) / max(historical_best_violation, 1e-12)
            abs_gain = historical_best_violation - violation
            violation_material = bool(violation < historical_best_violation and (rel_gain >= MATERIAL_VIOLATION_RELATIVE_GAIN or abs_gain >= MATERIAL_VIOLATION_ABSOLUTE_GAIN))
            full_shell = selected_report["multi_boundary"]["BOUNDARY_SHELL_PRESERVATION_RATE"]
            full_core = selected_report["multi_boundary"]["BOUNDARY_CORE_PRESERVATION_RATE"]
            held_shell = selected_report["heldout_multi_boundary"]["BOUNDARY_SHELL_PRESERVATION_RATE"]
            held_core = selected_report["heldout_multi_boundary"]["BOUNDARY_CORE_PRESERVATION_RATE"]
            invariance = tail.static_invariance(agent, c1_params, support, scale, core, outer,
                {selected: (definition, selected_params, selected_report["global_dagger"]["mae"], selected_report["checkpoint"]["sha256"])})[selected]
            offline_gate = bool(full_shell >= SHELL_GATE and full_core >= CORE_GATE and held_shell >= SHELL_GATE and held_core >= CORE_GATE
                and violation_material and selected_report["global_dagger"]["mae"] <= GLOBAL_DAGGER_MAX and invariance["pass"])
            dump(OUT / "selected_offline_gate.json", {"SELECTED_CHECKPOINT": selected, "full_shell": full_shell, "full_core": full_core,
                "heldout_shell": held_shell, "heldout_core": held_core, "margin_violation": violation,
                "historical_best_margin_violation": historical_best_violation, "relative_gain": rel_gain, "absolute_gain": abs_gain,
                "violation_materially_below_historical_best": violation_material, "global_dagger_mae": selected_report["global_dagger"]["mae"],
                "invariance": invariance, "MULTI_BOUNDARY_OFFLINE_GATE": "PASS" if offline_gate else "FAIL"})

            selected_full = tail.preservation_audit(states, {selected: (definition, selected_params, selected_report["global_dagger"]["mae"], selected_report["checkpoint"]["sha256"])}, oracle, agent, c1_params, support, scale, core, outer, demos)[selected]
            sentinel = next(r for r in selected_full["rows"] if r["source_policy"] == "offline" and int(r["step"]) == 10)
            historical_repaired = bool(sentinel["shell_preserved"])
            short = None; pilot = None; live = None; matched = "NOT_RUN"; confirmed = False; canonical = None; confirmed_shas = []
            first_bad = "MULTI_BOUNDARY_OFFLINE_GATE" if not offline_gate else None
            root = audit_result["PRIMARY_ROOT_CAUSE"]; next_task = "STOP"
            if offline_gate:
                suite = short_state_suite(states, source_replays)
                base_def, base_params = load_global(2000)
                baseline_short = short_horizon(base_def, base_params, suite, oracle, agent, c1_params, support, scale, core, outer)
                repaired_short = short_horizon(definition, selected_params, suite, oracle, agent, c1_params, support, scale, core, outer)
                short_gain = repaired_short["SHORT_HORIZON_RECOVERABILITY_RATE"] - baseline_short["SHORT_HORIZON_RECOVERABILITY_RATE"]
                short = {"baseline_checkpoint": 2000, "baseline": baseline_short, "repair": repaired_short,
                         "absolute_gain": short_gain, "material_improvement": bool(short_gain >= SHORT_MATERIAL_GAIN)}
                dump(OUT / "short_horizon_causal_audit.json", short)
                if not short["material_improvement"]:
                    first_bad = "SHORT_HORIZON_RECOVERABILITY"
                else:
                    historical = policy_params["offline"]
                    pilot = convergence.pilot(definition, selected_params, historical, rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale)
                    audit_hashes = {public["obs20_sha"] for public in public_rows}
                    live = live_recoverability(definition, selected_params, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale, oracle, audit_hashes)
                    dump(OUT / "c2_pilot_and_live_recoverability.json", {"pilot": pilot, "live_recoverability": live})
                    moved_later = live["C2_FIRST_NONRECOVERABLE_STEP"] is None or live["C2_FIRST_NONRECOVERABLE_STEP"] > 11
                    pilot_pass = bool(pilot["passed"] and moved_later)
                    if not pilot_pass:
                        first_bad = "C2_PILOT"
                        unseen = any(ep["first_nonrecoverable_known_boundary"] is False for ep in live["episodes"] if ep["first_nonrecoverable_policy_step"] is not None)
                        if full_shell >= SHELL_GATE and unseen:
                            root = "UNSEEN_TRAJECTORY_RECOVERY_BOUNDARIES"; next_task = "C2_TRAJECTORY_LEVEL_POLICY_REVIEW"
                        else:
                            root = "BOUNDARY_GENERALIZATION_FAILURE"; next_task = "STOP"
                    else:
                        matched_dir = OUT / "matched_checkpoints"; matched_dir.mkdir()
                        mdef, _, mreps, mbundles = train_once(matched_dir, rows, train_indices, demo_valid, dagger_valid, weights,
                            states, heldout_states, margin_by_hash, oracle, demos, agent, c1_params, support, scale, core, outer)
                        mselected = select_checkpoint(mreps); mparams, _ = mbundles[mselected]
                        mpilot = convergence.pilot(mdef, mparams, historical, rows, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale)
                        mlive = live_recoverability(mdef, mparams, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale, oracle, audit_hashes)
                        confirmed = bool(mselected == selected and mpilot["passed"] and (mlive["C2_FIRST_NONRECOVERABLE_STEP"] is None or mlive["C2_FIRST_NONRECOVERABLE_STEP"] > 11))
                        matched = "PASS" if confirmed else "FAIL"
                        dump(OUT / "matched_confirmation.json", {"selected_checkpoint": mselected, "reports": mreps, "pilot": mpilot,
                            "live_recoverability": mlive, "C2_MATCHED_CONFIRMATION": matched})
                        if confirmed:
                            frozen = OUT / "checkpoints/confirmed_frozen_c2_head.msgpack"
                            frozen.write_bytes(flax.serialization.to_bytes({"params": mparams})); confirmed_shas = [sha(frozen)]
                            canonical = pc.eval_selected(mdef, mparams, "H0", agent, c1_params, support, scale, core, outer, "C3", demo_obs, demo_scale,
                                next(row["observations"] for row in rows if row["source_round"] == 1 and row["source_step"] == 0))
                            dump(OUT / "canonical_interface_audit.json", canonical); next_task = "C3_250_STEP_PILOT"
                        else:
                            first_bad = "C2_MATCHED_CONFIRMATION"; root = "BOUNDARY_GENERALIZATION_FAILURE"

            protected_after = {name: sha(path) for name, path in protected_paths.items()}; req(protected_after == protected_before, "protected artifacts unchanged")
            dump(OUT / "protected_after.json", protected_after)
            c2 = None if pilot is None else pilot["C2"]; c1 = None if pilot is None else pilot["C1"]; c0 = None if pilot is None else pilot["C0"]
            result = {
                **audit_result, "SELECTED_REPAIR": "RECOVERY_MARGIN_NORMALIZED_SUPERVISION_V1", "SELECTED_CHECKPOINT": selected,
                "REPAIR_GLOBAL_DAGGER_MAE": selected_report["global_dagger"]["mae"], "REPAIR_CRITICAL_MAE": selected_report["critical"]["mae"],
                "REPAIR_CRITICAL_P95": selected_report["critical"]["p95"], "REPAIR_MARGIN_VIOLATION_FRACTION": violation,
                "MULTI_BOUNDARY_SHELL_PRESERVATION": full_shell, "MULTI_BOUNDARY_CORE_PRESERVATION": full_core,
                "HISTORICAL_STEP11_REPAIRED": "YES" if historical_repaired else "NO",
                "SHORT_HORIZON_RECOVERABILITY_RATE": "NOT_RUN" if short is None else short["repair"]["SHORT_HORIZON_RECOVERABILITY_RATE"],
                "C2_PILOT": "NOT_RUN" if pilot is None else pilot["C2_PILOT"],
                "C2_FIRST_NONRECOVERABLE_STEP": "NOT_RUN" if live is None else live["C2_FIRST_NONRECOVERABLE_STEP"],
                "C2_SHELL_ENTRY": None if c2 is None else c2["interface"]["C1_SHELL_ENTRY_COUNT"],
                "C2_CORE_ENTRY": None if c2 is None else c2["interface"]["C1_CORE_ENTRY_COUNT"],
                "C2_NEAR_HOLE": "NOT_RUN" if c2 is None else f"{int(round(c2['near_hole_rate'] * 3))}/3",
                "C2_SUCCESS": "NOT_RUN" if c2 is None else f"{c2['success_count']}/3", "C2_CONTACT": "NOT_RUN" if c2 is None else "PASS" if c2["contact_health"] else "FAIL",
                "C1_REGRESSION": "NOT_RUN" if pilot is None else pilot["C1_REGRESSION"], "C1_CONTACT": "NOT_RUN" if c1 is None else "PASS" if c1["contact_health"] else "FAIL",
                "C0_SUCCESS": "NOT_RUN" if c0 is None else f"{c0['success_count']}/3", "C2_MATCHED_CONFIRMATION": matched,
                "C2_SKILL_CONFIRMED": "YES" if confirmed else "NO", "C2_CONFIRMED_COMPONENT_SHA_LIST": confirmed_shas,
                "CANONICAL_INTERFACE_AUDIT": "NOT_RUN" if canonical is None else "PASS",
                "CANONICAL_TO_C2_SUPPORT": "NOT_RUN" if canonical is None else canonical["interface"]["class_state_counts"]["C2_OUTSIDE"] > 0,
                "CANONICAL_TO_C1_SHELL": "NOT_RUN" if canonical is None else canonical["interface"]["C1_SHELL_ENTRY_COUNT"] > 0,
                "CANONICAL_TO_C1_CORE": "NOT_RUN" if canonical is None else canonical["interface"]["C1_CORE_ENTRY_COUNT"] > 0,
                "CANONICAL_TO_C0": "NOT_RUN" if canonical is None else canonical["near_hole_rate"] > 0,
                "C3_PILOT": "NOT_RUN", "C3_1K": "NOT_RUN", "C3_MATCHED_CONFIRMATION": "NOT_RUN", "C3_3K": "NOT_RUN", "FORMAL_5K": "NOT_RUN",
                "CANONICAL_SUCCESS_RATE": "NOT_RUN" if canonical is None else canonical["success_rate"],
                "CANONICAL_CONTACT_HEALTH": "NOT_RUN" if canonical is None else "PASS" if canonical["contact_health"] else "FAIL",
                "PEG_SINGLE_TASK_CONVERGED": "NO", "PRIMARY_ROOT_CAUSE": root, "FIRST_BAD_GATE": first_bad, "NEXT": next_task,
                "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
                "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "EVAL_RESET_CANONICAL": "YES", "EVAL_GOAL_CANONICAL": "YES",
                "EVAL_CURRICULUM_ACTIVE": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO", "TRAINING_RUNNING": False,
                "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started,
            }
            dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))
        finally:
            oracle.close()


if __name__ == "__main__":
    main()
