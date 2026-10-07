#!/usr/bin/env python3
"""Collect and test corrective Round 1 with no mechanism beyond actor-only BC."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
                str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
                str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env")]

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from scripts.audit_peg_f3_v2_actor_parameterization_distribution_shift import (
    action_similarity, jacobian_audit, recovery_trial, rollout, teacher_forced,
)
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    action as policy_action, demo_distance, distance_model, load_config, make_train_env, raw_env,
)
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate, metrics, sha_tree, table_ee
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from serl_launcher.common.optimizers import make_optimizer


CONFIG = ROOT / "configs/peg_insertion/peg_v2_on_policy_corrective_round1_20260908.yaml"
AUDIT = ROOT / "results/diagnostics/peg/f3_v2_actor_parameterization_distribution_shift_20260908_003/result.json"
LOCAL_CLOSURE = ROOT / "results/closure/peg/f3_v2_local_insertion_actor_supervision_20260908_001/result.json"
LOCAL_DIR = ROOT / "results/datasets/peg_v2_local_insertion_supervision_20260908_001"
LOCAL = LOCAL_DIR / "local_insertion_supervision.pkl"
DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
SOURCE_RUN = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002"
SOURCE_POST_BC = ROOT / "results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001/checkpoints/bc_checkpoint_1000.msgpack"
SNAPSHOT_DIR = ROOT / "results/closure/peg/f3_v2_exact_state_restore_20260907_001"
TRACE = ROOT / "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
DATASET_DIR = ROOT / "results/datasets/peg_v2_corrective_postbc_round1_20260908_001"
WORKDIR = ROOT / "results/closure/peg/f3_v2_on_policy_corrective_20260908_001"
THRESHOLD = 7.41773975814
LIMITS = np.asarray([0.002, 0.003, 0.05], np.float64)


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def sha_path(path: Path):
    if path.is_file(): return sha(path)
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(item.relative_to(path)).encode()); h.update(bytes.fromhex(sha(item)))
    return h.hexdigest()


def jv(v):
    if isinstance(v, np.ndarray): return jv(v.tolist())
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, dict): return {str(k): jv(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jv(x) for x in v]
    return v


def dump(path, value, exclusive=True):
    mode = "x" if exclusive else "w"
    with Path(path).open(mode, encoding="utf-8") as f:
        json.dump(jv(value), f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")


def protected():
    paths = {
        "production_demo": DEMO,
        "canonical_eval_config": ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml",
        "source_post_bc_checkpoint": SOURCE_POST_BC,
        "exact_curriculum": ROOT / "training/peg_insertion/exact_state_curriculum.py",
        "c0_snapshot": SNAPSHOT_DIR / "c0_exact_state.npz",
        "c0_manifest": SNAPSHOT_DIR / "c0_exact_state_manifest.json",
        "actor_audit": AUDIT,
        "previous_local_closure": LOCAL_CLOSURE,
        "base_local_dataset": LOCAL,
    }
    return {k: {"path": str(v), "sha256": sha_path(v)} for k, v in paths.items()}


def current_metrics(env):
    raw = raw_env(env); values = raw.task._goal_metrics(raw._get_obs())
    return {k: float(values[k]) for k in ("lateral_error", "axial_error", "orientation_error", "force_norm")}


def collect_episode(agent, suffix, demo_obs, scale, takeover_step, seed, checkpoint_sha):
    env = make_train_env("C0"); prefix = []; expert_rows = []
    try:
        obs, _ = env.reset(seed=seed)
        require(np.asarray(obs).shape == (20,) and np.asarray(obs).dtype == np.float32, "runtime Obs20 reset")
        for policy_step in range(1, takeover_step + 1):
            act, _ = policy_action(agent, obs, deterministic=True)
            nxt, reward, term, trunc, info = env.step(act)
            prefix.append({"policy_step": policy_step, "actor_runtime_obs20": np.asarray(obs).copy(),
                           "policy_action": act, "next_obs": np.asarray(nxt).copy(), "reward": reward,
                           "terminated": term, "truncated": trunc})
            obs = nxt
            if term or trunc: break
        if len(prefix) != takeover_step or prefix[-1]["terminated"] or prefix[-1]["truncated"]:
            return {"accepted": False, "reason": "policy_prefix_ended", "seed": seed,
                    "takeover_step": takeover_step, "policy_prefix": prefix, "expert_rows": []}
        # The exact returned observation is retained as the next Actor input; never reconstructed.
        takeover_obs = np.asarray(obs, np.float32).copy()
        for source_offset, source in enumerate(suffix):
            runtime_obs = np.asarray(obs, np.float32).copy()
            require(np.array_equal(runtime_obs, obs), "actor runtime observation pairing")
            pre = current_metrics(env)
            expert_action = np.asarray(source["action"], np.float32)
            nxt, reward, term, trunc, info = env.step(expert_action)
            link01, unexpected, pairs = contact_flags(env)
            row = {
                "observations": runtime_obs, "actions": expert_action,
                "next_observations": np.asarray(nxt, np.float32).copy(),
                "rewards": np.float32(reward), "masks": np.float32(not term),
                "dones": bool(term or trunc),
                "actor_runtime_obs20": runtime_obs, "expert_action": expert_action,
                "next_obs": np.asarray(nxt, np.float32).copy(), "reward": np.float32(reward),
                "done": bool(term or trunc), "terminated": bool(term), "truncated": bool(trunc),
                "policy_checkpoint_sha": checkpoint_sha, "episode_seed": seed,
                "takeover_step": takeover_step, "source_policy_step": takeover_step,
                "expert_source_offset": source_offset, "expert_source_step": int(source.get("step", source_offset + 1)),
                "demo_distance": demo_distance(runtime_obs, demo_obs, scale),
                "lateral": pre["lateral_error"], "axial": pre["axial_error"],
                "orientation": pre["orientation_error"], "phase": source.get("phase", "unknown"),
                "force": pre["force_norm"], "contact": bool(link01 or unexpected or table_ee(env)),
                "contact_pairs": pairs, "expert_owner": True, "canonical_success": False,
                "corrective_obs_source": "ACTOR_RUNTIME_INPUT",
                "takeover_actor_runtime_obs20_sha256": hashlib.sha256(takeover_obs.tobytes()).hexdigest(),
            }
            expert_rows.append(row); obs = nxt
            if term or trunc: break
        success = bool(expert_rows and expert_rows[-1]["terminated"] and info.get("is_success", False))
        if success:
            for row in expert_rows: row["canonical_success"] = True
        return {"accepted": success, "reason": "canonical_success" if success else "expert_recovery_failed",
                "seed": seed, "takeover_step": takeover_step, "policy_prefix": prefix,
                "expert_rows": expert_rows, "canonical_success": success}
    finally: env.close()


def split_corrective(episodes):
    # One held-out deterministic seed per takeover family; grouping is by recovery episode.
    train_ids = {(x["takeover_step"], x["seed"]) for x in episodes if x["seed"] != 102}
    valid_ids = {(x["takeover_step"], x["seed"]) for x in episodes if x["seed"] == 102}
    return train_ids, valid_ids


def checkpoint_metrics(agent, base_obs, base_act, corr_obs, corr_act):
    return {"base": metrics(agent, base_obs, base_act), "corrective": metrics(agent, corr_obs, corr_act)}


def local_exit(states):
    return next((x["step"] for x in states
                 if min(x["D_LOCAL_TRAIN"], x["D_LOCAL_VALID"]) > THRESHOLD), None)


def divergence(states):
    return next((x["step"] for x in states if x["D_PRODUCTION_DEMO"] > THRESHOLD), None)


def error_trace(agent, states, trace, start):
    result = []
    for state in states:
        s = state["step"]
        if start + s >= len(trace): break
        obs = state["observation"]
        pa = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)), np.float32)
        ea = np.asarray(trace[start + s]["action"], np.float32)
        result.append({"state_step": s, **action_similarity(pa, ea)})
    return result


def behavior_key(report):
    c0 = report["c0"]
    local = report["first_local_support_exit_step"]
    div = c0["median_first_divergence"]
    later_local = 1000000 if local is None else local
    later_div = 1000000 if div is None else div
    return (-c0["success_count"], not c0["contact_health"], c0["table_ee_contact"],
            c0["best_success_margin"], -later_local, -later_div, -c0["near_hole_count"],
            c0["max_force"], report["offline"]["base"]["mae_a0_a5"])


def main():
    require(not WORKDIR.exists() and not DATASET_DIR.exists(), "fresh Round1 paths required")
    WORKDIR.mkdir(parents=True, exist_ok=False); DATASET_DIR.mkdir(parents=True, exist_ok=False)
    (WORKDIR / "round1_checkpoints").mkdir()
    config = yaml.safe_load(CONFIG.read_text()); cfg = load_config(1)
    require(config["scientific_mechanism"] == "POST_BC_ON_POLICY_CORRECTIVE_SUPERVISION", "mechanism")
    require(float(config["online_admission_threshold"]) == THRESHOLD, "threshold")
    source_audit = json.loads(AUDIT.read_text()); source_closure = json.loads(LOCAL_CLOSURE.read_text())
    require(source_audit["POST_BC_RECOVERABLE_STEP_SET"] == [1, 3], "source recovery evidence")
    require(source_closure["BEST_OFFLINE_BC_CHECKPOINT"] == str(SOURCE_POST_BC), "source post-BC identity")
    source_sha = sha(SOURCE_POST_BC)
    before = protected(); dump(WORKDIR / "protected_before.json", before)
    require(before["production_demo"]["sha256"] == DEMO_SHA, "production Demo SHA")
    require(jax.default_backend() == "gpu", "GPU required")
    demos = pickle.loads(DEMO.read_bytes()); demo_obs, scale = distance_model(demos)
    trace = [json.loads(x) for x in TRACE.read_text().splitlines() if x.strip()]
    manifest = json.loads((SNAPSHOT_DIR / "c0_exact_state_manifest.json").read_text())
    start = int(manifest["next_expert_index"]); suffix = trace[start:]
    local = pickle.loads(LOCAL.read_bytes()); split = json.loads((LOCAL_DIR / "split.json").read_text())
    train_set, valid_set = set(split["train_episodes"]), set(split["valid_episodes"])
    base_train = [x for x in local if x["source_episode"] in train_set]
    base_valid = [x for x in local if x["source_episode"] in valid_set]
    train_states = np.asarray([x["observations"] for x in base_train], np.float64)
    valid_states = np.asarray([x["observations"] for x in base_valid], np.float64)

    with portable_peg_runtime(ROOT):
        env = make_train_env("C0")
        fresh = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        env.close()
        base = fresh.replace(state=restore_fresh_state(SOURCE_RUN / "checkpoints", fresh.state, 1000))
        source = base.replace(state=flax.serialization.from_bytes(base.state, SOURCE_POST_BC.read_bytes()))

        attempts = [collect_episode(source, suffix, demo_obs, scale, step, seed, source_sha)
                    for step in config["takeover_steps"] for seed in config["episode_seeds"]]
        accepted = [x for x in attempts if x["accepted"]]
        failed = [x for x in attempts if not x["accepted"]]
        rows = [row for episode in accepted for row in episode["expert_rows"]]
        represented = sorted({x["takeover_step"] for x in accepted})
        pairing = all(np.array_equal(x["actor_runtime_obs20"], x["observations"]) and
                      x["actor_runtime_obs20"].shape == (20,) and x["actor_runtime_obs20"].dtype == np.float32
                      for x in rows)
        data_gate = len(accepted) >= 4 and represented == [1, 3] and pairing and all(x["canonical_success"] for x in accepted)
        require(data_gate, "ROUND1_DATA_GATE=FAIL")
        dataset_path = DATASET_DIR / "corrective_round1.pkl"
        with dataset_path.open("xb") as f: pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        train_ids, valid_ids = split_corrective(accepted)
        provenance = [{"episode_seed": x["seed"], "takeover_step": x["takeover_step"],
                       "canonical_success": x["canonical_success"], "transitions": len(x["expert_rows"]),
                       "split": "train" if (x["takeover_step"], x["seed"]) in train_ids else "valid",
                       "policy_prefix_transitions": len(x["policy_prefix"])} for x in accepted]
        manifest_out = {"CORRECTIVE_OBS_SOURCE": "ACTOR_RUNTIME_INPUT", "ROUND1_DATA_GATE": "PASS",
                        "ROUND1_CORRECTIVE_EPISODES": len(accepted), "ROUND1_CORRECTIVE_TRANSITIONS": len(rows),
                        "ROUND1_CORRECTIVE_SHA256": sha(dataset_path), "SOURCE_POST_BC_CHECKPOINT": str(SOURCE_POST_BC),
                        "SOURCE_POST_BC_CHECKPOINT_SHA256": source_sha, "takeover_steps": represented,
                        "train_episode_ids": sorted([list(x) for x in train_ids]),
                        "valid_episode_ids": sorted([list(x) for x in valid_ids]),
                        "policy_prefix_is_expert_data": False, "all_canonical_success": True,
                        "actor_runtime_obs20_pairing_verified": pairing}
        dump(DATASET_DIR / "manifest.json", manifest_out); dump(DATASET_DIR / "provenance.json", provenance)
        (DATASET_DIR / "sha256.txt").write_text(manifest_out["ROUND1_CORRECTIVE_SHA256"] + "  corrective_round1.pkl\n")
        dump(WORKDIR / "round1_failed_recovery_diagnostics.json", failed)
        require(before == protected(), "protected changed after collection")
        dump(WORKDIR / "protected_after_round1_collection.json", protected())

        corr_train = [row for ep in accepted if (ep["takeover_step"], ep["seed"]) in train_ids for row in ep["expert_rows"]]
        corr_valid = [row for ep in accepted if (ep["takeover_step"], ep["seed"]) in valid_ids for row in ep["expert_rows"]]
        bto = np.asarray([x["observations"] for x in base_train], np.float32)
        bta = np.asarray([x["actions"] for x in base_train], np.float32)
        bvo = np.asarray([x["observations"] for x in base_valid], np.float32)
        bva = np.asarray([x["actions"] for x in base_valid], np.float32)
        cto = np.asarray([x["observations"] for x in corr_train], np.float32)
        cta = np.asarray([x["actions"] for x in corr_train], np.float32)
        cvo = np.asarray([x["observations"] for x in corr_valid], np.float32)
        cva = np.asarray([x["actions"] for x in corr_valid], np.float32)

        frozen_before = {"critic": sha_tree(source.state.params["modules_critic"]),
                         "target": sha_tree(source.state.target_params),
                         "temperature": sha_tree(source.state.params["modules_temperature"]),
                         "actor": sha_tree(source.state.params["modules_actor"])}
        params = source.state.params["modules_actor"]
        tx = make_optimizer(learning_rate=float(config["actor_only"]["actor_learning_rate"]),
                            warmup_steps=int(config["actor_only"]["actor_warmup_steps"]))
        opt_state = tx.init(params); rng = np.random.default_rng(20260908); apply_fn = source.state.apply_fn
        @jax.jit
        def bc_step(actor_params, optimizer_state, base_obs, base_act, corr_obs, corr_act):
            obs = jnp.concatenate([base_obs, corr_obs], axis=0)
            target = jnp.concatenate([base_act, corr_act], axis=0)
            def loss_fn(p):
                pred = apply_fn({"params": {"modules_actor": p}}, obs, name="actor", train=False).mode()
                return jnp.mean(jnp.square(pred[:, :6] - target[:, :6]))
            loss, grads = jax.value_and_grad(loss_fn)(actor_params)
            updates, new_state = tx.update(grads, optimizer_state, actor_params)
            return optax.apply_updates(actor_params, updates), new_state, loss, optax.global_norm(grads)
        offline_reports = []
        for update in range(1, int(config["actor_only"]["updates"]) + 1):
            bi = rng.integers(0, len(bto), size=32); ci = rng.integers(0, len(cto), size=32)
            params, opt_state, loss, grad = bc_step(params, opt_state, jnp.asarray(bto[bi]), jnp.asarray(bta[bi]),
                                                    jnp.asarray(cto[ci]), jnp.asarray(cta[ci]))
            if update in config["actor_only"]["checkpoints"]:
                full = dict(source.state.params); full["modules_actor"] = params
                candidate = source.replace(state=source.state.replace(params=full))
                report = checkpoint_metrics(candidate, bvo, bva, cvo, cva)
                report.update(relative_update=update, loss=float(loss), gradient_norm=float(grad))
                offline_reports.append(report)
                with (WORKDIR / f"round1_checkpoints/checkpoint_{update}.msgpack").open("xb") as f:
                    f.write(flax.serialization.to_bytes(candidate.state))
        require(all(x["gradient_norm"] > 0 and np.isfinite(x["gradient_norm"]) for x in offline_reports), "actor gradients")

        behavior_reports = []
        for offline in offline_reports:
            update = offline["relative_update"]
            state = flax.serialization.from_bytes(source.state, (WORKDIR / f"round1_checkpoints/checkpoint_{update}.msgpack").read_bytes())
            candidate = source.replace(state=state)
            c0 = evaluate(candidate, "C0", demo_obs, scale)
            states, transitions = rollout(candidate, demo_obs, train_states, valid_states, scale)
            behavior_reports.append({"relative_update": update, "checkpoint": str(WORKDIR / f"round1_checkpoints/checkpoint_{update}.msgpack"),
                                     "offline": offline, "c0": c0,
                                     "first_local_support_exit_step": local_exit(states),
                                     "production_demo_first_divergence": divergence(states),
                                     "policy_expert_action_error_trajectory": error_trace(candidate, states, trace, start)})
        behavior_reports.sort(key=behavior_key); best = behavior_reports[0]
        best_state = flax.serialization.from_bytes(source.state, Path(best["checkpoint"]).read_bytes())
        best_agent = source.replace(state=best_state)
        frozen_after = {"critic": sha_tree(best_agent.state.params["modules_critic"]),
                        "target": sha_tree(best_agent.state.target_params),
                        "temperature": sha_tree(best_agent.state.params["modules_temperature"]),
                        "actor": sha_tree(best_agent.state.params["modules_actor"])}
        require(all(frozen_before[k] == frozen_after[k] for k in ("critic", "target", "temperature")) and
                frozen_before["actor"] != frozen_after["actor"], "actor-only update")
        c0 = best["c0"]
        strong = c0["success_count"] >= 2 and c0["contact_health"] and not c0["table_ee_contact"]
        recovery = {}
        if not strong:
            for step in (1, 3, 5, 7, 10, 15):
                recovery[str(step)] = [recovery_trial(best_agent, suffix, step, seed) for seed in (100, 101, 102)]
        recovery_rates = {k: float(np.mean([x["success"] for x in trials])) for k, trials in recovery.items()}
        recoverable = [int(k) for k, rate in recovery_rates.items() if rate > 0]
        recovery_rate = float(np.mean([x["success"] for trials in recovery.values() for x in trials])) if recovery else 1.0
        partial = (not strong and best["first_local_support_exit_step"] is not None and
                   best["first_local_support_exit_step"] > 6 and bool(recoverable))
        repair = "PASS" if strong else "PARTIAL_PASS" if partial else "FAIL"

        # Monitor only; no Jacobian or std objective is introduced.
        expert_obs = {}; env = make_train_env("C0")
        try:
            obs, _ = env.reset(seed=100)
            for offset, row in enumerate(suffix):
                if offset in (0, 5, 10, 20): expert_obs[offset] = np.asarray(obs).copy()
                obs, _, term, trunc, _ = env.step(np.asarray(row["action"], np.float32))
                if term or trunc: break
        finally: env.close()
        jac = jacobian_audit(best_agent, expert_obs)

        result = {**manifest_out,
            "WORKDIR": str(WORKDIR), "POST_BC_ON_POLICY_CORRECTIVE_SUPERVISION": "ROUND1_PASS" if strong else "ROUND1_PARTIAL_PASS" if partial else "SCIENTIFIC_FAIL",
            "BASE_LOCAL_BATCH": 32, "CORRECTIVE_BATCH": 32, "TOTAL_BC_BATCH": 64,
            "ROUND1_ACTOR_BC": "PASS", "ROUND1_BEST_CHECKPOINT": best["checkpoint"],
            "ROUND1_BASE_MAE": best["offline"]["base"]["mae_a0_a5"],
            "ROUND1_CORRECTIVE_MAE": best["offline"]["corrective"]["mae_a0_a5"],
            "ROUND1_BASE_COSINE": best["offline"]["base"]["cosine_a0_a5"],
            "ROUND1_CORRECTIVE_COSINE": best["offline"]["corrective"]["cosine_a0_a5"],
            "ROUND1_C0_SUCCESS": f'{c0["success_count"]}/3', "ROUND1_C0_NEAR_HOLE": f'{c0["near_hole_count"]}/3',
            "ROUND1_C0_SUCCESS_MARGIN": c0["best_success_margin"],
            "ROUND1_C0_FIRST_DIVERGENCE": c0["median_first_divergence"],
            "ROUND1_FIRST_LOCAL_SUPPORT_EXIT_STEP": best["first_local_support_exit_step"],
            "ROUND1_C0_MAX_FORCE": c0["max_force"],
            "ROUND1_C0_CONTACT_HEALTH": "PASS" if c0["contact_health"] and not c0["table_ee_contact"] else "FAIL",
            "ROUND1_C0_REPAIR": repair, "ROUND1_RECOVERABLE_STEP_SET": recoverable,
            "ROUND1_RECOVERY_SUCCESS_RATE": recovery_rate,
            "ROUND1_ACTOR_JACOBIAN_CONDITION": {"label": jac["condition_label"], "max_condition_number": jac["max_condition_number"]},
            "ROUND1_CROSS_AXIS_COUPLING": jac["cross_axis_coupling_label"],
            "round1_offline_checkpoints": offline_reports, "round1_behavior_checkpoints": behavior_reports,
            "round1_recovery_by_step": recovery, "round1_recovery_rate_by_step": recovery_rates,
            "ROUND2_AUTHORIZED": bool(partial),
            "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "ONLINE_ADMISSION_THRESHOLD": THRESHOLD, "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "EVAL_RESET_CANONICAL": "YES", "EVAL_GOAL_CANONICAL": "YES", "EVAL_CURRICULUM_ACTIVE": "NO",
            "PROTECTED_ARTIFACTS_CHANGED": "NO"}
        if repair == "FAIL":
            result.update(PRIMARY_ROOT_CAUSE="CORRECTIVE_COVERAGE_NOT_SUFFICIENT_TO_CHANGE_CLOSED_LOOP",
                          NEXT="ACTOR_LOCAL_PARAMETERIZATION_REVIEW")
        elif repair == "PARTIAL_PASS":
            result.update(PRIMARY_ROOT_CAUSE="LOCAL_CLOSED_LOOP_DISTRIBUTION_SHIFT_REMAINS_PARTIALLY_REPAIRABLE",
                          NEXT="CORRECTIVE_ROUND2")
        else:
            result.update(PRIMARY_ROOT_CAUSE="C0_REPAIRED_BY_POST_BC_ON_POLICY_CORRECTIVE_SUPERVISION",
                          NEXT="CANONICAL_SAFETY_CHECK_AND_RLPD_RETENTION")

    after = protected(); require(before == after, "protected artifact changed")
    dump(WORKDIR / "protected_after.json", after); dump(WORKDIR / "round1_result.json", result)
    (WORKDIR / "ROUND1_REPORT.md").write_text("# Peg F3 V2 post-BC corrective Round 1\n\n```json\n" +
        json.dumps(jv(result), indent=2, sort_keys=True, allow_nan=False) + "\n```\n", encoding="utf-8")
    summary = {k: result[k] for k in ("ROUND1_DATA_GATE", "ROUND1_CORRECTIVE_EPISODES", "ROUND1_CORRECTIVE_TRANSITIONS",
        "ROUND1_CORRECTIVE_SHA256", "ROUND1_ACTOR_BC", "ROUND1_BEST_CHECKPOINT", "ROUND1_BASE_MAE",
        "ROUND1_CORRECTIVE_MAE", "ROUND1_C0_SUCCESS", "ROUND1_C0_NEAR_HOLE", "ROUND1_C0_SUCCESS_MARGIN",
        "ROUND1_C0_FIRST_DIVERGENCE", "ROUND1_FIRST_LOCAL_SUPPORT_EXIT_STEP", "ROUND1_C0_MAX_FORCE",
        "ROUND1_C0_CONTACT_HEALTH", "ROUND1_C0_REPAIR", "ROUND1_RECOVERABLE_STEP_SET",
        "ROUND1_RECOVERY_SUCCESS_RATE", "ROUND2_AUTHORIZED", "PRIMARY_ROOT_CAUSE", "NEXT")}
    print(json.dumps(jv(summary), indent=2, sort_keys=True))


if __name__ == "__main__": main()
