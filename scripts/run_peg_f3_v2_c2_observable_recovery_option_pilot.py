#!/usr/bin/env python3
"""No-training pilot for the unique observable C2 recovery option."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env")]

import flax
import jax
import numpy as np

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_trajectory_recovery_head_v1_protocol_corrected as recovery
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    distance_model, evaluate, load_config, make_train_env,
)
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.c2_observable_recovery_option_v1 import (
    RecoveryPhase, lower_action, recovery_option_step,
)
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_observable_recovery_option_pilot_20260909_001"
FEASIBILITY = ROOT / "results/diagnostics/peg/f3_v2_c2_history_and_option_feasibility_20260909_001"
R1_CP = recovery.OUT / "checkpoints/trajectory_recovery_head_r1_3000.msgpack"
SEEDS = (100, 101, 102)
BASELINE_FIRST_BAD = 10


def req(value, message):
    if not value: raise RuntimeError(message)


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
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def tree_hash(tree):
    return hashlib.sha256(flax.serialization.to_bytes(jax.device_get(tree))).hexdigest()


def reset_observations():
    result = []
    for level in ("C0", "C1", "C2"):
        env = make_train_env(level)
        try:
            for seed in SEEDS:
                obs, _ = env.reset(seed=seed)
                result.append(np.asarray(obs, np.float32).copy())
        finally: env.close()
    return np.asarray(result, np.float32)


class OptionPolicy:
    def __init__(self, common, reset_obs):
        self.common = common
        self.reset_obs = reset_obs
        self.phase = RecoveryPhase.APPROACH
        self.trace = []

    def reset(self):
        self.phase = RecoveryPhase.APPROACH

    def _is_reset(self, obs):
        return bool(np.any(np.all(self.reset_obs == np.asarray(obs, np.float32), axis=1)))

    def act(self, obs):
        if self._is_reset(obs): self.reset()
        self.phase, action, info = recovery_option_step(self.phase, obs, **self.common)
        self.trace.append(info)
        return action

    def sample_actions(self, observations, seed=None, deterministic=False, argmax=False, **_):
        del seed, deterministic, argmax
        return self.act(np.asarray(observations, np.float32))

    def stats(self):
        return {
            "steps": len(self.trace),
            "approach_steps": sum(x["phase"] == "APPROACH" for x in self.trace),
            "recovery_steps": sum(x["phase"] == "RECOVERY" for x in self.trace),
            "lower_handoff_steps": sum(x["phase"] == "LOWER_HANDOFF" for x in self.trace),
            "recovery_initiations": sum(x["transition"] == "APPROACH_TO_RECOVERY" for x in self.trace),
            "lower_handoffs": sum(x["transition"] == "RECOVERY_TO_LOWER_HANDOFF" for x in self.trace),
            "trace": self.trace,
        }


def rollout(common, seed):
    policy = OptionPolicy(common, np.zeros((0, 20), np.float32))
    policy.reset()
    env = make_train_env("C2"); rows = []
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(500):
            action = policy.act(obs)
            link, unexpected, pairs = contact_flags(env)
            rows.append({"seed": seed, "policy_step": step,
                "observation": np.asarray(obs, np.float32).copy(),
                "snapshot": capture(env, obs), "policy_action": action,
                "option": policy.trace[-1], "contact_before": bool(link or unexpected),
                "contact_pairs": pairs})
            if link or unexpected: break
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated: break
    finally: env.close()
    return rows, policy.stats()


def evaluate_all(common, reset_obs, demo_obs, demo_scale, baseline_c1):
    records = {}
    for level in ("C0", "C1", "C2"):
        policy = OptionPolicy(common, reset_obs)
        records[level] = evaluate(policy, level, demo_obs, demo_scale, seeds=SEEDS)
        records[level]["option"] = policy.stats()
    c1_keys = ("success_count", "near_hole_rate", "median_first_divergence",
        "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
    c1_regression = all(records["C1"][key] == baseline_c1[key] for key in c1_keys)
    c2_option = records["C2"]["option"]
    passed = bool(records["C0"]["success_count"] == 3 and records["C0"]["contact_health"]
        and c1_regression and records["C2"]["contact_health"]
        and c2_option["recovery_initiations"] >= 2 and c2_option["lower_handoffs"] >= 2)
    return {**records, "C1_REGRESSION": "PASS" if c1_regression else "FAIL",
        "interface_pass": passed}


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    prior = json.loads((FEASIBILITY / "result.json").read_text())
    req(prior["NEXT"] == "C2_OBSERVABLE_RECOVERY_OPTION_PILOT", "authoritative next")
    OUT.mkdir(parents=True); started = time.time()
    protected = {"production_demo": p4.DEMO, "global_c2_2k": recovery.CHECKPOINT,
        "r1_recovery_head": R1_CP, "c1_support": review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl"}
    before = {k: sha(v) for k, v in protected.items()}; dump(OUT / "protected_before.json", before)
    req(before["production_demo"] == tail.DEMO_SHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    c1_support = np.asarray(support_bundle["observations"], np.float32)
    c1_scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    c1_core = float(preflight["C1_CORE_THRESHOLD"]); c1_outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    r1 = flax.serialization.msgpack_restore(R1_CP.read_bytes())
    cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common_agent, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common_agent), review.SOURCE_CP.read_bytes())
        agent = common_agent.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        global_def, global_params = tail.load_head_checkpoint(recovery.CHECKPOINT)
        frozen_hashes = {"base_actor": geo.thash(agent.state.params["modules_actor"]),
            "c1": geo.thash(c1_params), "global_c2": tree_hash(global_params), "local_r1": tree_hash(r1["params"])}
        global_fn = lambda obs: tail.checkpoint_action(global_def, global_params, obs, agent,
            c1_params, c1_support, c1_scale, c1_core, c1_outer)
        common = dict(global_action_fn=global_fn, local_params=r1["params"],
            recovery_support=np.asarray(r1["support"], np.float32), recovery_scale=c1_scale,
            recovery_outer=float(r1["recovery_outer"]), agent=agent, c1_params=c1_params,
            c1_support=c1_support, c1_scale=c1_scale, c1_core=c1_core, c1_outer=c1_outer)

        resets = reset_observations()
        # Exact C0/C1 protection: on every easier reset, the option must equal
        # the already frozen lower policy bit-for-bit.
        static_rows = []
        for obs in resets[:6]:
            policy = OptionPolicy(common, resets); option_action = policy.act(obs)
            frozen_lower = lower_action(agent, c1_params, obs)
            static_rows.append(float(np.max(np.abs(option_action - frozen_lower))))
        static_max = max(static_rows); req(static_max == 0.0, "C0/C1 action invariance")

        baseline_c1 = pc.eval_selected(global_def, global_params, "H0", agent, c1_params,
            c1_support, c1_scale, c1_core, c1_outer, "C1", demo_obs, demo_scale, np.zeros(20, np.float32))
        pilot = evaluate_all(common, resets, demo_obs, demo_scale, baseline_c1)
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, c1_support, c1_scale, c1_core, c1_outer)
        try:
            episodes = []; first_bad = []
            for seed in SEEDS:
                rows, stats = rollout(common, seed)
                boundary, audited = recovery.first_boundary(rows, oracle)
                first_bad.append(None if boundary is None else boundary[1])
                episodes.append({"seed": seed, "steps": len(rows), "option": stats,
                    "first_nonrecoverable": None if boundary is None else boundary[1], "recoverability_audit": audited})
        finally: oracle.close()
        later = all(step is None or step > BASELINE_FIRST_BAD for step in first_bad)
        pilot_pass = bool(pilot["interface_pass"] and later)
        dump(OUT / "pilot.json", {"evaluation": pilot, "trajectory_audit": episodes,
            "first_nonrecoverable_steps": first_bad, "materially_later_than_baseline": later})

        matched = "NOT_RUN"
        if pilot_pass:
            confirmation = evaluate_all(common, resets, demo_obs, demo_scale, baseline_c1)
            matched = "PASS" if confirmation == pilot else "FAIL"
            dump(OUT / "matched_confirmation.json", {"evaluation": confirmation,
                "exact_match_to_pilot": confirmation == pilot})
            pilot_pass = bool(pilot_pass and matched == "PASS")
        if pilot_pass:
            option_config = {"version": "OBSERVABLE_INITIATION_TERMINATION_OPTION_CONTROLLER_V1",
                "initiation": "D_REC <= frozen_R1_recovery_outer and g2 > 0",
                "termination": "D_C1 <= frozen_C1_shell_outer or g0 == 0",
                "runtime_inputs": "Obs20 only", "global_c2_checkpoint": str(recovery.CHECKPOINT),
                "local_recovery_checkpoint": str(R1_CP), "frozen_hashes": frozen_hashes,
                "recovery_outer": float(r1["recovery_outer"]), "c1_core": c1_core, "c1_outer": c1_outer}
            dump(OUT / "C2_CONFIRMED_OBSERVABLE_RECOVERY_OPTION_V1.json", option_config)

        req(geo.thash(agent.state.params["modules_actor"]) == frozen_hashes["base_actor"]
            and geo.thash(c1_params) == frozen_hashes["c1"]
            and tree_hash(global_params) == frozen_hashes["global_c2"]
            and tree_hash(r1["params"]) == frozen_hashes["local_r1"], "frozen model mutation")
        after = {k: sha(v) for k, v in protected.items()}; req(after == before, "protected artifact mutation")
        dump(OUT / "protected_after.json", after)
        root = "NONE" if pilot_pass else "OBSERVABLE_RECOVERY_OPTION_CANNOT_PRESERVE_RECOVERABILITY_TO_C1_HANDOFF"
        next_task = "CANONICAL_INTERFACE_REACHABILITY_AUDIT" if pilot_pass else "SHORT_HORIZON_MODEL_BASED_RECOVERY_REVIEW"
        result = {"AUTHORITATIVE": "YES", "C2_OBSERVABLE_RECOVERY_OPTION_PILOT": "PASS" if pilot_pass else "FAIL",
            "C2_SKILL_CONFIRMED": "YES" if pilot_pass else "NO", "C2_MATCHED_CONFIRMATION": matched,
            "C0_SUCCESS": f"{pilot['C0']['success_count']}/3", "C1_REGRESSION": pilot["C1_REGRESSION"],
            "C1_CONTACT": "PASS" if pilot["C1"]["contact_health"] else "FAIL",
            "C2_SUCCESS": f"{pilot['C2']['success_count']}/3", "C2_CONTACT": "PASS" if pilot["C2"]["contact_health"] else "FAIL",
            "C2_RECOVERY_INITIATIONS": pilot["C2"]["option"]["recovery_initiations"],
            "C2_LOWER_HANDOFFS": pilot["C2"]["option"]["lower_handoffs"],
            "FIRST_NONRECOVERABLE_STEPS": first_bad,
            "FIRST_NONRECOVERABLE_MATERIALLY_LATER_THAN_BASELINE": "YES" if later else "NO",
            "C0_C1_STATIC_MAX_ACTION_DELTA": static_max, "TRAINING_STARTED": False,
            "PRIMARY_ROOT_CAUSE": root, "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "C0_C1_GLOBAL_C2_R1_FROZEN": "YES",
            "PROTECTED_ARTIFACTS_CHANGED": "NO", "PEG_SINGLE_TASK_CONVERGED": "NO",
            "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started}
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
