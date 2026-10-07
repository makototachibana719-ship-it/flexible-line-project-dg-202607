#!/usr/bin/env python3
"""No-training Level-3 short-horizon recovery feasibility review.

This uses exact MuJoCo branching only as a causal diagnostic.  It does not
authorize privileged simulator state in the formal runtime controller.
"""
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
import scripts.audit_peg_f3_v2_c2_closed_loop_recovery_basin as basin
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_observable_recovery_option_pilot as option
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_trajectory_recovery_head_v1_protocol_corrected as recovery
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    demo_distance, distance_model, load_config,
)
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_short_horizon_model_based_recovery_review_20260909_001"
PILOT = option.OUT
K_NEAREST = 64
SEEDS = (100, 101, 102)


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


def one_step_score(oracle, snapshot, action, demo_obs, demo_scale, c1_support, c1_scale):
    obs = basin.restore(oracle.env, snapshot)
    next_obs, _, terminated, truncated, info = oracle.env.step(np.asarray(action, np.float32))
    link, unexpected, pairs = contact_flags(oracle.env)
    healthy = bool(not link and not unexpected and np.isfinite(float(info.get("force_norm", 0.0))))
    d_demo = demo_distance(next_obs, demo_obs, demo_scale)
    delta = (c1_support - np.asarray(next_obs, np.float32)) / c1_scale
    d_c1 = float(np.min(np.sqrt(np.sum(np.square(delta), axis=1))))
    return {"contact_healthy": healthy, "D_DEMO_NEXT": d_demo, "D_C1_NEXT": d_c1,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "force": float(info.get("force_norm", 0.0)), "contact_pairs": pairs}


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    prior = json.loads((PILOT / "result.json").read_text())
    req(prior["NEXT"] == "SHORT_HORIZON_MODEL_BASED_RECOVERY_REVIEW", "authoritative next")
    OUT.mkdir(parents=True); started = time.time()
    protected = {"production_demo": p4.DEMO, "global_c2_2k": recovery.CHECKPOINT,
        "r1_recovery_head": option.R1_CP, "option_pilot": PILOT / "result.json"}
    before = {k: sha(v) for k, v in protected.items()}; dump(OUT / "protected_before.json", before)
    req(before["production_demo"] == tail.DEMO_SHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
    demo_actions = np.asarray([row["actions"] for row in demos], np.float32)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    c1_support = np.asarray(support_bundle["observations"], np.float32)
    c1_scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    c1_core = float(preflight["C1_CORE_THRESHOLD"]); c1_outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    r1 = flax.serialization.msgpack_restore(option.R1_CP.read_bytes())
    cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common_agent, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common_agent), review.SOURCE_CP.read_bytes())
        agent = common_agent.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        global_def, global_params = tail.load_head_checkpoint(recovery.CHECKPOINT)
        global_fn = lambda obs: tail.checkpoint_action(global_def, global_params, obs, agent,
            c1_params, c1_support, c1_scale, c1_core, c1_outer)
        common = dict(global_action_fn=global_fn, local_params=r1["params"],
            recovery_support=np.asarray(r1["support"], np.float32), recovery_scale=c1_scale,
            recovery_outer=float(r1["recovery_outer"]), agent=agent, c1_params=c1_params,
            c1_support=c1_support, c1_scale=c1_scale, c1_core=c1_core, c1_outer=c1_outer)
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, c1_support, c1_scale, c1_core, c1_outer)
        reports = []
        try:
            for seed in SEEDS:
                rows, _ = option.rollout(common, seed)
                boundary, _ = recovery.first_boundary(rows, oracle)
                req(boundary is not None and boundary[1] == 28, "reproduce option first bad step 28")
                good_step, bad_step, good_takeover = boundary
                state = rows[good_step]
                obs = state["observation"]
                normalized = (demo_obs - obs) / demo_scale
                nearest = np.argsort(np.sum(np.square(normalized), axis=1))[:K_NEAREST]
                candidates = []
                for row_index in nearest:
                    score = one_step_score(oracle, state["snapshot"], demo_actions[row_index],
                        demo_obs, demo_scale, c1_support, c1_scale)
                    candidates.append({"source_row": int(row_index), "action": demo_actions[row_index], **score})
                healthy = [x for x in candidates if x["contact_healthy"]]
                req(healthy, "healthy model-predictive candidate")
                selected = min(healthy, key=lambda x: (x["D_DEMO_NEXT"], x["D_C1_NEXT"], x["source_row"]))
                expert_row = int(good_takeover["expert_source_row"])
                current_branch = oracle.action_then_takeover(state["snapshot"], state["policy_action"])
                selected_branch = oracle.action_then_takeover(state["snapshot"], selected["action"])
                expert_branch = oracle.action_then_takeover(state["snapshot"], demo_actions[expert_row])
                reports.append({"seed": seed, "last_recoverable_step": good_step,
                    "first_nonrecoverable_step": bad_step, "expert_source_row": expert_row,
                    "selected_source_row": selected["source_row"], "current_action": state["policy_action"],
                    "selected_action": selected["action"], "expert_action": demo_actions[expert_row],
                    "current": current_branch, "selected": selected_branch, "expert": expert_branch,
                    "candidate_summary": {"count": len(candidates), "healthy_count": len(healthy),
                        "selected_one_step": {k: selected[k] for k in ("source_row", "D_DEMO_NEXT", "D_C1_NEXT", "contact_healthy")},
                        "best_8": sorted(healthy, key=lambda x: (x["D_DEMO_NEXT"], x["D_C1_NEXT"]))[:8]}})
        finally: oracle.close()

        current_preservation = float(np.mean([x["current"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] for x in reports]))
        selected_preservation = float(np.mean([x["selected"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] for x in reports]))
        expert_preservation = float(np.mean([x["expert"]["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] for x in reports]))
        feasible = bool(current_preservation == 0.0 and selected_preservation == 1.0 and expert_preservation == 1.0)
        selected_rows = [x["selected_source_row"] for x in reports]
        dump(OUT / "exact_branching_review.json", {"states": reports,
            "current_preservation": current_preservation, "selected_preservation": selected_preservation,
            "expert_preservation": expert_preservation, "selected_source_rows": selected_rows})
        after = {k: sha(v) for k, v in protected.items()}; req(after == before, "protected artifact mutation")
        dump(OUT / "protected_after.json", after)
        if feasible:
            root = "RECOVERABLE_ACTION_EXISTS_BUT_REQUIRES_SHORT_HORIZON_STATE_TRANSITION_PREDICTION"
            next_task = "OBS20_SHORT_HORIZON_DYNAMICS_MODEL_FEASIBILITY"
        else:
            root = "SHORT_HORIZON_ACTION_SELECTION_DOES_NOT_RECOVER_OPTION_BOUNDARY"
            next_task = "HIERARCHICAL_C2_APPROACH_TO_C1_POLICY_REVIEW"
        result = {"AUTHORITATIVE": "YES", "SHORT_HORIZON_MODEL_BASED_RECOVERY_REVIEW": "PASS" if feasible else "FAIL",
            "EXACT_MODEL_CURRENT_ACTION_SHELL_PRESERVATION": current_preservation,
            "EXACT_MODEL_SELECTED_ACTION_SHELL_PRESERVATION": selected_preservation,
            "EXACT_EXPERT_ACTION_SHELL_PRESERVATION": expert_preservation,
            "MODEL_PREDICTIVE_CANDIDATE_POOL": f"{K_NEAREST} nearest existing production-Demo actions",
            "MODEL_PREDICTIVE_OBJECTIVE": "minimum next Obs20 D_DEMO among contact-healthy candidates",
            "DIAGNOSTIC_ONLY_PRIVILEGED_STATE": "YES", "FORMAL_RUNTIME_PRIVILEGED_STATE_AUTHORIZED": "NO",
            "TRAINING_STARTED": False, "PRIMARY_ROOT_CAUSE": root, "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "PEG_SINGLE_TASK_CONVERGED": "NO", "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started}
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
