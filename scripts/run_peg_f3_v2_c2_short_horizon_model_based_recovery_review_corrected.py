#!/usr/bin/env python3
"""Corrected no-training Level-3 short-horizon plan-selection review."""
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
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, raw_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_short_horizon_model_based_recovery_review_20260909_002"
PILOT = option.OUT
HORIZONS = (1, 2, 4, 8)
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


def metrics(obs, support, scale):
    d_c1 = float(np.asarray(jax.device_get(support_distance(obs, support, scale))))
    g0 = float(np.asarray(jax.device_get(gate_value(obs))))
    return d_c1, g0


def simulate_prefix(env, snapshot, actions, start, end, support, scale):
    obs = basin.restore(env, snapshot); healthy = True; executed = 0; pairs = []
    for index in range(start, end):
        obs, _, terminated, truncated, info = env.step(actions[index]); executed += 1
        link, unexpected, pairs = contact_flags(env)
        force = float(info.get("force_norm", 0.0))
        healthy &= bool(not link and not unexpected and np.isfinite(force))
        if terminated or truncated or not healthy: break
    d_c1, g0 = metrics(obs, support, scale)
    return {"observation": np.asarray(obs, np.float32), "healthy": healthy,
        "executed": executed, "D_C1": d_c1, "g0": g0, "pairs": pairs}


def fixed_suffix_recovery(env, snapshot, actions, start, ranges, support, scale, outer):
    obs = basin.restore(env, snapshot)
    _, _, _, episode_end = basin.row_provenance(ranges, start)
    remaining = max(0, 500 - int(raw_env(env)._elapsed_steps))
    healthy = True; first_shell = None; executed = 0
    for index in range(start, min(episode_end, start + remaining)):
        obs, _, terminated, truncated, info = env.step(actions[index]); executed += 1
        d_c1, g0 = metrics(obs, support, scale)
        link, unexpected, _ = contact_flags(env)
        healthy &= bool(not link and not unexpected and np.isfinite(float(info.get("force_norm", 0.0))))
        if healthy and (d_c1 <= outer or g0 == 0.0): first_shell = executed; break
        if terminated or truncated or not healthy: break
    return {"recoverable": bool(healthy and first_shell is not None),
        "contact_healthy": healthy, "first_shell_step": first_shell, "executed": executed}


def prefix_then_fixed_suffix(env, snapshot, actions, source_row, horizon, ranges, support, scale, outer):
    _, _, _, episode_end = basin.row_provenance(ranges, source_row)
    prefix_end = min(episode_end, source_row + horizon)
    prefix = simulate_prefix(env, snapshot, actions, source_row, prefix_end, support, scale)
    if not prefix["healthy"]:
        return {"prefix": prefix, "recoverable": False, "suffix": None}
    # Capture the exact post-prefix state and continue the same contiguous plan.
    post = basin.capture(env, prefix["observation"])
    suffix = fixed_suffix_recovery(env, post, actions, prefix_end, ranges, support, scale, outer)
    return {"prefix": prefix, "suffix": suffix, "recoverable": suffix["recoverable"]}


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
    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos)
    actions = np.asarray([row["actions"] for row in demos], np.float32)
    ranges = basin.episode_ranges(demos)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    r1 = flax.serialization.msgpack_restore(option.R1_CP.read_bytes()); cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common_agent, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common_agent), review.SOURCE_CP.read_bytes())
        agent = common_agent.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        global_def, global_params = tail.load_head_checkpoint(recovery.CHECKPOINT)
        global_fn = lambda obs: tail.checkpoint_action(global_def, global_params, obs, agent,
            c1_params, support, scale, core, outer)
        common = dict(global_action_fn=global_fn, local_params=r1["params"],
            recovery_support=np.asarray(r1["support"], np.float32), recovery_scale=scale,
            recovery_outer=float(r1["recovery_outer"]), agent=agent, c1_params=c1_params,
            c1_support=support, c1_scale=scale, c1_core=core, c1_outer=outer)
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, support, scale, core, outer)
        env = oracle.env; reports = []
        try:
            for seed in SEEDS:
                rows, _ = option.rollout(common, seed)
                boundary, _ = recovery.first_boundary(rows, oracle)
                req(boundary is not None and boundary[1] == 28, "reproduce step28 boundary")
                good, bad, takeover = boundary; state = rows[good]
                source_row = int(takeover["expert_source_row"])
                fixed_control = fixed_suffix_recovery(env, state["snapshot"], actions, source_row,
                    ranges, support, scale, outer)
                req(fixed_control["recoverable"], "known fixed expert suffix control")
                nearest = np.argsort(np.sum(np.square((demo_obs - state["observation"]) / demo_scale), axis=1))[:K_NEAREST]
                horizon_reports = {}
                for horizon in HORIZONS:
                    scored = []
                    for row_index in nearest:
                        _, _, _, episode_end = basin.row_provenance(ranges, int(row_index))
                        prefix = simulate_prefix(env, state["snapshot"], actions, int(row_index),
                            min(episode_end, int(row_index) + horizon), support, scale)
                        scored.append({"source_row": int(row_index), "prefix": prefix})
                    healthy = [x for x in scored if x["prefix"]["healthy"]]
                    req(healthy, "healthy exact-model candidates")
                    selected = min(healthy, key=lambda x: (x["prefix"]["D_C1"], x["source_row"]))
                    branch = prefix_then_fixed_suffix(env, state["snapshot"], actions,
                        selected["source_row"], horizon, ranges, support, scale, outer)
                    horizon_reports[horizon] = {"selected_source_row": selected["source_row"],
                        "selected_prefix_D_C1": selected["prefix"]["D_C1"],
                        "selected_prefix_g0": selected["prefix"]["g0"],
                        "selected_recoverable": branch["recoverable"], "selected_branch": branch,
                        "healthy_candidates": len(healthy)}
                reports.append({"seed": seed, "last_recoverable_step": good,
                    "first_nonrecoverable_step": bad, "authoritative_source_row": source_row,
                    "fixed_expert_suffix_control": fixed_control, "horizons": horizon_reports})
        finally: oracle.close()
        preservation = {h: float(np.mean([x["horizons"][h]["selected_recoverable"] for x in reports])) for h in HORIZONS}
        feasible_horizons = [h for h in HORIZONS if preservation[h] == 1.0]
        feasible = bool(feasible_horizons); selected_horizon = min(feasible_horizons) if feasible else None
        dump(OUT / "fixed_suffix_exact_branching_review.json", {"states": reports,
            "preservation_by_horizon": preservation, "selected_shortest_horizon": selected_horizon})
        after = {k: sha(v) for k, v in protected.items()}; req(after == before, "protected artifact mutation")
        dump(OUT / "protected_after.json", after)
        if feasible:
            root = "OBS20_ACTION_LIBRARY_CONTAINS_RECOVERY_PLAN_REQUIRING_SHORT_HORIZON_TRANSITION_SELECTION"
            next_task = "OBS20_SHORT_HORIZON_DYNAMICS_MODEL_FEASIBILITY"
        else:
            root = "SHORT_HORIZON_PLAN_SELECTION_CANNOT_RECOVER_OPTION_BOUNDARY"
            next_task = "HIERARCHICAL_C2_APPROACH_TO_C1_POLICY_REVIEW"
        result = {"AUTHORITATIVE": "YES", "SHORT_HORIZON_MODEL_BASED_RECOVERY_REVIEW": "PASS" if feasible else "FAIL",
            "FIXED_EXPERT_SUFFIX_CONTROL": "PASS", "PRESERVATION_BY_HORIZON": preservation,
            "SELECTED_SHORTEST_HORIZON": selected_horizon,
            "MODEL_PREDICTIVE_CANDIDATE_POOL": f"{K_NEAREST} nearest existing production-Demo contiguous plans",
            "MODEL_PREDICTIVE_OBJECTIVE": "minimum exact-model D_C1 after fixed horizon",
            "DIAGNOSTIC_ONLY_PRIVILEGED_STATE": "YES", "FORMAL_RUNTIME_PRIVILEGED_STATE_AUTHORIZED": "NO",
            "TRAINING_STARTED": False, "PRIMARY_ROOT_CAUSE": root, "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "PEG_SINGLE_TASK_CONVERGED": "NO", "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started}
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
