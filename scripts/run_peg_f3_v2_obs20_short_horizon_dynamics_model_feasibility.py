#!/usr/bin/env python3
"""No-training Obs20 empirical one-step dynamics feasibility audit."""
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
import scripts.run_peg_f3_v2_c2_short_horizon_model_based_recovery_review_corrected as exact
import scripts.run_peg_f3_v2_c2_trajectory_recovery_head_v1_protocol_corrected as recovery
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config
from training.peg_insertion.progressive_frozen_residual_v2 import support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_obs20_short_horizon_dynamics_model_feasibility_20260909_001"
EXACT = exact.OUT
K_CANDIDATES = 64
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


class EmpiricalDynamics:
    """Deterministic nearest-transition delta model in normalized Obs20+action."""
    def __init__(self, observations, actions, next_observations, obs_scale):
        self.observations = observations
        self.actions = actions[:, :6]
        self.deltas = next_observations - observations
        self.obs_scale = obs_scale

    def predict(self, observation, action, excluded=None):
        obs_term = np.sum(np.square((self.observations - observation) / self.obs_scale), axis=1)
        action_term = np.sum(np.square(self.actions - np.asarray(action)[:6]), axis=1)
        distance = obs_term + action_term
        if excluded is not None: distance[np.asarray(excluded, np.int64)] = np.inf
        row = int(np.argmin(distance))
        return np.asarray(observation, np.float32) + self.deltas[row], row, float(np.sqrt(distance[row]))


def d_c1(obs, support, scale):
    return float(np.asarray(jax.device_get(support_distance(obs, support, scale))))


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    prior = json.loads((EXACT / "result.json").read_text())
    req(prior["NEXT"] == "OBS20_SHORT_HORIZON_DYNAMICS_MODEL_FEASIBILITY", "authoritative next")
    OUT.mkdir(parents=True); started = time.time()
    protected = {"production_demo": p4.DEMO, "exact_review": EXACT / "result.json",
        "global_c2_2k": recovery.CHECKPOINT, "r1_recovery_head": option.R1_CP}
    before = {k: sha(v) for k, v in protected.items()}; dump(OUT / "protected_before.json", before)
    req(before["production_demo"] == tail.DEMO_SHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes())
    observations = np.asarray([row["observations"] for row in demos], np.float32)
    actions = np.asarray([row["actions"] for row in demos], np.float32)
    next_observations = np.asarray([row["next_observations"] for row in demos], np.float32)
    demo_obs, demo_scale = distance_model(demos); ranges = basin.episode_ranges(demos)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    model = EmpiricalDynamics(observations, actions, next_observations, demo_scale)

    # Episode-held-out diagnostic: prediction cannot retrieve any transition
    # from the query episode.  This is descriptive, not an outcome-tuned gate.
    loo = []
    for episode, start, end in ranges:
        excluded = np.arange(start, end)
        for row in range(start, end):
            pred, source_row, distance = model.predict(observations[row], actions[row], excluded)
            error = (pred - next_observations[row]) / demo_scale
            loo.append({"episode": int(episode), "row": row, "source_row": source_row,
                "input_distance": distance, "normalized_next_l2": float(np.linalg.norm(error)),
                "D_C1_error": abs(d_c1(pred, support, scale) - d_c1(next_observations[row], support, scale))})
    loo_summary = {"rows": len(loo),
        "normalized_next_l2_median": float(np.median([x["normalized_next_l2"] for x in loo])),
        "normalized_next_l2_p95": float(np.quantile([x["normalized_next_l2"] for x in loo], .95)),
        "D_C1_error_median": float(np.median([x["D_C1_error"] for x in loo])),
        "D_C1_error_p95": float(np.quantile([x["D_C1_error"] for x in loo], .95))}

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
        boundary_reports = []
        try:
            for seed in SEEDS:
                rows, _ = option.rollout(common, seed)
                boundary, _ = recovery.first_boundary(rows, oracle)
                req(boundary is not None and boundary[1] == 28, "reproduce boundary")
                good, bad, _ = boundary; state = rows[good]; obs = state["observation"]
                candidate_rows = np.argsort(np.sum(np.square((observations - obs) / demo_scale), axis=1))[:K_CANDIDATES]
                predictions = []
                for source_row in candidate_rows:
                    predicted, transition_row, input_distance = model.predict(obs, actions[source_row])
                    predictions.append({"source_row": int(source_row), "transition_row": transition_row,
                        "input_distance": input_distance, "predicted_D_C1": d_c1(predicted, support, scale)})
                selected = min(predictions, key=lambda x: (x["predicted_D_C1"], x["source_row"]))
                branch = exact.prefix_then_fixed_suffix(oracle.env, state["snapshot"], actions,
                    selected["source_row"], 1, ranges, support, scale, outer)
                boundary_reports.append({"seed": seed, "last_recoverable_step": good,
                    "first_nonrecoverable_step": bad, "selected": selected,
                    "selected_exact_prefix_D_C1": branch["prefix"]["D_C1"],
                    "selected_exact_recoverable": branch["recoverable"],
                    "prediction_absolute_D_C1_error": abs(selected["predicted_D_C1"] - branch["prefix"]["D_C1"])})
        finally: oracle.close()
    preservation = float(np.mean([x["selected_exact_recoverable"] for x in boundary_reports]))
    feasible = bool(preservation == 1.0)
    dump(OUT / "episode_heldout_prediction.json", {"summary": loo_summary, "rows": loo})
    dump(OUT / "boundary_action_ranking.json", {"states": boundary_reports,
        "exact_recoverability": preservation})
    after = {k: sha(v) for k, v in protected.items()}; req(after == before, "protected artifact mutation")
    dump(OUT / "protected_after.json", after)
    if feasible:
        root = "OBS20_EMPIRICAL_ONE_STEP_MODEL_SELECTS_RECOVERABLE_ACTION_PLAN"
        next_task = "C2_OBS20_EMPIRICAL_MPC_OPTION_PILOT"
    else:
        root = "OBS20_EMPIRICAL_DYNAMICS_CANNOT_RANK_RECOVERABLE_BOUNDARY_ACTION"
        next_task = "HIERARCHICAL_C2_APPROACH_TO_C1_POLICY_REVIEW"
    result = {"AUTHORITATIVE": "YES", "OBS20_SHORT_HORIZON_DYNAMICS_MODEL_FEASIBILITY": "PASS" if feasible else "FAIL",
        "BOUNDARY_SELECTED_ACTION_EXACT_RECOVERABILITY": preservation,
        "EPISODE_HELDOUT_PREDICTION": loo_summary,
        "MODEL": "nearest-transition delta in normalized Obs20+action6",
        "MODEL_TRAINING": "NONE", "NEW_DATA": "NONE", "SELECTED_HORIZON": 1,
        "FORMAL_RUNTIME_INPUTS": "Obs20 plus candidate normalized action only",
        "PRIMARY_ROOT_CAUSE": root, "NEXT": next_task,
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "PEG_SINGLE_TASK_CONVERGED": "NO", "TRAINING_RUNNING": False,
        "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started}
    dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
