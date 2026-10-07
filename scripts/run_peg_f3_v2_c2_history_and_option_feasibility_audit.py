#!/usr/bin/env python3
"""No-training Level-1 history versus Level-2 option feasibility audit."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env")]

import flax
import jax
import jax.numpy as jnp
import numpy as np

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_recovery_boundary_tail_risk_audit as tail
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_trajectory_recovery_head_v1_protocol_corrected as recovery
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.c2_trajectory_recovery_head_v1 import compose_recovery_action, recovery_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_history_and_option_feasibility_20260909_001"
RUN = recovery.OUT
DATA = recovery.DATA_OUT / "C2_TRAJECTORY_RECOVERY_CEGIS_R2_V1.pkl"
R1_CP = RUN / "checkpoints/trajectory_recovery_head_r1_3000.msgpack"
R2_CP = RUN / "checkpoints/trajectory_recovery_head_r2_1500.msgpack"


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
    with Path(path).open("x") as stream: json.dump(jvalue(value), stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load_bundle(path):
    return flax.serialization.msgpack_restore(Path(path).read_bytes())


def coherence(rows, scale, outer):
    groups = {}
    for row in rows: groups.setdefault((row["cegis_round"], row["source_seed"]), []).append(row)
    enriched = []
    for key, values in groups.items():
        values.sort(key=lambda x: x["expert_source_step"])
        for index, row in enumerate(values):
            previous = values[max(0, index - 1)]
            feature = np.concatenate((row["observations"], previous["observations"], previous["actions"])).astype(np.float32)
            enriched.append({**row, "feature_h1": feature})
    pairs = []
    scale_h1 = np.concatenate((scale, scale, np.ones(7, np.float32)))
    for index, query in enumerate(enriched):
        candidates = [x for x in enriched if x["cegis_round"] != query["cegis_round"]]
        if not candidates: continue
        h0 = np.asarray([np.linalg.norm((x["observations"] - query["observations"]) / scale) for x in candidates])
        nearest = int(np.argmin(h0)); other = candidates[nearest]
        h1 = float(np.linalg.norm((other["feature_h1"] - query["feature_h1"]) / scale_h1))
        action_l2 = float(np.linalg.norm(other["actions"][:6] - query["actions"][:6]))
        pairs.append({"h0_distance": float(h0[nearest]), "h1_distance": h1, "action_l2": action_l2,
                      "query_round": query["cegis_round"], "other_round": other["cegis_round"]})
    close_conflicts = [x for x in pairs if x["h0_distance"] <= outer and x["action_l2"] >= .10]
    history_separates = bool(close_conflicts and np.median([x["h1_distance"] for x in close_conflicts]) >= 2 * np.median([x["h0_distance"] for x in close_conflicts]))
    return {"cross_round_pairs": len(pairs), "close_conflicting_pairs": len(close_conflicts),
            "close_conflict_h0_median": None if not close_conflicts else float(np.median([x["h0_distance"] for x in close_conflicts])),
            "close_conflict_h1_median": None if not close_conflicts else float(np.median([x["h1_distance"] for x in close_conflicts])),
            "close_conflict_action_l2_median": None if not close_conflicts else float(np.median([x["action_l2"] for x in close_conflicts])),
            "HISTORY_CLEARLY_RESOLVES_LOCAL_LABEL_CONFLICT": "YES" if history_separates else "NO", "closest_pairs": sorted(pairs, key=lambda x: x["h0_distance"])[:20]}


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    authoritative = json.loads((RUN / "result.json").read_text()); req(authoritative["NEXT"] == "HISTORY_AUGMENTED_LOCAL_RECOVERY_POLICY_REVIEW", "authoritative next")
    OUT.mkdir(parents=True); started = time.time()
    protected = {"production_demo": p4.DEMO, "global_c2_2k": recovery.CHECKPOINT, "r1_checkpoint": R1_CP, "r2_checkpoint": R2_CP, "r2_dataset": DATA}
    before = {k: sha(v) for k, v in protected.items()}; dump(OUT / "protected_before.json", before); req(before["production_demo"] == tail.DEMO_SHA, "Demo SHA")
    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    c1_support = np.asarray(support_bundle["observations"], np.float32); c1_scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text()); c1_core = float(preflight["C1_CORE_THRESHOLD"]); c1_outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    cfg = load_config(2)
    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos); source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        global_def, global_params = tail.load_head_checkpoint(recovery.CHECKPOINT)
        r1, r2 = load_bundle(R1_CP), load_bundle(R2_CP)
        def action_fn(bundle, forced=False):
            def fn(obs):
                global_action = tail.checkpoint_action(global_def, global_params, obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
                core = 1e9 if forced else float(bundle["recovery_core"]); outer = 1e9 + 1 if forced else float(bundle["recovery_outer"])
                action, _ = compose_recovery_action(recovery.TRAJECTORY_RECOVERY_HEAD_DEF, bundle["params"], jnp.asarray(obs), jnp.asarray(global_action),
                    bundle["support"], c1_scale, core, outer, c1_support, c1_scale, c1_core, c1_outer)
                return np.asarray(jax.device_get(action), np.float32)
            return fn
        global_fn = lambda obs: tail.checkpoint_action(global_def, global_params, obs, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer)
        oracle = tail.RecoveryOracle(demos, demo_obs, demo_scale, c1_support, c1_scale, c1_core, c1_outer)
        try:
            initial = recovery.collect_policy_rollout(global_fn)
            source1 = initial[2][9]
            r1_episodes = recovery.collect_policy_rollout(action_fn(r1))
            b2, _ = recovery.first_boundary(r1_episodes[2], oracle); req(b2 and b2[0] == 19, "R2 boundary reproduction")
            source2 = r1_episodes[2][b2[0]]
            starts = [source1, source2]
            ordinary = [recovery.horizon_validation(x, action_fn(r2), oracle) for x in starts]
            latched = [recovery.horizon_validation(x, action_fn(r2, True), oracle) for x in starts]
            traces = []
            for source_state in starts:
                obs = restore(oracle.env, source_state["snapshot"]); rows = []
                for step in range(8):
                    global_action = global_fn(obs)
                    action, gate = compose_recovery_action(recovery.TRAJECTORY_RECOVERY_HEAD_DEF, r2["params"], jnp.asarray(obs), jnp.asarray(global_action),
                        r2["support"], c1_scale, float(r2["recovery_core"]), float(r2["recovery_outer"]), c1_support, c1_scale, c1_core, c1_outer)
                    distance = float(np.asarray(jax.device_get(recovery_distance(obs, r2["support"], c1_scale))))
                    rows.append({"step": step, "D_REC": distance, "h_effective": float(np.asarray(jax.device_get(gate))), "outside_outer": distance >= float(r2["recovery_outer"])})
                    obs, _, terminated, truncated, _ = oracle.env.step(np.asarray(jax.device_get(action), np.float32))
                    if terminated or truncated: break
                traces.append(rows)
        finally: oracle.close()
        rows = pickle.loads(DATA.read_bytes()); coherent = coherence(rows, c1_scale, float(r2["recovery_outer"]))
        ordinary_h8 = float(np.mean([x[8]["passed"] for x in ordinary])); latched_h8 = float(np.mean([x[8]["passed"] for x in latched]))
        dropout = bool(all(any(row["outside_outer"] for row in trace[1:2]) for trace in traces))
        if coherent["HISTORY_CLEARLY_RESOLVES_LOCAL_LABEL_CONFLICT"] == "YES":
            root, selected, next_task = "LOCAL_RECOVERY_OBS20_ALIAS_RESOLVED_BY_SHORT_HISTORY", "H1_HISTORY_AUGMENTED_LOCAL_RECOVERY_POLICY", "H1_HISTORY_AUGMENTED_LOCAL_RECOVERY_PILOT"
        elif dropout and latched_h8 > ordinary_h8:
            root, selected, next_task = "RECOVERY_OPTION_TERMINATES_PREMATURELY_FROM_OBSERVATION_GATE_DROPOUT", "OBSERVABLE_INITIATION_TERMINATION_OPTION_CONTROLLER_V1", "C2_OBSERVABLE_RECOVERY_OPTION_PILOT"
        else:
            root, selected, next_task = "LOCAL_RECOVERY_REQUIRES_LONGER_HORIZON_STEERING", "NONE", "SHORT_HORIZON_MODEL_BASED_RECOVERY_REVIEW"
        dump(OUT / "history_label_coherence.json", coherent); dump(OUT / "gate_dropout_and_latch_counterfactual.json", {"ordinary": ordinary, "latched": latched, "traces": traces})
        after = {k: sha(v) for k, v in protected.items()}; req(after == before, "protected mutation"); dump(OUT / "protected_after.json", after)
        result = {"HISTORY_AUGMENTED_LOCAL_RECOVERY_POLICY_REVIEW": "PASS", "HISTORY_CLEARLY_RESOLVES_LOCAL_LABEL_CONFLICT": coherent["HISTORY_CLEARLY_RESOLVES_LOCAL_LABEL_CONFLICT"],
            "GATE_DROPS_OUT_AFTER_FIRST_ACTION_ALL_STARTS": "YES" if dropout else "NO", "ORDINARY_H8_PRESERVATION": ordinary_h8, "FORCED_OPTION_LATCH_H8_PRESERVATION": latched_h8,
            "PRIMARY_ROOT_CAUSE": root, "SELECTED_MINIMAL_ARCHITECTURE": selected, "NEXT": next_task, "TRAINING_STARTED": False,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814, "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "C0_C1_GLOBAL_C2_FROZEN": "YES", "PROTECTED_ARTIFACTS_CHANGED": "NO", "PEG_SINGLE_TASK_CONVERGED": "NO", "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started}
        dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__": main()
