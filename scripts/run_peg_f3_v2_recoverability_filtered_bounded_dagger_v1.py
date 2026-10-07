#!/usr/bin/env python3
"""At most three recoverability-filtered, source-balanced C2 DAgger rounds."""
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

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.audit_peg_f3_v2_c2_closed_loop_recovery_basin as basin
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1 as corrective_run
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config
from training.peg_insertion.gated_absolute_c2_action_head_v1 import ABSOLUTE_C2_HEAD_DEF
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


BASIN_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_closed_loop_recovery_basin_20260908_002"
DATA_ROOT = ROOT / "results/datasets/peg_v2_recoverability_filtered_bounded_dagger_v1_20260908_001"
OUT = ROOT / "results/training/peg_f3_v2_recoverability_filtered_bounded_dagger_v1_20260908_001"
MAX_ROUNDS = 3
EPISODES_PER_ROUND = 3
UPDATES_PER_ROUND = 1000
DAGGER_VALID_MAE_MAX = 0.05


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
        return {str(key): jvalue(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jvalue(item) for item in value]
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


def episode_balanced_indices(rows, allowed_episodes, count, rng):
    by_episode = {episode: [index for index, row in enumerate(rows) if row["sampling_episode"] == episode] for episode in allowed_episodes}
    req(all(by_episode.values()), "nonempty episode-balanced groups")
    chosen_episodes = rng.choice(list(allowed_episodes), size=count, replace=True)
    return np.asarray([rng.choice(by_episode[episode]) for episode in chosen_episodes], np.int64)


def existing_rows(subset):
    return [
        {
            "observations": np.asarray(row["observations"], np.float32),
            "actions": np.asarray(row["actions"], np.float32),
            "sampling_episode": f"demo_{int(row['source_episode'])}",
            "source_episode": int(row["source_episode"]),
            "source_step": int(row["source_step"]),
        }
        for row in subset
    ]


def dagger_row(public, expert_action, round_index):
    return {
        "observations": np.asarray(public["observation"], np.float32),
        "actions": np.asarray(expert_action, np.float32),
        "runtime_obs20": np.asarray(public["observation"], np.float32),
        "frozen_expert_action": np.asarray(expert_action, np.float32),
        "policy_action": np.asarray(public["policy_action"], np.float32),
        "g2": float(public["g2"]),
        "source_episode": int(public["seed"]),
        "source_step": int(public["policy_step"]),
        "sampling_episode": f"r{round_index}_seed_{int(public['seed'])}",
        "round": round_index,
        "EXPERT_SHELL_RECOVERABLE": bool(public["EXPERT_SHELL_RECOVERABLE"]),
        "EXPERT_CORE_RECOVERABLE": bool(public["EXPERT_CORE_RECOVERABLE"]),
        "EXPERT_C0_RECOVERABLE": bool(public["EXPERT_C0_RECOVERABLE"]),
        "EXPERT_SUCCESS_RECOVERABLE": bool(public["EXPERT_SUCCESS_RECOVERABLE"]),
        "EXPERT_CONTACT_HEALTH": bool(public["EXPERT_CONTACT_HEALTH"]),
        "expert_source_row": int(public["expert_source_row"]),
        "expert_source_episode": int(public["expert_source_episode"]),
        "expert_source_step": int(public["expert_source_step"]),
        "pre_contact": not bool(public["contact_before"]),
        "c2_authorized": float(public["g2"]) > review.NUMERICAL_EPS,
    }


def make_round1(demos):
    payload = json.loads((BASIN_RUN / "occupancy_and_full_takeover.json").read_text())
    rows = []
    for public in payload["all_audited_rows"]:
        if public["contact_before"] or public["g2"] <= review.NUMERICAL_EPS or not public["EXPERT_SHELL_RECOVERABLE"]:
            continue
        rows.append(dagger_row(public, demos[int(public["expert_source_row"])]["actions"], 1))
    return rows


def collect_filtered_round(round_index, agent, c1_params, params, support, scale, core, outer, demos, demo_obs, demo_scale):
    episodes = basin.collect_occupancy(agent, c1_params, params, support, scale, core, outer, demo_obs, demo_scale)
    unique, aliases = basin.deduplicate_occupancy(episodes)
    takeover = {key: basin.expert_takeover(row["snapshot"], demos, demo_obs, demo_scale, support, scale, core, outer) for key, row in unique.items()}
    rows = []
    audit = []
    for episode in episodes:
        for state in episode:
            key = aliases[(state["seed"], state["policy_step"])]
            public = {name: value for name, value in state.items() if name != "snapshot"}
            public.update(takeover[key])
            accepted = bool(not public["contact_before"] and public["g2"] > review.NUMERICAL_EPS and public["EXPERT_SHELL_RECOVERABLE"])
            audit.append({"seed": public["seed"], "policy_step": public["policy_step"], "accepted": accepted, "EXPERT_SHELL_RECOVERABLE": public["EXPERT_SHELL_RECOVERABLE"], "EXPERT_SUCCESS_RECOVERABLE": public["EXPERT_SUCCESS_RECOVERABLE"]})
            if accepted:
                rows.append(dagger_row(public, demos[int(public["expert_source_row"])]["actions"], round_index))
    return rows, audit


def save_round_dataset(round_index, rows, audit):
    req(rows, f"round {round_index} has recoverable states")
    path = DATA_ROOT / f"C2_DAGGER_ROUND{round_index}_V1.pkl"
    path.write_bytes(pickle.dumps(rows, pickle.HIGHEST_PROTOCOL))
    manifest = {
        "name": f"C2_DAGGER_ROUND{round_index}_V1",
        "sha256": sha(path),
        "rows": len(rows),
        "episodes": EPISODES_PER_ROUND,
        "train_episode_seeds": [100, 101],
        "heldout_episode_seeds": [102],
        "pre_contact_only": True,
        "expert_shell_recoverable_only": True,
        "c2_authorized_only": True,
        "production_demo_modified": False,
        "audit": audit,
    }
    dump(DATA_ROOT / f"C2_DAGGER_ROUND{round_index}_V1_manifest.json", manifest)
    return path, manifest


def arrays(rows, indices):
    return (
        np.asarray([rows[index]["observations"] for index in indices], np.float32),
        np.asarray([rows[index]["actions"] for index in indices], np.float32),
    )


def train_round(round_index, agent, c1_params, offline_params, existing, aggregate, support, scale, core, outer):
    params = offline_params
    opt_state = review.TX.init(params)
    rng = np.random.default_rng(261000 + round_index)
    existing_train_episodes = sorted({row["sampling_episode"] for row in existing if row["source_episode"] < 9})
    dagger_train_episodes = sorted({row["sampling_episode"] for row in aggregate if row["source_episode"] in (100, 101)})
    existing_valid_indices = [index for index, row in enumerate(existing) if row["source_episode"] >= 9]
    dagger_valid_indices = [index for index, row in enumerate(aggregate) if row["source_episode"] == 102]
    req(existing_valid_indices and dagger_valid_indices, "dual heldout validation")
    history = []
    for update in range(1, UPDATES_PER_ROUND + 1):
        existing_indices = episode_balanced_indices(existing, existing_train_episodes, 32, rng)
        dagger_indices = episode_balanced_indices(aggregate, dagger_train_episodes, 32, rng)
        existing_obs, existing_actions = arrays(existing, existing_indices)
        dagger_obs, dagger_actions = arrays(aggregate, dagger_indices)
        batch_obs = np.concatenate((existing_obs, dagger_obs))
        batch_actions = np.concatenate((existing_actions, dagger_actions))
        params, opt_state, loss, gradient_norm = review.bc_step(params, opt_state, jnp.asarray(batch_obs), jnp.asarray(batch_actions), agent, c1_params, support, scale, core, outer)
        if update in (250, 500, 1000):
            demo_valid_obs, demo_valid_actions = arrays(existing, existing_valid_indices)
            dagger_valid_obs, dagger_valid_actions = arrays(aggregate, dagger_valid_indices)
            history.append({
                "update": update,
                "loss": float(loss),
                "gradient_norm": float(gradient_norm),
                "demo_validation": review.action_metrics(params, demo_valid_obs, demo_valid_actions, agent, c1_params, support, scale, core, outer),
                "dagger_validation": review.action_metrics(params, dagger_valid_obs, dagger_valid_actions, agent, c1_params, support, scale, core, outer),
            })
    checkpoint = OUT / "checkpoints" / f"c2_dagger_round{round_index}_head_1000.msgpack"
    checkpoint.write_bytes(flax.serialization.to_bytes({"c2_head_params": params, "c2_head_opt_state": opt_state}))
    return params, opt_state, history, {"path": str(checkpoint), "sha256": sha(checkpoint)}


def same_c1(reference, candidate):
    keys = ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
    return bool(candidate["contact_health"] and all(candidate[key] == reference[key] for key in keys))


def strong_pass(c2_eval, c1_ok, c0_eval):
    return bool(c1_ok and c0_eval["success_count"] == 3 and c0_eval["contact_health"] and c2_eval["contact_health"] and c2_eval["interface"]["C1_SHELL_ENTRY_COUNT"] > 0)


def main():
    req(Path.cwd() == ROOT and not OUT.exists() and not DATA_ROOT.exists(), "fresh outputs and project cwd")
    req(jax.default_backend() == "gpu", "GPU backend required")
    basin_result = json.loads((BASIN_RUN / "result.json").read_text())
    req(basin_result["RECOVERABILITY_FILTERED_DAGGER_AUTHORIZED"] == "YES" and basin_result["PRIMARY_ROOT_CAUSE"] == "POLICY_EXITS_VALID_EXPERT_RECOVERY_BASIN", "DAgger authorization")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    DATA_ROOT.mkdir(parents=True)
    started = time.time()
    parameter_result = json.loads((review.OUT / "result.json").read_text())
    protected_paths = {
        "production_demo": p4.DEMO,
        "existing_c2_subset": review.EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl",
        "offline_checkpoint": Path(parameter_result["SELECTED_OFFLINE_CHECKPOINT"]["path"]),
        "recovery_probe": BASIN_RUN / "C2_RECOVERY_BASIN_PROBE_V1.pkl",
    }
    protected_before = {name: sha(path) for name, path in protected_paths.items()}
    dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
    subset = pickle.loads((review.EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl").read_bytes())
    existing = existing_rows(subset)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32)
    scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text())
    core = float(preflight["C1_CORE_THRESHOLD"])
    outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    cfg = load_config(2)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"])
        c1_params = source["c1_params"]
        offline_params = corrective_run.load_head(parameter_result["SELECTED_OFFLINE_CHECKPOINT"]["path"])["c2_head_params"]
        base_guard = geo.thash(agent.state.params["modules_actor"])
        c1_guard = geo.thash(c1_params)
        baseline_c1 = review.eval_record(agent, c1_params, offline_params, support, scale, core, outer, "C1", demo_obs, demo_scale)
        aggregate = []
        rounds = []
        selected_params = None
        confirmed = False
        canonical = None
        for round_index in range(1, MAX_ROUNDS + 1):
            if round_index == 1:
                round_rows = make_round1(demos)
                round_audit = [{"seed": row["source_episode"], "policy_step": row["source_step"], "accepted": True, "EXPERT_SHELL_RECOVERABLE": True, "EXPERT_SUCCESS_RECOVERABLE": row["EXPERT_SUCCESS_RECOVERABLE"]} for row in round_rows]
            else:
                req(selected_params is not None, "previous round policy")
                round_rows, round_audit = collect_filtered_round(round_index, agent, c1_params, selected_params, support, scale, core, outer, demos, demo_obs, demo_scale)
            dataset_path, manifest = save_round_dataset(round_index, round_rows, round_audit)
            aggregate.extend(round_rows)
            params, _, history, checkpoint = train_round(round_index, agent, c1_params, offline_params, existing, aggregate, support, scale, core, outer)
            selected_params = params
            c2_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C2", demo_obs, demo_scale)
            c1_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C1", demo_obs, demo_scale)
            c0_eval = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C0", demo_obs, demo_scale)
            c1_ok = same_c1(baseline_c1, c1_eval)
            passed = strong_pass(c2_eval, c1_ok, c0_eval)
            round_report = {
                "round": round_index,
                "dataset": {"path": str(dataset_path), **manifest},
                "virtual_aggregate_rows": len(aggregate),
                "source_sampling": {"existing_expert_probability": 0.5, "aggregated_dagger_probability": 0.5, "within_source": "uniform episode-balanced"},
                "fresh_start_checkpoint": parameter_result["SELECTED_OFFLINE_CHECKPOINT"],
                "fixed_supervised_updates": UPDATES_PER_ROUND,
                "training": history,
                "DEMO_VALID_MAE": history[-1]["demo_validation"]["action_mae"],
                "DAGGER_VALID_MAE": history[-1]["dagger_validation"]["action_mae"],
                "checkpoint": checkpoint,
                "C2": c2_eval,
                "C1": c1_eval,
                "C0": c0_eval,
                "C1_REGRESSION": "PASS" if c1_ok else "FAIL",
                "STRONG_EARLY_PASS": passed,
            }
            rounds.append(round_report)
            dump(OUT / f"dagger_round{round_index}_report.json", round_report)
            if passed:
                matched_c2 = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C2", demo_obs, demo_scale)
                keys = ("success_count", "near_hole_rate", "median_first_divergence", "median_corridor_steps", "median_min_position", "median_approach_reduction", "contact_health")
                reproducible = all(matched_c2[key] == c2_eval[key] for key in keys)
                matched_c1 = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C1", demo_obs, demo_scale)
                matched_c0 = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C0", demo_obs, demo_scale)
                confirmed = bool(reproducible and same_c1(baseline_c1, matched_c1) and matched_c0["success_count"] == 3 and matched_c0["contact_health"] and matched_c2["contact_health"])
                round_report["matched_confirmation"] = {"C2": matched_c2, "C1": matched_c1, "C0": matched_c0, "reproducible": reproducible, "confirmed": confirmed}
                if confirmed:
                    canonical = review.eval_record(agent, c1_params, params, support, scale, core, outer, "C3", demo_obs, demo_scale)
                break

        final = rounds[-1]
        if final["STRONG_EARLY_PASS"] and confirmed:
            acquisition = "PASS"
            first_bad = None
            root = "NONE"
            next_task = "C3_ACQUISITION_PILOT" if canonical["interface"]["C1_SHELL_ENTRY_COUNT"] == 0 else "C3_INTERFACE_RESULT_REVIEW"
        elif final["STRONG_EARLY_PASS"] and not confirmed:
            acquisition = "FAIL"
            first_bad = "C2_MATCHED_CONFIRMATION"
            root = "C2_ACQUISITION_NOT_REPRODUCIBLE"
            next_task = "STOP"
        else:
            acquisition = "FAIL"
            first_bad = "DAGGER_R3_C2_SHELL_ENTRY" if len(rounds) == 3 else f"DAGGER_R{len(rounds)}_DATA"
            if final["DAGGER_VALID_MAE"] > DAGGER_VALID_MAE_MAX:
                root = "C2_ON_POLICY_CAPACITY_FAILURE"
            else:
                final_episodes = basin.collect_occupancy(agent, c1_params, selected_params, support, scale, core, outer, demo_obs, demo_scale)
                final_unique, _ = basin.deduplicate_occupancy(final_episodes)
                final_takeovers = [basin.expert_takeover(row["snapshot"], demos, demo_obs, demo_scale, support, scale, core, outer) for row in final_unique.values()]
                first_loss = next((index for index, takeover in enumerate(final_takeovers) if not takeover["EXPERT_SHELL_RECOVERABLE"]), None)
                root = "C2_RECOVERY_BASIN_TOO_NARROW_FOR_CURRENT_POLICY_CLASS" if first_loss is not None and first_loss < 11 else "C2_LONG_HORIZON_COMPOUNDING_PERSISTS"
            next_task = "STOP"
        req(geo.thash(agent.state.params["modules_actor"]) == base_guard and geo.thash(c1_params) == c1_guard, "protected model mutation")
        final_c2 = final["C2"]
        final_c1 = final["C1"]
        final_c0 = final["C0"]
        result = {
            "C2_RECOVERY_BASIN_REVIEW": "PASS",
            "C2_RESET_EXPERT_RECOVERABLE": basin_result["C2_RESET_EXPERT_RECOVERABLE"],
            "RECOVERABLE_PROBE_FRACTION": basin_result["RECOVERABLE_PROBE_FRACTION"],
            "PRE_DIVERGENCE_RECOVERABLE_FRACTION": basin_result["PRE_DIVERGENCE_RECOVERABLE_FRACTION"],
            "POST_DIVERGENCE_RECOVERABLE_FRACTION": basin_result["POST_DIVERGENCE_RECOVERABLE_FRACTION"],
            "FIRST_POLICY_STEP_NOT_EXPERT_SHELL_RECOVERABLE": basin_result["FIRST_POLICY_STEP_NOT_EXPERT_SHELL_RECOVERABLE"],
            "FIRST_POLICY_STEP_NOT_EXPERT_CORE_RECOVERABLE": basin_result["FIRST_POLICY_STEP_NOT_EXPERT_CORE_RECOVERABLE"],
            "RECOVERY_LOSS_TYPE": basin_result["RECOVERY_LOSS_TYPE"],
            "CORRECTIVE_RECOVERABLE_FRACTION": basin_result["CORRECTIVE_RECOVERABLE_FRACTION"],
            "CORRECTIVE_PRE_RECOVERY_LOSS_FRACTION": basin_result["CORRECTIVE_PRE_RECOVERY_LOSS_FRACTION"],
            "CORRECTIVE_POST_RECOVERY_LOSS_FRACTION": basin_result["CORRECTIVE_POST_RECOVERY_LOSS_FRACTION"],
            "CORRECTIVE_EFFECTIVE_INFLUENCE": basin_result["CORRECTIVE_EFFECTIVE_INFLUENCE"],
            "PRIMARY_C2_RECOVERY_FAILURE_MODE": basin_result["PRIMARY_C2_RECOVERY_FAILURE_MODE"],
            "RECOVERABILITY_FILTERED_DAGGER_USED": "YES",
            "DAGGER_R1": "PASS" if len(rounds) >= 1 else "NOT_RUN",
            "DAGGER_R2": "PASS" if len(rounds) >= 2 else "NOT_RUN",
            "DAGGER_R3": "PASS" if len(rounds) >= 3 else "NOT_RUN",
            "DAGGER_ROUNDS_COMPLETED": len(rounds),
            "C2_DAGGER_ACQUISITION": acquisition,
            "C2_SHELL_ENTRY": final_c2["interface"]["C1_SHELL_ENTRY_COUNT"],
            "C2_CORE_ENTRY": final_c2["interface"]["C1_CORE_ENTRY_COUNT"],
            "C2_NEAR_HOLE": final_c2["near_hole_rate"],
            "C2_SUCCESS": f"{final_c2['success_count']}/3",
            "C2_CONTACT": "PASS" if final_c2["contact_health"] else "FAIL",
            "C1_REGRESSION": final["C1_REGRESSION"],
            "C1_CONTACT": "PASS" if final_c1["contact_health"] else "FAIL",
            "C0_SUCCESS": f"{final_c0['success_count']}/3",
            "C2_MATCHED_CONFIRMATION": "PASS" if confirmed else "NOT_RUN" if not final["STRONG_EARLY_PASS"] else "FAIL",
            "C2_SKILL_CONFIRMED": "YES" if confirmed else "NO",
            "C2_FROZEN_SHA": review.tree_hash(selected_params) if confirmed else None,
            "CANONICAL_INTERFACE_AUDIT": "COMPLETE" if canonical is not None else "NOT_RUN",
            "PEG_SINGLE_TASK_CONVERGED": "NO",
            "FIRST_BAD_GATE": first_bad,
            "PRIMARY_ROOT_CAUSE": root,
            "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "ONLINE_ADMISSION_THRESHOLD": p4.TH,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE",
            "elapsed_seconds": time.time() - started,
        }
        dump(OUT / "all_rounds.json", rounds)
        if canonical is not None:
            dump(OUT / "canonical_interface_audit.json", canonical)
        protected_after = {name: sha(path) for name, path in protected_paths.items()}
        req(protected_after == protected_before, "protected artifact changed")
        dump(OUT / "protected_after.json", protected_after)
        dump(OUT / "result.json", result)
        print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
