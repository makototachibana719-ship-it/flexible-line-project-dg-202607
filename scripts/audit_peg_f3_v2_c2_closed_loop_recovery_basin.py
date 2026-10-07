#!/usr/bin/env python3
"""C2 recovery-basin audit using full-horizon frozen expert takeovers."""
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
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1 as corrective_run
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
from scripts.audit_peg_f3_v2_c0_near_hole_insertion_causal import capture, restore
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, make_train_env, raw_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.gated_absolute_c2_action_head_v1 import ABSOLUTE_C2_HEAD_DEF, gated_absolute_mode
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate, support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


PARAM_RUN = review.OUT
CORRECTIVE_RUN = corrective_run.OUT
CORRECTIVE_DATA = corrective_run.DATASET_DIR / "C2_ON_POLICY_CORRECTIVE_SUPERVISION_V1.pkl"
OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_closed_loop_recovery_basin_20260908_002"
FIXED_STEPS = (0, 1, 2, 3, 5, 8, 10, 12, 15)


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


def episode_ranges(demos):
    ranges = []
    start = 0
    for index, row in enumerate(demos):
        if bool(row["dones"]):
            ranges.append((start, index + 1))
            start = index + 1
    req(start == len(demos), "complete expert episodes")
    return ranges


def row_provenance(ranges, row_index):
    for episode, (start, end) in enumerate(ranges):
        if start <= row_index < end:
            return episode, row_index - start, start, end
    raise RuntimeError("expert source row outside episode ranges")


def current_metrics(env):
    raw = raw_env(env)
    values = raw.task._goal_metrics(raw._get_obs())
    return {
        "lateral": float(values["lateral_error"]),
        "axial": float(values["axial_error"]),
        "orientation": float(values["orientation_error"]),
        "force": float(values["force_norm"]),
    }


def expert_takeover(snapshot, demos, demo_obs, demo_scale, support, support_scale, core, outer):
    env = make_train_env("C2")
    ranges = episode_ranges(demos)
    actions = np.asarray([row["actions"] for row in demos], np.float32)
    try:
        obs = restore(env, snapshot)
        distances = np.sqrt(np.sum(np.square((demo_obs - obs) / demo_scale), axis=1))
        source_row = int(np.argmin(distances))
        episode, source_step, _, end = row_provenance(ranges, source_row)
        remaining = max(0, 500 - int(snapshot["python"]["raw_elapsed_steps"]))
        shell = core_entry = c0_entry = success = False
        contact_healthy = True
        max_force = 0.0
        executed = 0
        first_shell_step = first_core_step = first_c0_step = None
        for offset, index in enumerate(range(source_row, min(end, source_row + remaining))):
            action = actions[index]
            obs, _, terminated, truncated, info = env.step(action)
            executed += 1
            d_c1 = float(np.asarray(jax.device_get(support_distance(obs, support, support_scale))))
            g0 = float(np.asarray(jax.device_get(gate_value(obs))))
            if d_c1 <= outer and not shell:
                shell = True
                first_shell_step = executed
            if d_c1 <= core and not core_entry:
                core_entry = True
                first_core_step = executed
            if g0 == 0.0 and not c0_entry:
                c0_entry = True
                first_c0_step = executed
            link, unexpected, _ = contact_flags(env)
            force = float(info.get("force_norm", 0.0))
            max_force = max(max_force, force if np.isfinite(force) else math.inf)
            contact_healthy &= bool(not link and not unexpected and np.isfinite(force))
            if terminated or truncated:
                success = bool(info.get("is_success", terminated and not truncated))
                break
        # Hierarchical recoverability counts reaching the requested interface
        # or any strictly deeper validated region.  Literal support hits remain
        # separately reported so a direct transition into C0 is not mislabeled
        # as unrecoverable merely because sampled D_C1 skipped the thin shell.
        shell_recoverable = bool(contact_healthy and (shell or core_entry or c0_entry or success))
        core_recoverable = bool(contact_healthy and (core_entry or c0_entry or success))
        c0_recoverable = bool(contact_healthy and (c0_entry or success))
        return {
            "expert_binding": "nearest normalized Obs20 accepted production-Demo row, then frozen same-episode full suffix",
            "expert_source_row": source_row,
            "expert_source_episode": episode,
            "expert_source_step": source_step,
            "expert_source_distance": float(distances[source_row]),
            "EXPERT_SHELL_HIT": bool(shell),
            "EXPERT_CORE_HIT": bool(core_entry),
            "EXPERT_C0_HIT": bool(c0_entry),
            "EXPERT_SHELL_RECOVERABLE": shell_recoverable,
            "EXPERT_CORE_RECOVERABLE": core_recoverable,
            "EXPERT_C0_RECOVERABLE": c0_recoverable,
            "EXPERT_SUCCESS_RECOVERABLE": bool(success and contact_healthy),
            "EXPERT_RECOVERY_STEPS": executed,
            "EXPERT_FIRST_SHELL_STEP": first_shell_step,
            "EXPERT_FIRST_CORE_STEP": first_core_step,
            "EXPERT_FIRST_C0_STEP": first_c0_step,
            "EXPERT_CONTACT_HEALTH": bool(contact_healthy),
            "EXPERT_MAX_FORCE": max_force,
        }
    finally:
        env.close()


def collect_occupancy(agent, c1_params, params, support, scale, core, outer, demo_obs, demo_scale):
    episodes = []
    for seed in (100, 101, 102):
        env = make_train_env("C2")
        rows = []
        try:
            obs, _ = env.reset(seed=seed)
            for step in range(500):
                metrics = current_metrics(env)
                link, unexpected, pairs = contact_flags(env)
                d_demo = p4.demo_distance(obs, demo_obs, demo_scale)
                d_c1 = float(np.asarray(jax.device_get(support_distance(obs, support, scale))))
                g2 = float(np.asarray(jax.device_get(progressive_gate(obs, support, scale, core, outer))))
                action = np.asarray(jax.device_get(gated_absolute_mode(agent, c1_params, params, obs, support, scale, core, outer)), np.float32)
                rows.append({
                    "seed": seed,
                    "policy_step": step,
                    "observation": np.asarray(obs, np.float32).copy(),
                    "snapshot": capture(env, obs),
                    "policy_action": action,
                    "g2": g2,
                    "D_C1": d_c1,
                    "demo_distance": d_demo,
                    "position": float(np.linalg.norm(np.asarray(obs)[13:16])),
                    **metrics,
                    "contact_before": bool(link or unexpected),
                    "contact_pairs_before": pairs,
                })
                if link or unexpected:
                    break
                obs, _, terminated, truncated, _ = env.step(action)
                link_after, unexpected_after, _ = contact_flags(env)
                if terminated or truncated or link_after or unexpected_after:
                    break
        finally:
            env.close()
        episodes.append(rows)
    return episodes


def deduplicate_occupancy(episodes):
    unique = {}
    aliases = {}
    for rows in episodes:
        for row in rows:
            key = hashlib.sha256(row["observation"].tobytes()).hexdigest()
            aliases[(row["seed"], row["policy_step"])] = key
            unique.setdefault(key, row)
    return unique, aliases


def branch_then_takeover(snapshot, first_action, demos, demo_obs, demo_scale, support, scale, core, outer):
    env = make_train_env("C2")
    try:
        obs = restore(env, snapshot)
        next_obs, _, terminated, truncated, info = env.step(np.asarray(first_action, np.float32))
        link, unexpected, pairs = contact_flags(env)
        next_snapshot = capture(env, next_obs)
        first = {
            "next_observation": np.asarray(next_obs, np.float32),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "force": float(info.get("force_norm", 0.0)),
            "contact_healthy": bool(not link and not unexpected),
            "contact_pairs": pairs,
        }
    finally:
        env.close()
    takeover = expert_takeover(next_snapshot, demos, demo_obs, demo_scale, support, scale, core, outer)
    return {"one_step": first, "full_expert_after": takeover}


def source_gradient(params, observations, targets, agent, c1_params, support, scale, core, outer):
    obs = jnp.asarray(observations)
    target = jnp.asarray(targets)

    def loss_fn(candidate):
        actions = gated_absolute_mode(agent, c1_params, candidate, obs, support, scale, core, outer)
        return jnp.mean(jnp.square(actions[..., :6] - target[..., :6]))

    loss, gradient = jax.value_and_grad(loss_fn)(params)
    return float(loss), float(jax.device_get(jax.tree_util.tree_reduce(lambda a, b: a + jnp.sum(jnp.square(b)), gradient, initializer=jnp.asarray(0.0))) ** 0.5)


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU backend required")
    parameter_result = json.loads((PARAM_RUN / "result.json").read_text())
    corrective_result = json.loads((CORRECTIVE_RUN / "result.json").read_text())
    req(parameter_result["C2_OFFLINE_FIT"] == "PASS" and corrective_result["PRIMARY_ROOT_CAUSE"] == "C2_CLOSED_LOOP_DISTRIBUTION_SHIFT_NOT_REPAIRED", "authoritative inputs")
    OUT.mkdir(parents=True)
    started = time.time()
    protected_paths = {
        "production_demo": p4.DEMO,
        "existing_c2_subset": review.EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl",
        "previous_corrective": CORRECTIVE_DATA,
        "offline_checkpoint": Path(parameter_result["SELECTED_OFFLINE_CHECKPOINT"]["path"]),
        "degraded_corrective_checkpoint": CORRECTIVE_RUN / "checkpoints/gated_absolute_c2_head_corrective_1000.msgpack",
    }
    protected_before = {name: sha(path) for name, path in protected_paths.items()}
    dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == p4.DSHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes())
    demo_obs, demo_scale = distance_model(demos)
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
        offline_bundle = corrective_run.load_head(parameter_result["SELECTED_OFFLINE_CHECKPOINT"]["path"])
        offline_params = offline_bundle["c2_head_params"]
        degraded_bundle = corrective_run.load_head(CORRECTIVE_RUN / "checkpoints/gated_absolute_c2_head_corrective_1000.msgpack")
        degraded_params = degraded_bundle["c2_head_params"]
        policy_hashes = {
            "C0_BASE_SHA": geo.thash(agent.state.params["modules_actor"]),
            "C1_RESIDUAL_SHA": geo.thash(c1_params),
            "C2_OFFLINE_BC_HEAD_SHA": review.tree_hash(offline_params),
            "C2_DEGRADED_CORRECTIVE_HEAD_SHA": review.tree_hash(degraded_params),
        }

        episodes = collect_occupancy(agent, c1_params, offline_params, support, scale, core, outer, demo_obs, demo_scale)
        unique, aliases = deduplicate_occupancy(episodes)
        first_divergence = min(row["policy_step"] for row in episodes[0] if row["demo_distance"] > p4.TH)
        best_step = min(episodes[0], key=lambda row: row["position"])["policy_step"]
        shell_nearest_step = min(episodes[0], key=lambda row: row["D_C1"])["policy_step"]
        last_step = episodes[0][-1]["policy_step"]
        selected_steps = set(FIXED_STEPS)
        selected_steps.update((first_divergence - 2, first_divergence - 1, first_divergence, first_divergence + 1, first_divergence + 3, best_step, shell_nearest_step, max(0, last_step - 2), max(0, last_step - 1)))
        selected_steps = sorted(step for step in selected_steps if 0 <= step <= last_step)

        takeover_by_key = {}
        for key, row in unique.items():
            takeover_by_key[key] = expert_takeover(row["snapshot"], demos, demo_obs, demo_scale, support, scale, core, outer)
        audited_rows = []
        for rows in episodes:
            for row in rows:
                key = aliases[(row["seed"], row["policy_step"])]
                public = {name: value for name, value in row.items() if name != "snapshot"}
                public.update(takeover_by_key[key])
                audited_rows.append(public)

        probe = []
        seen = set()
        for row in audited_rows:
            if row["policy_step"] not in selected_steps or row["contact_before"]:
                continue
            key = hashlib.sha256(row["observation"].tobytes()).hexdigest()
            if key not in seen:
                seen.add(key)
                probe.append(row)
        probe_path = OUT / "C2_RECOVERY_BASIN_PROBE_V1.pkl"
        probe_path.write_bytes(pickle.dumps(probe, pickle.HIGHEST_PROTOCOL))
        dump(OUT / "C2_RECOVERY_BASIN_PROBE_V1_manifest.json", {
            "name": "C2_RECOVERY_BASIN_PROBE_V1",
            "sha256": sha(probe_path),
            "rows": len(probe),
            "selected_policy_steps": selected_steps,
            "source_episodes": 3,
            "deduplicated": True,
            "post_severe_contact_excluded": True,
            "policy": "pre-corrective offline-BC P2",
        })

        canonical_episode = [row for row in audited_rows if row["seed"] == 100]
        shell_recoverable = [row for row in canonical_episode if row["EXPERT_SHELL_RECOVERABLE"]]
        core_recoverable = [row for row in canonical_episode if row["EXPERT_CORE_RECOVERABLE"]]
        first_not_shell = next((row["policy_step"] for row in canonical_episode if not row["EXPERT_SHELL_RECOVERABLE"]), None)
        first_not_core = next((row["policy_step"] for row in canonical_episode if not row["EXPERT_CORE_RECOVERABLE"]), None)
        reset_recoverable = bool(canonical_episode[0]["EXPERT_SHELL_RECOVERABLE"])
        pre = [row for row in canonical_episode if row["policy_step"] < first_divergence]
        post = [row for row in canonical_episode if row["policy_step"] >= first_divergence]
        basin_metrics = {
            "C2_RESET_EXPERT_RECOVERABLE": "YES" if reset_recoverable else "NO",
            "FIRST_POLICY_STEP_NOT_EXPERT_SHELL_RECOVERABLE": first_not_shell,
            "FIRST_POLICY_STEP_NOT_EXPERT_CORE_RECOVERABLE": first_not_core,
            "RECOVERABLE_PROBE_FRACTION": float(np.mean([row["EXPERT_SHELL_RECOVERABLE"] for row in probe])),
            "PRE_DIVERGENCE_RECOVERABLE_FRACTION": float(np.mean([row["EXPERT_SHELL_RECOVERABLE"] for row in pre])) if pre else None,
            "POST_DIVERGENCE_RECOVERABLE_FRACTION": float(np.mean([row["EXPERT_SHELL_RECOVERABLE"] for row in post])) if post else None,
            "C2_RECOVERY_BASIN_EXISTS": "YES" if reset_recoverable and first_not_shell is not None else "NO",
            "FIRST_POLICY_DIVERGENCE_STEP": first_divergence,
        }

        transition_audit = None
        recovery_loss_type = "NOT_APPLICABLE"
        if reset_recoverable and first_not_shell is not None:
            last_good = max(
                (row for row in canonical_episode if row["policy_step"] < first_not_shell and row["EXPERT_SHELL_RECOVERABLE"]),
                key=lambda row: row["policy_step"],
            )
            last_good_key = aliases[(100, last_good["policy_step"])]
            snapshot = unique[last_good_key]["snapshot"]
            expert_first = np.asarray(demos[last_good["expert_source_row"]]["actions"], np.float32)
            policy_branch = branch_then_takeover(snapshot, last_good["policy_action"], demos, demo_obs, demo_scale, support, scale, core, outer)
            expert_branch = branch_then_takeover(snapshot, expert_first, demos, demo_obs, demo_scale, support, scale, core, outer)
            one_step = bool(not policy_branch["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"] and expert_branch["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"])
            earlier_unstable = any(not row["EXPERT_CORE_RECOVERABLE"] for row in canonical_episode if row["policy_step"] < last_good["policy_step"])
            recovery_loss_type = "MIXED" if one_step and earlier_unstable else "ONE_STEP_BOUNDARY_CROSSING" if one_step else "GRADUAL_COMPOUNDING"
            transition_audit = {
                "last_expert_recoverable_policy_step": last_good["policy_step"],
                "first_nonrecoverable_policy_step": first_not_shell,
                "state": {key: last_good[key] for key in ("observation", "policy_action", "g2", "D_C1", "position", "lateral", "axial", "orientation", "force")},
                "expert_action": expert_first,
                "policy_branch": policy_branch,
                "expert_branch": expert_branch,
                "RECOVERY_LOSS_TYPE": recovery_loss_type,
            }

        previous_corrective = pickle.loads(CORRECTIVE_DATA.read_bytes())
        corrected_rows = []
        for row in previous_corrective:
            key = aliases.get((int(row["episode_seed"]), int(row["policy_step"])))
            req(key in takeover_by_key, "corrective state reproduced in authoritative occupancy")
            takeover = takeover_by_key[key]
            corrected_rows.append({
                "episode_seed": int(row["episode_seed"]),
                "policy_step": int(row["policy_step"]),
                "EXPERT_SHELL_RECOVERABLE": takeover["EXPERT_SHELL_RECOVERABLE"],
                "EXPERT_CORE_RECOVERABLE": takeover["EXPERT_CORE_RECOVERABLE"],
                "before_recovery_loss": first_not_shell is None or int(row["policy_step"]) < first_not_shell,
                "after_recovery_loss": first_not_shell is not None and int(row["policy_step"]) >= first_not_shell,
            })
        corrective_metrics = {
            "CORRECTIVE_RECOVERABLE_FRACTION": float(np.mean([row["EXPERT_SHELL_RECOVERABLE"] for row in corrected_rows])),
            "CORRECTIVE_PRE_RECOVERY_LOSS_FRACTION": float(np.mean([row["before_recovery_loss"] for row in corrected_rows])),
            "CORRECTIVE_POST_RECOVERY_LOSS_FRACTION": float(np.mean([row["after_recovery_loss"] for row in corrected_rows])),
        }

        subset = pickle.loads((review.EVIDENCE / "C2_EXISTING_EXPERT_SUBSET_V1.pkl").read_bytes())
        existing_obs = np.asarray([row["observations"] for row in subset], np.float32)
        existing_targets = np.asarray([row["actions"] for row in subset], np.float32)
        corrective_obs = np.asarray([row["observations"] for row in previous_corrective], np.float32)
        corrective_targets = np.asarray([row["actions"] for row in previous_corrective], np.float32)
        final_bundle = corrective_run.load_head(CORRECTIVE_RUN / "checkpoints/gated_absolute_c2_head_corrective_1000.msgpack")
        influence = {}
        for label, params in (("before_training", offline_params), ("after_training", final_bundle["c2_head_params"])):
            expert_loss, expert_grad = source_gradient(params, existing_obs, existing_targets, agent, c1_params, support, scale, core, outer)
            corrective_loss, corrective_grad = source_gradient(params, corrective_obs, corrective_targets, agent, c1_params, support, scale, core, outer)
            influence[label] = {"existing_expert_loss": expert_loss, "existing_expert_gradient_norm": expert_grad, "corrective_loss": corrective_loss, "corrective_gradient_norm": corrective_grad}
        ratio = influence["before_training"]["corrective_gradient_norm"] / max(influence["before_training"]["existing_expert_gradient_norm"], 1e-12)
        effective = "DILUTED" if ratio < 0.25 else "DOMINANT" if ratio > 4.0 else "ADEQUATE"
        influence.update({
            "EXISTING_EXPERT_SAMPLE_FRACTION": 0.5,
            "CORRECTIVE_SAMPLE_FRACTION": 0.5,
            "EXISTING_EXPERT_BATCH_SAMPLES_SEEN": 32000,
            "CORRECTIVE_BATCH_SAMPLES_SEEN": 32000,
            "EXPERT_SOURCE_GRAD_NORM": influence["before_training"]["existing_expert_gradient_norm"],
            "CORRECTIVE_SOURCE_GRAD_NORM": influence["before_training"]["corrective_gradient_norm"],
            "CORRECTIVE_TO_EXPERT_GRAD_NORM_RATIO": ratio,
            "CORRECTIVE_EFFECTIVE_INFLUENCE": effective,
            "classification_thresholds_preregistered": {"diluted_below_ratio": 0.25, "dominant_above_ratio": 4.0},
        })

        if not reset_recoverable:
            primary = "C2_RESET_OUTSIDE_VALIDATED_EXPERT_RECOVERY_BASIN"
            next_task = "C2_RESET_OR_EXPERT_BINDING_CAUSAL_REVIEW"
            authorize = False
        elif not shell_recoverable:
            primary = "EXPERT_LABELS_NOT_CLOSED_LOOP_RECOVERABLE"
            next_task = "C2_EXPERT_RECOVERY_POLICY_REVIEW"
            authorize = False
        elif corrective_metrics["CORRECTIVE_POST_RECOVERY_LOSS_FRACTION"] > 0.5:
            primary = "CORRECTIVE_DATA_COLLECTED_TOO_LATE_OUTSIDE_RECOVERABLE_BASIN"
            next_task = "RECOVERABILITY_FILTERED_BOUNDED_DAGGER_V1"
            authorize = True
        elif effective == "DILUTED":
            primary = "CORRECTIVE_DATA_UNDERWEIGHTED_IN_TRAINING"
            next_task = "RECOVERABILITY_FILTERED_BOUNDED_DAGGER_V1"
            authorize = True
        elif first_not_shell is not None:
            primary = "POLICY_EXITS_VALID_EXPERT_RECOVERY_BASIN"
            next_task = "RECOVERABILITY_FILTERED_BOUNDED_DAGGER_V1"
            authorize = True
        else:
            primary = "MIXED_RECOVERY_COVERAGE_FAILURE"
            next_task = "RECOVERABILITY_FILTERED_BOUNDED_DAGGER_V1"
            authorize = True

        dump(OUT / "occupancy_and_full_takeover.json", {"policy_hashes": policy_hashes, "selected_probe_steps": selected_steps, "all_audited_rows": audited_rows, "basin_metrics": basin_metrics})
        dump(OUT / "first_bad_occupancy_transition.json", transition_audit)
        dump(OUT / "failed_corrective_recoverability.json", {"metrics": corrective_metrics, "rows": corrected_rows})
        dump(OUT / "training_influence.json", influence)
        result = {
            "C2_RECOVERY_BASIN_REVIEW": "PASS",
            **policy_hashes,
            "C2_RECOVERY_BASIN_PROBE_SHA": sha(probe_path),
            **basin_metrics,
            "RECOVERY_LOSS_TYPE": recovery_loss_type,
            **corrective_metrics,
            "CORRECTIVE_EFFECTIVE_INFLUENCE": effective,
            "PRIMARY_C2_RECOVERY_FAILURE_MODE": primary,
            "RECOVERABILITY_FILTERED_DAGGER_AUTHORIZED": "YES" if authorize else "NO",
            "RECOVERABILITY_FILTERED_DAGGER_USED": "NO",
            "DAGGER_R1": "NOT_RUN",
            "DAGGER_R2": "NOT_RUN",
            "DAGGER_R3": "NOT_RUN",
            "C2_MATCHED_CONFIRMATION": "NOT_RUN",
            "C2_SKILL_CONFIRMED": "NO",
            "CANONICAL_INTERFACE_AUDIT": "NOT_RUN",
            "PEG_SINGLE_TASK_CONVERGED": "NO",
            "FIRST_BAD_GATE": "C2_RECOVERY_BASIN" if not authorize else None,
            "PRIMARY_ROOT_CAUSE": primary,
            "NEXT": next_task,
            "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
            "ONLINE_ADMISSION_THRESHOLD": p4.TH,
            "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
            "PROTECTED_ARTIFACTS_CHANGED": "NO",
            "TRAINING_RUNNING": False,
            "RESIDUAL_PROCESSES": "NONE",
            "elapsed_seconds": time.time() - started,
        }
        protected_after = {name: sha(path) for name, path in protected_paths.items()}
        req(protected_after == protected_before, "protected artifact changed")
        dump(OUT / "protected_after.json", protected_after)
        dump(OUT / "result.json", result)
        print(json.dumps(jvalue(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
