#!/usr/bin/env python3
"""C2 multi-boundary recoverability tail-risk audit; no training."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
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

import scripts.audit_peg_f3_v2_actor_critic_closed_loop_geometry as geo
import scripts.audit_peg_f3_v2_c2_closed_loop_recovery_basin as basin
import scripts.run_peg_f3_v2_actor_update_timescale_v1 as p4
import scripts.run_peg_f3_v2_c2_on_policy_corrective_supervision_v1 as corrective
import scripts.run_peg_f3_v2_c2_policy_class_capacity_observability_review as pc
import scripts.run_peg_f3_v2_c2_residual_output_parameterization_review as review
import scripts.run_peg_f3_v2_c2_supervised_convergence_closure as convergence
import scripts.run_peg_f3_v2_c2_supervised_optimization_review as optimization
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import distance_model, load_config, make_train_env, raw_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.gated_absolute_c2_action_head_v1 import gated_absolute_mode, lower_policy_components
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate, support_distance
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


OUT = ROOT / "results/diagnostics/peg/f3_v2_c2_recovery_boundary_tail_risk_review_20260909_001"
DATASET_PATH = OUT / "C2_MULTI_BOUNDARY_AUDIT_SET_V1.pkl"
CONVERGENCE_RUN = convergence.OUT
DAGGER_RUN = ROOT / "results/training/peg_f3_v2_recoverability_filtered_bounded_dagger_v1_20260908_001"
DAGGER_DATA = ROOT / "results/datasets/peg_v2_recoverability_filtered_bounded_dagger_v1_20260908_001"
BASIN_RUN = ROOT / "results/diagnostics/peg/f3_v2_c2_closed_loop_recovery_basin_20260908_002"
AUDIT_DATA = pc.AUDIT_DATA
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
PERTURBATIONS = (0.0025, 0.005, 0.01, 0.02, 0.05)
CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
GRADIENT_STEPS = (0, 500, 1000, 2000, 3000)
BOUNDARY_TARGET = 16  # preregistered minimum and fastest valid multi-state set
SHELL_PASS = 0.90
CORE_PASS = 0.80
GLOBAL_DAGGER_CATASTROPHIC_MAX = 0.06
MATERIALLY_SMALL_BATCH_FRACTION = 0.10
TAIL_CONCENTRATION_MIN = 0.50
EPS = 1e-12


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


def obs_hash(obs):
    return hashlib.sha256(np.asarray(obs, np.float32).tobytes()).hexdigest()


def snapshot_hash(snapshot):
    digest = hashlib.sha256()
    digest.update(np.asarray(snapshot["mujoco"], np.float64).tobytes())
    digest.update(np.asarray(snapshot["observation"], np.float32).tobytes())
    digest.update(pickle.dumps(snapshot["python"], pickle.HIGHEST_PROTOCOL))
    return digest.hexdigest()


def tree_dot(a, b):
    return float(sum(np.vdot(np.asarray(x), np.asarray(y)).real for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))))


def tree_norm(tree):
    return math.sqrt(max(tree_dot(tree, tree), 0.0))


def tree_cosine(a, b):
    return tree_dot(a, b) / max(tree_norm(a) * tree_norm(b), EPS)


def layer_cosines(a, b):
    return {key: tree_cosine(a[key], b[key]) for key in sorted(a)}


def load_head_checkpoint(path, key="params"):
    definition, params, tx, state = convergence.init_model()
    template = {"params": params, "opt_state": state}
    loaded = flax.serialization.from_bytes(template, Path(path).read_bytes())
    return definition, loaded[key]


def load_dagger_head(path):
    bundle = corrective.load_head(path)
    return bundle["c2_head_params"]


class RecoveryOracle:
    """Reusable exact-state oracle; avoids recompiling a MuJoCo model per branch."""

    def __init__(self, demos, demo_obs, demo_scale, support, scale, core, outer):
        self.env = make_train_env("C2")
        self.demos = demos
        self.demo_obs = demo_obs
        self.demo_scale = demo_scale
        self.support = support
        self.scale = scale
        self.core = core
        self.outer = outer
        self.ranges = basin.episode_ranges(demos)
        self.actions = np.asarray([row["actions"] for row in demos], np.float32)

    def close(self):
        self.env.close()

    def _takeover_current(self, obs):
        distances = np.sqrt(np.sum(np.square((self.demo_obs - obs) / self.demo_scale), axis=1))
        source_row = int(np.argmin(distances))
        episode, source_step, _, end = basin.row_provenance(self.ranges, source_row)
        elapsed = int(raw_env(self.env)._elapsed_steps)
        remaining = max(0, 500 - elapsed)
        shell = core_entry = c0_entry = success = False
        healthy, max_force, executed = True, 0.0, 0
        first_shell = first_core = first_c0 = None
        for index in range(source_row, min(end, source_row + remaining)):
            obs, _, terminated, truncated, info = self.env.step(self.actions[index])
            executed += 1
            d_c1 = float(np.asarray(jax.device_get(support_distance(obs, self.support, self.scale))))
            g0 = float(np.asarray(jax.device_get(gate_value(obs))))
            if d_c1 <= self.outer and not shell: shell, first_shell = True, executed
            if d_c1 <= self.core and not core_entry: core_entry, first_core = True, executed
            if g0 == 0.0 and not c0_entry: c0_entry, first_c0 = True, executed
            link, unexpected, _ = contact_flags(self.env)
            force = float(info.get("force_norm", 0.0))
            max_force = max(max_force, force if np.isfinite(force) else math.inf)
            healthy &= bool(not link and not unexpected and np.isfinite(force))
            if terminated or truncated:
                success = bool(info.get("is_success", terminated and not truncated))
                break
        return {
            "expert_source_row": source_row, "expert_source_episode": episode, "expert_source_step": source_step,
            "expert_source_distance": float(distances[source_row]), "EXPERT_SHELL_HIT": shell,
            "EXPERT_CORE_HIT": core_entry, "EXPERT_C0_HIT": c0_entry,
            "EXPERT_SHELL_RECOVERABLE": bool(healthy and (shell or core_entry or c0_entry or success)),
            "EXPERT_CORE_RECOVERABLE": bool(healthy and (core_entry or c0_entry or success)),
            "EXPERT_C0_RECOVERABLE": bool(healthy and (c0_entry or success)),
            "EXPERT_SUCCESS_RECOVERABLE": bool(healthy and success), "EXPERT_RECOVERY_STEPS": executed,
            "EXPERT_FIRST_SHELL_STEP": first_shell, "EXPERT_FIRST_CORE_STEP": first_core,
            "EXPERT_FIRST_C0_STEP": first_c0, "EXPERT_CONTACT_HEALTH": bool(healthy),
            "EXPERT_MAX_FORCE": max_force,
        }

    def takeover(self, snapshot):
        obs = basin.restore(self.env, snapshot)
        return self._takeover_current(obs)

    def action_then_takeover(self, snapshot, action):
        obs = basin.restore(self.env, snapshot)
        next_obs, _, terminated, truncated, info = self.env.step(np.asarray(action, np.float32))
        link, unexpected, pairs = contact_flags(self.env)
        first_healthy = bool(not link and not unexpected and np.isfinite(float(info.get("force_norm", 0.0))))
        if terminated or truncated:
            success = bool(info.get("is_success", terminated and not truncated))
            takeover = {
                "EXPERT_SHELL_RECOVERABLE": bool(first_healthy and success),
                "EXPERT_CORE_RECOVERABLE": bool(first_healthy and success),
                "EXPERT_SUCCESS_RECOVERABLE": bool(first_healthy and success),
                "EXPERT_CONTACT_HEALTH": first_healthy, "EXPERT_RECOVERY_STEPS": 0,
            }
        else:
            takeover = self._takeover_current(next_obs)
        return {
            "one_step": {"next_observation": np.asarray(next_obs, np.float32), "terminated": bool(terminated),
                         "truncated": bool(truncated), "force": float(info.get("force_norm", 0.0)),
                         "contact_healthy": first_healthy, "contact_pairs": pairs},
            "full_expert_after": takeover,
        }


def public_state(row, categories, source_policy):
    return {
        "source_policy": source_policy, "seed": int(row["seed"]), "policy_step": int(row["policy_step"]),
        "observation": np.asarray(row["observation"], np.float32), "policy_action": np.asarray(row["policy_action"], np.float32),
        "g2": float(row["g2"]), "D_C1": float(row["D_C1"]), "demo_distance": float(row["demo_distance"]),
        "position": float(row["position"]), "contact_before": bool(row["contact_before"]),
        "categories": sorted(set(categories)), "obs20_sha": obs_hash(row["observation"]),
        "exact_state_sha": snapshot_hash(row["snapshot"]),
    }


def build_source_replays(agent, c1_params, policies, support, scale, core, outer, demo_obs, demo_scale):
    return {label: basin.collect_occupancy(agent, c1_params, params, support, scale, core, outer, demo_obs, demo_scale) for label, params in policies.items()}


def replay_index(episodes):
    return {(int(row["seed"]), int(row["policy_step"])): row for episode in episodes for row in episode}


def verify_existing_rows(source_label, episodes, dataset_path):
    replay_hashes = {obs_hash(row["observation"]) for episode in episodes for row in episode}
    rows = pickle.loads(Path(dataset_path).read_bytes())
    missing = [obs_hash(row["observations"]) for row in rows if obs_hash(row["observations"]) not in replay_hashes]
    req(not missing, f"{source_label} deterministic replay must exactly cover existing DAgger rows")
    return {"dataset": str(dataset_path), "rows": len(rows), "exact_obs20_matches": len(rows), "missing": 0}


def audit_map_from_json():
    payload = json.loads((BASIN_RUN / "occupancy_and_full_takeover.json").read_text())
    return {(int(r["seed"]), int(r["policy_step"])): bool(r["EXPERT_SHELL_RECOVERABLE"]) for r in payload["all_audited_rows"]}


def audit_map_from_manifest(round_index):
    payload = json.loads((DAGGER_DATA / f"C2_DAGGER_ROUND{round_index}_V1_manifest.json").read_text())
    return {(int(r["seed"]), int(r["policy_step"])): bool(r["EXPERT_SHELL_RECOVERABLE"]) for r in payload["audit"]}


def candidate_pool(replays, known_maps):
    candidates = {}

    def add(label, seed, step, category):
        row = replay_index(replays[label]).get((seed, step))
        if row is None or row["contact_before"]:
            return
        key = (obs_hash(row["observation"]), snapshot_hash(row["snapshot"]))
        if key not in candidates:
            candidates[key] = {"row": row, "source_policy": label, "categories": set()}
        candidates[key]["categories"].add(category)

    for step in (0, 3, 5): add("offline", 100, step, "A_RESET_EARLY")
    for step in range(7, 13): add("offline", 100, step, "B_HISTORICAL_7_12")
    for label, amap in known_maps.items():
        by_seed = defaultdict(list)
        for (seed, step), recoverable in amap.items(): by_seed[seed].append((step, recoverable))
        for seed, seq in by_seed.items():
            seq.sort()
            first_bad = next((step for step, ok in seq if not ok), None)
            if first_bad is not None:
                for step in range(max(0, first_bad - 3), first_bad):
                    add(label, seed, step, "C_LAST3_BEFORE_LOSS")
                add(label, seed, first_bad - 1, "D_NEAREST_UNRECOVERABLE_SUCCESSOR")
            recoverable_rows = [replay_index(replays[label]).get((seed, step)) for step, ok in seq if ok]
            recoverable_rows = [row for row in recoverable_rows if row is not None and not row["contact_before"]]
            for row in sorted(recoverable_rows, key=lambda x: x["D_C1"])[:2]:
                add(label, seed, int(row["policy_step"]), "E_CLOSEST_C1_SHELL")
            if first_bad is not None and recoverable_rows:
                last = max((row for row in recoverable_rows if row["policy_step"] < first_bad), key=lambda x: x["policy_step"], default=None)
                if last is not None: add(label, seed, int(last["policy_step"]), "F_PRE_DOWNSTREAM_CONTACT_RECOVERABLE")

    for label in replays:
        rows = [r for ep in replays[label] for r in ep if not r["contact_before"]]
        for row in sorted(rows, key=lambda x: (x["D_C1"], -x["policy_step"]))[:8]:
            add(label, int(row["seed"]), int(row["policy_step"]), "E_CLOSEST_C1_SHELL")
        for row in rows[-6:]:
            add(label, int(row["seed"]), int(row["policy_step"]), "F_PRE_DOWNSTREAM_CONTACT_RECOVERABLE")
    return list(candidates.values())


def select_authoritative(pool, oracle):
    audited = []
    for candidate in pool:
        takeover = oracle.takeover(candidate["row"]["snapshot"])
        audited.append({**candidate, "takeover": takeover})
    recoverable = [x for x in audited if x["takeover"]["EXPERT_SHELL_RECOVERABLE"]]
    category_order = (
        "A_RESET_EARLY", "B_HISTORICAL_7_12", "C_LAST3_BEFORE_LOSS",
        "D_NEAREST_UNRECOVERABLE_SUCCESSOR", "E_CLOSEST_C1_SHELL", "F_PRE_DOWNSTREAM_CONTACT_RECOVERABLE",
    )
    selected, seen = [], set()
    for category in category_order:
        item = next((x for x in recoverable if category in x["categories"] and id(x) not in seen), None)
        if item is not None:
            selected.append(item); seen.add(id(item))
    remaining = sorted((x for x in recoverable if id(x) not in seen), key=lambda x: (x["row"]["D_C1"], -x["row"]["policy_step"]))
    selected.extend(remaining[:max(0, BOUNDARY_TARGET - len(selected))])
    req(len(selected) >= BOUNDARY_TARGET, "minimum 16 proven recoverable boundary states")
    selected = selected[:BOUNDARY_TARGET]
    covered = set().union(*(x["categories"] for x in selected))
    req(set(category_order).issubset(covered), "all six boundary categories represented")
    return selected, audited


def checkpoint_action(definition, params, obs, agent, c1_params, support, scale, core, outer):
    return np.asarray(jax.device_get(pc.compose_actions(definition, params, jnp.asarray(obs), jnp.asarray(obs), agent, c1_params, support, scale, core, outer)), np.float32)


def preservation_audit(states, checkpoints, oracle, agent, c1_params, support, scale, core, outer, demos):
    reports = {}
    for step, (definition, params, global_mae, checkpoint_sha) in checkpoints.items():
        rows = []
        for state in states:
            obs = state["row"]["observation"]
            action = checkpoint_action(definition, params, obs, agent, c1_params, support, scale, core, outer)
            expert = np.asarray(demos[state["takeover"]["expert_source_row"]]["actions"], np.float32)
            branch = oracle.action_then_takeover(state["row"]["snapshot"], action)
            error = np.abs(action[:6] - expert[:6])
            rows.append({"source_policy": state["source_policy"], "seed": state["row"]["seed"], "step": state["row"]["policy_step"],
                         "obs20_sha": obs_hash(obs), "policy_action": action, "expert_action": expert,
                         "per_axis_error": error, "action_error_l2": float(np.linalg.norm(error)),
                         "shell_preserved": bool(branch["one_step"]["contact_healthy"] and branch["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"]),
                         "core_preserved": bool(branch["one_step"]["contact_healthy"] and branch["full_expert_after"]["EXPERT_CORE_RECOVERABLE"]),
                         "success_preserved": bool(branch["one_step"]["contact_healthy"] and branch["full_expert_after"]["EXPERT_SUCCESS_RECOVERABLE"]),
                         "branch": branch})
        errors = np.asarray([r["per_axis_error"] for r in rows])
        l2 = np.asarray([r["action_error_l2"] for r in rows])
        reports[step] = {
            "checkpoint_sha": checkpoint_sha, "global_dagger_mae": global_mae,
            "BOUNDARY_SHELL_PRESERVATION_RATE": float(np.mean([r["shell_preserved"] for r in rows])),
            "BOUNDARY_CORE_PRESERVATION_RATE": float(np.mean([r["core_preserved"] for r in rows])),
            "BOUNDARY_SUCCESS_PRESERVATION_RATE": float(np.mean([r["success_preserved"] for r in rows])),
            "BOUNDARY_ACTION_MAE": float(np.mean(errors)), "BOUNDARY_ACTION_P95": float(np.quantile(errors, .95)),
            "BOUNDARY_ACTION_MAX": float(np.max(errors)), "per_axis_mae": np.mean(errors, axis=0),
            "per_axis_p95": np.quantile(errors, .95, axis=0), "rows": rows,
        }
    return reports


def margin_audit(states, oracle, demos):
    output = []
    for state in states:
        expert = np.asarray(demos[state["takeover"]["expert_source_row"]]["actions"], np.float32)
        margins = np.full((6, 2), 0.05, np.float32)
        probes = []
        for axis in range(6):
            for direction_index, direction in enumerate((1.0, -1.0)):
                for magnitude in PERTURBATIONS:
                    action = expert.copy()
                    action[axis] = np.clip(action[axis] + direction * magnitude, -1.0, 1.0)
                    effective = abs(float(action[axis] - expert[axis]))
                    branch = oracle.action_then_takeover(state["row"]["snapshot"], action)
                    preserved = bool(branch["one_step"]["contact_healthy"] and branch["full_expert_after"]["EXPERT_SHELL_RECOVERABLE"])
                    probes.append({"axis": axis, "direction": int(direction), "requested_magnitude": magnitude,
                                   "effective_magnitude": effective, "shell_preserved": preserved})
                    if not preserved:
                        margins[axis, direction_index] = magnitude
                        break
        conservative = np.min(margins, axis=1)
        minimum = float(np.min(conservative))
        classification = "EXTREME" if minimum <= .005 else "HIGH" if minimum <= .01 else "MEDIUM" if minimum <= .02 else "LOW"
        output.append({"source_policy": state["source_policy"], "seed": state["row"]["seed"],
                       "step": state["row"]["policy_step"], "obs20_sha": obs_hash(state["row"]["observation"]),
                       "expert_action": expert, "margin_pos_neg": margins, "margin_axis": conservative,
                       "margin_min": minimum, "sensitivity_class": classification, "probes": probes})
    return output


def normalized_risk(preservation, margins):
    by_key = {row["obs20_sha"]: row for row in margins}
    reports = {}
    for step, report in preservation.items():
        values = []
        for row in report["rows"]:
            margin = np.asarray(by_key[row["obs20_sha"]]["margin_axis"], np.float32)
            normalized = np.asarray(row["per_axis_error"], np.float32) / np.maximum(margin, EPS)
            values.append({**row, "recovery_margin_axis": margin, "normalized_error": normalized})
        array = np.asarray([x["normalized_error"] for x in values])
        reports[step] = {"NORMALIZED_ERROR_MEAN": float(np.mean(array)), "NORMALIZED_ERROR_P95": float(np.quantile(array, .95)),
                         "NORMALIZED_ERROR_MAX": float(np.max(array)), "MARGIN_VIOLATION_FRACTION": float(np.mean(array >= 1.0)),
                         "rows": values}
    return reports


def sampling_representation(states, margins, dataset, train_indices, scale):
    observations = np.asarray([row["observations"] for row in dataset], np.float32)
    normalized = observations / scale
    train = np.asarray(train_indices, np.int64)
    critical_all, critical_train, extreme_high_train = set(), set(), set()
    margin_by_hash = {m["obs20_sha"]: m for m in margins}
    matches = []
    for state in states:
        query = state["row"]["observation"] / scale
        distance = np.sqrt(np.sum(np.square(normalized - query), axis=1))
        nearest = int(np.argmin(distance)); nearest_train = int(train[np.argmin(distance[train])])
        critical_all.add(nearest); critical_train.add(nearest_train)
        cls = margin_by_hash[obs_hash(state["row"]["observation"])]["sensitivity_class"]
        if cls in ("EXTREME", "HIGH"): extreme_high_train.add(nearest_train)
        matches.append({"obs20_sha": obs_hash(state["row"]["observation"]), "nearest_dataset_index": nearest,
                        "nearest_distance": float(distance[nearest]), "nearest_train_index": nearest_train,
                        "nearest_train_distance": float(distance[nearest_train]), "sensitivity_class": cls})

    demo_train = [i for i in train_indices if dataset[i]["source_type"] == "existing_expert"]
    dagger_train = [i for i in train_indices if dataset[i]["source_type"] == "recoverable_dagger"]

    def source_probability(indices, selected):
        groups = defaultdict(list)
        for index in indices: groups[dataset[index]["sampling_episode"]].append(index)
        return sum((1.0 / len(groups)) * (sum(i in selected for i in group) / len(group)) for group in groups.values())

    critical_fraction = .5 * source_probability(demo_train, critical_train) + .5 * source_probability(dagger_train, critical_train)
    extreme_fraction = .5 * source_probability(demo_train, extreme_high_train) + .5 * source_probability(dagger_train, extreme_high_train)
    return {"CRITICAL_DATASET_FRACTION": len(critical_all) / len(dataset),
            "CRITICAL_EFFECTIVE_BATCH_FRACTION": critical_fraction,
            "CRITICAL_EXAMPLES_PER_1000_UPDATES": 64 * 1000 * critical_fraction,
            "EXTREME_HIGH_EFFECTIVE_BATCH_FRACTION": extreme_fraction,
            "critical_train_indices": sorted(critical_train), "extreme_high_train_indices": sorted(extreme_high_train),
            "matches": matches}


def gradient_audit(dataset, train_indices, critical_indices, checkpoints, agent, c1_params, support, scale, core, outer):
    critical = sorted(set(critical_indices))
    rest = sorted(set(train_indices) - set(critical))
    if not critical:
        return {"status": "NO_EXTREME_HIGH_STATES", "by_checkpoint": {}, "CRITICAL_REST_GRAD_CONFLICT": "MIXED"}

    def grad_for(definition, params, indices, axis=None):
        obs = jnp.asarray(pc.observations(dataset, indices)); target = jnp.asarray(pc.targets(dataset, indices))
        def loss_fn(candidate):
            pred = pc.compose_actions(definition, candidate, obs, obs, agent, c1_params, support, scale, core, outer)
            error = jnp.square(pred[..., :6] - target[..., :6])
            return jnp.mean(error if axis is None else error[..., axis])
        loss, gradient = jax.value_and_grad(loss_fn)(params)
        return float(loss), gradient

    reports = {}
    for step in GRADIENT_STEPS:
        definition, params = checkpoints[step]
        critical_loss, gc = grad_for(definition, params, critical)
        rest_loss, gr = grad_for(definition, params, rest)
        axis = []
        for output_axis in range(6):
            _, gca = grad_for(definition, params, critical, output_axis)
            _, gra = grad_for(definition, params, rest, output_axis)
            axis.append({"axis": output_axis, "critical_norm": tree_norm(gca), "rest_norm": tree_norm(gra), "cosine": tree_cosine(gca, gra)})
        reports[step] = {"critical_loss": critical_loss, "rest_loss": rest_loss, "critical_norm": tree_norm(gc),
                         "rest_norm": tree_norm(gr), "cosine": tree_cosine(gc, gr), "per_layer_cosine": layer_cosines(gc, gr),
                         "per_output_axis": axis}
    cosines = [reports[s]["cosine"] for s in GRADIENT_STEPS]
    conflict = bool(sum(x < 0 for x in cosines) >= 3 and float(np.median(cosines)) <= -0.10)
    return {"status": "OK", "by_checkpoint": reports, "cosines": cosines,
            "CRITICAL_REST_GRAD_CONFLICT": "YES" if conflict else "NO"}


def anisotropy_and_localization(margins, risks, source_states):
    margin_array = np.asarray([row["margin_axis"] for row in margins], np.float32)
    classes = [row["sensitivity_class"] for row in margins]
    eh = np.asarray([i for i, x in enumerate(classes) if x in ("EXTREME", "HIGH")], np.int64)
    ratios = np.max(margin_array, axis=1) / np.maximum(np.min(margin_array, axis=1), EPS)
    eh_ratio_fraction = float(np.mean(ratios[eh] >= 4.0)) if len(eh) else 0.0
    medians = np.median(margin_array, axis=0)
    one_axis = [bool(medians[i] <= .25 * np.median(np.delete(medians, i))) for i in range(6)]
    anisotropy = bool(eh_ratio_fraction >= .25 or any(one_axis))
    risk3 = risks[3000]["rows"]
    normalized = np.asarray([row["normalized_error"] for row in risk3])
    violations = np.sum(normalized >= 1.0, axis=0)
    p95 = np.quantile(normalized, .95, axis=0)
    max_violation = max(int(np.max(violations)), 1)
    dominant = [int(i) for i in range(6) if violations[i] >= .5 * max_violation and (violations[i] > 0 or p95[i] == np.max(p95))]
    sensitive = [i for i in range(6) if one_axis[i] or medians[i] <= 1.25 * np.min(medians)]
    aligns = bool(set(dominant) & set(sensitive))
    state_scores = np.max(normalized, axis=1)
    order = np.argsort(-state_scores)
    top_count = max(1, math.ceil(len(order) * .25))
    total_excess = float(np.sum(np.maximum(state_scores - 1.0, 0.0)))
    top_excess = float(np.sum(np.maximum(state_scores[order[:top_count]] - 1.0, 0.0)))
    concentration = top_excess / max(total_excess, EPS)
    state_by_hash = {obs_hash(x["row"]["observation"]): x for x in source_states}
    top10 = []
    for i in order[:10]:
        risk = risk3[int(i)]; state = state_by_hash[risk["obs20_sha"]]
        top10.append({"source": state["source_policy"], "episode": state["row"]["seed"], "step": state["row"]["policy_step"],
                      "Obs20_hash": risk["obs20_sha"], "g2": state["row"]["g2"], "expert_action": risk["expert_action"],
                      "policy_action": risk["policy_action"], "per_axis_error": risk["per_axis_error"],
                      "per_axis_recovery_margin": risk["recovery_margin_axis"], "normalized_error": risk["normalized_error"],
                      "next_state_recoverability": risk["shell_preserved"]})
    return {"axis_median_margin": medians, "axis_p10_margin": np.quantile(margin_array, .10, axis=0),
            "axis_min_margin": np.min(margin_array, axis=0), "global_margin_max_min_ratio": float(np.max(margin_array) / max(float(np.min(margin_array)), EPS)),
            "extreme_high_state_ratio_ge4_fraction": eh_ratio_fraction, "one_axis_quarter_rule": one_axis,
            "RECOVERY_MARGIN_ANISOTROPY": "YES" if anisotropy else "NO", "DOMINANT_RISK_AXES": dominant,
            "sensitive_axes": sensitive, "policy_failures_align_sensitive_axes": aligns,
            "top_quartile_tail_excess_concentration": concentration, "TOP_10_BOUNDARY_RISK_STATES": top10}


def static_invariance(agent, c1_params, support, scale, core, outer, checkpoints):
    obs = np.asarray(support[:min(64, len(support))], np.float32)
    lower = np.asarray(jax.device_get(lower_policy_components(agent, c1_params, jnp.asarray(obs))[1]), np.float32)
    g2 = np.asarray(jax.device_get(progressive_gate(obs, support, scale, core, outer)))
    indices = np.flatnonzero(g2 == 0.0)
    req(len(indices) > 0, "C1 core invariant probes")
    reports = {}
    for step, (definition, params, _, _) in checkpoints.items():
        action = np.asarray(jax.device_get(pc.compose_actions(definition, params, jnp.asarray(obs), jnp.asarray(obs), agent, c1_params, support, scale, core, outer)), np.float32)
        reports[step] = {"g2_zero_probe_count": len(indices), "max_action_delta": float(np.max(np.abs(action[indices] - lower[indices]))),
                         "pass": bool(np.array_equal(action[indices], lower[indices]))}
    return reports


def main():
    req(Path.cwd() == ROOT and not OUT.exists(), "fresh output and project cwd")
    req(jax.default_backend() == "gpu", "GPU required")
    OUT.mkdir(parents=True); started = time.time()
    protected_paths = {
        "production_demo": p4.DEMO, "audit_dataset": AUDIT_DATA,
        "dagger_r1": DAGGER_DATA / "C2_DAGGER_ROUND1_V1.pkl",
        "dagger_r2": DAGGER_DATA / "C2_DAGGER_ROUND2_V1.pkl",
        "dagger_r3": DAGGER_DATA / "C2_DAGGER_ROUND3_V1.pkl",
        "checkpoint_1k": CONVERGENCE_RUN / "checkpoints/constant_plateau_c2_1000.msgpack",
        "checkpoint_2k": CONVERGENCE_RUN / "checkpoints/constant_plateau_c2_2000.msgpack",
        "checkpoint_3k": CONVERGENCE_RUN / "checkpoints/constant_plateau_c2_3000.msgpack",
    }
    protected_before = {name: sha(path) for name, path in protected_paths.items()}; dump(OUT / "protected_before.json", protected_before)
    req(protected_before["production_demo"] == DEMO_SHA, "production Demo SHA")

    demos = pickle.loads(p4.DEMO.read_bytes()); demo_obs, demo_scale = distance_model(demos)
    dataset = pickle.loads(AUDIT_DATA.read_bytes()); train_indices, demo_valid, dagger_valid = pc.split_indices(dataset)
    support_bundle = pickle.loads((review.PREFLIGHT / "C1_CONFIRMED_SUPPORT_V1.pkl").read_bytes())
    support = np.asarray(support_bundle["observations"], np.float32); scale = np.asarray(support_bundle["normalization_scale"], np.float32)
    preflight = json.loads((review.PREFLIGHT / "result.json").read_text()); core = float(preflight["C1_CORE_THRESHOLD"]); outer = float(preflight["C1_SHELL_OUTER_THRESHOLD"])
    convergence_report = json.loads((CONVERGENCE_RUN / "convergence_3k.json").read_text())
    cfg = load_config(2)

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        common, _, _ = geo.build_common_start(cfg, demos)
        source = flax.serialization.from_bytes(review.init_loaded(common), review.SOURCE_CP.read_bytes())
        agent = common.replace(state=source["agent_state"]); c1_params = source["c1_params"]
        base_hash, c1_hash = geo.thash(agent.state.params["modules_actor"]), geo.thash(c1_params)
        parameter = json.loads((pc.PARAM_RUN / "result.json").read_text())
        offline = load_dagger_head(parameter["SELECTED_OFFLINE_CHECKPOINT"]["path"])
        policies = {
            "offline": offline,
            "dagger_r1": load_dagger_head(DAGGER_RUN / "checkpoints/c2_dagger_round1_head_1000.msgpack"),
            "dagger_r2": load_dagger_head(DAGGER_RUN / "checkpoints/c2_dagger_round2_head_1000.msgpack"),
            "dagger_r3": load_dagger_head(DAGGER_RUN / "checkpoints/c2_dagger_round3_head_1000.msgpack"),
        }
        replays = build_source_replays(agent, c1_params, policies, support, scale, core, outer, demo_obs, demo_scale)
        replay_verification = {
            "offline_to_r1": verify_existing_rows("offline", replays["offline"], DAGGER_DATA / "C2_DAGGER_ROUND1_V1.pkl"),
            "r1_to_r2": verify_existing_rows("dagger_r1", replays["dagger_r1"], DAGGER_DATA / "C2_DAGGER_ROUND2_V1.pkl"),
            "r2_to_r3": verify_existing_rows("dagger_r2", replays["dagger_r2"], DAGGER_DATA / "C2_DAGGER_ROUND3_V1.pkl"),
        }
        dump(OUT / "deterministic_existing_trajectory_replay.json", replay_verification)
        known_maps = {"offline": audit_map_from_json(), "dagger_r1": audit_map_from_manifest(2), "dagger_r2": audit_map_from_manifest(3)}
        pool = candidate_pool(replays, known_maps)
        oracle = RecoveryOracle(demos, demo_obs, demo_scale, support, scale, core, outer)
        try:
            states, all_candidates = select_authoritative(pool, oracle)
            public = []
            for state in states:
                entry = public_state(state["row"], state["categories"], state["source_policy"])
                entry.update(state["takeover"]); public.append(entry)
            DATASET_PATH.write_bytes(pickle.dumps(public, pickle.HIGHEST_PROTOCOL)); dataset_sha = sha(DATASET_PATH)
            dump(OUT / "multi_boundary_manifest.json", {"name": "C2_MULTI_BOUNDARY_AUDIT_SET_V1", "sha256": dataset_sha,
                 "rows": len(public), "selection_target": BOUNDARY_TARGET, "candidate_count": len(all_candidates),
                 "expert_shell_recoverable_only": True, "exact_snapshot_reconstructed_from_existing_trajectory": True,
                 "categories": sorted(set().union(*(set(x["categories"]) for x in public))),
                 "source_counts": {label: sum(x["source_policy"] == label for x in public) for label in policies}})

            checkpoints = {}
            gradient_checkpoints = {}
            definition0, params0, _, _ = convergence.init_model()
            gradient_checkpoints[0] = (definition0, params0)
            for step in sorted(set(CHECKPOINT_STEPS) | set(GRADIENT_STEPS[1:])):
                path = CONVERGENCE_RUN / f"checkpoints/constant_plateau_c2_{step:04d}.msgpack"
                definition, params = load_head_checkpoint(path)
                gradient_checkpoints[step] = (definition, params)
                if step in CHECKPOINT_STEPS:
                    global_mae = float(convergence_report["reports"][str(step)]["dagger_valid"]["mae"])
                    checkpoints[step] = (definition, params, global_mae, sha(path))
            invariance = static_invariance(agent, c1_params, support, scale, core, outer, checkpoints)
            req(all(x["pass"] for x in invariance.values()), "C0/C1 exact g2=0 invariance")
            dump(OUT / "checkpoint_resolution_and_invariance.json", {"C0_BASE_SHA": base_hash, "C1_RESIDUAL_SHA": c1_hash,
                 "P2_COMPOSITION": "IDENTICAL", "checkpoints": {s: {"sha": checkpoints[s][3], "global_dagger_mae": checkpoints[s][2]} for s in checkpoints},
                 "invariance": invariance})

            preservation = preservation_audit(states, checkpoints, oracle, agent, c1_params, support, scale, core, outer, demos)
            dump(OUT / "existing_checkpoint_multi_boundary_audit.json", preservation)
            passing = [s for s, r in preservation.items() if r["BOUNDARY_SHELL_PRESERVATION_RATE"] >= SHELL_PASS and r["BOUNDARY_CORE_PRESERVATION_RATE"] >= CORE_PASS and r["global_dagger_mae"] <= GLOBAL_DAGGER_CATASTROPHIC_MAX and invariance[s]["pass"]]
            passing.sort(key=lambda s: (-preservation[s]["BOUNDARY_SHELL_PRESERVATION_RATE"], -preservation[s]["BOUNDARY_CORE_PRESERVATION_RATE"], preservation[s]["BOUNDARY_ACTION_P95"], preservation[s]["global_dagger_mae"]))
            existing_selected = passing[0] if passing else None
            existing_pilot = None
            if existing_selected is not None:
                definition, params, _, _ = checkpoints[existing_selected]
                historical = offline
                existing_pilot = convergence.pilot(definition, params, historical, dataset, agent, c1_params, support, scale, core, outer, demo_obs, demo_scale)
                dump(OUT / "existing_boundary_checkpoint_pilot.json", existing_pilot)

            if existing_selected is not None and existing_pilot["passed"]:
                margins = []; risks = {}; representation = {}; gradients = {}; anisotropy = {}
                root = "GLOBAL_MAE_CHECKPOINT_SELECTION_WAS_MISALIGNED"; selected_repair = "NO_NEW_TRAINING"
            else:
                margins = margin_audit(states, oracle, demos); dump(OUT / "recovery_action_sensitivity_map.json", margins)
                risks = normalized_risk(preservation, margins); dump(OUT / "checkpoint_error_margin_risk.json", risks)
                representation = sampling_representation(states, margins, dataset, train_indices, scale); dump(OUT / "critical_training_representation.json", representation)
                gradients = gradient_audit(dataset, train_indices, representation["extreme_high_train_indices"], gradient_checkpoints, agent, c1_params, support, scale, core, outer); dump(OUT / "critical_vs_rest_gradient_audit.json", gradients)
                anisotropy = anisotropy_and_localization(margins, risks, states); dump(OUT / "anisotropy_and_tail_localization.json", anisotropy)
                conflict = gradients["CRITICAL_REST_GRAD_CONFLICT"] == "YES"
                anisotropic = anisotropy["RECOVERY_MARGIN_ANISOTROPY"] == "YES" and anisotropy["policy_failures_align_sensitive_axes"]
                underrepresented = bool(not conflict and anisotropy["RECOVERY_MARGIN_ANISOTROPY"] == "NO" and representation["CRITICAL_EFFECTIVE_BATCH_FRACTION"] < MATERIALLY_SMALL_BATCH_FRACTION and anisotropy["top_quartile_tail_excess_concentration"] >= TAIL_CONCENTRATION_MIN)
                if conflict:
                    root = "GLOBAL_C2_AND_BOUNDARY_RECOVERY_POLICY_PARAMETER_CONFLICT"; selected_repair = "C2_LOCAL_BOUNDARY_RECOVERY_HEAD_V1"
                elif anisotropic:
                    root = "RECOVERY_BOUNDARY_AXIS_SENSITIVITY_NOT_CAPTURED_BY_UNIFORM_ACTION_MSE"; selected_repair = "RECOVERY_MARGIN_NORMALIZED_SUPERVISION_V1"
                elif underrepresented:
                    root = "RECOVERY_BOUNDARY_STATES_UNDERREPRESENTED"; selected_repair = "RECOVERY_BOUNDARY_BALANCED_SUPERVISION_V1"
                else:
                    root = "MIXED_RECOVERY_BOUNDARY_FAILURE"; selected_repair = "NONE"

            class_fraction = {name: (sum(x["sensitivity_class"] == name for x in margins) / len(margins) if margins else None) for name in ("EXTREME", "HIGH", "MEDIUM", "LOW")}
            result = {
                "C2_RECOVERY_BOUNDARY_TAIL_RISK_REVIEW": "PASS", "MULTI_BOUNDARY_SET_SHA": dataset_sha,
                "BOUNDARY_STATE_COUNT": len(states), "EXTREME_STATE_FRACTION": class_fraction["EXTREME"],
                "HIGH_STATE_FRACTION": class_fraction["HIGH"], "MEDIUM_STATE_FRACTION": class_fraction["MEDIUM"],
                "LOW_STATE_FRACTION": class_fraction["LOW"], "DOMINANT_RISK_AXES": anisotropy.get("DOMINANT_RISK_AXES", "NOT_RUN"),
                "CHECKPOINT_1K_SHELL_PRESERVATION": preservation[1000]["BOUNDARY_SHELL_PRESERVATION_RATE"],
                "CHECKPOINT_2K_SHELL_PRESERVATION": preservation[2000]["BOUNDARY_SHELL_PRESERVATION_RATE"],
                "CHECKPOINT_3K_SHELL_PRESERVATION": preservation[3000]["BOUNDARY_SHELL_PRESERVATION_RATE"],
                "CHECKPOINT_1K_CORE_PRESERVATION": preservation[1000]["BOUNDARY_CORE_PRESERVATION_RATE"],
                "CHECKPOINT_2K_CORE_PRESERVATION": preservation[2000]["BOUNDARY_CORE_PRESERVATION_RATE"],
                "CHECKPOINT_3K_CORE_PRESERVATION": preservation[3000]["BOUNDARY_CORE_PRESERVATION_RATE"],
                "EXISTING_BOUNDARY_CHECKPOINT_PASS": "YES" if existing_selected is not None else "NO",
                "EXISTING_BOUNDARY_CHECKPOINT_SELECTED": existing_selected,
                "EXISTING_BOUNDARY_CHECKPOINT_PILOT": "NOT_RUN" if existing_pilot is None else existing_pilot["C2_PILOT"],
                "CRITICAL_DATASET_FRACTION": representation.get("CRITICAL_DATASET_FRACTION"),
                "CRITICAL_EFFECTIVE_BATCH_FRACTION": representation.get("CRITICAL_EFFECTIVE_BATCH_FRACTION"),
                "CRITICAL_EXAMPLES_PER_1000_UPDATES": representation.get("CRITICAL_EXAMPLES_PER_1000_UPDATES"),
                "CRITICAL_REST_GRAD_COSINE_BY_CHECKPOINT": gradients.get("cosines", "NOT_RUN"),
                "CRITICAL_REST_GRAD_CONFLICT": gradients.get("CRITICAL_REST_GRAD_CONFLICT", "NOT_RUN"),
                "RECOVERY_MARGIN_ANISOTROPY": anisotropy.get("RECOVERY_MARGIN_ANISOTROPY", "NOT_RUN"),
                "PRIMARY_ROOT_CAUSE": root, "SELECTED_REPAIR": selected_repair,
                "SELECTED_CHECKPOINT": existing_selected if selected_repair == "NO_NEW_TRAINING" else None,
                "NEXT": "MATCHED_C2_CONFIRMATION" if selected_repair == "NO_NEW_TRAINING" else "STOP" if selected_repair == "NONE" else selected_repair,
                "PRODUCTION_DEMO_SHA_UNCHANGED": "PENDING_FINAL_CHECK", "ONLINE_ADMISSION_THRESHOLD": 7.41773975814,
                "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO", "EVAL_RESET_CANONICAL": "YES", "EVAL_GOAL_CANONICAL": "YES",
                "EVAL_CURRICULUM_ACTIVE": "NO", "PROTECTED_ARTIFACTS_CHANGED": "PENDING_FINAL_CHECK",
                "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE", "elapsed_seconds": time.time() - started,
            }
            protected_after = {name: sha(path) for name, path in protected_paths.items()}; req(protected_after == protected_before, "protected artifact mutation")
            dump(OUT / "protected_after.json", protected_after)
            result["PRODUCTION_DEMO_SHA_UNCHANGED"] = "YES"; result["PROTECTED_ARTIFACTS_CHANGED"] = "NO"
            dump(OUT / "result.json", result); print(json.dumps(jvalue(result), indent=2, sort_keys=True))
        finally:
            oracle.close()


if __name__ == "__main__":
    main()
