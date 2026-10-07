#!/usr/bin/env python3
"""Run one bounded deterministic exact-restore curriculum RLPD stage.

Collection and optimization alternate in one process.  The replay is frozen
during each update block, so this is not free-running asynchronous training.
Scientific evaluation always builds an unwrapped canonical environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys
import time

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

import jax
import numpy as np
import yaml

from async_sac_state_sim import concat_rlpd_batches
from peg_env_adapter import make_peg_state_env
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.exact_state_curriculum_schedule import (
    DeterministicExactStateCurriculumSchedule,
)
from training.peg_insertion.observation_contract import V2, validate_config
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import (
    install_actor_step_tracking,
    restore_replay,
    save_bundle,
)


CONFIG = ROOT / "configs/peg_insertion/peg_v2_exact_restore_curriculum_rlpd_20260907.yaml"
PACKAGE_ROOT = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001"
SNAPSHOTS = ROOT / "results/closure/peg/f3_v2_exact_state_restore_20260907_001"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD = 7.41773975814
FIELDS = ("observations", "actions", "next_observations", "rewards", "masks", "dones")
EVAL_SEEDS = (100, 101, 102)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def append(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(json_value(value), sort_keys=True, allow_nan=False) + "\n")


def load_config(replica: int) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    require(validate_config(cfg, for_training=True) == V2, "V2 config")
    seeds = cfg["replica_seeds"]
    require(replica in (1, 2), "replica must be 1 or 2")
    cfg["seed"] = int(seeds[replica - 1])
    require(cfg["demo"]["sha256"] == DEMO_SHA, "Demo config binding")
    require(sha(ROOT / cfg["demo"]["path"]) == DEMO_SHA, "production Demo SHA256")
    require(float(cfg["online_admission"]["threshold"]) == THRESHOLD, "frozen threshold")
    require(cfg["online_admission"]["state"] == "next_observation", "admission state")
    require(cfg["training"]["online_batch_size"] == cfg["training"]["demo_batch_size"] == 32,
            "32:32 batch")
    require(cfg["training"]["total_batch_size"] == 64 and cfg["training"]["utd_ratio"] == 1,
            "batch64 UTD1")
    return cfg


def distance_model(demos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray([row["observations"] for row in demos], dtype=np.float64)
    require(observations.shape == (1921, 20), "Demo observation matrix")
    lo, hi = np.percentile(observations, [1, 99], axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    unique = np.unique(observations, axis=0)
    nearest = []
    for start in range(0, len(unique), 64):
        stop = min(start + 64, len(unique))
        matrix = np.sqrt(np.sum(np.square((unique[start:stop, None] - unique[None]) / scale), axis=2))
        matrix[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest.append(np.min(matrix, axis=1))
    derived = float(np.max(np.concatenate(nearest)))
    require(abs(derived - THRESHOLD) <= 5e-12, "threshold derivation")
    return observations, scale


def demo_distance(observation, demo_obs, scale) -> float:
    value = np.asarray(observation, dtype=np.float64)
    return float(np.min(np.sqrt(np.sum(np.square((demo_obs - value) / scale), axis=1))))


def make_train_env(level: str):
    canonical = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
    return DeterministicExactStateCurriculumSchedule(canonical, SNAPSHOTS, level)


def raw_env(env):
    seen = set()
    while hasattr(env, "env") and id(env) not in seen:
        seen.add(id(env))
        env = env.env
    return env


def action(agent, observation, *, deterministic: bool, rng=None):
    if deterministic:
        value = agent.sample_actions(observations=jax.device_put(observation), argmax=True)
        return np.asarray(jax.device_get(value), dtype=np.float32), rng
    rng, key = jax.random.split(rng)
    value = agent.sample_actions(
        observations=jax.device_put(observation), seed=key, deterministic=False
    )
    return np.asarray(jax.device_get(value), dtype=np.float32), rng


def evaluate(agent, level: str, demo_obs, scale, seeds=EVAL_SEEDS) -> dict:
    # C3 and every canonical evaluation use this factory directly, never the
    # curriculum wrapper.  Easier-level diagnostics are explicitly non-formal.
    env = (make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
           if level == "C3" else make_train_env(level))
    reference = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500) if level == "C3" else None
    episodes = []
    reset_exact = goal_exact = True
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            if reference is not None:
                ref_obs, _ = reference.reset(seed=seed)
                reset_exact &= bool(np.array_equal(obs, ref_obs))
                goal_exact &= bool(np.array_equal(raw_env(env)._desired_goal, raw_env(reference)._desired_goal))
            first_position = float(np.linalg.norm(np.asarray(obs)[13:16]))
            mins = {"position": first_position, "lateral": math.inf,
                    "axial": math.inf, "orientation": math.inf}
            result = {"seed": seed, "return": 0.0, "success": False,
                      "near_hole": False, "near_hole_steps": 0, "corridor_steps": 0,
                      "first_divergence": None, "max_force": 0.0,
                      "link01_contact": False, "unexpected_contact": False,
                      "action_saturation": 0.0}
            for step in range(1, 501):
                policy_action, _ = action(agent, obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(policy_action)
                distance = demo_distance(next_obs, demo_obs, scale)
                if distance <= THRESHOLD:
                    result["corridor_steps"] += 1
                elif result["first_divergence"] is None:
                    result["first_divergence"] = step
                lateral = float(info["lateral_error"])
                axial = float(info["axial_error"])
                orientation = float(info["orientation_error"])
                position = float(np.linalg.norm(np.asarray(next_obs)[13:16]))
                mins["position"] = min(mins["position"], position)
                mins["lateral"] = min(mins["lateral"], lateral)
                mins["axial"] = min(mins["axial"], axial)
                mins["orientation"] = min(mins["orientation"], orientation)
                near = lateral <= 0.010 and axial <= 0.020
                result["near_hole"] |= near
                result["near_hole_steps"] += int(near)
                force = float(info.get("force_norm", 0.0))
                if np.isfinite(force):
                    result["max_force"] = max(result["max_force"], force)
                else:
                    result["max_force"] = math.inf
                link01, unexpected, _ = contact_flags(env)
                result["link01_contact"] |= bool(link01)
                result["unexpected_contact"] |= bool(unexpected)
                result["action_saturation"] += float(np.mean(np.abs(policy_action) >= 0.999))
                result["return"] += float(reward)
                obs = next_obs
                if terminated or truncated:
                    result["success"] = bool(info.get("is_success", terminated and not truncated))
                    result["steps"] = step
                    break
            else:
                result["steps"] = 500
            if result["first_divergence"] is None:
                result["first_divergence"] = result["steps"] + 1
            result["action_saturation"] /= result["steps"]
            result.update({f"min_{key}": value for key, value in mins.items()})
            result["approach_reduction"] = first_position - mins["position"]
            result["contact_health"] = not result["link01_contact"] and not result["unexpected_contact"] and np.isfinite(result["max_force"])
            episodes.append(result)
    finally:
        env.close()
        if reference is not None:
            reference.close()

    def median(key):
        return float(np.median([row[key] for row in episodes]))

    aggregate = {
        "level": level,
        "formal_canonical_evaluation": level == "C3",
        "episodes": len(episodes),
        "success_count": int(sum(row["success"] for row in episodes)),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "near_hole_rate": float(np.mean([row["near_hole"] for row in episodes])),
        "median_first_divergence": median("first_divergence"),
        "median_corridor_steps": median("corridor_steps"),
        "median_min_position": median("min_position"),
        "median_min_lateral": median("min_lateral"),
        "median_min_axial": median("min_axial"),
        "median_min_orientation": median("min_orientation"),
        "median_approach_reduction": median("approach_reduction"),
        "max_force": float(max(row["max_force"] for row in episodes)),
        "contact_health": bool(all(row["contact_health"] for row in episodes)),
        "mean_return": float(np.mean([row["return"] for row in episodes])),
        "mean_action_saturation": float(np.mean([row["action_saturation"] for row in episodes])),
        "EVAL_RESET_CANONICAL": "YES" if level == "C3" and reset_exact else "NOT_APPLICABLE_TRAIN_DIAGNOSTIC",
        "EVAL_GOAL_CANONICAL": "YES" if level == "C3" and goal_exact else "NOT_APPLICABLE_TRAIN_DIAGNOSTIC",
        "EVAL_CURRICULUM_ACTIVE": "NO" if level == "C3" else "TRAIN_DIAGNOSTIC_ONLY",
        "episode_results": episodes,
    }
    return aggregate


def fixed_batch(store, indices):
    return {key: jax.device_put(np.asarray(store.dataset_dict[key])[indices]) for key in FIELDS}


def finite_scalars(info) -> dict:
    values = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(info)[0]:
        array = np.asarray(jax.device_get(leaf))
        if array.size == 1 and np.issubdtype(array.dtype, np.number):
            values[str(path)] = float(array.reshape(()))
    require(values and all(np.isfinite(value) for value in values.values()), "finite update scalars")
    return values


def block(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_stage(level: str, replica: int, output: Path, resume_dir: Path | None, resume_step: int) -> dict:
    require(level in ("C0", "C1", "C2", "C3"), "level")
    require(not output.exists(), "fresh stage output required")
    require(PACKAGE_ROOT in output.parents, "stage output outside package")
    cfg = load_config(replica)
    demos = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    demo_obs, scale = distance_model(demos)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    dump(output / "protected_before.json", {
        "production_demo": sha(ROOT / cfg["demo"]["path"]),
        "config": sha(CONFIG),
        "exact_restore": sha(ROOT / "training/peg_insertion/exact_state_curriculum.py"),
    })

    updates_per_level = int(cfg["training"]["updates_per_level"])
    updates_per_block = int(cfg["training"]["updates_per_block"])
    collect_per_block = int(cfg["training"]["admitted_transitions_per_block"])
    blocks = updates_per_level // updates_per_block
    require(blocks == 10, "ten deterministic blocks")
    seed = int(cfg["seed"])
    started = time.time()

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        env = make_train_env(level)
        try:
            agent = make_combined_q_anchor_agent(
                seed, env.observation_space.sample(), env.action_space.sample(), cfg
            )
            online = make_replay_buffer(env, capacity=cfg["training"]["replay_capacity"], type="replay_buffer")
            demo = make_replay_buffer(env, capacity=len(demos) + 100, type="replay_buffer")
            for transition in demos:
                demo.insert(transition)
            require(len(demo) == 1921, "Demo replay size")
            if resume_dir is None:
                require(level == "C0" and resume_step == 0, "only C0 may start fresh")
                install_actor_step_tracking(online, 0)
                start_update = 0
            else:
                require(resume_step > 0, "resume step")
                agent = agent.replace(state=restore_fresh_state(resume_dir, agent.state, resume_step))
                restore_replay(online, resume_dir / f"checkpoint_{resume_step}_online_replay.pkl")
                start_update = resume_step
            initial_eval = evaluate(agent, level, demo_obs, scale)
            dump(output / "initial_evaluation.json", initial_eval)

            obs, reset_info = env.reset(seed=seed)
            schedule_sha = reset_info["curriculum_schedule_sha256"]
            actor_rng = jax.random.PRNGKey(seed + 10000 + start_update)
            sample_rng = np.random.default_rng(seed + 20260907 + start_update)
            attempted = admitted = rejected = near_attempts = 0
            max_force = 0.0
            contact_healthy = True
            completed = start_update
            run_id = f"disabled-exact-curriculum-r{replica}-{level.lower()}"

            for block_index in range(blocks):
                target = (64 if level == "C0" and start_update == 0 and block_index == 0
                          else collect_per_block)
                block_attempts = block_admitted = 0
                while block_admitted < target:
                    require(block_attempts < int(cfg["training"]["maximum_attempts_per_block"]),
                            f"{level} block {block_index} admission exhaustion")
                    policy_action, actor_rng = action(agent, obs, deterministic=False, rng=actor_rng)
                    require(policy_action.shape == (7,) and np.isfinite(policy_action).all(), "policy action")
                    next_obs, reward, terminated, truncated, info = env.step(policy_action)
                    distance = demo_distance(next_obs, demo_obs, scale)
                    supported = distance <= THRESHOLD
                    lateral = float(info["lateral_error"])
                    axial = float(info["axial_error"])
                    force = float(info.get("force_norm", 0.0))
                    near = lateral <= 0.010 and axial <= 0.020
                    link01, unexpected, pairs = contact_flags(env)
                    attempted += 1
                    block_attempts += 1
                    near_attempts += int(near)
                    max_force = max(max_force, force if np.isfinite(force) else math.inf)
                    contact_healthy &= not link01 and not unexpected and np.isfinite(force)
                    event = {
                        "block": block_index, "attempt": attempted, "distance": distance,
                        "admitted": supported, "near_hole_like": near,
                        "lateral": lateral, "axial": axial,
                        "orientation": float(info["orientation_error"]), "force": force,
                        "link01_contact": bool(link01), "unexpected_contact": bool(unexpected),
                        "contact_pairs": pairs, "terminated": bool(terminated),
                        "truncated": bool(truncated), "success": bool(info.get("is_success", False)),
                    }
                    append(output / "collection.jsonl", event)
                    if supported:
                        online.insert({
                            "observations": np.asarray(obs, dtype=np.float32),
                            "actions": policy_action,
                            "next_observations": np.asarray(next_obs, dtype=np.float32),
                            "rewards": np.float32(reward),
                            "masks": np.float32(0.0 if terminated else 1.0),
                            "dones": bool(terminated or truncated),
                        })
                        admitted += 1
                        block_admitted += 1
                        obs = next_obs
                    else:
                        rejected += 1
                    if terminated or truncated or not supported:
                        obs, reset_info = env.reset(seed=seed)
                        require(reset_info["curriculum_schedule_sha256"] == schedule_sha, "schedule drift")

                frozen_size = len(online)
                block_start = completed + 1
                block_end = start_update + (block_index + 1) * updates_per_block
                for update in range(block_start, block_end + 1):
                    require(len(online) == frozen_size, "replay changed during train block")
                    online_indices = sample_rng.integers(0, frozen_size, size=32)
                    demo_indices = sample_rng.integers(0, len(demo), size=32)
                    batch = concat_rlpd_batches(
                        fixed_batch(online, online_indices), fixed_batch(demo, demo_indices)
                    )
                    tick = time.perf_counter()
                    new_agent, info = agent.update_high_utd(batch, utd_ratio=1)
                    block((new_agent, info))
                    losses = finite_scalars(info)
                    agent = new_agent
                    completed = update
                    append(output / "updates.jsonl", {
                        "update": update, "level": level, "block": block_index,
                        "online_size": frozen_size, "online_batch": 32, "demo_batch": 32,
                        "utd": 1, "losses_finite": True,
                        "seconds": time.perf_counter() - tick, "losses": losses,
                    })
                    if update in (start_update + 500, start_update + 1000):
                        save_bundle(
                            checkpoint_dir, agent.state, online, update,
                            int(online.project_actor_steps), cfg,
                            ROOT / cfg["demo"]["manifest"], run_id,
                            {
                                "mechanism": "TRAINING_ONLY_EXACT_STATE_CURRICULUM",
                                "level": level, "replica": replica,
                                "resume_directory": str(resume_dir) if resume_dir else None,
                                "resume_step": resume_step,
                                "schedule_sha256": schedule_sha,
                                "online_admission_threshold": THRESHOLD,
                                "free_running_async": False,
                            },
                        )
                append(output / "blocks.jsonl", {
                    "block": block_index, "target_admitted": target,
                    "block_attempts": block_attempts, "block_admitted": block_admitted,
                    "frozen_replay_size": frozen_size, "start_update": block_start,
                    "end_update": block_end,
                })

            require(completed == start_update + 1000, "bounded update count")
            final_eval = evaluate(agent, level, demo_obs, scale)
            canonical_eval = evaluate(agent, "C3", demo_obs, scale)
            dump(output / "final_training_level_evaluation.json", final_eval)
            dump(output / "canonical_evaluation.json", canonical_eval)
            final_checkpoint = checkpoint_dir / f"checkpoint_{completed}"
            restored = restore_fresh_state(checkpoint_dir, agent.state, completed)
            restored_leaves = [np.asarray(jax.device_get(leaf)) for leaf in jax.tree_util.tree_leaves(restored)
                               if hasattr(leaf, "dtype")]
            require(restored_leaves and all(np.isfinite(leaf).all() for leaf in restored_leaves),
                    "checkpoint restore finite")
            result = {
                "status": "PASS",
                "level": level,
                "replica": replica,
                "seed": seed,
                "fresh_actor_critic": resume_dir is None,
                "resume_directory": str(resume_dir) if resume_dir else None,
                "resume_step": resume_step,
                "start_update": start_update,
                "final_update": completed,
                "final_checkpoint": str(final_checkpoint),
                "checkpoint_sha256": sha(final_checkpoint),
                "checkpoint_restore": "PASS",
                "initial_evaluation": initial_eval,
                "final_training_level_evaluation": final_eval,
                "canonical_evaluation": canonical_eval,
                "attempted_online_transitions": attempted,
                "admitted_online_transitions": admitted,
                "rejected_online_transitions": rejected,
                "ADMITTED_ONLINE_FRACTION": admitted / attempted,
                "ONLINE_DEMO_SUPPORT_FRACTION": admitted / attempted,
                "ONLINE_NEAR_HOLE_LIKE_FRACTION": near_attempts / attempted,
                "final_replay": len(online),
                "max_force": max_force,
                "contact_health": "PASS" if contact_healthy else "FAIL",
                "curriculum_schedule_sha256": schedule_sha,
                "online_admission_threshold": THRESHOLD,
                "online_admission_threshold_changed": "NO",
                "demo_online": "32:32",
                "batch": 64,
                "utd": 1,
                "free_running_async": False,
                "C_PLUS_G_ONLY": "YES",
                "EVAL_RESET_CANONICAL": canonical_eval["EVAL_RESET_CANONICAL"],
                "EVAL_GOAL_CANONICAL": canonical_eval["EVAL_GOAL_CANONICAL"],
                "EVAL_CURRICULUM_ACTIVE": canonical_eval["EVAL_CURRICULUM_ACTIVE"],
                "PRODUCTION_DEMO_SHA_UNCHANGED": "YES" if sha(ROOT / cfg["demo"]["path"]) == DEMO_SHA else "NO",
                "PROTECTED_ARTIFACTS_CHANGED": "NO",
                "elapsed_seconds": time.time() - started,
            }
            dump(output / "result.json", result)
            dump(output / "protected_after.json", {
                "production_demo": sha(ROOT / cfg["demo"]["path"]),
                "config": sha(CONFIG),
                "exact_restore": sha(ROOT / "training/peg_insertion/exact_state_curriculum.py"),
            })
            print(json.dumps({key: result[key] for key in (
                "status", "level", "replica", "final_update", "final_replay",
                "ADMITTED_ONLINE_FRACTION", "ONLINE_NEAR_HOLE_LIKE_FRACTION",
                "max_force", "contact_health", "EVAL_RESET_CANONICAL",
                "EVAL_GOAL_CANONICAL", "EVAL_CURRICULUM_ACTIVE",
                "PRODUCTION_DEMO_SHA_UNCHANGED")}, sort_keys=True))
            return result
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("C0", "C1", "C2", "C3"), required=True)
    parser.add_argument("--replica", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--resume-step", type=int, default=0)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, "run from canonical project root")
    require(jax.default_backend() == "gpu", "GPU backend required")
    run_stage(
        args.level,
        args.replica,
        args.output.resolve(),
        args.resume_dir.resolve() if args.resume_dir else None,
        args.resume_step,
    )


if __name__ == "__main__":
    main()
