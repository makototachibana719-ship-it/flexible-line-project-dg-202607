#!/usr/bin/env python3
"""Canonical safety, immutable repaired handoff, and bounded +500 C+G retention."""
from __future__ import annotations

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
    str(ROOT), str(ROOT / "external/serl/serl_launcher"), str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

import flax
import jax
import numpy as np
import yaml

from async_sac_state_sim import concat_rlpd_batches
from scripts.audit_peg_f3_v2_actor_parameterization_distribution_shift import rollout
from scripts.run_peg_f3_v2_exact_restore_curriculum_stage import (
    action, append, block, demo_distance, distance_model, dump, finite_scalars, fixed_batch,
    load_config, make_train_env,
)
from scripts.run_peg_f3_v2_local_insertion_actor_bc import evaluate as behavior_evaluate
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from serl_launcher.utils.launcher import make_replay_buffer
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
from training.peg_insertion.resume import restore_replay, save_bundle


CONFIG = ROOT / "configs/peg_insertion/peg_v2_corrective_safety_retention_20260908.yaml"
BASE_CONFIG = ROOT / "configs/peg_insertion/peg_v2_exact_restore_curriculum_rlpd_20260907.yaml"
DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD = 7.41773975814
SOURCE_RUN = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001/replica1/C0_retry002"
SOURCE_POST_BC = ROOT / "results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001/checkpoints/bc_checkpoint_1000.msgpack"
ROUND1_DIR = ROOT / "results/closure/peg/f3_v2_on_policy_corrective_20260908_001"
ROUND1_RESULT = ROUND1_DIR / "round1_result.json"
ROUND1_DATA = ROOT / "results/datasets/peg_v2_corrective_postbc_round1_20260908_001/corrective_round1.pkl"
LOCAL_DIR = ROOT / "results/datasets/peg_v2_local_insertion_supervision_20260908_001"
LOCAL_DATA = LOCAL_DIR / "local_insertion_supervision.pkl"
OUT = ROUND1_DIR / "canonical_safety_and_retention"
HANDOFF = OUT / "repaired_handoff"
RETENTION = OUT / "retention_c0_plus500"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tree(path: Path) -> str:
    if path.is_file():
        return sha(path)
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(item.relative_to(path)).encode())
        h.update(bytes.fromhex(sha(item)))
    return h.hexdigest()


def protected() -> dict:
    paths = {
        "production_demo": DEMO,
        "base_training_config": BASE_CONFIG,
        "source_rlpd_checkpoint": SOURCE_RUN / "checkpoints/checkpoint_1000",
        "source_rlpd_replay": SOURCE_RUN / "checkpoints/checkpoint_1000_online_replay.pkl",
        "source_post_bc_checkpoint": SOURCE_POST_BC,
        "round1_checkpoint": Path(json.loads(ROUND1_RESULT.read_text())["ROUND1_BEST_CHECKPOINT"]),
        "round1_dataset": ROUND1_DATA,
        "base_local_dataset": LOCAL_DATA,
    }
    return {name: {"path": str(path), "sha256": sha_tree(path)} for name, path in paths.items()}


def load_agents(cfg, round1_checkpoint: Path):
    env = make_train_env("C0")
    try:
        fresh = make_combined_q_anchor_agent(
            cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg
        )
    finally:
        env.close()
    base_state = restore_fresh_state(SOURCE_RUN / "checkpoints", fresh.state, 1000)
    source_state = flax.serialization.from_bytes(base_state, SOURCE_POST_BC.read_bytes())
    repaired_state = flax.serialization.from_bytes(base_state, round1_checkpoint.read_bytes())
    return fresh.replace(state=source_state), fresh.replace(state=repaired_state)


def safety_gate(old: dict, new: dict, limits: dict) -> tuple[bool, dict]:
    old_approach = float(old["median_approach_reduction"])
    new_approach = float(new["median_approach_reduction"])
    approach_floor = float(limits["minimum_approach_reduction_ratio"]) * max(old_approach, 0.0)
    position_ceiling = float(old["median_min_position"]) + float(limits["maximum_min_position_regression"])
    force_ceiling = max(
        float(old["max_force"]) * float(limits["maximum_force_multiplier"]),
        float(old["max_force"]) + float(limits["maximum_force_additive_regression"]),
    )
    checks = {
        "approach_not_catastrophic": new_approach >= approach_floor,
        "position_not_catastrophic": float(new["median_min_position"]) <= position_ceiling,
        "no_new_contact_failure": bool(new["contact_health"]) or not bool(old["contact_health"]),
        "no_new_table_ee_contact": not bool(new["table_ee_contact"]) or bool(old["table_ee_contact"]),
        "force_finite": bool(np.isfinite(new["max_force"])),
        "force_not_severe_regression": float(new["max_force"]) <= force_ceiling,
    }
    details = {"checks": checks, "approach_floor": approach_floor,
               "position_ceiling": position_ceiling, "force_ceiling": force_ceiling}
    return all(checks.values()), details


def support_exit(agent, demo_obs, scale) -> int | None:
    rows = pickle.loads(LOCAL_DATA.read_bytes())
    split = json.loads((LOCAL_DIR / "split.json").read_text())
    train_ids, valid_ids = set(split["train_episodes"]), set(split["valid_episodes"])
    train = np.asarray([x["observations"] for x in rows if x["source_episode"] in train_ids], np.float64)
    valid = np.asarray([x["observations"] for x in rows if x["source_episode"] in valid_ids], np.float64)
    states, _ = rollout(agent, demo_obs, train, valid, scale)
    return next((x["step"] for x in states
                 if min(x["D_LOCAL_TRAIN"], x["D_LOCAL_VALID"]) > THRESHOLD), None)


def main() -> None:
    require(Path.cwd() == ROOT, "run from canonical project root")
    require(not OUT.exists(), "fresh safety/retention output required")
    require(jax.default_backend() == "gpu", "GPU backend required")
    specification = yaml.safe_load(CONFIG.read_text())
    cfg = load_config(1)
    require(float(specification["online_admission_threshold"]) == THRESHOLD, "frozen threshold")
    require(int(specification["retention_updates"]) == 500, "bounded +500")
    require(specification["retention_checkpoints"] == [1100, 1250, 1500], "retention checkpoints")
    require(sha(DEMO) == DEMO_SHA, "production Demo binding")
    round1 = json.loads(ROUND1_RESULT.read_text())
    require(round1["ROUND1_C0_REPAIR"] == "PASS", "Round1 strong pass required")
    require(round1["ROUND2_AUTHORIZED"] is False, "Round2 must be skipped after strong pass")
    round1_checkpoint = Path(round1["ROUND1_BEST_CHECKPOINT"])
    require(round1_checkpoint.is_file(), "Round1 checkpoint")
    before = protected()

    OUT.mkdir(parents=True, exist_ok=False)
    HANDOFF.mkdir()
    RETENTION.mkdir()
    checkpoint_dir = RETENTION / "checkpoints"
    checkpoint_dir.mkdir()
    dump(OUT / "protected_before.json", before)
    demos = pickle.loads(DEMO.read_bytes())
    demo_obs, scale = distance_model(demos)
    started = time.time()

    with portable_peg_runtime(ROOT), repaired_replay_datastore_insert():
        source, repaired = load_agents(cfg, round1_checkpoint)
        source_c3 = behavior_evaluate(source, "C3", demo_obs, scale)
        repaired_c3 = behavior_evaluate(repaired, "C3", demo_obs, scale)
        safety_pass, safety_details = safety_gate(
            source_c3, repaired_c3, specification["canonical_safety"]
        )
        canonical_safety = {
            "CANONICAL_SAFETY_CHECK": "PASS" if safety_pass else "FAIL",
            "EVAL_RESET_CANONICAL": "YES", "EVAL_GOAL_CANONICAL": "YES",
            "EVAL_CURRICULUM_ACTIVE": "NO", "source_post_bc": source_c3,
            "round1_repaired": repaired_c3, **safety_details,
        }
        dump(OUT / "canonical_safety.json", canonical_safety)
        require(safety_pass, "canonical safety gate failed; STOP before retention")

        # Round1 changed actor parameters outside RLPD. Reset only that optimizer's
        # moments so its state is mathematically consistent with the repaired actor.
        opt_states = dict(repaired.state.opt_states)
        opt_states["actor"] = repaired.state.txs["actor"].init(repaired.state.params)
        repaired = repaired.replace(state=repaired.state.replace(opt_states=opt_states))

        handoff_env = make_train_env("C0")
        try:
            online = make_replay_buffer(
                handoff_env, capacity=cfg["training"]["replay_capacity"], type="replay_buffer"
            )
            restore_info = restore_replay(
                online, SOURCE_RUN / "checkpoints/checkpoint_1000_online_replay.pkl"
            )
            handoff_checkpoint = save_bundle(
                HANDOFF, repaired.state, online, 1000, int(online.project_actor_steps), cfg,
                ROOT / cfg["demo"]["manifest"], "disabled-postbc-corrective-handoff-r1-c0",
                {"mechanism": "POST_BC_ON_POLICY_CORRECTIVE_SUPERVISION_HANDOFF",
                 "round1_checkpoint": str(round1_checkpoint),
                 "round1_checkpoint_sha256": sha(round1_checkpoint),
                 "source_replay": str(SOURCE_RUN / "checkpoints/checkpoint_1000_online_replay.pkl"),
                 "source_replay_sha256": restore_info["sha256"],
                 "actor_optimizer_reinitialized": True,
                 "other_optimizer_states_preserved": True,
                 "online_admission_threshold": THRESHOLD},
            )
        finally:
            handoff_env.close()
        handoff = {"checkpoint": str(handoff_checkpoint), "checkpoint_sha256": sha(handoff_checkpoint),
                   "replay": str(HANDOFF / "checkpoint_1000_online_replay.pkl"),
                   "replay_sha256": sha(HANDOFF / "checkpoint_1000_online_replay.pkl"),
                   "source_replay_size": restore_info["size"],
                   "actor_optimizer_reinitialized": True,
                   "corrective_bc_active": False}
        dump(OUT / "repaired_handoff.json", handoff)

        env = make_train_env("C0")
        try:
            fresh = make_combined_q_anchor_agent(
                cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg
            )
            agent = fresh.replace(state=restore_fresh_state(HANDOFF, fresh.state, 1000))
            online = make_replay_buffer(
                env, capacity=cfg["training"]["replay_capacity"], type="replay_buffer"
            )
            replay_restore = restore_replay(online, HANDOFF / "checkpoint_1000_online_replay.pkl")
            demo = make_replay_buffer(env, capacity=len(demos) + 100, type="replay_buffer")
            for transition in demos:
                demo.insert(transition)
            require(len(demo) == 1921 and len(online) == 604, "source replay sizes")
            initial_c0 = behavior_evaluate(agent, "C0", demo_obs, scale)
            dump(RETENTION / "initial_c0_evaluation.json", initial_c0)

            obs, reset_info = env.reset(seed=int(cfg["seed"]))
            schedule_sha = reset_info["curriculum_schedule_sha256"]
            actor_rng = jax.random.PRNGKey(int(cfg["seed"]) + 11000)
            sample_rng = np.random.default_rng(int(cfg["seed"]) + 20261908)
            attempted = admitted = rejected = near_attempts = 0
            contact_healthy = True
            max_force = 0.0
            completed = 1000
            evaluations = {}
            checkpoints = set(specification["retention_checkpoints"])

            for block_index in range(5):
                block_admitted = block_attempts = 0
                while block_admitted < int(specification["admitted_transitions_per_block"]):
                    require(block_attempts < int(cfg["training"]["maximum_attempts_per_block"]),
                            f"retention block {block_index} admission exhaustion")
                    act, actor_rng = action(agent, obs, deterministic=False, rng=actor_rng)
                    next_obs, reward, terminated, truncated, info = env.step(act)
                    distance = demo_distance(next_obs, demo_obs, scale)
                    supported = distance <= THRESHOLD
                    lateral, axial = float(info["lateral_error"]), float(info["axial_error"])
                    force = float(info.get("force_norm", 0.0))
                    link01, unexpected, pairs = contact_flags(env)
                    near = lateral <= 0.010 and axial <= 0.020
                    attempted += 1
                    block_attempts += 1
                    near_attempts += int(near)
                    max_force = max(max_force, force if np.isfinite(force) else math.inf)
                    contact_healthy &= not link01 and not unexpected and np.isfinite(force)
                    append(RETENTION / "collection.jsonl", {
                        "block": block_index, "attempt": attempted, "distance": distance,
                        "admitted": supported, "near_hole_like": near, "lateral": lateral,
                        "axial": axial, "orientation": float(info["orientation_error"]),
                        "force": force, "link01_contact": bool(link01),
                        "unexpected_contact": bool(unexpected), "contact_pairs": pairs,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                        "success": bool(info.get("is_success", False)),
                    })
                    if supported:
                        online.insert({"observations": np.asarray(obs, np.float32), "actions": act,
                                       "next_observations": np.asarray(next_obs, np.float32),
                                       "rewards": np.float32(reward),
                                       "masks": np.float32(0.0 if terminated else 1.0),
                                       "dones": bool(terminated or truncated)})
                        admitted += 1
                        block_admitted += 1
                        obs = next_obs
                    else:
                        rejected += 1
                    if terminated or truncated or not supported:
                        obs, reset_info = env.reset(seed=int(cfg["seed"]))
                        require(reset_info["curriculum_schedule_sha256"] == schedule_sha, "schedule drift")

                frozen_size = len(online)
                block_start = completed + 1
                block_end = 1000 + (block_index + 1) * 100
                for update in range(block_start, block_end + 1):
                    require(len(online) == frozen_size, "replay changed during train block")
                    oi = sample_rng.integers(0, frozen_size, size=32)
                    di = sample_rng.integers(0, len(demo), size=32)
                    batch = concat_rlpd_batches(fixed_batch(online, oi), fixed_batch(demo, di))
                    tick = time.perf_counter()
                    new_agent, info = agent.update_high_utd(batch, utd_ratio=1)
                    block((new_agent, info))
                    losses = finite_scalars(info)
                    agent = new_agent
                    completed = update
                    append(RETENTION / "updates.jsonl", {
                        "update": update, "block": block_index, "online_size": frozen_size,
                        "online_batch": 32, "demo_batch": 32, "utd": 1,
                        "corrective_bc_active": False, "losses_finite": True,
                        "seconds": time.perf_counter() - tick, "losses": losses,
                    })
                    if update in checkpoints:
                        saved = save_bundle(
                            checkpoint_dir, agent.state, online, update,
                            int(online.project_actor_steps), cfg, ROOT / cfg["demo"]["manifest"],
                            "disabled-postbc-corrective-retention-r1-c0",
                            {"mechanism": "UNCHANGED_C_PLUS_G_RETENTION",
                             "handoff_checkpoint": str(handoff_checkpoint),
                             "relative_retention_updates": update - 1000,
                             "corrective_bc_active": False,
                             "online_admission_threshold": THRESHOLD,
                             "demo_online": "32:32", "utd": 1,
                             "free_running_async": False},
                        )
                        evaluations[str(update)] = {
                            "checkpoint": str(saved), "checkpoint_sha256": sha(saved),
                            "c0": behavior_evaluate(agent, "C0", demo_obs, scale),
                        }
                        dump(RETENTION / f"evaluation_{update}.json", evaluations[str(update)])
                append(RETENTION / "blocks.jsonl", {
                    "block": block_index, "target_admitted": 60,
                    "block_attempts": block_attempts, "block_admitted": block_admitted,
                    "frozen_replay_size": frozen_size, "start_update": block_start,
                    "end_update": block_end,
                })

            require(completed == 1500, "exactly +500 retention updates")
            final_c0 = evaluations["1500"]["c0"]
            final_c3 = behavior_evaluate(agent, "C3", demo_obs, scale)
            first_exit = support_exit(agent, demo_obs, scale)
            gate = specification["retention_gate"]
            retention_checks = {
                "c0_success": int(final_c0["success_count"]) >= int(gate["minimum_c0_successes_of_3"]),
                "c0_near_hole": int(final_c0["near_hole_count"]) >= int(gate["require_near_hole_successes_of_3"]),
                "contact_health": bool(final_c0["contact_health"]),
                "no_table_ee_contact": not bool(final_c0["table_ee_contact"]),
                "local_support_healthy": first_exit is None or first_exit > 6,
            }
            retention_pass = all(retention_checks.values())
            result = {
                "status": "PASS" if retention_pass else "STOP",
                "CANONICAL_SAFETY_CHECK": "PASS", "ROUND2_SKIPPED": "YES_STRONG_ROUND1_PASS",
                "REPAIRED_HANDOFF_CHECKPOINT": str(handoff_checkpoint),
                "REPAIRED_HANDOFF_CHECKPOINT_SHA256": sha(handoff_checkpoint),
                "RLPD_RETENTION": "PASS" if retention_pass else "FAIL",
                "RLPD_RETENTION_UPDATES": 500, "RLPD_RETENTION_FINAL_UPDATE": 1500,
                "RLPD_RETENTION_FINAL_CHECKPOINT": evaluations["1500"]["checkpoint"],
                "RLPD_RETENTION_FINAL_CHECKPOINT_SHA256": evaluations["1500"]["checkpoint_sha256"],
                "RETENTION_C0_SUCCESS": f'{final_c0["success_count"]}/3',
                "RETENTION_C0_NEAR_HOLE": f'{final_c0["near_hole_count"]}/3',
                "RETENTION_C0_CONTACT_HEALTH": "PASS" if final_c0["contact_health"] and not final_c0["table_ee_contact"] else "FAIL",
                "RETENTION_FIRST_LOCAL_SUPPORT_EXIT_STEP": first_exit,
                "retention_checks": retention_checks, "checkpoint_evaluations": evaluations,
                "initial_c0": initial_c0, "final_c0": final_c0, "final_canonical_c3": final_c3,
                "attempted_online_transitions": attempted, "admitted_online_transitions": admitted,
                "rejected_online_transitions": rejected,
                "ADMITTED_ONLINE_FRACTION": admitted / attempted,
                "ONLINE_NEAR_HOLE_LIKE_FRACTION": near_attempts / attempted,
                "final_replay_size": len(online), "collection_max_force": max_force,
                "collection_contact_health": "PASS" if contact_healthy else "FAIL",
                "C_PLUS_G_ONLY": "YES", "CORRECTIVE_BC_ACTIVE_DURING_RLPD": "NO",
                "demo_online": "32:32", "batch": 64, "utd": 1,
                "ONLINE_ADMISSION_THRESHOLD": THRESHOLD,
                "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
                "EVAL_RESET_CANONICAL": "YES", "EVAL_GOAL_CANONICAL": "YES",
                "EVAL_CURRICULUM_ACTIVE": "NO",
                "PRODUCTION_DEMO_SHA_UNCHANGED": "YES" if sha(DEMO) == DEMO_SHA else "NO",
                "PROTECTED_ARTIFACTS_CHANGED": "NO" if before == protected() else "YES",
                "TRAINING_RUNNING": False, "RESIDUAL_PROCESSES": "NONE",
                "elapsed_seconds": time.time() - started,
                "PRIMARY_ROOT_CAUSE": ("C0_REPAIRED_BY_POST_BC_ON_POLICY_CORRECTIVE_SUPERVISION_AND_RETAINED"
                                       if retention_pass else "LOCAL_ACTOR_SUPERVISION_NOT_RETAINED_DURING_RLPD"),
                "NEXT": "C1_EXACT_RESTORE_CURRICULUM" if retention_pass else "STOP",
            }
            require(result["PROTECTED_ARTIFACTS_CHANGED"] == "NO", "protected artifact changed")
            dump(OUT / "result.json", result)
            dump(OUT / "protected_after.json", protected())
            print(json.dumps({key: result[key] for key in (
                "status", "CANONICAL_SAFETY_CHECK", "RLPD_RETENTION",
                "RETENTION_C0_SUCCESS", "RETENTION_C0_NEAR_HOLE",
                "RETENTION_C0_CONTACT_HEALTH", "RETENTION_FIRST_LOCAL_SUPPORT_EXIT_STEP",
                "ADMITTED_ONLINE_FRACTION", "ONLINE_NEAR_HOLE_LIKE_FRACTION",
                "RLPD_RETENTION_FINAL_CHECKPOINT", "PRIMARY_ROOT_CAUSE", "NEXT",
            )}, indent=2, sort_keys=True))
        finally:
            env.close()


if __name__ == "__main__":
    main()
