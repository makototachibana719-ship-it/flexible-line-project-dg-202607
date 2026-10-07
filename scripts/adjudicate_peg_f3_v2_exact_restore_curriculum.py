#!/usr/bin/env python3
"""Fail-stop adjudication for exact-restore curriculum training."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import sys

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

import scripts.run_peg_f3_v2_exact_restore_curriculum_stage as stage
from training.peg_insertion.checkpoint import restore_fresh_state


PACKAGE = ROOT / "results/training/peg_f3_v2_exact_restore_curriculum_rlpd_20260907_001"
R1 = PACKAGE / "replica1/C0_retry002/result.json"
R2 = PACKAGE / "replica2/C0/result.json"
DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD = 7.41773975814
PROTECTED = {
    DEMO: DEMO_SHA,
    ROOT / "training/peg_insertion/exact_state_curriculum.py": "a050c29a6158ee7fb79e3dcb8f9e539fb247da7fe67a2309b8f970e0f7611b37",
    ROOT / "scripts/close_peg_f3_v2_exact_state_restore.py": "d6c719fdd87bac39b62e0d70fcc277c55eceb206b27e50cf5a03bdc3c4c680e6",
    ROOT / "scripts/preflight_peg_f3_v2_exact_state_curriculum.py": "b76a25726336709745c821af8c8e9bb0b624ca64a7790ae4c66944e445b53750",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def localize_first_contact() -> dict:
    cfg = stage.load_config(1)
    env = stage.make_train_env("C0")
    try:
        agent = stage.make_combined_q_anchor_agent(
            cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg
        )
        checkpoint_dir = PACKAGE / "replica1/C0_retry002/checkpoints"
        agent = agent.replace(state=restore_fresh_state(checkpoint_dir, agent.state, 1000))
        observation, _ = env.reset(seed=100)
        for step_index in range(1, 501):
            policy_action, _ = stage.action(agent, observation, deterministic=True)
            observation, _reward, terminated, truncated, info = env.step(policy_action)
            link01, unexpected, pairs = stage.contact_flags(env)
            if link01 or unexpected:
                return {
                    "seed": 100,
                    "step": step_index,
                    "link01_contact": bool(link01),
                    "unexpected_contact": bool(unexpected),
                    "contact_pairs": pairs,
                    "force": float(info.get("force_norm", 0.0)),
                }
            if terminated or truncated:
                break
    finally:
        env.close()
    return {"contact_found": False}


def main() -> None:
    require(Path.cwd() == ROOT, "run from canonical project root")
    require(all(sha(path) == expected for path, expected in PROTECTED.items()), "protected precheck")
    preflight = json.loads((PACKAGE / "package_preflight.json").read_text())
    r1 = json.loads(R1.read_text())
    require(preflight["CURRICULUM_PACKAGE_PREFLIGHT"] == "PASS", "package preflight")
    require(r1["status"] == "PASS" and r1["final_update"] == 1000, "R1 engineering completion")
    require(r1["EVAL_RESET_CANONICAL"] == r1["EVAL_GOAL_CANONICAL"] == "YES", "canonical eval identity")
    require(r1["EVAL_CURRICULUM_ACTIVE"] == "NO", "canonical eval curriculum separation")

    contact = localize_first_contact()
    require(contact.get("contact_pairs") == [["table", "end_effector"]], "unexpected contact localization")
    (PACKAGE / "c0_r1_contact_localization.json").write_text(
        json.dumps(contact, indent=2, sort_keys=True) + "\n"
    )
    initial = r1["initial_evaluation"]
    final = r1["final_training_level_evaluation"]
    behavior_improved = (
        final["median_corridor_steps"] > initial["median_corridor_steps"]
        and final["median_min_lateral"] < initial["median_min_lateral"]
        and final["median_min_axial"] < initial["median_min_axial"]
    )
    r1_gate = (
        behavior_improved
        and r1["admitted_online_transitions"] > 0
        and final["contact_health"]
        and r1["contact_health"] == "PASS"
    )
    r2_gate = False
    c0_pass = r1_gate and r2_gate
    require(not c0_pass, "adjudicator is only for the observed C0 fail-stop")

    result = {
        "EXACT_STATE_RESTORE_REVIEW": "PASS",
        "C0_SNAPSHOT_COUNT": preflight["C0_SNAPSHOT_COUNT"],
        "C1_SNAPSHOT_COUNT": preflight["C1_SNAPSHOT_COUNT"],
        "C2_SNAPSHOT_COUNT": preflight["C2_SNAPSHOT_COUNT"],
        "CURRICULUM_SCHEDULE_SHA256": preflight["CURRICULUM_SCHEDULE_SHA256"],
        "C0_TRAINING": "FAIL",
        "C1_TRAINING": "NOT_RUN",
        "C2_TRAINING": "NOT_RUN",
        "C3_TRAINING": "NOT_RUN",
        "C0_R1_ENGINEERING_RUN": "PASS_1000_UPDATES",
        "C0_R1_BEHAVIOR_IMPROVED": "YES" if behavior_improved else "NO",
        "C0_R1_GATE": "FAIL_CONTACT_HEALTH",
        "C0_R2": "NOT_RUN_FAIL_STOP_R1_ALREADY_PREVENTS_BOTH_REPLICAS_GATE",
        "FIRST_BAD_CURRICULUM_LEVEL": "C0",
        "TRAIN_FULL_RESET_READY": "NO",
        "C3_1K": "NOT_RUN",
        "C3_1K_CONFIRMATION": "NOT_RUN",
        "C3_3K": "NOT_RUN",
        "FORMAL_5K_PREFLIGHT": "NOT_RUN",
        "FORMAL_5K": "NOT_RUN",
        "R1_FINAL_REPLAY": r1["final_replay"],
        "R2_FINAL_REPLAY": "NOT_RUN",
        "ADMITTED_ONLINE_FRACTION": r1["ADMITTED_ONLINE_FRACTION"],
        "ONLINE_DEMO_SUPPORT_FRACTION": r1["ONLINE_DEMO_SUPPORT_FRACTION"],
        "ONLINE_NEAR_HOLE_LIKE_FRACTION": r1["ONLINE_NEAR_HOLE_LIKE_FRACTION"],
        "FIRST_DIVERGENCE": final["median_first_divergence"],
        "CORRIDOR": final["median_corridor_steps"],
        "MIN_LATERAL": final["median_min_lateral"],
        "MIN_AXIAL": final["median_min_axial"],
        "ORIENTATION": final["median_min_orientation"],
        "MAX_FORCE": final["max_force"],
        "TRAIN_COLLECTION_MAX_FORCE": r1["max_force"],
        "CONTACT_HEALTH": "FAIL_TABLE_END_EFFECTOR_CONTACT",
        "FIRST_UNEXPECTED_CONTACT": contact,
        "SUCCESS_RATE": final["success_rate"],
        "NEAR_HOLE_RATE": final["near_hole_rate"],
        "BEST_CHECKPOINT": "NONE_C0_GATE_FAILED",
        "PEG_SINGLE_TASK_CONVERGED": "NO",
        "SUCCESSFUL_INSERTION": "NO",
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES" if sha(DEMO) == DEMO_SHA else "NO",
        "ONLINE_ADMISSION_THRESHOLD": THRESHOLD,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "EVAL_RESET_CANONICAL": "YES",
        "EVAL_GOAL_CANONICAL": "YES",
        "EVAL_CURRICULUM_ACTIVE": "NO",
        "PROTECTED_ARTIFACTS_CHANGED": "NO" if all(
            sha(path) == expected for path, expected in PROTECTED.items()
        ) else "YES",
        "PRIMARY_ROOT_CAUSE": "LOCAL_INSERTION_POLICY_LEARNING_FAILURE",
        "DETAIL_ROOT_CAUSE": "C0_POLICY_IMPROVED_GEOMETRIC_ERRORS_BUT_DID_NOT_INSERT_AND_DEVELOPED_TABLE_END_EFFECTOR_COLLISION",
        "NEXT": "STOP_PRESERVE_EVIDENCE_NO_C1_C2_C3_3K_5K",
    }
    (PACKAGE / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = "# Peg F3 V2 Exact-Restore Curriculum RLPD Closure\n\n```json\n" + json.dumps(result, indent=2, sort_keys=True) + "\n```\n"
    (PACKAGE / "REPORT.md").write_text(report)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
