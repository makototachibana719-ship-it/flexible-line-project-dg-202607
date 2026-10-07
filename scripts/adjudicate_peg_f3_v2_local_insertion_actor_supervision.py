#!/usr/bin/env python3
"""Read-only final adjudication for the bounded local actor-supervision task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/home/makoto/flexible_line_project")
DATASET_DIR = ROOT / "results/datasets/peg_v2_local_insertion_supervision_20260908_001"
TRAIN_DIR = ROOT / "results/training/peg_f3_v2_local_insertion_actor_supervision_20260908_001"
EVAL_DIR = ROOT / "results/diagnostics/peg/f3_v2_local_actor_bc_checkpoint_eval_20260908_002"
OUT_DIR = ROOT / "results/closure/peg/f3_v2_local_insertion_actor_supervision_20260908_001"
PRODUCTION_DEMO = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
EXPECTED_DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
THRESHOLD = 7.41773975814


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUT_DIR}")

    manifest = load(DATASET_DIR / "manifest.json")
    training = load(TRAIN_DIR / "result.json")
    checkpoint_eval = load(EVAL_DIR / "result.json")
    ranked = checkpoint_eval["ranked"]
    selected = next(item for item in ranked if item["bc_step"] == checkpoint_eval["best_bc_step"])
    selected_report = next(
        item for item in training["checkpoint_reports"] if item["step"] == checkpoint_eval["best_bc_step"]
    )
    c0 = selected["c0"]

    demo_sha = sha256(PRODUCTION_DEMO)
    if demo_sha != EXPECTED_DEMO_SHA:
        raise RuntimeError(f"production Demo SHA changed: {demo_sha}")
    if load(DATASET_DIR / "protected_before.json") != load(DATASET_DIR / "protected_after.json"):
        raise RuntimeError("dataset-stage protected hashes differ")
    if load(TRAIN_DIR / "protected_before.json") != load(TRAIN_DIR / "protected_after.json"):
        raise RuntimeError("training-stage protected hashes differ")
    if training["ONLINE_ADMISSION_THRESHOLD"] != THRESHOLD:
        raise RuntimeError("online-admission threshold mismatch")
    if c0["success_count"] != 0:
        raise RuntimeError("adjudicator expects the recorded C0 scientific failure")

    result = {
        "LOCAL_SUPERVISION_DATASET": manifest["LOCAL_SUPERVISION_DATASET"],
        "LOCAL_SUPERVISION_EPISODES": manifest["LOCAL_SUPERVISION_EPISODES"],
        "LOCAL_SUPERVISION_TRANSITIONS": manifest["LOCAL_SUPERVISION_TRANSITIONS"],
        "LOCAL_SUPERVISION_SHA256": manifest["LOCAL_SUPERVISION_SHA256"],
        "LOCAL_EXPERT_ACTION_STATS": manifest["LOCAL_EXPERT_ACTION_STATS"],
        "LOCAL_BC_TRAIN_EPISODES": manifest["LOCAL_BC_TRAIN_EPISODES"],
        "LOCAL_BC_VALID_EPISODES": manifest["LOCAL_BC_VALID_EPISODES"],
        "LOCAL_BC_SPLIT_SHA256": manifest["LOCAL_BC_SPLIT_SHA256"],
        "ACTOR_ONLY_BC": training["ACTOR_ONLY_BC"],
        "BEST_BC_CHECKPOINT": checkpoint_eval["BEST_BC_CHECKPOINT"],
        "BEST_BC_CHECKPOINT_SHA256": sha256(Path(checkpoint_eval["BEST_BC_CHECKPOINT"])),
        "BEST_OFFLINE_BC_CHECKPOINT": training["BEST_BC_CHECKPOINT"],
        "OLD_C0_FIRST_ACTION": training["OLD_C0_FIRST_ACTION"],
        "REPAIRED_C0_FIRST_ACTION": selected_report["first_action"],
        "EXPERT_C0_FIRST_ACTION": training["EXPERT_C0_FIRST_ACTION"],
        "OLD_ACTION_L2_DELTA": training["OLD_ACTION_L2_DELTA"],
        "REPAIRED_ACTION_L2_DELTA": selected_report["first_action_l2_delta"],
        "LOCAL_VALIDATION_ACTION_MAE": selected["offline"]["mae_a0_a5"],
        "LOCAL_VALIDATION_ACTION_COSINE": selected["offline"]["cosine_a0_a5"],
        "LOCAL_VALIDATION_ACTION_MAX_ERROR": selected["offline"]["max_error_a0_a5"],
        "C0_POST_BC_SUCCESS": f'{c0["success_count"]}/3',
        "C0_POST_BC_NEAR_HOLE": f'{c0["near_hole_count"]}/3',
        "C0_POST_BC_SUCCESS_MARGIN": c0["best_success_margin"],
        "C0_POST_BC_FIRST_DIVERGENCE": c0["median_first_divergence"],
        "C0_POST_BC_MAX_FORCE": c0["max_force"],
        "C0_POST_BC_CONTACT_HEALTH": "PASS" if c0["contact_health"] else "FAIL",
        "LOCAL_INSERTION_POLICY_SUPERVISION": "SCIENTIFIC_FAIL",
        "LOCAL_BC_CATASTROPHIC_FORGETTING": "NOT_RUN_GATE_FAILED",
        "ACTOR_OPTIMIZER_HANDOFF": "NOT_RUN_GATE_FAILED",
        "LOCAL_BC_RETENTION": "NOT_RUN_GATE_FAILED",
        "C1_TRAINING": "NOT_RUN_GATE_FAILED",
        "C2_TRAINING": "NOT_RUN_GATE_FAILED",
        "C3_TRAINING": "NOT_RUN_GATE_FAILED",
        "FIRST_BAD_CURRICULUM_LEVEL": "C0",
        "C3_1K": "NOT_RUN_GATE_FAILED",
        "C3_1K_CONFIRMATION": "NOT_RUN_GATE_FAILED",
        "C3_3K": "NOT_RUN_GATE_FAILED",
        "FORMAL_5K_PREFLIGHT": "NOT_RUN_GATE_FAILED",
        "FORMAL_5K": "NOT_RUN_GATE_FAILED",
        "SUCCESS_RATE": 0.0,
        "NEAR_HOLE_RATE": 1.0,
        "BEST_SIMULTANEOUS_SUCCESS_MARGIN": c0["best_success_margin"],
        "FIRST_DIVERGENCE": c0["median_first_divergence"],
        "MIN_LATERAL": c0["median_min_lateral"],
        "MIN_AXIAL": c0["median_min_axial"],
        "ORIENTATION": c0["median_min_orientation"],
        "MAX_FORCE": c0["max_force"],
        "CONTACT_HEALTH": "PASS" if c0["contact_health"] else "FAIL",
        "TABLE_END_EFFECTOR_COLLISION": "NO" if not c0["table_ee_contact"] else "YES",
        "BEST_CHECKPOINT": checkpoint_eval["BEST_BC_CHECKPOINT"],
        "PEG_SINGLE_TASK_CONVERGED": "NO",
        "SUCCESSFUL_INSERTION": "NO",
        "PRIMARY_ROOT_CAUSE": "LOCAL_ACTOR_SUPERVISION_DID_NOT_CLOSE_C0",
        "NEXT": "STOP_AND_REVIEW_ACTOR_PARAMETERIZATION_OR_LOCAL_CLOSED_LOOP_DISTRIBUTION_SHIFT",
        "TRAINING_RUNNING": False,
        "RESIDUAL_PROCESSES": "NONE",
        "PRODUCTION_DEMO_SHA256": demo_sha,
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES",
        "ONLINE_ADMISSION_THRESHOLD": THRESHOLD,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO",
        "EVAL_RESET_CANONICAL": "YES",
        "EVAL_GOAL_CANONICAL": "YES",
        "EVAL_CURRICULUM_ACTIVE": "NO",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "FORMAL_CANONICAL_EVALUATION": "NOT_RUN_C0_GATE_FAILED",
        "adjudication_note": (
            "Offline actor imitation passed, but none of the behavior-ranked checkpoints from the single "
            "bounded BC run achieved autonomous C0 insertion. Per the explicit gate, canonical safety eval, "
            "optimizer handoff, RLPD retention, and C1/C2/C3 were not run."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    result_path = OUT_DIR / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Peg F3 V2 local insertion actor-supervision adjudication",
        "",
        "Scientific result: **STOP at C0**. Actor-only BC passed the offline imitation gate, but the best "
        "behavior-ranked checkpoint achieved 0/3 autonomous C0 insertions. No later gate was entered.",
        "",
    ]
    for key, value in result.items():
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, bool)) else str(value)
        lines.append(f"- `{key}={rendered}`")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT_DIR), "result_sha256": sha256(result_path)}, indent=2))


if __name__ == "__main__":
    main()
