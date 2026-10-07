"""Calibrate pitch response from _001 and freeze the minimal revised plan."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion.v2_recorder import PHASES

OLD_PLAN_RELATIVE = Path("results/validation/peg/f3_rotation_pitch_command_plan_20260905/command_plan.json")
OLD_PLAN_SHA256 = "9ce4e9583563c900b75d977b7bf450b29f63c298a26193748afdb9bf95bcc1d9"
TRACE_RELATIVE = Path("results/demos/peg_v2_pilot/rotation_pitch_post_exclusion_20260905_001/transitions.jsonl")
TRACE_SHA256 = "e5364818f23fde9a8a06b4c413e24573fe23cc25fc2371900c28dc3dc44762e6"
CONTACT_RELATIVE = Path("results/diagnostics/peg/f3_v2_rotation_pitch_contact_20260905_001/contacts.jsonl")
CONTACT_SHA256 = "a42c3766f01fb90b231441834f28de5550cd9b9dfb194e87e05e4925e2564be9"
OBSERVER_RELATIVE = Path("results/diagnostics/peg/f3_v2_rotation_pitch_contact_20260905_001/telemetry/steps.jsonl")
OBSERVER_SHA256 = "e3c7cd59ce7a2962a975206951b1d4adea108d78c82c91422f73a34d79718ebc"
OUTPUT_RELATIVE = Path("results/validation/peg/f3_rotation_pitch_revised_plan_20260906")
ROTATION_SCALE_RAD = 0.017453292519943295
PREFIX_COUNT = 106
OLD_CHALLENGE_COUNT = 4
REVISED_CHALLENGE_COUNT = 5
REVISED_CORRECTION = [-0.8] * 6 + [-0.2]
SETTLE_COUNT = 5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(command, index):
    require(set(command) == {"phase", "action"}, f"extra/missing fields at {index}")
    require(command["phase"] in PHASES, f"unknown phase at {index}")
    action = command["action"]
    require(isinstance(action, list) and len(action) == 7, f"action dimension at {index}")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in action), f"nonfinite action at {index}")
    require(all(-1 <= x <= 1 for x in action), f"action bounds at {index}")


def quat_mat(value):
    w, x, y, z = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    require(norm > 1e-12, "invalid quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def matrix_rpy(matrix):
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1, 1)))
    roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return [roll, pitch, yaw]


def durable_exclusive_json(path, value):
    raw = (json.dumps(value, allow_nan=False, indent=2) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    old_path = ROOT / OLD_PLAN_RELATIVE
    trace_path = ROOT / TRACE_RELATIVE
    contact_path = ROOT / CONTACT_RELATIVE
    observer_path = ROOT / OBSERVER_RELATIVE
    output_dir = ROOT / OUTPUT_RELATIVE
    for path, expected, label in (
        (old_path, OLD_PLAN_SHA256, "old plan"),
        (trace_path, TRACE_SHA256, "pitch _001 trace"),
        (contact_path, CONTACT_SHA256, "pitch _001 contacts"),
        (observer_path, OBSERVER_SHA256, "pitch _001 observer telemetry"),
    ):
        require(path.is_file() and digest(path) == expected, f"{label} binding mismatch")
    require(not output_dir.exists(), "exclusive revised-plan output already exists")

    old = json.loads(old_path.read_bytes())
    old_commands = old["commands"]
    require(len(old_commands) == 179, "old plan count mismatch")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    contacts = [json.loads(line) for line in contact_path.read_text().splitlines() if line]
    observer = [json.loads(line) for line in observer_path.read_text().splitlines() if line]
    require(len(rows) == len(contacts) == len(observer) == 179, "_001 evidence length mismatch")
    require(all(row.get("complete") and not row.get("observer_errors") and not row.get("step_exception") for row in observer),
            "observer telemetry incomplete")

    challenge_rows = rows[PREFIX_COUNT:PREFIX_COUNT + OLD_CHALLENGE_COUNT]
    require(all(row["action"][4] == 1.0 for row in challenge_rows), "_001 pitch challenge actions mismatch")
    errors = [float(row["after"]["metrics"]["orientation_error"]) for row in challenge_rows]
    before_error = float(rows[PREFIX_COUNT - 1]["after"]["metrics"]["orientation_error"])
    increments = [errors[0] - before_error] + [b - a for a, b in zip(errors, errors[1:])]
    require(all(math.isfinite(x) and x > 0 for x in increments), "pitch response is not finite/monotonic")
    require(all(float(row["after"]["metrics"]["lateral_error"]) < 0.002 for row in challenge_rows),
            "unexpected challenge lateral behavior")
    require(not any(row["link01_contact"] or row["unexpected_contact"] or row["contacts"] for row in contacts[PREFIX_COUNT:PREFIX_COUNT + OLD_CHALLENGE_COUNT]),
            "unexpected challenge contact")

    base_actual = quat_mat(rows[PREFIX_COUNT - 1]["after"]["actual_ee_pose"][3:7])
    base_target = quat_mat(rows[PREFIX_COUNT - 1]["after"]["target_pose"][3:7])
    calibration_steps = []
    for offset, row in enumerate(challenge_rows, 1):
        after = row["after"]
        actual_rpy = matrix_rpy(base_actual.T @ quat_mat(after["actual_ee_pose"][3:7]))
        target_rpy = matrix_rpy(base_target.T @ quat_mat(after["target_pose"][3:7]))
        values = actual_rpy + target_rpy
        require(all(math.isfinite(x) for x in values), "nonfinite RPY response")
        calibration_steps.append({
            "step": PREFIX_COUNT + offset,
            "action_pitch": row["action"][4],
            "orientation_error": errors[offset - 1],
            "orientation_increment": increments[offset - 1],
            "actual_relative_rpy": actual_rpy,
            "target_relative_rpy": target_rpy,
            "lateral_error": float(after["metrics"]["lateral_error"]),
            "axial_error": float(after["metrics"]["axial_error"]),
        })

    old_planned_peak = float(old["rotation"]["planned_peak_rad"])
    old_realized_peak = max(errors)
    ratio = old_realized_peak / old_planned_peak
    revised_planned_peak = REVISED_CHALLENGE_COUNT * ROTATION_SCALE_RAD
    revised_expected_peak = revised_planned_peak * ratio
    require(0.06 <= revised_expected_peak <= 0.08, "minimal revised challenge is not evidenced inside protocol")
    require((OLD_CHALLENGE_COUNT * ROTATION_SCALE_RAD) * ratio < 0.06, "old count unexpectedly sufficient")

    challenge = [{"phase": "orientation_correction", "action": [0, 0, 0, 0, 1.0, 0, -1]}
                 for _ in range(REVISED_CHALLENGE_COUNT)]
    correction = [{"phase": "orientation_correction", "action": [0, 0, 0, 0, value, 0, -1]}
                  for value in REVISED_CORRECTION]
    settle = [{"phase": "orientation_correction", "action": [0, 0, 0, 0, 0, 0, -1]}
              for _ in range(SETTLE_COUNT)]
    suffix = old_commands[120:]
    commands = old_commands[:PREFIX_COUNT] + challenge + correction + settle + suffix
    require(len(commands) == 182, "revised plan count mismatch")
    for index, command in enumerate(commands, 1):
        validate(command, index)
    require(commands[:PREFIX_COUNT] == old_commands[:PREFIX_COUNT], "standard prefix changed")
    require(commands[-len(suffix):] == suffix, "standard suffix changed")
    require(math.isclose(sum(x["action"][4] for x in challenge + correction), 0.0, abs_tol=1e-15),
            "revised correction does not cancel challenge")

    plan = {
        "status": "PASS",
        "protocol_version": "PEG_V2_PILOT_1",
        "episode_type": "orientation_pitch",
        "revision": "challenge_calibrated_v2",
        "old_plan": OLD_PLAN_RELATIVE.as_posix(),
        "old_plan_sha256": OLD_PLAN_SHA256,
        "calibration_trace": TRACE_RELATIVE.as_posix(),
        "calibration_trace_sha256": TRACE_SHA256,
        "structure": {
            "standard_prefix_commands": PREFIX_COUNT,
            "challenge_commands": list(range(107, 112)),
            "correction_commands": list(range(112, 119)),
            "settle_commands": list(range(119, 124)),
            "standard_suffix_start": 124,
        },
        "calibration": {
            "steps": calibration_steps,
            "response_monotonic": True,
            "old_planned_peak_rad": old_planned_peak,
            "old_realized_peak_rad": old_realized_peak,
            "realized_to_planned_ratio": ratio,
            "orientation_growth_rad": errors[-1] - before_error,
            "mean_error_growth_per_step_rad": float(np.mean(increments)),
            "revised_expected_peak_rad": revised_expected_peak,
        },
        "rotation": {
            "axis": "local_pitch/action[4]",
            "rotation_scale_rad": ROTATION_SCALE_RAD,
            "challenge_action": 1.0,
            "challenge_count": REVISED_CHALLENGE_COUNT,
            "revised_planned_peak_rad": revised_planned_peak,
            "correction_actions": REVISED_CORRECTION,
            "correction_count": len(REVISED_CORRECTION),
            "net_planned_rotation_rad": 0.0,
        },
        "command_count": len(commands),
        "commands": commands,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "command_plan_v2.json"
    durable_exclusive_json(plan_path, plan)
    plan_hash = digest(plan_path)
    manifest = {
        "status": "PASS",
        "artifact": "F3_V2_ROTATION_PITCH_REVISED_COMMAND_PLAN",
        "command_plan": plan_path.name,
        "command_plan_sha256": plan_hash,
        "commands": len(commands),
        "old_plan_sha256": OLD_PLAN_SHA256,
        "calibration_trace_sha256": TRACE_SHA256,
        "contact_evidence_sha256": CONTACT_SHA256,
        "observer_telemetry_sha256": OBSERVER_SHA256,
        "old_planned_peak_rad": old_planned_peak,
        "old_realized_peak_rad": old_realized_peak,
        "realized_to_planned_ratio": ratio,
        "revised_challenge_steps": REVISED_CHALLENGE_COUNT,
        "revised_correction_steps": len(REVISED_CORRECTION),
        "revised_expected_peak_rad": revised_expected_peak,
        "simulator_started": False,
    }
    durable_exclusive_json(output_dir / "manifest.json", manifest)
    descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"status": "PASS", "path": str(plan_path), "sha256": plan_hash, "commands": len(commands)}))


if __name__ == "__main__":
    main()
