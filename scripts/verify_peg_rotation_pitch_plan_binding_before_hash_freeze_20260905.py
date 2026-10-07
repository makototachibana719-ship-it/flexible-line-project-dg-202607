"""Read-only canonical verifier for the frozen local-pitch pilot plan."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion.v2_recorder import PHASES

SOURCE_RELATIVE = Path(
    "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
)
SOURCE_SHA256 = "c1e66e2e6e7a43202d2b140162a4ac9e9c6aca1937da077f76c9865bb24f9b4f"
ROLL_RELATIVE = Path("results/validation/peg/f3_rotation_roll_command_plan_20260905/command_plan.json")
ROLL_SHA256 = "2c5e9f3ec9a165cab794a883b1a549e87248eb152d41bba0b29c25d698847e3d"
PLAN_RELATIVE = Path("results/validation/peg/f3_rotation_pitch_command_plan_20260905/command_plan.json")
MANIFEST_RELATIVE = PLAN_RELATIVE.with_name("manifest.json")
EXPECTED_PLAN_SHA256 = "__TO_BE_FROZEN__"
EXPECTED_COUNT = 179
PREFIX_COUNT = 106
ROTATION_SCALE_RAD = 0.017453292519943295


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


def main():
    source_path = ROOT / SOURCE_RELATIVE
    roll_path = ROOT / ROLL_RELATIVE
    plan_path = ROOT / PLAN_RELATIVE
    manifest_path = ROOT / MANIFEST_RELATIVE
    require(source_path.is_file() and digest(source_path) == SOURCE_SHA256, "standard source binding mismatch")
    require(roll_path.is_file() and digest(roll_path) == ROLL_SHA256, "roll structure binding mismatch")
    require(plan_path.is_file() and manifest_path.is_file(), "pitch artifact missing")
    require(EXPECTED_PLAN_SHA256 != "__TO_BE_FROZEN__", "verifier hash is not frozen")
    require(digest(plan_path) == EXPECTED_PLAN_SHA256, "pitch plan hash mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("command_plan_sha256") == EXPECTED_PLAN_SHA256, "manifest hash mismatch")
    require(manifest.get("commands") == EXPECTED_COUNT, "manifest count mismatch")
    require(manifest.get("source_trace_sha256") == SOURCE_SHA256, "manifest source mismatch")
    require(manifest.get("structure_source_sha256") == ROLL_SHA256, "manifest roll binding mismatch")

    source_rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    source = [{"phase": row["phase"], "action": row["action"]} for row in source_rows]
    roll = json.loads(roll_path.read_bytes())["commands"]
    plan = json.loads(plan_path.read_bytes())
    commands = plan.get("commands")
    require(plan.get("status") == "PASS" and plan.get("episode_type") == "orientation_pitch", "invalid plan envelope")
    require(plan.get("structure_source_sha256") == ROLL_SHA256, "plan roll binding mismatch")
    require(isinstance(commands, list) and len(commands) == EXPECTED_COUNT, "plan count mismatch")
    for index, command in enumerate(commands, 1):
        validate(command, index)
    require(commands[:PREFIX_COUNT] == source[:PREFIX_COUNT] == roll[:PREFIX_COUNT], "verified prefix changed")
    require(commands[PREFIX_COUNT + 14:] == source[PREFIX_COUNT:] == roll[PREFIX_COUNT + 14:], "verified suffix changed")

    challenge = commands[106:110]
    correction = commands[110:115]
    settle = commands[115:120]
    require(all(x == {"phase": "orientation_correction", "action": [0, 0, 0, 0, 1.0, 0, -1]} for x in challenge),
            "pitch challenge mismatch")
    require(all(x == {"phase": "orientation_correction", "action": [0, 0, 0, 0, -0.8, 0, -1]} for x in correction),
            "pitch correction mismatch")
    require(all(x == {"phase": "orientation_correction", "action": [0, 0, 0, 0, 0, 0, -1]} for x in settle),
            "pitch settle mismatch")
    require(all(c["phase"] == r["phase"] for c, r in zip(commands, roll)), "roll phase structure changed")
    peak = len(challenge) * ROTATION_SCALE_RAD
    recovered = sum(x["action"][4] for x in correction) * ROTATION_SCALE_RAD
    require(0.06 <= peak <= 0.08, "planned pitch peak outside frozen range")
    require(len(correction) >= 5 and all(abs(x["action"][4]) >= 0.02 for x in correction), "correction excitation insufficient")
    require(math.isclose(peak + recovered, 0.0, abs_tol=1e-15), "correction does not cancel challenge")
    print(json.dumps({
        "status": "PASS",
        "path": str(plan_path),
        "sha256": EXPECTED_PLAN_SHA256,
        "commands": len(commands),
        "phase_validation": "PASS",
        "action_validation": "PASS",
        "rotation_challenge_plan": "PASS",
        "rotation_correction_plan": "PASS",
        "planned_peak_rad": peak,
        "correction_transitions": len(correction),
        "simulator_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
