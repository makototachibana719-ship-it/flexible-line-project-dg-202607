"""Build the frozen local-pitch pilot plan from verified standard/roll structure."""
import hashlib
import json
import math
import os
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
PROTOCOL_RELATIVE = Path("results/validation/peg/f3_data_review_20260904/v2_pilot_protocol.json")
OUTPUT_RELATIVE = Path("results/validation/peg/f3_rotation_pitch_command_plan_20260905")
PREFIX_COUNT = 106
ROTATION_SCALE_RAD = 0.017453292519943295
CHALLENGE_COUNT = 4
CHALLENGE_VALUE = 1.0
CORRECTION_COUNT = 5
CORRECTION_VALUE = -0.8
SETTLE_COUNT = 5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(command, index):
    require(set(command) == {"phase", "action"}, f"extra/missing fields at {index}")
    require(command["phase"] in PHASES, f"unknown phase at {index}")
    action = command["action"]
    require(isinstance(action, list) and len(action) == 7, f"action dimension at {index}")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in action), f"nonfinite action at {index}")
    require(all(-1 <= x <= 1 for x in action), f"action bounds at {index}")


def durable_exclusive_json(path, value):
    raw = (json.dumps(value, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    source_path = ROOT / SOURCE_RELATIVE
    roll_path = ROOT / ROLL_RELATIVE
    protocol_path = ROOT / PROTOCOL_RELATIVE
    output_dir = ROOT / OUTPUT_RELATIVE
    require(source_path.is_file() and sha256(source_path) == SOURCE_SHA256, "accepted standard trace binding mismatch")
    require(roll_path.is_file() and sha256(roll_path) == ROLL_SHA256, "accepted roll structure binding mismatch")
    require(protocol_path.is_file(), "frozen V2 protocol missing")
    require(not output_dir.exists(), "exclusive output already exists")

    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    require(len(rows) == 165, "standard trace length mismatch")
    source = [{"phase": row["phase"], "action": row["action"]} for row in rows]
    roll = json.loads(roll_path.read_bytes())["commands"]
    require(len(roll) == 179, "roll structure length mismatch")
    require(roll[:PREFIX_COUNT] == source[:PREFIX_COUNT], "roll prefix differs from standard")
    require(roll[PREFIX_COUNT + 14:] == source[PREFIX_COUNT:], "roll suffix differs from standard")

    challenge = [
        {"phase": "orientation_correction", "action": [0, 0, 0, 0, CHALLENGE_VALUE, 0, -1]}
        for _ in range(CHALLENGE_COUNT)
    ]
    correction = [
        {"phase": "orientation_correction", "action": [0, 0, 0, 0, CORRECTION_VALUE, 0, -1]}
        for _ in range(CORRECTION_COUNT)
    ]
    settle = [
        {"phase": "orientation_correction", "action": [0, 0, 0, 0, 0, 0, -1]}
        for _ in range(SETTLE_COUNT)
    ]
    commands = source[:PREFIX_COUNT] + challenge + correction + settle + source[PREFIX_COUNT:]
    require(len(commands) == 179, "pitch plan length mismatch")
    for index, command in enumerate(commands, 1):
        validate(command, index)
    require(commands[:PREFIX_COUNT] == source[:PREFIX_COUNT], "standard prefix changed")
    require(commands[PREFIX_COUNT + 14:] == source[PREFIX_COUNT:], "standard insertion/success suffix changed")
    planned_peak = CHALLENGE_COUNT * CHALLENGE_VALUE * ROTATION_SCALE_RAD
    planned_correction = CORRECTION_COUNT * CORRECTION_VALUE * ROTATION_SCALE_RAD
    require(0.06 <= planned_peak <= 0.08, "planned pitch challenge outside frozen range")
    require(CORRECTION_COUNT >= 5 and abs(CORRECTION_VALUE) >= 0.02, "correction excitation insufficient")
    require(math.isclose(planned_peak + planned_correction, 0.0, abs_tol=1e-15), "planned correction does not cancel challenge")

    plan = {
        "status": "PASS",
        "protocol_version": "PEG_V2_PILOT_1",
        "episode_type": "orientation_pitch",
        "source_trace": SOURCE_RELATIVE.as_posix(),
        "source_trace_sha256": SOURCE_SHA256,
        "structure_source": ROLL_RELATIVE.as_posix(),
        "structure_source_sha256": ROLL_SHA256,
        "structure": {
            "standard_prefix_commands": PREFIX_COUNT,
            "challenge_commands": list(range(107, 111)),
            "correction_commands": list(range(111, 116)),
            "settle_commands": list(range(116, 121)),
            "standard_suffix_start": 121,
        },
        "rotation": {
            "axis": "local_pitch/action[4]",
            "rotation_scale_rad": ROTATION_SCALE_RAD,
            "challenge_action": CHALLENGE_VALUE,
            "challenge_count": CHALLENGE_COUNT,
            "planned_peak_rad": planned_peak,
            "correction_action": CORRECTION_VALUE,
            "correction_count": CORRECTION_COUNT,
            "planned_correction_rad": planned_correction,
        },
        "command_count": len(commands),
        "commands": commands,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "command_plan.json"
    durable_exclusive_json(plan_path, plan)
    plan_hash = sha256(plan_path)
    manifest = {
        "status": "PASS",
        "artifact": "F3_V2_ROTATION_PITCH_COMMAND_PLAN",
        "command_plan": plan_path.name,
        "command_plan_sha256": plan_hash,
        "commands": len(commands),
        "source_trace": SOURCE_RELATIVE.as_posix(),
        "source_trace_sha256": SOURCE_SHA256,
        "structure_source": ROLL_RELATIVE.as_posix(),
        "structure_source_sha256": ROLL_SHA256,
        "protocol": PROTOCOL_RELATIVE.as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "planned_pitch_peak_rad": planned_peak,
        "planned_pitch_correction_rad": planned_correction,
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
