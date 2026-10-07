"""Build the frozen lateral -Y plan from the accepted lateral +Y plan."""
import copy
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
    "results/validation/peg/f3_lateral_plus_auto_closure_20260905/command_plan.json"
)
SOURCE_SHA256 = "a7494bc04c469012dad7ca36f23f7e322ffcfd51d93f4f5627384440820aeea7"
PROTOCOL_RELATIVE = Path("results/validation/peg/f3_data_review_20260904/v2_pilot_protocol.json")
OUTPUT_RELATIVE = Path("results/validation/peg/f3_lateral_minus_command_plan_20260905")
CHALLENGE = range(106, 109)  # one-based commands 107..109
CORRECTION = range(109, 112)  # one-based commands 110..112


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_command(command, index):
    require(set(command) == {"phase", "action"}, f"extra/missing fields at {index}")
    require(command["phase"] in PHASES, f"unknown phase at {index}")
    action = command["action"]
    require(isinstance(action, list) and len(action) == 7, f"action dimension at {index}")
    require(
        all(isinstance(value, (int, float)) and math.isfinite(value) for value in action),
        f"nonfinite action at {index}",
    )
    require(all(-1 <= value <= 1 for value in action), f"action bounds at {index}")


def durable_exclusive_json(path, value):
    payload = (json.dumps(value, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    source_path = ROOT / SOURCE_RELATIVE
    protocol_path = ROOT / PROTOCOL_RELATIVE
    output_dir = ROOT / OUTPUT_RELATIVE
    require(source_path.is_file(), "accepted +Y canonical plan missing")
    require(sha256(source_path) == SOURCE_SHA256, "accepted +Y canonical hash mismatch")
    require(protocol_path.is_file(), "frozen V2 pilot protocol missing")
    require(not output_dir.exists(), "exclusive output already exists")

    source = json.loads(source_path.read_bytes())
    commands = source.get("commands")
    require(source.get("status") == "PASS" and isinstance(commands, list), "invalid source envelope")
    require(len(commands) == 176, "source command count mismatch")
    for index, command in enumerate(commands, 1):
        validate_command(command, index)

    plus_challenge = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    plus_correction = [0, -1, 0, 0, 0, 0, -1]
    require(all(commands[i] == {"phase": "lateral_alignment", "action": plus_challenge} for i in CHALLENGE),
            "+Y challenge boundary mismatch")
    require(all(commands[i] == {"phase": "lateral_alignment", "action": plus_correction} for i in CORRECTION),
            "+Y correction boundary mismatch")

    mirrored = copy.deepcopy(commands)
    changed = []
    for index in list(CHALLENGE) + list(CORRECTION):
        before = mirrored[index]["action"][1]
        mirrored[index]["action"][1] = -before
        changed.append({"command": index + 1, "component": 1, "from": before, "to": -before})

    require([item["phase"] for item in mirrored] == [item["phase"] for item in commands],
            "phase ordering changed")
    for index, (before, after) in enumerate(zip(commands, mirrored), 1):
        validate_command(after, index)
        if index - 1 in CHALLENGE or index - 1 in CORRECTION:
            require(after["action"][1] == -before["action"][1], f"lateral sign not mirrored at {index}")
            require(after["action"][:1] + after["action"][2:] == before["action"][:1] + before["action"][2:],
                    f"non-lateral action changed at {index}")
        else:
            require(after == before, f"unauthorized command change at {index}")

    plan = {
        "status": "PASS",
        "protocol_version": "PEG_V2_PILOT_1",
        "episode_type": "lateral_minus_y",
        "source_plan": SOURCE_RELATIVE.as_posix(),
        "source_sha256": SOURCE_SHA256,
        "symmetry": {
            "axis": "action[1]",
            "challenge_commands": [107, 108, 109],
            "correction_commands": [110, 111, 112],
            "changed_commands": changed,
            "all_other_commands": "bit-exact JSON values preserved",
        },
        "command_count": len(mirrored),
        "commands": mirrored,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "command_plan.json"
    durable_exclusive_json(plan_path, plan)
    plan_hash = sha256(plan_path)
    manifest = {
        "status": "PASS",
        "artifact": "F3_V2_LATERAL_MINUS_COMMAND_PLAN",
        "command_plan": plan_path.name,
        "command_plan_sha256": plan_hash,
        "commands": len(mirrored),
        "source_plan": SOURCE_RELATIVE.as_posix(),
        "source_plan_sha256": SOURCE_SHA256,
        "protocol": PROTOCOL_RELATIVE.as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "semantic_delta": "negate action[1] only at commands 107..112",
        "simulator_started": False,
    }
    durable_exclusive_json(output_dir / "manifest.json", manifest)
    descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"status": "PASS", "path": str(plan_path), "sha256": plan_hash, "commands": len(mirrored)}))


if __name__ == "__main__":
    main()
