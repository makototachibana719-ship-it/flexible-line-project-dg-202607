"""Read-only verifier for the frozen lateral -Y command plan."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion.v2_recorder import PHASES

SOURCE_RELATIVE = Path(
    "results/validation/peg/f3_lateral_plus_auto_closure_20260905/command_plan.json"
)
SOURCE_SHA256 = "a7494bc04c469012dad7ca36f23f7e322ffcfd51d93f4f5627384440820aeea7"
PLAN_RELATIVE = Path(
    "results/validation/peg/f3_lateral_minus_command_plan_20260905/command_plan.json"
)
MANIFEST_RELATIVE = PLAN_RELATIVE.with_name("manifest.json")
EXPECTED_PLAN_SHA256 = "dcf3a7981098935cc8fe919a99e82fcad2c7e81783dd15c9047305bc439fef81"
EXPECTED_COUNT = 176
MIRRORED = frozenset(range(106, 112))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_command(command, index):
    require(set(command) == {"phase", "action"}, f"extra/missing fields at {index}")
    require(command["phase"] in PHASES, f"unknown phase at {index}")
    action = command["action"]
    require(isinstance(action, list) and len(action) == 7, f"action dimension at {index}")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in action), f"nonfinite action at {index}")
    require(all(-1 <= x <= 1 for x in action), f"action bounds at {index}")


def main():
    source_path = ROOT / SOURCE_RELATIVE
    plan_path = ROOT / PLAN_RELATIVE
    manifest_path = ROOT / MANIFEST_RELATIVE
    require(source_path.is_file() and digest(source_path) == SOURCE_SHA256, "+Y source binding mismatch")
    require(plan_path.is_file() and manifest_path.is_file(), "lateral -Y artifact missing")
    require(EXPECTED_PLAN_SHA256 != "__TO_BE_FROZEN__", "verifier hash is not frozen")
    require(digest(plan_path) == EXPECTED_PLAN_SHA256, "lateral -Y plan hash mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("command_plan_sha256") == EXPECTED_PLAN_SHA256, "manifest hash mismatch")
    require(manifest.get("commands") == EXPECTED_COUNT, "manifest count mismatch")
    require(manifest.get("source_plan_sha256") == SOURCE_SHA256, "manifest source mismatch")

    source = json.loads(source_path.read_bytes())["commands"]
    value = json.loads(plan_path.read_bytes())
    commands = value.get("commands")
    require(value.get("status") == "PASS" and value.get("episode_type") == "lateral_minus_y", "invalid plan envelope")
    require(isinstance(commands, list) and len(commands) == EXPECTED_COUNT, "plan count mismatch")
    require(len(source) == EXPECTED_COUNT, "source count mismatch")
    require([x["phase"] for x in commands] == [x["phase"] for x in source], "phase sequence changed")

    changed = []
    for offset, (before, after) in enumerate(zip(source, commands)):
        validate_command(after, offset + 1)
        if offset in MIRRORED:
            require(after["action"][1] == -before["action"][1], f"lateral sign mismatch at {offset + 1}")
            require(after["action"][:1] + after["action"][2:] == before["action"][:1] + before["action"][2:],
                    f"non-lateral action changed at {offset + 1}")
            changed.append(offset + 1)
        else:
            require(after == before, f"unauthorized change at {offset + 1}")
    require(changed == [107, 108, 109, 110, 111, 112], "symmetry change set mismatch")
    require(all(commands[i]["phase"] == "lateral_alignment" for i in MIRRORED), "illegal correction phase")
    print(json.dumps({
        "status": "PASS",
        "path": str(plan_path),
        "sha256": EXPECTED_PLAN_SHA256,
        "commands": len(commands),
        "phase_validation": "PASS",
        "action_validation": "PASS",
        "symmetry_audit": "PASS",
        "changed_commands": changed,
        "simulator_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
