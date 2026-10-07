"""Read-only verifier for the calibrated local-pitch v2 command plan."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion.v2_recorder import PHASES

OLD_PLAN = Path("results/validation/peg/f3_rotation_pitch_command_plan_20260905/command_plan.json")
OLD_PLAN_SHA256 = "9ce4e9583563c900b75d977b7bf450b29f63c298a26193748afdb9bf95bcc1d9"
PLAN = Path("results/validation/peg/f3_rotation_pitch_revised_plan_20260906/command_plan_v2.json")
MANIFEST = PLAN.with_name("manifest.json")
EXPECTED_SHA256 = "__TO_BE_FROZEN__"
EXPECTED_COUNT = 182
PREFIX_COUNT = 106
SUFFIX_COUNT = 59


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(command, index):
    require(set(command) == {"phase", "action"}, f"fields at {index}")
    require(command["phase"] in PHASES, f"phase at {index}")
    action = command["action"]
    require(isinstance(action, list) and len(action) == 7, f"shape at {index}")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and -1 <= x <= 1 for x in action),
            f"action at {index}")


def main():
    old_path, plan_path, manifest_path = ROOT / OLD_PLAN, ROOT / PLAN, ROOT / MANIFEST
    require(old_path.is_file() and digest(old_path) == OLD_PLAN_SHA256, "old plan binding mismatch")
    require(plan_path.is_file() and manifest_path.is_file(), "revised artifact missing")
    require(EXPECTED_SHA256 != "__TO_BE_FROZEN__", "verifier hash not frozen")
    require(digest(plan_path) == EXPECTED_SHA256, "revised plan hash mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("command_plan_sha256") == EXPECTED_SHA256, "manifest hash mismatch")
    require(manifest.get("commands") == EXPECTED_COUNT, "manifest count mismatch")
    require(manifest.get("old_plan_sha256") == OLD_PLAN_SHA256, "manifest old-plan mismatch")

    old = json.loads(old_path.read_bytes())["commands"]
    plan = json.loads(plan_path.read_bytes())
    commands = plan.get("commands")
    require(plan.get("status") == "PASS" and plan.get("revision") == "challenge_calibrated_v2", "plan envelope")
    require(isinstance(commands, list) and len(commands) == EXPECTED_COUNT, "plan count")
    for index, command in enumerate(commands, 1):
        validate(command, index)
    require(commands[:PREFIX_COUNT] == old[:PREFIX_COUNT], "prefix changed")
    require(commands[-SUFFIX_COUNT:] == old[-SUFFIX_COUNT:], "suffix changed")

    challenge = commands[106:111]
    correction = commands[111:118]
    settle = commands[118:123]
    require(all(x == {"phase": "orientation_correction", "action": [0, 0, 0, 0, 1.0, 0, -1]} for x in challenge),
            "challenge mismatch")
    expected_correction = [-0.8] * 6 + [-0.2]
    require([x["action"][4] for x in correction] == expected_correction, "correction mismatch")
    require(all(x["phase"] == "orientation_correction" and x["action"][:4] == [0, 0, 0, 0]
                and x["action"][5:] == [0, -1] for x in correction), "correction non-pitch component changed")
    require(all(x == {"phase": "orientation_correction", "action": [0, 0, 0, 0, 0, 0, -1]} for x in settle),
            "settle mismatch")
    require(math.isclose(sum(x["action"][4] for x in challenge + correction), 0.0, abs_tol=1e-15),
            "net pitch rotation is not zero")

    calibration = plan["calibration"]
    steps = calibration["steps"]
    errors = [x["orientation_error"] for x in steps]
    require(len(steps) == 4 and all(b > a for a, b in zip(errors, errors[1:])), "response not monotonic")
    require(all(math.isfinite(v) for step in steps for value in step.values()
                for v in (value if isinstance(value, list) else [value]) if isinstance(v, (int, float))),
            "nonfinite calibration")
    expected_peak = float(calibration["revised_expected_peak_rad"])
    require(0.06 <= expected_peak <= 0.08, "revised expected peak outside protocol")
    require(manifest.get("revised_challenge_steps") == 5 and manifest.get("revised_correction_steps") == 7,
            "revised step counts mismatch")
    print(json.dumps({
        "status": "PASS",
        "path": str(plan_path),
        "sha256": EXPECTED_SHA256,
        "commands": len(commands),
        "pitch_challenge_calibration": "PASS",
        "pitch_response_monotonic": "PASS",
        "phase_validation": "PASS",
        "action_validation": "PASS",
        "semantic_audit": "PASS",
        "old_planned_peak": calibration["old_planned_peak_rad"],
        "old_realized_peak": calibration["old_realized_peak_rad"],
        "realized_to_planned_ratio": calibration["realized_to_planned_ratio"],
        "revised_challenge_steps": len(challenge),
        "revised_correction_steps": len(correction),
        "revised_expected_peak": expected_peak,
        "simulator_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
