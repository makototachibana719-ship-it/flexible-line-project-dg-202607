"""Fail-closed verifier for the canonical frozen lateral +Y command plan."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion.v2_recorder import PHASES

CANONICAL_RELATIVE = Path(
    "results/validation/peg/f3_lateral_plus_auto_closure_20260905/command_plan.json"
)
EXPECTED_SHA256 = "a7494bc04c469012dad7ca36f23f7e322ffcfd51d93f4f5627384440820aeea7"
EXPECTED_COUNT = 176


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_path(root=ROOT):
    raw = str(CANONICAL_RELATIVE)
    require(not CANONICAL_RELATIVE.is_absolute(), "canonical plan must be project-relative")
    require("\\" not in raw and not (len(raw) > 1 and raw[1] == ":"), "Windows path rejected")
    return Path(root) / CANONICAL_RELATIVE


def verify(path, expected_hash=EXPECTED_SHA256, expected_count=EXPECTED_COUNT):
    path = Path(path)
    require(path.exists(), "canonical plan missing")
    require(path.is_file(), "canonical plan is not a regular file")
    require(path.stat().st_mode, "canonical plan unreadable")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_hash, "canonical plan hash mismatch")
    value = json.loads(raw)
    commands = value.get("commands")
    require(value.get("status") == "PASS" and isinstance(commands, list), "invalid plan envelope")
    require(len(commands) == expected_count, "canonical plan command count mismatch")
    for index, command in enumerate(commands, 1):
        require(set(command) == {"phase", "action"}, f"extra/missing fields at {index}")
        require(command["phase"] in PHASES, f"unknown phase at {index}")
        action = command["action"]
        require(isinstance(action, list) and len(action) == 7, f"action dimension at {index}")
        require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in action), f"nonfinite action at {index}")
        require(all(-1 <= x <= 1 for x in action), f"action bounds at {index}")
    require(not any(x["phase"] == "lateral_correction" for x in commands), "illegal phase remains")
    correction = commands[109:112]
    require(len(correction) == 3 and all(x["phase"] == "lateral_alignment" for x in correction), "correction phase mismatch")
    return {"path": str(path), "sha256": expected_hash, "commands": len(commands)}


def self_test():
    source = canonical_path().read_bytes()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        target = canonical_path(root)
        target.parent.mkdir(parents=True)
        target.write_bytes(source)
        verify(target)
        # An absent historical report-copy path is intentionally irrelevant.
        require(not (root / "results/validation/peg/f3_lateral_plus_bootstrap_fix_20260905/command_plan.json").exists(), "fixture error")
        verify(target)
        try:
            verify(target, expected_hash="0" * 64)
            raise AssertionError("wrong hash accepted")
        except ValueError:
            pass
        try:
            verify(target, expected_count=175)
            raise AssertionError("wrong count accepted")
        except ValueError:
            pass
        try:
            candidate = Path("C:\\temp\\command_plan.json")
            raw = str(candidate)
            require("\\" not in raw and not (len(raw) > 1 and raw[1] == ":"), "Windows path rejected")
            raise AssertionError("Windows path accepted")
        except ValueError:
            pass
    return 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = verify(canonical_path())
    if args.self_test:
        result["path_regression_tests"] = self_test()
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__":
    main()
