"""Stable launcher for one frozen lateral +Y V2 formal-demo recording.

Default execution performs CPU/static pre-record validation only. Recording
still requires both explicit recorder authorization flags.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.peg_insertion.v2_recorder import PHASES


PLAN = ROOT / "results/validation/peg/f3_lateral_plus_auto_closure_20260905/command_plan.json"
PLAN_SHA256 = "a7494bc04c469012dad7ca36f23f7e322ffcfd51d93f4f5627384440820aeea7"
PLAN_COMMANDS = 176
VERIFIER = ROOT / "scripts/verify_peg_lateral_plus_plan_binding.py"
WRAPPER = ROOT / "scripts/run_peg_v2_standard_rerecord_with_contact_audit.py"
DEMO_ROOT = ROOT / "results/demos/peg_v2_pilot"
CONTACT_ROOT = ROOT / "results/diagnostics/peg"
EXPECTED_PYTHON = Path("/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--contact-audit-output", type=Path, required=True)
    value.add_argument("--episode-type", choices=("lateral_plus_y",), required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--max-steps", type=int, required=True)
    value.add_argument("--record", action="store_true")
    value.add_argument("--authorize-recording", action="store_true")
    return value


def project_output(path, parent, label):
    raw = str(path)
    require(
        not (len(raw) >= 3 and raw[0].isalpha() and raw[1] == ":" and raw[2] in "\\/"),
        f"Windows absolute {label} rejected",
    )
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve(strict=False)
    require(resolved.parent == parent.resolve(), f"{label} must be a direct child of {parent.relative_to(ROOT)}")
    require(not resolved.exists(), f"{label} must not already exist")
    return resolved


def frozen_commands():
    require(PLAN.is_file(), "frozen lateral +Y command plan missing")
    require(hashlib.sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA256, "frozen lateral +Y plan hash mismatch")
    envelope = json.loads(PLAN.read_text())
    commands = envelope.get("commands")
    require(envelope.get("status") == "PASS" and isinstance(commands, list), "invalid command plan envelope")
    require(len(commands) == PLAN_COMMANDS and envelope.get("command_count") == PLAN_COMMANDS, "command count mismatch")
    result = []
    for index, command in enumerate(commands, 1):
        require(set(command) == {"phase", "action"}, f"extra/missing command fields at {index}")
        phase, action = command["phase"], command["action"]
        require(phase in PHASES, f"unknown phase at command {index}")
        require(isinstance(action, list) and len(action) == 7, f"action must be 7D at command {index}")
        require(
            all(isinstance(x, (int, float)) and math.isfinite(x) and -1 <= x <= 1 for x in action),
            f"invalid action at command {index}",
        )
        result.append((phase, action))
    return result


def validate(args):
    require(Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), "unexpected WSL project Python")
    require(args.seed == 0, "lateral +Y formal-demo seed must be 0")
    require(args.max_steps == 500, "lateral +Y formal-demo max_steps must be 500")
    require(args.record == args.authorize_recording, "--record and --authorize-recording must be supplied together")
    require(VERIFIER.is_file() and WRAPPER.is_file(), "verifier or recorder wrapper missing")
    verifier = subprocess.run([sys.executable, str(VERIFIER)], cwd=ROOT, capture_output=True, text=True)
    require(verifier.returncode == 0, "canonical verifier failed: " + verifier.stderr.strip())
    verifier_result = json.loads(verifier.stdout)
    require(verifier_result.get("status") == "PASS", "canonical verifier did not report PASS")
    output = project_output(args.output, DEMO_ROOT, "output")
    contact = project_output(args.contact_audit_output, CONTACT_ROOT, "contact output")
    commands = frozen_commands()
    require(len(commands) <= args.max_steps, "command plan exceeds max_steps")
    command_text = "".join(phase + " " + " ".join(str(x) for x in action) + "\n" for phase, action in commands)
    recorder_argv = [
        sys.executable,
        str(WRAPPER),
        "--output",
        str(output),
        "--contact-audit-output",
        str(contact),
        "--episode-type",
        args.episode_type,
        "--record",
        "--authorize-recording",
    ]
    return output, contact, commands, command_text, recorder_argv


def main(argv=None):
    args = parser().parse_args(argv)
    output, contact, commands, command_text, recorder_argv = validate(args)
    result = {
        "status": "PASS",
        "shell_boundary": "PASS",
        "python_binding": "PASS",
        "canonical_verifier_executed": "PASS",
        "pre_record_invocation_validation": "PASS",
        "episode_type": args.episode_type,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "commands": len(commands),
        "command_plan": str(PLAN),
        "command_plan_sha256": PLAN_SHA256,
        "python": sys.executable,
        "output": str(output),
        "contact_output": str(contact),
        "recording_authorized": bool(args.record and args.authorize_recording),
        "simulator_started": False,
        "demo_recording_started": False,
    }
    print(json.dumps(result, indent=2))
    if not args.record:
        return
    subprocess.run(recorder_argv, cwd=ROOT, input=command_text, text=True, check=True)


if __name__ == "__main__":
    main()
