"""Stable launcher for one frozen standard V2 formal-demo recording.

Default execution is a CPU/static pre-record validation. Recording still
requires both explicit recorder authorization flags.
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


STREAM = ROOT / "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001/transitions.jsonl"
STREAM_SHA256 = "c1e66e2e6e7a43202d2b140162a4ac9e9c6aca1937da077f76c9865bb24f9b4f"
WRAPPER = ROOT / "scripts/run_peg_v2_standard_rerecord_with_contact_audit.py"
DEMO_ROOT = ROOT / "results/demos/peg_v2_pilot"
CONTACT_ROOT = ROOT / "results/diagnostics/peg"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--contact-audit-output", type=Path, required=True)
    value.add_argument("--episode-type", choices=("standard",), required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--max-steps", type=int, required=True)
    value.add_argument("--record", action="store_true")
    value.add_argument("--authorize-recording", action="store_true")
    return value


def project_output(path, parent, label):
    raw = str(path)
    require(not (len(raw) >= 3 and raw[0].isalpha() and raw[1] == ":" and raw[2] in "\\/"),
            f"Windows absolute {label} rejected")
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve(strict=False)
    require(resolved.parent == parent.resolve(), f"{label} must be a direct child of {parent.relative_to(ROOT)}")
    require(not resolved.exists(), f"{label} must not already exist")
    return resolved


def frozen_commands():
    require(STREAM.is_file(), "frozen standard command stream missing")
    require(hashlib.sha256(STREAM.read_bytes()).hexdigest() == STREAM_SHA256,
            "frozen standard command stream hash mismatch")
    rows = [json.loads(line) for line in STREAM.read_text().splitlines() if line.strip()]
    require(rows, "frozen standard command stream empty")
    commands = []
    for index, row in enumerate(rows, 1):
        phase = row.get("phase")
        action = row.get("action")
        require(phase in PHASES, f"unknown phase at command {index}")
        require(isinstance(action, list) and len(action) == 7, f"action must be 7D at command {index}")
        require(all(isinstance(x, (int, float)) and math.isfinite(x) and -1 <= x <= 1 for x in action),
                f"invalid action at command {index}")
        commands.append((phase, action))
    return commands


def validate(args):
    require(args.seed == 0, "standard formal-demo seed must be 0")
    require(args.max_steps == 500, "standard formal-demo max_steps must be 500")
    require(args.record == args.authorize_recording,
            "--record and --authorize-recording must be supplied together")
    require(WRAPPER.is_file(), "existing recorder wrapper missing")
    output = project_output(args.output, DEMO_ROOT, "output")
    contact = project_output(args.contact_audit_output, CONTACT_ROOT, "contact output")
    commands = frozen_commands()
    require(len(commands) <= args.max_steps, "command stream exceeds max_steps")
    text = "".join(phase + " " + " ".join(str(x) for x in action) + "\n" for phase, action in commands)
    argv = [
        sys.executable,
        str(WRAPPER),
        "--output", str(output),
        "--contact-audit-output", str(contact),
        "--episode-type", args.episode_type,
        "--record",
        "--authorize-recording",
    ]
    return output, contact, commands, text, argv


def main(argv=None):
    args = parser().parse_args(argv)
    output, contact, commands, command_text, recorder_argv = validate(args)
    result = {
        "status": "PASS",
        "argument_parse": "PASS",
        "command_stream_binding": "PASS",
        "output_binding": "PASS",
        "pre_record_invocation_validation": "PASS",
        "episode_type": args.episode_type,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "commands": len(commands),
        "command_stream": str(STREAM),
        "command_stream_sha256": STREAM_SHA256,
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
