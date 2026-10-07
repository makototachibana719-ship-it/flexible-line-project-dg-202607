"""Build and statically validate the one-shot lateral +Y closure command plan."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.peg_insertion.v2_recorder import PHASES


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-trace", type=Path, required=True)
    parser.add_argument("--standard-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")

    failed = [json.loads(line) for line in args.failed_trace.read_text().splitlines()]
    standard = [json.loads(line) for line in args.standard_trace.read_text().splitlines()]
    require(len(failed) == 109, "expected preserved 109-step failed attempt")
    require(len(standard) == 165, "expected accepted 165-step standard trace")

    commands = [dict(phase=row["phase"], action=row["action"]) for row in failed]
    # The sole correction fix: use the existing legal lateral_alignment phase.
    commands += [dict(phase="lateral_alignment", action=[0, -1, 0, 0, 0, 0, -1]) for _ in range(3)]
    commands += [dict(phase="lateral_alignment", action=[0, 0, 0, 0, 0, 0, -1]) for _ in range(5)]
    commands += [dict(phase="insertion", action=row["action"]) for row in standard[106:]]

    for index, command in enumerate(commands, 1):
        require(set(command) == {"phase", "action"}, f"extra/missing token fields at {index}")
        require(command["phase"] in PHASES, f"unknown phase at {index}")
        action = command["action"]
        require(isinstance(action, list) and len(action) == 7, f"action count at {index}")
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in action), f"nonfinite action at {index}")
        require(all(-1 <= v <= 1 for v in action), f"action bounds at {index}")
    require(commands[0]["phase"] == "setup", "setup segment missing")
    require(len(commands) == 176, "command sequence incomplete")
    require(all(c["phase"] != "lateral_correction" for c in commands), "illegal phase remains")

    payload = {"status": "PASS", "phase_set": list(PHASES), "command_count": len(commands), "commands": commands}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"status": "PASS", "command_count": len(commands), "output": str(args.output)}))


if __name__ == "__main__":
    main()
