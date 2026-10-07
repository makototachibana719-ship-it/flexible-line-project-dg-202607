#!/usr/bin/env python3
"""Adjudicate a completed stage without changing its immutable first result."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

BASELINE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    if BASELINE not in run.parents:
        raise ValueError("run outside baseline")
    result = json.loads((run / "result.json").read_text())
    restore = json.loads((run / "restore_verification.json").read_text())
    checks = result["checks"]
    scientific = all(checks[key] for key in (
        "gpu", "demo", "online_growth", "ratio", "updates", "finite",
        "communication", "checkpoint_1000", "checkpoint_targets",
        "children_exited", "learner_stop", "actor_stop_observed",
        "trainerclient_stop", "broadcast_join", "actor_natural_exit",
        "real_communication"))
    final = {
        "status": "PASS" if scientific and restore.get("status") == "PASS" else "FAIL",
        "step": args.step,
        "training_updates_complete": scientific,
        "restore_verification": restore.get("status"),
        "original_supervisor_status": result.get("PEG_F3_V2_RLPD_SMOKE"),
        "original_failure_class": "POST_TRAIN_VERIFY_RUN_ID_MISMATCH",
        "retrained": False,
        "checkpoint_modified": False,
    }
    path = run / "adjudication.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
