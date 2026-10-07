#!/usr/bin/env python3
"""Read-only final adjudication of frozen Peg F3 V2 smoke evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results/validation/peg/f3_v2_rlpd_smoke_1000_20260906"
RESTORE = ROOT / "results/validation/peg/f3_v2_smoke_shutdown_restore_20260906"
TEARDOWN = ROOT / "results/validation/peg/f3_reqrep_teardown_review_20260906"
NORMAL = ROOT / "results/validation/peg/f3_reqrep_normal_path_revalidation_20260906_002"
DATASET = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
EXPECTED_DATASET = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"


def load(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: adjudicate_peg_f3_v2_smoke_final.py OUTPUT")
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pre = load(SMOKE / "preflight.json")
    learner = load(SMOKE / "learner_result.json")
    actor = load(SMOKE / "actor_result.json")
    updates = jsonl(SMOKE / "updates.jsonl")
    communication = jsonl(SMOKE / "communication.jsonl")
    checkpoint_events = jsonl(SMOKE / "checkpoint_events.jsonl")
    restore = load(SMOKE / "restore_verification.json")
    normal = load(NORMAL / "validation_normal_path.json")
    teardown_report = (TEARDOWN / "REPORT.md").read_text()
    restore_report = (RESTORE / "REPORT.md").read_text()
    checkpoints = {row.get("checkpoint") for row in checkpoint_events}
    chains = {
        "COMPUTE_CHAIN": (pre["backend"] == learner["backend"] == actor["backend"] == "gpu"
                          and learner["completed_updates"] == 1000
                          and len(updates) == 1000
                          and [row["update"] for row in updates] == list(range(1, 1001))
                          and all(row["losses_finite"] for row in updates)),
        "DATA_CHAIN": (sha(DATASET) == EXPECTED_DATASET == pre["dataset_sha256"]
                       and pre["episodes"] == 11 and pre["transitions"] == 1921
                       and learner["demo_size"] == 1921),
        "REPLAY_CHAIN": (learner["final_online_size"] > learner["initial_online_size"]
                         and all(row["demo_batch"] == row["online_batch"] == 32
                                 for row in updates)),
        "COMMUNICATION_CHAIN": (actor["received_networks"] > 0
                                and any(row.get("event") == "live_insert"
                                        for row in communication)),
        "CHECKPOINT_CHAIN": (checkpoints == {500, 1000}
                             and all((SMOKE / "checkpoints" / name).is_file()
                                     for step in (500, 1000)
                                     for name in (f"checkpoint_{step}",
                                                  f"checkpoint_{step}_online_replay.pkl",
                                                  f"checkpoint_{step}_resume_metadata.json"))),
        "RESTORE_CHAIN": (restore["status"] == "PASS"
                          and restore["restored_numeric_leaves"] == 198
                          and restore["training_resumed"] is False
                          and "PEG_F3_CHECKPOINT_1000_RESTORE_ACCEPTANCE=PASS" in restore_report),
        "SHUTDOWN_CHAIN": (normal["PEG_F3_REQREP_NORMAL_PATH_REVALIDATION"] == "PASS"
                           and normal["ACTOR_NATURAL_EXIT"] == "PASS"
                           and normal["ZMQ_REAPER_AFTER_EXIT"] == "NO"
                           and normal["ZMQ_IO_AFTER_EXIT"] == "NO"
                           and normal["RESIDUAL_PROCESSES"] == "NONE"
                           and "PEG_F3_REQREP_FIX_REVIEW=PASS" in teardown_report),
    }
    passed = all(chains.values())
    result = {
        "PEG_F3_V2_RLPD_SMOKE_FINAL": "PASS" if passed else "FAIL",
        **{key: "PASS" if value else "FAIL" for key, value in chains.items()},
        "TRAINING_READY": "YES" if passed else "NO",
        "NEXT": "FORMAL_5K_PREFLIGHT_READY" if passed else "BLOCKED",
        "training_started": False,
        "source_evidence": [str(SMOKE), str(RESTORE), str(TEARDOWN), str(NORMAL)],
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    report = "# Stage 3 — RLPD Smoke Final Adjudication\n\n" + "\n".join(
        f"- `{key}={value}`" for key, value in result.items()
        if key not in ("source_evidence", "training_started")) + "\n\nNo training was run.\n"
    (output / "REPORT.md").write_text(report)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
