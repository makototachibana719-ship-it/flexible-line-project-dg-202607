#!/usr/bin/env python3
"""Analyze the one-shot fixed/dynamic fresh runtime reliability arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compile_counts(log: str) -> dict:
    lines = log.splitlines()
    start0 = next((i for i, line in enumerate(lines) if "UPDATE_START index=0" in line), None)
    done0 = next((i for i, line in enumerate(lines) if "UPDATE_COMPLETE:" in line), None)
    if start0 is None:
        return {"before_update_0": None, "initial_update": None, "post_initial_recompile": None}
    if done0 is None:
        done0 = len(lines)
    count = lambda segment: sum("Compiling " in line for line in segment)
    return {
        "before_update_0": count(lines[:start0]),
        "initial_update": count(lines[start0:done0]),
        "post_initial_recompile": count(lines[done0:]),
        "initial_update_high_utd": sum(
            "Compiling update_high_utd" in line for line in lines[start0:done0]
        ),
        "post_initial_update_high_utd": sum(
            "Compiling update_high_utd" in line for line in lines[done0:]
        ),
        "total_started": log.count("Compiling "),
        "total_finished": log.count("Finished XLA compilation"),
    }


def summarize(root: Path, mode: str) -> dict:
    case = root / mode
    all_events = events(case / "updates.jsonl")
    completed = [event for event in all_events if event.get("event") == "complete"]
    timeout_path = case / "timeout.json"
    result_path = case / "result.json"
    log = (case / "probe.log").read_text(errors="replace")
    post_classes = [event["latency_classification"] for event in completed if event["index"] > 0]
    gate_pass = (
        len(completed) == 20
        and all(event["losses_finite"] for event in completed)
        and not any(value in {"ANOMALOUS", "TIMEOUT"} for value in post_classes)
    )
    return {
        "exit_status": int((case / "exit_status.txt").read_text()),
        "gate": "PASS" if gate_pass else "FAIL",
        "completed_updates": len(completed),
        "updates": completed,
        "post_compile_latency_classes": post_classes,
        "max_post_compile_seconds": max(
            (event["true_update_seconds"] for event in completed if event["index"] > 0),
            default=None,
        ),
        "compile_counts": compile_counts(log),
        "timeout": json.loads(timeout_path.read_text()) if timeout_path.is_file() else None,
        "result": json.loads(result_path.read_text()) if result_path.is_file() else None,
        "preflight": json.loads((case / "preflight.json").read_text()),
        "thirty_second_snapshots": sorted(path.name for path in case.glob("update_*_30s.*")),
        "probe_log_sha256": sha256(case / "probe.log"),
        "gpu_dmon_sha256": sha256(case / "nvidia_smi_dmon.log"),
        "cpu_threads_sha256": sha256(case / "cpu_threads.log"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    r1 = summarize(root, "R1")
    r2 = summarize(root, "R2")
    comparable = (
        r1["preflight"]["fresh_initialization"]
        and r2["preflight"]["fresh_initialization"]
        and not r1["preflight"]["resume"]
        and not r2["preflight"]["resume"]
        and r1["preflight"]["initial_rng"] == r2["preflight"]["initial_rng"]
        and r1["preflight"]["fixed_batch_sha256"] == r2["preflight"]["fixed_batch_sha256"]
    )
    any_stall = any(
        case["timeout"] is not None
        or any(value in {"ANOMALOUS", "TIMEOUT"} for value in case["post_compile_latency_classes"])
        for case in (r1, r2)
    )
    post_recompile = any(
        (case["compile_counts"].get("post_initial_recompile") or 0) > 0 for case in (r1, r2)
    )
    if not comparable:
        classification = "UNRESOLVED"
    elif r1["gate"] == "PASS" and r2["gate"] == "PASS" and not post_recompile:
        classification = "STABLE"
    elif r1["gate"] == "PASS" and r2["gate"] == "FAIL" and any_stall:
        classification = "DYNAMIC_BATCH_DEPENDENT"
    elif post_recompile and (any_stall or r1["gate"] == "FAIL" or r2["gate"] == "FAIL"):
        classification = "RECOMPILE"
    elif any_stall:
        classification = "INTERMITTENT_STALL"
    else:
        classification = "UNRESOLVED"
    summary = {
        "fixed_batch_gate": r1["gate"],
        "dynamic_batch_gate": r2["gate"],
        "classification": classification,
        "comparable": comparable,
        "cases": {"R1": r1, "R2": r2},
    }
    (root / "runtime_reliability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("PEG_RUNTIME_FIXED_BATCH_20=" + r1["gate"])
    print("PEG_RUNTIME_DYNAMIC_BATCH_20=" + r2["gate"])
    print("PEG_RUNTIME_RELIABILITY_CLASSIFICATION=" + classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
