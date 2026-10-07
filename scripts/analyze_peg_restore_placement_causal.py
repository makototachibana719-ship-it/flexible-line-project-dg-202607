#!/usr/bin/env python3
"""Classify the one-shot D1/D2 restore-placement causal diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_BATCH_SHA256 = "854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def structural_signature(signature: dict) -> list[tuple]:
    return [
        (leaf["path"], tuple(leaf["shape"]), leaf["dtype"])
        for leaf in signature["leaves"]
    ]


def case_summary(root: Path, mode: str) -> dict:
    case = root / mode
    preflight = json.loads((case / "preflight.json").read_text())
    events = read_events(case / "updates.jsonl")
    completed = [event for event in events if event.get("event") == "complete"]
    started = [event for event in events if event.get("event") == "start"]
    timeout_path = case / "timeout.json"
    result_path = case / "result.json"
    log = (case / "probe.log").read_text(errors="replace")
    return {
        "exit_status": int((case / "exit_status.txt").read_text()),
        "preflight": preflight,
        "started_updates": len(started),
        "completed_updates": len(completed),
        "updates": completed,
        "all_losses_finite": bool(completed) and all(event["losses_finite"] for event in completed),
        "timeout": json.loads(timeout_path.read_text()) if timeout_path.is_file() else None,
        "result": json.loads(result_path.read_text()) if result_path.is_file() else None,
        "compile_started": log.count("Compiling "),
        "compile_finished": log.count("Finished XLA compilation"),
        "stack_phases": [line for line in log.splitlines() if line.startswith("[STACK]")],
        "probe_log_sha256": sha256(case / "probe.log"),
        "dmon_sha256": sha256(case / "nvidia_smi_dmon.log"),
        "cpu_threads_sha256": sha256(case / "cpu_threads.log"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    d1 = case_summary(root, "D1")
    d2 = case_summary(root, "D2")
    before_d1 = d1["preflight"]["state_signature_before_placement"]
    before_d2 = d2["preflight"]["state_signature_before_placement"]
    same_restored_structure = structural_signature(before_d1) == structural_signature(before_d2)
    same_rng = d1["preflight"]["checkpoint_rng"] == d2["preflight"]["checkpoint_rng"]
    same_batch = (
        d1["preflight"]["batch_sha256"]
        == d2["preflight"]["batch_sha256"]
        == EXPECTED_BATCH_SHA256
    )
    d1_host = d1["preflight"]["placement_counts_after"] == {
        "leaf_count": 198,
        "cuda_jax_array_count": 0,
        "host_numpy_count": 198,
    }
    d2_gpu = d2["preflight"]["placement_counts_after"] == {
        "leaf_count": 198,
        "cuda_jax_array_count": 198,
        "host_numpy_count": 0,
    }
    comparable = same_restored_structure and same_rng and same_batch and d1_host and d2_gpu
    c1 = d1["completed_updates"]
    c2 = d2["completed_updates"]
    if not comparable or not d1["all_losses_finite"] or not d2["all_losses_finite"]:
        classification = "INCONCLUSIVE"
    elif c2 > c1:
        classification = "IMPROVES_STABILITY"
    elif c2 < c1:
        classification = "REGRESSION"
    elif c1 == c2 == 5 and d1["exit_status"] == d2["exit_status"] == 0:
        classification = "NO_EFFECT"
    elif c1 == c2 and d1["timeout"] is not None and d2["timeout"] is not None:
        classification = "NO_EFFECT"
    else:
        classification = "INCONCLUSIVE"
    summary = {
        "classification": classification,
        "comparable": comparable,
        "same_batch": same_batch,
        "same_checkpoint_rng": same_rng,
        "same_restored_shape_dtype_structure": same_restored_structure,
        "D1_host_control_confirmed": d1_host,
        "D2_explicit_gpu_placement_confirmed": d2_gpu,
        "cases": {"D1": d1, "D2": d2},
    }
    output = root / "causal_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("COMPARABLE:", comparable)
    print("D1_COMPLETED_UPDATES:", c1)
    print("D2_COMPLETED_UPDATES:", c2)
    print("PEG_RESTORE_PLACEMENT_CAUSAL_DIAGNOSTIC=" + classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
