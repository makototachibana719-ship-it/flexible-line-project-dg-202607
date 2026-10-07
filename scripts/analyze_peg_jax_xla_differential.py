#!/usr/bin/env python3
"""Create immutable manifests and a conservative classification for Peg A/B probes."""
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


def manifest(directory: Path) -> list[dict]:
    return [
        {"path": str(path.relative_to(directory)), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def structural_signature(preflight: dict) -> list[tuple]:
    return [
        (
            leaf["path"],
            tuple(leaf["shape"]),
            leaf["dtype"],
            leaf["type"],
            leaf["device"],
            leaf["sharding_type"],
            leaf["sharding"],
        )
        for leaf in preflight["agent_state_signature"]["leaves"]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result_dir = parse_args().result_dir.resolve()
    batch = json.loads((result_dir / "batch_manifest.json").read_text())
    cases = {}
    for mode in ("A", "B"):
        run_dir = result_dir / mode
        files = manifest(run_dir)
        (run_dir / "xla_manifest.json").write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
        preflight = json.loads((run_dir / "preflight.json").read_text())
        log = (run_dir / "probe.log").read_text(errors="replace")
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
        dispatch = re.findall(r"DISPATCH_SECONDS:\s*([0-9.]+)", log)
        cases[mode] = {
            "exit_status": int((run_dir / "exit_status.txt").read_text()),
            "preflight": preflight,
            "result": result,
            "dispatch_seconds_from_log": float(dispatch[-1]) if dispatch else None,
            "compile_started": log.count("Compiling "),
            "compile_finished": log.count("Finished XLA compilation"),
            "stack_phases": re.findall(r"\[STACK\].*phase=([^\s]+)", log),
            "xla_file_count": len(files),
            "xla_manifest_sha256": sha256(run_dir / "xla_manifest.json"),
        }
    same_batch = cases["A"]["preflight"]["batch_sha256"] == cases["B"]["preflight"]["batch_sha256"] == batch["batch_sha256"]
    same_structure = structural_signature(cases["A"]["preflight"]) == structural_signature(cases["B"]["preflight"])
    same_optimizer = cases["A"]["preflight"]["optimizer_state_structure"] == cases["B"]["preflight"]["optimizer_state_structure"]
    a_pass = cases["A"]["exit_status"] == 0 and cases["A"]["result"] is not None
    b_timeout = cases["B"]["exit_status"] == 124
    b_sync_wait = "full_block_until_ready" in cases["B"]["stack_phases"]
    if not same_structure or not same_optimizer:
        classification = "RESTORE_DEVICE_PLACEMENT_DIFFERENCE"
    elif a_pass and b_timeout and b_sync_wait and cases["B"]["compile_finished"]:
        classification = "XLA_EXECUTOR_STALL"
    elif a_pass and b_timeout and not cases["B"]["compile_finished"]:
        classification = "RECOMPILE_DIFFERENCE"
    elif a_pass and b_timeout and b_sync_wait:
        classification = "HOST_RUNTIME_WAIT"
    else:
        classification = "UNRESOLVED"
    summary = {
        "classification": classification,
        "same_batch": same_batch,
        "same_agent_state_structure_placement": same_structure,
        "same_optimizer_structure": same_optimizer,
        "experiment_A_pass": a_pass,
        "experiment_B_timeout": b_timeout,
        "experiment_B_sync_wait": b_sync_wait,
        "cases": cases,
    }
    (result_dir / "differential_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    all_files = manifest(result_dir)
    (result_dir / "artifact_manifest.json").write_text(json.dumps(all_files, indent=2, sort_keys=True) + "\n")
    print("SAME_BATCH:", same_batch)
    print("SAME_STATE_STRUCTURE_PLACEMENT:", same_structure)
    print("SAME_OPTIMIZER_STRUCTURE:", same_optimizer)
    print("EXPERIMENT_A_PASS:", a_pass)
    print("EXPERIMENT_B_TIMEOUT:", b_timeout)
    print("EXPERIMENT_B_SYNC_WAIT:", b_sync_wait)
    print("PEG_JAX_XLA_DIFFERENTIAL_CLASSIFICATION=" + classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
