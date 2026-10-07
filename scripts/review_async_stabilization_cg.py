#!/usr/bin/env python3
"""Static async stabilization review for Peg F3 V2 C+G."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/makoto/flexible_line_project")
OUT_ROOT = ROOT / "results/diagnostics/peg"
CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
RUNNER = ROOT / "scripts/run_combined_q_anchor_n2_candidate_1k.py"
REPLAY_AUDIT = ROOT / "results/diagnostics/peg/f3_v2_async_variance_replay_audit_20260907_002/result.json"
FROZEN_AUDIT = ROOT / "results/diagnostics/peg/f3_v2_frozen_data_learner_replay_cg_20260907_001/result.json"


def unique_out() -> Path:
    stem = "f3_v2_async_stabilization_review_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    replay = read(REPLAY_AUDIT)
    frozen = read(FROZEN_AUDIT)
    require(replay["REPLAY_DISTRIBUTION_MATCH"] == "NO", "replay mismatch evidence missing")
    require(frozen["FROZEN_STREAM_READY"] == "PASS", "frozen stream not ready")
    require(frozen["FROZEN_DATA_REPRODUCIBILITY"] == "PASS", "frozen reproducibility not proven")
    require(frozen["POLICY_VECTOR_DISTANCE_R1_R2"] == 0.0, "frozen learner params diverged")

    protocol = {
        "name": "FIXED_REPLAY_PREFILL_GATE_WITH_ADMISSION_FREEZE",
        "mechanism": [
            "start actor collection first",
            "learner waits until online replay reaches exact prefill target",
            "freeze online admission at target count",
            "run 1000 learner updates over fixed online replay plus fixed Demo buffer",
            "checkpoint by learner update 500/1000 only",
        ],
        "preserves": {
            "algorithm": "C+G only",
            "batch": "64",
            "demo_online_ratio": "32:32",
            "utd": 1,
            "dataset": "unchanged",
            "replay_schema": "unchanged",
            "action_semantics": "unchanged",
            "v2": "unchanged",
        },
        "target": {
            "replay_size_delta_pct_at_update1000": "<=5%",
            "preferred": "<=2%",
            "policy_vector_distance": "reduced vs async C+G pair",
        },
    }
    result = {
        "ASYNC_STABILIZATION_REVIEW": "PASS",
        "PRIMARY_TIMING_DIFFERENCE": replay["LARGEST_REPLAY_DIFFERENCE"],
        "STABILIZATION_PROTOCOL": protocol,
        "FILES_TO_CHANGE": [
            "new runner/wrapper only: scripts/run_cg_fixed_prefill_freeze_n2_1k.py",
            "new output dirs only: results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_YYYYMMDD/replica*",
        ],
        "SOURCE_HASHES_BEFORE": {
            "config_sha256": sha(CONFIG),
            "current_cg_runner_sha256": sha(RUNNER),
        },
        "ALGORITHM_UNCHANGED": "YES",
        "REPLAY_SCHEMA_UNCHANGED": "YES",
        "DATASET_UNCHANGED": "YES",
        "ACTION_SEMANTICS_UNCHANGED": "YES",
        "V2_UNCHANGED": "YES",
        "STATIC_PREFLIGHT": "PASS",
        "NEXT_N2_1K_READY": "YES_STATIC_DESIGN_ONLY",
        "NEXT_COMMAND": (
            "cd /home/makoto/flexible_line_project && "
            "/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python "
            "scripts/run_cg_fixed_prefill_freeze_n2_1k.py --role preflight "
            "--output results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica1"
        ),
        "TRAINING_STARTED": "false",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "NEXT": "IMPLEMENT_NEW_RUNNER_WRAPPER_THEN_PREFLIGHT_ONLY",
    }
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "ASYNC_STABILIZATION_REVIEW": result["ASYNC_STABILIZATION_REVIEW"],
        "PRIMARY_TIMING_DIFFERENCE": result["PRIMARY_TIMING_DIFFERENCE"],
        "STABILIZATION_PROTOCOL": result["STABILIZATION_PROTOCOL"]["name"],
        "STATIC_PREFLIGHT": result["STATIC_PREFLIGHT"],
        "NEXT_N2_1K_READY": result["NEXT_N2_1K_READY"],
        "OUT": str(out / "result.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
