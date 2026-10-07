"""Fail-closed verifier for the Peg continuous formal-candidate GPU gate."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jaxlib
import numpy as np
import wandb
import yaml
from flax.training import checkpoints


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/peg_insertion/peg_rlpd_baseline_seed0.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def running_training_processes() -> list[int]:
    matches = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "training.peg_insertion.run_peg_training" in command:
            matches.append(int(entry.name))
    return matches


def verify() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    candidate = cfg["formal_candidate"]
    result_dir = ROOT / candidate["result_dir"]
    checkpoint_dir = ROOT / candidate["checkpoint_dir"]
    result = json.loads((result_dir / "result.json").read_text())
    records = [json.loads(line) for line in (result_dir / "updates.jsonl").read_text().splitlines() if line]
    if result.get("gate") == "FAIL":
        raise RuntimeError(
            f'candidate reported {result.get("reason")} at update {result.get("failed_update")} '
            f'with limit {result.get("timeout_seconds")}s'
        )

    assert jax.default_backend() == "gpu", jax.devices()
    assert jax.__version__ == "0.4.35"
    assert jaxlib.__version__ == "0.4.35"
    assert result["profile"] == "smoke" and result["formal_candidate"] is True
    assert result["fresh_initialization"] is True
    assert result["single_learner_process"] is True and result["process_launch_count"] == 1
    assert result["normal_rng_mode"] is True and result["diagnostic_enabled"] is False
    assert result["updates_completed"] == result["updates_expected"] == 20
    assert len(records) == 20 and [item["update"] for item in records] == list(range(20))
    assert {item["pid"] for item in records} == {result["learner_process_pid"]}
    assert all(0 <= item["sample_seconds"] for item in records)
    assert all(0 < item["update_seconds"] <= 120 for item in records)
    assert all(item["total_seconds"] >= item["update_seconds"] for item in records)
    assert all(item["losses_finite"] and item["losses"] for item in records)
    assert all(np.isfinite(value) for item in records for value in item["losses"].values())
    assert all(item["rng_before"] != item["rng_after"] for item in records)
    assert all(records[index - 1]["rng_after"] == records[index]["rng_before"] for index in range(1, 20))

    checkpoint = checkpoint_dir / "checkpoint_19"
    replay = checkpoint_dir / "checkpoint_19_online_replay.pkl"
    metadata_path = checkpoint_dir / "checkpoint_19_resume_metadata.json"
    assert all(path.is_file() and path.stat().st_size > 0 for path in (checkpoint, replay, metadata_path))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["checkpoint_step"] == metadata["learner_update_steps"] == 19
    assert metadata["wandb_name"] == candidate["wandb_run_name"]
    assert metadata["source_snapshot"]["diagnostic_enabled"] is False
    assert metadata["source_snapshot"]["formal_candidate"] is True
    assert metadata["online_replay_sha256"] == sha256(replay)
    restored = checkpoints.restore_checkpoint(checkpoint_dir, target=None, step=19)
    arrays = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(restored) if hasattr(leaf, "dtype")]
    assert arrays and all(np.all(np.isfinite(array)) for array in arrays)

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.peg_insertion.run_peg_training",
            "--profile",
            "smoke",
            "--formal-candidate",
            "--role",
            "learner",
            "--require-gpu",
            "--dry-run",
            "--resume-from",
            str(checkpoint),
            "--wandb-run-id",
            result["wandb_run_id"],
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=True,
    )
    assert "PEG_TRAINING_DRY_RUN: PASS" in dry_run.stdout

    run_path = f'{result["wandb_entity"]}/{result["wandb_project"]}/{result["wandb_run_id"]}'
    run = wandb.Api(timeout=30).run(run_path)
    history = list(run.scan_history())
    steps = sorted({row.get("_step") for row in history if row.get("_step") is not None})
    assert run.name == candidate["wandb_run_name"] and run.state == "finished"
    assert steps == list(range(20)), steps
    online_losses = [value for row in history for key, value in row.items() if "loss" in key.lower() and value is not None]
    assert online_losses and all(np.isfinite(value) for value in online_losses)
    assert running_training_processes() == []

    verified = dict(result)
    verified.update(
        gate="PASS",
        checkpoint_sha256=sha256(checkpoint),
        replay_sha256=sha256(replay),
        checkpoint_array_leaves=len(arrays),
        wandb_steps=steps,
        residual_training_processes=[],
    )
    tmp = result_dir / "result.json.tmp"
    tmp.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, result_dir / "result.json")
    print("JAX_DEVICES:", jax.devices())
    print("JAX/JAXLIB:", jax.__version__, jaxlib.__version__)
    print("SINGLE_LEARNER_PROCESS: PASS")
    print("NORMAL_RNG_PROGRESSION: PASS")
    print("DIAGNOSTIC_DISABLED: PASS")
    print("UPDATES: 20/20")
    print("FINITE_LOSSES: PASS")
    print("CHECKPOINT_BUNDLE_HASH: PASS")
    print("RESUME_DRY_RUN: PASS")
    print("WANDB_STEPS: 0..19 PASS")
    print("RESIDUAL_TRAINING_PROCESSES: NONE")
    print("PEG_FORMAL_CANDIDATE_GPU_GATE=PASS")


def main() -> int:
    try:
        verify()
    except Exception as error:
        print(f"PEG_FORMAL_CANDIDATE_GPU_GATE=FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
