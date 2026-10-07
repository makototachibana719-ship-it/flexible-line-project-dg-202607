#!/usr/bin/env python3
"""Read-only/fail-closed preflight for the frozen Peg F3 V2 formal 5K run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python")
CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
LAUNCHER = ROOT / "scripts/run_peg_v2_canonical_wsl.sh"
HARNESS = ROOT / "scripts/run_peg_f3_v2_rlpd_smoke.py"
EXPECTED_DATASET_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
FIELDS = ("observations", "actions", "next_observations", "rewards", "masks", "dones")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    require(ROOT in evidence.parents and not evidence.exists(), "fresh evidence directory required")
    cfg = yaml.safe_load(CONFIG.read_text())
    dataset = ROOT / cfg["demo"]["path"]
    manifest_path = ROOT / cfg["demo"]["manifest"]
    run_dir = ROOT / cfg["outputs"]["run_dir"]
    runtime_dir = run_dir / "run"
    checks = {}
    checks["canonical_launcher"] = LAUNCHER.is_file() and HARNESS.is_file()
    checks["python_env"] = Path(sys.executable).resolve() == PYTHON.resolve() and PYTHON.is_file()
    import jax
    backend = jax.default_backend()
    checks["gpu"] = backend == "gpu"
    checks["dataset_hash"] = dataset.is_file() and sha(dataset) == EXPECTED_DATASET_SHA == cfg["demo"]["sha256"]
    rows = pickle.loads(dataset.read_bytes())
    manifest = json.loads(manifest_path.read_text())
    checks["dataset_schema"] = (
        isinstance(rows, list) and len(rows) == 1921 and cfg["demo"]["episodes"] == 11 and
        manifest.get("episodes") == 11 and manifest.get("transitions") == 1921 and
        manifest.get("v2_only") is True and manifest.get("dataset_sha256") == EXPECTED_DATASET_SHA and
        all(set(row) == set(FIELDS) and row["observations"].shape == row["next_observations"].shape == (20,)
            and row["actions"].shape == (7,) and row["observations"].dtype == np.float32
            and row["next_observations"].dtype == np.float32 and row["actions"].dtype == np.float32
            and all(np.isfinite(np.asarray(row[key])).all() for key in FIELDS) for row in rows))
    from training.peg_insertion.observation_contract import V2, validate_config
    checks["v2_contract"] = validate_config(cfg, for_training=True) == V2
    checks["fresh_init"] = cfg["fresh_initialization"] is True and cfg["load_legacy_checkpoint"] is False
    def string_values(value):
        if isinstance(value, dict):
            return [item for nested in value.values() for item in string_values(nested)]
        if isinstance(value, list):
            return [item for nested in value for item in string_values(nested)]
        return [value.lower()] if isinstance(value, str) else []
    configured_values = string_values(cfg)
    checks["no_v1"] = (cfg["load_legacy_checkpoint"] is False and
                        cfg["demo"]["observation_contract_version"] == V2 and
                        not any(any(token in value for token in
                                    ("checkpoint_1000", "v1_legacy", "--resume", "resume_from"))
                                for value in configured_values))
    training = cfg["training"]
    checks["sampling"] = (training["demo_batch_size"] == training["online_batch_size"] == 32 and
                           training["total_batch_size"] == 64 and training["demo_fraction"] == .5 and
                           training["utd_ratio"] == 1 and training["critic_actor_ratio"] == 1)
    checks["checkpoint_plan"] = (cfg["formal"]["target_updates"] == 5000 and
                                  training["checkpoint_period"] == 1000 and
                                  cfg["formal"]["checkpoint_targets"] == [1000, 2000, 3000, 4000, 5000])
    actor_source = (ROOT / "training/peg_insertion/actor_shutdown.py").read_text()
    checks["shutdown_fix"] = all(token in actor_source for token in
                                  ("BoundedTrainerClient", "_close_req_rep_client", "linger"))
    checks["output_isolation"] = (run_dir.is_dir() and not runtime_dir.exists() and
                                   "smoke" not in str(run_dir).lower())
    free = shutil.disk_usage(run_dir).free
    checks["disk_space"] = free >= 10 * 1024**3
    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True).stdout
    residual = [line.strip() for line in ps.splitlines()
                if ("run_peg_f3_v2_rlpd_smoke.py" in line or "async_sac_state_sim.py" in line)
                and str(os.getpid()) not in line]
    checks["residual"] = not residual
    expected_command = (f"{LAUNCHER} formal-run {CONFIG} {runtime_dir} --authorize-formal-5k")
    probe_env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                     XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl",
                     WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1",
                     PYTHONPATH=f"{ROOT}:{ROOT / 'external/serl/serl_launcher'}:{ROOT / 'external/serl/franka_sim'}")
    parse_probe = subprocess.run(
        [str(PYTHON), str(HARNESS), "--role", "preflight", "--formal-5k-config", str(CONFIG),
         "--output", str(runtime_dir)], cwd="/tmp", env=probe_env, capture_output=True,
        text=True, timeout=180)
    guard_probe = subprocess.run(
        [str(LAUNCHER), "formal-run", str(CONFIG), str(runtime_dir)], cwd="/tmp",
        capture_output=True, text=True, timeout=30)
    checks["launch_command"] = (all(path.is_absolute() for path in (LAUNCHER, CONFIG, runtime_dir)) and
                                parse_probe.returncode == 0 and guard_probe.returncode == 2 and
                                "requires --authorize-formal-5k" in guard_probe.stderr)
    failed = next((key for key, value in checks.items() if not value), None)
    require(failed is None, "first failed gate: " + str(failed))
    evidence.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "PREPARED_NOT_STARTED.json"
    marker.write_text(json.dumps({"training_started": False, "config_sha256": sha(CONFIG)}, indent=2) + "\n")
    result = {
        "PEG_F3_V2_FORMAL_5K_PREFLIGHT": "PASS", "CANONICAL_LAUNCHER": "PASS",
        "PYTHON_ENV": "PASS", "GPU_BACKEND": backend, "DATASET_HASH": "PASS",
        "DATASET_SCHEMA": "PASS", "V2_CONTRACT": "PASS", "FRESH_INIT": "PASS",
        "NO_V1_CONTAMINATION": "PASS", "SAMPLING_CONTRACT": "PASS",
        "CHECKPOINT_PLAN": "PASS", "SHUTDOWN_FIX_BINDING": "PASS",
        "OUTPUT_ISOLATION": "PASS", "DISK_SPACE": "PASS", "DISK_FREE_BYTES": free,
        "RESIDUAL_PROCESSES": "NONE", "FORMAL_CONFIG_PATH": str(CONFIG),
        "FORMAL_CONFIG_SHA256": sha(CONFIG), "FORMAL_RUN_DIR": str(run_dir),
        "FORMAL_LAUNCH_COMMAND": expected_command, "FORMAL_LAUNCH_COMMAND_VALIDATED": "PASS",
        "TRAINING_STARTED": False, "FORMAL_5K": "READY", "checks": checks,
        "PARSE_PROBE_RETURN_CODE": parse_probe.returncode,
        "UNAUTHORIZED_GUARD_RETURN_CODE": guard_probe.returncode,
    }
    (evidence / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (evidence / "command.txt").write_text(expected_command + "\n")
    (evidence / "config_snapshot.yaml").write_bytes(CONFIG.read_bytes())
    (evidence / "config_sha256.txt").write_text(sha(CONFIG) + "\n")
    (evidence / "REPORT.md").write_text("# Peg F3 V2 Formal 5K Preflight\n\n```json\n" +
                                        json.dumps(result, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
