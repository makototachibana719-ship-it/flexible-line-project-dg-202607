"""Verify Cable handoff inputs without reading credentials or starting training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOFF = PROJECT_ROOT / "deliverables/gpu_training_handoff/cable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    required = [
        HANDOFF / "README.md",
        HANDOFF / "demo_manifest.json",
        HANDOFF / "environment_spec.md",
        HANDOFF / "baseline_config_snapshot.yaml",
        PROJECT_ROOT / "training/cable/run_cable_training.py",
        PROJECT_ROOT / "evaluation/cable/evaluate_policy.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads((HANDOFF / "demo_manifest.json").read_text(encoding="utf-8"))
    if manifest["totals"] != {"files": 5, "transitions": 10521, "successes": 5}:
        raise AssertionError("Demo manifest totals mismatch")
    for item in manifest["files"]:
        path = PROJECT_ROOT / item["relative_path"]
        if path.stat().st_size != item["size_bytes"]:
            raise AssertionError(f"Demo size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise AssertionError(f"Demo SHA256 mismatch: {path}")
        print(f"DEMO_INTEGRITY: {path.name} PASS")

    config = yaml.safe_load((HANDOFF / "baseline_config_snapshot.yaml").read_text())
    if config["wandb"] != {
        "project": "flexible-line-cable",
        "run_name": "cable_baseline_5demo_seed0",
        "smoke_run_name": "cable_gpu_smoke_5demo_seed0",
        "mode": "online",
    }:
        raise AssertionError("W&B contract mismatch")
    for key in ("checkpoint_dir", "log_dir", "smoke_checkpoint_dir", "smoke_log_dir"):
        output = PROJECT_ROOT / config["outputs"][key]
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing non-empty output: {output}")

    for module in ("jax", "flax", "gym", "gymnasium", "mujoco", "wandb"):
        if importlib.util.find_spec(module) is None:
            raise ImportError(module)
    backend = jax.default_backend()
    devices = [str(device) for device in jax.devices()]
    print(f"JAX_DEVICES: {devices}")
    print(f"JAX_BACKEND: {backend}")
    if args.require_gpu and backend == "cpu":
        raise RuntimeError("GPU required but JAX default backend is CPU")
    print("WANDB_CONFIG: PASS (credentials not inspected)")
    print("OUTPUT_ISOLATION: PASS")
    print("CABLE_HANDOFF_VERIFY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
