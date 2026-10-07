"""GPU, Object ResNet-10 DrQ initialization, and inference preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess

import jax
import numpy as np

from experiments.peg_vision_drq_demo_v1.drq_runtime import (
    OBJECT_RESNET10,
    make_object_drq_agent,
)
from experiments.peg_vision_drq_demo_v1.peg_vision_env import PROJECT_ROOT


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if PROJECT_ROOT not in output.parents or output.exists():
        raise RuntimeError("fresh project output required")
    output.mkdir(parents=True)
    with args.demo.open("rb") as stream:
        transition = pickle.load(stream)[0]
    observation = transition["observations"]
    action_template = np.zeros(7, dtype=np.float32)
    smi = subprocess.run(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"JAX backend is {backend}, expected gpu")
    agent = make_object_drq_agent(0, observation, action_template)
    action = agent.sample_actions(observations=observation, seed=None, argmax=True)
    action = np.asarray(jax.device_get(action))
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid inference action: {action.shape}")
    leaves = jax.tree_util.tree_leaves(agent.state.params)
    result = {
        "GPU_PREFLIGHT": "PASS",
        "nvidia_smi": smi,
        "jax_default_backend": backend,
        "jax_devices": [str(device) for device in jax.devices()],
        "drq_agent": type(agent).__name__,
        "encoder": "resnet-pretrained",
        "resnet10_path": str(OBJECT_RESNET10),
        "resnet10_sha256": sha(OBJECT_RESNET10),
        "parameter_count": int(sum(np.prod(leaf.shape) for leaf in leaves)),
        "action_shape": list(action.shape),
        "action_finite": bool(np.isfinite(action).all()),
        "action_within_normalized_bounds": bool(np.all(np.abs(action) <= 1.0)),
        "sample_action": action.tolist(),
        "fresh_initialization": True,
        "training_run": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
