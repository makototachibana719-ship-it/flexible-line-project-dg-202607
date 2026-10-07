"""Collect bounded visual online transitions without importing the learner stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.peg_vision_drq_demo_v1.peg_vision_env import (
    PROJECT_ROOT,
    experiment_peg_runtime,
    make_object_contract_env,
)
from experiments.peg_vision_drq_demo_v1.reconstruct_visual_demo import (
    contact_flags,
    table_ee_contact,
)


def clone_obs(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if PROJECT_ROOT not in output.parents or output.exists():
        raise RuntimeError("fresh project output required")
    output.mkdir(parents=True)
    rows = []
    max_force = 0.0
    unexpected_seen = link01_seen = table_seen = False
    with experiment_peg_runtime():
        env = make_object_contract_env(max_episode_steps=max(500, args.transitions + 1))
        obs, _ = env.reset(seed=0)
        for _ in range(args.transitions):
            action = np.zeros(7, dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            link01, unexpected, _ = contact_flags(env)
            link01_seen |= bool(link01)
            unexpected_seen |= bool(unexpected)
            table_seen |= table_ee_contact(env)
            max_force = max(max_force, float(info.get("force_norm", 0.0)))
            rows.append(
                {
                    "observations": clone_obs(obs),
                    "actions": action.copy(),
                    "next_observations": clone_obs(next_obs),
                    "rewards": np.float32(reward),
                    "masks": np.float32(1.0 - float(terminated)),
                    "dones": np.bool_(terminated or truncated),
                }
            )
            if terminated or truncated:
                raise RuntimeError("bounded zero-action collection ended unexpectedly")
            obs = next_obs
        env.close()
    data_path = output / "safe_online_visual.pkl"
    with data_path.open("xb") as stream:
        pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
    contact_health = not link01_seen and not unexpected_seen and not table_seen and np.isfinite(max_force)
    if not contact_health:
        raise RuntimeError("safe online collection contact-health failure")
    result = {
        "SAFE_ONLINE_VISUAL_COLLECTION": "PASS",
        "transitions": len(rows),
        "action": "zero Action7",
        "contact_health": bool(contact_health),
        "max_force": max_force,
        "dataset": str(data_path),
        "dataset_sha256": sha(data_path),
        "training_run": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
