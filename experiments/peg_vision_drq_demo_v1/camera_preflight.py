"""No-training camera and observation-contract preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from experiments.peg_vision_drq_demo_v1.peg_vision_env import (
    FRONT_CAMERA_NAME,
    PROJECT_ROOT,
    WRIST_CAMERA_NAME,
    experiment_peg_runtime,
    make_object_contract_env,
)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def image_metrics(image: np.ndarray) -> dict:
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "finite": bool(np.isfinite(image).all()),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "unique_colors": int(np.unique(image.reshape(-1, 3), axis=0).shape[0]),
        "sha256": digest(image),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if PROJECT_ROOT not in output.parents:
        raise ValueError("preflight output must be below project root")
    output.mkdir(parents=True, exist_ok=False)

    with experiment_peg_runtime():
        env = make_object_contract_env(max_episode_steps=500)
        first, info_first = env.reset(seed=0)
        second, info_second = env.reset(seed=0)
        zero_next, reward, terminated, truncated, step_info = env.step(
            np.zeros(7, dtype=np.float32)
        )
        front = np.asarray(first["front"])[0]
        wrist = np.asarray(first["wrist"])[0]
        iio.imwrite(output / "front_reset.png", front)
        iio.imwrite(output / "wrist_reset.png", wrist)
        checks = {
            "state_contract": list(first["state"].shape) == [1, 20]
            and first["state"].dtype == np.float32,
            "front_contract": list(first["front"].shape) == [1, 128, 128, 3]
            and first["front"].dtype == np.uint8,
            "wrist_contract": list(first["wrist"].shape) == [1, 128, 128, 3]
            and first["wrist"].dtype == np.uint8,
            "action_contract": list(env.action_space.shape) == [7]
            and env.action_space.dtype == np.float32,
            "front_nonblank": float(front.std()) > 1.0,
            "wrist_nonblank": float(wrist.std()) > 1.0,
            "same_seed_state_stable": bool(
                np.array_equal(first["state"], second["state"])
            ),
            "same_seed_front_stable": bool(
                np.array_equal(first["front"], second["front"])
            ),
            "same_seed_wrist_stable": bool(
                np.array_equal(first["wrist"], second["wrist"])
            ),
            "zero_step_single_transition": bool(
                list(zero_next["state"].shape) == [1, 20]
                and np.isfinite(reward)
            ),
        }
        result = {
            "PEG_VISION_CAMERA_PREFLIGHT": "PASS" if all(checks.values()) else "FAIL",
            "training_run": False,
            "front_camera": FRONT_CAMERA_NAME,
            "wrist_camera": WRIST_CAMERA_NAME,
            "image_chunk_length": 1,
            "checks": checks,
            "front": image_metrics(front),
            "wrist": image_metrics(wrist),
            "reward_after_zero_action": float(reward),
            "terminated_after_zero_action": bool(terminated),
            "truncated_after_zero_action": bool(truncated),
            "reset_info_keys": sorted(set(info_first) | set(info_second)),
            "step_info_keys": sorted(step_info),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        env.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["PEG_VISION_CAMERA_PREFLIGHT"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
