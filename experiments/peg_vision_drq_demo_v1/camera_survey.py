"""Render every existing Peg camera at canonical reset for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco

from experiments.peg_vision_drq_demo_v1.peg_vision_env import (
    PROJECT_ROOT,
    PegVisionDrQEnvV1,
    experiment_peg_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if PROJECT_ROOT not in output.parents:
        raise ValueError("survey output must be below project root")
    output.mkdir(parents=True, exist_ok=False)
    with experiment_peg_runtime():
        env = PegVisionDrQEnvV1()
        env.reset(seed=0)
        names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
            for idx in range(env.model.ncam)
        ]
        renderer = mujoco.Renderer(env.model, height=128, width=128)
        for name in names:
            renderer.update_scene(env.data, camera=name)
            iio.imwrite(output / f"{name}.png", renderer.render())
        renderer.close()
        env.close()
    (output / "cameras.json").write_text(json.dumps(names, indent=2) + "\n")
    print(json.dumps(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
