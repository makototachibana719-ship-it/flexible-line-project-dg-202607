"""Minimal lifecycle smoke test for the delivered Object Grasping environment."""

from __future__ import annotations

import sys
from pathlib import Path

from evaluation.smoke import base_report, classify_exception, finish, run_lifecycle


PACKAGE = Path("external/packages/object_grasping/object_grasp_package/code/env").resolve()


def main() -> None:
    report = base_report("Object Grasping", str(PACKAGE))
    try:
        sys.path.insert(0, str(PACKAGE))
        import gym
        import franka_sim  # noqa: F401 - registration side effect is under test

        report["IMPORT"] = "PASS"
        run_lifecycle(
            report,
            lambda: gym.make("PandaPickCubeVision-v0"),
            steps=1,
        )
    except Exception as exc:
        classify_exception(report, exc)
    finish(report)


if __name__ == "__main__":
    main()
