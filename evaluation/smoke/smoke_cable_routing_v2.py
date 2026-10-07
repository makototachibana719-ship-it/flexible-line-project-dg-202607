"""Minimal lifecycle smoke test for the isolated Cable Routing V2 package."""

from __future__ import annotations

import sys
from pathlib import Path

from evaluation.smoke import base_report, classify_exception, finish, run_lifecycle


PACKAGE = Path(
    "external/packages/cable_routing_v2/"
    "CableRoute_最终交接包_V2_20260820_221714"
).resolve()
FRANKA_SIM_PARENT = PACKAGE / "franka_sim"


def main() -> None:
    report = base_report("Cable Routing V2", str(PACKAGE))
    try:
        sys.path.insert(0, str(FRANKA_SIM_PARENT))
        import franka_sim  # noqa: F401 - import performs Gym registration
        import gym

        report["IMPORT"] = "PASS"
        run_lifecycle(
            report,
            lambda: gym.make(
                "PandaCableRouteVision-v0",
                render_mode="rgb_array",
            ),
            steps=1,
        )
    except Exception as exc:
        classify_exception(report, exc)
    finish(report)


if __name__ == "__main__":
    main()
