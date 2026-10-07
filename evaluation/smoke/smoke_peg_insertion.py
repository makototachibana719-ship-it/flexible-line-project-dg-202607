"""Minimal lifecycle smoke test for the delivered Peg Insertion adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from evaluation.smoke import base_report, classify_exception, finish, run_lifecycle


PACKAGE = Path("external/packages/peg_insertion/peg_insertion").resolve()
EXAMPLE = PACKAGE / "code/example/async_rlpd_peg_sim"
MUJOCO_ENV_PARENT = PACKAGE / "code/env"


def main() -> None:
    report = base_report("Peg Insertion", str(PACKAGE))
    try:
        sys.path.insert(0, str(EXAMPLE))
        sys.path.insert(0, str(MUJOCO_ENV_PARENT))
        from peg_env_adapter import make_peg_state_env

        report["IMPORT"] = "PASS"
        run_lifecycle(
            report,
            lambda: make_peg_state_env(
                render_mode="rgb_array",
                max_episode_steps=5,
            ),
            steps=1,
        )
    except Exception as exc:
        classify_exception(report, exc)
    finish(report)


if __name__ == "__main__":
    main()
