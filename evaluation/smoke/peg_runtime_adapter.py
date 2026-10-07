"""Project-owned Peg smoke adapter with isolated generated-XML output."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from evaluation.smoke import base_report, classify_exception, finish, run_lifecycle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT / "external/packages/peg_insertion/peg_insertion"
EXAMPLE = PACKAGE / "code/example/async_rlpd_peg_sim"
MUJOCO_ENV_PARENT = PACKAGE / "code/env"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/peg_insertion"
GENERATED_DIR = RUNTIME_ROOT / "generated"
LOGS_DIR = RUNTIME_ROOT / "logs"


def _assert_project_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    runtime_root = RUNTIME_ROOT.resolve()
    if resolved != runtime_root and runtime_root not in resolved.parents:
        raise ValueError(f"Peg runtime output escapes project staging: {resolved}")
    return resolved


@contextmanager
def isolated_robot_assembler_output(output_dir: Path) -> Iterator[None]:
    """Redirect only RobotAssembler's generated XML, then restore its method."""
    output_dir = _assert_project_runtime_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    from mujoco_env.tasks import base_task

    assembler_class = base_task.RobotAssembler
    original_build = assembler_class.build_robot_scene

    def build_in_project_runtime(self, scene_name, robot_config, _source_output_dir):
        return original_build(self, scene_name, robot_config, output_dir)

    assembler_class.build_robot_scene = build_in_project_runtime
    try:
        yield
    finally:
        assembler_class.build_robot_scene = original_build


def main() -> None:
    report = base_report("Peg Insertion", str(PACKAGE))
    try:
        sys.path.insert(0, str(EXAMPLE))
        sys.path.insert(0, str(MUJOCO_ENV_PARENT))
        from peg_env_adapter import make_peg_state_env

        report["IMPORT"] = "PASS"
        with isolated_robot_assembler_output(GENERATED_DIR):
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
