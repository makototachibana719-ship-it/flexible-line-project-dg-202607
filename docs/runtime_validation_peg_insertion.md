# Peg Insertion Runtime Validation

Date: 2026-08-15  
Runtime: `serl_mujoco` with `MUJOCO_GL=egl`  
Scope: factory/adapter import/create/reset/one-step/close only; no SAC/RLPD/BC policy, checkpoint, demo, training, or evaluation.

## Result

`PASS` — `ENVIRONMENT_SMOKE_VALIDATED`.

```text
PROCESS: Peg Insertion
PYTHON_PATH: /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python
PYTHON_VERSION: 3.10.20
ENV_CREATE: PASS
RESET: PASS
RESET_RETURN_TYPE: tuple(len=2)
OBS_TYPE: ndarray
OBS_KEYS: []
OBS_SHAPES: {'<root>': '(20,)'}
OBS_DTYPES: {'<root>': 'float32'}
ACTION_SPACE: Box(-1.0, 1.0, (7,), float32)
ACTION_SHAPE: (7,)
STEP: PASS (1)
STEP_RETURN_LENGTH: 5
REWARD_TYPE: float: -13.793326324537661
TERMINATED: False
TRUNCATED: False
INFO_KEYS: ['axial_error', 'elapsed_steps', 'episode', 'force_norm', 'is_success', 'lateral_error', 'orientation_error', 'position_error', 'terminated', 'truncated']
CLOSE: PASS
RESULT: PASS
```

## XML generation call chain

1. Delivered `make_peg_state_env()` constructs `PegStateSERLEnv`.
2. `PegStateSERLEnv.__init__()` calls `make_aubo_i5_assemble_hole_env()`.
3. The factory constructs `PegInsertionTask`.
4. `BaseTask.__init__()` obtains package-relative assets from `get_assets_path()` and derives `output_dir = assets_dir.parent / "tasks" / self.name`.
5. `RobotAssembler.build_robot_scene(scene_name, robot_config, output_dir)` derives `aubo_i5_assemble_hole.xml` and writes it with `ElementTree.write()`.
6. `make_sim_env()` loads the generated path from `task.model_path`.

The original API exposes no `output_dir`, `xml_path`, runtime-root environment variable, or cwd-derived override. The original destination is package-relative and resolves to `/home/ubuntu/mujoco_env/mujoco_env/tasks/PegInsertion/` because the Conda editable installation resolves `mujoco_env` there.

## Project-side isolation

`evaluation/smoke/peg_runtime_adapter.py` implements solution C without modifying delivered or external source. During environment construction only, it wraps the `RobotAssembler` class referenced by `BaseTask` and substitutes the project-owned output directory. A `finally` block restores the original method immediately after construction/lifecycle execution.

Generated output:

```text
/home/ubuntu/flexible_line_project/runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml
```

The generated XML keeps relative read-only references to the original assets. Before/after path-size-mtime fingerprints for `external/packages/peg_insertion` and `/home/ubuntu/mujoco_env` were identical (`6d5d4b631d2edbfa332942a6f633c3d97b1f4dbf13fa23da83e7b5423daa2795`).

## Scope boundary

This validates only the environment lifecycle and current runtime compatibility. Checkpoints, policies, demos, reward correctness, success rate, and formal evaluation remain UNVERIFIED.
