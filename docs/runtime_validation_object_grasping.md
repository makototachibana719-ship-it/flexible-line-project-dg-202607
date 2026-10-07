# Object Grasping Runtime Validation

Date: 2026-08-15  
Runtime: `serl`  
Scope: import/create/reset/one-step/close only; no DPRL/DrQ policy, checkpoint, demo, training, or evaluation.

## Result

`PASS` — `ENVIRONMENT_SMOKE_VALIDATED`.

The first SSH/headless attempt without a backend override failed at create with GLFW `gladLoadGL error` because `DISPLAY` is absent. A second attempt in the same environment with process-local `MUJOCO_GL=egl` completed the full lifecycle. This did not modify Conda or source files.

```text
PROCESS: Object Grasping
PYTHON_PATH: /home/ubuntu/miniconda3/envs/serl/bin/python
PYTHON_VERSION: 3.10.20
ENV_CREATE: PASS
RESET: PASS
OBS_TYPE: dict
OBS_KEYS: ['state', 'state/panda/tcp_pos', 'state/panda/tcp_vel', 'state/panda/gripper_pos', 'images', 'images/front', 'images/wrist']
OBS_SHAPES: {'state/panda/tcp_pos': '(3,)', 'state/panda/tcp_vel': '(3,)', 'state/panda/gripper_pos': '()', 'images/front': '(128, 128, 3)', 'images/wrist': '(128, 128, 3)'}
OBS_DTYPES: {'state/panda/tcp_pos': 'float32', 'state/panda/tcp_vel': 'float32', 'state/panda/gripper_pos': 'float32', 'images/front': 'uint8', 'images/wrist': 'uint8'}
ACTION_SPACE: Box(-1.0, 1.0, (4,), float32)
ACTION_SHAPE: (4,)
STEP: PASS (1)
STEP_RETURN_LENGTH: 5
REWARD_TYPE: float64
REWARD_VALUE: 1.4951120665357853e-05
TERMINATED: False
TRUNCATED: False
INFO_KEYS: ['block_height', 'lift_height', 'success']
CLOSE: PASS
RESULT: PASS
```

## Runtime findings

- Two actual RGB cameras are PRESENT: `images/front` and `images/wrist`, both `(128,128,3)` uint8.
- State components total seven scalar values: TCP position 3, TCP velocity 3, gripper position 1.
- Actual `panda/gripper_pos` array shape is scalar `()`, while the declared observation space expects `(1,)`.
- Gym emitted warnings that reset and step observations are not within `observation_space`; this mismatch is recorded and not modified.
- Action shape `(4,)` and five-value step API are runtime confirmed.
- On the single zero-action step, `success` was false, `terminated` false, and `truncated` false.

This validates only the environment lifecycle, not checkpoint, policy, demo, DPRL identity, or success rate.
