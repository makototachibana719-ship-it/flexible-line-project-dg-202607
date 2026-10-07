# Cable Routing Runtime Validation

Date: 2026-08-21  
Package: Cable Routing V2  
Runtime: `serl` with `MUJOCO_GL=egl`  
Scope: import/create/reset/one-step/close only; no policy, checkpoint, demo, training, or evaluation.

## Result

`PASS` — `ENVIRONMENT_SMOKE_VALIDATED`.

V1 history: import passed but create failed because `franka_emika_panda/assets/link1.stl` and its asset tree were absent. V2 supplies the tree and completed the lifecycle.

```text
PROCESS: Cable Routing V2
PYTHON_PATH: /home/ubuntu/miniconda3/envs/serl/bin/python
PYTHON_VERSION: 3.10.20
GYM: 0.26.2
GYMNASIUM: 1.3.0
MUJOCO: 2.3.7
JAX: 0.4.35
FLAX: 0.8.5
IMPORT: PASS
ENV_CREATE: PASS
RESET: PASS
RESET_RETURN_TYPE: tuple(len=2)
OBS_TYPE: dict
OBS_KEYS: ['state', 'state/panda/tcp_pos', 'state/panda/tcp_vel', 'state/panda/gripper_pos', 'images', 'images/front', 'images/wrist']
OBS_SHAPES: {'state/panda/tcp_pos': '(3,)', 'state/panda/tcp_vel': '(3,)', 'state/panda/gripper_pos': '(1,)', 'images/front': '(128, 128, 3)', 'images/wrist': '(128, 128, 3)'}
OBS_DTYPES: {'state/panda/tcp_pos': 'float32', 'state/panda/tcp_vel': 'float32', 'state/panda/gripper_pos': 'float32', 'images/front': 'uint8', 'images/wrist': 'uint8'}
ACTION_SPACE: Box(-1.0, 1.0, (3,), float32)
ACTION_SHAPE: (3,)
STEP: PASS (1)
STEP_RETURN_LENGTH: 5
REWARD_TYPE: float: 0.07527805155610397
TERMINATED: False
TRUNCATED: False
INFO_KEYS: ['cable_tip_pos', 'distance_to_slot', 'root_error', 'slot_pos', 'success']
CLOSE: PASS
RESULT: PASS
```

## Runtime selection

V2 declares Python 3.10.20, MuJoCo 2.3.7, Gym 0.26.2, Gymnasium 1.3.0, and JAX 0.4.35. The existing `serl` Conda environment matches those versions exactly and was therefore safer than the primary general candidate `serl_mujoco`, which has MuJoCo 3.10.0.

## Runtime notes

- A zero `(3,) float32` action was used for one step.
- Both RGB cameras and the complete seven-value state were runtime confirmed.
- The observed step did not succeed or terminate/truncate.
- `env.close()` completed and was recorded PASS. After report completion, MuJoCo's EGL context destructor emitted a non-fatal cleanup exception; headless renderer cleanup remains a known risk.
- Mesa could not create `/home/ubuntu/.cache/mesa_shader_cache` in the restricted environment and disabled that cache; lifecycle execution continued.

## Validation boundary

Environment smoke validation does not validate DrQ checkpoints, policy behavior, demo data, training results, success rate, or the SERL observation wrapper.

`DEMO_DATA = MISSING`  
`DEMO_SOURCE = SELF_RECORD_REQUIRED`  
`DEMO_ACQUISITION = TODO`
