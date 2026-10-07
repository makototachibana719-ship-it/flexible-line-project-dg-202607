# MILESTONE 1A Runtime Validation Summary

Date: 2026-08-15

## Runtime environment status

Runtime Environment Discovery is VALIDATED.

| Candidate | Python | Gym | Gymnasium | MuJoCo | JAX | Flax | Use |
|---|---:|---:|---:|---:|---:|---:|---|
| `serl_mujoco` | 3.10.20 | 0.26.2 | 1.3.0 | 3.10.0 | 0.4.35 | 0.8.5 | Primary; Cable and Peg smoke |
| `serl` | 3.10.20 | 0.26.2 | 1.3.0 | 2.3.7 | 0.4.35 | 0.8.5 | Object version-matched runtime |

Earlier `/usr/bin/python3` results are INVALID as project dependency conclusions.

## Validation matrix

| Process | Import | Create | Reset | Step | Close | Result | Environment status |
|---|---|---|---|---|---|---|---|
| Cable Routing V2 | PASS | PASS with completed Franka assets and EGL | PASS | PASS (1) | PASS | PASS | ENVIRONMENT_SMOKE_VALIDATED |
| Object Grasping | PASS | PASS with headless EGL | PASS | PASS (1) | PASS | PASS | ENVIRONMENT_SMOKE_VALIDATED |
| Peg Insertion | PASS | PASS with project runtime staging | PASS | PASS (1) | PASS | PASS | ENVIRONMENT_SMOKE_VALIDATED |

## Actual runtime interfaces

Cable Routing V2 and Object Grasping reached reset and step. Cable V2 runtime interface:

- Observation: nested state plus `images/front` and `images/wrist`.
- State: TCP position `(3,)`, TCP velocity `(3,)`, gripper position `(1,)`, all float32.
- Images: front/wrist `(128,128,3)` uint8.
- Action: `Box(-1,1,(3,),float32)`.
- Step: five values; observed reward float, terminated false, truncated false, and `success` plus distance/root/position details in info.

Object Grasping runtime interface:

- Observation: nested dict with state and two images.
- State: TCP position `(3,)` float32, TCP velocity `(3,)` float32, gripper scalar `()` float32; seven total values.
- Images: front/wrist `(128,128,3)` uint8.
- Action: `Box(-1,1,(4,),float32)`.
- Step: five values; observed reward float64, terminated false, truncated false, info keys `block_height`, `lift_height`, `success`.
- Observation-space membership warnings occurred because actual gripper scalar shape differs from declared `(1,)`.

Peg runtime confirmed a state-only `(20,)` float32 observation, `(7,)` float32 action, and five-value step API. The observed step returned a Python float reward, `terminated=False`, `truncated=False`, and task/error metrics in `info`.

Cable V1's missing-asset failure is superseded for current use by the isolated V2 lifecycle result. Cable demo data remains MISSING and acquisition remains TODO.

Peg XML output isolation is implemented by the project-side `evaluation/smoke/peg_runtime_adapter.py`. Generated XML is confined to `runtime/peg_insertion/generated/`; external assets remain read-only references.

No source, Conda environment, checkpoint, demo, reward/success logic, policy, or external repository was modified.
