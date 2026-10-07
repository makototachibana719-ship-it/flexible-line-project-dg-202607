# Cable Routing Demo Acquisition Plan

Date: 2026-08-21  
Status: project recorder `VALIDATED` by bounded dry run; `DEMO_DATA = MISSING`; `DEMO_SOURCE = SELF_RECORD_REQUIRED`; `DEMO_ACQUISITION = TODO`

This is a static design record only. No demo, rollout, empty pickle, or synthetic trajectory was created.

## Delivered recorder capability

| Capability | Status | Evidence |
|---|---|---|
| Cable keyboard teleoperation | MISSING | No Cable keyboard controller found |
| Joystick control | MISSING | No joystick/gamepad interface found |
| Human control | MISSING | Supplied human-view test targets PickCube, samples random actions, and only binds Space to reset |
| Trajectory/demo recorder | MISSING | No transition recorder or Cable collection entry point found |
| Trajectory save/dataset builder | MISSING | No Cable save or dataset-build implementation found |
| Replay buffer consumer | PRESENT | DrQ learner can accept `demo_path`/RLDS input, but this is loading/training code, not a recorder |
| Actual Cable demo data | MISSING | `resnet10_params.pkl` is network initialization data, not a demo |

## Runtime interface to record

- Environment: `PandaCableRouteVision-v0` from the isolated V2 `franka_sim` package.
- Reset: `(observation, info)`.
- Observation:
  - `state/panda/tcp_pos`: `(3,) float32`
  - `state/panda/tcp_vel`: `(3,) float32`
  - `state/panda/gripper_pos`: `(1,) float32`
  - `images/front`: `(128,128,3) uint8`
  - `images/wrist`: `(128,128,3) uint8`
- Action: normalized xyz delta, `Box(-1,1,(3,),float32)`; gripper remains closed.
- Step: `(next_observation, reward, terminated, truncated, info)`.
- Success evidence: `info["success"]`; supporting keys include `distance_to_slot`, `root_error`, `cable_tip_pos`, and `slot_pos`.

## Required project-owned recorder design

1. Design a bounded teleoperation controller that emits only valid `(3,) float32` actions.
2. Record authentic human-controlled transitions with the native observation dictionaries preserved.
3. Store at least `observations`, `actions`, `next_observations`, `rewards`, `masks`, and `dones`; separately preserve `terminated`, `truncated`, `info["success"]`, and episode boundaries.
4. Define `mask` explicitly against the downstream replay-buffer contract; do not silently treat truncation as task failure or success.
5. Validate observation/image/action shapes and dtypes before accepting an episode.
6. Mark success only from the real environment success field and review failed/aborted trajectories separately.
7. Save data under a project-owned demo-data location only after the recorder implementation is reviewed and explicitly authorized.

## Open design risks

- Current SERL `SERLObsWrapper` compatibility with nested `images/{front,wrist}` remains unresolved.
- The delivered learner accepts demos but does not establish the project recorder's serialization contract by itself.
- Teleoperation mapping, safety bounds, episode acceptance criteria, file format, and dataset versioning still require implementation and validation.

No acquisition is performed as part of MILESTONE 1A-FIX-CABLE-V2.

## MILESTONE 1B-CABLE update

The delivered package still has no recorder. A project-owned keyboard recorder and static validator now exist under `tools/cable_demo/`. Real environment reset/step, TEST_ONLY serialization, schema validation, and insertion through the SERL replay consumer path passed. Interactive keyboard/display usability over a desktop session remains to be preflighted before formal acquisition.
