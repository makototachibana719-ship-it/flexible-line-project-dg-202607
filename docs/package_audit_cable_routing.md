# Cable Routing Package Audit

Overall status: UNVERIFIED. Static presence/absence findings are validated; runtime behavior and model quality are not.

## Inventory

| Component | Status | Evidence / finding |
|---|---|---|
| README | MISSING | No README file delivered; manifest and unchecked handover form are PRESENT. |
| Python/environment | PRESENT | `panda_cable_route_gym_env.py`; `PandaCableRouteGymEnv`. |
| Registration | PRESENT | `PandaCableRoute-v0` and `PandaCableRouteVision-v0`, max 500 steps. |
| MuJoCo model/assets | PRESENT | `cable_route.xml`, `arena.xml`, `panda.xml`; runtime asset resolution UNVERIFIED. |
| Observation | PRESENT | Native nested Dict: state plus optional `images/front,wrist`. |
| Action | PRESENT | float32 Box `[-1,1]`, shape `(3,)`: xyz deltas, closed gripper. |
| reset/step | PRESENT | Gymnasium-style `(obs, info)` and five-value step. Runtime UNVERIFIED. |
| Reward | PRESENT | `exp(-10*distance_to_slot)` plus `5.0` on success. |
| Success | PRESENT | Cable-tip to slot-center distance `< 0.03 m`; returned as `info["success"]`. |
| termination/truncation | PRESENT | Success terminates; time limit truncates. |
| Demo Recorder | MISSING | No recorder/collection script exists in the package; project-owned recorder design is TODO. |
| Demo Data | MISSING | Human-confirmed intentional absence, not a packaging omission. `resnet10_params.pkl` is network initialization data, not a demo. |
| Demo Acquisition | TODO | The project will record authentic Cable Routing demos after environment validation and recorder implementation; not waiting for teammate resend. |
| Checkpoints | UNVERIFIED | Non-empty `cable_route_shutdown_test` (steps 0/10/20) and `cable_route_smoke` (0/100/200); `cable_route_v0` and `cable_route_v0_20k` are empty. |
| DRQ/SERL actor/learner | PRESENT | One async DrQ program plus actor/learner launch scripts; no delivered modified SERL tree. |
| Evaluation | UNVERIFIED | Generic 5-episode evaluation call exists; acceptance checklist is unchecked and no formal metrics are supplied. |
| W&B | PRESENT | Logger code and environment records exist; run identity/results are MISSING. |
| Requirements/environment | PRESENT | Conda history/full YAML, pip freeze, system versions. |

## Native and learner-facing interface

- Native state keys: `panda/tcp_pos (3)`, `panda/tcp_vel (3)`, `panda/gripper_pos (1)`, all float32.
- Native image keys: nested `images/front` and `images/wrist`, each `(128,128,3)` uint8 according to renderer configuration/code.
- Training intends `SERLObsWrapper` then `ChunkingWrapper(obs_horizon=1)`, yielding the expected learner-facing pattern `state (1,7)`, `front (1,128,128,3)`, `wrist (1,128,128,3)` only if a compatible wrapper flattens nested images.
- Runtime transition keys are `observations`, `actions`, `next_observations`, `rewards`, `masks`, `dones`; `mask = 1 - terminated`, while `dones = terminated or truncated`.

## Blocking findings

- Current project SERL wrapper expects singular `image`, not nested `images`; direct use is INCOMPATIBLE.
- No actual demo or demo recorder is delivered. This is confirmed package scope, not a missing resend; recorder design and acquisition are project TODO items.
- No candidate checkpoint is identified as final; two expected named directories are empty.
- No independent success-rate evaluation evidence exists.
