# Cable Routing Action Pipeline Diagnosis

Date: 2026-08-21  
Milestone: MILESTONE 1B-CABLE-ACTION-PIPELINE-DIAGNOSIS  
Scope: static trace plus bounded normal-`env.step()` diagnostics; no source/controller/state mutation and no Demo output.

## Result

Primary classification: `A. ACTION_MAGNITUDE_TOO_SMALL`  
Secondary classification: `F. TCP_OBSERVATION_ERROR` in the project recorder's debug display only  
TCP debug display: `FIXED` by MILESTONE 1B-CABLE-TELEOP-MINIMAL-FIX  
Robot teleop motion: `OPERATOR_VALIDATION_REQUIRED`

The action reaches the environment, target, operational-space controller, actuator ctrl, MuJoCo simulation, joints, and TCP. The current 0.05 teleop magnitude changes the Cartesian target by only 0.05 mm per control step. Separately, the recorder incorrectly labels the first three values of flattened state as TCP; the first value is actually gripper position.

## Static Action Pipeline

1. Project recorder `tools/cable_demo/record_cable_demo_viewer.py`, `run_interactive()`:
   - `action_from_pressed_keys()` creates `(3,) float32`.
   - the action is clipped to `env.action_space` and passed unchanged to `env.step(action)`.
2. Project wrapper `tools/cable_demo/__init__.py`, `make_cable_training_env()`:
   - `ChunkingWrapper` has `act_exec_horizon=None`; runtime evidence shows the action reaches the base environment.
3. Cable V2 `franka_sim/franka_sim/envs/panda_cable_route_gym_env.py`, `PandaCableRouteGymEnv.step()`, lines 286–371:
   - lines 297–311 convert to float32, enforce `(3,)`, and clip to `[-1,1]`;
   - line 316 reads `pos = data.mocap_pos[0]`;
   - line 318 computes `dpos = action * self._action_scale`;
   - lines 320–325 clip `pos + dpos` to Cartesian bounds and write `data.mocap_pos[0]`;
   - lines 333–343 call `opspace(..., pos=data.mocap_pos[0], ori=data.mocap_quat[0])`;
   - lines 345–350 write torque to seven Panda actuator ctrl entries and call `mujoco.mj_step()`.
4. Cable V2 `franka_sim/franka_sim/controllers/opspace.py`, `opspace()`, lines 59–180:
   - lines 75–78 assign the passed mocap position to `x_des`;
   - lines 112–124 compute the `pinch` site Jacobian;
   - lines 126–135 apply Cartesian position PD with gains `(200,200,200)` and critical damping;
   - lines 150–176 map task acceleration through inertia/Jacobian and add nullspace joint control;
   - lines 178–180 add gravity compensation and return joint torque `tau`.
5. Cable V2 observation, `panda_cable_route_gym_env.py`, `_compute_observation()`, lines 391–423:
   - TCP comes from sensor `2f85/pinch_pos`;
   - XML defines that sensor as `<framepos ... objtype="site" objname="pinch"/>`;
   - runtime confirms sensor data equals `data.site_xpos[_pinch_site_id]`.

There is no `use_target_as_ctrl` option in this implementation. The persistent target is `data.mocap_pos[0]`; `mocap_quat[0]` is held constant. Action does not directly modify qpos, qvel, or ctrl. Controller output subsequently modifies ctrl and physics modifies qpos/TCP.

## Configuration

- `action_scale`: scalar `0.001 m` for X, Y, and Z.
- Cartesian bounds: lower `[0.2,-0.3,0.0]`, upper `[0.6,0.3,0.5]` m.
- `control_dt`: 0.02 s.
- `physics_dt`: 0.002 s.
- physics substeps per control step: 10.
- Panda actuators: seven direct motors.
- ctrl-limited: true for all seven.
- ctrl ranges: joints 1–4 `[-87,87]`, joints 5–7 `[-12,12]`.
- motor gain parameter: 1.0.
- explicit force limiting: false; `actuator_forcerange` is unused/zero.

## Magnitude Calculation

For action magnitude 0.05:

```text
0.05 * 0.001 m = 0.00005 m = 0.05 mm per control step
20 steps * 0.00005 m = 0.001 m = 1 mm target displacement
```

At 50 control steps/s, holding one key requests only 2.5 mm/s target velocity. This is small relative to scene size, cable motion, controller transient error, and visual resolution.

## Single-Step Runtime Evidence

All six directional cases changed target/mocap on exactly the requested axis by approximately `±5.0000002e-05 m`. Thus:

- wrapper action loss: not observed;
- target update failure: not observed;
- target reset inside step: not observed;
- controller input mismatch: not observed;
- ctrl overwrite to zero: not observed;
- qpos frozen: not observed.

Example X-positive step:

- raw action: `[0.05,0,0]`;
- target/mocap delta: `[+5.0000002e-05,0,0]` m;
- TCP delta: approximately `[-2.04e-06,-3.57e-06,-3.07e-06]` m during this transient step;
- ctrl delta: nonzero on the Panda motors, including joint-2 `+2.153e-02`;
- qpos delta: nonzero on all seven joints at roughly `4e-07` to `1.17e-05` rad.

Zero action itself produced micrometer-scale drift, confirming that single-step TCP movement at magnitude 0.05 is comparable to ongoing dynamics and cannot be judged visually from one step.

## 20-Step Held-Action Evidence

### X positive

- target/mocap delta: `[+0.001,0,0]` m exactly;
- TCP sensor/site delta: `[+0.0001371462,+0.0002398250,-0.0003527445]` m;
- qpos delta: nonzero, with joint-2 approximately `+0.00198545` rad;
- target was moving away from an initial controller offset, so it was not fully tracked in the bounded 0.4-second window.

### X negative

- target/mocap delta: `[-0.001,0,0]` m exactly;
- TCP sensor/site delta: `[-0.0011671995,+0.0002392335,-0.0003328894]` m;
- qpos delta: nonzero, including joint-2 approximately `-0.00224015` rad and joint-6 approximately `-0.00249759` rad.

The negative case visibly tracks the requested X direction and magnitude. The positive/negative asymmetry over only 20 steps is consistent with initial tracking offset, transient dynamics, gravity/cable loading, and coupled Cartesian control; it is not evidence that the controller ignores the target.

## TCP Debug Display Error

The learner-facing flattened state is runtime-confirmed as:

```text
[gripper_pos, tcp_x, tcp_y, tcp_z, tcp_vx, tcp_vy, tcp_vz]
```

At reset:

```text
flattened state = [0.0, 0.30796209, -0.000147215, 0.44400612, 0, 0, 0]
true TCP sensor = [0.30796210, -0.000147215, 0.44400611]
true TCP site   = [0.30796210, -0.000147215, 0.44400611]
```

The recorder currently prints `state[:3]` as TCP, producing approximately `[0.0,0.307,0.0]`, exactly matching the operator observation. This is a diagnostic-label/indexing error, not a Cable environment observation error and not a learner-schema error. The formal stored state remains the already validated seven-value flattened schema.

## Root Cause Classification

### A. ACTION_MAGNITUDE_TOO_SMALL

Supported. Magnitude 0.05 requests only 0.05 mm/step and 1 mm over 20 steps. The target and robot do move, but this is difficult to see against flexible-cable motion.

### F. TCP_OBSERVATION_ERROR

Supported specifically for the recorder's terminal debug display. It reads the wrong flattened indices. Native TCP sensor and site agree exactly.

### Rejected by current evidence

- `B. ACTION_SCALE_ERROR`: no; the declared 0.001 scalar is applied consistently on all axes.
- `C. TARGET_NOT_UPDATED`: no; target/mocap changes exactly match scaled action.
- `D. CONTROLLER_NOT_TRACKING_TARGET`: no as a total failure; ctrl, qpos, and TCP respond, although bounded transient tracking is imperfect.
- `E. CONTROL_OVERWRITTEN`: no evidence; final ctrl values remain nonzero and action-dependent.
- `G. WRAPPER_ACTION_ERROR`: no; target deltas prove the base environment receives the requested action.

## Minimal Fix Status

Implemented only the project-side display correction:

- `TCP_POS_REAL` now reads `2f85/pinch_pos` directly;
- `TCP_SITE` reads `data.site_xpos[pinch]`;
- `TCP_MATCH` validates equality at runtime;
- bounded zero-action smoke returned `TCP_MATCH=True`;
- learner-facing state and transition schema are unchanged.

Still requires operator execution:

1. Perform a bounded operator-only magnitude test, starting at 0.25 and then 0.5 if needed. With the unchanged environment scale:
   - 0.25 = 0.25 mm/step, 5 mm over 20 steps;
   - 0.5 = 0.5 mm/step, 10 mm over 20 steps.
2. Keep the action inside `[-1,1]`; do not modify action scale, controller gains, qpos, ctrl, reward, or success logic during that test.

Robot teleop motion must remain `OPERATOR_VALIDATION_REQUIRED` until the operator confirms clear six-direction response in the Viewer.
