# Cable Routing MuJoCo Viewer Teleoperation

Date: 2026-08-21  
Milestone: Cable recorder cleanup fix  
Viewer rendering: `IMPLEMENTED`  
Pynput teleoperation: `VALIDATED` by operator evidence  
Action construction: `VALIDATED` by operator evidence  
Robot teleop motion: `OPERATOR_VALIDATION_REQUIRED`  
TCP debug display: `FIXED`  
Key suppression: `IMPLEMENTED`  
Operator suppression validation: `REQUIRED`  
Operator episode limit: `20000 control steps`  
Gear control: `IMPLEMENTED` (`1x`, `2x`, `5x`; default Gear 3)  
Formal demo data: `PRESENT / UNVERIFIED` (one operator-confirmed success file)

## Scope

`tools/cable_demo/record_cable_demo_viewer.py` is a project-owned recorder for the real Cable V2 environment. It does not modify Cable V2, SERL, checkpoints, reward, success logic, or replay-buffer internals. It reuses the validated project observation/transition helpers and formal dataset writer.

## Runtime and Viewer

- Python: `/home/ubuntu/miniconda3/envs/serl/bin/python`
- Environment: `PandaCableRouteVision-v0`
- Viewer: `mujoco.viewer.launch_passive(base.model, base.data)`
- MuJoCo version checked: 2.3.7
- Control period: environment `control_dt`, currently 0.02 seconds

The Viewer is created once and remains attached to the real environment model/data across resets. It is used only for scene visualization, camera operation, and `viewer.sync()`. It is not a teleoperation input source. Closing the Viewer discards only the active unfinished attempt and exits safely; every earlier successful episode has already been written independently.

The desktop command does not force `MUJOCO_GL=egl`. A visible passive Viewer requires the Ubuntu X11 desktop/window backend. EGL is used only for the current headless bounded preflight.

## Root Cause and Input Architecture

Operator testing showed that W/A/S/D/Q/E passed through the MuJoCo Viewer path trigger built-in visualization shortcuts, changing colors, background, or grid state without moving the robot. The Viewer native callback route is therefore `FAILED_FOR_TELEOP` and has been removed from robot control.

The replacement architecture is:

```text
pynput global Listener (`suppress=True`)
  -> thread-safe pressed-key set
  -> fixed-rate main control loop
  -> clipped Cable action (3,) float32
  -> env.step(action)
  -> viewer.sync()
```

`on_press` and `on_release` only update state. They never call `env.step()`. The main loop samples the current set once per environment control period, so operating-system key autorepeat cannot determine the action frequency. Holding a key produces continuous movement; releasing it returns that axis to zero. Simultaneous keys such as W+A or W+Q generate combined actions.

## Keyboard Control

| Key | Behavior |
|---|---|
| W / S | X positive / negative delta |
| A / D | Y positive / negative delta |
| Q / E | Z positive / negative delta |
| R | Discard active attempt and reset |
| Enter | Discard and end the current unsuccessful attempt |
| ESC | Discard active attempt and exit recorder |
| P | Print detailed current state |
| 1 | Gear 1: one `env.step(action)` per control-loop tick |
| 2 | Gear 2: two `env.step(action)` calls per control-loop tick |
| 3 | Gear 3: five `env.step(action)` calls per control-loop tick; default |

Every control period produces one clipped `(3,) float32` action from the currently held keys. The default and recommended magnitude is 1.0, the maximum meaningful value in action space `[-1,1]`. With no movement key held, the recorder sends a zero action; this preserves the current mocap target under the environment's delta semantics. Gripper control is never exposed and remains fixed closed by the environment.

Gear changes repetition count only; it never multiplies the action. Action remains clipped to `[-1,1]`. Every repeated call uses the normal environment path and appends its own complete learner-facing transition, including its own observation, next observation, reward, mask, and done. Success/termination/truncation stops a gear batch immediately. Startup defaults to Gear 3.

Raw callbacks print `RAW_PRESS` and `RAW_RELEASE`; movement-state changes print `LOGICAL_KEYS`. Gear changes print `GEAR`. Periodic and P-requested status output reports `GEAR`, `STEP`, `ACTION`, `TCP_POS_REAL`, `TCP_SITE`, their consistency flag, mocap target, cable tip, slot, distance, reward, return, and success. `TCP_POS_REAL` comes directly from sensor `2f85/pinch_pos`; it no longer slices the learner-facing flattened state.

## Key Event Suppression

The installed pynput 1.8.2 Listener supports `suppress=True`; the X11 backend implements the corresponding keyboard grab. This also suppresses Gear keys 1/2/3 from other X11 clients. The recorder enables this global suppression before opening the environment and prints:

```text
PYNPUT_SUPPRESS: ENABLED
KEY_SUPPRESSION = ACTIVE
```

While the recorder is active, W/S/A/D/Q/E, R, P, Enter, and ESC are captured for teleoperation and are expected not to reach the Viewer, terminal, or other X11 clients. Because pynput's supported `suppress=True` mode is global rather than per-key, other keyboard events are also suppressed during the listener lifetime. The listener is always stopped and joined in `finally`, including startup failure, exception, Viewer close, and manual exit paths. ESC remains a recorder callback: it requests safe exit, then cleanup releases the global grab.

## Action Semantics

Static source inspection confirms the environment computes:

```text
dpos = action * action_scale
new_mocap_position = clip(old_mocap_position + dpos, cartesian_bounds)
```

The action is therefore normalized xyz delta, not an absolute target. The recorder never writes `qpos`, `qvel`, mocap state, or success state directly; all motion is produced through `env.step(action)`.

## Success, Failure, and Saving

Only `info["success"] is True` accepts an episode. Reward is never used as a success proxy.

On success, the last real transition is included and the episode is immediately saved as an independent one-episode formal dataset under `results/cable_demo/`. Immediate per-success saving prevents later Viewer closure from losing completed work and avoids overwriting prior demos. The recorder then resets and continues until `--episodes` successful episodes have been collected; unsuccessful attempts do not count toward this target.

Demo outcome and process cleanup are reported independently. A completed save prints `DEMO_RESULT: PASS`. Listener, Viewer, or environment shutdown warnings print `CLEANUP_RESULT: WARNING` but cannot invalidate or remove an already saved Demo. If no successful Demo was saved, a listener cleanup failure remains a real raised error. The X11 suppression listener receives `stop()` followed by a five-second bounded join; a still-alive thread is a warning after success rather than a post-save traceback.

Reset, manual end, Viewer close, operator exit, max-step limit, unsuccessful termination, truncation, and exceptions are excluded from formal pickle data. Normal discard paths may write lightweight JSON-only records under `runtime/cable_demo/discarded_logs/`; these files are explicitly marked as non-learner datasets.

## Operator Episode Time Limit

The delivered environment has two independent 500-step boundaries: Gym registration applies `TimeLimit(max_episode_steps=500)`, and the native environment defaults to `time_limit=10.0` seconds. With `control_dt=0.02`, the native boundary is also 500 control steps. Recorder `--max-steps` is only an outer loop and cannot override either source by itself.

For operator recording only, the project recorder updates its own environment instance to:

- Gym `TimeLimit._max_episode_steps = 20000`;
- native simulation `_time_limit = 20000 * control_dt = 400.0` seconds;
- recorder `--max-steps = 20000`.

Cable V2 source and global registration are unchanged. Success termination remains untouched. Immediately before every episode boundary, the recorder prints `RECORDER_STEP`, `TERMINATED`, `TRUNCATED`, `INFO`, and `RESET_REASON`.

## Terminal Semantics

Transitions retain the environment values without rewriting them:

- `masks = float32(1.0 - terminated)`
- `dones = bool(terminated or truncated)`

The current Cable V2 source sets `terminated = info["success"]` and sets `truncated` only from the time limit. Therefore a real success currently produces a final `dones=True`, `masks=0.0` transition. Metadata separately records `environment_terminated`, `environment_truncated`, and `dataset_boundary_reason="info_success"`; the recorder does not silently fabricate a terminal flag if future environment behavior changes.

## Learner-Facing Schema

The pickle remains a flat list of transitions with exactly:

```python
{
    "observations": {"state": ..., "front": ..., "wrist": ...},
    "actions": ...,
    "next_observations": {"state": ..., "front": ..., "wrist": ...},
    "rewards": ...,
    "masks": ...,
    "dones": ...,
}
```

Shapes and dtypes remain:

- `state`: `(1, 7) float32`
- `front`: `(1, 128, 128, 3) uint8`
- `wrist`: `(1, 128, 128, 3) uint8`
- `actions`: `(3,) float32`
- `rewards`: scalar float32
- `masks`: scalar float32
- `dones`: scalar bool

## Bounded Operator Motion Test Commands

These commands are for motion validation only. Stop without attempting a formal successful Demo. Start with Stage A and use Stage B only if Stage A is still visibly too slow.

Recommended operator command — magnitude 1.0, the maximum meaningful value for action space `[-1,1]`:

```bash
cd ~/flexible_line_project

PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/miniconda3/envs/serl/bin/python \
-m tools.cable_demo.record_cable_demo_viewer \
--episodes 1 \
--action-magnitude 1.0 \
--max-steps 20000 \
--print-every 5
```

For the required operator speed check, hold W for two seconds separately in Gear 1, Gear 2, and Gear 3, resetting between trials. Expected ordering is `Gear 1 < Gear 2 < Gear 3`. Because workspace clipping can saturate long positive-X motion, compare early TCP displacement and `STEP` rate as well as the final pose.

Do not add `MUJOCO_GL=egl` to the desktop Viewer command unless separately proven compatible with visible Viewer creation on that desktop.

## Validation Command

After a successful episode, validate the exact printed pickle path:

```bash
cd ~/flexible_line_project

PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/miniconda3/envs/serl/bin/python \
-m tools.cable_demo.validate_cable_demo \
results/cable_demo/cable_routing_01_demos_YYYY-MM-DD_HH-MM-SS.pkl
```

## Bounded Preflight Evidence

The current headless SSH session had empty `DISPLAY`, empty `WAYLAND_DISPLAY`, and `XDG_SESSION_TYPE=tty`. The following were actually verified:

- syntax and CLI import: PASS;
- `mujoco.viewer.launch_passive` Viewer API: PRESENT;
- Viewer teleoperation input: DISABLED;
- real environment create/reset: PASS;
- real model/data exposure: PASS;
- one zero-action step: PASS;
- close: PASS;
- formal demo write: NO;
- formal output directory contamination: NO.
- real TCP sensor/site equality after the minimal display fix: PASS (`TCP_MATCH=True`).

Visible Viewer creation and real keyboard events were not possible during that earlier headless session. Its local `ModuleNotFoundError` finding is superseded for the operator desktop runtime by later human evidence: pynput produced real key-state logs and correctly signed nonzero actions for W/S/A/D. Keyboard input and action construction are therefore VALIDATED; robot motion is separately FAILED / UNDER_DIAGNOSIS.

## Known Risks

- MuJoCo Viewer native input is `FAILED_FOR_TELEOP` and must not be re-enabled for robot control.
- X11 global pynput capture and action construction are VALIDATED by operator evidence.
- `KEY_SUPPRESSION = IMPLEMENTED`; Viewer shortcut suppression remains `OPERATOR_SUPPRESSION_VALIDATION = REQUIRED`.
- While the listener is active, `suppress=True` globally suppresses keyboard delivery, not only the teleoperation key set; cleanup failure would temporarily affect all X11 keyboard clients.
- Robot motion at 0.25/0.50 remains `OPERATOR_VALIDATION_REQUIRED`; see `docs/cable_action_pipeline_diagnosis.md`.
- The Viewer and offscreen dual-camera rendering must coexist on the selected desktop GL backend.
- At 0.02-second control periods, key-tap responsiveness and the selected magnitude require human usability tuning without changing environment semantics.
- The EGL destructor exception is a known non-blocking exit warning; it is recorded rather than addressed by modifying the MuJoCo environment.
- Every generated pickle must pass the unchanged validator before learner use.
