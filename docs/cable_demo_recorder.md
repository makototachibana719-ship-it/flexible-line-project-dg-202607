# Cable Routing Demo Recorder

Date: 2026-08-21  
Milestone: MILESTONE 1B-CABLE  
Recorder status: `VALIDATED` for code/dry-run lifecycle  
Formal demo status: `MISSING`

## Scope

The recorder is project-owned under `tools/cable_demo/`. It imports the isolated Cable V2 package and does not modify Cable source, SERL, checkpoints, reward, or success logic.

Only episodes where the real environment returns `info["success"] is True` can enter a formal dataset. Random actions, dry-run transitions, timeouts, manual resets, manual early endings, exceptions, and unsuccessful termination are never accepted as formal demonstrations.

## Architecture

```text
PandaCableRouteVision-v0
  -> CableTrainingObservationWrapper
       native state dict -> Gym flatten -> top-level state
       images/front       -> top-level front
       images/wrist       -> top-level wrist
  -> SERL ChunkingWrapper(obs_horizon=1)
  -> keyboard action (3,) float32
  -> six-field deep-copied transition
  -> success-only episode gate
  -> flat list pickle + separate episode metadata JSON
```

The project wrapper is necessary because the current mainline `SERLObsWrapper` expects a singular native `image` key, while Cable V2 exposes nested `images/front,wrist`. The project wrapper follows the same Gym `flatten()` operation for state and produces the top-level image keys required by the memory-efficient replay buffer. SERL source is unchanged.

## Keyboard mapping

| Key | Action |
|---|---|
| W / S | X positive / negative |
| A / D | Y positive / negative |
| Q / E | Z positive / negative |
| R | Discard and reset current episode |
| Enter | End and discard current episode unless real success already auto-ended it |
| ESC | Exit; discard current episode |

The magnitude defaults to 0.25 in normalized action units. Every action is converted to float32 and clipped to the real action space. Gripper control is not exposed.

## Episode acceptance

- Success source: only `info["success"]` from the environment.
- Success threshold remains environment-owned: cable-tip distance to slot center below 0.03 m.
- Successful episodes auto-end and are accepted.
- Timeout/truncation, non-success termination, recorder max steps, reset, manual end, or exit are discarded.
- Each accepted episode records ID, transition range, steps, return, success, duration, and final distance to slot.

## Output

Formal output directory: `results/cable_demo/`.

```text
cable_routing_XX_demos_YYYY-MM-DD_HH-MM-SS.pkl
cable_routing_XX_demos_YYYY-MM-DD_HH-MM-SS_metadata.json
```

The pickle is a flat list of transition dictionaries because the delivered Cable learner loads it and inserts every list element directly. Episode boundaries and audit metadata are stored in JSON so replay insertion sees exactly the expected six transition keys.

## Commands

Manual recording requires a working desktop/X11 display for the OpenCV keyboard window:

```bash
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/miniconda3/envs/serl/bin/python \
-m tools.cable_demo.record_cable_demo --episodes 1
```

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/miniconda3/envs/serl/bin/python \
-m tools.cable_demo.validate_cable_demo \
results/cable_demo/cable_routing_01_demos_YYYY-MM-DD_HH-MM-SS.pkl
```

Only load pickle files produced by this project or otherwise explicitly trusted; Python pickle is not safe for untrusted input.

## Dry run evidence

The bounded dry run completed real env reset, one zero-action step, transition deepcopy, TEST_ONLY pickle/metadata output, environment close, and insertion through SERL's `populate_data_store` plus `MemoryEfficientReplayBufferDataStore` path. The validator passed only with `--allow-test-only` and correctly rejected the same artifact in formal/default mode.

Interactive OpenCV key capture was not exercised over the current headless SSH session. Manual display/keyboard usability remains a required operator preflight before authentic acquisition.
