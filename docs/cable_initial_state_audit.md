# Cable Routing Initial-State Randomization Audit

Date: 2026-08-21  
Scope: Cable Routing V2 reset/initialization only  
Runtime: `/home/ubuntu/miniconda3/envs/serl/bin/python`, `MUJOCO_GL=egl`  
Reset samples: 20  
Control actions executed: 0

## Conclusion

```text
INITIAL_STATE = FIXED
DEMO_DIVERSITY_RISK = HIGH
RECOMMENDED_DEMO_COLLECTION_STRATEGY = Preserve the five authentic demos, then design and validate a project-side bounded reset-randomization adapter before collecting a stratified second dataset across approved initial-condition bins.
```

All 20 resets produced exactly identical component values, including when reset was called with seeds 0 through 19. Every measured component had an exact `max - min` of zero. Tiny displayed standard deviations around `1e-16` are floating-point reduction roundoff from averaging identical values, not observed state variation.

## Static Reset Audit

The V2 environment reset implementation is `PandaCableRouteGymEnv.reset()` in `panda_cable_route_gym_env.py` lines 183–284. It performs this deterministic sequence:

1. `mujoco.mj_resetData()` restores the compiled model defaults.
2. All seven Panda joints are assigned the constant `_PANDA_HOME = [0, -0.785, 0, -2.35, 0, 1.57, pi/4]`.
3. All velocities are zeroed.
4. The cable free root position is assigned the current fixed pinch position, and its quaternion is assigned the fixed robot-base quaternion.
5. The mocap target is assigned the current TCP pose.
6. The gripper is fixed closed.
7. Exactly 1000 deterministic MuJoCo settling steps run under the same controller and home target.
8. Velocities are zeroed again and simulation time is reset to zero.

The method accepts `seed`, but does not use it, does not call a seeded superclass reset, and performs no random draw. The base class constructs `self._random`, but Cable reset never consumes it.

## Initial-State Findings

| Quantity | Finding | Evidence |
|---|---|---|
| Robot TCP position | Fixed | Same sensor value after all 20 resets. |
| Robot joint qpos | Fixed | Constant home assignment followed by identical deterministic settling; measured qpos identical. |
| Cable root position | Fixed | Root is assigned to the fixed pinch pose, then settles identically. |
| Cable root orientation | Fixed | Assigned from the fixed base quaternion; measured free-joint quaternion identical. |
| Complete cable qpos/configuration | Fixed | All 63 cable qpos components had exact zero span across 20 resets. |
| Cable tip position | Fixed | Identical after every settling phase. |
| Slot position | Fixed | Defined by static MJCF body/site positions and identical at runtime. |
| Cameras | Fixed | Camera model position, quaternion, and FOV arrays had exact zero span. The rendered `front` and robot-mounted wrist camera therefore begin from the same geometry/robot pose. |

## Randomization Search

| Mechanism | Cable V2 status |
|---|---|
| Random reset | MISSING |
| Reset noise | MISSING |
| Domain randomization | MISSING |
| Position randomization | MISSING |
| Orientation randomization | MISSING |
| Seed-dependent initialization | MISSING / seed ignored by Cable reset |

The training script has random action exploration and JAX policy seeds, but those affect actions/learning, not reset initialization. The separate Panda pick environment contains position sampling; that code is not used by Cable Routing.

## Twenty-Reset Runtime Statistics

The vector entries below are `[x, y, z]` unless otherwise noted. Runs used `reset(seed=0)` through `reset(seed=19)` without any intervening control action.

### TCP initial position

| Statistic | Value |
|---|---|
| min | `[0.3079620992548457, -0.00014721541451238667, 0.44400611146599406]` |
| max | `[0.3079620992548457, -0.00014721541451238667, 0.44400611146599406]` |
| mean | `[0.30796209925484574, -0.00014721541451238667, 0.44400611146599384]` |
| std | `[5.55e-17, 0.0, 2.22e-16]` |
| exact maximum component span | `0.0` |

### Robot joint qpos

Joint order is Panda joint1 through joint7.

| Statistic | Value |
|---|---|
| min | `[0.0002526638834715, -0.7848890115835246, -0.0002610422586482, -2.350553951711717, -0.0005773323984060, 1.5709724772213047, 0.7855841980397829]` |
| max | same as min |
| mean | `[0.0002526638834715, -0.7848890115835249, -0.0002610422586482, -2.350553951711717, -0.0005773323984060, 1.5709724772213043, 0.7855841980397825]` |
| std | all components `<= 4.44e-16` |
| exact maximum component span | `0.0` |

The measured post-settling qpos is slightly different from the assigned `_PANDA_HOME`, but is identical on every reset.

### Cable root world position

| Statistic | Value |
|---|---|
| min | `[0.30837762713463185, -0.00032192824706351827, 0.4435682033096803]` |
| max | same as min |
| mean | `[0.30837762713463174, -0.00032192824706351816, 0.4435682033096803]` |
| std | `[1.11e-16, 1.08e-19, 0.0]` |
| exact maximum component span | `0.0` |

### Cable root free-joint qpos

Order is position xyz followed by quaternion wxyz.

| Statistic | Value |
|---|---|
| min | `[0.30837762713463185, -0.00032192824706351827, 0.4435682033096803, -0.6812988859315509, -0.49953560120489415, -0.31490021899124915, 0.4325897169796758]` |
| max | same as min |
| mean | `[0.30837762713463174, -0.00032192824706351816, 0.4435682033096803, -0.6812988859315507, -0.4995356012048939, -0.3149002189912491, 0.43258971697967585]` |
| std | all components `<= 2.78e-16` |
| exact maximum component span | `0.0` |

### Complete cable configuration/qpos

The model has `nq=78`; the cable begins at qpos address 15 and contributes 63 measured qpos components.

| Statistic | Value |
|---|---|
| component-wise min | exactly equal to component-wise max for all 63 components |
| component-wise max | exactly equal to component-wise min for all 63 components |
| component-wise mean | equal to the fixed vector within floating-point reduction roundoff |
| component-wise std | maximum observed reduction roundoff `5.55e-16` |
| exact maximum component span | `0.0` |

### Cable tip initial position

| Statistic | Value |
|---|---|
| min | `[0.47329906240967157, -0.08370332123530967, 0.2362947065262273]` |
| max | same as min |
| mean | `[0.4732990624096713, -0.08370332123530971, 0.23629470652622722]` |
| std | `[2.78e-16, 4.16e-17, 8.33e-17]` |
| exact maximum component span | `0.0` |

### Slot position

| Statistic | Value |
|---|---|
| min | `[0.5, 0.1, 0.055]` |
| max | `[0.5, 0.1, 0.055]` |
| mean | `[0.5, 0.10000000000000002, 0.055000000000000014]` |
| std | `[0.0, 1.39e-17, 1.39e-17]` |
| exact maximum component span | `0.0` |

### Camera model parameters

All compiled model cameras were included, while Cable observations use `front` and `handcam_rgb` (exposed as wrist).

| Quantity | min | max | std | Exact max span |
|---|---|---|---|---|
| camera positions | equal to fixed model values | same as min | all components `<= 2.22e-16` | `0.0` |
| camera quaternions | equal to fixed model values | same as min | all components `<= 2.22e-16` | `0.0` |
| camera FOV | `[45.0, 62.0, 42.5]` | `[45.0, 62.0, 42.5]` | `[0.0, 0.0, 0.0]` | `0.0` |

## Training Versus Evaluation Reset

The delivered training entry creates its environment with `gym.make(FLAGS.env)` (or the same call with `render_mode="human"`) and then applies observation/chunking wrappers. Its evaluation environment independently calls the same `gym.make(FLAGS.env)` and applies the same Cable observation/chunking wrappers plus `RecordEpisodeStatistics`.

Both use Gym ID `PandaCableRouteVision-v0`, whose entry point is the same `PandaCableRouteGymEnv`. None of these wrappers changes physical reset initialization. Therefore training and evaluation use the same fixed reset logic.

The project recorder also instantiates the same Gym ID. Its project-side time-limit adjustment changes truncation only and does not alter reset state.

## Risk Assessment

`DEMO_DIVERSITY_RISK = HIGH` because five successful demonstrations from one deterministic initial condition provide trajectory/control variation but no initial-state coverage. A learner can overfit to the single TCP/cable/slot geometry, and current validation does not establish robustness to manufacturing placement variation, cable-shape variation, camera perturbation, or robot calibration offsets.

## Recommended Collection Strategy

1. Preserve the existing five authentic successful demos and their provenance; do not rewrite or relabel them. The first has documented validator PASS evidence; validate the other four separately before learner use.
2. In a separate authorized milestone, design a project-side reset-randomization adapter rather than editing Cable V2 directly.
3. Define physically safe, bounded perturbation ranges separately for robot/TCP start, slot position, cable root pose/configuration, and—only if required—camera parameters.
4. Validate every sampled initial state for collision, workspace bounds, cable attachment, deterministic seed replay, reset stability, and unchanged success semantics before recording.
5. Use stratified initial-condition bins and record the seed/parameters in metadata so coverage is measurable rather than relying on uncontrolled randomness.
6. Collect successful human demos across those approved bins and validate each output with the unchanged Demo validator before combining datasets.
7. Keep fixed-start and randomized-start datasets distinguishable for later ablation and evaluation.

No randomization adapter or new Demo was created in this audit.
