# Object Grasp Demo Validation

Date: 2026-08-22  
Milestone: `M1D-B`  
Runtime: `/home/ubuntu/miniconda3/envs/serl/bin/python`

## Frozen asset

- Package root: `external/packages/object_grasping/object_grasp_package/`
- Archive source: `external/incoming/object_grasping_package.zip`
- Archive size/SHA256: `239862488` bytes / `16ecaa47295265664ecf095a0f98e790f626c4d8abaeaa567aae2140e7b088c3`
- Demo: `external/packages/object_grasping/object_grasp_package/demos/franka_lift_cube_image_20_trajs.pkl`
- Demo size/SHA256: `199111375` bytes / `f5ea03a2534b1696f31520ff68386fafe20ab06618a39cd0cad18e899ddb0d64`
- The second copy under `code/example/` has the same SHA256. Neither copy was modified.

## Environment contract

`PandaPickCubeVision-v0` is registered by `franka_sim.__init__` with factory `franka_sim.envs:PandaPickCubeGymEnv`, `image_obs=True`, and an outer Gym `TimeLimit(max_episode_steps=100)`. Native control/physics periods are 0.02/0.002 seconds and the native 10-second limit is checked as `terminated`; native `truncated` is always false. In the registered environment, the outer 100-step limit occurs first and is represented as truncation.

Native vision observation is nested `state/{panda/tcp_pos(3),panda/tcp_vel(3),panda/gripper_pos}` plus `images/{front,wrist}`, each camera `(128,128,3)` uint8. The training-time flattened/chunked contract evidenced by the log and Demo is `{front,state,wrist}` with shapes `(1,128,128,3)`, `(1,7)`, `(1,128,128,3)`. Action is float32 `(4,)`, `[x,y,z,grasp]`, bounded by `[-1,1]`.

## Gripper shape mismatch

The actual value is created by `np.array(ctrl / 255, dtype=np.float32)`, giving scalar shape `()`. Both vision observation-space declarations require `(1,)`. Gym therefore reports that reset/step observations are outside the space.

`gym.spaces.flatten` was directly tested and still produces the intended `(7,)` state; the delivered Demo has `(1,7)`. All 2,000 transitions insert and sample successfully, and the constructed DrQ input and checkpoint target use `(1,7)`. The mismatch therefore did not block Demo replay, batch stacking, DrQ construction, or checkpoint structural construction in this audit. It remains a native contract/evaluation warning. The separately missing delivered multi-camera wrapper revision is an integration problem, not evidence that the scalar itself breaks flattening.

```text
OBJECT_GRIPPER_SHAPE_MISMATCH = BENIGN
```

No source was changed.

## Actual Demo schema

The trusted repository-local pickle was deserialized read-only. It is a flat list of 2,000 transition dictionaries, not a list of 20 episode objects.

Every transition contains exactly:

```text
observations, actions, next_observations, rewards, masks, dones
```

Observation and next-observation keys are `front`, `state`, `wrist`. Images are uint8 `(1,128,128,3)`, state is float32 `(1,7)`, action is float32 `(4,)`, reward is scalar float32, mask is a Python float, and done is a Python bool. All numeric values are finite. Actions stay within `[-1,1]`.

## Episode boundaries and success

All 2,000 records have `dones=False` and `masks=1.0`. There are no trajectory IDs, boundary markers, `success`, `block_pos`, `block_height`, or `lift_height`. Consequently:

- claimed trajectories in package documentation: 20;
- episodes recoverable from the actual pickle: 0;
- trailing transitions without a boundary: 2,000;
- episode length min/max/mean/median: `UNVERIFIED`;
- Demo success count/failure count: `UNVERIFIED`.

The source success condition requires `block_height >= z_init + 0.2`. That value cannot be reconstructed from the visual 7D state, and neither reward, `done`, filename, nor README is accepted as a substitute.

```text
DEMO_SUCCESS_COUNT = UNVERIFIED
DEMO_FAILURE_COUNT = UNVERIFIED
DEMO_SUCCESS_VALIDATION_METHOD = BLOCKED_MISSING_BLOCK_STATE_SUCCESS_AND_BOUNDARIES
OBJECT_DEMO_SCHEMA = VALIDATED
OBJECT_DEMO_SUCCESS = UNVERIFIED
```

The data is replay-compatible, so immediate re-recording is not recommended. First request authoritative boundary/success sidecar data or upstream provenance. Only if that cannot be obtained and success-labelled demonstrations are required should new manual recording be proposed.

## Reward and terminal semantics

- `z_success = z_init + 0.2 m`.
- `r_close = exp(-20 * ||block_pos - tcp_pos||)`.
- `r_lift = clip((block_z-z_init)/(z_success-z_init), 0, 1)`.
- reward is `0.3*r_close + 0.7*r_lift`.
- there is no additional success bonus.
- native success is exposed only in `info["success"]` with `block_height` and `lift_height`.
- success does not terminate the episode.
- registered episodes normally end through outer 100-step truncation, whereas the actor records `mask = 1-terminated` and `done = terminated or truncated`.

The delivered pickle's all-one masks/all-false dones do not preserve those registered episode boundaries.

## Replay compatibility

The actual `MemoryEfficientReplayBuffer` was created with `front` and `wrist` pixel keys. All 2,000 transitions inserted. Horizon-one initialization added one record, producing replay size 2,001. An 8-item packed sample returned actions `(8,4)`, state `(8,1,7)`, and each packed camera `(8,2,128,128,3)`.

```text
RAW_DEMO_TRANSITIONS = 2000
REPLAY_SIZE = 2001
HORIZON_ONE_INITIALIZATION_RECORDS = 1
REPLAY_BUFFER_COMPATIBILITY = PASS
```

Raw evidence: `results/object_m1d_b/demo_audit.json` and `results/object_m1d_b/replay_audit.json`.
