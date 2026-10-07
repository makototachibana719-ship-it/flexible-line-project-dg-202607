# Object Checkpoint Closure Audit

Date: 2026-08-22  
Checkpoint: `checkpoint_4500`  
Scope: structural diagnosis, one zero-gradient optimizer-state test, one bounded restore retry, at most one inference.

## Three-path diagnosis

The paths are exactly:

```text
opt_states/actor/inner_state/0/1
opt_states/critic/inner_state/0/1
opt_states/temperature/inner_state/0/1
```

In checkpoint metadata, all three have:

```text
value_type = None
skip_deserialize = true
```

They correspond to the `EmptyState` second element of each Optax Adam `chain`, not actor/critic/temperature parameter arrays. Default Flax `flatten_dict` drops empty nodes, which produced the earlier 598 count. With `keep_empty_nodes=True`, the fresh target contains the same three structural paths from initial construction.

```text
OPTIMIZER_STATE_EXPLANATION = OPTAX_EMPTY_STATE_SENTINELS
LAZY_OPTIMIZER_INITIALIZATION = FALSE
VERSION_DIFFERENCE_REQUIRED_TO_EXPLAIN = FALSE
```

## Pre/post optimizer comparison

One zero-gradient `JaxRLTrainState.apply_gradients` call was executed for actor, critic, and temperature. It did not start a learner loop or evaluate a loss.

| Tree view | Before | After | Checkpoint |
|---|---:|---:|---:|
| Array paths | 598 | 598 | 598 |
| Structural paths including empty nodes | 601 | 601 | 601 |

No path was added by the update. All 601 structural paths match exactly.

```text
PRE_UPDATE_ARRAY_PATHS = 598
POST_UPDATE_ARRAY_PATHS = 598
PRE_UPDATE_STRUCTURAL_PATHS = 601
POST_UPDATE_STRUCTURAL_PATHS = 601
CHECKPOINT_PATHS = 601
CHECKPOINT_ARCHITECTURE_MATCH = MATCH
```

This corrects the earlier false mismatch; no checkpoint path was deleted or ignored.

## Bounded restore retry

After exact structural match, one retry used:

```text
flax.training.checkpoints.restore_checkpoint
Orbax OCDBT / TensorStore backend
target = matching JaxRLTrainState
step = 4500
```

The restore did not return within the requested 300-second restore window. Python `SIGALRM` could not interrupt the native TensorStore blocking call, so the process was manually interrupted after the bounded window. No restored state or leaf values were returned, and the original checkpoint hash/content was unchanged.

```text
CHECKPOINT_RESTORE = UNVERIFIED_TIMEOUT
RESTORED_LEAF_COUNT = UNAVAILABLE
RESTORED_PATH_SHAPE_DTYPE_MATCH = UNAVAILABLE
```

## Policy inference

Because restore did not return, environment reset/inference was not executed.

```text
POLICY_INFERENCE = NOT_RUN
POLICY_QUALITY = UNVERIFIED
```

## Gates

```text
OBJECT_CHECKPOINT_MAPPING_GATE = VALIDATED_MATCH
OBJECT_CHECKPOINT_RESTORE_GATE = UNVERIFIED_TIMEOUT
OBJECT_POLICY_INFERENCE_GATE = NOT_RUN
```

Raw evidence: `results/object_m1d_b2/optimizer_tree_audit.json` and `results/object_m1d_b2/restore_audit.json`.
## M1D-B4 producing-runtime result — 2026-08-23

An isolated runtime exactly reproduced the delivered versions, including TensorStore 0.1.78; the original `serl` environment remained unchanged. Direct OCDBT and representative leaf reads passed, but Orbax metadata timed out at 60 seconds, no-target restore at 120 seconds, and matching-target restore at 300 seconds. No inference ran. Therefore TensorStore version drift is not proven as the root cause, and restore/inference gates remain BLOCKED.

The original 2,000-transition Demo is `ACCEPTED_FOR_BASELINE` by project decision while `OBJECT_DEMO_SUCCESS_PROVENANCE = UNVERIFIED` remains unchanged.

## M1D-C legacy policy

Project decision freezes `checkpoint_4500` as a preserved legacy artifact. No further full restore, compatible-reader, conversion, or inference work will be performed. Architecture MATCH and unresolved native restore remain historical facts; the legacy checkpoint is not required for fresh Object baseline training and is no longer a GPU-handoff blocker.
