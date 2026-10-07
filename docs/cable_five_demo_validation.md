# Cable Five-Demo Validation

Date: 2026-08-21  
Milestone: `MILESTONE 1B-CABLE-FIVE-DEMO-VALIDATION`  
Runtime: `/home/ubuntu/miniconda3/envs/serl/bin/python`  
Validator: existing `tools/cable_demo/validate_cable_demo.py` unchanged

## Result

```text
FORMAL_DEMO_COUNT = 5
FORMAL_DEMO_SUCCESS_COUNT = 5
VALIDATED_FORMAL_DEMO_COUNT = 5
UNVERIFIED_FORMAL_DEMO_COUNT = 0
DEMO_BATCH_VALIDATION = VALIDATED
SCHEMA_CONSISTENT = YES
REPLAY_BUFFER_COMPATIBLE = YES
MERGE_COMPATIBLE = YES
```

No Demo, metadata, validator, recorder, environment, reset, SERL, or checkpoint file was modified.

## Inventory and Validator Results

| Demo file | Bytes | SHA256 | Metadata | Episodes | Transitions | Successes | Validator |
|---|---:|---|---|---:|---:|---:|---|
| `cable_routing_01_demos_2026-08-21_07-14-17.pkl` | 517,944,804 | `415b2adf5c2453db6532a6c5b9347b6ba6c2b97b393331c98c956e9ef71e54e0` | `cable_routing_01_demos_2026-08-21_07-14-17_metadata.json` | 1 | 2,629 | 1 | PASS |
| `cable_routing_01_demos_2026-08-21_07-37-47.pkl` | 961,221,808 | `16ce3c0f9978e3f7b10c5e9c11ccd37307cdcc01b83e2308a1f22c0516da253b` | `cable_routing_01_demos_2026-08-21_07-37-47_metadata.json` | 1 | 4,879 | 1 | PASS |
| `cable_routing_01_demos_2026-08-21_07-43-30.pkl` | 249,023,422 | `b4f3f835d62c97aab1add8ea0af1587a2d6e5a15166353c6a732b69d4063e7e7` | `cable_routing_01_demos_2026-08-21_07-43-30_metadata.json` | 1 | 1,264 | 1 | PASS |
| `cable_routing_01_demos_2026-08-21_07-46-57.pkl` | 170,612,644 | `30210b87f980621e7402ed03bc361f99bc1fa1fbf78f785e7ee29e3df19d4c60` | `cable_routing_01_demos_2026-08-21_07-46-57_metadata.json` | 1 | 866 | 1 | PASS |
| `cable_routing_01_demos_2026-08-21_07-50-43.pkl` | 173,961,848 | `86935541b3da64a95146b1ede1967b90d00b4e06897eb8c4c292163bc031b27f` | `cable_routing_01_demos_2026-08-21_07-50-43_metadata.json` | 1 | 883 | 1 | PASS |

Every validator invocation reported `TEST_ONLY=False`. The unchanged validator passed required keys, top-level type, non-empty transitions, shapes, dtypes, finite values, action bounds, mask/done semantics, episode boundaries, success-only policy, metadata counts, and terminal transition checks.

## Schema Consistency

All transitions in all five files passed the validator, and direct schema fingerprints from all five first transitions were identical:

| Field | Shape | dtype |
|---|---|---|
| `observations/state` | `(1, 7)` | `float32` |
| `observations/front` | `(1, 128, 128, 3)` | `uint8` |
| `observations/wrist` | `(1, 128, 128, 3)` | `uint8` |
| `actions` | `(3,)` | `float32` |
| `next_observations/state` | `(1, 7)` | `float32` |
| `next_observations/front` | `(1, 128, 128, 3)` | `uint8` |
| `next_observations/wrist` | `(1, 128, 128, 3)` | `uint8` |
| `rewards` | scalar `()` | `float32` |
| `masks` | scalar `()` | `float32` |
| `dones` | scalar `()` | `bool` |

Observation keys and next-observation keys are exactly `front`, `state`, and `wrist` in every file. No NaN/Inf, shape mismatch, dtype mismatch, or action-range error was found.

## Episode Boundaries

| Episode | Merged start | Merged end (exclusive) | Terminal done | Terminal mask | Success | Terminated | Truncated |
|---:|---:|---:|---|---:|---|---|---|
| 1 | 0 | 2,629 | `True` | 0.0 | `True` | `True` | `False` |
| 2 | 2,629 | 7,508 | `True` | 0.0 | `True` | `True` | `False` |
| 3 | 7,508 | 8,772 | `True` | 0.0 | `True` | `True` | `False` |
| 4 | 8,772 | 9,638 | `True` | 0.0 | `True` | `True` | `False` |
| 5 | 9,638 | 10,521 | `True` | 0.0 | `True` | `True` | `False` |

Each metadata boundary reason is `info_success`.

## Replay Buffer Compatibility

The test used the current Cable learner-facing path:

1. real Cable training environment and project observation wrapper;
2. SERL `make_replay_buffer(..., type="memory_efficient_replay_buffer")`;
3. image keys `front` and `wrist`;
4. SERL `populate_data_store`;
5. `MemoryEfficientReplayBufferDataStore.insert`.

| File | Raw transitions | Replay before | Replay after | Replay increase |
|---|---:|---:|---:|---:|
| 07-14-17 | 2,629 | 0 | 2,630 | 2,630 |
| 07-37-47 | 4,879 | 2,630 | 7,510 | 4,880 |
| 07-43-30 | 1,264 | 7,510 | 8,775 | 1,265 |
| 07-46-57 | 866 | 8,775 | 9,642 | 867 |
| 07-50-43 | 883 | 9,642 | 10,526 | 884 |

```text
RAW_TRANSITIONS = 10521
HORIZON_ONE_EPISODE_INITIALIZATION_RECORDS = 5
EXPECTED_REPLAY_SIZE = 10526
ACTUAL_REPLAY_SIZE = 10526
REPLAY_BUFFER_COMPATIBILITY = PASS
```

The extra one record per episode is the expected horizon-one initialization-frame behavior; it is not an input transition count mismatch or data loss. No shape, image-key, dtype, or insertion error occurred. No learner or training loop was started.

The environment emitted existing optional TFDS/protobuf and RLDS-logger import warnings during setup. They did not prevent the actual Cable pickle insertion path from completing.

## Merge Feasibility

The five transition lists were concatenated in memory, in filename/episode order, without writing a merged pickle. The result contained 10,521 transitions. Required keys were present throughout; the five contiguous boundaries covered `[0, 10521)` exactly; every boundary ended with success, `done=True`, and `mask=0.0`.

```text
MERGE_COMPATIBLE = YES
MERGED_FILE_CREATED = NO
ORIGINAL_FILES_OVERWRITTEN = NO
```

## Initial-State Distribution Assessment

```text
INITIAL_STATE_FIXED = TRUE
TRAIN_RESET = FIXED
EVALUATION_RESET = FIXED
RECORDER_RESET = FIXED
CURRENT_BASELINE_DEMOS = IN_DISTRIBUTION
CURRENT_EVALUATION_RISK = LOW  # initial-state distribution mismatch only
OUT_OF_DISTRIBUTION_GENERALIZATION_RISK = HIGH
```

The LOW current-evaluation rating is narrowly scoped to initial-state distribution: baseline training, evaluation, and recording use the same fixed reset. It does not validate policy quality, checkpoint restoration, training convergence, success rate, or robustness. Fixed-start data is therefore not a blocker for the current fixed-start baseline, but robustness to changed TCP/cable/slot/camera states remains unvalidated and high risk.

Reset randomization was not implemented and remains a later robustness/ablation candidate only.
