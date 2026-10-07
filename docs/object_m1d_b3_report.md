# M1D-B3 Object Checkpoint Read-Stall Report

Date: 2026-08-23

## Outcome

The checkpoint is committed and its OCDBT storage is readable. Direct reads of a scalar, a 9 MB encoder leaf, and a 9 MB optimizer leaf all pass quickly. Full-tree restore stalls with and without a fresh target. The target-free run reads actively at first, then enters a stable sleeping/no-I/O/no-CPU state.

- `CHECKPOINT_STORAGE_INTEGRITY = PASS`
- `PRODUCING_RUNTIME = KNOWN`
- `METADATA_READ_ONLY_PASS = PASS` (direct metadata; Orbax-native metadata API timed out)
- `SMALL_LEAF_READ_PASS = PASS`
- `LARGE_LEAF_READ_PASS = PASS`
- `NO_TARGET_INSPECTION = TIMEOUT`
- `READ_STALL_LAYER = PARALLEL_FULL_TREE_RESTORE / ORBAX-TENSORSTORE COORDINATION`
- `RESOURCE_BEHAVIOR = NATIVE_DEADLOCK`
- `GPU_EXPECTED_TO_FIX_RESTORE = NO`
- `RECOMMENDED_RESOLUTION = A. producing runtime reproduction`
- `CHECKPOINT_RESTORE_STATUS = BLOCKED_READ_STALL`

Only TensorStore differs: producing 0.1.78 versus current 0.1.51. This is the primary compatibility hypothesis but has not yet been experimentally proven. No checkpoint, Object package, SERL source, Demo, or environment package was changed.

Exact next task: reproduce the producing dependency set in an isolated runtime and repeat bounded read-only restore diagnostics. Do not enter M1D-C.
