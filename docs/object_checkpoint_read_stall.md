# Object Checkpoint Read-Stall Diagnosis

Date: 2026-08-23

Scope: read-only diagnosis of `checkpoint_4500`. No training, rollout, evaluation, conversion, package/SERL mutation, or dependency change was performed.

## Findings

`CHECKPOINT_STORAGE_INTEGRITY = PASS`

The committed checkpoint contains 15 files (105,193,080 bytes), including `_CHECKPOINT_METADATA`, `_METADATA`, `_sharding`, root/process OCDBT manifests, and nine OCDBT data files. The commit metadata has non-null init and commit timestamps. No tmp, incomplete, or lock marker was found. OCDBT opened, all 1,196 keys were enumerated, 598 Zarr arrays were discovered, and representative arrays from parameter and optimizer branches were read successfully. The frozen tree hash remains `f23f676447cfcefe6568e5946de48b3ffd08ce5177ce2220d9caa76e0b7e831c`.

`PRODUCING_RUNTIME = KNOWN`

The delivered `environment_versions.txt`, `pip_freeze.txt`, and `environment.yml` agree on Python 3.10.20, JAX/JAXlib 0.4.35, Flax 0.8.5, Orbax-checkpoint 0.6.4, TensorStore 0.1.78, and Optax 0.2.3.

## Version Drift Table

| Component | Producing | Current `serl` | Drift |
|---|---:|---:|---|
| Python | 3.10.20 | 3.10.20 | none |
| JAX | 0.4.35 | 0.4.35 | none |
| JAXlib | 0.4.35 | 0.4.35 | none |
| Flax | 0.8.5 | 0.8.5 | none |
| Orbax-checkpoint | 0.6.4 | 0.6.4 | none |
| TensorStore | 0.1.78 | 0.1.51 | current is older |
| Optax | 0.2.3 | 0.2.3 | none |

TensorStore is the only evidenced version drift. This is a strong compatibility lead, not proof of causality.

## Isolated Reads

| Test | Open | Read | Elapsed | Result |
|---|---|---|---:|---|
| OCDBT root / metadata keys | PASS | 1,196 keys | 0.0002 s open; 0.0092 s list | PASS |
| `step` scalar | PASS | PASS | 0.0139 s open; 0.0041 s read | PASS |
| encoder kernel, 9,437,184 bytes | PASS | PASS | 0.0081 s open; 0.0846 s read | PASS |
| optimizer kernel, 9,437,184 bytes | PASS | PASS | 0.0084 s open; 0.0565 s read | PASS |

`METADATA_READ_ONLY_PASS = PASS` for direct JSON/OCDBT metadata and traversal. Orbax-native `PyTreeCheckpointer.metadata()` separately timed out at 30 seconds.

`SMALL_LEAF_READ_PASS = PASS`

`LARGE_LEAF_READ_PASS = PASS`

## Restore and Resource Behavior

Flax `restore_checkpoint(..., target=None, step=4500)` uses the legacy compatibility entry but reaches Orbax for this checkpoint. It timed out after 60 seconds. During the first 17 seconds it accumulated 371,388,416 `read_bytes`, reached 517,648 KiB RSS, and used CPU. For the remaining 43 samples, disk-read bytes were flat; after about 37 seconds RSS and CPU ticks were also flat while the process slept.

`NO_TARGET_INSPECTION = TIMEOUT`

`READ_STALL_LAYER = PARALLEL_FULL_TREE_RESTORE / ORBAX-TENSORSTORE COORDINATION`

The stall is after metadata/OCDBT open, key traversal, and individual small/large tensor reads. Because target-free restore also stalls, fresh-target reconstruction is not required to reproduce it. The evidence is most consistent with native wait/deadlock after full-tree parallel reads, rather than active slow I/O or CPU compilation.

`RESOURCE_BEHAVIOR = NATIVE_DEADLOCK` (evidence-based classification; exact internal wait site remains uninstrumented).

## GPU Implication and Resolution

`GPU_EXPECTED_TO_FIX_RESTORE = NO`

Checkpoint parsing and storage reads are CPU/storage/TensorStore work. No observed behavior indicates GPU compute is limiting the restore.

`RECOMMENDED_RESOLUTION = A. producing runtime reproduction`

Reproduce the delivered runtime in an isolated environment, especially TensorStore 0.1.78, then repeat the same bounded metadata, no-target, and matching-target reads. Do not change the current environment or convert the checkpoint. If the producing runtime also stalls, collect native thread stacks before selecting a different resolution.

Raw evidence is under `results/object_m1d_b3/`.

## M1D-B4 addendum — producing version reproduced

TensorStore 0.1.78 was isolated in `object_ckpt_compat` with all other required versions exact. Leaf reads remained fast, but Orbax metadata (60 s), no-target restore (120 s), and matching-target restore (300 s) all timed out. Thus version drift alone does not close the root cause. See `object_producing_runtime_restore.md` and `results/object_m1d_b4/`.
