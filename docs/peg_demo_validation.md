# Peg Demo Validation (M1D-D)

Date: 2026-08-23  
Scope: read-only audit; no Demo/package/environment mutation and no training/evaluation.

## Frozen assets

- `PACKAGE_ROOT = external/packages/peg_insertion/peg_insertion`
- archive: `external/incoming/peg_insertion_package.zip`, 337,343,679 bytes
- archive SHA256: `17ba3da5cbcaafd334492cee5e91f3403740f9aa6836a18a3170ae74982e0c06`
- transition PKL: 1,714,786 bytes, SHA256 `44ee651768d9f231caccf81d5a58460c8f44bc17463de7e08ea157b3c2a53ce7`
- BC PKL: 743,082 bytes, SHA256 `39dc643476417b5cdef269e575945ab2d53183df14739731584e52215c1f64b0`

## NPZ inventory and result

Exactly 20 files, `demo_000.npz` through `demo_019.npz`, were opened independently. Each is one episode. Lengths are:

`318, 260, 227, 169, 354, 218, 453, 225, 296, 358, 381, 193, 385, 228, 181, 197, 284, 222, 354, 209` (total 5,512).

`PEG_NPZ_DEMO_COUNT = 20`  
`PEG_NPZ_SUCCESS_COUNT = 20`  
`PEG_NPZ_SCHEMA = VALIDATED`

Common schema:

| Key | Per-episode shape | dtype |
|---|---:|---|
| observations / next_observations | `(T, 20)` | float32 |
| actions | `(T, 7)` | float32 |
| rewards / masks | `(T,)` | float32 |
| dones / terminated / truncated | `(T,)` | bool |
| position/lateral/axial/orientation_error, force_norm | `(T,)` | float32 |
| initial_observation | `(20,)` | float32 |
| metadata_json | scalar | Unicode JSON |

All 20 final records have `terminated=True`, `truncated=False`, and independently satisfy the executed source thresholds: lateral `<=0.002 m`, axial `<=0.003 m`, orientation `<=0.05 rad`. Force is measured but is not a success predicate.

## PKL mapping

`peg_insert_20_demos_state.pkl` is a flat list of 5,512 six-field transitions: `observations`, `actions`, `next_observations`, `rewards`, `masks`, `dones`. Sequential flattening of the 20 NPZ files is exactly equal transition-by-transition, including dtype/value. Episode boundaries are recoverable from terminal fields and the validated NPZ lengths. It directly matches the RLPD loader.

`peg_insert_bc_consistent.pkl` is also a flat six-field transition list, not an obs/action-only file. Its 2,388 records exactly equal nine complete NPZ episodes: `001, 003, 005, 006, 007, 013, 015, 016, 018`. It directly matches the BC trainer, which consumes observations/actions/next_observations (the remaining fields are retained but unused by its loss).

`PEG_BC_DATASET_MAPPING = EXACT_9_EPISODE_SUBSET_OF_NPZ`

No additional dataset/replay binary was found. `trajectory_replay.py` is source code, not a dataset.

## Success method

`SUCCESS_VALIDATION_METHOD = executed-state-source thresholds + terminal arrays`, not README names or `metadata_json`. The live call chain is `SimEnv.step -> _compute_reward/_check_terminated -> PegInsertionTask.compute_reward_from_obs/is_success_from_obs`; timeout is independently emitted as `truncated` at 500 steps by the training-created environment.

