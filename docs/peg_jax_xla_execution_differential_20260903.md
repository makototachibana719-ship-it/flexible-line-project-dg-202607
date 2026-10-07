# Peg JAX/XLA Execution-Layer Differential Diagnostic — 2026-09-03

## Scope and safety constraints

This diagnostic compared one fresh-initialized `SACAgent.update_high_utd` call (A) with one fresh-process call after restoring the formal-candidate `checkpoint_0` (B). Both calls used the original update implementation, normal RNG progression, one identical immutable 32-online + 32-Demo batch, JAX GPU, disabled Triton GEMM, explicit full-tree `block_until_ready`, independent XLA dump directories, and a 130-second hard timeout.

Formal training, W&B, checkpoint writes, retries, dependency changes, algorithm/configuration changes, and `runtime_gpu_formal.env` creation were prohibited and did not occur.

## Inputs held constant

- GPU/backend: `CudaDevice(id=0)` / `gpu`
- JAX/JAXlib: existing locked 0.4.35 environment
- XLA runtime setting: `--xla_gpu_enable_triton_gemm=false`
- allocation setting: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- batch SHA-256: `854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c`
- batch composition: 32 online + 32 Demo
- observation/action shapes: `(64, 20)` / `(64, 7)`
- online replay SHA-256: `21fe7eeb0885b0ba3f75d1daa3633499620b7e2f6051beabeca5f5c5fa6f9561`
- Demo SHA-256: `44ee651768d9f231caccf81d5a58460c8f44bc17463de7e08ea157b3c2a53ce7`
- agent state leaves: 198 in A and B
- optimizer-state leaves: 150 in A and B

## Results

| Evidence | A — fresh initialization | B — restored candidate checkpoint_0 |
|---|---:|---:|
| Exit/status | 0 / PASS | 0 / PASS |
| Dispatch seconds | 5.177128 | 5.289444 |
| Explicit sync seconds | 0.012155 | 0.011980 |
| True update seconds | 5.189362 | 5.301514 |
| XLA compilations started/finished | 99 / 99 | 99 / 99 |
| XLA dump files | 1,194 | 1,194 |
| Finite losses | PASS | PASS |
| RNG before | `[1278412471, 2182328957]` | `[595175366, 1025945940]` |
| RNG after | `[595175366, 1025945940]` | `[1738621784, 2837200184]` |
| GPU memory before/after | 146 / 190 MiB | 146 / 190 MiB |

A losses were actor `0.1566278338`, critic `132.1410217285`, and temperature `0.0182768479`. B losses were actor `0.1597654223`, critic `143.1664276123`, and temperature `0.0225411169`; all were finite.

No 30/90-second stack timer fired because both calls completed in about 5.3 seconds. The 1-second `nvidia-smi dmon` logs and per-thread CPU snapshots were retained. They show short compilation/execution activity rather than a prolonged synchronization wait.

## Differential findings

### Restore placement

The only decisive pre-update structural/runtime difference was placement:

- A: model and optimizer leaves were JAX `ArrayImpl` values on `cuda:0` with `SingleDeviceSharding`.
- B: all 198 restored state leaves, including all 150 optimizer leaves, were host NumPy values before the update call; device and sharding fields were absent.
- Shape, dtype, leaf path, leaf count, model structure, and optimizer structure remained compatible. The B update transferred/compiled the host-restored inputs and returned a finite GPU-backed updated state.

This is a real restore lifecycle difference, not an algorithm, batch, or checkpoint-content change.

### Compile and executable comparison

Both processes compiled `jit(update_high_utd)` exactly once. Compilation completed in 3.516035 seconds for A and 3.668261 seconds for B.

For `module_0197.jit_update_high_utd`, A and B have identical hashes for:

- before-optimization HLO text: `801a2fef328ff8dfd79e907a694b4440f0c464a648e6356c19ec53f2036bf2c7`
- optimized HLO text: `e1bbf2fdd486ef457147016bf0d777f873c8c63b603ac5f76645627bb2961c88`
- optimized LLVM IR: `9fb5449b7234516b423c1fe2ac0f1b2e1174f59edca6ab7ea6addd02385425c8`
- PTX: `ae70d704c3f1bfc8ef53dc4c18f263fa87cf2dcfeb1a24e87ac8b55ed39da977`
- thunk sequence: `4d8ceeab486969d84614041a6af1c72815d1240dde8eeb892a40db0bf24e69d5`

Therefore this run does not show an `update_high_utd` HLO, generated PTX, or thunk difference. The complete A/B XLA manifest hashes differ because the full processes include different initialization/restore-side artifacts:

- A XLA manifest: `c700c257cb0307dbd5ae2cdfba1ea1e41610a1ef7b0f29d10d4e6101942d85a7`
- B XLA manifest: `c15ec78cb715cf84bd4eff9ae0915b17b8277f54f4f72875d1882dec0453f0b1`
- overall artifact manifest: `18d416124f160428462346576a654dd41d20f52d7e0d171df040d5db36e5ec8a`

## Classification

`PEG_JAX_XLA_DIFFERENTIAL_CLASSIFICATION=RESTORE_DEVICE_PLACEMENT_DIFFERENCE`

This classification means that the experiment positively identified host-versus-GPU placement at the update boundary. It does **not** establish that placement caused the earlier 130-second stall: B completed successfully in this single no-retry trial, so the earlier stall was not reproduced. The evidence in this run does not support `XLA_EXECUTOR_STALL`, `CUDA_STREAM_EVENT_WAIT`, or `HOST_RUNTIME_WAIT`; it also does not show a materially different compiled update executable.

The prior timeout remains an intermittent/unresolved formal-readiness risk. A single successful restored update is insufficient to validate uninterrupted multi-update formal execution.

## Gate decision

- Differential diagnostic: COMPLETE
- Restored host-state placement difference: CONFIRMED
- Candidate next-update timeout reproduced: NO
- Root cause of the prior intermittent stall: UNRESOLVED
- `PEG_FORMAL_CANDIDATE_GPU_GATE`: remains FAIL
- Formal training: remains BLOCKED / NOT STARTED
- `runtime_gpu_formal.env`: must remain absent

## Evidence location

`/home/makoto/flexible_line_project/results/diagnostics/peg_jax_xla_differential_20260903`

The earlier setup-only import-path failure is preserved separately at `results/diagnostics/peg_jax_xla_differential_20260903_setup_failure_import_path`; it did not invoke either update experiment.
