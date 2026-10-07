# Peg Candidate Update 2 Diagnostic — 2026-09-03

## Scope

This diagnostic tested whether the formal-candidate `checkpoint_0` next learner update only stalls when it follows update 1 in the same process. It did not start formal training, write W&B history, save a checkpoint, change RNG, retry an update, or modify the algorithm and frozen training hyperparameters.

## Exact isolation

- Source: `results/checkpoints/peg_rlpd_gpu_formal_candidate_seed0/checkpoint_0`
- Profile: `diagnostic`
- Process: one fresh learner process
- Updates: one maximum
- RNG: restored unchanged from checkpoint metadata/state
- Update implementation: original `SACAgent.update_high_utd`
- XLA: `--xla_gpu_enable_triton_gemm=false`
- W&B: disabled
- Checkpoint period: zero
- External timeout: 130 seconds, no retry
- Python stack dumps: 30 and 90 seconds
- GPU sampling interval: approximately 10 seconds

## Observed result

Replay restoration, 32 online + 32 Demo sampling and batch concatenation completed. The original update call returned asynchronous outputs, but both Python stack dumps placed the main thread in `jax.block_until_ready(new_agent)`. The process did not return before the external limit and exited 124.

During the blocked interval, GPU memory stabilized at 196 MiB. Reported GPU utilization was mostly 4–8 percent and power was approximately 2.28–2.43 W, rather than sustained compute utilization. No learner process remained afterward. The source checkpoint bundle modification timestamps were unchanged.

## Classification

`PEG_CANDIDATE_CHECKPOINT0_NEXT_UPDATE_FRESH_PROCESS = FAIL_TIMEOUT`

The failure reproduces in a fresh process and without W&B or checkpoint saving. Therefore same-process continuity, W&B logging and the step-0 checkpoint save are not sufficient explanations. The wait occurs after asynchronous JAX dispatch, at device-result synchronization. The available evidence does not yet distinguish XLA compilation/runtime coordination, CPU-side executor work, CUDA stream/event waiting, or a driver/runtime interaction; the root cause remains unresolved.

## Evidence

- `results/diagnostics/peg_candidate_checkpoint0_update2_fresh_process_20260903/learner.log`
- `results/diagnostics/peg_candidate_checkpoint0_update2_fresh_process_20260903/gpu.csv`
- `results/diagnostics/peg_candidate_checkpoint0_update2_fresh_process_20260903/exit_status.txt`

Evidence SHA-256:

- learner log: `7850d9145112eae8487225055eafe2a04add63b21ada7e10eeb7f44f0ae6aa2b`
- GPU samples: `0249e3d81f69243b355dc526cb5041ca20fedd6c874e60b8b45156c6470d7636`
- exit status: `ca2ebdf97d7469496b1f4b78958f9dc8447efdcb623953fee7b6996b762f6fff`

## Gate decision

Formal training remains blocked. The diagnostic must not be converted into a formal runtime workaround. A subsequent task may add XLA compilation/profile evidence under the same no-save/no-W&B/one-update boundary, but must not upgrade dependencies or change the algorithm.
