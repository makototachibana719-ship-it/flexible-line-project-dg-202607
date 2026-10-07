# Peg F0-Observe — 2026-09-04

## Result

```
PEG_F0_OBSERVE=COMPLETE
PEG_F0_OBSERVED_STALL_PHASE=FULL_SYNCHRONIZATION
PEG_F0_OBSERVE_CONTINUOUS_50=FAIL
PEG_F0_OBSERVE_EXACT_ROOT_CAUSE=UNRESOLVED
PEG_FORMAL_TRAINING=BLOCKED
```

Exactly one fresh production-path attempt; learner PID377 exit124; actor PID419 terminated by the supervisor (exit -15). No retry, resume or formal run. Original F0 output and failed Gate remain intact; no formal runtime file was created.

## Instrumentation and invariants

Existing entry and verifier backed up as `.before_f0_observe_20260904` on Windows/WSL. Added acceptance-only `--observe-update1`, metadata-only state/batch signatures (type, shape, dtype, weak_type, devices, sharding, PyTree structure), phase JSONL, separate dispatch/sync timings and 30-second Python/GPU/CPU snapshots. Enabled JAX_LOG_COMPILES=1. New output root: `/home/makoto/flexible_line_project/results/validation/peg/f0_observe_20260904`.

Original update_high_utd invocation, including explicit pmap_axis=None, remains unchanged. Existing jax.block_until_ready(new_agent) plus info-leaf synchronization remains unchanged. No placement, RNG, replay bootstrap, algorithm, hyperparameter, publication/log/save ordering, dependency or cadence changes. The original 50-update bound and actor 1500-step bound remain. New W&B run ID3ux2jgwl uses the same display name but independent identity/output; final cloud status was not queried. Checkpoint_0 was saved normally; it was not resumed. Observation adds I/O/timing overhead, so it is not a zero-perturbation experiment.

## Phase evidence

| Phase | Update 0 | Update 1 |
|---|---|---|
| Before original update | observed | observed |
| Original Python return | dispatch 5.321879 s | dispatch 4.429683 s |
| Before full synchronization | observed | observed |
| Synchronization complete | 0.010958 s | not observed |
| After-update signature/RNG | captured | unavailable because synchronization never completed |

The 120-second watchdog covers original update plus synchronization, not 120 additional seconds after dispatch. At timeout, update1 had spent approximately 115.5 seconds after Python return without completing the synchronization marker.

Update1 log explicitly reports tracing, MLIR conversion, and `Finished XLA compilation of jit(update_high_utd) in 3.110994816 sec`, followed by Python-return and before-sync markers. Thus this attempt's long wait is after that compilation completed, not an unfinished Python update call. Do not extrapolate this to every historical stall.

At 30 seconds, the learner main-thread stack is:

```
jax/_src/api.py:2754 block_until_ready
run_peg_training.py:196 candidate_update
delivered async_sac_state_sim.py:430 learner
```

Installed JAX source line2754 calls `xc.batched_block_until_ready(arrays)`. The info synchronization on the same source line follows this agent barrier. Since the agent has198 JAX-array leaves, the bulk barrier stack matches the first agent synchronization. Successful harness instead iterates leaf.block_until_ready(). This is an evidence-backed investigation target, NOT proof of a bug in the batched barrier itself: it may be waiting for incomplete GPU work.

GPU snapshot: RTX4060, 417MiB/8188MiB, 8% utilization, P8. CPU main thread: futex_do_wait; reported26.9% is ps lifetime-average CPU, not instantaneous utilization. W&B asyncio thread is in selector wait, Agentlace request thread in receive loop; these snapshots do not establish W&B or communication deadlock. No native CUDA stream/event trace was collected, so CUDA_STREAM_EVENT_WAIT is not proven.

## Signatures

- Full train state has198 leaves; pre-update1 all198 are on GPU with replicated PositionalSharding.
- Batch metadata signatures at indices0/1 are identical; this does not mean identical sampled values.
- Post-update0 and pre-update1 state leaf metadata are identical; this does not compare parameter values.
- Initial state to pre-update1 changes8 temperature-related scalar weak_type flags true -> false (params/target lagrange plus six optimizer mu/nu lagrange leaves). Dtype staysfloat32. This is compatible with specialization/recompilation but is not causal proof of the stall.
- RNG0: [1278412471,2182328957] -> [595175366,1025945940]. RNG before1 equals that after0. RNG after1 is unavailable; do not infer completion or a second optimizer update from Python return alone.

## Tests and preserved evidence

Six existing policy/history tests passed. Python syntax compilation passed for the modified entry and new observer/runner. Exactly one live experiment failed at index1. No additional training test/retry. Raw result retains inherited profile=smoke label; CLI is acceptance plus observe flag. No Gate promotion. Lightweight logs/signatures/stack/process records are mirrored to Windows; model/replay remain in WSL.

## Next scoped task

Review batched versus per-leaf synchronization and relevant state/executable readiness. A future single-variable experiment would require separate authorization, keep placement/RNG/data/call form/order fixed, change only synchronization implementation, and retain the same one-shot timeout. This report does not authorize that experiment, dependency changes, automatic retry or formal training.
