# Peg Checkpoint Restore Device-Placement Causal Diagnostic — 2026-09-03

## Scope

This diagnostic tested only whether explicitly moving a restored Peg train state to GPU improves consecutive-update stability.

- D1 restored formal-candidate `checkpoint_0` and retained the existing host NumPy state at the first update boundary.
- D2 restored the same checkpoint, explicitly applied `jax.device_put` to the complete train state, blocked every leaf until ready, and confirmed all 198 leaves as JAX arrays on `cuda:0` before the first update.
- Each arm ran once in a fresh Python process, used normal checkpoint RNG progression, the original `SACAgent.update_high_utd`, one learner process, the same immutable batch, at most five consecutive updates, and a 120-second hard timeout per update.

Formal training, the formal launcher, W&B, checkpoint writes, fixed diagnostic RNG, automatic retry, dependency or algorithm changes, and `runtime_gpu_formal.env` were excluded.

## Comparability checks

All required comparisons passed:

- Same candidate checkpoint and initial RNG: `[595175366, 1025945940]`
- Same batch SHA-256: `854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c`
- Same 32 online + 32 Demo contract
- Same state leaf paths, shapes and dtypes before placement
- Same GPU/backend and disabled Triton GEMM setting
- D1 post-control placement: 198 host NumPy, 0 GPU JAX arrays
- D2 post-placement: 0 host NumPy, 198 GPU JAX arrays
- D2 placement and full synchronization: 0.028590 seconds

## Update results

| Update | D1 true / dispatch / sync (s) | D2 true / dispatch / sync (s) |
|---:|---:|---:|
| 0 | 5.061709 / 5.047213 / 0.014492 | 5.135037 / 5.123948 / 0.011085 |
| 1 | 0.027453 / 0.008193 / 0.019254 | 0.021685 / 0.003256 / 0.018424 |
| 2 | 0.014816 / 0.001998 / 0.012814 | 0.015822 / 0.002044 / 0.013774 |
| 3 | 0.013413 / 0.002075 / 0.011334 | 0.013979 / 0.001693 / 0.012282 |
| 4 | 0.016410 / 0.002170 / 0.014235 | 0.013688 / 0.002045 / 0.011640 |

Both arms completed 5/5 updates with exit status 0, no timeout, and 99/99 logged compilations completed. Update 0 includes compilation; subsequent cached updates completed in approximately 0.014–0.027 seconds.

The complete RNG sequence and all actor/critic/temperature losses were bit-for-bit identical between D1 and D2. The final RNG in both arms was `[3350198582, 799687021]`. Every reported loss was finite.

## Resource evidence

The short runs showed no prolonged GPU or host wait:

- D1: 18 GPU samples, mean/max SM utilization 5.00%/18%; 18 main-thread CPU samples, mean/max 61.26%/70.9%.
- D2: 15 GPU samples, mean/max SM utilization 4.53%/21%; 15 main-thread CPU samples, mean/max 78.83%/100%.
- No 30-second stack timer fired in either arm because all updates completed before the threshold.
- No residual training process remained.

Evidence hashes:

- D1 probe log: `47ed5d66ab4047a35f62ebfd2bf433866189e5d183109c7e617e766de2e40aa5`
- D1 GPU dmon: `f684d31d2ea304dd863218487d2fac57b2b3e81f174d9677a85ce17a5e132d52`
- D1 CPU snapshots: `6e16ff3af112660bda4fa2e23d9a809edcbf892888b089fb72c4f470a170317f`
- D2 probe log: `1434239c994c43a7143b3bc1c0e16111264c3b524da933e8f8f98a539c35bdcf`
- D2 GPU dmon: `3b09824de5d5abea5a265baef05033b0813facf68954689cd2029519affd1499`
- D2 CPU snapshots: `347ebeebabea61903c33e3c77dbdb4aad5a5d5c414fb805571ccda629dea8d3c`
- Causal summary: `4c75e00c3adea627a26f9c9031cd8ce22fb2b8636565749561f333101aeb79b0`

## Classification

`PEG_RESTORE_PLACEMENT_CAUSAL_DIAGNOSTIC=NO_EFFECT`

Within this bounded one-shot comparison, explicit GPU placement did not improve or reduce stability. D1 and D2 completed the same number of updates, produced identical RNG/loss sequences, and had comparable timing after compilation. The restored host placement therefore is not a sufficient cause of the previously observed intermittent stall.

This does not prove that placement can never interact with the failure. Neither arm reproduced the intermittent timeout, and five successful updates do not satisfy the existing 20-update formal-candidate Gate.

## Safety verification and gate state

- No checkpoint was written; the source checkpoint/model/replay/metadata hashes remained unchanged.
- No W&B process or output was created.
- No dependency, algorithm, network, optimizer, learning rate, batch, replay, reward or success setting changed.
- The formal launcher was not modified or invoked.
- `runtime_gpu_formal.env` remains absent.
- Formal training remains NOT STARTED and BLOCKED.
- `PEG_FORMAL_CANDIDATE_GPU_GATE` remains FAIL.

## Evidence directory

`/home/makoto/flexible_line_project/results/diagnostics/peg_restore_placement_causal_20260903`
