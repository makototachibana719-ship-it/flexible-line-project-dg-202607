# Peg runtime-only continuous update reliability diagnostic

Report finalized 2026-09-04. One-shot experiment evidence is retained under `results/diagnostics/peg_runtime_reliability_20260903`.

## Result

```text
PEG_RUNTIME_FIXED_BATCH_20=PASS
PEG_RUNTIME_DYNAMIC_BATCH_20=PASS
PEG_RUNTIME_RELIABILITY_CLASSIFICATION=RECOMPILE
```

Both fresh-process arms completed 20/20 original `SACAgent.update_high_utd` calls with finite losses, uninterrupted normal RNG progression, no restart, and no retry. Both had one real post-initial compilation at index 1. No stall was observed. `RECOMPILE` identifies this observed behavior, not a proven explanation for the earlier intermittent timeout.

## Controlled scope

- Fresh agent initialized with the normal seed-0 factory; no agent state was restored and no diagnostic RNG was assigned.
- Original SAC update, UTD=1, unchanged network/optimizer/LR, 32 online + 32 Demo.
- GPU backend, `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, preallocation disabled, EGL, and `JAX_LOG_COMPILES=1`.
- R1 reused immutable batch SHA-256 `854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c`.
- R2 populated new in-memory replay buffers from 121 read-only online transitions and the unchanged 5,512-transition Demo. Every iteration used the delivered `get_iterator(batch_size=32, device=sharding.replicate())` for each buffer and the delivered `concat_rlpd_batches`. Loading transition data is not agent/checkpoint resume; neither model state nor saved replay RNG was restored.
- The source online snapshot and Demo hashes were checked before insertion. No existing smoke/candidate artifact was written.
- No actor, W&B initialization, checkpoint saving, formal launch, formal runtime file, dependency upgrade, or split update.

## Timing and compilation

| Metric | R1 fixed | R2 dynamic |
|---|---:|---:|
| Completed updates / exit | 20/20 / 0 | 20/20 / 0 |
| index 0 true seconds | 5.087420 | 4.950054 |
| index 1 dispatch seconds | 4.260347 | 4.289753 |
| index 1 sync seconds | 0.015863 | 0.016802 |
| index 1 true seconds / class | 4.276215 / WARNING | 4.306559 / WARNING |
| index 2–19 true seconds | 0.013366–0.022967 | 0.014674–0.019520 |
| index 2–19 class | 18 NORMAL | 18 NORMAL |
| Initial update compile count | 1 | 1 |
| Post-initial recompile count | 1 | 1 |
| Bootstrap compile count | 98 | 102 |
| Total compile starts / finishes | 100 / 100 | 104 / 104 |
| 30-second captures / timeout | none / none | none / none |

Every update's index, dispatch/sync/true time, RNG before/after, finite actor/critic/temperature losses, sample time and latency class is retained in each arm's `updates.jsonl` and the combined JSON summary. PASS here means 20 completed finite updates with no ANOMALOUS or TIMEOUT latency; the observed WARNING is retained rather than hidden.

The compile logs place the second `Compiling update_high_utd` at index 1 in both arms. Logged weak-typed argument occurrences change from 9 at index 0 to 1 at index 1. This is evidence of an input abstract-signature change consistent with recompilation; it is not a controlled causal test of individual leaves. No further compilation is logged for indices 2–19.

XLA compilation durations at indices 0/1 were 3.418175/3.072392 seconds in R1 and 3.435742/3.076190 seconds in R2. Thus the index-1 delay is accompanied by compilation, not a long measured device-result synchronization wait.

## RNG, GPU and integrity

Both arms start at `[1278412471, 2182328957]` and end at `[279913639, 4177495616]`. Every RNG-before equals the preceding RNG-after, and each update advances the key. All 20 loss records in each arm are finite. Different batches are not expected to produce equal loss values.

Both report GPU backend on the RTX 4060 Laptop GPU. GPU memory before/after is 146/254 MiB in each arm. Continuous GPU dmon and CPU-thread logs were captured. The automatic 30-second Python/GPU/CPU snapshot code was not triggered because no update exceeded 30 seconds; it was not runtime-tested by inducing a failure.

Log SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| R1 probe | `e41c22c47bd5a4fb4db516f785674e18617a751ad0ac2aca649994d1f61c9b74` |
| R1 GPU | `09cb69620ea89d42ec9bf14efddac2e3cc3dce5e96a9f5d51aef207c93762638` |
| R1 CPU | `26d59a465b2fb8ed576ab44427ff3ed9978c171870cd581f3ce866358378f7cf` |
| R2 probe | `0ac3b9d720f669795344b261bfc3889ec61ecdb52f9aa7288dbf2096f950f03a` |
| R2 GPU | `447fcf492bb97f62f68b5229791853fe032d51626c7ff290b14fa449b2d28de8` |
| R2 CPU | `6b97fb51ce5b112a26042b8dda62ba46b7d758e02e396c099ea5c6a3e0a58d9f` |

Candidate checkpoint model/replay/metadata hashes still match their pre-experiment values. No residual runtime-probe or Peg training process remained. `runtime_gpu_formal.env` remains absent.

## Analysis correction and limitations

The initial analyzer returned UNRESOLVED because it only selected RECOMPILE when a timeout/failure also occurred. Raw logs prove recompilation even though both completion gates pass. The analyzer was backed up and its classification branch corrected; the original JSON summary was preserved as `runtime_reliability_summary_before_classification_fix_20260904.json`. Only analysis of existing logs was rerun. Neither GPU experiment was retried.

This provides bounded evidence for runtime-only execution with a frozen online dataset, including normal dynamic replay sampling. It does not validate live actor communication, W&B, periodic checkpoint I/O, resume, long-duration reliability or policy performance. No data supports DYNAMIC_BATCH_DEPENDENT or INTERMITTENT_STALL in this trial.

`PEG_FORMAL_CANDIDATE_GPU_GATE` remains FAIL; formal training remains BLOCKED / NOT STARTED. Integration must be separately accepted before any formal launch.
