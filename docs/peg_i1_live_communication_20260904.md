# Peg I1 — Live Actor Communication Integration Test

Date: 2026-09-04. Result: `PEG_I1_LIVE_COMMUNICATION_20=PASS`.

`PEG_FORMAL_CANDIDATE_GPU_GATE=FAIL`; `PEG_FORMAL_TRAINING=BLOCKED` remain unchanged.

## Scope and method

One fresh learner process (PID 375) and one real MuJoCo actor process (PID 587), each launched once. The isolated `scripts/peg_i1_live_communication.py` harness reuses the existing R2 runner, without changing its original `SACAgent.update_high_utd(batch, utd_ratio=1)`, sampling iterators, factory, full synchronization, or normal learner RNG progression.

The only added integration is Agentlace TrainerServer/TrainerClient on ports 5595/5596: actor action sampling and environment transitions, uploads every 30 actor steps, initial parameter publication and publication after each completed learner update. The actor applies incoming parameters to its agent. Publication is timed separately, outside the original update dispatch/synchronization timing. The R2 functions are wrapped only inside this diagnostic process; no persistent edits to the R2 runner, delivered algorithm, or formal launcher occurred.

R2's initial 121 online transitions and 5,512 Demo transitions were retained for comparability, then live transitions were appended. This is not an empty-replay startup test. The learner waited for at least 31 live transitions; 61 had arrived by the recorded preflight, giving 182 online transitions. Every update resampled 32 online + 32 Demo transitions using the original replay iterators. Batch observations `(64,20)`, actions `(64,7)`.

JAX 0.4.35, GPU `cuda:0` on both processes, `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MUJOCO_GL=egl`, `JAX_LOG_COMPILES=1`. W&B initialization/logging is never invoked; the subprocess environment also sets `WANDB_MODE=disabled`. No model checkpoint restore/save, fixed learner RNG override, split update, dependency change, or retry. Existing R2 replay-sidecar data and checkpoint integrity hashes were read; no checkpoint bundle was written.

## Observed results

| Check | Evidence |
|---|---|
| Continuous original learner updates | 20/20; one learner PID; exit 0 |
| Actor | 1,140 actual environment steps; exit 0; stopped by learner completion |
| Live uploads accepted | 1,111 transitions; 32 uploads overlapped learner update interval |
| Steady-state live overlap | During indices 2–19: 4 uploads / 120 transitions, 18 received parameter callbacks |
| Network publication / reception | 1 initial + 20 updated publications; actor received 21 callbacks |
| Index 0 | 5.548642 s, initial compilation |
| Index 1 | 4.626373 s, WARNING, one real `update_high_utd` recompile |
| Indices 2–19 | 0.015297–0.032732 s; mean 0.020938 s; all NORMAL |
| Parameter publication overhead | 0.012811–0.028728 s, mean 0.019630 s; separate from update time |
| Compile count for `update_high_utd` | Initial 1, post-initial 1 at index 1, none at indices 2–19 |
| Losses | All 20 critic/actor/temperature losses finite |
| RNG | All before/after transitions advance and chain continuously |
| Timeouts | None; no 30-second snapshot triggered |
| Cleanup | Both child exit codes 0; independent process check found no residual I1/Peg trainer |
| Protection | Candidate checkpoint_0 bundle hashes unchanged; no formal runtime file |

Initial learner RNG `[1278412471,2182328957]`; final `[279913639,4177495616]`, matching the R2 normal 20-update RNG progression.

Final losses (diagnostic values, not policy quality): critic `204.1107635498047`, actor `0.2668692171573639`, temperature `0.017581524327397346`.

The actor's last 29 transitions remained in its upload queue when the learner finished. There was deliberately no post-completion flush/retry; actor step count is not represented as the learner's received-transition count.

## Evidence and reproducibility

WSL evidence: `/home/makoto/flexible_line_project/results/diagnostics/peg_i1_live_20260904_114647/`.

Windows evidence mirror: `D:/桌面/小挑/flexible_line_project/flexible_line_project/results/diagnostics/peg_i1_live_20260904_114647/`.

Files: `updates.jsonl` (per-step dispatch/sync/true-update/sample time, RNG, losses), `communication.jsonl`, `actor_network.jsonl`, `preflight.json`, `result.json`, `actor_result.json`, `i1_result.json`, `learner.log`, `actor.log`, `gpu_dmon.log`, `cpu_threads.log`, `protected_before.json`. GPU dmon and CPU thread snapshots were collected at approximately 1-second intervals.

SHA256:

| File | SHA256 |
|---|---|
| learner.log | `648f3db4899027348665c13783fc143cfcc83753ef23e567335cf4fcfb47ef04` |
| updates.jsonl | `276280832fe7dbd91ad2abbf3c0a0c6232189ba8d4805c71e11cde901861a61a` |
| communication.jsonl | `e9eeabc865897879c0c1ef1f03cf8f1ecf8bb077be7d0fa92975f72d63ee6ac8` |
| i1_result.json | `493e58fea62d67873b8b68d291f4a2d8f2347d065212f6fbfa5be0f2307393a0` |

Executed once (not an instruction to rerun automatically):

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/makoto/flexible_line_project/external/serl/serl_launcher:/home/makoto/flexible_line_project/external/serl/franka_sim \
  /home/makoto/miniconda3/envs/serl_peg_gpu/bin/python -u \
  /home/makoto/flexible_line_project/scripts/peg_i1_live_communication.py --role supervise
```

Per-update dispatch + full synchronization has a 120-second watchdog and 30-second Python/GPU/CPU snapshot trigger inherited from R2. The supervisor has a 300-second process deadline and owns cleanup of its children. Actor safety cap: 3,000 steps or 180 seconds; actual termination followed learner completion. No automatic restart or retry.

## Interpretation and limits

Live communication did **not reproduce** intermittent stall in this one 20-update run. Its timing pattern remains initial compile, one index-1 recompile, then normal steady state. This is evidence against communication alone being a sufficient trigger under these tested conditions, not proof that communication can never contribute to a rare stall.

This diagnostic omits W&B, checkpoint serialization, formal launch configuration, long-duration load, and full candidate lifecycle. It does not establish their correctness or release formal training. A subsequent independently authorized integration rung should add only one omitted subsystem (W&B or checkpoint I/O) while holding the rest fixed; neither was executed here.

## Changes and rollback

Only a new isolated diagnostic harness, this report, new evidence, and project tracking updates were added. Existing tracking documents were backed up as adjacent `*.before_peg_i1_20260904` files before modification. No algorithm, network, optimizer, LR, batch ratio, Demo, reward, success, existing smoke/candidate output, formal launcher, or dependency was changed.
