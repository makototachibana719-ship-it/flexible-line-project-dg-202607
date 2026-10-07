# Peg I2 — W&B Integration Test

Date: 2026-09-04. `PEG_I2_WANDB_20=PASS`.

Classification: **NO_STALL_OBSERVED_IN_THIS_RUN**. Formal candidate remains FAIL; formal training remains BLOCKED / NOT STARTED.

## Method and boundaries

Added only online W&B initialization, per-update scalar logging, and finish to the I1 learner. New isolated harness: `scripts/peg_i2_wandb_integration.py`. It directly reuses the unchanged I1 actor and learner, R2 dynamic replay sampling, fresh agent factory, normal RNG, original `SACAgent.update_high_utd(batch, utd_ratio=1)`, full synchronization and 20-update loop. No formal launcher was called or modified.

One learner PID 381, one actor PID 631. A separate read-only W&B verification process PID 1025 ran after both completed; it is not another learner. All three exited 0. No process restart, experiment retry or repeated online verification query. The installed W&B SDK's transport implementation was not modified.

Preserved I1 runtime: JAX 0.4.35; both actor and learner GPU `cuda:0`; `--xla_gpu_enable_triton_gemm=false`; preallocation false; EGL; JAX_LOG_COMPILES=1. Installed W&B reports 0.28.0; no dependency installation or upgrade occurred.

Replay starts with the same 121 online snapshot transitions and 5,512 Demo transitions as R2/I1, then accepts real actor data. Each update dynamically samples 32 online + 32 Demo. No model checkpoint restore/save; only the existing replay data and candidate bundle integrity hashes are read. No new checkpoint, formal runtime file, fixed RNG, split update, algorithm/network/optimizer/LR/batch/reward/success changes.

W&B logging executes after the synchronized update record and before I1's parameter publication. Its host enqueue-call latency is recorded separately from dispatch/sync update latency. `run.finish()` flushes the asynchronous logging path; cloud history is then verified. This is not a test of synchronous network acknowledgement per update or of all production logger hooks.

## W&B evidence

- Project: `flexible-line-peg-insertion`
- New run name: `peg_rlpd_gpu_i2_wandb_seed0`
- Run ID: `21p57czu`
- [Online run](https://wandb.ai/makototachibana719-beihang-university/flexible-line-peg-insertion/runs/21p57czu)
- Mode: online; API terminal state: finished.
- Cloud `_step` and `learner_update`: exactly 0–19, 20 rows.
- All cloud critic/actor/temperature losses match local logged values within the verifier tolerance (relative 1e-7, absolute 1e-9), and are finite.
- Initialization: 3.585391 seconds.
- 20 `run.log` calls: 0.000414–0.000701 seconds, average 0.000493 seconds. These measure SDK enqueue calls, not end-to-end network latency.
- Finish/flush: 2.591387 seconds.
- SDK output: 5 W&B files synced, 0 media, 0 artifacts. No checkpoint uploaded.

## Results and comparison

| Measurement | I1 | I2 |
|---|---:|---:|
| Continuous learner updates | 20/20 | 20/20 |
| Update 0, initial compile | 5.548642 s | 5.553555 s |
| Update 1, recompile | 4.626373 s | 4.356799 s |
| Updates 2–19 | 0.015297–0.032732 s | 0.011545–0.028590 s |
| Mean updates 2–19 | 0.020938 s | 0.020811 s |
| Initial/post-initial update compile count | 1 / 1 | 1 / 1 |
| Live transitions accepted | 1,111 | 1,201 |
| Parameter callbacks received by actor | 21 | 21 |
| Timeout/stall observed | No | No |

I2 actor completed 1,212 environment steps; final 11 transitions were still queued at normal stop, so they are not counted as accepted uploads. There were 35 accepted uploads during the learner update interval. Initial parameter publication plus 20 updated publications were recorded.

All 20 loss triples are finite. Initial learner RNG `[1278412471,2182328957]`; final RNG `[279913639,4177495616]`, with uninterrupted before/after progression matching I1/R2.

Final losses: critic `232.7230987548828`; actor `0.27666524052619934`; temperature `0.012736695818603039`. These are diagnostic values, not policy-quality evaluation.

Compilation counts refer specifically to `update_high_utd`: once at index 0, once at index 1, none at 2–19. Index 1 is WARNING by the existing 1–10 second threshold; all subsequent updates are NORMAL. No 30-second stack trigger or 120-second update timeout occurred. W&B operations also have bounded watchdogs; process supervision is bounded to 300 seconds.

## Evidence, command, integrity

WSL directory: `/home/makoto/flexible_line_project/results/diagnostics/peg_i2_wandb_20260904_115851/`.

Windows mirror: `D:/桌面/小挑/flexible_line_project/flexible_line_project/results/diagnostics/peg_i2_wandb_20260904_115851/`.

Contains `i2_result.json`, `updates.jsonl`, `communication.jsonl`, `actor_network.jsonl`, `actor_result.json`, `preflight.json`, `wandb_run.json`, `wandb_calls.jsonl`, `wandb_finish.json`, `wandb_remote.json`, learner/actor/verify logs, GPU dmon, CPU thread snapshots and local W&B files.

SHA256:

- `i2_result.json`: `bb43fad0187612b21d4b6a414278abce3a99702953803b67a2f45083c2669418`
- `updates.jsonl`: `16a06853a9c14b7bdca254c08a86453a75e43e4288bbd60bdd1ccdcccb03e743`
- `wandb_remote.json`: `1c46f4490117928944c69651460c1ca96ed23ff5f03369afface1d0870b293f7`

Executed once:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/makoto/flexible_line_project/external/serl/serl_launcher:/home/makoto/flexible_line_project/external/serl/franka_sim \
  /home/makoto/miniconda3/envs/serl_peg_gpu/bin/python -u \
  /home/makoto/flexible_line_project/scripts/peg_i2_wandb_integration.py --role supervise
```

Independent post-run process search found no I2 trainer, Peg formal trainer, or wandb-core residual. Candidate checkpoint_0 bundle hashes remained unchanged. I1 source SHA256 stayed `080072126202c323b6ad89f78440aeebf93101ee3db5afaea25c61793a9ccd4c`. No formal runtime file exists.

## Interpretation and next boundary

Adding online W&B init/log/finish to I1 did not reproduce the prior intermittent stall in this one short run. The compile/timing pattern remains consistent with I1/R2. It does not prove that W&B, its production integration, or network conditions can never contribute to a rare stall.

Checkpoint saving and full candidate integration remain untested in this ladder. A separately authorized I3 could add only checkpoint I/O, with independent output and preservation of existing checkpoints. I3 and formal training were not started.

## Files and rollback

Added only the I2 harness/report/new evidence; updated PROJECT_STATUS.md, TASKS.md and docs/change_log.md after adjacent `.before_peg_i2_20260904` backups. No existing training source was edited. Rollback of tracking changes can use these backups; retain experiment evidence and the online run.
