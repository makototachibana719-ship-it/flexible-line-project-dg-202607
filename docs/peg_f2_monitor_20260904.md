# Peg F2-Monitor — monitoring and STOP protection

2026-09-04. PASS (CPU/static hardening). Formal training NOT STARTED.

## Files Changed

- Added `training/peg_insertion/formal_monitor.py`, `scripts/test_peg_f2_monitor.py`, `scripts/build_peg_f2_gate.py`.
- Modified `training/peg_insertion/run_peg_training.py`, `formal_policy.py`, `update_observer.py`, `scripts/test_peg_f1.py`.
- Added `results/validation/peg/formal_readiness_v3.json` and `results/validation/peg/f2_monitor_20260904/` evidence; updated three tracking documents.

Existing modified files backed up on Windows/WSL as `.before_f2_monitor_20260904`. v2 and all old bundles preserved.

## Reason / implementation

F1 lacked the newly requested per-update execution guards. Formal-only wrapper calls the unmodified original update with exactly the received arguments (including omission of pmap_axis). It returns the original agent/info, without changing RNG generation, math, replay sampling, publication/log/save ordering or W&B binding. Existing per-leaf function waits on returned agent/info; delivered loop retains its ready-state wait through JaxView. Shared sync implementation unchanged; native cause UNRESOLVED.

Every formal learner process gets an exclusive UUID directory under its formal log directory, printed as FORMAL_MONITOR_DIRECTORY. `formal_monitor_updates.jsonl` records every successfully guarded global update, UTC time, dispatch/sync/true-update seconds, before/after RNG, three required finite scalar losses, replay size/actor counter read together under existing lock, startup_window for global0–9 and PID. Resume numbering starts at checkpoint step+1; separate directories preserve prior monitor evidence. W&B stays log_period10.

First failure writes `formal_status.json`: formal_status=FAIL, reason, failed_update, PID/time. Missing/NaN/Inf required losses stop with non_finite_loss; mismatched/nonadvancing RNG stops with rng_continuity_failure. Ordinary exceptions unwind through existing server/W&B cleanup. The120s watchdog spans entry RNG read, original dispatch, per-leaf sync and validation; native timeout writes failure then os._exit(124), without retry or saving incomplete state. Hard exit necessarily bypasses Python finally/W&B finish; separately authorized F2 supervisor must stop associated actor and audit run state.

Existing UpdateObserver snapshot implementation reused at30/60/110s. Files: update_N_DELAYs.stack.txt / gpu.txt / cpu.txt. F0 default remains30s. Collection is best effort;110s collection can be cut short at120s. No new independent snapshot implementation. true_update_seconds is entry-to-sync completion, not record-writing overhead.

## Commands Executed

Using WSL Ubuntu-22.04/makoto and locked serl_peg_gpu Python:

```text
python scripts/test_peg_f2_monitor.py
env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false MUJOCO_GL=egl python scripts/build_peg_f2_gate.py
```

No training, checkpoint resume, JAX GPU execution, W&B initialization or cloud mutation. Fault-injection subprocesses are synthetic CPU functions, with test-only0.05s deadlines; production remains120s. Snapshot command calls mocked in tests.

## Tests

73/73 PASS:22 monitoring +37 role/Gate/cadence +8 W&B schema +6 old policy. Includes dispatch/sync hard-exit subprocesses; finite/missing losses; RNG mismatch/nonadvance/normal chain; every-update/startup/global resume records; no retry; canceled-timer race; snapshot schedule/files; actual wrapper argument forwarding; frozen source/runtime/config/Demo; stale v2 and wrong monitor/source/runtime/config/Demo rejection.

Python AST and shell syntax PASS. Actual fresh learner preflight PASS (127.0.0.1:5488); no service started. Other roles fixture-tested and must recheck at launch.

Evidence: `results/validation/peg/f2_monitor_20260904/tests.json`, `syntax.json`, `actual_fresh_preflight.json`, `protected_before.json`, `result.json`.

v3 SHA256:9967d06cb4e35715b1ed7fdb262ce1ae6f3462f10e5a5e3955b5a94871a9039c.

Preserved v2 SHA256:1e6bf0c8ce3e80b233984984ca2ca51231b1568549a2b7e4dc88b000a350f916. v2 remains historical PASS but stale for current source. v3 binds current sources including monitor/observer/tests/builder, monitor hash/flags, unchanged runtime/config/Demo, historical GPU evidence and new CPU evidence. No new GPU acceptance claimed.

## Result

```text
PEG_F2_FORMAL_WATCHDOG=PASS
PEG_F2_FINITE_LOSS_GUARD=PASS
PEG_F2_RNG_CONTINUITY_GUARD=PASS
PEG_F2_LOCAL_STEP_MONITOR=PASS
PEG_F2_WANDB_CADENCE_UNCHANGED=PASS
PEG_F2_CHECKPOINT_CADENCE_UNCHANGED=PASS
PEG_F2_MONITOR_TESTS=PASS
PEG_F2_GATE_V3=PASS
PEG_FORMAL_READINESS_V3=PASS
PEG_F2_EXECUTION=READY_TO_START
formal_training_started=false
wandb_run_created=false
algorithm_changed=false
sync_policy_changed=false
max_steps_changed=false
checkpoint_period_changed=false
wandb_log_period_changed=false
```

## Remaining Issues

Native sync cause UNRESOLVED. CPU hardening is not new GPU/live-supervisor validation. Real launch must recheck v3/roles and supervise both processes, all requested STOP conditions and checkpoint audits. Hard timeout cannot promise graceful W&B finalization. No visible execution terminal opened because training is deferred. max_steps10000/checkpoint1000/W&B10 unchanged.

## Next Recommended Task

Await separate F2 execution authorization; then show execution/monitor terminals. No automatic training/resume/evaluation. Rollback uses backups plus explicit policy revalidation; do not bypass v3 or treat old v2 as current.
