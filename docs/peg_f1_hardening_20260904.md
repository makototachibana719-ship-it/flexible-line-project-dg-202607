# Peg F1 — Formal Readiness Hardening

## Result

```text
PEG_F1_FORMAL_SYNC_POLICY=PASS
PEG_F1_ROLE_AWARE_PREFLIGHT=PASS
PEG_F1_RUN_BINDING=PASS
PEG_F1_GLOBAL_CHECKPOINT_CADENCE=PASS
PEG_F1_FORMAL_RUNTIME_FREEZE=PASS
PEG_F1_GATE_ENFORCEMENT=PASS
PEG_FORMAL_READINESS=PASS
PEG_FORMAL_TRAINING=READY_TO_START
formal_ready=true
formal_training_started=false
gpu_training_run=false
sync_exact_root_cause=UNRESOLVED
```

Gate: `/home/makoto/flexible_line_project/results/validation/peg/formal_readiness_v2.json`.
Evidence: `/home/makoto/flexible_line_project/results/validation/peg/f1_hardening_20260904/`.

READY_TO_START is readiness, not training completion or authorization to auto-launch. No formal learner/actor, GPU training, new W&B run, cloud write, or checkpoint30 resume occurred in F1.

## Evidence scope

- Existing B `hr88a6u1`:50-update production acceptance PASS_WITH_PER_LEAF_SYNC, based on corrected existing-artifact re-audit.
- Existing F0-R `2243685c`:5-update production checkpoint branch resume PASS. NOT same-run continuation of hr88a6u1.
- Current F1: Python AST/shell syntax,33 new CPU policy tests,8 existing schema tests,6 existing policy tests: **47/47 PASS**.
- Actual clean fresh-learner output paths and new Gate were checked read-only. Actor join/resume learner/resume actor were exercised using synthetic CPU fixtures with mocked endpoint/cloud boundaries; no live role launch is claimed. Every real future launch repeats the corresponding checks.
- New source hashes bind current F1 code. Historical GPU evidence is explicitly historical, not an assertion that F1 code was GPU-rerun.

## Changes

1. `training/peg_insertion/sync_policy.py`: shared per-leaf waits, used by the project update wrapper and delivered learner's JAX view, for formal/acceptance/resume. No batched full-tree wait fallback. No placement, network, SAC math, optimizer, LR, batch, reward, success or Demo changes.
2. `formal_preflight.py`: four fail-closed role policies and Linux process locks. General production Python entry repeats preflight while holding the corresponding role lock, preventing shell-preflight/launch races. Actor scheduling reads metadata progress, not guessed actor steps.
3. `run_peg_training.py`: source-bound default formal config only; learner session publication; explicit fresh run identity with `resume=never`; same-run restore with `resume=must`; global logger/checkpoint indices, normal formal server/W&B cleanup.
4. Delivered `async_sac_state_sim.py`: only progress plumbing changed: two start-index flags, actor range offset, learner `update_steps` initialized from global start. Original SAC call, sample/publication/log/save order and update loop body remain unchanged. A small delivered-code edit is required because the save predicate lives inside that loop; remapping filenames alone cannot fix cadence.
5. `resume.py`: new bundles record model SHA256 in addition to existing replay SHA256. Formal saves additionally bind source/runtime/config/Demo through metadata `source_snapshot.formal_binding`. Legacy evidence and bundles were not rewritten or relabeled as F1 formal artifacts.
6. `formal_policy.py`: v2 Gate enforcement with19 source-file hashes, runtime/config/Demo/manifest hashes, historical and current test evidence hashes, role policy results, fixed run binding and global cadence. Missing/false/stale Gate or mismatch fails closed.
7. Four general formal launchers explicitly source the frozen formal runtime, require role-aware preflight, and enter the guarded Python path. Smoke resume remains separate. Formal resume actor automatically reads metadata actor_steps.
8. Added `scripts/test_peg_f1.py` and `scripts/build_peg_f1_gate.py`. Builder creates a new Gate only after passing tests and actual fresh-output checks; refuses overwrite.

Existing changed files were backed up as `.before_f1_20260904` on Windows and WSL before modification. Tracking documentation uses the same backup suffix.

## Role-aware policy

| Role | Required checks |
|---|---|
| Fresh learner |Gate/runtime/source/config/Demo PASS; canonical config and fixed fresh run name; clean checkpoint/log/actor-log dirs; no existing learner/actor lock; locked interpreter and JAX versions|
| Actor join |Existing learner output permitted; live learner lock/session and process-start identity; matching run/project/env/config/source/endpoint; localhost:5488 reachable; exclusive actor lock|
| Resume learner |Existing checkpoint dir permitted; exact formal output path; complete triplet and model/replay hashes; config/Demo/source/runtime/run binding; coherent replay counters and RNG presence; no actor still running; same-run cloud history cannot be ahead of source checkpoint|
| Resume actor |Same triplet/binding checks; metadata actor_steps; matching live learner source checkpoint/session/endpoint; exclusive actor lock|

Endpoint policy intentionally accepts only localhost/127.0.0.1 for the validated single-host setup. Remote actor deployment is not silently authorized. Actor join is expected to be invoked after learner initialization/service startup, with its actual WANDB_RUN_ID; a missing/unready session is rejected, not retried automatically.

Fresh W&B: `project=flexible-line-peg-insertion`, `name=peg_rlpd_baseline_seed0`. Resume: exact source run ID, `resume=must`. If cloud history extends beyond the checkpoint, preflight rejects it and requires a separately approved branch policy rather than offsetting steps or overwriting history. F1 performed no live cloud query; the boundary logic is tested using fixtures/mocks.

## Checkpoint cadence

Formal config remains `max_steps=10000`, `checkpoint_period=1000` (config SHA256 unchanged).

- Fresh outer learner indices0–9999 save at0,1000,…,9000.
- Restore checkpointN: next outer learner indexN+1, remaining budget10000−(N+1).
- Delivered save predicate now sees the global `update_steps`, not local process index.
- Logger and save wrapper take that global value directly; no double resume offset.
- Unit regression uses the actual delivered save-condition AST: resumed26–30 saves no periodic checkpoint;999–1001 saves1000;1001–3000 saves2000 and3000.
- Bounded F0-R's explicit final checkpoint behavior is historical and separate from formal cadence. Old one-shot validation scripts are preserved and are not instructions to rerun old output directories under changed source.

## Runtime freeze

`configs/peg_insertion/runtime_gpu_formal.env` contains exactly:

```text
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
MUJOCO_GL=egl
```

File SHA256: `ed29e50af02dd02b5583850ad19ec4572908c4758f2d4b8b80c3ef0d4fcb23d1`.
Launchers use the locked `serl_peg_gpu` interpreter; preflight checks JAX/JAXlib0.4.35 through package metadata without initializing GPU. Unapproved inherited JAX_DISABLE_JIT/JAX_PLATFORMS/JAX_PLATFORM_NAME/CUDA_VISIBLE_DEVICES overrides are rejected. No dependencies were installed/upgraded.

## Tests / commands

- `python scripts/test_peg_f1.py`:33 tests PASS, including four roles, dirty output, missing triplet, wrong model/source/run/counter, duplicate locks, wrong runtime/env/config/Demo/evidence, missing/stale Gate, shared per-leaf dispatch and global cadence.
- `python scripts/test_peg_f0_schema.py`:8 PASS.
- `python scripts/test_peg_f0_policy.py`:6 PASS.
- `python scripts/build_peg_f1_gate.py`: AST parse of bound Python files, `bash -n` of bound shell files, repeat47 tests, actual fresh-output preflight, protected artifact hash check, exclusive v2 Gate creation, final require_gate PASS.
- Read-only process inspection: no production learner/actor found; no actual formal coordination directory was created.

Old `formal_readiness.json` FAIL is preserved. B and F0-R checkpoint triplets are SHA256 unchanged. No historical FAIL/PASS evidence was deleted or modified. New Gate is the current approval artifact; older entries in status documents remain historical.

## Remaining issues / next action

- Native cause of the batched synchronization stall remains UNRESOLVED; per-leaf is the approved tested policy.
- F0-R proved branch continuation, not same-run continuity of the original B history. General same-run guards were CPU-tested here; this is not a new live same-run experiment.
- Formal performance, completion, independent evaluation and final training delivery are not yet measured.
- A later explicit user authorization may start formal training. F1 itself does not do so. Any bound source/config/runtime/Demo/evidence change invalidates the Gate and requires re-audit; do not edit the Gate to bypass a failure.

Rollback requires restoring named backups and treating v2 as stale; do not delete old evidence or silently preserve `formal_ready=true` after code rollback.
