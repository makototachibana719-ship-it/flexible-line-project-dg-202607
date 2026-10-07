# Peg F0-R — Production-path checkpoint branch resume acceptance

## Result

```text
PEG_FORMAL_EXACT_PATH_CHECKPOINT_RESUME_5=PASS
PEG_F0_R_EXACT_PATH_CHECKPOINT_RESTORE_5=PASS
PEG_F0_R_WANDB_MODE=NEW_BRANCH_RUN
PEG_F0_R_SOURCE_RUN=hr88a6u1
PEG_F0_R_SOURCE_CHECKPOINT=checkpoint_25
PEG_F0_R_LEARNER_UPDATES=26-30
PEG_F0_R_SAME_RUN_HISTORY_CONTINUATION=NOT_TESTED
PEG_FORMAL_TRAINING=BLOCKED
sync_exact_root_cause=UNRESOLVED
retry=false
restart=false
formal_training_started=false
```

This is a new branch from historical model progress25, NOT a continuation of the original run's model at49. Original run hr88a6u1 and its steps0–49 remain unchanged. The previously suggested mapping to same-run steps50–54 was not implemented.

## Provenance and execution path

- Source: `/home/makoto/flexible_line_project/results/validation/peg/f0_sync_20260904/checkpoints/checkpoint_25`.
- New W&B run: `2243685c`, name `peg_rlpd_gpu_f0_resume_from25_seed0`, project `flexible-line-peg-insertion`, state `finished`.
- URL: https://wandb.ai/makototachibana719-beihang-university/flexible-line-peg-insertion/runs/2243685c
- W&B config and new bundle `source_snapshot` contain `validation_type=exact_path_resume_branch`, `resume_branch=true`, source run/path/step and `same_run_history_continuation=false`.
- Output: `/home/makoto/flexible_line_project/results/validation/peg/f0_r_branch_20260904`.
- Actual execution: bounded supervisor → `run_peg_f0_branch_role.sh` → **production `run_peg_training.py --resume-from`** → unchanged delivered learner/actor → original `SACAgent.update_high_utd`.
- No replacement learner loop, split update, diagnostic RNG, replay bootstrap change, or one-update-per-process. Per-leaf synchronization remains enabled at both learner synchronization points. Runtime policy and baseline config unchanged.
- Existing general formal/smoke resume shell launchers were not changed or exercised; this acceptance uses a dedicated bounded shell into the shared production Python resume path. It does not certify the outstanding general formal launcher hardening.

## Minimal scoped changes

- Backed up `run_peg_training.py` as `.before_f0_r_branch_20260904` on both hosts before changing it.
- Added explicit acceptance-only branch flag; exact B source bundle hash and prior re-audit checks; source-config validation remains strict and distinct from new run metadata.
- Fresh learner restores via existing `load_bundle` and `restore_fresh_state`; online replay restores via unchanged `restore_replay` before being exposed to communication threads. GPU placement remains the delivered main's normal placement.
- New W&B identity uses `resume='never'`, not the source run. Logical update and `_step` are26–30. Late actor stats beyond the bounded end are not logged as phantom update31.
- Actor scheduling starts at1261 using an actor-scoped range adapter; actor's environment and sampling RNG are newly initialized normally. No claim of restoring simulator state or actor sampling RNG (not present in source bundle).
- Existing relative periodic-save behavior is retained: local index0 saves actual checkpoint26. An explicitly bounded final save after the original learner loop produces checkpoint30. No early diagnostic checkpoint, no formal cadence modification. The general resume cadence's use of local versus global modulo is not certified by this test.
- Observer/assertions record exact restored state and replay; final saved state is compared against the actual post-update30 fingerprint.

## Tests and evidence

CPU-only prelaunch tests: **11/11 branch tests**, **8/8 schema tests**, **6/6 existing policy tests**, Python AST and shell syntax PASS. An initial test command had a shell-quoting error before tests ran; no training was launched by that command.

Exactly one learner PID390 and actor PID412 were launched. Learner exit0; actor was stopped by supervisor SIGTERM (exit-15), no restart/retry. Residual process query returned no production learner/actor.

| Check | Observed evidence |
|---|---|
| Model/optimizer/RNG restore |198/198 serialized state leaves exactly match source, both immediately after restore and before first update|
| Learner RNG before update26 |`[4098270448, 1939745508]`, exact source value|
| Online replay |All contents equal before communication|
| Replay RNG |Exact equality before normal iterator sampling|
| Source size/index/actor_steps |1261/1261/1261|
| Normal RNG progression |Each update advances; all adjacent before/after links match|
| Actor reconnect |Network callback markers present, new replay inserts accepted|
| W&B |Exactly five logical rows26–30; all three dotted loss fields finite and match local logs; no duplicate warning; runfinished|
| Final model state |198 leaves finite and match post-update30 fingerprints|
| Internal train/optimizer counters |52→62; these internal optimizer counters are distinct from outer learner-update indices25→30|
| Old artifacts |Source checkpoint triplets, original raw audit/cloud files and formal Gate hashes unchanged|
| Immutable code |Delivered learner/actor, resume.py, baseline config, runtime policy, formal policy, verifier and four general launchers match pre-task re-audit hashes|

| Actual learner update | Full update seconds | Critic loss | Actor loss | Temperature loss |
|---|---:|---:|---:|---:|
|26|5.797474|236.229446|0.357664|0.027262|
|27|0.033839|205.794342|0.304695|0.016764|
|28|0.019267|202.423416|0.424562|0.030183|
|29|0.020520|203.084351|0.415497|0.022823|
|30|0.018696|212.660202|0.431872|0.016840|

New bundles:

| Checkpoint | Replay size/index/actor_steps | Additional entries relative to source |
|---|---|---:|
|26|1651/1651/1651|390|
|30|1651/1651/1651|390|

`checkpoint_30` model SHA256: `852f9257567707a5d75357a87676793466079c73b071661a126d6e2059356f97`.

Both snapshots have the same replay count but distinct replay RNG sidecar hashes because normal sampling advances replay RNG. The live buffer continued receiving during W&B finish; summary records660 accepted entries relative to source (1921 total). This later live count is **not** the checkpoint snapshot count and must not overwrite metadata1651.

On shutdown actor logged send warnings after learner service closed. These are preserved in `actor.log`; they are not an update timeout. No residual process remained.

Primary evidence: `audit.json`, `offline_final.json`, `state_restore.json`, `state_before_first_update.json`, `replay_restore.json`, `final_state_fingerprint.json`, `learner/updates.jsonl`, `learner/phases.jsonl`, `branch_cloud.json`, `branch_cloud_logical_steps.json`, `source_cloud_before.json`, `source_cloud_after.json`, `protected_before.json`, `processes.json`, `residual.json`. Lightweight evidence is mirrored to the Windows project; model/replay binaries remain in WSL.

## Commands executed

1. CPU test scripts `test_peg_f0_branch.py`, `test_peg_f0_schema.py`, `test_peg_f0_policy.py` under the locked `serl_peg_gpu` interpreter.
2. **Once only**: `bash scripts/run_peg_f0_branch_acceptance.sh`.
3. Read-only supplemental artifact audit: `python scripts/audit_peg_f0_branch_offline.py`.
4. Source/old bundle/Gate hashes, cloud source history before/after, process cleanup checks and evidence mirroring.

## Remaining issues / next task

The exact production Python checkpoint restore/continuation path is accepted for this bounded branch. This is not same-run continuation of hr88a6u1. No new same-run continuation claim is made.

Formal training remains BLOCKED. Original failed formal Gate is deliberately preserved; no `runtime_gpu_formal.env` was created. Per-leaf native root cause remains UNRESOLVED. Next separately authorized task should review role-aware fresh/actor/resume preflight, output/run binding and checkpoint cadence semantics, then decide how to produce a new source-bound readiness artifact. Do not automatically start formal or resume this new checkpoint.

Rollback: restore the entry from its named backup only with authorization, retaining all new evidence. Tracking docs also have `.before_f0_r_branch_20260904` backups.
