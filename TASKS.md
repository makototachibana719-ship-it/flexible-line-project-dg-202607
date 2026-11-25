# Project Tasks

Dedicated environment backup 2026-11-25: preserve VMware `serl_peg` independently of all other environments, verify sender/receiver/archive-part hashes and local reassembly before allowing any subsequent guarded environment deletion. Leave `serl`, `serl_clean`, base and shared caches untouched.

Storage archival 2026-11-25: preserve the 1,606 scoped VMware Peg files under `archives/vmware/2026-11-25/`, verify transfer and per-file SHA-256, and check staged Git blob hashes before remote publication. No further guest deletion belongs to this upload task.

Peg Vision DrQ + Demo Replay V1 STOPPED/FAIL at engineering GPU smoke: compatibility, isolated visual env, camera preflight, five distinct physical Demo replays (878 transitions), Object replay, AgentLace online transport and GPU inference all PASS. Two unchanged-recipe batch256 first-update probes timed out at 15 minutes with 0 completed updates, so 100-update smoke, checkpoints, scientific training and canonical evaluation are NOT_RUN. `NEXT=VISION_DRQ_FAILURE_REVIEW`; do not start 500+ training from these artifacts. Evidence: `results/evaluation/peg_vision_drq_demo_v1_20260910_190000/MASTER_REPORT.md`.

Peg F3 minimal actor Demo-action anchor design VALIDATED: isolated all-Demo additive MSE anchor, `lambda_small=0.07`, gradient contribution 9.8207%, baseline optimizer/critic/environment/data semantics preserved. A fresh 1K ablation requires separate user authorization.

Peg F3 Demo-anchor 1K closure remains BLOCKED in Stage 1. The authorized `dedent` fix and one static rerun passed parsing but failed the next proof: after wrapper installation the inspected callable is not the immutable production factory, so no production `create_states` call is present. Prelaunch/workload/checkpoints/audit remain NOT_RUN (`UPDATES=0`). Next: immutable production-factory identity/proof-order Gate fix.

Peg F3 async RLPD binding fix VALIDATED; Q-bias config-only 1K retry completed but scientific result FAILED. The full runtime/checkpoint/shutdown chain passed, but Q bias and Demo-action alignment regressed at matched step. Fresh 5K R2 remains BLOCKED.

Peg F3 runtime Python binding Gate equivalence fix VALIDATED: symlink-equivalent binaries within the exact `serl_peg_gpu` prefix are accepted; system/base/other/Windows interpreters are rejected. `ABLATION_RETRY=READY`, not started.

F3 V2 Smoke Shutdown/Restore VALIDATED: read-only evidence locates the shutdown boundary after actor result persistence and before `env.close`, at `TrainerClient.stop()` / unbounded `BroadcastClient` thread join. Training computation and stop observation are proven complete; exact receive-callback state remains unavailable. Existing checkpoint_1000 restore acceptance PASS in a fresh non-training process with exact step/config/dataset/V2 bindings, 198 finite leaves, and 9331 restored replay entries. `SMOKE_ADJUDICATION=PASS_WITH_SHUTDOWN_LIMITATION`; `TRAINING_READY=YES_AFTER_SHUTDOWN_FIX`. Next requires a separately authorized minimal shutdown lifecycle fix and bounded static/cleanup acceptance, not a 1000-update rerun. Evidence `results/validation/peg/f3_v2_smoke_shutdown_restore_20260906/REPORT.md`.

F3 V2 RLPD Smoke 1000 STOPPED/FAIL: one fresh bounded GPU run reached 1000 updates with canonical V2 Demo=1921, online replay 91->9361, exact 32:32 sampling, finite losses/critic/temperature, live actor communication and durable checkpoint_500/checkpoint_1000. The actor did not exit inside the 30-second shutdown boundary after writing its result, so fail-stop skipped restore verification (`NOT_RUN`) and left `TRAINING_READY=NO`. No retry, residual process, formal training, resume or 5k run. Next requires separate authorization for a read-only shutdown-boundary review and checkpoint restore acceptance using the preserved bundle. Evidence `results/validation/peg/f3_v2_rlpd_smoke_1000_20260906/REPORT.md`.

F3 V2 Dataset Rebuild VALIDATED: canonical V2 dataset publication completed from exactly 11 frozen accepted episodes (1921 transitions; 3/2/2/2/2 by standard/+Y/-Y/roll/pitch). Pre-publication source/provenance/schema/contact/success Gates and post-publication hash/full-load/replay sample Gates PASS. `TRAINING_SMOKE=READY` for separate authorization; no RLPD smoke, training or resume ran. Evidence `results/validation/peg/f3_v2_dataset_rebuild_20260906/REPORT.md`.

F3 V2 Dataset Provenance Routing Fix VALIDATED: explicit Five-Pilot versus formal acceptance authorities now route all 11 accepted episodes without weakening path/hash/plan checks. Ten negative/positive route categories pass, including rejection of unknown source, wrong manifest, wrong SHA and V1. `DATASET_REBUILD=READY`; rebuilding remains separately authorized. Evidence `results/validation/peg/f3_v2_provenance_routing_fix_20260906/REPORT.md`.

F3 V2 Dataset Build BLOCKED: the first build attempt stopped before output publication because the builder used the Five-Pilot standard identity rule for a formal standard episode. No dataset was created; loader/replay tests and training are NOT_RUN. Next task is a minimal provenance-authority routing fix with CPU/static tests, followed by separately authorized build retry. Evidence `results/validation/peg/f3_v2_dataset_build_20260906/REPORT.md`.

F3 V2 Formal Rotation Pitch +1 VALIDATED: one independent 182-transition `_003` episode passed calibrated frozen-plan, V2, pitch peak/correction, near-hole, success, contact and axial-tracking audits. Accepted total=11; rotation pitch=2. `V2_DATASET_BUILD=READY` for a separately authorized task. Evidence `results/validation/peg/f3_v2_formal_rotation_pitch_20260906_003/REPORT.md`.

F3 V2 Formal Rotation Roll +1 VALIDATED: one independent 179-transition `_002` episode passed frozen-plan, V2, rotation peak/correction, near-hole, success, contact and axial-tracking audits. Accepted total=10; rotation roll=2. `NEXT_FORMAL_DEMO=ROTATION_PITCH_READY` for a separate task. Evidence `results/validation/peg/f3_v2_formal_rotation_roll_20260906_002/REPORT.md`.

F3 V2 Formal Lateral -Y +1 VALIDATED: one independent 176-transition `_002` episode passed canonical symmetry/plan, V2, challenge/correction, near-hole, success, contact and axial-tracking audits. Accepted total=9; lateral -Y=2. `NEXT_FORMAL_DEMO=ROTATION_ROLL_READY` for a separate task. Evidence `results/validation/peg/f3_v2_formal_lateral_minus_20260906_002/REPORT.md`.

F3 V2 Formal Lateral +Y +1 VALIDATED: one independent 176-transition `_004` episode passed frozen-plan action/terminal-success semantics, V2, challenge/correction, near-hole, success, contact and axial-tracking audits. Accepted total=8; lateral +Y=2. `NEXT_FORMAL_DEMO=LATERAL_MINUS_READY` for a separate task. Evidence `results/validation/peg/f3_v2_formal_lateral_plus_20260906_004/REPORT.md`.

F3 Formal Lateral +Y Invocation Fix VALIDATED: the new invocation-only launcher binds the accepted 176-command +Y canonical plan and fixed WSL Python directly, executes the existing verifier, validates fresh output paths, and passes dry-run without starting the simulator. `LATERAL_PLUS_FORMAL_RERECORD=READY`; recording remains a separate task. Evidence `results/validation/peg/f3_lateral_plus_formal_invocation_fix_20260906/REPORT.md`.

F3 V2 Formal Standard Second VALIDATED: one new independent 165-transition standard episode passed exact frozen-stream, V2, near-hole, success, contact and axial tracking audits. Accepted total=7; standard=3. `NEXT_FORMAL_DEMO=LATERAL_PLUS_READY` for a separate task. Evidence `results/validation/peg/f3_v2_formal_standard_second_20260906_003/REPORT.md`.

F3 V2 Formal Standard +1 VALIDATED: one new 165-transition standard episode completed successfully with exact frozen-stream binding, V2 continuity/sensor/hash Gates, 29 near-hole rows, zero contacts and healthy axial tracking. Accepted demos=6; standard=2. `NEXT_FORMAL_DEMO=STANDARD_SECOND_READY` for separate authorization only. Evidence `results/validation/peg/f3_v2_formal_standard_20260906_002/REPORT.md`.

F3 Standard Formal Invocation Fix VALIDATED: stable launcher argv/stdin binding, frozen 165-command stream hash, seed0/max_steps500/type/output parsing, extra-token rejection and pre-record dry-run all PASS. `STANDARD_FORMAL_RERECORD=READY` for separate authorization; no recording or collection retry was run. Evidence `results/validation/peg/f3_standard_formal_invocation_fix_20260906/REPORT.md`.

F3 LateralPlus PlanPathFix VALIDATED: canonical WSL-relative binding, exact hash/count/content validation and five path regressions PASS. Missing report copies are irrelevant; wrong hash/count and Windows paths fail closed. Plan unchanged. LATERAL_PLUS_RERECORD=READY for separate authorization; no simulator/`_002`/training. Evidence `results/validation/peg/f3_lateral_plus_plan_path_fix_20260905/REPORT.md`.

F3 V2 LateralPlus Rerecord-002 STOPPED/FAIL at preflight: invoked plan path was absent in WSL although the accepted176-command plan remains intact at the earlier auto-closure validation path. No path switch/retry was allowed; simulator, `_002`, recording, audit and training were NOT_RUN. Next requires separate authorization using the exact accepted WSL plan path. Evidence `results/validation/peg/f3_v2_lateral_plus_rerecord_002_20260905/REPORT.md`.

F3 LateralPlus BootstrapFix VALIDATED: standard project-root bootstrap is the only helper change. Direct WSL compile/import and independent176-command validation PASS; all phases legal, correction is `lateral_alignment`, actions are7D finite/in bounds, and preserved prefix/insertion action parity holds. LATERAL_PLUS_RERECORD=READY for separate authorization only; simulator and `_002` were not started. Evidence `results/validation/peg/f3_lateral_plus_bootstrap_fix_20260905/REPORT.md`.

F3 LateralPlus-AutoClosure STOPPED/FAIL at CPU/static validation: planned correction now uses legal `lateral_alignment`, but direct WSL execution of the new validator failed to import the project `training` package before the plan could be validated or published. No retry allowed in this task; simulator/recording/audit/later pilots/training were NOT_RUN. Next requires separate authorization for the minimal import/bootstrap repair and a new static-first closure attempt. Evidence `results/validation/peg/f3_lateral_plus_auto_closure_20260905/REPORT.md`.

F3 V2 RemainingPilots STOPPED/FAIL: lateral +Y produced109 durable transitions and established the frozen 4–8mm challenge without contact, then failed before the first correction action because `lateral_correction` is not a frozen recorder phase. Episode is INTERRUPTED/ineligible/unapproved. Sequential stop applied: lateral -Y, roll and pitch NOT_RUN; no retry, promotion or training. Next requires a separately authorized collection after command-sequence static validation using only existing phases. Evidence `results/validation/peg/f3_v2_remaining_pilots_20260905/REPORT.md`.

F3 PostExclusionPilotAudit VALIDATED (offline): protocol/reset/goal/control/V2 bindings PASS and all126x7 actions match exactly. Only intended loaded production semantic change is the Link01 exclusion. The prior12mm-to0.00882149mm axial nonresponse became12.0825768mm realized post-exclusion. STANDARD_V2_RERECORD=READY, but requires separate authorization for one standard pilot; do not auto-record it or continue the other four pilots. Evidence `results/validation/peg/f3_post_exclusion_pilot_audit_20260905/REPORT.md`.

F3 AxialBoundedDiagnostic-AfterTelemetryHardening DONE / diagnostic PASS: one authorized fixed-reset run completed126 frozen actions with complete passive telemetry. Axial request12.0000005mm realized12.0825768mm; historical nonresponse was NOT reproduced. IK/joints/actuation/conditioning/contact are not blocking, Link01 did not reappear, and exact action parity with the preserved failed pilot supports Link01 exclusion causal contribution. Next requires separately authorized read-only PilotAudit and decision on standard-only re-recording; do not auto-record remaining pilots or train. Evidence `results/diagnostics/peg/f3_axial_bounded_after_telemetry_hardening_20260905/REPORT.md`.

AxialDiagnosticContractFix VALIDATED: live reset durable-first ordering and semantic post-exclusion gate implemented; 12 CPU tests PASS in Windows/WSL and real saved pre/post fixtures pass frozen tolerances. Production/guard/exclusion/telemetry hashes unchanged. AXIAL_DIAGNOSTIC=READY for separate authorization only; do not automatically run it, record pilots, or train.

AxialBoundedDiagnostic-Retry STOPPED/FAIL: exactly one invocation, fixed reset, then PRE_ACTION fail-closed on an invalid full-observation equality check against pre-exclusion evidence. No axial/IK telemetry and 0 actions; causal fields remain UNRESOLVED. Next proposed P0: runner-only initial snapshot persistence plus invariant/tolerance compatibility gate review, followed by separately authorized diagnostic. Do not rerun, record pilots, or train automatically.

Link01ExclusionFix VALIDATED: exactly one source-level `base_link`/`shoulder_Link` exclusion propagated through generation; XML compile and reset-only contact validation PASS. Reset has no contacts, original guard remains unchanged/not triggered, and no action was dispatched. AXIAL_DIAGNOSTIC=READY for separate authorization; do not automatically rerun it, record pilots, or train. Evidence `results/validation/peg/f3_link01_exclusion_fix_20260905/REPORT.md`.

GeometryFixReview COMPLETE: static evidence supports LIKELY_MISSING_EXCLUSION. Official Aubo SRDF excludes all adjacent arm pairs; local MJCF contains none, and reset link0/link1 contact is exact. Next proposed P0: separately authorize a one-pair source exclusion review plus compile-only/reset-only validation. Do not edit the guard or rerun axial diagnosis automatically.

Link01CollisionIntentAudit COMPLETE: static configuration established, semantic intent unresolved. Next proposed P0: read-only link0/link1 collision-mesh/upstream design review before any narrowly scoped exclusion proposal. Do not modify guard/geometry or rerun axial diagnostic.

BaselineContactReview-Retry COMPLETE: evidence collection PASS, guard triggered, zero actions. Next proposed P0 is static pair-aware collision intent review for link0-link1; do not loosen guard or rerun bounded diagnostic automatically.

DiagnosticSerializationFix COMPLETE (CPU19 PASS). Await separate reset-only review authorization with new exclusive output path. Do not automatically run baseline/axial diagnostic, pilots or training.

BaselineContactReview STOPPED: serialization error before guard, zero actions. Next proposed task: diagnostic JSON NumPy scalar normalization and CPU fixture regression, then separately authorized reset-only review. Do not retry or adjust guard automatically.

InitEvidenceFix COMPLETE: CPU persistence/guard tests 7/7 PASS. Await separate authorization for initial baseline contact collection/review; do not rerun bounded diagnostic or remaining pilots automatically. Historical FAIL preserved.

AxialBoundedDiagnostic FAIL: one attempt stopped before action1 on initial penetration guard; no IK/axial trace. P0 proposed diagnostic-runner fix: persist complete initial contact/joint snapshot before guard; identify triggering pair and baseline semantics. No automatic repair/retry or remaining pilots. Root cause UNRESOLVED.

AxialTelemetry VALIDATED (CPU/static): passive observer and18 tests complete. Next P0: seek explicit bounded simulator diagnostic authorization using observer.step and completeness checks. Do not continue four pilots or retrain. CPU tests do not establish real trajectory parity or axial root cause.

AxialTrackingAudit PARTIAL: target audit PASS; deeper localization BLOCKED on missing historical joint/IK/actuator/contact telemetry. Next proposed P0: authorize telemetry-only instrumentation and separately bounded diagnosis; do not change IK/controller/limits/geometry speculatively. No new pilot in this audit.

F3 V2 DemoPilot STOPPED: standard FAIL (126 transitions, controlled tracking-divergence stop); lateral +/- and roll/pitch NOT_RUN. Next P0: read-only axial IK/limits/geometry/target tracking review using this trace, then seek authorization before any repair or new pilot. No automatic retry, collection expansion or training.

F3 V2 RecorderReady DONE: independent V2 recorder and CPU preflight passed. Next: separately authorize the frozen five-pilot collection protocol; do not auto-record. Success/rotation coverage review and dataset release remain TODO; training BLOCKED.

Last updated: 2026-09-04

F3 DataReview DONE:20-episode offline quality audit and V2 pilot/acceptance spec frozen. Next P0 TODO: separately authorize V2 recorder readiness hardening + CPU/preflight, then pilot recording. Collection BLOCKED, no automatic Demo/retraining. Evidence `results/validation/peg/f3_data_review_20260904/REPORT.md`.

F3 SensorFix VALIDATED (CPU): sensor address/dim lookup and observation contract versioning; legacy direct V2 use rejected. Next P0: V2 data/compatibility plan, missing Ty/Tz not recoverable from current demos. No automatic retraining or legacy formal evaluation. Evidence `results/validation/peg/f3_sensorfix_20260904/REPORT.md`.

F3 ContractAudit DONE: static chain,5 Demo episodes/full-data rotation coverage and1 fixed9000 trace. Confirmed torque address bug; controller tracking quality FAIL, policy/data causality unresolved. P0 TODO: separately authorize minimal sensor-address CPU repair and schema compatibility review; do not silently change old-checkpoint observations. Final comparison BLOCKED. Evidence `results/validation/peg/f3_contract_audit_20260904/REPORT.md`.

F3 Diagnostic VALIDATED:2x1 fixed-scene episodes; complete per-step data, key states and SVG plots. Best joint ratios5000=26.2649,9000=13.3361; skill still failed. Next separately scoped action/controller/Demo/sensor audit; no automatic training/screening/final evaluation. See `results/validation/peg/f3_diagnostic_20260904/REPORT.md`.

F3 Stage2 VALIDATED execution / skill NOT PASSED:5000–9000 each10 episodes, all0/10; protected hashes unchanged. Ranking is diagnostic only, no final-eval Top2. Next separately authorize trajectory failure-mode review; no automatic50-episode evaluation or training. See `results/validation/peg/f3_stage2_20260904/REPORT.md`.

F3 Stage1 VALIDATED: checkpoint9000 inference smoke10/10 execution completed,0/10 task success. Preflight/load PASS, screening READY but not started per latest instruction. Next separately authorize Stage2 and freeze undefined-success/precision tie-break handling; fixed-reset limitation must be disclosed. See `results/validation/peg/f3_stage1_20260904/REPORT.md`.

F2-RetentionFix VALIDATED (CPU):62/62 tests PASS; bundle-aware retention and fail-closed partial detection implemented. No historical checkpoint recovery or deletion. Next: separately review surviving3000–9000 artifacts/missing-history exception and source-bound approval (include new retention module); evaluation remains BLOCKED. See `results/validation/peg/f2_retention_20260904/REPORT.md`.

F2-ExitFix VALIDATED:83 CPU tests; minimal normal-SystemExit classification fix. Readiness v4/evaluation BLOCKED by independent retention bug: keep20 includes sidecars, removes early bundle files. Retained3000–9000 hashes valid; no training retry/resume/evaluation. Next requires explicit bundle-aware retention repair scope. See `results/validation/peg/f2_exitfix_20260904_review/REPORT.md`.

F2-Run DONE / overall FAIL: formal10000 updates and10 checkpoint audits PASS; W&B loss history complete but final statefailed due success-SystemExit misclassification in unchanged finally. No retry/resume/evaluation. Next scoped task: separately authorized exit-status CPU repair/reapproval; preserve run e6fea100 and checkpoint9000. See `results/validation/peg/f2_run_20260904/REPORT.md`.

F2-Monitor VALIDATED (CPU/static):73 tests plus syntax/actual fresh preflight PASS; v3 monitoring Gate READY_TO_START. Formal-only watchdog/finite-loss/RNG/local monitor, no math/sync implementation/runtime/cadence changes. No GPU/formal/W&B run. v2 preserved but stale; await separate execution authorization. See `docs/peg_f2_monitor_20260904.md`.

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `VALIDATED`.

`DONE` means implementation or documentation is complete. `VALIDATED` requires relevant testing and evidence.

## P0 — Blocker

F3 AxialTelemetryFailureEvidenceFix VALIDATED: 44 CPU/static tests PASS. Passive observer failures now have frozen stage classification, exact exception/availability/missing-field evidence, strict NumPy-safe serialization, exclusive temp+fsync+atomic no-clobber publish+directory fsync, and explicit non-durable fail-closed status if evidence publication itself fails. Normal schema and single env.step/controller/solver behavior are unchanged. AXIAL_DIAGNOSTIC=READY for separate authorization only; no simulator/Demo/training. Evidence `results/validation/peg/f3_axial_telemetry_failure_evidence_20260905/REPORT.md`.

F3 AxialBoundedDiagnostic-FinalRetry DONE / diagnostic FAIL: exactly one post-exclusion run passed fixed-reset persistence, repaired invariant gate and unchanged contact guard, then stopped at setup step1 on incomplete passive telemetry. One action only, no axial phase, no contacts, Link01 did not reappear. Axial nonresponse and downstream cause remain unresolved. P0 next: separately authorize passive-observer failure-artifact hardening and CPU/static proof; no automatic simulator retry or Demo/training. Evidence `results/diagnostics/peg/f3_axial_bounded_final_retry_20260905/REPORT.md`.

F1 Formal Readiness Hardening VALIDATED:47/47 CPU tests, source/shell syntax and actual fresh-output preflight PASS. v2 source-bound Gate created; formal_ready=true; formal READY_TO_START but NOT STARTED. Four general launchers guarded; per-leaf sync shared; resume global cadence fixed without changing period1000/max_steps10000. Old Gate/evidence preserved. Next: await explicit formal-launch authorization; no automatic launch. See `docs/peg_f1_hardening_20260904.md`.

F0-R branch restore VALIDATED: production `run_peg_training.py` resume, 5/5 true updates26–30, new W&B branch2243685c, exact model/optimizer/RNG/replay restore and checkpoint30 integrity PASS. Not same-run continuation of hr88a6u1. No retry/restart; general formal launchers not accepted by this bounded result. Formal BLOCKED; next scoped task is role-aware entry/readiness review. See `docs/peg_f0_r_branch_20260904.md`.

F0-Verifier-Fix VALIDATED:8 schema tests plus6 existing policy tests PASS. Existing B artifact re-audit PASS_WITH_PER_LEAF_SYNC without retraining. Formal remains BLOCKED; exact-path resume and role-aware preflight remain unvalidated. Original failed audit/Gate preserved; no formal runtime file.

F0-Sync B arm VALIDATED for one50-update run: per-leaf synchronization passes update1 and all later updates; no timeout/retry/resume. Complete Gate still FAIL due empty filtered W&B history query; investigate read-only, do not assume missing cloud data. Formal BLOCKED; workaround remains acceptance-only. See `docs/peg_f0_sync_20260904.md`.

F0-Observe COMPLETE, training acceptance FAIL: update1 compiled and returned, but full synchronization did not finish before120s. Next proposed scope: batched/per-leaf synchronization investigation with no other changes; any further live experiment requires separate authorization. Formal BLOCKED. See `docs/peg_f0_observe_20260904.md`.

Peg F0: implementation PARTIAL; exact-path bounded acceptance FAIL / formal BLOCKED. One and only one 50-update attempt completed 1 update and timed out at update index 1 (120 s). No F0-R/resume or retry permitted by this result. Preserve `results/validation/peg/f0_acceptance` and failed `formal_readiness.json`. Next scoped work: read-only exact-path/harness differential diagnosis; separately review role-aware fresh/resume preflight (current all-output-empty check is overly restrictive) and timeout record profile label. Existing six policy tests do not validate these future formal paths. See `docs/peg_f0_20260904.md`.

Peg I4 new-bundle resume lifecycle: VALIDATED WITH WARNING. Five updates20–24, complete learner/optimizer/RNG/replay restore, normal RNG, real actor and new counter-consistent checkpoint24 passed. Same W&B run finished, new five steps complete; raw verifier FAIL due to duplicate identical historical steps0–19 is retained. Pure cached-history audit result `PASS_WITH_WANDB_HISTORY_WARNING`. Investigate historical duplication read-only before full readiness approval. No retry; formal BLOCKED. See `docs/peg_i4_resume_20260904.md`.

Peg I3-Fix snapshot/counter repair: VALIDATED. Same-lock insert/counter/sampling/snapshot boundary, detached memory copy before disk I/O, no live-counter rewind. 5/5 concurrency regressions preceded one bounded I3 PASS; new bundle counter audits pass. See `docs/peg_i3_snapshot_fix_20260904.md`. Historical I3 FAIL is retained below. New-bundle resume remains NOT STARTED; formal BLOCKED.

Peg I3 normal checkpoint I/O: experiment DONE, acceptance BLOCKED / FAIL. 20 updates, communication, W&B and saves passed, but live actor-counter invariant failed (30 then 60 undercount). Raw script PASS is superseded by `i3_counter_audit.json`. Normal save snapshot/counter repair and concurrent-insert test require separate authorization; no resume or formal allowed on this result. See `docs/peg_i3_checkpoint_integration_20260904.md`.

Peg I2 W&B integration is VALIDATED for one bounded run: I1 + online W&B completed 20/20, run `21p57czu` finished with verified cloud steps/losses, no stall, normal RNG, no checkpoint saving. See `docs/peg_i2_wandb_integration_20260904.md`. Checkpoint-I/O integration is NOT STARTED and requires separate authorization. Formal remains BLOCKED.

Peg I1 live communication diagnostic is VALIDATED for one bounded 20-update run: 1,111 received transitions, 21 received networks, finite losses and normal RNG, no stall. See `docs/peg_i1_live_communication_20260904.md`. Full candidate remains FAIL and formal training BLOCKED. Further integration tests require separate scope/authorization; W&B and checkpoint I/O were not enabled in I1.

| Task | Status | Notes |
|---|---|---|
| Cable Routing environment smoke validation | VALIDATED | Cable V2 supplies Franka assets and completed import/create/reset/one-step/close in `serl` with EGL. |
| Object Grasping environment smoke validation | VALIDATED | ENVIRONMENT_SMOKE_VALIDATED in `serl` with `MUJOCO_GL=egl`; observation-space warning remains. |
| Peg Insertion environment smoke validation | VALIDATED | ENVIRONMENT_SMOKE_VALIDATED in `serl_mujoco` with EGL and project-owned generated-XML staging. |
| Define three-task GPU handoff gate | DONE | Gate A–G and evidence rules are documented; each process remains NOT_READY until every required item is validated/prepared. |
| M1D-A Cable GPU handoff finalize | VALIDATED | All handoff preparation gates, including bounded safe resume, have evidence. READY does not validate GPU smoke or policy quality. |

## P1 — Critical

| Task | Status | Notes |
|---|---|---|
| Resolve package-specific dependency environments | DONE | Runtime Environment Discovery VALIDATED; candidate compatibility still requires per-process smoke. |
| Validate visual observation adapter contract | TODO | Resolve `images/{front,wrist}` versus current SERL `image` wrapper using an adapter/wrapper, not package source edits. |
| Runtime Environment Discovery | VALIDATED | Found Conda, `serl_mujoco`, and `serl`; direct imports/version checks passed in both environments. |
| Cable Routing Demo Recorder design | VALIDATED | Project keyboard recorder, success-only gate, TEST_ONLY dry run, static validator, and replay-consumer insertion passed; desktop key-capture preflight remains before acquisition. |
| Cable MuJoCo Viewer rendering | DONE | Passive Viewer remains project-side and display-only; native key callback is disabled after operator evidence showed Viewer shortcut conflicts. Visible desktop rendering still needs operator confirmation. |
| Cable pynput teleoperation | VALIDATED | Human logs confirm pynput key states and correct signed nonzero actions for W/S/A/D; action construction is VALIDATED. Viewer native callback remains FAILED_FOR_TELEOP. |
| Cable action pipeline diagnosis | DONE | Target/mocap, opspace ctrl, qpos, and true TCP all respond. Root causes: 0.05 magnitude is too small for clear motion, and recorder debug display uses wrong flattened TCP indices. |
| Cable TCP debug display fix | VALIDATED | `TCP_POS_REAL` now reads `2f85/pinch_pos`; bounded smoke confirmed equality with pinch site (`TCP_MATCH=True`). Formal observation/transition schema unchanged. |
| Cable key-event suppression | DONE | `KEY_SUPPRESSION = IMPLEMENTED` with pynput 1.8.2 `suppress=True` and unconditional stop/join cleanup; `OPERATOR_SUPPRESSION_VALIDATION = REQUIRED`. |
| Cable operator episode time limit | DONE | Root cause is Gym 500 steps plus native 10.0s/500 steps. Recorder-owned instance now uses 20000 steps/400.0s and prints every episode boundary reason; operator confirmation remains required. |
| Cable recorder Gear speed | DONE | Gear 1/2/3 perform 1/2/5 real env steps per tick, default Gear 3, and preserve every transition. Two-second W displacement ordering remains OPERATOR_VALIDATION_REQUIRED. |
| Cable recorder cleanup result separation | VALIDATED | Five-second bounded listener join; post-save timeout reports cleanup WARNING without invalidating Demo, while zero-success cleanup failure still raises. |
| Cable robot teleop motion | TODO | OPERATOR_VALIDATION_REQUIRED: use magnitude 1.0 within action space `[-1,1]`, max 20000 steps, and print interval 5. |
| Cable Routing Demo acquisition | VALIDATED | Five real successful formal Demo files are PRESENT and all five pass the unchanged validator. |
| Cable initial-state randomization audit | VALIDATED | 20 reset-only samples with seeds 0–19 had exact zero span across TCP, robot/cable state, tip, slot, and cameras. FIXED_INITIAL_STATE; diversity risk HIGH. |
| Cable five-demo validation | VALIDATED | Five validator PASS results; 10,521 transitions, schema consistent, replay insertion size 10,526 as expected for horizon one, merge compatible. |
| Cable W&B training preflight | DONE | W&B 0.28.0/base URL and learner logging path were inspected without credentials or training. Online mode requires `debug=False`; target project/run identity still needs explicit configuration. |
| Cable project training launcher | DONE | Project entry/config implements explicit W&B identity, sequential five-file loading, observation adapter injection, isolated outputs, and assertions. Dry run PASS; real training path remains UNVERIFIED. |
| Cable multi-demo training input | DONE | Uses five configured pickle paths sequentially; no permanent merged duplicate. Dry run SHA/schema/count total 10,521 PASS; real buffer size assertion remains part of smoke startup. |
| Cable checkpoint save diagnosis/fix | DONE | Root cause `ORBAX_TENSORSTORE_MISMATCH`: Orbax 0.6.4 emits `assume_config`, rejected by TensorStore 0.1.51. Project legacy Flax save/restore PASS; no environment change. |
| Cable bounded full GPU smoke | TODO | Run only on the target GPU host after handoff acceptance. Historical local smoke remains PARTIAL; do not run the 100-update retry locally. |
| Cable project-side reset randomization design | TODO | Deferred robustness/ablation candidate; not a blocker for the fixed-reset baseline because train/eval/recorder initial states are in-distribution. |
| Cable automatic-demo checkpoint preflight | BLOCKED | Checkpoint inventory/call chain audited, but absolute-path restore of `cable_route_smoke/checkpoint_200` did not finish in bounded environment-derived or 180-second CPU-only attempts. Restore, architecture match, inference, and policy quality remain UNVERIFIED; zero episodes and no candidate demo. |
| Object Grasping algorithm mapping | VALIDATED | Code-level identity is `DRQ_PLUS_DEMO_REPLAY`: `DrQAgent` with 50/50 online/Demo batches. No distinct DPRL implementation evidence exists; the team-label meaning remains unverified. |
| Object Grasping checkpoint 4500 architecture mapping | VALIDATED | Exact match: 598 arrays plus three Optax EmptyState sentinels equals 601 structural paths in checkpoint and fresh target. Zero-gradient pre/post structure is unchanged. |
| Object Grasping checkpoint 4500 restore/inference | DONE | Frozen legacy policy: PRESENT, architecture MATCH, native restore UNRESOLVED, not required for fresh training and no longer a handoff blocker. No further restore work planned. |
| Object checkpoint read-stall diagnosis | VALIDATED | Storage/OCDBT and isolated leaves PASS; Orbax metadata and target-free full-tree restore time out. Resource plateau indicates native coordination wait. Producing/current drift is TensorStore 0.1.78/0.1.51; next action is isolated producing-runtime reproduction. |
| Object producing-runtime checkpoint restore | VALIDATED | Isolated exact runtime reproduction and original-serl integrity PASS. TensorStore 0.1.78 still stalls in Orbax metadata/no-target/matching-target paths; no inference. Diagnosis complete, restore gate remains BLOCKED. |
| Peg Insertion algorithm/checkpoint validation | VALIDATED | Fresh-training mapping and checkpoint roundtrip/inference pass; legacy checkpoint_9000 timeout is retained as nonblocking. |
| Demo schema validation for available demos | IN_PROGRESS | Peg 20 NPZ/5,512 RLPD/2,388 BC schema and mapping VALIDATED; Object success provenance remains UNVERIFIED. |
| Define common domain model | TODO | Use audited differences; preserve process-specific payloads. |
| Define Mock Skill framework | TODO | Must not imply real-skill validation. |
| Skill API draft | TODO | Adapter-based; do not expose SERL agent internals to scheduler. |
| Common Evaluation skeleton | TODO | Include explicit success semantics and required single-skill metrics. |
| Cable real-agent checkpoint roundtrip | VALIDATED | Same DrQ architecture: legacy save/rebuild/restore; 598 key paths/shapes/dtypes and all leaf hashes match; restored inference PASS. |
| Cable GPU handoff documentation/package | DONE | Manifest, environment/config specs, portable bounded/formal/eval scripts, operator README, checkpoint policy, return layout, and verifier are prepared. No large Demo was copied. |
| Cable evaluation entry | DONE | Independent CLI and raw/summary schemas implemented; reset plus one-step dry run PASS. Trained-checkpoint runtime remains UNVERIFIED. |
| Cable portable path audit | VALIDATED | Prepared handoff scripts derive PROJECT_ROOT and config/data/output references are relative; no runtime dependency on `/home/ubuntu`. |
| Cable formal training resume support | VALIDATED | Step-123 cross-process test restored real agent, online replay/RNG/actor progress, rebuilt 10,526 Demo replay, reconstructed W&B must-resume, saved step 125, preserved step 123, and rejected collision. |
| Cable GPU training handoff readiness | VALIDATED | `CABLE_GPU_TRAINING_HANDOFF_READY = READY`; target GPU smoke remains mandatory before formal training. |
| Object Demo schema audit | VALIDATED | Trusted frozen pickle: 2,000 finite six-field transitions; images/state/action shapes and bounds validated without mutation. |
| Object Demo boundary/success provenance | BLOCKED | Success provenance remains UNVERIFIED. Project decision accepts the original unsplit, unmodified 2,000 transitions for baseline use; acceptance is not success proof and no rerecording is planned. |
| Object replay compatibility and algorithm identity | VALIDATED | All 2,000 transitions inserted; horizon-one replay size 2,001; packed sample PASS. Classification `DRQ_PLUS_DEMO_REPLAY`. |
| Object GPU training handoff preparation | VALIDATED | All preparation gates including full safe resume are validated; `OBJECT_GPU_TRAINING_HANDOFF_READY = READY`. GPU smoke remains a target-host acceptance step. |
| Object fresh checkpoint save/restore | VALIDATED | Project legacy-Flax fresh 598-leaf save, destroy/rebuild, restore, structure comparison and one deterministic `(4,)` inference PASS. |
| Object safe training resume | VALIDATED | Cross-process step 123→125 restored model, replay/index/RNG, learner/actor steps, Demo rebuild and strict W&B identity; collision and six fail-closed tests plus inference PASS. |
| Peg NPZ/PKL Demo audit/schema/semantics | VALIDATED | 20/20 source-semantic successes; RLPD PKL exact 20-episode flatten; BC PKL exact nine-episode subset. |
| Peg algorithm/checkpoint/reward mapping | VALIDATED | RLPD primary, SAC/BC auxiliary, reward discrepancy NO, and BC/RLPD checkpoint ownership validated. |
| Peg replay compatibility repair | VALIDATED | Scoped project adapter preserves original datastore; 5,512 inserts and 32+32 sample PASS. |
| Peg fresh checkpoint compatibility | VALIDATED | Fresh 198-leaf SAC state legacy-Flax save/rebuild/restore/value equality and one `(7,)` inference PASS. |
| Peg legacy checkpoint restore | DONE | `UNRESOLVED_TIMEOUT_NONBLOCKING`; retained artifact, not loaded or required for fresh baseline. |
| Peg portable runtime | VALIDATED | Dynamic installed-asset discovery, project symlink/XML rewrite, synthetic-root resolution and lifecycle PASS. |
| Peg GPU training handoff preparation | VALIDATED | Launcher/config/manifest/spec/adapters/checkpoint+resume/verifier/scripts/evaluation/operator docs prepared; `PEG_GPU_TRAINING_HANDOFF_READY = READY`. |
| Peg safe training resume | VALIDATED | Cross-process 123→125 roundtrip, replay/RNG/Demo/W&B restoration, inference and six fail-closed cases PASS. |
| Three-model parallel GPU handoff package freeze | VALIDATED | Operators A/B/C are independently assigned to Cable/Object/Peg; self-contained README 0–13, transfer/task sheets, environment/W&B/data/return contracts, CPU verifiers, shell syntax, secret scan and independence audit pass. `THREE_MODEL_PARALLEL_GPU_HANDOFF_READY = READY`; remote GPU gates remain unrun. |
| Cable remote GPU gates/training/evaluation return | TODO | Operator A may proceed independently after Cable GPU verifier, env smoke, bounded smoke, checkpoint restore and online W&B PASS. |
| Object remote GPU gates/training/evaluation return | TODO | Operator B may proceed independently; fresh training only and never load legacy checkpoint 4500. |
| Peg remote GPU gates/training/evaluation return | BLOCKED | Platform, replay, checkpoint/resume and diagnostic-workaround bounded smoke pass, but `PEG_FORMAL_CANDIDATE_GPU_GATE = FAIL`. Restore placement causality was tested once: D1 retained 198 host NumPy leaves and D2 explicitly placed all 198 leaves on GPU; both completed 5/5 original consecutive updates with identical RNG/loss sequences and no timeout. `PEG_RESTORE_PLACEMENT_CAUSAL_DIAGNOSTIC = NO_EFFECT`; host restore placement is not a sufficient cause, while the intermittent stall remains unresolved because neither arm reproduced it. No `runtime_gpu_formal.env` exists; formal/evaluation must not start. Never load legacy checkpoints. |
| Peg runtime-only continuous update reliability | VALIDATED | R1 fixed and R2 normally resampled 32/32 batches each completed 20/20 fresh original updates with finite losses and normal RNG. Both showed one index-1 recompile (WARNING ~4.3 s), then 18 NORMAL updates and no timeout. Classification RECOMPILE; no W&B/checkpoint/resume/retry. This does not release the BLOCKED formal candidate integration Gate. |
| GPU returned-artifact intake validator | TODO | Check checkpoints, logs, W&B identity/export, environment record, evaluation metrics, hashes, commands, and per-model report. |

## Completed Audit Record

| Task | Status | Notes |
|---|---|---|
| Initialize project state documents | DONE | Initial documents remain present. |
| Package intake inventory and safe extraction | VALIDATED | Archives identified, hashed, checked, and extracted without changing originals. |
| Three package static audits | DONE | Presence, interfaces, demos, checkpoints, and risks documented. |
| Three-process compatibility audit | DONE | Static comparison and provisional integration plan completed; later smoke status is tracked separately above. |
| Cable Routing V2 intake and V1/V2 comparison | VALIDATED | Archive identified, hashed, safety-checked, isolated, and compared; all non-self manifest entries validate. |

## P2 — Important

| Task | Status | Notes |
|---|---|---|
| Robot Manager | TODO | Logical robot state and lifecycle. |
| Order Manager | TODO | Dynamic orders and task dependencies. |
| Resource lock / workspace reservation skeleton | TODO | Logical coordination mechanism. |
| Baseline scheduler interface | TODO | Fixed Assignment, FCFS, Greedy/Earliest Finish Time. |
| KPI mathematical definitions | TODO | Define before benchmarking. |
| Logical production-line simulator | TODO | Use Mock Skills until real integrations are validated. |
| Reproducible scenario definitions | TODO | Six required scenarios. |
| Benchmark result schema | TODO | No fabricated data. |
| Submission validator | TODO | Must not report READY FOR SUBMISSION while evidence is missing. |

## P3 — Optional / Later Phase

| Task | Status | Notes |
|---|---|---|
| Proposed autonomous decision method | TODO | After baselines and preceding milestones. |
| Ablation and stress-test framework | TODO | Requires proposed method and validated scenarios. |
| Multi-robot visualization and demo | TODO | Must not block logical scheduling/evaluation. |
| Technical report | TODO | Claims only from recorded results. |
| Final submission package | TODO | Requires validator evidence. |
## 2026-09-05 — Peg F3 lateral -Y pilot

| Task | Status | Notes |
|---|---|---|
| Freeze lateral -Y command plan | VALIDATED | 176 commands; only `action[1]` for commands 107-112 mirrors the accepted +Y challenge/correction; dedicated verifier PASS. |
| Record lateral -Y V2 pilot | TODO | `LATERAL_MINUS_RERECORD=READY`; requires separate authorized one-shot recording. |
## 2026-09-05 — Peg F3 V2 lateral -Y pilot completion

| Task | Status | Notes |
|---|---|---|
| Record and audit lateral -Y V2 pilot | VALIDATED | 176 transitions; challenge/correction/near-hole/success/contact/axial tracking Gates PASS. |
| Record rotation-roll V2 pilot | TODO | `ROTATION_ROLL_READY`; requires separately frozen and validated plan plus explicit authorization. |
## 2026-09-05 — Peg F3 rotation-roll pilot

| Task | Status | Notes |
|---|---|---|
| Freeze rotation-roll command plan | VALIDATED | 179 commands; planned peak 0.06981317 rad; five active correction transitions; dedicated verifier PASS. |
| Record rotation-roll V2 pilot | TODO | `ROTATION_ROLL_RERECORD=READY`; requires separate one-shot authorization and runtime challenge verification. |
## 2026-09-05 — Peg F3 V2 rotation-roll pilot completion

| Task | Status | Notes |
|---|---|---|
| Record and audit rotation-roll V2 pilot | VALIDATED | 179 transitions; runtime challenge/correction/near-hole/success/contact/axial tracking Gates PASS. |
| Record rotation-pitch V2 pilot | TODO | `ROTATION_PITCH_READY`; requires a separately frozen and validated plan and explicit authorization. |
## 2026-09-05 — Peg F3 rotation-pitch pilot

| Task | Status | Notes |
|---|---|---|
| Freeze rotation-pitch command plan | VALIDATED | 179 commands; planned peak 0.06981317 rad; five active correction transitions; standard/roll bindings and dedicated verifier PASS. |
| Record rotation-pitch V2 pilot | TODO | `ROTATION_PITCH_RERECORD=READY`; requires separate one-shot authorization and runtime challenge verification. |
## 2026-09-05 — Peg F3 V2 rotation-pitch pilot result

| Task | Status | Notes |
|---|---|---|
| Record and audit rotation-pitch V2 pilot | BLOCKED | Recorder completed 179 transitions and all non-peak Gates passed, but realized orientation peak was 0.05148208 rad versus required 0.06-0.08 rad. No retry. |
| Audit all five V2 pilots | BLOCKED | Pitch pilot has not passed its frozen challenge-peak Gate. |
## 2026-09-06 — Peg F3 pitch calibration and revised plan

| Task | Status | Notes |
|---|---|---|
| Calibrate pitch challenge from `_001` telemetry | VALIDATED | Four response steps finite/monotonic; no abnormal lateral/contact/tracking state; response ratio 0.73742645. |
| Freeze revised pitch plan | VALIDATED | 182 commands; five challenge and seven exactly cancelling correction steps; expected peak 0.06435260 rad; dedicated verifier PASS. |
| Rerecord rotation-pitch V2 pilot | TODO | `ROTATION_PITCH_RERECORD=READY`; requires separate one-shot authorization. |
| Audit all five V2 pilots | BLOCKED | Remains blocked until the revised pitch pilot passes runtime acceptance. |

## 2026-09-06 — Peg F3 actor shutdown fix

| Task | Status | Notes |
|---|---|---|
| Bound Actor/TrainerClient shutdown joins | VALIDATED | Explicit stop/close ordering, idempotent stop, bounded join and durable fail-closed thread evidence; CPU shutdown regression PASS. |
| Short shutdown acceptance | TODO | `READY_FOR_SHORT_SHUTDOWN_ACCEPTANCE`; do not rerun the completed 1000-update smoke. |

## 2026-09-06 — Peg F3 short shutdown acceptance result

| Task | Status | Notes |
|---|---|---|
| Real Actor/Learner short shutdown acceptance | BLOCKED | 20 updates and real communication passed; client stop/broadcast join passed, but Actor process remained alive beyond the 30-second natural-exit Gate. No retry. |
| Post-cleanup Actor process-exit investigation | TODO | Inspect runtime threads/process-exit boundary after durable `actor_cleanup_complete.json`; training remains blocked. |

## 2026-09-06 — Peg F3 post-cleanup exit investigation result

| Task | Status | Notes |
|---|---|---|
| Identify exact post-cleanup blocking object | BLOCKED | Single zero-update Actor reproduction exited its workload via uncaught TrainerClient connection error before cleanup instrumentation, but the Python process still remained alive to 180 s. Exact thread/native object was not captured; no retry. |

## 2026-09-06 — Peg F3 interpreter-exit dual-path result

| Task | Status | Notes |
|---|---|---|
| Dual-path interpreter-exit sampling | VALIDATED | Exception path completed finally/atexit with no Python-level blocker; native ZMQ threads persisted. Normal path stopped between cleanup-complete and outer finally. |
| Explicit request-side ZeroMQ cleanup fix | TODO | `INTERPRETER_EXIT_FIX_READY`; close `ReqRepClient` socket/context with bounded/fail-closed ordering on normal and partial-constructor paths. |

## 2026-09-06 — Peg F3 request-side ZeroMQ fix result

| Task | Status | Notes |
|---|---|---|
| Explicit request-side ZeroMQ cleanup | BLOCKED | Implementation added, but the first no-server constructor-exit regression exceeded 10 seconds. Remaining regression cases were NOT_RUN under fail-stop. |
| Short shutdown reacceptance | BLOCKED | Requires a separately authorized investigation of whether the child timed out before cleanup or remained blocked after it; do not retry automatically. |

## 2026-09-06 — Peg F3 ReqRep timeout phase localization

| Task | Status | Notes |
|---|---|---|
| Localize no-server 10-second timeout phase | VALIDATED | One reproduction only. T+10 last durable marker was `M10_CONTEXT_CLOSE_BEGIN`; faulthandler placed the main thread in `zmq.Context.destroy/term`. Classification `CLEANUP_TIMEOUT`; fix review is ready. |
| Review request-side context finalization fix | TODO | `REQREP_FIX_REVIEW_READY`; do not rerun normal path or training without separate authorization. |

## 2026-09-06 — Peg F3 ReqRep teardown review and minimal fix

| Task | Status | Notes |
|---|---|---|
| Resolve ReqRep context ownership and exception-path teardown | VALIDATED | Private `zmq.Context()` ownership confirmed. All reset/final sockets use linger=0; private context uses one explicit `term()`. No-server exception process exited naturally in 7.264547 s. |
| Revalidate normal Actor/Learner shutdown | TODO | `NORMAL_PATH_REVALIDATION=READY`; requires separate authorization. No normal-path run occurred in this task. |

## 2026-09-06 — Peg F3 ReqRep normal-path revalidation attempt

| Task | Status | Notes |
|---|---|---|
| Revalidate normal Actor/Learner shutdown | BLOCKED | Pre-workload WSL invocation lacked project-root binding and failed importing `scripts`. No learner/actor/output was started; runtime Gates are NOT_RUN. A separate invocation fix/preflight is required before another authorization. |

## 2026-09-06 — Peg F3 V2 autonomous pre-5K closure

| Task | Status | Notes |
|---|---|---|
| Canonical WSL launch binding | VALIDATED | Absolute project/Python/runtime/PYTHONPATH/cwd bindings; import and pre-workload validation PASS. |
| ReqRep normal-path shutdown | VALIDATED | 20/20 updates; real communication/replay growth; Actor natural exit 1.659183331 s; no target residual. |
| Final 1000-update smoke adjudication | VALIDATED | Compute, data, replay, communication, checkpoint, restore and shutdown chains PASS; no rerun. |
| Formal V2 5K preflight | VALIDATED | Frozen config SHA256 `eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de`; guarded exact command and isolated destination PASS. |
| Start formal 5K | READY | Explicit user authorization required; `TRAINING_STARTED=false`. |

## 2026-09-06 — Peg F3 V2 formal Stage-1 completion

| Task | Status | Notes |
|---|---|---|
| Fresh formal 0→5000 | VALIDATED | 5000/5000, GPU, V2-only dataset, 32:32, UTD1, finite losses and communication PASS. |
| Checkpoints 1000–5000 | VALIDATED | Five complete bundles; exact metadata/hash bindings; each independently restored with 198 finite numeric leaves. |
| Normal shutdown | VALIDATED | Actor/Learner natural exit, ReqRep cleanup and env close PASS; no residual/ZMQ threads. |
| Checkpoint screening | READY | Separate authorization required. Formal 10K continuation is NOT_AUTHORIZED. |

## 2026-09-07 — Peg F3 V2 recovery and curriculum closure

| Task | Status | Notes |
|---|---|---|
| Recovery basin audit | BLOCKED | Frozen expert control succeeds 3/3 from canonical reset, but 0/3 after policy steps 1/2/5/10/15/25/50. No corrective dataset is scientifically admissible from the tested takeover rule. |
| Training-only curriculum feasibility | BLOCKED | C0/C1/C2 restored integration states reproduce the next transition and complete insertion; C3 full reset succeeds. Immediate restored physical observation differs by 4.14e-5 to 4.02e-4, so exact reset-observation Gate is not met after the maximum two stage fixes. |
| Corrective Demo collection and BC warm-start | NOT_RUN | Recovery Gate failed; canonical production Demo remains immutable. |
| Staged RLPD 1K/3K and formal 5K | NOT_RUN | Blocked before training by exact curriculum reset-observation verification. |
| Review exact training-state restoration | TODO | Determine which derived MuJoCo/controller state produces the pre-action pose discrepancy without changing eval, goal, success, controller, action semantics, or the frozen admission threshold. |

## 2026-09-07 — Peg F3 V2 exact-state restore closure

| Task | Status | Notes |
|---|---|---|
| Localize immediate restored observation mismatch | VALIDATED | Exact first cause is `OBSERVATION_TIMING`: pre-extra-forward step observation was compared with post-forward reconstruction. Per-dimension attribution is recorded for C0/C1/C2. |
| Full MuJoCo/Python state round trip | VALIDATED | Full `mjData` control PASS; official 139-value `mjSTATE_INTEGRATION` plus env/controller/wrapper/task state restores with integration and observation max delta 0.0. |
| Single-step and expert-suffix reproduction | VALIDATED | C0/C1/C2 observation/qpos/qvel/reward maxima are all 0.0; wrench exact; every suffix reaches canonical success. |
| Training-only curriculum wrapper | VALIDATED | New isolated wrapper uses exact snapshots for C0/C1/C2 and delegates C3 to canonical reset; goal/reward/success/controller/action semantics unchanged. |
| Curriculum static preflight | VALIDATED | All C0–C3 Gates PASS; train factory is isolated; formal eval entries do not import curriculum. `CURRICULUM_TRAINING_READY=YES`. |
| Start BC/RLPD curriculum training | NOT_RUN | Explicitly stopped before training. A separate scientific task must authorize any training. |

## 2026-09-08 — Peg F3 V2 exact-restore curriculum RLPD

| Task | Status | Notes |
|---|---|---|
| Curriculum package and snapshot-pool audit | VALIDATED | C0/C1/C2 each contain one unique exact-restorable successful source snapshot; expert suffix, canonical goal, V2 and controller Gates pass. Fixed R1/R2 schedule SHA `e536e469...bfcf`. |
| Deterministic staged admission runner | VALIDATED | New single-process collect/train blocks; C+G only, 32:32, batch64, UTD1; frozen `D_DEMO=7.41773975814`; no free-running async. |
| C0 replica 1 bounded 1K | BLOCKED | 1000 updates and checkpoints pass; replay 604 and geometry improve, but success is 0/3 and final policy produces repeated `table ↔ end_effector` collision. C0 Gate FAIL. |
| C0 replica 2 | NOT_RUN | Fail-stop: R1 already prevents the required both-replicas C0 PASS. |
| C1/C2 promotion | NOT_RUN | C0 Gate failed; no promotion authorized. |
| C3 full-reset training and canonical confirmation | NOT_RUN | `TRAIN_FULL_RESET_READY=NO`. |
| Fresh 3K / formal 5K | NOT_RUN | Upstream C0 failure; no preflight or run. |
| Peg single-task convergence | BLOCKED | `PEG_SINGLE_TASK_CONVERGED=NO`; `SUCCESSFUL_INSERTION=NO`; first bad level C0. |

## 2026-09-08 — Peg F3 V2 C0 near-hole insertion causal audit

| Task | Status | Notes |
|---|---|---|
| Time-aligned C0 audit | VALIDATED | Three deterministic episodes identical; best simultaneous margin 2.94584 at step 50; no success-predicate mismatch. |
| Frozen expert comparison | VALIDATED | Same exact C0 snapshot; expert suffix succeeds without table–EE contact; policy diverges at step 1. |
| Exact one-step and dimension branches | VALIDATED | Strict Obs20 equality; immediate force damage; no single action dimension repairs the conflicting force/margin effects. |
| Pre-contact and collision identity | VALIDATED | By step 179 the state is already far outside Demo support; table geom 2 contacts end-effector geom 74 at step 194. Geometry root cause NO. |
| Local critic ranking | VALIDATED | All critic heads rank policy below expert/Demo at all three critical states; critic bad-action preference NO. |
| Root-cause decision | VALIDATED | `LOCAL_ACTOR_MULTI_AXIS_CONTROL_FAILURE`; table contact is downstream. |
| Implement next fix | NOT_RUN | Recommended only: `LOCAL_INSERTION_POLICY_SUPERVISION_REVIEW`. No training or parameter change authorized in this audit. |

## 2026-09-08 — Peg F3 V2 local insertion actor supervision

| Task | Status | Notes |
|---|---|---|
| Build immutable local supervision subset | VALIDATED | 11 successful accepted V2 episodes, 330 real near-hole-to-success transitions, 9/2 episode-group split; production Demo unchanged. |
| Bounded actor-only BC | VALIDATED | One 1000-update run; critic/target/temperature exact-frozen; offline validation action error materially improved; a6 excluded from the six-axis objective. |
| C0 closed-loop repair gate | BLOCKED | Eligible checkpoints 250/500/1000 all achieved 0/3 insertion. Best behavior point 250 retained near-hole 3/3 and contact health but margin 4.684031 did not beat historical 2.945840 and first divergence remained 3. |
| Canonical safety eval and optimizer handoff | NOT_RUN | C0 insertion gate failed. |
| C0 +500 RLPD retention | NOT_RUN | C0 insertion gate failed; no second actor regularizer added. |
| C1/C2/C3 and formal evaluation | NOT_RUN | Fail-stop at C0; curriculum success was not reported as canonical success. |
| Review actor parameterization / local closed-loop distribution shift | TODO | Separate scientific task only; do not revisit critic penalties, per-axis penalties, collision exclusions, Demo mutation, or admission threshold. |

## 2026-09-08 — Peg F3 V2 actor parameterization and closed-loop shift audit

| Task | Status | Notes |
|---|---|---|
| Verify actor inference binding | VALIDATED | BC, validation, C0 deterministic eval and normal Actor runtime use the same tanh-distribution mode path; dynamic outputs are exact-equal. |
| Verify Obs20 pipeline identity | BLOCKED | Shape/dtype/semantics/order/wrapper match, but BC step-return observation differs from the forwarded C0 snapshot, max 0.0956383 at wrench index 7. The induced six-axis action change is only 0.0296215, so this is secondary. |
| Local coverage and action compounding | VALIDATED | First local support exit is state step 6; policy/expert L2 grows from 0.211242 at step0 to 0.575852 at step6. In-support one-step GOOD, first off-support one-step POOR. |
| Teacher-forced and checkpoint comparison | VALIDATED | Step1000 teacher-forced MAE 0.0172013/cosine 0.960566; offline fit improves monotonically while closed-loop behavior is NON_MONOTONIC. |
| Distribution and Jacobian audit | VALIDATED | Broad pre-tanh std is present but cannot affect argmax mode. Jacobian condition HIGH at 23.4094; cross-axis coupling SEVERE at 0.898617 energy fraction. |
| Post-BC recovery basin | VALIDATED | Frozen proven C0 suffix recovers policy states at steps 1 and 3 (3/3 each), but not 5/10/20/50. No corrective data collected. |
| Root-cause decision | VALIDATED | `MIXED_ACTOR_FAILURE`; local closed-loop distribution shift plus severe multi-axis conditioning. |
| Actor-local causal review | TODO | Separately authorize a fix-design review only; this audit did not train or implement a new objective. |

## 2026-09-08 — Peg F3 V2 post-BC on-policy corrective supervision

| Task | Status | Notes |
|---|---|---|
| Round 1 corrective collection | VALIDATED | 6 canonical-success expert recoveries from actual post-BC policy states at steps 1/3; 156 transitions in a separate immutable dataset; actor-runtime Obs20 pairing PASS. |
| Round 1 actor-only corrective BC | VALIDATED | Fixed 32 base + 32 corrective, maximum 500 updates; critic/target/temperature frozen. Best checkpoint 500 reached C0 success 3/3 with healthy contact. |
| Corrective Round 2 | NOT_RUN | Required skip because Round 1 met the strong C0 gate; no second dataset or second BC job was created. |
| Canonical safety check | VALIDATED | Original canonical reset/goal, no curriculum/expert/training; no catastrophic approach, force, or contact regression. Canonical insertion remained 0/3 and was not claimed. |
| Repaired C0 handoff | VALIDATED | New full checkpoint/replay bundle; repaired actor plus validated critic/target/temperature; actor optimizer reinitialized with unchanged optimizer/hyperparameters. |
| Unchanged C+G +500 retention | BLOCKED | Exactly +500, 32:32, batch64, UTD1, fixed threshold, no corrective BC. C0 success 3/3→0/3; near-hole/contact survived but insertion and local support did not. |
| C1/C2/C3 and formal evaluation | NOT_RUN | Mandatory fail-stop at C0 retention; no promotion, confirmation, 3K, or 5K run. |
| Policy distribution / RLPD actor-retention review | TODO | Separate authorization required. Do not add an actor regularizer, tune entropy/std, change architecture, relax threshold, or mutate Demo in this task. |
| Peg single-task convergence | BLOCKED | `PEG_SINGLE_TASK_CONVERGED=NO`; root cause `RLPD_DESTROYS_REPAIRED_LOCAL_ACTOR_SKILL`. |

## 2026-09-08 — Peg F3 V2 RLPD actor-retention causal review

| Task | Status | Notes |
|---|---|---|
| Optimizer handoff forensics | VALIDATED | Retention actor Adam was fresh and exactly matched a fresh counterfactual: counters/m/v zero; no optimizer-state shock. |
| Microscopic unchanged-RLPD replay | VALIDATED | C0 3/3 through update2; first measured loss at update5 (0/3); checkpoints 0/1/2/5/10/25/50/100 preserved. |
| Actor-loss decomposition | VALIDATED | Pre-loss gradient norms Q 3.69265, entropy 0.394639, other 0.085508; Q moved local actions away, entropy/other toward expert. |
| Critic local action-gradient geometry | VALIDATED | Median ensemble cosine `-0.312306` and directional scan confirm Q decreases toward expert action. |
| Actor-update coverage/interference | VALIDATED | Near-hole 48.5%, local/corrective support 78.94%; simple absence and one-step cross-region tradeoff are not primary. |
| LOCAL_RETENTION_ANCHOR_V1 | VALIDATED | Virtual-only references; 486 inputs deduplicated to 202 pairs; production Demo excluded; SHA `ae90fb88...daf1c`. |
| One-shot lambda calibration | VALIDATED | Lambda `9.408421811724581`; retention/RL gradient ratio exactly 0.20; no clamp, sweep, or success tuning. |
| Retention fix +100 | BLOCKED | `0/3` at +10 and +100 despite near-hole/contact health; final margin 4.82084 and support exit step6. Mandatory stop. |
| +250/+500 and matched replica | NOT_RUN | +100 primary Gate failed; no second candidate or lambda adjustment allowed. |
| C1/C2/C3 and formal evaluation | NOT_RUN | Upstream retention-fix Gate failed. |
| Next retention mechanism | BLOCKED | Requires separate scientific authorization; current task permits no further repair. |

## 2026-09-08 — Peg F3 V2 repaired-Actor critic-first handoff

| Task | Status | Notes |
|---|---|---|
| CRITIC_ALIGNMENT_PROBE_V1 | VALIDATED | 230 deduplicated diagnostic probes; existing data plus repaired successful rollout only; SHA `b1514a2d...d5944`. |
| Baseline critic geometry | VALIDATED | Median cosine -0.327955, anti-aligned 0.70, expert-not-worse 0.30. |
| Actor-frozen critic catch-up | VALIDATED | Original 32:32 replay passed at earliest update25: median +0.059038, anti-aligned 0.30, expert-not-worse 0.704348. |
| Corrective expert critic coverage | NOT_RUN | Original replay was sufficient; fallback was forbidden after primary PASS. |
| Frozen Actor sanity | VALIDATED | Actor/optimizer/temperature frozen; C0 behavior preserved at 3/3. |
| Ordinary Actor re-enable +100 | BLOCKED | Non-monotonic 3/3→0/3→3/3→0/3; final 0/3 with healthy contact. Geometry oscillated and insertion was not retained. |
| Actor re-enable +500 / matched confirmation | NOT_RUN | +100 Gate failed; mandatory stop. |
| C1/C2/C3 and formal evaluation | NOT_RUN | Upstream actor-reenable Gate failed. |
| Actor–critic closed-loop geometry review | TODO | Separate scientific authorization required; do not add critic penalty, tune retention/entropy/std, or alter architecture in this task. |

## 2026-09-08 — Peg F3 V2 Actor–Critic closed-loop geometry causal review

| Task | Status | Notes |
|---|---|---|
| COMMON_START reconstruction | VALIDATED | Repaired Actor plus deterministic critic-only update25; prior probe geometry reproduced and C0 remained 3/3. |
| C0_BEHAVIOR_CRITICAL_PROBE_V1 | VALIDATED | Three successful trajectories, all deduplicated local states plus critical tags; SHA `81b03c51...a43c1`. |
| Actor-only matched branch | BLOCKED | Fixed critic; C0 3/3 at update5, 0/3 at update10/25. Exact one-step damage increased before failure. |
| Critic-only matched branch | VALIDATED | Fixed Actor; C0 3/3 at update5/10/25 and zero action/one-step drift, even though online cosine median was negative at update25. |
| Joint matched branch | BLOCKED | C0 3/3→0/3 at update10, transiently 3/3 at update25; behavior does not monotonically follow cosine. |
| JVP/component attribution | VALIDATED | update9→10 JVP error `0.000108919`; normalized Q direction has the largest critical action effect. |
| Scientific repair | NOT_RUN | Diagnostic-only mandate; no hyperparameter, loss, data, architecture or environment change authorized. |
| Next review | TODO | `ACTOR_UPDATE_TIMESCALE_OR_TRUST_REGION_REVIEW`, under a separate authorization and without treating cosine as a success Gate. |

## 2026-09-08 — Peg F3 V2 Actor update timescale repair V1

| Task | Status | Notes |
|---|---|---|
| Period-4 static preflight | VALIDATED | Sole candidate; critic every learner update, Actor/temperature at `4,8,...`; index SHA `506f4d82...0a07b`. |
| Effective update semantics | VALIDATED | New helper isolates selected optimizers while reusing unchanged losses; full-tree target Polyak remains every critic step; production SAC unchanged. |
| COMMON_START | VALIDATED | Exact Actor/critic/target/replay hashes; canonical C0 3/3 and contact PASS. |
| Same Actor-count comparison | VALIDATED | Ordinary Actor10 0/3 versus period4 learner40/Actor10 3/3; timescale separation effective. |
| Period4 +100 | VALIDATED | 3/3 C0, 3/3 near-hole, margin 0.94831, support exit none, healthy contact, no table/EE collision. |
| Period4 +500 retention | BLOCKED | 0/3 by learner250 and at learner500; final support exit step3, margin 4.95559, exact one-step damage 1.10852. |
| Matched R2 | NOT_RUN | +500 R1 Gate failed. |
| C1/C2/C3/formal | NOT_RUN | Upstream retention Gate failed. |
| Alternate Actor periods | NOT_RUN | Forbidden; no period2/8/16 grid. |
| Next action | BLOCKED | `NEXT=STOP`; any trust-region review requires a separate scientific task. |
## 2026-09-08 — Peg F3 V2 behavior-critical Actor trust region V1

| Task | Status | Evidence |
|---|---|---|
| Freeze successful behavior envelope and sentinel | VALIDATED | Successful-only mean/P95/damage bounds frozen at `0.02406635507941246` / `0.02416462143883109` / `0.019941113951320415`; probe SHA `81b03c51...a43c1`, sentinel SHA `fc52c856...31a11`, fixed alpha `1..1/64`. |
| R1 C0 retention through learner500 | VALIDATED | C0 `3/3` at learner100 and learner500; 25/125 candidates accepted, 100 rejected, 24 nonzero accepts; all accepted states inside frozen bounds. |
| Matched R2 C0 retention through learner500 | VALIDATED | Identical seed/data/schedule/threshold reproduces R1 and retains C0 `3/3`; final replay handoff saved separately with 904 transitions. |
| Original canonical full-reset safety | VALIDATED | Reset/goal canonical, curriculum inactive, contact healthy, finite force, no action saturation. Canonical success and near-hole remain `0/3`; not convergence. |
| Exact-state C1 continuation | BLOCKED | 1000 updates completed with C0 trust region active. C0 remains `3/3`, but 250/250 C1 Actor candidates are rejected and C1 behavior has no progress. `FIRST_BAD_CURRICULUM_LEVEL=C1`. |
| C2/C3/3K/formal5K | NOT_RUN | Mandatory scientific stop at the C1 Gate. `PEG_SINGLE_TASK_CONVERGED=NO`. |
## 2026-09-08 — Peg F3 V2 C2 data/policy-capacity review

| Task | Status | Notes |
|---|---|---|
| C2 failure probe | VALIDATED | Seven deterministic corrected-`_002` pre-contact states; SHA `386c1e56...344c4`. |
| Expert relabel and exact-state one-step audit | VALIDATED | C2 reset binds to production Demo episode0/step42; expert correction improves the registered one-step criterion on 2/7 probes. |
| Gate-versus-output authority attribution | VALIDATED | Every selected probe has `g2=1`; Gate authority is sufficient, but required pre-tanh residual is outside `[-1,1]` (reset a1 `-8.435875`). |
| SAC direction and replay coverage audit | VALIDATED | Actor direction MIXED/mostly anti-aligned; useful corridor 59/1684 (`3.50356%`), data coverage MIXED. |
| Existing expert subset | VALIDATED | New physically separate subset extracted read-only from accepted Demo; SHA `733bd017...fcd02`; production Demo unchanged. |
| Offline representation fit / existing-expert BC | NOT_RUN | Forbidden by Decision 2 because required action is not representable. |
| On-policy corrective supervision | NOT_RUN | Not authorized after action-authority failure. |
| C2 acquisition / confirmation / canonical interface audit | BLOCKED | `PRIMARY_ROOT_CAUSE=C2_RESIDUAL_ACTION_AUTHORITY_LIMIT`; next separate task is `C2_RESIDUAL_OUTPUT_PARAMETERIZATION_REVIEW`. |
| Peg single-task convergence | BLOCKED | `PEG_SINGLE_TASK_CONVERGED=NO`; no C3 or formal training started. |
## Peg F3 V2 — C2 output parameterization and acquisition closure (2026-09-08)

| Task | Status | Evidence / decision |
|---|---|---|
| Exact current C2 action-composition audit | VALIDATED | Confirmed nested tanh/pre-tanh composition from effective source; environment action range/scaling unchanged. |
| P1/P2 static representability | VALIDATED | 205/205 controlled states; P1=1.0 but P95/max `delta_z=8.452254/8.453148`; P2=1.0 with 0 failures. |
| Select `GATED_ABSOLUTE_C2_ACTION_HEAD_V1` | VALIDATED | Selected strictly from static evidence before training; no scale sweep or behavioral selection. |
| Zero-init and lower-skill invariance | VALIDATED | C0 and C1-core max action delta `0.0`, including random C2-parameter perturbation. |
| Actual-forward representation rerun | VALIDATED | Reset reconstruction error `0.0`; existing expert representability fraction `1.0`. |
| Existing-data offline head fit | VALIDATED | 1000 selected by validation MAE; MAE `0.00460827`, cosine `0.99996036`, reset error `0.0170668`. |
| Exact short-horizon gate | VALIDATED | GOOD on 2/7 probes under fixed 12-step audit. |
| Initial closed-loop C2 acquisition | DONE | FAIL: `0/3`, shell/core `0`, contact unhealthy; classified closed-loop distribution shift. |
| Single C2 corrective collection | VALIDATED | Exactly one round / 3 episodes / 51 separate rows; SHA `08151c2e...958f5`; all runtime Obs20, C2-authorized, pre-contact, pre-threshold-exit. |
| Single combined corrective head fit and evaluation | VALIDATED | No progress check passed; C2 `0/3`, shell/core `0`, contact FAIL; mandatory STOP. |
| C2 matched confirmation / freeze | BLOCKED | Not authorized because corrective acquisition failed. |
| Canonical interface audit / C3 | BLOCKED | Not authorized because C2 was not confirmed. |
| Peg single-task convergence | BLOCKED | `PEG_SINGLE_TASK_CONVERGED=NO`; `PRIMARY_ROOT_CAUSE=C2_CLOSED_LOOP_DISTRIBUTION_SHIFT_NOT_REPAIRED`; `NEXT=STOP`. |

## 2026-09-08 — Peg F3 V2 C2 recovery-basin review and recoverability-filtered DAgger

| Task | Status | Evidence / decision |
|---|---|---|
| Exact-state C2 recovery-basin audit | VALIDATED | Reset recoverable; basin exists; probe recoverable fraction `0.4375`; first nonrecoverable policy state step 11. |
| Recovery-loss causal classification | VALIDATED | Policy/expert one-step branch identifies `ONE_STEP_BOUNDARY_CROSSING`; post-divergence recoverability `0.0`. |
| Prior 51-row corrective audit | VALIDATED | 33/51 shell-recoverable; pre/post loss fractions `0.647059/0.352941`; gradient influence `165.45x`, classified DOMINANT rather than diluted. |
| Recoverability-filtered DAgger R1 | DONE | 3 episodes, 33 accepted rows, fresh 1000-update head-only fit; C2 shell 3/3 but contact FAIL, success 0/3. |
| Recoverability-filtered DAgger R2 | DONE | 3 episodes, 108 accepted rows, fresh fixed-budget fit; near-hole 3/3 but shell/core 0 and contact FAIL. |
| Recoverability-filtered DAgger R3 | DONE | 3 episodes, 156 accepted rows, fresh fixed-budget fit; C2 0/3, shell/core 0, contact FAIL. |
| Lower-skill regression | VALIDATED | Every round: C0 3/3; C1 regression/contact PASS. |
| C2 matched confirmation / freeze | BLOCKED | Three-round strong Gate never passed; no C2 checkpoint promoted. |
| Canonical interface audit / C3 | NOT_RUN | Forbidden after C2 acquisition failure. |
| Final decision | BLOCKED | `PRIMARY_ROOT_CAUSE=C2_ON_POLICY_CAPACITY_FAILURE`; `NEXT=STOP`; `PEG_SINGLE_TASK_CONVERGED=NO`. |

## 2026-09-08 — Peg F3 V2 C2 policy-class/capacity/observability review

| Task | Status | Evidence / decision |
|---|---|---|
| Immutable policy-class audit dataset | VALIDATED | 495 rows from existing subset + R1/R2/R3, SHA `6e8ba2cf...638d`; episode-separated train/demo-valid/Dagger-valid = 360/36/99. |
| Exact/near-duplicate Obs20 audit | VALIDATED | 495/495 cross-episode exact neighbors; very-close action disagreement mean/P95/max `0/0/0`, every axis `0`. |
| Local conditional variance | VALIDATED | k3 P95 approximately zero; k5 mean/P95 `0.007007/0.0864`; worst 20 lie at C1 shell boundary. |
| Nonparametric Obs20 oracle | VALIDATED | Held-out 1-NN MAE `0.0`; 3-NN MAE `0.00413964`; labels are locally coherent. |
| Current 2x256 H0 fresh fit | VALIDATED | Train/valid/Dagger MAE `0.0577812/0.0727564/0.0945386`; material training underfit remains. |
| H1/H2 diagnostics | VALIDATED | Valid MAE `0.0652751/0.0702320`; DAgger gains 12.03%/7.61%, below fixed history Gate. |
| Sole 2x512 capacity diagnostic | VALIDATED | Train/valid/DAgger MAE `0.0551548/0.0675047/0.0854680`; only 4.55%/9.59% train/Dagger improvement, so capacity Gate fails. |
| Evidence-selected repair | BLOCKED | `C2_SUPERVISED_OPTIMIZATION_UNDERFIT`; no H1 or width repair authorized. |
| Boundary audit / C2 pilot / confirmation | NOT_RUN | Correctly stopped at policy-class root-cause decision before deployment. |
| Canonical/C3/formal continuation | NOT_RUN | C2 was not repaired or confirmed; `PEG_SINGLE_TASK_CONVERGED=NO`. |

## 2026-09-08 — Peg F3 V2 C2 supervised optimization review

| Task | Status | Evidence / decision |
|---|---|---|
| Exact H0 baseline reproduction | VALIDATED | Final train/Dagger MAE exactly `0.0577812/0.0945386`; registered checkpoints 0–1000 recorded. |
| Target saturation/tanh conditioning | VALIDATED | a1 substantially saturated, but full attenuation Gate fails; pathology `MIXED`, not selected. |
| Effective LR/warmup audit | VALIDATED | Warmup 2000 versus budget 1000; LR step1=0, step1000=`1.4985e-4`; integrated LR `0.074925`; 100% below half plateau. |
| Parameter/Adam mechanics | VALIDATED | Moments and counters advance; step1000 parameter delta/parameter norm `2.4987e-4`; fresh Adam is not better. |
| Tiny-subset memorization | VALIDATED | Train16 MAE `0.0137388`; train64 MAE `0.0598290`. |
| Demo-vs-Dagger conflict | VALIDATED | Cosines `[0.9217,0.8994,0.0687,-0.2911,0.2522]`; only 20% strong negative, so no persistent conflict. |
| Fixed-direction line scan | VALIDATED | Alpha2 gives lowest loss; direction is useful and ordinary movement is suboptimal. |
| Select constant plateau LR repair | VALIDATED | Causally selected before repair training; no LR value sweep and no alternate repair. |
| Single repair offline Gate | BLOCKED | Train/valid/Dagger `0.0440912/0.0547767/0.0703185`; misses `0.03/0.06`, a4 DAgger MAE `0.155622`. |
| Boundary/pilot/confirmation/canonical | NOT_RUN | Mandatory STOP at selected repair offline Gate; no second repair. |
| Final decision | BLOCKED | `SUPERVISED_EFFECTIVE_STEP_TOO_SMALL`; repair helped but was insufficient; `NEXT=STOP`, convergence NO. |
## 2026-09-09 — Peg F3 V2 C2 supervised convergence closure

| Task | Status | Notes |
|---|---|---|
| Existing 1K historical boundary replay | VALIDATED | Exact step10 branch: old unrecoverable, 1K repair and expert recoverable. |
| Existing 1K C2 pilot | FAILED | C2 `0/3`, near-hole `0/3`, shell/core `0/0`; C0 `3/3`, C1/contact regression PASS. |
| Fixed 3000-update convergence run | COMPLETED | One fresh constant-plateau `3e-4` trajectory only; step1000 exactly reproduces the prior checkpoint SHA. |
| Offline convergence Gate | PASSED | Selected step3000 by held-out DAgger MAE only: train `0.0249857`, DAgger `0.0390718`, all DAgger axes `<0.15`. |
| Selected-checkpoint causal boundary | FAILED | Expert branch recoverable; selected 3K action branch unrecoverable. First bad Gate is `HISTORICAL_STEP11_BOUNDARY`. |
| C2 pilot / matched confirmation / freeze / canonical / C3 / RLPD | NOT_RUN | Mandatory fail-stop; `C2_SKILL_CONFIRMED=NO`, `PEG_SINGLE_TASK_CONVERGED=NO`. |
| Next action | STOP | Root: `OFFLINE_FIT_IMPROVED_BUT_CAUSAL_BOUNDARY_NOT_REPAIRED`. A new task is required for any further hypothesis. |
## 2026-09-09 — Peg F3 V2 C2 recovery-boundary tail-risk closure

| Task | Status | Notes |
|---|---|---|
| Existing-state multi-boundary set | VALIDATED | 16 exact-restorable, full-expert-shell-recoverable states; SHA `9ea19bc0...0aff4c1`; no new policy data. |
| Existing 1K/1.5K/2K/2.5K/3K audit | VALIDATED | 2K passes shell/core at `0.9375/0.8125` and is the zero-training selection. |
| Existing 2K C2 pilot | FAILED | C2 `0/3`, shell/core `0/0`; C0 `3/3`, C1 regression/contact PASS. |
| Recovery margin sensitivity | VALIDATED | EXTREME/HIGH each `1/16`; anisotropy YES; dominant tail-risk axes a0-a3. |
| Critical/rest gradient audit | VALIDATED | Cosines `[0.9339,0.1134,0.5486,-0.6355,-0.0850]`; pre-registered conflict criterion NO. |
| Evidence-selected repair | VALIDATED | Exactly Repair B: `RECOVERY_MARGIN_NORMALIZED_SUPERVISION_V1`; A/C NOT_RUN. |
| Repair B fixed 3K run | COMPLETED | Same P2/data/loader/Adam/LR; only fixed normalized margin weighting on 13 exact critical rows. |
| Repair B multi-boundary offline Gate | FAILED | Selected 3K shell/core `0.875/0.6875`; violation remains `0.28125`; step11 NO. |
| Short horizon / C2 pilot / matched / freeze / canonical / C3 | NOT_RUN | Fail-stop at offline Gate. |
| Next action | STOP | One-repair limit reached; `C2_SKILL_CONFIRMED=NO`, `PEG_SINGLE_TASK_CONVERGED=NO`. |
## 2026-09-09 — Active autonomous continuation

- [x] Repair/verify WSL DNS and HTTPS; GitHub connectivity is healthy.
- [x] Complete C2 trajectory-level recoverability closure (baseline first bad step `10`).
- [x] Stop failed feed-forward/history/observable-option/Level-3 branches with isolated evidence.
- [x] Confirm Level-4 hierarchical contiguous-plan C2 interface with matched confirmation; freeze C2.
- [x] Run original canonical reset/goal interface audit with no curriculum/expert.
- [ ] `CURRENT_NEXT`: build `C3_UPSTREAM_COMPONENT_PREFLIGHT` from frozen matched C2 evidence.
- [ ] Run exactly one bounded C3 250-step pilot only if static/offline preflight passes.
- [ ] Continue 1K → matched 1K → 3K → formal 5K only through the registered behavior Gates.
- [ ] Final five-episode canonical test requires success >=3/5 and contact PASS before packaging.

## 2026-09-09 — C3 upstream continuation

- [x] Union-gated C3 250 behavior progress and fixed 1K.
- [x] Fresh matched 1K confirmation: checkpoint/evidence exact match; canonical C2-support `3/3`, contact PASS.
- [x] Fixed 3K sustained-progress Gate: FAILED (`C2-support 0/3`); formal 5K NOT_RUN.
- [x] Exact initiation-state expert recoverability and forced frozen-C2 handoff counterfactual.
- [x] Existing-expert recovery-core extension preflight: 411 rows, static lower protection PASS.
- [x] Recovery-core fixed 250 pilot: FAILED with no recovery-core/C1/C0 progress; branch stopped before 1K.
- [ ] `CURRENT_NEXT`: C3 upstream policy-class / on-policy distribution-shift causal review.
- [ ] Do not run formal 5K until a new branch passes 250 -> 1K -> matched 1K -> sustained 3K.

## 2026-09-09 — Peg F3 V2 final closure

- [x] Monotonic Obs20 option primary 3/3 and exact matched confirmation.
- [x] Formal original-canonical evaluation: 5/5, contact PASS.
- [x] Freeze final policy/config/data/evidence/environment manifests.
- [x] Build and verify final reproducibility package: `/home/makoto/flexible_line_project/results/final_packages/peg_f3_v2_monotonic_obs20_option_20260909_001`.
- [x] `PEG_SINGLE_TASK_CONVERGED=YES`; `FINAL_PACKAGE_READY=YES`.

## 2026-09-09 — Final release delivery

- [x] Independent canonical release verification: 5/5, contact PASS.
- [x] Frozen artifact and production Demo hash verification.
- [x] Complete component-based GPU return directory.
- [x] Isolated package-root portability verifier and canonical 1/1 smoke.
- [x] `PEG_STATUS=FINAL_FROZEN`; `PACKAGE_PORTABILITY=PASS`.
- [x] `PROJECT_NEXT=PEG_DELIVERY_ONLY`; do not reopen training.
