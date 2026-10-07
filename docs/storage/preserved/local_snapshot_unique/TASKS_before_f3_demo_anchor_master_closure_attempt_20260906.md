# Project Tasks

Peg F3 minimal actor Demo-action anchor design VALIDATED: isolated all-Demo additive MSE anchor, `lambda_small=0.07`, gradient contribution 9.8207%, baseline optimizer/critic/environment/data semantics preserved. A fresh 1K ablation requires separate user authorization.

Peg F3 Demo-anchor 1K execution is BLOCKED at prelaunch. Frozen config hash and isolated destination passed; raw optimizer-block comparison incorrectly treated explicit baseline values versus inherited production defaults as unequal. Training/Actor/checkpoints/audit are NOT_RUN (`UPDATES=0`). Next is a separately authorized static effective-value Gate-equivalence fix; it must not start the workload.

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
