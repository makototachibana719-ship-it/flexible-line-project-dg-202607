# Project Status

Latest Peg F3 minimal actor Demo-action anchoring design PASS: production SAC applies `-mean(Q-alpha*log_pi)` over the full concatenated 32 Online + 32 Demo batch and has no Demo-specific actor term. Replay rows do not retain phase/source identity, so the isolated candidate anchors all Demo rows using the reliable fixed tail boundary with additive `lambda*MSE(mode(pi(obs_demo)), action_demo)`. A read-only checkpoint_1000 batch measured SAC/anchor gradient norms 1.678847/2.355341; `lambda_small=0.07` yields a 9.8207% weighted gradient ratio, while `lambda_medium=0.14` is reserved. Losses/gradients are finite, initialization/critic/environment/dataset semantics are unchanged, and no parameters were updated. `DEMO_ANCHOR_ABLATION_1K=READY`; training was not started. See `results/validation/peg/f3_v2_demo_anchor_minimal_fix_design_20260906/REPORT.md`.

Latest Peg F3 async RLPD binding fix PASS and 1K Q-bias ablation retry scientific FAIL: the runner now reuses the formal 5K `async_rlpd_peg_sim` binding and verifies the imported file identity from project root, `/home/makoto`, and `/tmp`. One authorized fresh run completed 1000/1000 GPU updates with Demo=1921, 32:32 sampling, online replay 91->8851, finite losses, live communication, checkpoints 500/1000 plus restore, natural shutdown, and no residual process. The critic updated on all 1000 steps. Matched-step Q policy-minus-demo worsened from +1.578540 to +2.438479, while overall/lateral/action[1] Demo alignment also regressed. `ABLATION_RESULT=FAIL`, `FRESH_5K_R2=BLOCKED`, next `MINIMAL_FIX_REDESIGN_REQUIRED`. Protected artifacts were unchanged. See `results/ablation/peg_f3_v2_qbias_ablation_1k_seed0_retry_002/MASTER_REPORT.md`.

Latest Peg F3 runtime Python binding Gate equivalence fix PASS: the ablation runner now validates the expected interpreter's existence, resolved-binary equivalence, exact Conda `sys.prefix`, and resolved binary containment within `serl_peg_gpu`. The expected `python` symlink, resolved `python3.10`, and an alternate same-environment symlink pass; system Python, base Conda, another Conda environment, and Windows Python fail closed. `ABLATION_RETRY=READY`; no learner, actor, or training workload ran. See `results/validation/peg/f3_runtime_python_binding_gate_fix_20260906/REPORT.md`.

Latest F3 ReqRep Normal-Path Revalidation STOPPED before workload: the one authorized supervisor invocation failed before harness import because the WSL command did not bind the project root, producing `ModuleNotFoundError: No module named 'scripts'`. Fail-stop prohibited correcting and rerunning it in the same task. Learner/actor/GPU workload and output directory were NOT_STARTED/NOT_CREATED; all normal-path Gates are NOT_RUN rather than FAIL. No residual workload process exists, and protected checkpoint_1000 hashes remain unchanged. `PEG_F3_REQREP_NORMAL_PATH_REVALIDATION=FAIL`; next is a separately authorized invocation-binding fix/preflight. See `results/validation/peg/f3_reqrep_normal_path_revalidation_invocation_fail_20260906/REPORT.md`.

Latest F3 ReqRep Teardown Review + Minimal Fix PASS: installed Agentlace creates a private `zmq.Context()` per `ReqRepClient`; its timeout recovery closed the superseded REQ socket with default linger, while project cleanup later called `Context.destroy()` and inherited destruction could call `term()` again. The project-owned client now gives every REQ socket linger=0, closes superseded/current sockets with linger=0, and explicitly calls `term()` exactly once on its private context; fallback destruction is idempotent. The sole authorized no-server exception regression propagated the expected connection error and naturally exited in 7.264547 s; context term took about 1.35 ms, the child PID vanished, and no ZMQ Reaper/IO or residual process remained. Fake partial-init/repeated-stop regression PASS. `NORMAL_PATH_REVALIDATION=READY`; it was not run. See `results/validation/peg/f3_reqrep_teardown_review_20260906/REPORT.md`.

Latest F3 ReqRep Timeout Phase Localization PASS: the single authorized no-server constructor reproduction reached imports at 7.584 s, construction at 7.586 s, constructor failure cleanup at 7.688 s, and socket close at 7.695 s. At the supervisor's T+10 s boundary the last durably visible marker was `M10_CONTEXT_CLOSE_BEGIN`; the SIGUSR1 traceback independently placed the main thread in `zmq.Context.destroy()` -> `context.term()` from `_close_req_rep_client`. `/proc` showed 108 tasks, main `poll_schedule_timeout.constprop.0`, plus `ZMQbg/Reaper` and `ZMQbg/IO/0`. Later markers completed only after the timeout boundary. Classification is `CLEANUP_TIMEOUT`, specifically request-side ZMQ context destruction/finalization. `REQREP_FIX_REVIEW_READY`; no normal-path run, training, dataset/checkpoint mutation, or production cleanup-semantic change occurred. See `results/validation/peg/f3_reqrep_timeout_phase_localization_report_20260906/REPORT.md`.

Latest F3 V2 Smoke Shutdown Investigation + Checkpoint 1000 Restore Acceptance PASS: existing evidence proves all 1000 learner updates completed and the actor observed the stop marker, left its step/reset loop, completed its final upload attempt, and durably wrote `actor_result.json`. The first unconfirmed operation is `TrainerClient.stop()`; current Agentlace enters `BroadcastClient.stop()` and an unbounded receive-thread `join()` before `env.close()`. This supports a late network-cleanup classification, while the exact callback/thread state is not recoverable without a timeout stack. A separate fresh process restored the preserved V2 checkpoint_1000 successfully: step/config/dataset/manifest/model/replay hashes, 198 finite numeric leaves, and 9331 replay transitions all PASS with no V1 contamination or training resume. Smoke is adjudicated `PASS_WITH_SHUTDOWN_LIMITATION`; `TRAINING_READY=YES_AFTER_SHUTDOWN_FIX`. No 1000-update rerun or code fix occurred. See `results/validation/peg/f3_v2_smoke_shutdown_restore_20260906/REPORT.md`.

Latest F3 V2 RLPD Smoke 1000 STOPPED/FAIL: the single authorized fresh V2 learner+actor run completed 1000/1000 GPU updates using the canonical 1921-transition Demo buffer and exact 32 Demo + 32 Online batches at UTD=1. Loss/critic/temperature metrics remained finite, online replay grew 91->9361, actor received 1001 networks, and V2 checkpoint bundles 500/1000 were durably saved with matching replay hashes. Fail-stop triggered because the actor wrote its terminal result but did not exit within the 30-second shutdown boundary; the supervisor terminated it and did not start restore verification. Therefore `CHECKPOINT_RESTORE=NOT_RUN`, `TRAINING_READY=NO`; no residual process, retry, formal training, resume or 5k run occurred. See `results/validation/peg/f3_v2_rlpd_smoke_1000_20260906/REPORT.md`.

Latest F3 V2 Dataset Rebuild PASS: the corrected builder exclusively collected the frozen 11 accepted V2 episodes and published 1921 transitions with type counts standard=3, lateral +Y=2, lateral -Y=2, roll=2, pitch=2. V2-only, float32 obs20/action7 schema, finiteness, continuity, sensor reconstruction, provenance 11/11, hash binding, success/non-truncation, contact health, full-load and replay-sample Gates PASS. Dataset SHA-256 is `4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878`. `TRAINING_SMOKE=READY` for a separate task; RLPD smoke, training and resume were not started. See `results/validation/peg/f3_v2_dataset_rebuild_20260906/REPORT.md`.

Latest F3 V2 Dataset Provenance Routing Fix PASS: the builder now uses an explicit `five_pilot`/`formal` source classification. Five-Pilot entries require the frozen manifest path/hash plus exact episode/stream/canonical identity; formal entries require an exact allowlisted PASS report containing their episode path, transition SHA and, for correction episodes, canonical-plan SHA. All 11 positive routes and wrong-source/wrong-manifest/wrong-hash/V1 fail-closed regressions PASS. `DATASET_REBUILD=READY` for a separate task; no dataset was published and no training/resume/RLPD smoke ran. See `results/validation/peg/f3_v2_provenance_routing_fix_20260906/REPORT.md`.

Latest F3 V2 Dataset Build STOPPED/FAIL before publication: the new allowlisted builder validated the original accepted standard pilot, then incorrectly applied that Five-Pilot manifest identity check to the first formal standard episode and raised `standard frozen-stream provenance`. The formal episode itself was not rejected; this is a builder provenance-routing defect. Fail-stop prevented retry and publication: dataset path does not exist, loader/replay smoke are NOT_RUN, and all 11 episodes remain untouched. `TRAINING_SMOKE=BLOCKED` pending a separately authorized minimal builder fix and new build attempt. See `results/validation/peg/f3_v2_dataset_build_20260906/REPORT.md`.

Latest F3 V2 Formal Rotation Pitch +1 PASS: exactly one new `_003` episode completed with 182 transitions using the frozen calibrated plan. Canonical SHA/count, V2 contract, continuity, raw sensor reconstruction, current bindings, active pitch challenge/correction, 0.065045005 rad protocol-range peak, 29-row near-hole coverage, success/non-truncation, zero contact and 1.071806 axial tracking all PASS. The formal collection now contains 11 accepted demos: standard=3, lateral +Y=2, lateral -Y=2, roll=2, pitch=2. `V2_DATASET_BUILD=READY` for a separate task; dataset build, training and resume were not started. See `results/validation/peg/f3_v2_formal_rotation_pitch_20260906_003/REPORT.md`.

Latest F3 V2 Formal Rotation Roll +1 PASS: exactly one new `_002` episode completed with 179 transitions. The frozen canonical plan SHA/count, V2 contract, continuity, raw sensor reconstruction, current bindings, rotation challenge/correction, 0.074964087 rad protocol-range peak, 29-row near-hole coverage, success/non-truncation, zero contact and 0.995735 axial tracking all PASS. Accepted demos now total 10, including rotation roll=2. `ROTATION_PITCH_READY` requires separate authorization; no pitch, training, resume, or dataset build ran. See `results/validation/peg/f3_v2_formal_rotation_roll_20260906_002/REPORT.md`.

Latest F3 V2 Formal Lateral -Y +1 PASS: exactly one new `_002` episode completed with 176 transitions. Canonical verifier/symmetry and preflight passed; every V2 transition passed contract, continuity, sensor reconstruction, finiteness, current hash, reward and success validation. Frozen actions and accepted terminal-success phase semantics match, while challenge/correction, 29-row near-hole, success/non-truncation, zero contact and 0.997420 axial tracking PASS. Accepted demos now total 9, including lateral -Y=2. `ROTATION_ROLL_READY` requires separate authorization; no roll, pitch, training, resume, or dataset build ran. See `results/validation/peg/f3_v2_formal_lateral_minus_20260906_002/REPORT.md`.

Latest F3 V2 Formal Lateral +Y +1 PASS: exactly one new `_004` episode completed with 176 transitions through the fixed formal launcher. Every V2 transition passed contract, continuity, sensor reconstruction, finiteness, current hash, reward and success validation; all actions and the first 175 phases match the frozen canonical plan, while the terminal phase changes from planned `insertion` to recorder-defined `success` exactly as in the accepted +Y pilot. Challenge/correction, 29-row near-hole, success/non-truncation, zero contact and 0.996831 axial tracking PASS. Accepted demos now total 8, including lateral +Y=2. `LATERAL_MINUS_READY` requires separate authorization; no other recording, training, resume, or dataset build ran. See `results/validation/peg/f3_v2_formal_lateral_plus_20260906_004/REPORT.md`.

Latest F3 Formal Lateral +Y Invocation Shell-Boundary Fix PASS: a dedicated Python launcher now invokes the existing canonical verifier and recorder wrapper with the exact WSL project interpreter and `subprocess.run(input=...)`; no PowerShell/Bash `$PY` variable crosses the boundary. The dry-run verified the canonical 176-command plan and SHA-256, seed0/max_steps500/type, fresh episode/contact outputs, and stopped before simulator startup. `LATERAL_PLUS_FORMAL_RERECORD=READY` for separate authorization; no recording, collection retry, training, resume, dataset build, canonical plan, recorder, or production semantic change occurred. See `results/validation/peg/f3_lateral_plus_formal_invocation_fix_20260906/REPORT.md`.

Latest F3 V2 Formal Standard Second PASS: exactly one additional standard episode completed in `standard_post_exclusion_20260906_003`. Its 165 transitions match the same frozen stream and SHA-256 exactly; V2 continuity/sensor/hash, 29-row near-hole, success/non-truncation, zero contact, and 0.997009 axial tracking Gates PASS. Accepted demos now total 7 with standard=3. `LATERAL_PLUS_READY` requires separate authorization; no +Y or other recording, training, resume, or dataset build ran. See `results/validation/peg/f3_v2_formal_standard_second_20260906_003/REPORT.md`.

Latest F3 V2 Formal Standard +1 PASS: exactly one new standard episode was recorded through the validated formal launcher into `standard_post_exclusion_20260906_002`. All 165 V2 transitions match the frozen accepted stream phase/action sequence and SHA-256 exactly; continuity, sensor reconstruction, common hashes, 29-row near-hole coverage, success/non-truncation, zero contact and 0.997009 axial tracking ratio PASS. Accepted total is now 6 (standard=2); `STANDARD_SECOND_READY` requires separate authorization. No other recording, training, resume, or dataset build ran. See `results/validation/peg/f3_v2_formal_standard_20260906_002/REPORT.md`.

Latest F3 Standard Formal Invocation Fix PASS: a new invocation-only launcher reads the accepted 165-transition standard stream at its frozen SHA-256 and supplies commands to the unchanged contact-audit recorder through `subprocess.run(input=...)`, removing the failed shell `python -c` quoting boundary. Argparse, exact stream, output, seed0/max_steps500/type, extra-token rejection and pre-record dry-run Gates PASS. No simulator, Demo, collection retry, training, resume, or production/canonical change occurred. `STANDARD_FORMAL_RERECORD=READY` for separate authorization. See `results/validation/peg/f3_standard_formal_invocation_fix_20260906/REPORT.md`.

Latest F3 LateralPlus PlanPathFix PASS: a read-only verifier now binds exclusively to the canonical project-relative WSL plan path and accepts no path override. File/readability, exact SHA-256,176-command count, legal phases, correction label,7D finite/bounded actions and five path fail-closed regressions PASS. The plan was not rewritten. LATERAL_PLUS_RERECORD=READY; simulator/recording/`_002`/training/resume were not started. See `results/validation/peg/f3_lateral_plus_plan_path_fix_20260905/REPORT.md`.

Latest F3 V2 LateralPlus Rerecord-002 STOPPED/FAIL at preflight: the check referenced an absent WSL report-copy path instead of the intact accepted plan at `f3_lateral_plus_auto_closure_20260905/command_plan.json` (SHA-256 a7494b...,176 commands). Per no-retry policy, it did not switch paths: simulator/recording/audit were NOT_RUN, `_001` is unchanged and `_002` does not exist. See `results/validation/peg/f3_v2_lateral_plus_rerecord_002_20260905/REPORT.md`.

Latest F3 LateralPlus BootstrapFix PASS (CPU/static only): the plan helper now uses the project's standard root bootstrap before importing `training`; direct WSL compile/import checks pass. The complete176-command plan has zero unknown phases, dimension/finite/bounds errors, and zero `lateral_correction` labels; three correction commands use `lateral_alignment` with unchanged action values. Prefix109 commands and final59 insertion actions exactly match preserved evidence. LATERAL_PLUS_RERECORD=READY for separate authorization; simulator/recording/training were not started and `_002` does not exist. See `results/validation/peg/f3_lateral_plus_bootstrap_fix_20260905/REPORT.md`.

Latest F3 LateralPlus-AutoClosure STOPPED/FAIL: the planned correction label was changed to legal `lateral_alignment` and local syntax compilation passed, but the first WSL CPU/static-plan invocation failed before validation/publish with `ModuleNotFoundError: training` because the new helper lacked direct-script project-root bootstrap. The no-retry rule stopped the task: simulator and `_002` recording were not started, `_001` remained untouched, and no later pilot/training/resume ran. See `results/validation/peg/f3_lateral_plus_auto_closure_20260905/REPORT.md`.

Latest F3 V2 RemainingPilots STOPPED/FAIL: the separately authorized lateral +Y pilot started once in a fresh directory and durably recorded 109 transitions. A 4.50795mm +Y challenge was established at 37.9950mm axial distance with zero contacts, but the first intended correction command used unsupported phase `lateral_correction`; the frozen recorder rejected it before action dispatch and marked the episode `INTERRUPTED`, ineligible, and unapproved. Per protocol, lateral -Y/roll/pitch were NOT_RUN, no retry occurred, and no pilot was promoted or training started. See `results/validation/peg/f3_v2_remaining_pilots_20260905/REPORT.md`.

Latest PostExclusionPilotAudit PASS (offline): preserved failed standard pilot and successful bounded diagnostic use fixed reset seed0, identical goal, V2 contract, runtime/config and control/reward/success/action bindings. All126x7 actions are exactly equal. Loaded source semantic diff is only the authorized Link01 exclusion; the other manifest addition is its non-loaded backup. Axial realization changed from0.00882149mm to12.0825768mm for about12mm requested, so historical nonresponse is removed. STANDARD_V2_RERECORD=READY for separate authorization only; no simulator/Demo/training. See `results/validation/peg/f3_post_exclusion_pilot_audit_20260905/REPORT.md`.

Latest AxialBoundedDiagnostic-AfterTelemetryHardening PASS: exactly one fixed-reset run completed all 126 frozen actions through `observer.step`, with 126/126 complete telemetry rows and no observer error, clamp, contact, limit, solver, or conditioning blocker. Axial steps107--126 requested12.0000005mm and realized12.0825768mm, so historical nonresponse was not reproduced. All126 actions exactly match the preserved pre-exclusion failed pilot, where realized motion was0.00882149mm; Link01 remained absent, supporting exclusion causal contribution without claiming sole/general sufficiency. Diagnostic only/ineligible; no Demo/training/resume or control change. See `results/diagnostics/peg/f3_axial_bounded_after_telemetry_hardening_20260905/REPORT.md`.

Latest AxialTelemetryFailureEvidenceFix PASS: passive observer failures are now classified across RAW_STEP/IK/JOINT/ACTUATOR/CONTACT/POSTPROCESS/JSON_NORMALIZE/SERIALIZE/WRITE/FSYNC/PUBLISH and attempt an exact, strict, atomic no-clobber `failure_step_NNNNNN.json`. Artifact persistence failure is explicitly non-durable and cannot mask raw-step exceptions. Normal `steps.jsonl` schema, one original env.step per path, bit-exact controller output, and residual/Jacobian counts pass. 44 CPU/static tests PASS; production bindings, controller/IK, exclusion/guard/reward/success/action/V2 semantics unchanged. AXIAL_DIAGNOSTIC=READY for separate authorization; no simulator/Demo/training. See `results/validation/peg/f3_axial_telemetry_failure_evidence_20260905/REPORT.md`.

Latest AxialBoundedDiagnostic-FinalRetry FAIL: the single authorized post-exclusion run passed durable fixed-reset, repaired semantic invariant, and unchanged zero-contact guard boundaries, then dispatched exactly one setup action through `observer.step`. It stopped immediately because passive telemetry was incomplete; `cartesian.jsonl` is durable, but `telemetry/steps.jsonl` is 0 bytes. No axial phase was entered, axial nonresponse was NOT REPRODUCED, and all axial causal fields plus Link01 exclusion contribution remain UNRESOLVED. No Link01/other contact appeared through step1. No rerun, Demo, training, resume or production change. See `results/diagnostics/peg/f3_axial_bounded_final_retry_20260905/REPORT.md`.

Latest AxialDiagnosticContractFix PASS: retry runner now durably persists live reset before compatibility/unchanged guard and replaces invalid full-obs equality with frozen V2/shape/finite/reset/goal/binding/exclusion/action invariants plus EE tolerances (50um, 1e-4rad). obs[7:13] force/torque delta is recorded but not equality-gated. 12 CPU tests PASS in Windows and WSL; accepted historical post-exclusion fixture PASS. Production/exclusion/telemetry/guard/controller/IK/reward/success/action/V2 semantics unchanged. AXIAL_DIAGNOSTIC=READY, not run; no Demo/training. See `results/validation/peg/f3_axial_diagnostic_contract_fix_20260905/REPORT.md`.

Latest AxialBoundedDiagnostic-Retry FAIL: the single authorized post-exclusion invocation performed fixed reset but stopped PRE_ACTION because the new runner incorrectly required the full post-exclusion 20D observation to equal the pre-exclusion failed-pilot observation exactly. Exclusion/compile/V2/telemetry/process preflight PASS; 0 actions and 0 telemetry steps. Offline accepted-reset comparison shows micrometre-scale pose drift but largest changes in force/torque fields after removal of the former self-contact. Axial nonresponse NOT REPRODUCED; all downstream blockers and exclusion causal contribution UNRESOLVED. No automatic rerun/Demo/training. See `results/diagnostics/peg/f3_axial_bounded_retry_20260905/REPORT.md`.

Latest Link01ExclusionFix PASS: the Aubo source MJCF now contains exactly one `base_link`/`shoulder_Link` adjacent-body exclusion and the existing generator preserves it. Compile-only MuJoCo load PASS (`nexclude=1`). Accepted fixed-reset evidence has 0 total contacts, 0 link0-link1/peg-hole/robot-fixture contacts, unchanged guard not triggered, and 0 actions. Controller/IK/sensor/reward/success/action/V2 obs/Demo/checkpoints unchanged. AXIAL_DIAGNOSTIC=READY but not run; no Demo/training. See `results/validation/peg/f3_link01_exclusion_fix_20260905/REPORT.md`.

Latest GeometryFixReview PASS (static): exact saved reset contact plus six-angle STL AABB review confirms persistent link0/link1 overlap potential. Project mesh paths/transforms match the official Aubo URDF structure, while the official SRDF explicitly disables all six adjacent arm pairs, including base_link/shoulder_Link as Adjacent; local MJCF has none. Classification LIKELY_MISSING_EXCLUSION, not proven mesh defect. Other-pair exact overlap remains UNRESOLVED. Next EXCLUSION_FIX_REVIEW; no model/guard/simulator change. See `results/validation/peg/f3_geometry_fix_review_20260905/REPORT.md`.

Latest Link01CollisionIntentAudit PASS (static): link0-link1 direct parent/child, shoulder_joint, no exclude, collision enabled. All six Aubo adjacent pairs are configured ALLOW_ALL and generator preserves contact sections; no source exclusion was dropped. Semantic intent/benignity remains UNRESOLVED due no comment/doc/pair policy. Next GEOMETRY_FIX_REVIEW; no model/guard/simulator change. See `results/validation/peg/f3_link01_collision_intent_20260905/REPORT.md`.

Latest BaselineContactReview-Retry PASS: one fixed reset and zero actions. Durable full evidence precedes unchanged guard, which triggered on robot `base_link/link0` geom49 versus `shoulder_Link/link1` geom52 (4 contacts, distance -0.00649972274584152m). No peg-hole or robot-fixture reset contact. Benignity/root-cause UNRESOLVED; pair-aware static review recommended. No axial diagnostic/Demo/training. See `results/diagnostics/peg/f3_baseline_contact_review_retry_20260905/REPORT.md`.

Latest DiagnosticSerializationFix PASS: NumPy-safe strict JSON boundary and atomic exclusive snapshot publication;19 CPU tests PASS. Production/telemetry hashes, guard and runner action AST unchanged. BASELINE_CONTACT_REVIEW=READY for separately authorized reset-only collection in a new directory. No simulator run; historical FAIL and geometry UNRESOLVED preserved. See `results/validation/peg/f3_diagnostic_json_20260905/REPORT.md`.

Latest BaselineContactReview FAIL: one fixed reset, zero actions. Initial snapshot JSON serialization failed at NumPy bool_ joint controller_arm_member before guard evaluation; partial file preserved, no contacts available. Previous CPU PASS lacked real-shaped scalar coverage. Root cause and baseline contact classifications UNRESOLVED. No retry or production change. See `results/diagnostics/peg/f3_baseline_contact_review_20260905/REPORT.md`.

Latest InitEvidenceFix: PASS (7 CPU tests). Diagnostic runner fsyncs complete initial passive snapshot before unchanged initial guard; exact triggering contact event is durable before STOP. Production/telemetry hashes and action-loop AST unchanged. BASELINE_CONTACT_REVIEW=READY prospectively; no new simulator run or initial contact evidence. Root cause remains UNRESOLVED. See `results/validation/peg/f3_init_evidence_fix_20260905/REPORT.md`.

Latest AxialBoundedDiagnostic: FAIL, single initialization/fixed reset stopped at initial contact guard before action1 (0 actions). Guard distance<-5mm was conservative diagnostic policy; triggering contact details were not persisted, a runner evidence gap. Axial nonresponse NOT REPRODUCED, all downstream causal fields UNRESOLVED. No retry/new Demo/training; production unchanged. Next: initial contact evidence persistence/review, separately authorized retest only. See `results/diagnostics/peg/f3_axial_bounded_20260905/REPORT.md`.

Latest AxialTelemetry: independent opt-in observer implemented;18 CPU tests PASS, protected production bytes/failed pilot unchanged. AXIAL_DIAGNOSTIC=READY for separately authorized use, not executed. Solver/joint/ctrl/contact evidence fields available prospectively; historical root cause remains UNRESOLVED. No new simulator/model/reset/rollout or pilot/training. See `results/validation/peg/f3_axial_telemetry_20260905/REPORT.md`.

Latest AxialTrackingAudit: existing steps100–126 audited offline. Cartesian target accumulation PASS; adapter/base clipping not blocking. IK/joint audit incomplete (FAIL evidence gate), contacts UNRESOLVED because qpos/q_target/ctrl/contact pairs were not recorded. No root cause attribution beyond downstream-of-target blockage. Need separately authorized telemetry-only instrumentation; no rollout or repair performed. Evidence `results/validation/peg/f3_axial_tracking_audit_20260905/REPORT.md`.

Latest F3 V2 DemoPilot (2026-09-05): standard FAIL after126 transitions; agent-generated simulated demonstration, not human data. Axial command12mm produced0.0088mm EE displacement over steps107–126; target tracking error grew to12.51mm. Controlled q stop, INTERRUPTED/ineligible preserved. Other four pilots NOT_RUN. Contract/sensor/continuity/hash audit PASS. No retry, training, checkpoint load/change or data promotion. Collection BLOCKED pending axial tracking review. See `results/validation/peg/f3_v2_pilot_20260905/REPORT.md`.

Latest F3 V2 RecorderReady: schema/sensor/phase/hash/preflight PASS,27 synthetic CPU tests PASS. V2_DEMO_PILOT=READY_TO_RECORD (tool readiness only). Default CLI is preflight-only; explicit recording authorization required. No Demo/simulator/training started. Pilot success and coverage remain unverified; training remains BLOCKED. See `results/validation/peg/f3_v2_recorder_ready_20260904/REPORT.md`.

Last updated: 2026-09-04

Latest F3 DataReview:20 V1 demos/5512 transitions,20 successful terminals; zero rotation valid for near-aligned fixed reset but active correction coverage INADEQUATE (5213/5512 orientation<=0.05). Frozen near-hole10mm lateral/20mm axial contains1131 transitions. V2 five-pilot protocol/acceptance spec PASS; collection BLOCKED pending recorder V2/raw sensor/hash/phase readiness. No recording/rollout/training. See `results/validation/peg/f3_data_review_20260904/REPORT.md`.

Latest F3 SensorFix: address/dim helper and CPU tests PASS; env/config V2, immutable legacy formal models and Demo hash-labeled V1 in external registry. Default V1->V2 eval/restore blocked; explicit diagnostic-only mode is non-skill and NOT executed. Legacy Demo cannot reconstruct missing Ty/Tz from available PKL/NPZ. No training/resume/checkpoint/cloud changes; prior Gates retained/stale, V1 Demo/V2 training blocked. See `results/validation/peg/f3_sensorfix_20260904/REPORT.md`.

Latest F3 ContractAudit: action mapping PASS; torque sensor id1/adr3 sliced1:4 instead of3:6, observation/force mapping FAIL. First5 Demo episodes1328 transitions match source; full5512 actions have zero rotation. Rotation empirical check UNVERIFIED (gate FAIL), not proven rotation mismatch. One9000 episode reproduces prior trace; tracking error67=40.6mm/0.0947rad, commanded goal-orientation error0.683rad. No production fixes/training. See `results/validation/peg/f3_contract_audit_20260904/REPORT.md`.

Latest F3 Diagnostic:5000/9000 one fixed episode each, diagnostic PASS; best simultaneous success ratios26.2649/13.3361, neither<=1. Mixed approach/fine-alignment/orientation failure; lateral threshold never met. E saturation mostly gripper-driven, D force rejection unproven. Recommend action/controller/Demo and sensor-mapping audit before any fix; no training-code change. See `results/validation/peg/f3_diagnostic_20260904/REPORT.md`.

Latest F3 Stage2: fixed-reset screening execution PASS,5x10 episodes all0 successes. Diagnostic ranking9000>5000>7000>6000>8000; no final Top2.9000 min-position error26.6% lower than5000 but trend non-monotonic. CategoryB descriptive approach improvement; Policy Quality Review RECOMMENDED, FINAL_COMPARISON BLOCKED. No training/resume/cloud change/random reset/final evaluation. See `results/validation/peg/f3_stage2_20260904/REPORT.md`.

Latest F3 Stage1: evaluator preflight/model load/smoke execution PASS. checkpoint9000 only,10 episodes,0 successes, all500-step truncations. Fixed-reset deterministic trajectories repeat across seeds0–9; no generalization claim. Screening READY/NOT STARTED; no formal50-episode evaluation. Model/replay/cloud unchanged, no training. See `results/validation/peg/f3_stage1_20260904/REPORT.md`.

Latest F2-RetentionFix: bundle-aware retention repair PASS /62 CPU tests PASS. Isolated Flax staging plus numeric global-step bundle retention prevents sidecars consuming model keep slots; interrupted transactions fail closed. Original serialization/hash/binding semantics, config/runtime/Demo/sync/cadence unchanged. Historical missing0/1000 and partial2000 are NOT repaired;3000–9000 remain complete. Raw F2 FAIL and v3/v4 preserved; evaluation BLOCKED pending surviving-artifact adjudication and separate approval. No training/resume/cloud modification/evaluation. See `results/validation/peg/f2_retention_20260904/REPORT.md`.

Latest F2-ExitFix: exit classification PASS /83 CPU tests PASS; v4 generated formal_ready=false, evaluation BLOCKED. Re-audit discovered Flax keep20 counts sidecars and automatically removed8 early bundle files during F2; only3000–9000 remain complete (2000 metadata only). Save-time audits were PASS but final retention is not. Retained hashes/checkpoint9000,10000 updates and1000 cloud loss rows PASS; raw F2 FAIL/runfailed/v3 preserved. No training/resume/cloud change/evaluation. Separate bundle-aware retention repair authorization required. See `results/validation/peg/f2_exitfix_20260904_review/REPORT.md`.

Latest F2-Run: overall `PEG_FORMAL_TRAINING=FAIL` due W&B final-state classification. Actual formal updates10000/10000 PASS; all10 periodic bundles0–9000 PASS;1000 unique cloud loss steps match local data. Run e6fea100 statefailed despite learner exit0: frozen finally treats successful absl SystemExit as error and reports exit_code1. No timeout/NaN/RNG failure/retry/resume/evaluation; associated actor stopped, no training residuals. No frozen source/config/runtime/Demo/Gate changes. Separate exit-status repair authorization needed; see `results/validation/peg/f2_run_20260904/REPORT.md`.

Latest F2-Monitor: `PEG_FORMAL_READINESS_V3=PASS`, `PEG_F2_EXECUTION=READY_TO_START` (NOT STARTED; separate authorization required). Formal-only120s watchdog, required finite-loss/RNG guards and per-update local telemetry added; W&B10/global checkpoint1000/max10000/runtime/config/Demo/per-leaf implementation unchanged.73 CPU tests plus syntax/actual fresh preflight PASS. Source-bound v3 published; v2 and all historical bundles preserved, v2 now stale. No GPU training/W&B run. See `docs/peg_f2_monitor_20260904.md`.

Latest F1: `PEG_FORMAL_READINESS=PASS`, `PEG_FORMAL_TRAINING=READY_TO_START` (NOT STARTED). Shared per-leaf production sync, four-role guarded preflight, same-run binding, global checkpoint cadence1000/max_steps10000 and formal runtime freeze validated by47 CPU tests plus syntax/actual fresh-directory checks. New source-bound `results/validation/peg/formal_readiness_v2.json` has formal_ready=true; old failed Gate and all B/F0-R artifacts preserved. Historical GPU evidence reused explicitly; no F1 GPU run or checkpoint30 resume. Live actor/resume role checks were fixture-tested and repeat on actual launch. Native sync cause UNRESOLVED. See `docs/peg_f1_hardening_20260904.md`.

Latest F0-R branch acceptance: `PEG_FORMAL_EXACT_PATH_CHECKPOINT_RESUME_5=PASS`. Exactly one production Python resume from B checkpoint25 completed true updates26–30; new W&B validation branch2243685c finished with five matching unique steps. 198 state leaves/replay/RNG restored exactly; new checkpoint30 counter1651 consistent, model finite. Source hr88a6u1 and old artifacts/Gate unchanged. No same-run continuation claim, retry/restart/formal. Formal remains BLOCKED pending general entry hardening/readiness review. See `docs/peg_f0_r_branch_20260904.md`.

Latest F0-Verifier-Fix: schema fix PASS; existing B re-audit PASS, W&B history50 PASS; `PEG_FORMAL_EXACT_PATH_ACCEPTANCE_50=PASS_WITH_PER_LEAF_SYNC` for hr88a6u1 only. Dotted fields match50 unique steps/finite local losses, runfinished; checkpoint0/25 audits PASS. No training rerun/cloud writes/sync changes. Old raw FAIL and formal Gate unchanged; formal BLOCKED pending exact-path resume and entry hardening. Evidence: `results/diagnostics/peg_f0_verifier_fix_20260904/reaudit.json`.

Latest F0-Sync: `PEG_F0_SYNC_PER_LEAF_50=PASS`, one B arm only. Update1 dispatch4.163s/sync0.028s;50 updates finite/normal RNG; bundles0/25 counters751/1261 consistent. Supports targeted per-leaf workaround, not exact native root cause. Raw full verifier FAIL: filtered W&B query returnedempty rows, runfinished; history-loss not established. Old Gate unchanged; formal BLOCKED. See `docs/peg_f0_sync_20260904.md`.

Latest F0-Observe: observation COMPLETE; continuous run FAIL (1/50). Exactly one fresh instrumented production-path attempt reached update1 Python return in4.430s after XLA compilation completed, then timed out in full synchronization. 30s stack identifies JAX batched_block_until_ready; native cause UNRESOLVED. No retry/resume/formal; Gate remains failed. See `docs/peg_f0_observe_20260904.md`.

## Latest Peg F0 — exact production-path acceptance

2026-09-04: `PEG_FORMAL_EXACT_PATH_ACCEPTANCE_50=FAIL`; `PEG_FORMAL_TRAINING=BLOCKED`. Fixed runtime policy, source-bound fail-closed gate, preflight and an acceptance profile using the delivered learner were implemented. Six policy/history unit tests and syntax checks passed. Exactly one fresh bounded run completed update 0 (5.222 s), saved checkpoint_0, then timed out at update 1 after 120 s: 1/50 completed. Learner exited 124; supervisor stopped actor; no residual production processes. No retry, resume or formal training. No formal runtime file was created. Historical isolated I1–I4 PASS does not supersede this exact-path FAIL. Gate artifact has formal_ready=false. Hardening is PARTIAL: role-aware formal preflight/output handling and timeout profile-label cleanup still require review; not production-approved. See `docs/peg_f0_20260904.md`.

## Latest Peg I1 diagnostic

Latest I4 (2026-09-04): `PEG_I4_RESUME_LIFECYCLE_5=PASS_WITH_WANDB_HISTORY_WARNING`. Only repaired I3 checkpoint19 was restored; 198 state leaves plus replay/RNG/counters matched, five continuous updates20–24 passed, and new checkpoint24 has exact counter2102=2223−121. Same W&B44yrxufd resumed and finished; new steps20–24 unique/matching. Raw strict verifier FAIL is preserved because cached cloud history contains identical duplicates of old steps0–19; offline audit confirms lifecycle PASS with history-uniqueness warning. No retry/residual/formal. See `docs/peg_i4_resume_20260904.md`. Formal BLOCKED; historical NOT STARTED statements below are superseded only for this repaired source bundle.

Latest I3-Fix (2026-09-04): snapshot/counter repair VALIDATED by 5/5 CPU concurrency regressions, followed by exactly one new 20-update I3 run. `PEG_I3_CHECKPOINT_IO_20=PASS` for `peg_i3_checkpoint_20260904_132008`; counters 691=812−121 and 1171=1292−121. W&B `44yrxufd` finished with steps 0–19; no stall/residual. Normal save cadence unchanged; no resume executed. Old I3 FAIL remains preserved. See `docs/peg_i3_snapshot_fix_20260904.md`. Formal remains BLOCKED; actual new-bundle resume and full formal readiness are not validated.

Latest I3 audited result (2026-09-04): `PEG_I3_CHECKPOINT_IO_20=FAIL`. Continuous 20/20 updates, live communication, finished online W&B and periodic saves at 0/19 succeeded without stall. Independent audit supersedes the raw PASS: checkpoint_0 actor count 691 vs actual 721, checkpoint_19 1141 vs 1201. Existing save_bundle writes a stale actor counter after model I/O while live inserts continue. See `docs/peg_i3_checkpoint_integration_20260904.md`. No fix/retry/resume executed; new I3 bundles are NOT validated resume artifacts. Formal remains BLOCKED.

Latest subsequent result: `PEG_I2_WANDB_20=PASS` (2026-09-04). I1 plus online W&B init/log/finish completed 20/20; online run `21p57czu` is finished with cloud steps 0–19 and losses matching local records. 1,201 accepted live transitions, 21 parameter callbacks, normal RNG, no stall or checkpoint saving. Update 0/1: 5.554/4.357 s, then 0.012–0.029 s. One index-1 recompile remains. See `docs/peg_i2_wandb_integration_20260904.md`. Full candidate FAIL / formal BLOCKED remain unchanged; no I3 started.

`PEG_I1_LIVE_COMMUNICATION_20=PASS` (2026-09-04). One fresh learner completed 20 original updates with normal RNG and dynamically sampled 32 online + 32 Demo; real actor uploaded 1,111 transitions and received 21 parameter broadcasts. Live uploads overlapped both compilation and steady-state updates. Index 0/1: 5.549/4.626 s with one index-1 recompile; indices 2–19: 0.015–0.033 s. No stall, W&B, model checkpoint restore/save, restart, or retry. Both processes exited 0. See `docs/peg_i1_live_communication_20260904.md`. This single I1 run does not exclude intermittent failures or pass full candidate integration: `PEG_FORMAL_CANDIDATE_GPU_GATE=FAIL`, `PEG_FORMAL_TRAINING=BLOCKED` remain unchanged.

## Current Phase

MILESTONE 1D — THREE TASK GPU TRAINING PREPARATION is COMPLETE. The frozen delivery model is three independent parallel operators: A/Cable, B/Object, C/Peg. All per-task README 0–13 flows, transfer checklists, task sheets, independent environments/W&B/data contracts, return contracts, and CPU-reference verifiers pass. `THREE_MODEL_PARALLEL_GPU_HANDOFF_READY = READY`. Target-host GPU gates, formal training, and formal evaluation have not run.

MILESTONE 1C evidence remains valid: Cable online W&B, actor execution, and entry into the learner update path were observed; the checkpoint root cause and project-side legacy Flax save/restore unit path were validated. The prior smoke remains PARTIAL. `FULL_GPU_SMOKE`, `FORMAL_TRAINING`, and `FORMAL_EVALUATION` are `NOT_STARTED` for Cable.

## Incoming Process Packages

- Cable Routing V1: PRESERVED — original archive and isolated extraction remain unchanged.
- Cable Routing V2: PRESENT — new archive inventoried, safety-checked, and isolated under `external/packages/cable_routing_v2/`.
- Object Grasping: PRESENT and frozen — archive/package, Demo, training sources, and checkpoint tree were hashed in M1D-B; package source remains unchanged.
- Peg Insertion: PRESENT and frozen — archive/package, all Demo representations, algorithm entries, and checkpoint groups were audited in M1D-D.

Original archives remain unchanged in `external/incoming/`. Isolated extracted copies are under `external/packages/`.

## Completed

- Inventoried three incoming archives by byte size, magic-based file type, and SHA256.
- Tested archive integrity and rejected unsafe absolute or parent-traversal paths before extraction.
- Mapped and extracted all three packages without modifying incoming originals.
- Completed static package audits and a source-based interface compatibility comparison.
- Recorded package-specific findings, integration risks, process comparison, interface matrix, and a provisional integration plan.
- Recorded the project-member confirmation that Cable Routing intentionally has no delivered demo and that acquisition is a project-owned TODO.
- Recorded Object Grasping's project-member algorithm mapping as DPRL, with package-code identity marked NEEDS_VERIFICATION because the delivered implementation names DrQ components.
- Created external smoke harnesses for Cable Routing, Object Grasping, and Peg Insertion.
- Executed the three smoke harnesses in the required order and recorded exact runtime output.
- Runtime Environment Discovery: VALIDATED. Conda is at `/home/ubuntu/miniconda3/bin/conda`; existing `serl_mujoco` and `serl` environments contain the required core packages.
- Peg runtime output isolation implemented through a project-side adapter; generated XML is confined to `runtime/peg_insertion/generated/`.
- Peg completed import/create/reset/one-step/close in `serl_mujoco` with EGL.
- Cable V2 archive integrity/safety and isolated extraction completed; the V2 package supplies the missing Franka assets.
- Cable V2 completed import/create/reset/one-step/close in the version-matched `serl` runtime with EGL.
- Implemented a project-owned Cable keyboard recorder, learner-facing observation adapter, success-only episode gate, metadata writer, and static validator.
- Completed one TEST_ONLY real-environment transition dry run and inserted its pickle through SERL's population/replay path; no formal demo was created.
- Audited all Cable V2 checkpoint groups, the DrQ construction/restore/inference call chain, and selected the highest numbered delivered smoke checkpoint for a bounded read-only restore attempt.
- Added a project-owned MuJoCo passive-Viewer Cable recorder using the real environment, native Viewer key callback, delta actions, success-only immediate saving, truthful terminal semantics, and JSON-only discarded-attempt logs.
- Completed syntax/import and headless create/reset/one-zero-step/close preflight without writing any formal demo.
- Replaced the failed Viewer-native teleop callback with project-side pynput press/release state tracking and fixed-rate continuous/combined action generation; Viewer is now display-only.
- Traced and runtime-tested the full Cable action-to-target-to-opspace-to-ctrl-to-qpos-to-TCP pipeline with six single-step directions and two 20-step held actions, without modifying environment state directly or writing Demo data.
- Fixed only the project recorder's TCP operator display: `TCP_POS_REAL` uses `2f85/pinch_pos`, is checked against the pinch site, and reports mocap target separately. Bounded zero-action smoke returned `TCP_MATCH=True`.
- Implemented project-side global X11 key-event suppression with pynput `suppress=True`, explicit startup status, raw/logical key diagnostics, and unconditional listener stop/join cleanup.
- Diagnosed the premature Cable reset as two coincident 500-step limits and extended only the operator recorder instance's Gym/native truncation boundaries to 20000 control steps, with explicit pre-reset diagnostics.
- Added project-recorder Gear 1/2/3 selection on numeric keys, using 1/2/5 real `env.step` calls and one transition per call; action values remain clipped to the original space.
- Recorded operator evidence for the first real success: 2629 steps, `success=True`, `terminated=True`, `truncated=False`, final distance 0.029851879389882815, with the pickle preserved unchanged.
- Changed recorder cleanup to use a five-second bounded listener join and independent Demo/cleanup results; post-save cleanup timeout is a warning, while cleanup failure with zero saved successes remains an error.
- Synchronized the operator-confirmed validator PASS for the first formal Cable Demo: 1 episode, 2629 transitions, 1 success, `TEST_ONLY=False`, with the recorded SHA256 unchanged.
- Completed Cable initial-state randomization audit: source inspection and 20 resets with seeds 0–19 found exact zero span for TCP, robot qpos, cable root/configuration/tip, slot, and camera parameters. `INITIAL_STATE = FIXED`; `DEMO_DIVERSITY_RISK = HIGH`.
- Validated all five formal Cable demos individually and as a batch: 5 successful episodes, 10,521 raw transitions, consistent schema, expected replay size 10,526 with horizon-one initialization records, and safe in-memory episode-order concatenation.
- Completed a static Cable W&B preflight: confirmed learner-only online initialization, debug-disable behavior, current metric keys, learner-update global-step semantics, and missing train-return/buffer/success metrics. No W&B run or training process was started.
- Implemented the project-side Cable training launcher and single baseline config: explicit W&B project/name, sequential five-pickle loading without a merged duplicate, validated observation adapter reuse, isolated smoke/formal outputs, and fail-fast startup assertions. Dry run passed; no W&B run or learner update started.
- Recorded the real Cable smoke split result: online W&B run `flexible-line-cable/cable_smoke_5demo_seed0` created successfully; actor completed 500/500 and shut down cleanly; learner entered real updates but failed at its first checkpoint save.
- Diagnosed Flax 0.8.5 / Orbax 0.6.4 / TensorStore 0.1.51 checkpoint incompatibility. Exact `assume_config` rejection was reproduced; project-side legacy Flax save and restore passed without changing the Conda environment.

## In Progress

- Remote Operator A/B/C target-host acceptance, parallel formal training, independent evaluation, and artifact return are pending independently. None has started in repository evidence.
- Local MILESTONE 2 Unified Skill API work is the next project-side milestone; it must preserve task-specific interfaces and treat all trained-policy quality as UNVERIFIED until returns are accepted.

## Blocked

- Real Skill adapter integration — BLOCKED: runtime compatibility and checkpoint restoration have not been validated.
- Cable automatic-demo checkpoint evaluation — BLOCKED: absolute-path Orbax restore of `cable_route_smoke/checkpoint_200` did not finish within either the environment-derived attempt or the 180-second CPU-only diagnostic.
- Cable baseline smoke training — PARTIAL historical evidence: W&B online and actor pass; learner entered its update path and then failed on step-0 Orbax save. The legacy save fix is isolated-test PASS. The local retry is cancelled by strategy; a bounded full smoke must be performed on the target GPU host.

Cable Routing Demo Acquisition is TODO, not BLOCKED. It is an internally planned data-acquisition task and is not waiting for a package resend.

## Validated Components

- Archive identity, byte size, SHA256, magic-based type, integrity test, safe-path check, and extraction destination: VALIDATED.
- Presence/absence findings recorded by static filesystem and source inspection: VALIDATED.
- Runtime Environment Discovery: VALIDATED — `serl_mujoco` and `serl` were inspected using their own Python executables.

No delivered or trained policy checkpoint, policy quality, success rate, training result, or formal policy evaluation result is VALIDATED. Project-generated synthetic/full-architecture checkpoint and resume mechanics are validated only within their bounded preflight scope. Environment lifecycle validation covers Cable V2, Object Grasping, and Peg Insertion.

Object Grasping achieved ENVIRONMENT_SMOKE_VALIDATED in `serl` with headless EGL. M1D-B additionally validated the actual 2,000-transition schema, all-finite/action-bound checks, 2,001-record horizon-one replay insertion and packed sampling, 20-reset randomized block distribution, and code-level `DRQ_PLUS_DEMO_REPLAY` identity. This does not validate Demo success, checkpoint restore, policy, or success rate.

Peg Insertion M1D-D/D1 is `PASS_FOR_FRESH_GPU_TRAINING`: prior Demo/algorithm evidence remains valid; 5,512 official replay inserts plus 32/32 sampling pass through a scoped project repair; portable generated XML lifecycle passes; a fresh 198-leaf SAC state saves/restores exactly and produces one finite in-range `(7,)` action. Legacy checkpoint restore remains nonblocking and policy quality remains unverified.

Cable Routing V2 achieved ENVIRONMENT_SMOKE_VALIDATED in `serl` with headless EGL. This does not validate DrQ checkpoint, policy, demo, training result, success rate, or wrapper compatibility.

Cable Demo Recorder is VALIDATED. The first formal operator Demo also passed the existing validator with one successful episode and 2629 transitions.

Cable Routing formal Demo status:

- `DEMO_RECORDER = VALIDATED`
- `DEMO_DATA = PRESENT`
- `FORMAL_DEMO_COUNT = 5`
- `FORMAL_DEMO_SUCCESS_COUNT = 5` (operator-confirmed)
- `VALIDATED_FORMAL_DEMO_COUNT = 5`
- `UNVERIFIED_FORMAL_DEMO_COUNT = 0`
- `DEMO_BATCH_VALIDATION = VALIDATED`
- `DEMO_TRANSITIONS = 10521`
- `REPLAY_BUFFER_COMPATIBILITY = VALIDATED`
- `MERGE_COMPATIBLE = YES`

Cable Viewer rendering is IMPLEMENTED. Human test evidence validates `KEYBOARD_INPUT`, `PYNPUT`, and `ACTION_CONSTRUCTION`: real logs showed correct ±0.05 actions for W/S and corresponding nonzero actions for A/D. Robot teleop motion is not validated.

Cable `TCP_DEBUG_DISPLAY = FIXED`: real TCP sensor and pinch site matched during zero-action smoke. This validation covers display correctness only, not teleoperation motion.

Cable `KEY_SUPPRESSION = IMPLEMENTED` using pynput 1.8.2 `suppress=True`. Non-GUI construction and cleanup tests do not validate actual Viewer/X11 suppression; `OPERATOR_SUPPRESSION_VALIDATION = REQUIRED`.

Cable operator episode limit fix is IMPLEMENTED: Gym TimeLimit and native simulation time limit are both configured to 20000 steps on the recorder-owned instance. Success termination and Demo schema are unchanged.

Cable Gear speed is IMPLEMENTED: Gear 1/2/3 map to 1/2/5 environment transitions per fixed-rate tick, with Gear 3 default. Non-GUI tests validate selection, counts, terminal short-circuit, action bounds, and transition preservation; visible two-second displacement remains operator-required.

Cable recorder cleanup fix is VALIDATED by lifecycle tests. `DEMO_RESULT` remains authoritative after a completed save; listener timeout becomes `CLEANUP_RESULT: WARNING`. With no successful save, the same cleanup failure still raises.

## Unverified Components

- All checkpoint restore operations and policy performance.
- Cable automatic-demo policy: UNVERIFIED. No restored state was returned, no policy action was sampled, and zero evaluation episodes ran.
- Cable performance/generalization outside the single fixed initial state.
- Cable robot teleop motion at a usable magnitude. Viewer native key input is known `FAILED_FOR_TELEOP`; global pynput keyboard input is now VALIDATED by operator evidence.
- Cable Viewer shortcut suppression under the operator's real X11 session.
- Object Grasping Demo success and episode count: the 2,000-transition flat pickle has all `dones=False`, all masks 1, and no success/block/lift/boundary fields; the package's claimed 20 trajectories cannot be recovered or independently success-validated.
- Peg trained-policy quality and formal evaluation. Delivered legacy checkpoint restoration remains unresolved by policy and is not required for fresh training.
- All reported historical training results and W&B records.
- Cable W&B runtime metrics: code paths are statically audited, but no online run or metric emission has been runtime-validated in this project.
- Cross-version Gym/Gymnasium, JAX/Flax, SERL wrapper, replay-buffer, and agent compatibility.
- Object Grasping team-label DPRL meaning remains UNVERIFIED. The delivered code-level algorithm mapping is VALIDATED as `DRQ_PLUS_DEMO_REPLAY`: `DrQAgent` plus 50/50 online/Demo batches; no distinct DPRL class/loss/replay implementation is present.

## Known Issues

- Object legacy `checkpoint_4500`: PRESENT, architecture MATCH, native restore UNRESOLVED, required for new training NO. It is preserved and no further restore/conversion/inference work is planned.
- Object safe resume: cross-process step 123→125 restored agent values, learner/actor progress, online replay size/index/RNG, rebuilt Demo replay, W&B identity, collision protection, six failure cases and one `(4,)` inference. `OBJECT_RESUME_GATE = VALIDATED`.
- Team-reported historical Object success rate is approximately 70%; repository revalidation is not required and this is not a current experiment result.

- Object M1D-B4 reproduced Python/JAX/JAXlib/Flax/Orbax/TensorStore/Optax producing versions exactly in retained `object_ckpt_compat`; original `serl` freeze hash stayed unchanged. TensorStore 0.1.78 did not resolve the stall: Orbax metadata 60 s, no-target restore 120 s, and matching-target restore 300 s all timed out. `ROOT_CAUSE = UNRESOLVED_NATIVE_ORBAX_TENSORSTORE_COORDINATION`; inference not run.
- `OBJECT_DEMO_DATA = ACCEPTED_FOR_BASELINE` by project decision. This does not alter `OBJECT_DEMO_SUCCESS_PROVENANCE = UNVERIFIED`; the original 2,000 transitions will be used unchanged.

- Object M1D-B3: checkpoint 4500 storage integrity PASS; direct OCDBT metadata/key traversal and representative scalar/9 MB parameter/9 MB optimizer reads PASS. Orbax metadata timed out at 30 seconds and target-free Flax/Orbax restore timed out at 60 seconds after an initial 371,388,416 bytes of disk reads followed by a sleeping no-I/O/no-CPU plateau. Producing/current environments differ only in TensorStore 0.1.78 versus 0.1.51. `CHECKPOINT_RESTORE_STATUS = BLOCKED_READ_STALL`.

- Cable Routing had no delivered recorder or demo. The project-owned recorder has now saved one real operator-confirmed successful Demo; additional acquisition remains TODO.
- Cable Routing V2: `DEMO_DATA = PRESENT`; all five formal files `VALIDATED`; `DEMO_SOURCE = SELF_RECORDED`.
- Cable Routing V2 reset has no random reset/noise/domain randomization and ignores reset seed for initialization. Training, evaluation, and recorder share this fixed physical reset logic.
- Cable Routing contains two non-empty smoke/shutdown checkpoint groups and two empty named checkpoint directories; none is validated.
- Cable's highest delivered checkpoint is `cable_route_smoke/checkpoint_200`; its label indicates smoke data, not a final or validated policy. Its Orbax metadata appears structurally consistent with dual `encoder_front`/`encoder_wrist` DrQ, but full architecture compatibility remains UNVERIFIED because restore did not complete.
- Cable Routing has no README and its handover acceptance checklist is entirely unchecked.
- Object Grasping's success is exposed in `info`, but environment termination is time-limit based rather than success based.
- Object Grasping checkpoint 4500 exact architecture mapping is MATCH. Its 601 metadata paths are 598 arrays plus three Optax `EmptyState` sentinels (`None`, `skip_deserialize=true`); fresh target has the same structure when empty nodes are retained. Zero-gradient update leaves 598/601 counts unchanged, so this is not lazy post-update state. A new matching-target restore still did not return within 300 seconds; restore and inference remain UNVERIFIED.
- Object Grasping actual scalar gripper observation does not satisfy its declared `(1,)` Box, but bounded flattening produced `(7,)` and Demo/replay/DrQ structural paths passed; `OBJECT_GRIPPER_SHAPE_MISMATCH = BENIGN` for these tested consumers.
- The current project SERL `SERLObsWrapper` expects `image`, while Cable Routing and Object Grasping environments expose nested `images/{front,wrist}`.
- The Cable recorder uses a project-side `state/front/wrist` wrapper compatible with the replay buffer; the delivered training script still invokes the incompatible singular-`image` SERL wrapper before its demo loader.
- Cable learner W&B project is hard-coded as `serl_cable_route`; `--exp_name` supplies an ID prefix/tag but not an explicit `wandb.init(name=...)`. The requested `flexible-line-cable` / `cable_baseline_5demo_seed0` identity needs configuration before launch.
- Cable W&B currently logs raw critic/actor/temperature update metrics and timer/eval dictionaries at learner `update_steps`. It does not log `train/episode_return`, replay-buffer size, or training success. No missing metric is treated as present.
- `WANDB_ONLINE = VALIDATED`: real project `flexible-line-cable`, run `cable_smoke_5demo_seed0`.
- `ACTOR_SMOKE = PASS`; `ACTOR_STEPS = 500`; `ACTOR_SHUTDOWN = CLEAN`.
- `CABLE_TRAINING_LAUNCHER = IMPLEMENTED / UNVERIFIED`; `SMOKE_TRAINING = PARTIAL`; `LEARNER_SMOKE = PARTIAL`; `FORMAL_TRAINING = NOT_STARTED`; `FORMAL_EVALUATION = NOT_YET_VALIDATED`.
- `WANDB_ONLINE = VALIDATED`; `ACTOR_UPDATE_PATH = VALIDATED`; `LEARNER_UPDATE_PATH = VALIDATED`; `CHECKPOINT_ROOT_CAUSE = VALIDATED`; `CHECKPOINT_LEGACY_SAVE_RESTORE_UNIT_TEST = VALIDATED`.
- `FULL_GPU_SMOKE = NOT_STARTED`; `FORMAL_TRAINING = NOT_STARTED`; `FORMAL_EVALUATION = NOT_STARTED`.
- Cable real-agent legacy roundtrip: 598 serialized leaves; all key paths/shapes/dtypes and all leaf hashes matched after rebuild/restore; restored policy inference PASS. `REAL_AGENT_CHECKPOINT_ROUNDTRIP = VALIDATED`.
- Cable GPU handoff files use project-relative paths/`PROJECT_ROOT`; Demo manifest hash verification, frozen config dry run, output isolation, and local CPU/reference verifier passed. `PORTABLE_PATHS = VALIDATED` for prepared handoff files.
- Cable independent evaluation entry reset/one-step dry run passed. Metrics are defined, but trained-checkpoint evaluation is unverified: `EVALUATION_RUNTIME = PREPARED / TRAINED_CHECKPOINT_UNVERIFIED`.
- Cable safe resume: real agent checkpoint at step 123, complete online replay/RNG/actor progress sidecar, five-Demo rebuild, W&B `resume="must"` reconstruction, cross-process restore, next checkpoint 125, original preservation, and collision rejection passed. `TRAINING_RESUME = VALIDATED`.
- `CABLE_GPU_TRAINING_HANDOFF_READY = READY`. This does not validate GPU runtime, GPU smoke, formal training, trained policy, success rate, or formal evaluation.
- Peg Insertion is state-only SAC/RLPD with a 20D observation and 7D action, unlike the two visual DrQ packages.
- Peg Insertion uses a direct factory/adapter; the configured `AuboPegInsert-State-v0` name is not statically registered in the delivered adapter.
- Package scripts contain submitter-specific absolute paths requiring future configuration, not source mutation.
- Conda is not initialized on the current shell PATH but is available at `/home/ubuntu/miniconda3/bin/conda`.
- Primary candidate runtime: `serl_mujoco` — Python 3.10.20, Gym 0.26.2, Gymnasium 1.3.0, MuJoCo 3.10.0, JAX 0.4.35, Flax 0.8.5.
- Object Grasping compatible candidate: `serl` — Python 3.10.20, Gym 0.26.2, Gymnasium 1.3.0, MuJoCo 2.3.7, JAX 0.4.35, Flax 0.8.5.
- Peg declared versions differ from `serl_mujoco`; the current environment lifecycle is smoke compatible, but broader training/checkpoint compatibility remains UNVERIFIED.
- Cable V1 omits the `franka_emika_panda` mesh tree; V2 supplies 67 Franka asset files and resolves the previous `link1.stl` failure.
- Cable V2's supplied checksum manifest has a mismatching self-entry, while all 240 other listed files validate and archive-level integrity passes.
- Cable V2 headless smoke completed, but its EGL context destructor emitted a non-fatal cleanup exception after `env.close()` reported PASS.
- Object requires headless `MUJOCO_GL=egl` over this SSH session; actual gripper observation is scalar while its space declares `(1,)`.
- Peg factory normally derives generated XML output inside `/home/ubuntu/mujoco_env`; the project-side smoke adapter redirects only that generated output to `runtime/peg_insertion/generated/` and restores the original method afterward.
- Peg 20-reset initial state is fixed. The current project adapter still resolves read-only XML/assets through `/home/ubuntu/mujoco_env`, so GPU portability needs a project-side asset-root solution.
- Peg's installed SERL datastore retains an invalid sibling-class `super(...)` call; the project-scoped adapter repairs it without external mutation. Training must use that adapter.

## Latest Experiments

- 2026-09-04 Peg runtime-only reliability report: fresh fixed-batch R1 and normal replay-iterator dynamic-batch R2 each completed 20/20 original SAC updates, with finite losses and uninterrupted normal RNG. Both compiled once at index 0 and recompiled once at index 1 (4.276/4.307 s, WARNING); indices 2–19 were NORMAL at 0.013–0.023 s without further compilation. `PEG_RUNTIME_FIXED_BATCH_20=PASS`, `PEG_RUNTIME_DYNAMIC_BATCH_20=PASS`, `PEG_RUNTIME_RELIABILITY_CLASSIFICATION=RECOMPILE`. No timeout was reproduced. No resume, W&B, checkpoint writes, restart or retry occurred. This runtime-only result does not validate the integrated candidate Gate; formal remains BLOCKED / NOT STARTED.

- 2026-09-03 Peg target GPU acceptance: restored complete SERL source from upstream commit `1fa2af7496be042e43d36120a7c8d1b71f1bfb59`, applied only the delivered state-only `bc.py` patch with an explicit backup, installed the project-local `serl_launcher` and Peg `mujoco_env`, repaired legacy absolute-path pointers, aligned JAX/JAXlib/CUDA plugin to 0.4.35, and passed a real 2048x2048 GPU matrix computation on `CudaDevice(id=0)`. `scripts/verify_peg_handoff.py --require-gpu` reports all five contracts PASS and `PEG_HANDOFF_VERIFY: PASS`; `pip check` reports no broken requirements. The EGL environment smoke also reports `RESET_ONE_STEP_DRY_RUN: PASS`, `CONTROL_DT_SECONDS: 0.02`, and five-value step API. Formal training, online W&B validation, bounded learner/actor smoke, checkpoint save/restore, and formal evaluation remain NOT_STARTED.
- 2026-09-03 Peg target bounded-smoke checkpoint gate: W&B authentication and online run creation passed; the successful run ID is `2snda1i7` under `flexible-line-peg-insertion/peg_rlpd_gpu_smoke_seed0`. Actor completed 150/150 steps, learner received 121 online transitions, sampled exact 32 online + 32 Demo batches on JAX GPU, completed update step 0, and created fresh `/home/makoto/flexible_line_project/results/checkpoints/peg_rlpd_gpu_smoke_seed0/checkpoint_0` plus online-replay and resume-metadata sidecars. Per operator instruction, learner/actor were terminated immediately after the new checkpoint appeared, during update 1/10; no residual process remains and GPU memory returned to idle. Full 10-update smoke, W&B metric-sync validation, checkpoint restore, formal training, and formal evaluation remain pending. The launcher checkpoint recursion defect was fixed by retaining stable references to the original legacy-Flax save/restore functions; focused monkeypatch save/restore regression passed.
- 2026-09-03 Peg checkpoint Resume Gate follow-up: `checkpoint_0` metadata/replay hashes and the 121-transition replay snapshot were validated, then learner and actor dry-runs restored the bundle on `CudaDevice(id=0)`. The launcher now infers the smoke/formal resume profile from the checkpoint-bound W&B name, so the generic resume scripts preserve smoke limits and output paths. A real resume under the same W&B run `2snda1i7` restored model/replay state, completed one additional GPU optimizer update, emitted finite actor/critic/temperature/timer metrics at W&B step 1, and atomically created loadable `checkpoint_1` plus matching replay/metadata sidecars while preserving `checkpoint_0`. Actor continuation from recorded step 121 completed 29/29 bounded steps. A further attempt from `checkpoint_1` remained inside one JAX/XLA worker at approximately one CPU core for more than ten minutes without completing its next update or producing `checkpoint_2`; it was stopped and no training process remains. Therefore checkpoint restore/save and online metric-sync gates PASS, but the full 10-update bounded smoke remains PARTIAL. Formal training and formal evaluation remain NOT STARTED.
- 2026-09-03 Peg bounded GPU smoke completion: actor 150/150 and learner 10/10 are now complete under a strictly bounded, diagnostic GPU workaround. Investigation isolated the stall to the fused learner update on this RTX 4060/JAX 0.4.35 path: disabling Triton GEMM allowed updates to finish, while some PRNG states and the delivered periodic-checkpoint branch still produced non-returning GPU work. The project launcher therefore supports explicit diagnostic RNG provenance, one-update resume caps, and checkpointing immediately after synchronized update completion. Updates 2–9 were resumed one per process with `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` and the recorded safe key `[595175366, 1025945940]`; the frozen RLPD architecture, losses, optimizers, learning rates, batch size, 32/32 replay composition, Demo, environment, reward, and success logic were unchanged. Final `checkpoint_9` plus replay/metadata sidecars validates as 10 updates, 121 online transitions, matching replay SHA-256, 198 finite model-state leaves, GPU resume dry-run PASS, and W&B run `2snda1i7` finished with online actor/critic/temperature/timer history through step 9. `PEG_BOUNDED_GPU_SMOKE = PASS_WITH_DIAGNOSTIC_GPU_WORKAROUND`; nominal uninterrupted/formal training remains UNVERIFIED and MUST NOT start without resolving or explicitly accepting this workaround.

- 2026-09-03 Peg formal-readiness hardening and continuous candidate Gate: the launcher now requires an explicit `formal`, `smoke`, or `diagnostic` profile. All eight diagnostic update/RNG/save controls fail closed under formal or smoke, while an empty diagnostic profile also fails closed. The normal RNG path is unchanged. The candidate runtime contains only disabled JAX preallocation, disabled Triton GEMM, and EGL; no formal runtime file was created. A single fresh learner process using the original `SACAgent.update_high_utd`, normal RNG, the integrity-checked 121-transition smoke replay, no restart and no retry completed update 1/20 in 4.928487 s with finite losses, advanced RNG from `[1278412471, 2182328957]` to `[595175366, 1025945940]`, and created an independent `checkpoint_0` bundle whose GPU resume dry-run passes. In the same process, update 2/20 exceeded the 120 s hard limit and was terminated with exit 124. No residual training process remains. W&B run `tzv5bsvz` was created but currently reports `running` with no synced history after the forced exit. `PEG_FORMAL_CANDIDATE_GPU_GATE = FAIL`; formal training remains BLOCKED and was not started.
- 2026-09-03 Peg candidate checkpoint-0 next-update isolation: a fresh diagnostic learner restored candidate `checkpoint_0` with its normal RNG and ran exactly one original `SACAgent.update_high_utd` under disabled Triton GEMM, with W&B disabled, checkpoint period zero and no retry. Replay restore, 32+32 sampling and batch concatenation completed, but the update timed out at 130 s. Python stacks at 30 s and 90 s both showed the main thread in `jax.block_until_ready(new_agent)` after asynchronous update dispatch returned. GPU memory stabilized at 196 MiB while utilization remained mostly 4–8% and power about 2.3 W. This reproduces the failure in a new process and without W&B/checkpoint writes, ruling out same-process continuity, W&B, or the step-0 save as sufficient causes. Root cause remains unresolved at the JAX/XLA device-result synchronization layer; no residual process remains and formal stays BLOCKED.
- 2026-09-03 Peg JAX/XLA A/B differential: one no-retry fresh update and one fresh-process candidate-checkpoint restore update used the same immutable 32+32 batch, GPU, original `update_high_utd`, normal RNG, disabled Triton GEMM and explicit full-tree synchronization. A completed in 5.189362 s and B in 5.301514 s with finite losses; the earlier 130 s stall did not reproduce. The decisive pre-update difference is that fresh state is GPU `ArrayImpl`/`SingleDeviceSharding`, whereas all 198 restored leaves (including 150 optimizer leaves) are host NumPy values until the update boundary. The update HLO, optimized HLO, PTX and thunk hashes are identical across A/B, and sync itself took only about 0.012 s. Classification is `RESTORE_DEVICE_PLACEMENT_DIFFERENCE`, but causality for the earlier intermittent stall remains UNRESOLVED. `PEG_FORMAL_CANDIDATE_GPU_GATE = FAIL`; formal remains BLOCKED and was not started.
- 2026-09-03 Peg restore-placement causal diagnostic: D1 retained all 198 restored checkpoint leaves as host NumPy at the first update boundary; D2 explicitly placed and synchronized all 198 leaves on `cuda:0`. Both fresh-process arms used the same checkpoint RNG, immutable 32+32 batch, original `update_high_utd`, disabled Triton GEMM, no W&B/checkpoint writes/retry, and completed 5/5 consecutive updates. Update 0 took 5.062/5.135 s including compilation; later updates took 0.014–0.027 s. RNG progression and every finite loss were identical across arms. `PEG_RESTORE_PLACEMENT_CAUSAL_DIAGNOSTIC = NO_EFFECT`: explicit pre-placement neither improved nor regressed stability in this trial, and host restore placement is not a sufficient cause of the intermittent stall. Neither arm reproduced the timeout, so root cause remains unresolved and formal remains BLOCKED/NOT STARTED.

- 2026-08-23 M1D-F parallel freeze: Cable/Object/Peg CPU-reference handoff verifiers PASS (Object with required EGL), all three README step audits report 14/14, all environment YAML parses, all handoff shell scripts pass `bash -n`, no credential material was found, and no cross-task execution-order dependency was found. No training ran.

- 2026-08-23 Peg M1D-E: CPU verifier, full Demo launcher dry-run and evaluation reset+one-step PASS. Cross-process safe resume restored step 123/actor 17/replay RNG, rebuilt 5,512 Demo transitions, inferred once, committed step 125, preserved step 123, and passed six fail-closed tests. No training ran.

- 2026-08-23 Peg M1D-D1: 5,512 Demo inserts, 32 online + 32 Demo sampling, RNG/index/schema integrity, portable XML lifecycle, fresh 198-leaf checkpoint exact roundtrip, and restored `(7,)` inference PASS. `PEG_M1D_D_STATUS = PASS_FOR_FRESH_GPU_TRAINING`.

- 2026-08-23 Peg M1D-D: 20 NPZ/5,512 RLPD/2,388 BC transitions mapped exactly; Demo success 20/20; 20 resets fixed; algorithm/checkpoint mapping PASS. RLPD checkpoint_9000 restore TIMEOUT_120S and official replay insert TypeError, so status PARTIAL.

- 2026-08-23 Object M1D-B4: isolated producing runtime PASS and original-serl integrity PASS; direct reads PASS, full-tree restores TIMEOUT, inference gated off.

- 2026-08-23 Object M1D-B3: committed 105,193,080-byte OCDBT checkpoint inventory PASS; 1,196-key traversal and three isolated leaf reads PASS. Full-tree no-target restore timed out with native-wait behavior. GPU is not expected to fix the storage/runtime issue; next resolution is isolated producing-runtime reproduction.

- 2026-08-15 Runtime Environment Discovery: VALIDATED.
- 2026-08-22 Object M1D-B: 2,000 transitions deserialized read-only; schema/finite/action bounds PASS; no done/boundary/success/block fields; actual memory-efficient replay insertion/sample PASS at size 2,001.
- 2026-08-22 Object reset audit: 20 EGL resets; TCP fixed, block XY and scene images randomized, scalar gripper mismatch reproduced.
- 2026-08-22 Object checkpoint audit: checkpoint 4500 tree frozen at 15 files/105,193,080 bytes; 598/601 serialized-path comparison found three extra optimizer states; single restore timed out after 120 seconds and no inference ran.
- 2026-08-22 Object M1D-B2 provenance: no generator/writer or success-only guarantee is delivered. Exact 2,000 = 20 × 100 evidence makes boundaries strongly supported but not authoritative; success remains UNVERIFIED.
- 2026-08-22 Object M1D-B2 checkpoint closure: three apparent extra paths are Optax EmptyState sentinels. Exact structural mapping MATCH at 598 arrays/601 structural paths before and after a zero-gradient optimizer update. Matching-target Orbax restore exceeded 300 seconds; no state/inference returned.
- Earlier `/usr/bin/python3` smoke attempts used the wrong interpreter and are INVALID as dependency conclusions for all three processes; they did not validate or block any process runtime.
- Cable V1 in `serl_mujoco`: FAIL at create due missing `link1.stl` asset path; retained as historical V1 evidence.
- Object in `serl` with `MUJOCO_GL=egl`: PASS; ENVIRONMENT_SMOKE_VALIDATED.
- Peg in `serl_mujoco` with EGL and project runtime staging: PASS; ENVIRONMENT_SMOKE_VALIDATED. Actual observation `(20,)` float32, action `(7,)` float32, five-value step API.
- Cable V2 in `serl` with EGL: PASS; ENVIRONMENT_SMOKE_VALIDATED. Actual nested dual-camera observation, seven-value state, `(3,)` float32 action, and five-value step API.
- Cable recorder dry run in `serl` with EGL: PASS. One TEST_ONLY transition was serialized, statically validated, and inserted through the SERL population/memory-efficient replay path; success count 0 and no formal demo output.
- Cable automatic-demo preflight in `serl`: checkpoint inventory and DrQ call-chain audit completed. Absolute-path restore of `cable_route_smoke/checkpoint_200` did not complete in bounded attempts; CPU-only target construction reached 598 leaves, then timed out after 180 seconds. Episodes 0; candidate demo not generated.
- Cable Viewer recorder headless preflight in `serl` with EGL: syntax/import, Viewer API signature, model/data access, env create/reset, one zero-action step, and close passed. `DISPLAY` was unavailable, so visible Viewer/key input remain operator-required. No formal demo file was created.
- Cable Viewer teleop input fix: nine non-GUI action/state cases passed. An earlier headless shell lacked pynput, but this is superseded for the desktop operator runtime by later human logs validating pynput key states and correctly signed actions. Formal output remained empty.
- Cable action pipeline diagnosis: action scale is 0.001 m; magnitude 0.05 changes target by 0.05 mm/step. Target/mocap changed exactly, ctrl/qpos/TCP responded, and 20 negative-X steps moved true TCP X by -1.167 mm. Recorder debug output was reading `[gripper,tcp_x,tcp_y]` as TCP because it used flattened `state[:3]`.
- Cable teleop minimal fix: syntax/import/zero-action smoke passed; `TCP_POS_REAL=[0.3080,-0.0001,0.4440]` matched the pinch site and no formal file was written. Operator motion tests were not run.
- Cable key-event suppression: pynput 1.8.2 exposes `Listener(..., suppress=True)` and its X11 backend implements suppression. Syntax/import, suppressed-listener construction, normal cleanup, and exception cleanup passed without a GUI; operator Viewer suppression remains required.
- Cable premature-reset diagnosis: registration `max_episode_steps=500` and native `time_limit=10.0s` at `control_dt=0.02s` both caused truncation near step 500. Project-side configuration inspection confirmed both recorder-instance limits change to 20000/400.0s; no formal Demo was recorded.
- Cable Gear speed: static/non-GUI tests confirmed default Gear 3, key selection for all gears, 1/2/5 real step counts, one transition per step, and immediate stop on terminal output. No formal Demo was recorded.
- First real Cable Demo: `cable_routing_01_demos_2026-08-21_07-14-17.pkl`, 2629 transitions, one episode, one success, `TEST_ONLY=False`; existing validator result PASS is confirmed. SHA256 is `415b2adf5c2453db6532a6c5b9347b6ba6c2b97b393331c98c956e9ef71e54e0`.
- Cable cleanup fix: syntax and listener lifecycle tests confirm clean exit, post-save timeout warning without raise, and pre-save timeout error behavior.
- Cable initial-state audit: 20 reset-only samples, seeds 0–19, zero actions. Every measured component had exact span 0.0; classification `FIXED_INITIAL_STATE`. The subsequent five-demo milestone validated all five formal files.
- Cable five-demo validation: all validator invocations PASS; total 10,521 transitions and 5 successes. Real SERL population/memory-efficient replay insertion PASS with expected size 10,526. In-memory concat PASS with five intact terminal boundaries. No merged file was written.
- Cable W&B preflight: local W&B 0.28.0/base URL verified without credential access; online/disabled mode selection, log call sites, metric inventory, and global-step semantics statically confirmed. Training and W&B run creation were not executed.
- Cable project launcher dry run: all five SHA/schema checks passed sequentially with 10,521 transitions; real environment adapter returned `state/front/wrist`, action `(3,)`, replay capacity and output checks passed, and JAX exposed `TFRT_CPU_0`. Offline fake-W&B construction passed. W&B init and learner update were not called.
- Real Cable smoke: W&B online run creation PASS; actor 500/500 PASS with clean shutdown; learner checkpoint save FAIL at displayed 0/100. Minimal debug reproduced TensorStore `assume_config` rejection/default Orbax non-return and validated legacy Flax save+restore. Smoke overall PARTIAL.

## Next Milestone

Remote next steps are three independent target-host gates and parallel training by Operators A/B/C. Exact local next milestone is MILESTONE 2 — Unified Skill API draft/integration using audited task differences; it must not claim trained-policy validation while remote results are pending.
## 2026-09-05 — Peg F3 lateral -Y command plan freeze

- `PEG_F3_LATERAL_MINUS_COMMAND_PLAN = PASS`.
- Built an independent 176-command `lateral_minus_y` plan from the accepted +Y canonical plan. Only `action[1]` at commands 107-112 was sign-negated; every phase and all other action values remain identical.
- Frozen plan SHA256: `dcf3a7981098935cc8fe919a99e82fcad2c7e81783dd15c9047305bc439fef81`.
- Dedicated read-only verifier passed phase, 7D/finite/bounds, count, source binding, artifact hash, and exact symmetry checks.
- `LATERAL_MINUS_RERECORD = READY`; no simulator, roll, pitch, or training run occurred.
## 2026-09-05 — Peg F3 V2 lateral -Y pilot

- `PEG_F3_V2_LATERAL_MINUS = PASS`; recorder status `COMPLETED_SUCCESS` after 176 frozen-plan transitions.
- The -Y challenge reached `0.005036013234593285 m` lateral error; the opposite correction was dispatched and reduced it to `0.00003807880981983542 m`.
- Final lateral/axial/orientation errors were `0.00007743706718209285 m`, `0.0028496384193609403 m`, and `0.012833769154048885 rad`; success true, truncated false.
- All 176 contact rows were empty. V2 validation, continuity, sensor reconstruction, hash binding, near-hole coverage, and axial tracking passed.
- Pilot remains ineligible/unapproved for training. `NEXT_PILOT = ROTATION_ROLL_READY`; roll/pitch/training/resume were not started.
## 2026-09-05 — Peg F3 roll command plan freeze

- `PEG_F3_ROTATION_ROLL_COMMAND_PLAN = PASS`.
- Frozen 179-command roll plan preserves the accepted standard 106-command prefix and 59-command insertion/success suffix.
- Four normal `action[3]=+1` commands plan a `0.06981317007977318 rad` local-roll challenge; five `action[3]=-0.8` commands satisfy the correction-transition requirement and exactly cancel the planned rotation, followed by five settle commands.
- Dedicated hash-bound verifier passed phase, shape, finiteness, bounds, structure, source binding, challenge range and correction checks.
- `ROTATION_ROLL_RERECORD = READY`; simulator, pitch, and training were not started.
## 2026-09-05 — Peg F3 V2 rotation-roll pilot

- `PEG_F3_V2_ROTATION_ROLL = PASS`; recorder status `COMPLETED_SUCCESS` after 179 frozen-plan transitions.
- Runtime roll challenge peaked at `0.07496408700214947 rad`; correction reduced orientation error to `0.01331020038326052 rad`.
- Final lateral/axial/orientation errors were `0.00007743745606240142 m`, `0.002849578814716165 m`, and `0.012833770898787622 rad`; success true, truncated false.
- All 179 contact rows were empty. V2 validation, continuity, sensor reconstruction, hash binding, near-hole coverage, and axial tracking passed.
- Pilot remains ineligible/unapproved for training. `NEXT_PILOT = ROTATION_PITCH_READY`; pitch/training/resume were not started.
## 2026-09-05 — Peg F3 pitch command plan freeze

- `PEG_F3_ROTATION_PITCH_COMMAND_PLAN = PASS`.
- Frozen 179-command pitch plan binds the accepted standard trace and verified roll structure while preserving the exact 106-command prefix and 59-command insertion/success suffix.
- Four normal `action[4]=+1` commands plan a `0.06981317007977318 rad` local-pitch challenge; five `action[4]=-0.8` commands exactly cancel it, followed by five settle commands.
- Dedicated hash-bound verifier passed phase, 7D/finite/bounds, structure/source bindings, challenge range, and correction checks.
- `ROTATION_PITCH_RERECORD = READY`; simulator, recording, training, and resume were not started.
## 2026-09-05 — Peg F3 V2 rotation-pitch pilot

- `PEG_F3_V2_ROTATION_PITCH = FAIL`; recorder itself reached `COMPLETED_SUCCESS` after 179 frozen-plan transitions, but acceptance failed.
- Four pitch challenge actions were active and raised orientation error from `0.012340157970118806 rad` to `0.0514820779082296 rad`, below the frozen `0.06-0.08 rad` peak requirement.
- Correction reduced orientation error to `0.014141453782770853 rad`; near-hole, final success, no truncation, contact health, and axial tracking all passed.
- Final lateral/axial/orientation errors were `0.00007742949955367049 m`, `0.0028499364425848173 m`, and `0.012833750537051346 rad`.
- No retry was run. Pilot remains ineligible/unapproved for training. `FIVE_PILOT_AUDIT = BLOCKED`.
## 2026-09-06 — Peg F3 pitch challenge calibration and revised plan

- `PEG_F3_PITCH_CHALLENGE_CALIBRATION = PASS`; `_001` four-step response was finite, contact-free, and strictly monotonic.
- Old planned/realized peak: `0.06981317007977318 / 0.0514820779082296 rad`; response ratio `0.7374264461763124`.
- Minimum evidence-supported revision uses five `action[4]=+1.0` challenge steps, with expected realized peak `0.064352597385287 rad`.
- Seven correction steps (`6×-0.8`, `1×-0.2`) exactly cancel the five challenge units while preserving old correction actions as far as possible.
- Revised 182-command plan SHA256: `5e2aca80ac53564584faa2c3419e910f5c541d5351b31ac99fed258cb84e1415`; dedicated verifier PASS.
- `ROTATION_PITCH_RERECORD = READY`; no simulator, training, resume, promotion, or Five-Pilot Audit ran.

## 2026-09-06 — Peg F3 actor bounded shutdown cleanup

- `PEG_F3_ACTOR_SHUTDOWN_FIX = PASS`.
- Actor-side TrainerClient shutdown now signals receive/update loops, performs bounded joins, closes network resources in a defined order, and fails closed with durable thread-state evidence instead of hanging indefinitely.
- CPU/static shutdown regression passed normal stop, learner-stop handshake, repeated stop, already-dead thread, delayed-thread timeout, real Agentlace broadcast-client stop, and production integration checks.
- Learner-stop actor-like subprocess exited in `0.115235966 s`; delayed-thread timeout returned in `0.102985939 s`; residual processes: none.
- The prior 1000-update computation and checkpoint-1000 restore evidence were not rerun or modified. `SMOKE_FINALIZATION = READY_FOR_SHORT_SHUTDOWN_ACCEPTANCE`.

## 2026-09-06 — Peg F3 short shutdown acceptance

- `PEG_F3_SHORT_SHUTDOWN_ACCEPTANCE = FAIL`; `PEG_F3_V2_RLPD_SMOKE_FINAL = FAIL`; `TRAINING_READY = NO`.
- The one authorized short run completed 20 real GPU learner updates, 2,949 Actor steps, 21 network updates, and grew online replay from 91 to 2,941.
- Learner stop was observed. `TrainerClient.stop()` and broadcast join passed in `0.701639976 s`; Actor cleanup evidence was durable `1.509741702 s` after learner stop.
- Despite cleanup completion, the Actor process did not naturally exit within 30 seconds. The supervisor terminated it during fail-stop cleanup; no residual target processes remained.
- No retry, checkpoint write, resume, formal training, or 1000-update rerun occurred. Further readiness is blocked pending a separate post-cleanup process-exit investigation.

## 2026-09-06 — Peg F3 post-cleanup process-exit investigation

- `PEG_F3_POST_CLEANUP_EXIT_INVESTIGATION = FAIL`; exact blocking object remains `UNRESOLVED`.
- The single Actor-only, zero-learner/zero-update reproduction never reached cleanup instrumentation: `TrainerClient` failed to connect to the intentionally absent server and raised an uncaught exception.
- The Actor Python process nevertheless remained alive until the diagnostic supervisor's 180-second startup bound, showing that the exit hang is not uniquely dependent on completed `TrainerClient.stop()` or broadcast join.
- Required T+0/2/5/10 Python/native snapshots after cleanup were therefore not captured. No production code changed and no retry was run.
- Target residual processes: none after supervisor cleanup. `NEXT = BLOCKED` pending a separately authorized diagnostic design that can capture interpreter state on every exit path.

## 2026-09-06 — Peg F3 interpreter-exit dual-path sampling

- `PEG_F3_INTERPRETER_EXIT_DUAL_PATH_INVESTIGATION = PASS`.
- Exception path: top-level finally and atexit both entered/exited; no live non-daemon Python thread, multiprocessing child, asyncio task, or executor thread remained, but the process was still alive at T+10.
- Native T+10 evidence showed 108 threads, including `ZMQbg/Reaper` and `ZMQbg/IO/0`; the main TID waited in `poll_schedule_timeout.constprop.0`.
- Normal path: 20 learner updates and real cleanup completed, but control never returned to the outer top-level finally within 60 seconds.
- Installed Agentlace source confirms `TrainerClient.stop()` closes only broadcast/update paths, while request-side `ReqRepClient.__del__` performs `socket.close()` and unbounded `context.term()`. Both paths support request-side ZeroMQ finalization as the common blocker.
- `NEXT = INTERPRETER_EXIT_FIX_READY`; no production change, checkpoint, 1000-update rerun, resume, or formal training occurred.

## 2026-09-06 — Peg F3 request-side ZeroMQ finalization fix attempt

- `PEG_F3_ZMQ_FINALIZATION_FIX = FAIL`; `SHORT_SHUTDOWN_REACCEPTANCE = BLOCKED`.
- Project-owned `BoundedTrainerClient` now contains an explicit linger-0 request-socket/context cleanup path and constructor-exception cleanup, but the required regression did not validate it.
- The first no-server constructor child failed to exit naturally within its 10-second Gate. Per fail-stop, normal communication, repeated-stop, partial-init, and native-thread disappearance tests were `NOT_RUN`.
- The timed-out child was cleaned up and no test process or port remained. No retry, simulator, training, resume, dataset, or checkpoint operation occurred.

## 2026-09-06 — Peg F3 V2 autonomous pre-5K closure

- `PEG_F3_CANONICAL_LAUNCH_BINDING = PASS`; fixed WSL project root, Python, runtime, cwd and child-only `PYTHONPATH` work from project root, home and `/tmp`.
- `PEG_F3_REQREP_NORMAL_PATH_REVALIDATION = PASS`; 20/20 real GPU updates, 21 network updates, online replay 91→2941, finite losses, Actor natural exit in 1.659183331 s, and no residual processes.
- `PEG_F3_V2_RLPD_SMOKE_FINAL = PASS`; existing compute/data/replay/communication/checkpoint/restore evidence and new shutdown evidence all adjudicated PASS without rerunning 1000 updates.
- `PEG_F3_V2_FORMAL_5K_PREFLIGHT = PASS`; frozen V2 dataset/config, GPU, 32:32 sampling, 5K checkpoint plan, shutdown binding, isolated output and guarded launch command validated.
- `FORMAL_5K = READY`; `TRAINING_STARTED = false`. User authorization is required.

## 2026-09-06 — Peg F3 V2 formal Stage-1 0→5000

- `PEG_F3_V2_FORMAL_5K = PASS`; fresh V2 training completed exactly 5000/5000 updates and did not continue to 10K.
- GPU backend `gpu`; Demo buffer 1921; 32 Demo + 32 Online; online replay 91→36481; 5001 Actor network updates; all recorded losses finite.
- `checkpoint_1000`, `2000`, `3000`, `4000`, and `5000` are complete hash-bound bundles. Independent read-only restoration found 198/198 finite numeric leaves at every checkpoint.
- Learner and Actor stopped naturally; request socket/private context cleanup and environment close passed; no ZMQ or target process remained.
- `CHECKPOINT_SCREENING = READY`; formal 10K continuation is not authorized.
