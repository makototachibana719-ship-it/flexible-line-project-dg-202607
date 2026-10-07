# Change Log

## 2026-09-06 — F3 V2 formal standard +1

- Ran the validated standard formal launcher once after a clean dry-run preflight, creating a new independent episode and passive contact-audit directory.
- Read-only audit passed all 165 transitions for V2 schema, exact continuity, raw sensor reconstruction, phase/action stream binding, hashes, near-hole coverage, success, non-truncation, zero contact and axial tracking.
- No second standard or lateral/rotation episode was run; no training, resume, dataset build, production semantic, recorder, or frozen stream change.

## 2026-09-06 — F3 standard formal-demo invocation fix

- Added `scripts/run_peg_v2_standard_formal_demo.py`, an invocation-only launcher that feeds the frozen accepted standard command stream to the unchanged recorder wrapper using a direct subprocess stdin string instead of shell `python -c` quoting.
- The launcher fail-closes on stream/hash drift, invalid phase/action, wrong seed/max steps/type, existing or out-of-scope output paths, Windows paths, mismatched authorization flags, and extra CLI tokens.
- Tests: WSL `py_compile`, valid dry-run, parser/stream/scalar/path negative regressions, no-output check, and residual-process check PASS.
- No simulator, recording, collection retry, training, resume, canonical plan, pilot, or production semantic change.
- Rollback: remove the new launcher and restore the three `*_before_f3_standard_formal_invocation_fix_20260906.md` documentation backups.

## 2026-09-05 — F3 LateralPlus canonical plan path binding

- Added a read-only verifier hard-bound to the canonical project-relative WSL plan; no CLI path override, report copy or Windows path is accepted.
- Exact hash/count/content checks and five path regressions passed. The176-command artifact was not regenerated or modified.
- No simulator, recorder, `_002`, later pilot, training, or resume was started.

## 2026-09-05 — F3 V2 LateralPlus Rerecord-002 preflight stop

- Preflight used an absent WSL report-copy plan path. Post-failure read-only evidence confirmed the accepted plan remains intact at the earlier auto-closure path with the accepted hash and176 commands.
- No automatic path correction/retry: simulator, `_002`, recording, audit, later pilots, training and resume were not started.

## 2026-09-05 — F3 LateralPlus project-root bootstrap fix

- Reused the standard project root/sys.path bootstrap in the plan helper before its `training` import; no other script logic or planned action value changed.
- Local/WSL compile, direct-script imports, plan publication, and independent176-command phase/dimension/finite/bounds/order checks passed.
- Recording remains separately gated: no simulator, `_002`, later pilot, training, or resume was started.

## 2026-09-05 — F3 LateralPlus AutoClosure fail-stop

- Added a bounded plan builder whose intended correction commands use the existing `lateral_alignment` phase; recorder PHASES and all production semantics remain unchanged.
- Local Python compilation passed. The first WSL static-plan invocation failed before plan validation/publish because direct execution could not import `training`.
- Applied the explicit no-retry stop: no simulator, `_002` recording, audit, later pilot, training, or resume. Historical `_001` evidence remains untouched.

## 2026-09-05 — F3 V2 Remaining Pilots stopped on first pilot

- Parameterized only the passive contact-audit launcher so it could pass one explicitly selected frozen episode type to the unchanged V2 recorder; syntax and preflight passed.
- Started lateral +Y exactly once in new exclusive Demo/contact directories. It recorded109 transitions, established4.50795mm lateral challenge at37.9950mm axial distance, and observed zero contacts.
- The intended correction command used unsupported phase `lateral_correction`; recorder validation raised before dispatch, wrote INTERRUPTED/ineligible status, and exited. Sequential fail-stop prevented lateral -Y/roll/pitch; no retry, promotion, training, or resume.
- Rollback: restore the launcher to its former hard-coded `standard` episode type and revert the three dated status documents. Preserve the failed pilot evidence; it must not be overwritten or promoted.

## 2026-09-05 — Post-Exclusion Pilot Audit

- Offline-only comparison established fixed-reset seed0, exact goal, V2 contract, runtime/config and control/reward/success/action binding parity between the failed pilot and bounded diagnostic.
- All126 seven-dimensional actions match exactly. Full file-manifest comparison isolates the loaded semantic change to the authorized Link01 exclusion; its added backup file is not model-loaded.
- Historical axial realization0.00882149mm became12.0825768mm for about12mm requested. Standard-only V2 rerecord is READY for separate authorization; no simulator, recording, training, or production change occurred.

## 2026-09-05 — Axial Diagnostic After Telemetry Hardening

- Ran exactly one authorized fixed-reset diagnostic in a new exclusive directory; repaired reset boundary and unchanged contact guard passed, then126 frozen actions ran only through `observer.step`.
- 126/126 telemetry rows are complete. IK remained `G_TOL`, minimum joint margin was4.033rad, no actuator clipped, no contacts appeared, and insertion Jacobian condition number stayed at or below8.14.
- Requested12.0000005mm and realized12.0825768mm. All126 actions exactly equal the failed pre-exclusion pilot, which realized0.00882149mm; therefore Link01 exclusion causal contribution is supported, with no claim of sole/general sufficiency.
- Diagnostic-only runner telemetry-manifest binding was advanced to the already hardened observer. No production control/exclusion/guard/Demo/checkpoint/training change and no rerun.

## 2026-09-05 — AxialTelemetry Durable Failure Evidence

- Hardened only the opt-in passive observer: frozen RAW_STEP through PUBLISH stage classification; exact exception, phase, availability and missing-field failure artifacts; no fabricated telemetry.
- Failure artifacts use strict NumPy-safe normalization, serialize-before-write, exclusive temp, file fsync, atomic no-clobber publication and POSIX directory fsync. Self-persistence failures explicitly remain non-durable and do not mask original environment exceptions.
- Normal telemetry schema, single original step dispatch, controller output and solver/Jacobian counts pass 44 CPU/static tests. Production bindings/controller/IK/exclusion/guard/reward/success/action/V2 semantics unchanged; no simulator, Demo or training.

## 2026-09-05 — Post-Exclusion Axial Diagnostic Final Retry

- Ran the one authorized fixed-reset diagnostic in a new exclusive directory. Durable live reset, repaired semantic invariant gate, unchanged zero-contact guard, V2/exclusion/binding/process preflight all passed.
- Dispatched exactly one setup action through `observer.step`, then failed closed on `observer.last.complete=false`. Cartesian evidence is durable; passive `steps.jsonl` is empty, so the exact observer post-processing/publication exception is not recoverable and no axial causal claim is made.
- No axial phase/nonresponse reproduction, contact, Link01 reappearance, rerun, Demo, training, resume, or production behavior change. Next minimal scope is passive observer failure-artifact hardening with CPU/static noninvasiveness proof.

## 2026-09-05 — Axial Diagnostic Reset Contract Fix

- Retry runner now durably saves the live reset before compatibility and unchanged safety guard; persistence failures block both guard and action.
- Replaced full 20D pre/post-exclusion equality with fixed V2/schema/finite/reset/goal/binding/exclusion/action invariants and frozen EE tolerances. Force/torque indices7:13 are saved as old/new/delta but not equality-gated.
- 12 CPU tests PASS in Windows/WSL plus accepted historical fixture PASS. Production/exclusion/guard/telemetry/controller/IK/reward/success/action unchanged; no simulator, Demo or training.

## 2026-09-05 — Post-Exclusion Axial Diagnostic Retry

- Added an isolated one-shot diagnostic runner and invoked it exactly once. Source/generated exclusion, XML compile, V2 binding, telemetry hash and process preflight passed.
- The run performed fixed reset then stopped before action1 because a new compatibility assertion required full 20D equality to the pre-exclusion failed pilot. Zero actions/telemetry steps; no axial root-cause evidence.
- Offline accepted-reset comparison attributes the gate failure to expected reset-state/sensor differences across the collision-model change, not to a proven production fault. No automatic rerun or production/Demo/training change.

## 2026-09-05 — Link01 Adjacent Collision Exclusion Fix

- Added exactly one source MJCF exclusion for adjacent bodies `base_link` and `shoulder_Link`; existing assembly generation propagated only that pair.
- Compile-only load PASS. Accepted fixed reset produced zero total/link0-link1/peg-hole/robot-fixture contacts; unchanged guard did not trigger; zero actions.
- Regression confirmed controller, IK, sensor, reward, success, action, V2 observation, Demo, and checkpoint content unchanged. No axial diagnostic, Demo recording, resume, or training.

## 2026-09-05 — Geometry Fix Review

- Offline STL/hash/transform review found persistent link0/link1 broad-phase overlap across six shoulder samples; exact saved reset contact remains the only triangle/contact proof.
- Official Aubo URDF geometry structure matches the local paths/transforms, and official SRDF disables all six adjacent arm pairs. Local MJCF has no Aubo exclusions; classification is `LIKELY_MISSING_EXCLUSION`, with other-pair exact overlap unresolved.
- Added static review script/evidence/report and updated status documents after backups. No XML/STL/exclude/guard/simulator/controller/Demo/training change.

## 2026-09-05 — Link01 Collision Intent Audit

- Static-only XML/generator/exclude review: Aubo link0-link1 is direct parent-child, not excluded, collision enabled; all six adjacent arm pairs match ALLOW_ALL configuration.
- Assembler/splicer preserve `<contact>` when present; Aubo source has none. Semantic benignity remains UNRESOLVED; geometry review recommended.
- Added audit script/evidence only. No simulator, guard, geometry, Demo or training change.

## 2026-09-05 — BaselineContactReview Retry

- One new-directory fixed reset; complete initial snapshot and exact guard event durably saved, zero actions.
- Trigger is robot geom49 base_link/link0 vs geom52 shoulder_Link/link1, 4 contacts at -6.49972274584152mm. Not peg-hole/fixture; benignity UNRESOLVED.
- Production hashes unchanged; no automatic guard change or axial rerun.

## 2026-09-05 — DiagnosticSerializationFix

- Diagnostic-only NumPy normalization, strict unknown/NaN/Inf rejection and serialize-before-publish, file/directory fsync, atomic no-clobber hard-link publication.
- CPU19 PASS; production/telemetry hashes and guard/action AST unchanged. No simulator/reset/Demo/training. Historical evidence untouched.

## 2026-09-05 — BaselineContactReview

- One reset-only collection attempted; no action calls. Failed on numpy.bool_ serialization in initial joint snapshot before guard evaluation. Partial evidence preserved unchanged.
- Added standalone reset-only runner and failure report; production hashes unchanged. No retry, guard change, Demo or training.

## 2026-09-05 — AxialDiagnostic InitEvidenceFix

- Backed up and changed diagnostic runner only: durable exclusive initial joint/actuator/contact snapshot before unchanged distance < -0.005 m guard, separate exact triggering event on STOP.
- Added CPU tests/verifier; 7 PASS, production and telemetry hashes unchanged, action loop AST unchanged.
- No simulator, Demo, training, or historical evidence reconstruction. Root cause UNRESOLVED.

## 2026-09-05 — One-shot AxialBoundedDiagnostic

- Added standalone bounded diagnostic runner; syntax passed; one actual initialization/reset,0 observer.step calls due initial penetration guard.
- Execution FAIL; initial contact list missing from evidence because guard preceded persistence. Explicitly disclosed; no geometry cause claimed. No retry or production changes.
- Evidence under results/diagnostics/peg/f3_axial_bounded_20260905. Status documents backed up/updated; proposed runner-only persistence repair awaits direction.

## 2026-09-05 — Passive AxialTelemetry

- Added independent training/peg_insertion/axial_telemetry.py, CPU test and verification scripts; default production paths unchanged. No controller/IK monkeypatch or production source edit.
- Copies existing function inputs/returns via scoped thread profiler; records joints/ctrl/named contacts before/after step. SVD on copied existing Jacobians only, no extra IK/FK/physics calls.
-18 CPU tests PASS including actual production solve/dispatch AST with algebraic residual fixtures, real installed least_squares, off/on exact output and invocation-count parity, exception fallback, profiler restoration and exclusive output. Reports binding confirms protected source/spec/failed data unchanged.
- No simulator diagnostic or recording executed. Status documents backed up; historical root cause UNRESOLVED.

## 2026-09-05 — Offline AxialTrackingAudit

- Added analysis-only script and step100–126 JSON/static XML inventory/input hashes/results/report. Production code and raw Demo unchanged. Status documents backed up before update.
- Existing WSL Python executed audit_peg_axial_tracking.py only. Target reconstruction and hash checks PASS; historical IK/joint/contact evidence unavailable. No simulator/IK invocation, recording or training.

## 2026-09-05 — F3 V2 DemoPilot stopped

- Used unchanged recorder with explicit recording flags, agent-selected state-feedback actions. Standard stopped after126 steps due axial target/realized divergence; preserved entire setup and failed trace. No other pilot attempted.
- Added read-only `scripts/audit_peg_v2_pilot_20260905.py`, evidence/result/report; updated status/task documents after backups. No production source/config/runtime/controller/reward/success/Demo legacy/checkpoint edits.
- Audit command with existing WSL Python: PASS data integrity, FAIL pilot acceptance. No retry or promotion. Data and raw recorder log retained; no rollback/deletion needed.

## 2026-09-04 — F3 V2 RecorderReady (CPU/static only)

- Added independent V2 recorder module, console entry point,27 synthetic tests, and evidence verifier. V2-only/raw sensor reconstruction/phase/continuity/hash checks fail closed; interrupted data quarantined; all output exclusive.
- Default command performs preflight only. No existing recorder, task reward/success/controller/action, Demo or checkpoint bytes changed. No real episode recorded; no training.
- Evidence: `results/validation/peg/f3_v2_recorder_ready_20260904/REPORT.md`.

## 2026-09-04 — F3 DataReview (offline only)

- Added per-episode/near-hole V1 statistics, 5-pilot V2 protocol, acceptance spec and hash freeze.20/20 legacy success,5512 zero-rotation transitions; active rotational correction coverage inadequate, not invalid zero-action demos.
- PKL/NPZ, finite/continuity/reward/success/hash tests PASS. No simulation, recording, training, or production code changes. Status docs backed up `.before_f3_datareview_20260904`.
- V2 collection BLOCKED pending versioned recorder tooling; report `results/validation/peg/f3_data_review_20260904/REPORT.md`. No automatic pilot start.

## 2026-09-04 — F3 SensorFix + observation versioning

- Replaced sensor-id slicing with shared address/dim3 validation. Added V1/V2 metadata/config/hash registry, fail-closed training/restore/preflight/evaluator checks; explicit diagnostic mode marked non-skill.
- CPU tests and legacy file hashes PASS; no physics/policy execution. Existing Demo lacks raw Ty/Tz, reconstruction NO. 7 retained formal model bundles unchanged.
- Backups `.before_f3_sensorfix_20260904`; no checkpoint or old Gate rewritten. Rollback requires restoring the full backed-up version-bound code/config set together, not silently mixing sensor schema with legacy artifacts. Next task needs separate V2-data authorization. Report `results/validation/peg/f3_sensorfix_20260904/REPORT.md`.

## 2026-09-04 — F3 contract audit (no production fix)

- Added `peg_f3_contract_audit.py`, `analyze_peg_f3_contract.py`, read-only Demo and one9000 episode evidence. Action mapping PASS; torque ID/address bug confirmed; rotation actions allzero across5512 transitions. Tracking and policy-target deviations separately reported.
- Tests: existing inference guards, source/artifact hashes, per-step reproduction and offline action reconstruction. No training/resume/cloud/checkpoint/controller/reward edits. Source files unchanged; docs backed up `.before_f3_contract_20260904`.
- Reason: distinguish semantic bugs, tracking failures and missing data excitation. Next minimal fix is recommendation only. Preserve all history; report `results/validation/peg/f3_contract_audit_20260904/REPORT.md`.

## 2026-09-04 — F3 fixed-scene diagnostic

- Added passive trace recorder/diagnostic worker/supervisor and offline analyzer. Executed only5000 and9000,one500-step episode each; no training/environment contract change. Guards PASS, traces reproduce screening.
- Generated CSV/JSONL, full critical-state JSON and8 SVG figures (six requested categories). Joint best ratios26.2649/13.3361; mixed A/B/C, saturation E auxiliary and force D unproven. Offline Matplotlib missing, used dependency-free SVG with existing workspace Python, no package upgrades/re-runs.
- Docs backed up `.before_f3_diagnostic_20260904`. Next diagnostic recommendations only, no fix implemented. Evidence `results/validation/peg/f3_diagnostic_20260904/REPORT.md`; preserve all old artifacts.

## 2026-09-04 — F3 fixed-reset Stage2 screening

- Added parameterized Stage1-equivalent inference worker and bounded screening supervisor. AST parity check and ranking protocol frozen before runs. Existing evaluator/environment/algorithm unchanged; docs backed up `.before_f3_stage2_20260904`.
- Command: serl_peg_gpu Python scripts/run_peg_f3_stage2.py.5x10 episodes completed, all0 successes, all500-step truncation. Candidate hashes and state guards PASS, no retries or residual screening process.
- Diagnostic rank9000,5000,7000,6000,8000. CategoryB descriptive improvement relative5000, non-monotonic; quality review recommended, final comparison blocked. No training/resume/cloud mutation/random reset/50-episode evaluation. Evidence `results/validation/peg/f3_stage2_20260904/REPORT.md`. Preserve evidence; rollback only new scripts/docs via backups if separately requested.

## 2026-09-04 — F3 evaluator Stage1

- Added inference-only smoke metrics/guard script and600s one-shot supervisor; original evaluator, training, reward/success source unchanged. Tracking docs backed up `.before_f3_stage1_20260904`.
- Ran serl_peg_gpu Python scripts/run_peg_f3_stage1_once.py once. Preflight/load/smoke PASS;10 episodes,0 successes, all500-step truncation. Identical seeds0–9 trajectory aggregates with default fixed reset; report explicitly limits conclusions.
- Checkpoint/replay hashes unchanged; no training/cloud modification/other checkpoint evaluation. Screening READY, not executed. Evidence `results/validation/peg/f3_stage1_20260904/REPORT.md`; rollback only new scripts/docs via backups if separately requested, preserve evidence.

## 2026-09-04 — F2-RetentionFix

- Backed up resume.py/status/tasks/change_log before editing. Added bundle_retention.py and wrapped unchanged bundle serialization in isolated staging; numeric keep20 now counts bundles, retirement removes all3 exact members. Journal/rollback rejects interrupted or incomplete bundles.
- Command: `python scripts/audit_peg_retention_fix.py` in WSL serl_peg_gpu.62 CPU tests PASS (10 retention,5 snapshot,37 policy,10 exit); syntax PASS;88 protected-file hashes unchanged. No GPU training/resume/evaluation/cloud operations.
- Existing missing/partial formal bundles and raw F2 FAIL/v3/v4 remain unchanged. Evaluation BLOCKED; authorized repair changes source hashes and needs separate future source-bound approval. Rollback backups: `.before_f2_retention_20260904`; do not automatically revert to known retention bug. Evidence: `results/validation/peg/f2_retention_20260904/REPORT.md`.

## 2026-09-04 — F2-ExitFix / retention discovery

- Backed up changed entry/policy/test fixture/docs `.before_f2_exitfix_20260904`. Replaced only normal/SystemExit exception classification for W&B finish;83 CPU regressions and syntax PASS; hard timeout still exits124 without success finalization.
- New v4 source-bound Gate generated false/blocked. Preserved v3/raw F2 FAIL/cloudfailed. Re-audit discovered training-time Flax cleanup includes sidecars in keep20:0/1000 absent,2000 partial,3000–9000 complete. Historical save-time PASS is not final-retention PASS. No checkpoint cleanup/repair performed in this task.
- No training/resume/cloud change/evaluation. Report: `results/validation/peg/f2_exitfix_20260904_review/REPORT.md`. Further retention fix requires separate scope; no silent PASS adjudication.

## 2026-09-04 — F2 formal execution, final-state warning/failure

- One formal learner408 and actor684 launched after v3/role checks; visible execution/log/GPU terminals opened. No frozen source/config/runtime/Demo/Gate edit. Added only execution evidence scripts/report; tracking docs backed up `.before_f2_run_20260904`.
-10000/10000 updates, finite/RNG checks,10 periodic bundles PASS;0 retries/resumes/timeouts. Learner exit0; supervisor stopped associated actor; no training residuals.
- Run e6fea100 has1000 unique matching cloud loss rows but statefailed: SDK receivedexit1 because existing finally regards successful absl SystemExit as error. OverallFAIL preserved; no cloud repair or source change. No evaluation. Details/evidence: `results/validation/peg/f2_run_20260904/REPORT.md`.

## 2026-09-04 — F2-Monitor hardening, no training

- Backups `.before_f2_monitor_20260904` on Windows/WSL. Formal-only wrapper/local monitor,120s watchdog, required finite-loss/RNG checks; reuse existing UpdateObserver30/60/110s snapshots. Original SAC/source math, argument forwarding, per-leaf implementation, runtime/config/Demo, W&B10/max10000/checkpoint1000 unchanged.
- Policy now requires new source-bound v3 monitor hash/flags/evidence; v2 historical PASS preserved but stale. No checkpoint overwrite.
- `scripts/build_peg_f2_gate.py`:73 CPU tests, Python/shell syntax, actual fresh preflight PASS; v3 READY_TO_START. No GPU training/resume/W&B run. Scope/limitations/rollback: `docs/peg_f2_monitor_20260904.md`. Separate execution authorization required.

## 2026-09-04 — F1 formal readiness hardening, no training

- Backups `.before_f1_20260904` for entry/policy/resume/preflight/four launchers/delivered progress plumbing/tracking docs. Added shared per-leaf policy, four role-aware preflight/locks/session/run binding and source-bound formal config guard. Native sync cause remains UNRESOLVED.
- Delivered code only adds global learner/actor start indices; logger/save offsets removed to avoid double counting. Actual save predicate now uses global index; formal period1000/max_steps10000 unchanged. New formal bundle model/replay/source/runtime/run checks; no existing checkpoint modified.
- Created exact three-line formal runtime file and new source/evidence-bound formal_readiness_v2.json, leaving old Gate untouched.47 CPU tests and Python/shell syntax PASS; actual fresh clean outputs PASS; actor/resume cases tested with CPU fixtures, not live workers.
- Readiness PASS / READY_TO_START; no formal learner/actor, GPU training, checkpoint30 resume, dependency or cloud changes. Full scope and rollback in `docs/peg_f1_hardening_20260904.md`.

## 2026-09-04 — F0-R production checkpoint branch resume acceptance

- Backed up entry/tracking docs `.before_f0_r_branch_20260904`. Added acceptance-only branch permission/policy, exact source hash/config binding, isolated outputs/new W&B provenance, state/replay restore assertions and real global26–30 logging. Actor scheduling continues from1261. Per-leaf sync implementation, original learner/update, runtime, dependencies and formal Gate unchanged.
- Added branch role/supervisor shells, one-shot verifier,11 CPU policy tests and supplemental offline audit.11+8+6 tests PASS. One learner/actor launch only;5/5 resumed updates PASS, first5.797s then0.019–0.034s; new run2243685c finished, steps26–30 unique/matching.
- Preserved relative periodic save at new checkpoint26; final bounded post-loop checkpoint30 saved, both snapshot counters1651. Live counter later1921 during finish is not substituted for frozen snapshot.198 restored state leaves exact; final198 finite; internal counters52→62. Source bundle/run and old Gate preserved; no retry/restart/formal/env creation.
- Added `docs/peg_f0_r_branch_20260904.md`; exact Python resume accepted only as NEW_BRANCH_RUN, not same-run history continuity. General formal launcher hardening still pending. Rollback from named backups only if authorized; retain evidence.

## 2026-09-04 — F0 verifier dotted schema correction and existing-B re-audit

- Backed up verifier `.before_schema_fix_20260904`; changed three slash loss keys to dots, removed local dot-to-slash conversion. Added optional independent evidence destination; original outputs retained.
- Added8 schema tests covering complete50, missing step/field, step-only, nonfinite, loss mismatch, conflicting duplicate rejection and explicit identical-duplicate warning; all PASS, existing6 policy tests PASS.
- New audit-only runner checked existing hr88a6u1/50 updates/bundles0,25. PASS, finished run, no duplicate warnings, losses match; protected bundle/raw FAIL/Gate hashes unchanged. No training/resume/cloud mutation/sync modification/formal runtime. Current-source hashes explicitly labeled audit-time, not historic attestation.
- Updated tracking docs after `.before_verifier_fix_20260904` backups. Acceptance result PASS_WITH_PER_LEAF_SYNC is scoped to B; formal remains BLOCKED. Evidence `results/diagnostics/peg_f0_verifier_fix_20260904/reaudit.json`.

## 2026-09-04 — F0-Sync single-variable B arm PASS, overall verifier FAIL retained

- Backed up entry/docs before changes. Added acceptance-observed per-leaf sync flag and one-shot runner. Both learner full-tree sync points use per-leaf waits; global JAX/actor/update implementation/placement/RNG/order untouched.
- Syntax and6 policy tests PASS; one B run50/50 with update1 sync0.028s, steady0.011–0.029s. Offline counter/hash/model-finite/RNG audit PASS for checkpoints0/25.
- Raw full verifier FAIL retained: online runfinished but filtered query rows=[]; no repeated query or cloud mutation. Formal Gate untouched; no formal env/resume/retry. Report `docs/peg_f0_sync_20260904.md`. Rollback from `.before_f0_sync_20260904` only if authorized; preserve evidence.

## 2026-09-04 — F0-Observe, synchronization stall localized

- Backed up entry/verifier/tracking docs `.before_f0_observe_20260904`; added metadata-only observer and separate one-shot runner. Verifier can suppress Gate writing; default F0 behavior retained. No algorithm/runtime placement/order changes.
- Six policy tests and syntax checks PASS. One instrumented attempt:1/50, update1 dispatch4.430s, XLA compilation finished3.111s, no full-sync completion, watchdog120s. Stack: batched_block_until_ready; GPU8% snapshot; root cause unresolved.
- Saved independent evidence and report `docs/peg_f0_observe_20260904.md`; no retry/resume or formal runtime file. Prior failed Gate not overwritten. Rollback code/docs from named backups if authorized; preserve all new/old evidence.

## 2026-09-04 — F0 formal entry hardening, one exact-path acceptance FAIL

- Backed up existing changed training entry, four launchers and tracking docs as `.before_f0_20260904`. Added runtime policy, source-hash-bound fail-closed gate, preflight, acceptance supervisor/verifier and six CPU policy/history tests. No algorithm/config/dependency changes.
- Acceptance uses the production training module and delivered learner, real actor, normal RNG, online W&B, normal save path; bounded settings are 50 updates and save period 25. Production cadence remains unchanged. Six unit tests, syntax checks and runtime preflight passed; formal launch without approved gate was rejected.
- Exactly one run: update 0 completed in 5.222 s; checkpoint_0 saved; update 1 hit 120 s timeout. Learner exit124, actor terminated by owned-process supervisor, no residual processes. `PEG_FORMAL_EXACT_PATH_ACCEPTANCE_50=FAIL`, gate formal_ready=false. No retry/resume/formal or formal runtime file.
- Preserved raw timeout record's inherited `profile=smoke` label; actual CLI/log/metadata say acceptance. Post-run static review also found fresh actor/resume preflight output-emptiness checks overly restrictive. These are open hardening issues, not approved formal paths. Report: `docs/peg_f0_20260904.md`.

## 2026-09-04 — I4 one-shot resume lifecycle, historical W&B duplicate warning

- Added isolated `scripts/peg_i4_resume_validation.py`: source-locked repaired checkpoint19, exact full state/replay restore validation, real actor, original five updates20–24, same W&B resume, isolated terminal checkpoint24, strict audit. No production code/cadence/dependency/algorithm changes.
- Syntax passed; exactly one supervised experiment completed five updates, learner/actor exit0, finite losses and counters2102=2223−121, source hashes unchanged. Raw verification exit1/FAIL is retained: W&B history query returned45 rows due to identical old0–19 duplicates.
- Added pure offline `scripts/audit_peg_i4_cached_history.py`; cached source/current comparison proves old duplicate values identical, new20–24 unique/matching, run finished. No network query or training retry. Result `PEG_I4_RESUME_LIFECYCLE_5=PASS_WITH_WANDB_HISTORY_WARNING`; formal BLOCKED.
- Report `docs/peg_i4_resume_20260904.md`; light evidence mirrored from WSL. Tracking-doc backups `.before_i4_resume_20260904`; revert scripts/docs only if needed, preserve raw and new bundles. Old failed artifacts never loaded; new checkpoint24 not used for another resume.

## 2026-09-04 — I3-Fix snapshot/counter repair and one bounded acceptance PASS

- Modified only production source `training/peg_insertion/resume.py` after `.before_i3_snapshot_fix_20260904` backup. Removed stale live-counter overwrite; installed shared RLock before buffer concurrency; captured detached replay arrays, size/index/RNG/count under that lock before model/disk I/O. Metadata uses captured values.
- Added `scripts/test_peg_snapshot_consistency.py`: 5/5 actual-datastore CPU regressions PASS (atomic insert/count, sample/RNG lock, detached/wrapped snapshot, concurrent inserts during model I/O, collision).
- Only after regressions, ran new `scripts/peg_i3_fixed_acceptance.py --role supervise` once. 20/20, finite/normal RNG, live communication, W&B `44yrxufd` finished, cadence 0/19 unchanged, no timeout/retry/resume/formal.
- Stricter counter audit: checkpoint_0 812−121=691; checkpoint_19 1292−121=1171, both exact. `PEG_I3_CHECKPOINT_IO_20=PASS` for new run only. Old FAIL evidence and checkpoint hashes preserved.
- Report: `docs/peg_i3_snapshot_fix_20260904.md`. Tracking docs backed up with `.before_i3_snapshot_fix_20260904`. No dependency/algorithm/formal launcher changes. Formal remains BLOCKED; resume is not claimed as validated.

## 2026-09-04 — Peg I3 normal checkpoint I/O, final audit FAIL

- Added isolated `scripts/peg_i3_checkpoint_integration.py`, reusing I2 and the unmodified normal save_bundle at periodic steps 0/19. One fresh run completed 20/20 updates, live uploads and finished W&B run `b6ajsw8v`; no stall/retry/resume/formal.
- Saving took 0.199/0.189 s. Both bundles decode, have 198 finite leaves, correct update RNG and matching metadata/sidecar hashes. Initial script emitted PASS.
- Independent `scripts/verify_peg_i3_counter_consistency.py` found 30 then 60 live actor-step undercount. Final `PEG_I3_CHECKPOINT_IO_20=FAIL`; raw outputs preserved and superseded by the independent audit. Normal save_bundle resets live counter from a stale argument after model I/O; 30 live transitions arrived during each save.
- No save-function repair or rerun. Added report `docs/peg_i3_checkpoint_integration_20260904.md` and new I3 evidence; updated tracking docs after `.before_peg_i3_20260904` backups. Original checkpoints/functions unchanged. Formal BLOCKED; I3 artifacts not validated for resume.

## 2026-09-04 — Peg I2 online W&B integration PASS

- Original behavior: I1 had live actor communication but no W&B. Added isolated `scripts/peg_i2_wandb_integration.py`, reusing unchanged I1/R2 and adding online init, per-update metrics, finish and read-only cloud verification only.
- Executed `python scripts/peg_i2_wandb_integration.py --role supervise` once in locked WSL GPU environment. 20/20 updates, finite losses, normal RNG, 1,201 accepted live transitions and 21 received networks. New W&B run `21p57czu` finished; cloud steps 0–19/losses match local.
- Update 0/1: 5.554/4.357 s; indices 2–19: 0.012–0.029 s. Initial/post-initial compilation 1/1. No stall. W&B init 3.585 s, mean log enqueue 0.000493 s, finish 2.591 s.
- No checkpoint saving/resume, retry, dependencies/algorithms/formal launcher changes or formal runtime. No residual trainer/wandb-core. Full candidate FAIL / formal BLOCKED unchanged.
- Added `docs/peg_i2_wandb_integration_20260904.md` and new I2 evidence. Tracking docs backed up as `.before_peg_i2_20260904` before edits; these provide tracking rollback without touching experiment results.

## 2026-09-04 — Peg I1 live actor communication, one-shot PASS

- Added isolated `scripts/peg_i1_live_communication.py`, reusing the unmodified R2 update runner and adding real Agentlace actor uploads/parameter broadcasts. Formal launcher and algorithm remain untouched.
- One fresh learner completed 20/20 original updates; actor took 1,140 steps, learner accepted 1,111 transitions, actor received 21 parameter broadcasts. 32 uploads overlapped learner updates, including 4 during indices 2–19.
- Index 0/1: 5.549/4.626 s, one post-initial recompile. Indices 2–19: 0.015–0.033 s; losses finite; normal RNG chain; both processes exit 0; no residual trainer.
- No W&B, model checkpoint save/restore, dependency upgrade, fixed RNG, retry, or formal runtime. R2 replay snapshot was read and appended with live data; candidate bundle integrity hashes unchanged.
- Added report `docs/peg_i1_live_communication_20260904.md` and new diagnostic evidence `results/diagnostics/peg_i1_live_20260904_114647/`; updated status/tasks only after adjacent `.before_peg_i1_20260904` backups.
- Result: `PEG_I1_LIVE_COMMUNICATION_20=PASS`. This does not prove absence of rare communication-related stalls; full candidate FAIL / formal BLOCKED remain unchanged. No next integration rung automatically started.

## 2026-08-13 — Project state management initialization

### Date

2026-08-13

### File

- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`
- `docs/architecture/current_architecture.md`
- `docs/competition/deliverable_checklist.md`

### Original behavior

The project contained `AGENTS.md` but did not contain the requested state-management and documentation files.

### Modification

Created the initial project status, prioritized task register, change log, current architecture boundary, and competition deliverable checklist.

All three real process packages and their interfaces, checkpoints, demos, rewards, success conditions, and environments are recorded as PENDING, UNVERIFIED, or BLOCKED as appropriate.

### Reason

Establish truthful PHASE 1 project tracking and integration preparation without assuming unavailable process implementations.

### Test command

```bash
test -f PROJECT_STATUS.md \
  -a -f TASKS.md \
  -a -f docs/change_log.md \
  -a -f docs/architecture/current_architecture.md \
  -a -f docs/competition/deliverable_checklist.md

rg -n "PHASE 1 — Integration Infrastructure Preparation|Cable Routing: PENDING|Object Grasping: PENDING|Peg Insertion: PENDING|package not received|UNVERIFIED" \
  PROJECT_STATUS.md TASKS.md docs/architecture/current_architecture.md docs/competition/deliverable_checklist.md
```

### Test result

PASS — all five files exist, and the required phase, package status, blocked reason, and UNVERIFIED wording were found. The workspace is not a Git repository, so Git status was unavailable and was not used as validation evidence.

### Rollback method

Remove only the five initialization documents listed above if this initialization must be reverted. No source code, external repository, checkpoint, demo, model, or experiment artifact was modified.

## 2026-08-15 — Package intake and static compatibility audit

### Date

2026-08-15

### File

- `external/packages/cable_routing/` (created by extraction)
- `external/packages/object_grasping/` (created by extraction)
- `external/packages/peg_insertion/` (created by extraction)
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/package_inventory.md`
- `docs/package_audit_cable_routing.md`
- `docs/package_audit_object_grasping.md`
- `docs/package_audit_peg_insertion.md`
- `docs/interface_compatibility_report.md`
- `docs/integration_risk_report.md`
- `docs/process_comparison.md`
- `docs/interface_matrix.md`
- `docs/integration_plan.md`

### Original behavior

Project tracking recorded all three packages as not received. No isolated extracted copies or package/interface audit documents existed.

### Modification

Inventoried and safely extracted three incoming archives, then documented static package contents, demo/checkpoint status, interfaces, compatibility, and risks. Updated project/task state. Original archives and all process source/demo/checkpoint files were left unchanged.

Backups: `PROJECT_STATUS_backup_20260815_package_audit.md`, `TASKS_backup_20260815_package_audit.md`, and `docs/change_log_backup_20260815_package_audit.md`.

### Reason

Complete MILESTONE 0 package intake and compatibility audit using only truthful static evidence before any real Skill integration.

### Test command

```bash
unzip -t external/incoming/object_grasping_package.zip
unzip -t external/incoming/peg_insertion_package.zip
tar -tzf external/incoming/cable_routing_package.gz
sha256sum external/incoming/*
find external/packages -type f
rg -n "PRESENT|MISSING|UNVERIFIED|INCOMPATIBLE|VALIDATED" PROJECT_STATUS.md TASKS.md docs/package_*.md docs/interface_*.md docs/integration_*.md
```

### Test result

PASS for archive integrity, safe extraction, inventory evidence, and static documentation. Runtime environments, demos, checkpoints, policies, and evaluation remain UNVERIFIED or MISSING as recorded.

### Rollback method

Restore the three backed-up tracking documents and remove only the newly generated audit documents and isolated `external/packages/{cable_routing,object_grasping,peg_insertion}` extraction copies. Never remove or alter `external/incoming/` originals.

## 2026-08-15 — Process information corrections and task reprioritization

### Date

2026-08-15

### File

- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/package_audit_cable_routing.md`
- `docs/package_audit_object_grasping.md`
- `docs/process_comparison.md`
- `docs/interface_matrix.md`
- `docs/integration_plan.md`
- `docs/change_log.md`

### Original behavior

Cable Routing's missing demo was recorded without a confirmed acquisition owner. Object Grasping was described as ordinary DrQ based only on package naming and imports. Follow-up validation tasks were grouped as blocked single-skill validation.

### Modification

Recorded human-confirmed Cable demo scope: recorder MISSING, data MISSING, project-owned acquisition TODO, not awaiting resend. Recorded Object Grasping as DPRL at project level while marking the conflict with its delivered `DrQAgent`/`make_drq_agent` and 50/50 demo/online replay implementation as NEEDS_VERIFICATION. Reprioritized the three environment smoke validations to P0 and the specified recorder/demo/algorithm/checkpoint/schema tasks to P1.

Created one-for-one backups with suffix `_backup_20260815_process_corrections.md` for all eight modified files.

### Reason

Incorporate verified team knowledge without overriding contradictory package evidence or inventing a DPRL implementation mapping.

### Test command

```bash
rg -n -i "DPRL|DrQAgent|make_drq_agent|demo_buffer|replay_buffer|checkpoint" external/packages/object_grasping/object_grasp_package
rg -n "Demo Acquisition|NEEDS_VERIFICATION|environment smoke validation|DPRL" PROJECT_STATUS.md TASKS.md docs/package_audit_cable_routing.md docs/package_audit_object_grasping.md docs/process_comparison.md docs/interface_matrix.md docs/integration_plan.md
```

### Test result

PASS — Cable acquisition is TODO and not described as awaiting resend; Object algorithm is DPRL by human confirmation with code mapping NEEDS_VERIFICATION; requested P0/P1 tasks are present. No process source, SERL, mujoco_env, demo, checkpoint, or AGENTS.md file was modified.

### Rollback method

Restore the eight corresponding `_backup_20260815_process_corrections.md` files. No source or data rollback is required.

## 2026-08-15 — MILESTONE 1A isolated environment smoke attempt

### Date

2026-08-15

### File

- `evaluation/smoke/__init__.py`
- `evaluation/smoke/smoke_cable_routing.py`
- `evaluation/smoke/smoke_object_grasping.py`
- `evaluation/smoke/smoke_peg_insertion.py`
- `docs/runtime_validation_cable_routing.md`
- `docs/runtime_validation_object_grasping.md`
- `docs/runtime_validation_peg_insertion.md`
- `docs/runtime_validation_summary.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Three P0 environment smoke validations were TODO and no external lifecycle harness or runtime evidence existed.

### Modification

Created read-only external harnesses and executed them sequentially. All three stopped at import with BLOCKED_DEPENDENCY because the current Python environment lacks required Gym APIs. No package reached create/reset/step/close, and none was marked ENVIRONMENT_SMOKE_VALIDATED.

Backups: `PROJECT_STATUS_backup_20260815_milestone_1a.md`, `TASKS_backup_20260815_milestone_1a.md`, and `docs/change_log_backup_20260815_milestone_1a.md`.

### Reason

Establish truthful current-machine runtime evidence without modifying packages or installing dependencies.

### Test command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m evaluation.smoke.smoke_cable_routing
PYTHONDONTWRITEBYTECODE=1 python3 -m evaluation.smoke.smoke_object_grasping
PYTHONDONTWRITEBYTECODE=1 python3 -m evaluation.smoke.smoke_peg_insertion
```

### Test result

PARTIAL — all harnesses executed correctly, but Cable/Object were BLOCKED_DEPENDENCY on missing `gym`, and Peg was BLOCKED_DEPENDENCY on missing `gymnasium` after its `gym` fallback. No environment lifecycle was validated.

### Rollback method

Restore the three `_backup_20260815_milestone_1a.md` files and remove only the four new harness files and four runtime-validation documents. No process source, external repository, demo, checkpoint, or environment rollback is required.

## 2026-08-15 — Runtime Environment Discovery correction

### Date

2026-08-15

### File

- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

All three processes were marked BLOCKED_DEPENDENCY using `/usr/bin/python3`, which is not the project runtime.

### Modification

Removed those invalid current blockers. Marked Runtime Environment Discovery VALIDATED, restored all three process environment runtimes to UNVERIFIED, and recorded `serl_mujoco` as the primary candidate, `serl` as the Object-compatible candidate, and Peg version compatibility as UNVERIFIED.

### Reason

The actual project environments are existing Conda environments under `/home/ubuntu/miniconda3`, not the system Python.

### Test command

```bash
/home/ubuntu/miniconda3/bin/conda env list
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python -c "import gym, gymnasium, mujoco, jax, flax"
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c "import gym, gymnasium, mujoco, jax, flax"
```

### Test result

PASS — both candidate environments import all queried core packages. No package was installed or modified. Process runtimes remain UNVERIFIED until lifecycle smoke completion.

### Rollback method

Revert only this documentation entry and the associated status/task edits. No environment or source rollback is required.

## 2026-08-15 — MILESTONE 1A smoke in validated Conda runtimes

### Date

2026-08-15

### File

- `evaluation/smoke/__init__.py`
- `docs/runtime_validation_cable_routing.md`
- `docs/runtime_validation_object_grasping.md`
- `docs/runtime_validation_peg_insertion.md`
- `docs/runtime_validation_summary.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Runtime candidates had been discovered, but all process lifecycle results were UNVERIFIED after invalid system-Python attempts.

### Modification

Ran Cable in `serl_mujoco`, Object in `serl`, and Peg in `serl_mujoco`. Added `PYTHON_PATH` output and read-only-write classification to the external harness. Recorded Cable FAIL on missing mesh assets, Object PASS with headless EGL, and Peg BLOCKED_CODE on attempted external source-tree XML generation.

### Reason

Complete bounded environment lifecycle smoke testing with the actual existing Conda runtimes without changing packages or environments.

### Test command

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python -m evaluation.smoke.smoke_cable_routing
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.smoke.smoke_object_grasping
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python -m evaluation.smoke.smoke_peg_insertion
```

### Test result

PARTIAL — Object completed import/create/reset/one-step/close and is ENVIRONMENT_SMOKE_VALIDATED. Cable failed at create due a missing referenced mesh. Peg imports passed but create is BLOCKED_CODE because it attempts to write generated XML inside protected external source.

## 2026-08-15 — MILESTONE 1A-FIX-PEG runtime output isolation

### File

`evaluation/smoke/peg_runtime_adapter.py`; `runtime/peg_insertion/generated/aubo_i5_assemble_hole.xml`; `docs/runtime_validation_peg_insertion.md`; `docs/runtime_validation_summary.md`; `PROJECT_STATUS.md`; `TASKS.md`; `docs/change_log.md`

### Original behavior

Peg imports succeeded, but `BaseTask.__init__()` derived its XML output directory from the editable `/home/ubuntu/mujoco_env` assets location. `RobotAssembler` therefore attempted to write `mujoco_env/tasks/PegInsertion/aubo_i5_assemble_hole.xml`, and smoke stopped as BLOCKED_CODE.

### Modification

Added a project-owned smoke adapter that temporarily redirects only `RobotAssembler.build_robot_scene()` output to `runtime/peg_insertion/generated/`, then restores the original method in a `finally` block. No external source or asset was copied or modified.

### Reason

Isolate generated runtime output in a project-writable directory while retaining read-only access to the original MuJoCo assets and preserving package behavior.

### Test command

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('evaluation/smoke/peg_runtime_adapter.py').read_text())"
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python -m evaluation.smoke.peg_runtime_adapter
find external/packages/peg_insertion /home/ubuntu/mujoco_env -type f -printf '%p\\t%s\\t%T@\\n' | sha256sum
```

### Test result

PASS — import/create/reset/one-step/close completed. Observation is `(20,)` float32, action is `(7,)` float32, and step returns five values. Generated XML is under `runtime/peg_insertion/generated/`. The external-tree path-size-mtime fingerprint remained `6d5d4b631d2edbfa332942a6f633c3d97b1f4dbf13fa23da83e7b5423daa2795` before and after.

### Rollback method

Stop using and remove the project-owned adapter and runtime staging artifacts. No external source rollback is required because external code was not changed.

### Rollback method

Revert the harness reporting-only changes and documentation updates. No source, environment, package, demo, or checkpoint rollback is required.

## 2026-08-21 — MILESTONE 1A-FIX-CABLE-V2

### Date

2026-08-21

### File

- `external/packages/cable_routing_v2/` (new isolated extraction)
- `evaluation/smoke/smoke_cable_routing_v2.py`
- `docs/cable_v1_v2_comparison.md`
- `docs/cable_demo_acquisition_plan.md`
- `docs/runtime_validation_cable_routing.md`
- `docs/runtime_validation_summary.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Cable V1 imported but could not create its environment because the archive omitted `franka_emika_panda/assets/`, including `link1.stl`. Cable demo recorder and data were MISSING.

### Modification

Identified `external/incoming/cable_routing_package(1).gz` as Cable V2, recorded SHA256 `5110d10c1e30f2567014fb1d58e0e0cadaf1dfe3d5bcda03259023ecfe202fcb`, verified gzip/tar structure and member-path safety, and extracted it without overwriting V1. Compared V1/V2, statically audited demo acquisition capability, added an external V2 smoke harness, and ran one zero-action lifecycle step in the version-matched `serl` environment with EGL.

### Reason

Verify whether the new delivery repairs the missing Franka assets and establish truthful runtime evidence without modifying package source, algorithms, checkpoints, or demos.

### Test command

```bash
file external/incoming/cable_routing_package\(1\).gz
sha256sum external/incoming/cable_routing_package\(1\).gz
gzip -t external/incoming/cable_routing_package\(1\).gz
sha256sum -c SHA256SUMS.txt
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.smoke.smoke_cable_routing_v2
```

### Test result

PASS — safe extraction succeeded; V2 contains 67 Franka assets and 75 STL/OBJ files; `link1.stl` is present; import/create/reset/one-step/close all passed. Actual observations are nested seven-value state plus two `(128,128,3)` uint8 cameras, and action is `(3,)` float32. The supplied checksum file validates all 240 other entries but its checksum of itself does not match; this packaging-manifest issue is recorded separately. Demo recorder and demo data remain MISSING, with project acquisition TODO.

### Rollback method

Stop referencing the isolated V2 directory and the new external smoke harness; restore the prior documentation revisions if needed. V1 and both incoming archives remain untouched. No source, checkpoint, demo, SERL, or Conda rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE Demo Recorder Design & Dry Run

### Date

2026-08-21

### File

- `tools/cable_demo/__init__.py`
- `tools/cable_demo/record_cable_demo.py`
- `tools/cable_demo/validate_cable_demo.py`
- `runtime/cable_demo/TEST_ONLY/TEST_ONLY_cable_routing_01_demos_2026-08-21_01-09-05.pkl`
- `runtime/cable_demo/TEST_ONLY/TEST_ONLY_cable_routing_01_demos_2026-08-21_01-09-05_metadata.json`
- `results/cable_demo/` (empty formal output directory)
- `docs/cable_demo_recorder.md`
- `docs/cable_demo_schema.md`
- `docs/cable_demo_acquisition_plan.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Cable V2 environment lifecycle was validated, but no project or delivered Cable recorder existed. Formal demo data was MISSING. Native nested images were also incompatible with the current singular-`image` SERL observation wrapper.

### Modification

Added a project-side keyboard recorder with W/S, A/D, Q/E xyz control, reset/end/exit handling, action clipping, deep-copied six-field transitions, success-only formal episode acceptance, flat-list pickle output, and companion episode metadata. Added a static validator and a project observation wrapper that flattens state, promotes front/wrist to top-level keys, and applies horizon-one chunking without modifying SERL.

### Reason

Prepare safe, authentic Cable demo acquisition against the runtime-validated V2 environment while preventing random, failed, timed-out, manually aborted, or TEST_ONLY trajectories from entering formal data.

### Test command

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo --help
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_demo_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo --dry-run
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.validate_cable_demo runtime/cable_demo/TEST_ONLY/TEST_ONLY_cable_routing_01_demos_2026-08-21_01-09-05.pkl --allow-test-only
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.validate_cable_demo runtime/cable_demo/TEST_ONLY/TEST_ONLY_cable_routing_01_demos_2026-08-21_01-09-05.pkl
```

### Test result

PASS — syntax/CLI checks passed; the real V2 environment reset, stepped once, and closed; one explicitly TEST_ONLY transition and metadata file were written outside the formal directory; SERL `populate_data_store` and `MemoryEfficientReplayBufferDataStore` accepted the pickle. Validator passed with `--allow-test-only` and rejected it by default because it is not a successful formal demo. Formal `results/cable_demo/` remains empty. Existing TensorFlow-Datasets/protobuf import noise and the non-fatal EGL destructor warning were observed; neither prevented the bounded dry run.

### Rollback method

Stop using the project-owned recorder/validator, remove only their TEST_ONLY artifacts if explicitly authorized, and revert the listed project documents. No Cable package, SERL, checkpoint, formal demo, or Conda rollback is required.

## 2026-08-21 — MILESTONE 1C-CABLE-AUTO-DEMO-PREFLIGHT

### Date

2026-08-21

### File

- `docs/cable_auto_demo_preflight.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Cable's environment and project recorder were validated, but all delivered checkpoints, restore compatibility, policy inference, and automatic-demo feasibility were UNVERIFIED.

### Modification

Inventoried six non-empty checkpoint steps across the delivered `shutdown_test` and `smoke` groups plus two empty named groups. Traced the delivered DrQ training, Flax restore, and deterministic `sample_actions(..., argmax=True)` inference path. Attempted a read-only restore of the highest numbered delivered candidate, `cable_route_smoke/checkpoint_200`, first with the exact environment-derived agent and then with a CPU-only exact-shape agent.

### Reason

Determine whether a real restored Cable policy could safely be considered for automatic successful-demo generation without changing the package, SERL, checkpoints, environment semantics, or existing demo data.

### Test command

```bash
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_auto_demo_cache /home/ubuntu/miniconda3/envs/serl/bin/python -c '<construct exact wrapped Cable DrQAgent; restore absolute checkpoint step 200; compare tree leaves>'
timeout 180s env JAX_PLATFORMS=cpu XDG_CACHE_HOME=/tmp/cable_auto_demo_cache /home/ubuntu/miniconda3/envs/serl/bin/python -c '<construct exact-shape Cable DrQAgent; restore absolute checkpoint step 200; compare tree leaves>'
```

### Test result

UNVERIFIED — the exact CPU target state constructed successfully with 598 leaves, but Orbax did not return a restored state or structural error before the 180-second hard limit (exit 124). An earlier relative-path call was correctly rejected because Orbax requires an absolute path and is not treated as checkpoint failure. No policy inference or episode ran; no candidate or formal demo was created; schema and replay validation were therefore not applicable.

### Rollback method

Revert the three status/documentation updates and remove only `docs/cable_auto_demo_preflight.md` if desired. No source, package, SERL, environment, checkpoint, demo, reward, or success-logic rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE-VIEWER-TELEOP

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The validated project Cable recorder used an OpenCV image window for blocking keyboard input. It did not provide a true MuJoCo Viewer teleoperation workflow, and desktop Viewer/key delivery remained untested.

### Modification

Added a separate project-owned recorder using `mujoco.viewer.launch_passive` on the real Cable model/data and the Viewer native key callback. It provides normalized xyz-delta control, zero-action idle, environment-control-period pacing, periodic/full state feedback, success-only immediate one-episode formal dataset saving, unchanged learner schema, truthful terminated/truncated handling, safe Viewer/reset/exit behavior, and JSON-only discard logs. The original recorder and validator were not changed.

### Reason

Enable operator-attended Cable demonstration acquisition with a true MuJoCo scene Viewer without changing Cable V2, SERL, checkpoints, reward, success conditions, or replay logic.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<parse recorder source with ast>'
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo_viewer --help
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_viewer_preflight_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo_viewer --preflight --episodes 1 --action-magnitude 0.10 --max-steps 1 --print-every 1
```

### Test result

PARTIAL — syntax/import, Viewer API signature, model/data exposure, env create/reset, one real zero-action step, and close passed. The current SSH session has no DISPLAY, so visible Viewer creation and keyboard callback behavior remain `OPERATOR_VIEWER_PREFLIGHT_REQUIRED`. No formal demo or TEST_ONLY file was written, and `results/cable_demo/` remained empty.

### Rollback method

Stop using and remove the new project-side Viewer recorder, then revert the four documentation/status changes. The pre-existing recorder remains intact. No Cable V2, SERL, checkpoint, reward, success logic, replay buffer, formal demo, or Conda rollback is required.

## 2026-08-21 — Cable Viewer Teleoperation pynput Fix

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `tools/cable_demo/record_cable_demo_viewer_before_pynput_20260821.py`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The Viewer recorder passed W/A/S/D/Q/E through MuJoCo Viewer's native key callback. Operator testing showed those keys triggered Viewer visualization shortcuts (colors, black background, grid changes) while the robot did not move. This route is `FAILED_FOR_TELEOP`.

### Modification

Removed the Viewer key callback from `launch_passive`. Added a lazy-loaded pynput global Listener with thread-safe press/release state, fixed-control-period action construction, continuous held-key control, simultaneous-axis combinations, zero action on release, R/P/Enter/ESC one-shot requests, key-state debug output, listener startup failure protection, and safe stop/join. Changed the default action magnitude from 0.15 to 0.05. Viewer remains responsible only for visualization and synchronization.

### Reason

Prevent MuJoCo Viewer shortcuts and keyboard autorepeat from controlling or disrupting robot teleoperation while preserving the real environment step path and formal Demo semantics.

### Test command

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c '<AST parse; test PressedKeyTeleop and nine action combinations>'
/home/ubuntu/miniconda3/envs/serl/bin/python -c 'import pynput'
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_pynput_preflight_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo_viewer --preflight --episodes 1 --action-magnitude 0.05 --max-steps 1 --print-every 1
```

### Test result

PARTIAL — syntax and nine pressed-key/action state cases passed; Viewer teleop input reports DISABLED; real create/reset/one-zero-step/close passed; no formal output was written. Direct pynput import failed with `ModuleNotFoundError`, and the SSH session has no DISPLAY. Therefore `VIEWER_RENDERING = IMPLEMENTED`, `PYNPUT_TELEOP = IMPLEMENTED`, and `OPERATOR_KEYBOARD_VALIDATION = REQUIRED`, but runtime operation is not VALIDATED.

### Rollback method

Restore `tools/cable_demo/record_cable_demo_viewer.py` from its single dated pre-pynput backup and revert the four documentation/status changes. No Cable V2, SERL, checkpoint, reward, success logic, replay buffer, demo, or Conda rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE-ACTION-PIPELINE-DIAGNOSIS

### Date

2026-08-21

### File

- `evaluation/debug/debug_cable_action_pipeline.py`
- `docs/cable_action_pipeline_diagnosis.md`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Human testing validated pynput and signed action construction, but the displayed TCP appeared fixed near `[0,0.307,0]` while the cable continued moving under flexible dynamics. The fault location was unknown between action scaling, target, controller, actuation, and TCP observation.

### Modification

Added a project-side diagnostic harness that reads before/after target, mocap, ctrl, robot qpos, TCP sensor, and TCP site while applying only normal `env.step()` actions. Ran zero and six signed single-step actions plus independent 20-step positive/negative X held-action tests. Traced exact Cable environment, opspace controller, actuator, and sensor source locations. No production source was modified.

### Reason

Separate input, scaling, target update, controller tracking, actuation, and observation issues using bounded measured evidence before proposing any teleoperation change.

### Test command

```bash
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_action_pipeline_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.debug.debug_cable_action_pipeline
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_state_order_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c '<compare flattened state, Gym unflatten, native TCP sensor, and TCP site>'
```

### Test result

PASS for diagnosis. Action 0.05 became 0.00005 m target delta on the requested axis; target/mocap matched exactly; ctrl and qpos changed; true sensor/site TCP changed. Twenty X-negative steps requested -1 mm and produced approximately -1.167 mm true TCP X displacement. The apparent `[0,0.307,0]` TCP was reproduced as flattened `[gripper,tcp_x,tcp_y]`, proving a recorder debug-index error. No formal Demo was written. Overall teleop motion remains FAILED / UNDER_DIAGNOSIS until a bounded magnitude/display correction is approved and tested.

### Rollback method

Stop using and remove the project diagnostic harness and revert the five documentation/status changes. No Cable V2, SERL, recorder, Viewer, pynput mapping, checkpoint, reward, success logic, controller, MuJoCo state, or Demo rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE-TELEOP-MINIMAL-FIX

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `tools/cable_demo/record_cable_demo_viewer_before_tcp_display_fix_20260821.py`
- `docs/cable_viewer_teleop.md`
- `docs/cable_action_pipeline_diagnosis.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The recorder printed `state[:3]` as TCP even though flattened state order is `[gripper,tcp_x,tcp_y,tcp_z,vel_x,vel_y,vel_z]`, producing the misleading value near `[0,0.307,0]`. Magnitude 0.05 was already diagnosed as too small for clear visual motion.

### Modification

Changed operator/debug output only. `TCP_POS_REAL` now reads sensor `2f85/pinch_pos`; `TCP_SITE` independently reads `data.site_xpos[pinch]`; `TCP_MATCH` checks them at 1e-9 absolute tolerance; `MOCAP_TARGET` is printed separately. Periodic output now explicitly lists step, action, real TCP, target, cable tip, slot, distance, reward/return, and success. Added bounded 0.25 and 0.50 operator test instructions without changing the recorder default or environment scale.

### Reason

Remove a confirmed debug-index error and provide a safe magnitude-validation path while preserving the validated learner-facing schema and all environment/controller semantics.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<AST parse recorder source>'
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c 'from tools.cable_demo.record_cable_demo_viewer import read_real_tcp, print_state'
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_tcp_display_fix_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo_viewer --preflight --episodes 1 --action-magnitude 0.25 --max-steps 1 --print-every 5
```

### Test result

PASS for the minimal display fix: syntax/import/create/reset/one zero-action step/close passed; `TCP_POS_REAL` and `TCP_SITE` matched (`TCP_MATCH=True`); mocap target printed separately; formal output directory remained empty. Robot teleop motion remains `OPERATOR_VALIDATION_REQUIRED` because no human motion test was performed.

### Rollback method

Restore the recorder from `record_cable_demo_viewer_before_tcp_display_fix_20260821.py` and revert the five documentation/status updates. No Cable V2, SERL, schema, checkpoint, reward, success logic, controller, qpos, ctrl, MuJoCo state, or Demo rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE-KEY-EVENT-SUPPRESSION

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `tools/cable_demo/record_cable_demo_viewer_before_key_suppression_20260821.py`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The pynput listener captured Cable teleoperation keys and produced correct actions, but did not suppress delivery. Operator evidence showed the same events continued into MuJoCo Viewer shortcuts, toggling the background, geometry, grid, and visualization modes.

### Modification

Enabled pynput 1.8.2 `Listener(..., suppress=True)`, added `PYNPUT_SUPPRESS: ENABLED` and `KEY_SUPPRESSION = ACTIVE` startup output, retained raw press/release and logical held-key diagnostics, and placed listener startup plus the complete recorder lifecycle under unconditional `stop()`/`join()` cleanup. The helper raises if the listener remains alive after its bounded join.

### Reason

Keep teleoperation events out of the MuJoCo Viewer and other X11 clients while preserving the fixed-rate held-key action loop, environment behavior, and learner-facing Demo schema.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<AST parse recorder source>'
PYNPUT_BACKEND=dummy /home/ubuntu/miniconda3/envs/serl/bin/python -c '<construct Listener with suppress=True; validate callbacks and normal/exception cleanup>'
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo_viewer --help
```

### Test result

PASS for non-GUI implementation checks: pynput version 1.8.2 and Listener suppression support were confirmed; syntax/import, suppressed-listener construction, held-key callbacks, and normal/exception cleanup passed. Viewer/X11 suppression is not claimed VALIDATED and remains `OPERATOR_SUPPRESSION_VALIDATION = REQUIRED`. No environment episode, formal Demo, checkpoint, training, or evaluation ran.

### Rollback method

Restore the recorder from `record_cable_demo_viewer_before_key_suppression_20260821.py` and revert the four documentation/status updates. No Cable V2, SERL, schema, checkpoint, reward, success logic, controller, MuJoCo state, or Demo rollback is required.

## 2026-08-21 — Cable Operator Episode Time-Limit Fix

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `tools/cable_demo/record_cable_demo_viewer_before_operator_time_limit_20260821.py`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Increasing recorder `--max-steps` to 10000 did not prevent resets. Cable registration wrapped the environment with Gym `TimeLimit(max_episode_steps=500)`, while the native environment independently returned truncation at its default 10.0-second limit. At `control_dt=0.02`, both boundaries occurred near recorder step 500.

### Modification

For the operator recorder instance only, set Gym TimeLimit to 20000 steps and native time limit to `20000 * control_dt` (400.0 seconds). Set recorder defaults to 20000 steps and action magnitude 1.0. Added mandatory episode-boundary output for recorder step, terminated, truncated, info, and reset reason. Success termination, reward, controller, environment scale, and transition schema are unchanged.

### Reason

Prevent the delivered short training time limits from resetting a human demonstration attempt before the operator reaches the slot, without modifying Cable V2 source or task semantics.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<AST parse and reset-reason output test>'
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_operator_limit_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c '<create env; configure recorder limit; inspect wrapper/native values; reset; bounded zero-action step; close>'
```

### Test result

PASS for project-side configuration and bounded lifecycle checks. The recorder environment reported Gym limit 500 -> 20000 and native limit 10.0s -> 400.0s at control_dt 0.02; reset and a zero-action step passed. No GUI operator episode or formal Demo was run.

### Rollback method

Restore `tools/cable_demo/record_cable_demo_viewer.py` from `record_cable_demo_viewer_before_operator_time_limit_20260821.py` and revert the four documentation/status files. No Cable V2, SERL, environment source, reward, success, controller, action scale, checkpoint, or Demo rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE-GEAR-SPEED

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `tools/cable_demo/record_cable_demo_viewer_before_gear_speed_20260821.py`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The recorder executed one real `env.step(action)` per fixed-rate control-loop tick. Magnitude was already bounded by the environment action space, but the operator had no project-side speed gear.

### Modification

Added Gear 1/2/3 on numeric keys, mapped to 1/2/5 normal environment steps per control tick, with Gear 3 as startup default. Each repeated step produces and appends its own unchanged learner-facing transition. A success, termination, truncation, or 20000-step boundary stops the gear batch immediately. Periodic output now includes Gear alongside step, action, real TCP, distance, and success.

### Reason

Provide operator-selectable motion speed without changing Cable action scale, controller, reward, success logic, SERL, or Demo schema.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<AST parse; key/gear/default/action tests>'
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<fake-env 1/2/5 step and transition-preservation tests; terminal short-circuit test>'
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_gear_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.record_cable_demo_viewer --preflight --max-steps 1
```

### Test result

PASS for implementation and non-GUI behavior: Gear selection/default, action bounds, 1/2/5 real step counts, one transition per step, terminal short-circuit, syntax/import, and bounded real-environment preflight passed. No formal Demo was recorded. Two-second visible W displacement ordering remains `OPERATOR_VALIDATION_REQUIRED`.

### Rollback method

Restore `tools/cable_demo/record_cable_demo_viewer.py` from `record_cable_demo_viewer_before_gear_speed_20260821.py` and revert the four documentation/status files. No Cable V2, SERL, environment source, action scale, controller, reward, success, SERL, schema, checkpoint, or Demo rollback is required.

## 2026-08-21 — Cable Recorder Post-Success Cleanup Fix

### Date

2026-08-21

### File

- `tools/cable_demo/record_cable_demo_viewer.py`
- `tools/cable_demo/record_cable_demo_viewer_before_cleanup_fix_20260821.py`
- `docs/cable_viewer_teleop.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The first real successful Cable Demo was fully saved and the recorder printed PASS, but the suppressed pynput X11 listener remained alive beyond the two-second join timeout. Cleanup then raised `RuntimeError`, producing a traceback after the successful result even though the pickle was already safely on disk.

### Modification

Extended the bounded listener join to five seconds and made it return cleanup status instead of deciding Demo status. Added independent `DEMO_RESULT` and `CLEANUP_RESULT` reporting. Viewer, environment, and listener cleanup are all attempted even if another cleanup operation warns. A cleanup warning after at least one saved success does not raise; the same listener failure with zero saved successes still raises normally.

### Reason

Ensure a post-save X11 thread shutdown delay cannot invalidate or obscure a real saved Demo while retaining strict failure reporting before any successful save.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<AST parse; fake listener normal/timeout/exception lifecycle tests>'
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<post-save warning and zero-success error reporting tests>'
sha256sum results/cable_demo/cable_routing_01_demos_2026-08-21_07-14-17.pkl
```

### Test result

PASS. Normal listener cleanup returns PASS; a still-alive or exception-throwing listener returns WARNING; post-save warning does not raise; zero-success warning raises. The existing Demo remained byte-identical at SHA256 `415b2adf5c2453db6532a6c5b9347b6ba6c2b97b393331c98c956e9ef71e54e0`. No Demo was recorded or modified. EGL destructor output remains a known non-blocking exit warning.

### Rollback method

Restore the recorder from `record_cable_demo_viewer_before_cleanup_fix_20260821.py` and revert the four documentation/status changes. Do not alter the saved Demo. No Cable V2, SERL, environment, reward, success logic, schema, checkpoint, or Demo rollback is required.

## 2026-08-21 — Cable First Formal Demo Validation Status Sync

### Date

2026-08-21

### File

- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The first formal Cable Demo was recorded as PRESENT but validator status remained UNVERIFIED in project documentation.

### Modification

Synchronized the project-member-confirmed output from the existing validator: `RESULT=PASS`, `TEST_ONLY=False`, 1 episode, 2629 transitions, and 1 success. Recorded `DEMO_RECORDER=VALIDATED`, `DEMO_DATA=PRESENT`, `FORMAL_DEMO_COUNT=1`, `FORMAL_DEMO_SUCCESS_COUNT=1`, `DEMO_VALIDATION=VALIDATED`, and `DEMO_TRANSITIONS=2629`.

### Reason

Make project status reflect the completed formal Demo validation without extending that evidence to checkpoint, policy, success rate, or training.

### Test command

```bash
sha256sum results/cable_demo/cable_routing_01_demos_2026-08-21_07-14-17.pkl
stat -c '%n %s bytes' results/cable_demo/cable_routing_01_demos_2026-08-21_07-14-17_metadata.json
```

### Test result

Status synchronization PASS. The pickle SHA256 remains `415b2adf5c2453db6532a6c5b9347b6ba6c2b97b393331c98c956e9ef71e54e0`; metadata exists. The validator was not rerun in this status-only task. No source, validator, recorder, package, SERL, checkpoint, or Demo file was modified.

### Rollback method

Revert only this status-sync entry and the corresponding `PROJECT_STATUS.md` / `TASKS.md` text. No code or data rollback is required.

## 2026-08-21 — Cable Initial-State Randomization Audit

### Date

2026-08-21

### File

- `docs/cable_initial_state_audit.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Five formal successful Cable Demo files were present, but it was unclear whether reset produced diverse initial robot, cable, slot, or camera states.

### Modification

Added a read-only static/runtime audit. Traced the Cable reset, training environment, and evaluation environment construction. Executed 20 consecutive resets with seeds 0–19 and zero control actions; measured TCP, Panda qpos, cable root pose, all 63 cable qpos components, cable tip, slot, and camera parameters. Documented statistics, classification, diversity risk, and a future collection recommendation. Synchronized the presence of five operator-confirmed successful files while keeping the four later files UNVERIFIED.

### Reason

Determine whether the existing successful demonstrations cover multiple initial conditions before planning further data acquisition.

### Test command

```bash
rg -n 'def reset|random|uniform|normal|noise|seed' external/packages/cable_routing_v2/.../franka_sim
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_initial_state_audit_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c '<20 reset-only samples and min/max/mean/std>'
```

### Test result

PASS. All measured component-wise maximum spans were exactly `0.0`; seed values 0–19 did not affect initialization. `INITIAL_STATE = FIXED`, `DEMO_DIVERSITY_RISK = HIGH`. Training and evaluation instantiate the same environment/reset implementation. No step action, source change, randomization, or Demo creation/modification occurred. The known non-blocking EGL destructor warning appeared after close.

### Rollback method

Remove only `docs/cable_initial_state_audit.md` and revert this status/task/change-log synchronization. No source or data rollback is required.

## 2026-08-21 — MILESTONE 1B-CABLE-FIVE-DEMO-VALIDATION

### Date

2026-08-21

### File

- `docs/cable_five_demo_validation.md`
- `docs/cable_demo_quality_summary.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Five formal successful Cable Demo files were present, but only the first had documented validator evidence. Batch schema consistency, quality statistics, real replay insertion, and merge feasibility were unverified.

### Modification

Enumerated and hashed all five formal pickles, ran the unchanged validator separately on each, compared actual schema/dtypes, computed episode and zero-action statistics, inserted all files through SERL `populate_data_store` into the current memory-efficient replay implementation, and tested episode-ordered in-memory concatenation without writing a merged file. Documented fixed-reset in-distribution versus OOD risk separately.

### Reason

Establish a truthful, learner-facing five-demo baseline dataset before any training work, without conflating fixed-start baseline compatibility with robustness randomization.

### Test command

```bash
for demo in results/cable_demo/*.pkl; do sha256sum "$demo"; /home/ubuntu/miniconda3/envs/serl/bin/python -m tools.cable_demo.validate_cable_demo "$demo"; done
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<per-file schema, episode quality, and zero-action statistics>'
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python -c '<real make_replay_buffer + populate_data_store for all five files>'
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<in-memory concat and episode-boundary checks>'
```

### Test result

PASS. Five of five validators returned PASS with `TEST_ONLY=False`; schemas were identical; no shape/dtype/finite/action/mask/done/terminal/metadata error occurred. Raw transitions total 10,521. Replay insertion produced exactly 10,526 records, accounting for one horizon-one initialization frame per episode. Merge compatibility is YES with five contiguous success-terminal boundaries. No learner, training, checkpoint restore, reset randomization, new recording, or formal merged file was executed.

### Rollback method

Remove only the two new audit documents and revert this status/task/change-log synchronization. Original Demo files and all code remain unchanged; no data or source rollback is required.

## 2026-08-21 — Cable W&B Training Preflight

### Date

2026-08-21

### File

- `docs/cable_wandb_preflight.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The five Cable demos were validated, but W&B initialization, uploaded metrics, step semantics, and the requested online run identity had not been audited before training.

### Modification

Documented the learner-only `WandBLogger` call chain, online/disabled debug behavior, actual flattened metric keys, asynchronous actor-stat handling, and learner-update global-step semantics. Recorded missing train return, buffer-size, and training-success metrics without inventing replacements. Verified the local W&B version/base URL without accessing credentials. Identified the hard-coded project, absent explicit run name, single-pickle demo flag, and observation-wrapper mismatch as pre-training gaps.

### Reason

Ensure a later bounded Cable baseline run uses visible online W&B with truthful metric names and does not start under an incorrect project/run identity or incompatible data path.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c 'import wandb; print(wandb.__version__); print(wandb.Settings().base_url)'
rg -n 'wandb|WandBLogger|wandb_logger.log|update_steps|eval|running_return|replay_buffer' external/packages/cable_routing_v2/... external/serl/serl_launcher/...
```

### Test result

PASS for static preflight. W&B 0.28.0 and `https://api.wandb.ai` were confirmed. No credential file/API key, learner, actor, W&B run, training, checkpoint, evaluation episode, Demo, or external source was accessed or modified beyond the stated read-only source inspection.

### Rollback method

Remove `docs/cable_wandb_preflight.md` and revert only the associated status/task/change-log entries. No code, environment, credential, checkpoint, or Demo rollback is required.

## 2026-08-21 — MILESTONE 1C-CABLE-PROJECT-LAUNCHER

### Date

2026-08-21

### File

- `training/__init__.py`
- `training/cable/__init__.py`
- `training/cable/run_cable_training.py`
- `configs/cable/cable_baseline_5demo.yaml`
- `docs/cable_project_training_launcher.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The delivered Cable entry hard-coded W&B project `serl_cable_route`, did not pass an explicit visible run name, accepted one demo pickle, and applied the incompatible singular-`image` SERL wrapper. No launch-ready project-side baseline configuration existed.

### Modification

Added one project-side Python launcher and one YAML baseline config. The launcher explicitly constructs W&B project/name/mode, loads the five validated pickles sequentially into one demo buffer without writing a merged duplicate, reuses the validated `state/front/wrist` wrapper for actor/learner/evaluation environments, preserves delivered DrQ hyperparameters, isolates smoke/formal outputs, and applies fail-fast data/schema/capacity/output/W&B/JAX assertions. Added bounded smoke design for 100 learner updates and 500 actor steps. No training was run.

### Reason

Resolve the three confirmed Cable smoke-training integration blockers without changing Cable V2, SERL, formal Demos, checkpoints, environment semantics, or credentials.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -m py_compile training/__init__.py training/cable/__init__.py training/cable/run_cable_training.py
/home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --help
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_launcher_dryrun_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --dry-run --smoke
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<offline fake-W&B explicit project/name/mode and flattening assertions>'
```

### Test result

PASS for syntax, import, config parse, all five SHA/schema/count checks, 10,521 transition accounting, real observation adapter/reset, `(3,)` action space, replay capacity, output isolation, JAX device discovery, and offline W&B construction. `WANDB_INIT=NOT_CALLED` and `LEARNER_UPDATE=NOT_STARTED`. Known optional TFDS/protobuf and EGL cleanup warnings did not prevent dry-run PASS.

### Rollback method

Remove the new `training/` launcher files, YAML config, and launcher document, then revert only the associated status/task/change-log entries. No Cable V2, SERL, environment, Conda, W&B credential, Demo, checkpoint, or result rollback is required.

## 2026-08-21 — MILESTONE 1C-CABLE-CHECKPOINT-SAVE-DIAGNOSIS

### Date

2026-08-21

### File

- `evaluation/debug/debug_cable_checkpoint_save.py`
- `training/cable/run_cable_training.py`
- `training/cable/run_cable_training_before_checkpoint_fix_20260821.py`
- `configs/cable/cable_baseline_5demo.yaml`
- `configs/cable/cable_baseline_5demo_before_checkpoint_fix_20260821.yaml`
- `docs/cable_checkpoint_save_diagnosis.md`
- `docs/cable_project_training_launcher.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The real smoke created online W&B successfully and the actor completed 500/500 with clean shutdown. Learner entered the real update path but failed on its first Flax/Orbax checkpoint save with TensorStore rejecting `kvstore.assume_config`; tqdm still displayed 0/100.

### Modification

Audited exact versions and the save call. Added an isolated debug reproduction under `runtime/checkpoint_debug/`, reproduced the exact TensorStore field rejection, bounded the non-returning default Orbax save in a child process, and validated Flax legacy save/restore. Wrapped only project-launched Cable saves with temporary legacy Flax serialization, restoring the global config afterward. Preserved the failed smoke outputs and configured a new non-overwriting checkpoint-fix retry run/path.

### Reason

Resolve checkpoint persistence without modifying Cable V2, SERL, Conda packages, formal Demos, delivered checkpoints, W&B credentials, or existing smoke evidence.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -c '<print Python/Flax/Orbax/TensorStore/JAX versions and Flax checkpoint backend flag>'
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -u evaluation/debug/debug_cable_checkpoint_save.py
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -c '<project legacy saver save/restore assertion>'
```

### Test result

PASS for diagnosis and isolated fix: TensorStore 0.1.51 rejected the Orbax 0.6.4 `assume_config` member; default Orbax save did not return within 15 seconds; legacy Flax saved and restored the equivalent pytree; the launcher wrapper independently saved/restored. No actor, learner, W&B run, training, formal checkpoint, Demo, or environment modification was executed in this diagnosis.

### Rollback method

Restore `training/cable/run_cable_training.py` and the YAML config from their two dated backups, remove only the new debug/diagnosis files, and revert the associated status/task/change-log entries. Preserve all existing smoke W&B logs and incomplete Orbax output as evidence unless the user separately authorizes cleanup.

## 2026-08-22 — MILESTONE 1D THREE-TASK GPU TRAINING PREPARATION STRATEGY

### Date

2026-08-22

### File

- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/three_task_gpu_handoff_preflight.md`
- `docs/gpu_training_operator_contract.md`
- `docs/three_task_training_preparation_roadmap.md`
- `docs/gpu_return_artifact_spec.md`
- `docs/change_log.md`

### Original behavior

The immediate project milestone was a local Cable 100-update CPU smoke retry. No common three-process GPU handoff gate, operator contract, return-artifact contract, or ordered M1D roadmap existed.

### Modification

Changed the execution strategy so this machine performs development, data/preflight/evaluation preparation, and GPU handoff preparation only. Preserved all prior Cable PARTIAL smoke evidence while moving bounded full smoke and formal training to the GPU host. Defined the A–G `GPU_TRAINING_HANDOFF_READY` gate, Cable blockers, ordered Object/Peg preparation, GPU operator workflow, returned artifacts, training order, handoff directory specification, and M1D-A through M1D-F roadmap. No deliverable payload directory or large artifact copy was created.

### Reason

Formal long-running training will be performed by a project member with a GPU. The repository must provide reproducible, evidence-based handoff inputs and acceptance criteria without treating local CPU training as the main line.

### Test command

```bash
rg -n "MILESTONE 1D|GPU_TRAINING_HANDOFF_READY|100-update|FULL_GPU_SMOKE|M1D-[A-F]" PROJECT_STATUS.md TASKS.md docs/three_task_gpu_handoff_preflight.md docs/gpu_training_operator_contract.md docs/three_task_training_preparation_roadmap.md docs/gpu_return_artifact_spec.md
find docs -maxdepth 1 -type f -name '*gpu*' -o -name '*training_preparation*'
```

### Test result

Documentation-only change. Structural/content checks are required; no training, environment episode, checkpoint operation, W&B call, Demo access, dependency change, or large-file copy is authorized or performed.

### Rollback method

Remove the four new M1D documents and revert the associated `PROJECT_STATUS.md`, `TASKS.md`, and change-log entries. Preserve all prior Cable smoke logs, incomplete Orbax output, Demos, checkpoints, packages, and environments.

## 2026-08-22 — M1D-A CABLE GPU HANDOFF FINALIZE

### Date

2026-08-22

### File

- `evaluation/cable/__init__.py`
- `evaluation/cable/checkpoint_roundtrip.py`
- `evaluation/cable/evaluate_policy.py`
- `configs/cable/environment_gpu.yaml`
- `deliverables/gpu_training_handoff/cable/README.md`
- `deliverables/gpu_training_handoff/cable/demo_manifest.json`
- `deliverables/gpu_training_handoff/cable/environment_spec.md`
- `deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml`
- `deliverables/gpu_training_handoff/cable/run_gpu_smoke_learner.sh`
- `deliverables/gpu_training_handoff/cable/run_gpu_smoke_actor.sh`
- `deliverables/gpu_training_handoff/cable/run_train_learner.sh`
- `deliverables/gpu_training_handoff/cable/run_train_actor.sh`
- `deliverables/gpu_training_handoff/cable/run_eval.sh`
- `deliverables/gpu_training_handoff/cable/expected_outputs.md`
- `scripts/verify_cable_handoff.py`
- `docs/cable_gpu_handoff_final.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Cable had validated Demos and a project launcher, but no portable GPU operator package, real full-agent legacy checkpoint roundtrip, independent evaluation CLI, handoff verifier, frozen Demo manifest/config snapshot, checkpoint selection policy, or returned layout. The canonical config contained a local Python path. Formal resume was not supported.

### Modification

Created a lightweight handoff package without copying Demos/packages. Added five-file size/SHA/count/success manifest, portable frozen config, GPU base environment spec, bounded 10-update/400-step smoke scripts, formal learner/actor scripts, evaluation script, step-by-step operator README, checkpoint policy and return layout. Added a verifier that checks required files, all Demo hashes, W&B identity, empty outputs, modules, and optional GPU enforcement without reading credentials. Added an independent evaluation CLI and a full-real-agent checkpoint roundtrip harness.

The final bounded roundtrip saved the exact Cable DrQ state through legacy Flax, destroyed/rebuilt the agent, restored 598 leaves, matched every serialized key path/shape/dtype and every leaf hash, and passed restored deterministic policy inference. Raw PyTree auxiliary metadata differs because rebuilt function/optimizer object identities differ; this is retained as diagnostics. Resume remains BLOCKED because learner update numbering, replay/actor progress, output collision, and W&B continuation are not persisted.

### Reason

Close all truthful Cable GPU handoff gaps that can be validated locally without training, dependency changes, external-source edits, Demo mutation, or large artifact copying.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -m py_compile evaluation/cable/*.py scripts/verify_cable_handoff.py
bash -n deliverables/gpu_training_handoff/cable/run_*.sh
/home/ubuntu/miniconda3/envs/serl/bin/python scripts/verify_cable_handoff.py
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml --dry-run --smoke
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.cable.evaluate_policy --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml --dry-run
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.cable.checkpoint_roundtrip --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml --output-dir runtime/cable_handoff/checkpoint_roundtrip_final
```

### Test result

PASS for syntax/shell/config parsing, five Demo hashes and 10,521 transition dry-run validation, portable output isolation, explicit W&B contract construction, evaluation reset/one-step schema, full-agent legacy save/restore, all-leaf equality, and restored policy inference. GPU backend, online GPU smoke, formal training, trained-policy evaluation, and resume are NOT_STARTED/UNVERIFIED. Existing optional TFDS/protobuf and EGL destructor warnings remained non-blocking. `CABLE_GPU_TRAINING_HANDOFF_READY = NOT_READY` solely because `TRAINING_RESUME = BLOCKED`.

### Rollback method

Stop using the new handoff/evaluation/verifier files and revert the three status documents. Preserve all original Demos, delivered packages/checkpoints, historical smoke evidence, and bounded roundtrip outputs. No external source, canonical Cable config, SERL core, environment, reward, or success logic was modified.

## 2026-08-22 — M1D-A2 CABLE SAFE TRAINING RESUME

### Date

2026-08-22

### File

- `training/cable/run_cable_training.py`
- `training/cable/run_cable_training_before_safe_resume_20260822.py`
- `training/cable/resume.py`
- `evaluation/cable/resume_roundtrip.py`
- `scripts/verify_cable_handoff.py`
- `scripts/verify_cable_handoff_before_safe_resume_20260822.py`
- `deliverables/gpu_training_handoff/cable/run_resume_learner.sh`
- `deliverables/gpu_training_handoff/cable/run_resume_actor.sh`
- `deliverables/gpu_training_handoff/cable/README.md`
- `deliverables/gpu_training_handoff/cable/expected_outputs.md`
- `docs/cable_training_resume.md`
- `docs/cable_gpu_handoff_final.md`
- `docs/cable_project_training_launcher.md`
- `docs/gpu_return_artifact_spec.md`
- `docs/three_task_gpu_handoff_preflight.md`
- `docs/gpu_training_operator_contract.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

The project launcher could save a legacy Flax model state but had no resume CLI or sidecars. Learner/checkpoint/W&B steps restarted at zero; actor random exploration restarted; online replay and its sampling state were lost; Demo replay had no explicit resume contract; checkpoint collisions and same-run W&B continuation were not controlled.

### Modification

Added fail-closed `--resume-from`, `--resume-actor-step`, and `--wandb-run-id` handling without changing the delivered learner/actor loops. Each new checkpoint commits a legacy model file, active chronological online-replay snapshot, and metadata sidecar. Metadata binds environment, algorithm, seed, observation/action schema, config, Demo manifest, W&B identity, learner/actor progress, replay hash/capacity, source hash, and timestamp. Resume restores full agent and replay/RNG state, rebuilds Demo replay, offsets global steps, avoids repeated random exploration, uses W&B `resume="must"`, and refuses missing/mismatching/colliding artifacts.

### Reason

Online replay is semantically required by the delivered 50/50 online/Demo learner and by its 300-transition startup gate. Silent replay reset would block or change formal training. Safe GPU handoff therefore required complete state continuity rather than model-only restoration.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -m py_compile training/cable/run_cable_training.py training/cable/resume.py evaluation/cable/resume_roundtrip.py scripts/verify_cable_handoff.py
bash -n deliverables/gpu_training_handoff/cable/run_resume_learner.sh deliverables/gpu_training_handoff/cable/run_resume_actor.sh
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.cable.resume_roundtrip --output-dir runtime/cable_handoff/resume_roundtrip_v2
/home/ubuntu/miniconda3/envs/serl/bin/python scripts/verify_cable_handoff.py
```

### Test result

PASS. Separate create/resume processes used a real Cable agent checkpoint at step 123. Full model inference, a bounded actual memory-efficient replay (four actor transitions/five records), replay RNG/progress, all five Demo files (10,521 raw/10,526 replay), W&B same-run `must` kwargs, next checkpoint 125, original checkpoint hash preservation, and collision rejection passed. No learner update, training loop, W&B online initialization, GPU smoke, formal training, dependency change, external-source edit, or Demo mutation occurred. `TRAINING_RESUME = VALIDATED`; `CABLE_GPU_TRAINING_HANDOFF_READY = READY`.

### Rollback method

Restore `training/cable/run_cable_training.py` and `scripts/verify_cable_handoff.py` from their dated pre-resume backups, stop using/remove only the new resume module/test/scripts/documentation, and revert the associated status entries. Preserve all Demos, delivered checkpoints, historical smoke evidence, and bounded runtime evidence.

## 2026-08-22 — M1D-B OBJECT DEMO + ALGORITHM VALIDATION

### Date

2026-08-22

### File

- `scripts/audit_object_m1d_b.py`
- `results/object_m1d_b/*.json`
- `docs/object_demo_validation.md`
- `docs/object_algorithm_mapping.md`
- `docs/object_checkpoint_audit.md`
- `docs/object_demo_quality_summary.md`
- `docs/object_m1d_b_report.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Object had only environment lifecycle validation and package-supplied claims. Demo contents/success/boundaries, replay insertion, DPRL-to-code identity, checkpoint architecture/restore/inference, and reset distribution were unverified.

### Modification

Added a project-side read-only audit runner and frozen raw JSON evidence. Hashed the archive, critical sources, Demo, and checkpoint tree. Deserialized the trusted Demo, audited every transition, inserted all records into the actual memory-efficient visual replay, sampled a packed batch, ran 20 EGL resets, mapped the delivered DrQ/demo-mixing algorithm, compared checkpoint metadata with a freshly constructed visual DrQ state, and made one bounded restore attempt. Created the five requested reports and synchronized project status/tasks.

### Reason

Advance Object through M1D-B using existing delivered assets without training, GPU smoke, package mutation, Demo/checkpoint conversion, or unsupported DPRL/success claims.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -m py_compile scripts/audit_object_m1d_b.py
PYTHONPATH=external/serl/serl_launcher /home/ubuntu/miniconda3/envs/serl/bin/python scripts/audit_object_m1d_b.py demo --output results/object_m1d_b/demo_audit.json
PYTHONPATH=external/serl/serl_launcher /home/ubuntu/miniconda3/envs/serl/bin/python scripts/audit_object_m1d_b.py replay --output results/object_m1d_b/replay_audit.json
MUJOCO_GL=egl PYTHONPATH=external/packages/object_grasping/object_grasp_package/code/env /home/ubuntu/miniconda3/envs/serl/bin/python scripts/audit_object_m1d_b.py resets --output results/object_m1d_b/reset_audit.json
PYTHONPATH=external/serl/serl_launcher /home/ubuntu/miniconda3/envs/serl/bin/python scripts/audit_object_m1d_b.py structure --output results/object_m1d_b/checkpoint_structure_audit.json
```

### Test result

PARTIAL. Demo schema/finite/action bounds PASS; actual replay insertion/sample PASS at 2,001 records; 20-reset distribution audit PASS; code-level `DRQ_PLUS_DEMO_REPLAY` mapping VALIDATED. The pickle contains no true episode boundary or success/block/lift fields, so 20-episode success validation is BLOCKED. Checkpoint exact mapping is MISMATCH by three optimizer inner-state paths; the only read-only restore attempt was stopped after 120 seconds with no returned state, so restore/inference remain UNVERIFIED. Existing TFDS/protobuf and EGL/cache warnings were non-blocking for completed tests.

### Rollback method

Stop using/remove only the new project audit script, raw M1D-B JSON evidence, and five Object reports, then revert the M1D-B entries in the three status documents. The frozen archive, extracted Object package, Demo, checkpoint, external SERL, environment, reward, and success logic were never modified.

## 2026-08-22 — M1D-B2 OBJECT DEMO PROVENANCE + CHECKPOINT CLOSURE

### Date

2026-08-22

### File

- `scripts/audit_object_m1d_b2.py`
- `results/object_m1d_b2/*.json`
- `docs/object_demo_provenance.md`
- `docs/object_checkpoint_closure.md`
- `docs/object_m1d_b2_report.md`
- `docs/object_checkpoint_audit.md`
- `docs/object_m1d_b_report.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

M1D-B had no generator/success provenance for the claimed 20 trajectories. Checkpoint metadata appeared to have three more paths than the fresh target, was marked MISMATCH, and a 120-second restore returned no state.

### Modification

Searched the full delivered package and local SERL consumer/documentation for the official Demo's generator, writer, boundary, and success-only persistence logic. Added a bounded optimizer/checkpoint audit that retains Flax empty nodes, performs one zero-gradient three-optimizer update without a learner/loss loop, and retries restore only after exact structural match. Added raw evidence and three closure reports; corrected prior checkpoint mapping statements.

### Reason

Resolve the two remaining M1D-B blockers without guessing 20×100 boundaries, deleting checkpoint paths, converting artifacts, training, evaluation, dependency changes, or external-source mutation.

### Test command

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -m py_compile scripts/audit_object_m1d_b2.py
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=external/serl/serl_launcher /home/ubuntu/miniconda3/envs/serl/bin/python scripts/audit_object_m1d_b2.py optimizer --output results/object_m1d_b2/optimizer_tree_audit.json
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=external/serl/serl_launcher /home/ubuntu/miniconda3/envs/serl/bin/python scripts/audit_object_m1d_b2.py restore --timeout-seconds 300 --output results/object_m1d_b2/restore_audit.json
```

### Test result

PARTIAL. No Demo generator or success-only persistence guarantee was found. The 20×100 hypothesis is strongly supported by exact counts, documentation, and TimeLimit but is not authoritative, so no reconstructed episodes were asserted. Checkpoint architecture mapping is MATCH: 598 array paths plus three Optax EmptyState sentinels equal all 601 metadata paths both before and after zero-gradient update. The matching-target Orbax/TensorStore restore still exceeded 300 seconds; the Python alarm did not interrupt the native call, so it was manually stopped after the bounded window. No state or inference returned. Package, Demo, checkpoint, external SERL, reward, and success code remained unchanged.

### Rollback method

Stop using/remove only the M1D-B2 project audit script, raw JSON, and three reports, then revert the M1D-B2 status/addendum entries. Preserve all frozen delivered artifacts and earlier M1D-B evidence.
## 2026-08-23 — M1D-B3 OBJECT CHECKPOINT READ-STALL DIAGNOSIS

### Date

2026-08-23

### File

- `scripts/diagnose_object_checkpoint_read_stall.py`
- `results/object_m1d_b3/*.json`
- `docs/object_checkpoint_read_stall.md`
- `docs/object_m1d_b3_report.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

Architecture mapping was exact, but matching-target restore exceeded 300 seconds with no known storage layer, runtime drift, or resource behavior.

### Modification

Added project-side read-only OCDBT inventory, isolated TensorStore leaf reads, bounded Orbax metadata and target-free Flax compatibility tests, and `/proc` CPU/RAM/I/O/state monitoring. Recorded producing/current runtime comparison and synchronized reports/status.

### Reason

Localize the restore stall without modifying or converting the checkpoint, package, SERL, Demo, or dependencies.

### Test command

See `docs/object_checkpoint_read_stall.md`; all reads used `scripts/diagnose_object_checkpoint_read_stall.py` with 20/30/60-second bounds.

### Test result

PARTIAL. Storage integrity and isolated reads PASS. Full-tree Orbax paths TIMEOUT with native-wait behavior. Restore/inference remain blocked.

### Rollback method

Stop using/remove only the new project-side script, raw M1D-B3 JSON, and two reports, then revert the M1D-B3 status entries. Delivered artifacts and environments were not modified.
## 2026-08-23 — M1D-B4 OBJECT PRODUCING-RUNTIME CHECKPOINT RESTORE

### Date

2026-08-23

### File

- `results/object_m1d_b4/*.json`
- `docs/object_producing_runtime_restore.md`
- `docs/object_m1d_b4_report.md`
- `docs/object_checkpoint_closure.md`
- `docs/object_checkpoint_read_stall.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `docs/change_log.md`

### Original behavior

TensorStore 0.1.51 was the only known producing/current drift and the primary untested restore hypothesis.

### Modification

Cloned `serl` into isolated `object_ckpt_compat`, installed only TensorStore 0.1.78 with `--no-deps`, verified the equality gate and original freeze hash, then repeated bounded read-only storage, metadata, no-target, and matching-target tests. Recorded project acceptance of the unchanged Object Demo without upgrading success provenance.

### Reason

Test the producing-runtime hypothesis without changing the original environment or delivered artifacts.

### Test result

PARTIAL. Environment reproduction and isolated reads PASS. Orbax metadata and both full-tree restore forms TIMEOUT; inference was not run. TensorStore drift is not the sufficient root cause.

### Rollback method

Stop using the retained compatibility environment and remove/revert only M1D-B4 project evidence/documentation if desired. Original `serl`, checkpoint, Demo, Object package, and SERL source were unchanged.
## 2026-08-23 — M1D-C OBJECT GPU TRAINING HANDOFF PREPARATION

### Date

2026-08-23

### File

- `configs/object_grasping/*`
- `training/object_grasping/*`
- `evaluation/object_grasping/*`
- `scripts/verify_object_handoff.py`
- `deliverables/gpu_training_handoff/object_grasping/*`
- `docs/object_gpu_handoff_final.md`
- `docs/object_training_resume.md`
- `docs/object_training_launcher.md`
- `docs/object_checkpoint_closure.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Original behavior

Object GPU preparation was blocked on legacy checkpoint restore and had no project launcher, frozen GPU config, new-checkpoint policy, evaluation entry, or operator handoff.

### Modification

Adopted fresh training and preserved the old checkpoint as a nonblocking legacy artifact. Added the accepted immutable Demo manifest, actual DrQ+Demo configuration, observation adapter, explicit W&B launcher boundary, GPU scripts/spec/verifier, evaluation entry, and fresh legacy-Flax checkpoint roundtrip. Documented rather than overstated the incomplete resume integration.

### Reason

Prepare reproducible WSL2 GPU baseline training without changing delivered algorithm/environment/Demo or relying on the unresolved old checkpoint.

### Test result

PARTIAL. Launcher dry run, Demo/contract checks, fresh 598-leaf checkpoint save/restore/inference, evaluation reset+one-step, shell syntax and CPU-mode handoff verifier PASS. Resume bundle roundtrip is missing, so handoff is not yet READY. No training/GPU smoke/formal evaluation ran.

### Rollback method

Stop using/remove the new project-side Object M1D-C files and revert status additions. Preserve all delivered Object artifacts and historical diagnosis evidence.
## 2026-08-23 — M1D-C2 OBJECT SAFE RESUME BUNDLE

Implemented Object-specific safe resume by reusing the validated Cable sidecar primitives and adding Object config/Demo/action constraints. Added atomic three-file bundles, strict metadata/hash/W&B validation, replay index/RNG restoration, Demo rebuild, actor random-step continuation, learner/W&B offsets, collision protection, launcher integration, and cross-process bounded evidence under `runtime/object_handoff/resume_roundtrip_m1dc2/`.

Test result: VALIDATED. Step 123/actor 17 restored; online replay and 2,001 Demo replay validated; step 125 committed without changing step 123; six failure cases failed closed; one restored `(4,)` inference passed. No learner loop, training, GPU smoke, formal evaluation, package/SERL/Demo/legacy-checkpoint mutation, or dependency change occurred.

Rollback: stop using/revert the project-side Object resume module, roundtrip test and launcher/docs changes. Preserve delivered artifacts and runtime evidence.

## 2026-08-23 — M1D-D PEG DEMO + ALGORITHM VALIDATION

### File

- `docs/peg_demo_validation.md`
- `docs/peg_algorithm_mapping.md`
- `docs/peg_checkpoint_audit.md`
- `docs/peg_demo_quality_summary.md`
- `docs/peg_m1d_d_report.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Original behavior

Peg had only environment lifecycle validation; Demo contents, success, algorithms, replay and checkpoints were unverified.

### Modification

Performed read-only archive/source hashes, audited all 20 NPZ files and both PKLs, proved exact cross-dataset mappings, validated source-derived success/reward semantics, mapped RLPD/SAC/BC and both checkpoint groups, audited 20 resets, attempted one bounded primary restore, and tested the exact primary replay insertion path. No package, SERL, Demo, checkpoint, reward or success source was modified.

### Test commands and result

Read-only Python/NumPy audits: PASS (20/20 Demo success; exact 5,512 and 2,388 mappings). EGL 20-reset audit: PASS, fixed. BC stack: PASS. RLPD checkpoint_9000 restore: TIMEOUT at 120 s. Official replay datastore: FAIL on first insert with existing invalid `super(...)` call. Overall `PEG_M1D_D_STATUS = PARTIAL`.

### Rollback method

Remove the five new reports and revert only the M1D-D status/task/change-log entries. Delivered archives, package, Demo, checkpoints, environments and SERL remain unchanged.

## 2026-08-23 — M1D-D1 PEG REPLAY + FRESH INFRASTRUCTURE CLOSURE

### Files

- `training/peg_insertion/{peg_replay_adapter.py,peg_runtime_adapter.py,checkpoint.py}`
- `configs/peg_insertion/peg_rlpd_baseline_seed0.yaml`
- `scripts/validate_peg_m1d_d1.py`
- `docs/peg_replay_repair.md`, `docs/peg_portable_runtime.md`
- `docs/peg_new_checkpoint_strategy.md`, `docs/peg_m1d_d1_report.md`
- `docs/peg_checkpoint_audit.md`, `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Modification and reason

Re-scoped the delivered checkpoint as nonblocking legacy. Added a scoped repair for the installed SERL sibling-class `super(...)` typo, username-independent runtime asset discovery/symlink/XML rewriting, legacy-Flax fresh-state I/O, a frozen RLPD baseline config, and bounded validation. This closes actual fresh-training blockers without altering delivered or external sources.

### Tests and result

`validate_peg_m1d_d1.py replay`: PASS, 5,512 inserts and exact 32+32 schema. `portability`: PASS, current lifecycle and synthetic-root resolution; generated XML has no username literal. `checkpoint`: PASS, fresh save/rebuild/restore, 198 exact leaves, one finite bounded `(7,)` inference. No training or learner update ran. `PEG_M1D_D_STATUS = PASS_FOR_FRESH_GPU_TRAINING`.

### Rollback

Stop using/remove only the new project-side Peg training/config/validation/report files and runtime symlink/roundtrip evidence, then revert the synchronized status entries. Preserve all package, SERL, Demo, reward/success and delivered legacy checkpoints.

## 2026-08-23 — M1D-E PEG GPU TRAINING HANDOFF PREPARATION

### Files

Added the Peg launcher/resume, evaluation/resume-roundtrip, handoff verifier, environment config/spec, 22-asset manifest, seven operator scripts, README/output contract and three M1D-E documents. Updated the frozen Peg config, checkpoint retention helper, project status/tasks/change log, and preserved pre-change config/checkpoint helper backups.

### Modification and reason

Prepared a no-code-edit GPU workflow around the validated fresh RLPD contract. Runtime and replay fixes activate automatically; W&B identity and GPU backend are explicit; checkpoints include replay and metadata sidecars; evaluation uses source semantics and `control_dt=0.02`. Legacy checkpoints remain nonblocking.

### Tests

Python compile, shell syntax, JSON/YAML parsing PASS. CPU handoff verifier PASS. Launcher dry-run populated 5,512 transitions and passed all contracts. Evaluation reset+one-step PASS. Cross-process resume v2 restored exact agent/replay/RNG/W&B state, rebuilt Demo, inferred once, saved 125 while preserving 123, and all six failures failed closed. The first `keep=2` attempt correctly failed old-checkpoint preservation; evidence was retained and the backed-up helper changed to keep 20. No learner update, GPU smoke, formal training/evaluation, install, or external mutation occurred.

### Rollback

Stop using/remove the project-side M1D-E files and runtime evidence, restore the two explicit backups, and revert status entries. Preserve all delivered packages, Demo and legacy checkpoints.
## 2026-08-23 — M1D-F three-model parallel GPU handoff freeze

**Files:** `deliverables/gpu_training_handoff/` root coordination documents; Cable/Object/Peg `README.md`, `TRANSFER_CHECKLIST.md`, and `GPU_OPERATOR_TASK_SHEET.md`; `configs/peg_insertion/environment_gpu.yaml`; `PROJECT_STATUS.md`; `TASKS.md`; this log.

**Original behavior:** The project planned a serial Cable → Object → Peg operator workflow. The root handoff README, parallel assignment/timeline/tracker, and per-task transfer/task sheets were absent. Cable documentation used an older step order, Object commands were abbreviated, and Peg environment YAML was evidence rather than a directly creatable Conda spec.

**Modification/reason:** Frozen three independent tracks (A/Cable, B/Object, C/Peg), each with self-contained Steps 0–13 including exact command/expected output/stop condition, unique environment and W&B identity, task-only verifier/smoke/train/resume/eval commands, data transfer exclusions, independent returns, and common fail-closed escalation. Converted Peg GPU YAML to an independent `serl_peg_gpu` Conda spec and retained its previous evidence file as `environment_gpu_before_parallel_handoff_20260823.yaml`. No training source, Demo, checkpoint, algorithm, reward, success logic, external package, or credential was modified.

**Test commands/results:** `/home/ubuntu/miniconda3/envs/serl/bin/python scripts/verify_cable_handoff.py` PASS; `MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python scripts/verify_object_handoff.py` PASS (the first invocation without EGL failed at GL initialization and is superseded); `MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher /home/ubuntu/miniconda3/envs/serl_mujoco/bin/python scripts/verify_peg_handoff.py` PASS. README step count 14/14 each, three YAML parse checks PASS, all handoff `.sh` files `bash -n` PASS, credential-content scan PASS, static cross-task execution dependency audit PASS. `CABLE_INDEPENDENCE = PASS`; `OBJECT_INDEPENDENCE = PASS` (shared resume source only, no Cable runtime/order prerequisite); `PEG_INDEPENDENCE = PASS`. No learner, actor, evaluation episode, W&B login/run, checkpoint write, or training was started.

**Result:** `THREE_MODEL_PARALLEL_GPU_HANDOFF_READY = READY`; `MILESTONE_1D = COMPLETE`. GPU-host acceptance and policy results remain UNVERIFIED.

**Rollback:** Restore Peg YAML from its dated backup; remove only the new parallel coordination/checklist/task-sheet documents; revert the three task README and associated status/task/log entries. No data/checkpoint rollback is needed.

## 2026-09-03 — Peg target GPU dependency and verifier acceptance

### Files

- `external/serl/` restored from upstream commit `1fa2af7496be042e43d36120a7c8d1b71f1bfb59`
- `external/serl/serl_launcher/serl_launcher/agents/continuous/bc.py`
- `external/serl/serl_launcher/serl_launcher/agents/continuous/bc_before_peg_state_bc_20260902.py`
- `external/mujoco_env`, `external/mujoco_env.pointer_legacy`
- `runtime/peg_insertion/assets`, `runtime/peg_insertion/assets.pointer_legacy`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Original behavior

The target WSL project retained legacy text pointers for SERL, `mujoco_env`, and Peg runtime assets. Complete SERL source was absent after an interrupted Git clone. The isolated GPU environment could see CUDA but had JAXlib 0.4.34 and lacked the local package/runtime dependencies required by the Peg verifier.

### Modification and reason

Downloaded the exact SERL commit archive from GitHub codeload after Git transport failures, restored the complete source tree, backed up the upstream BC implementation, and applied only the delivered state-only BC patch. Installed `serl_launcher` and the delivered Peg `mujoco_env` as editable local packages, converted legacy absolute-path text pointers into project-relative symlinks while preserving their original text, and installed the precise frozen runtime dependencies needed by SERL and the verifier. JAXlib was aligned to 0.4.35 to match JAX and the CUDA plugin.

### Tests and result

Package imports resolve to `/home/makoto/flexible_line_project/external/...`; `pip check` reports `No broken requirements found.` JAX/JAXlib are 0.4.35, devices are `[CudaDevice(id=0)]`, backend is `gpu`, and a 2048x2048 matrix product returned 2048.0 on `cuda:0`. With `MUJOCO_GL=egl`, `scripts/verify_peg_handoff.py --require-gpu` passed Demo manifest, config, runtime adapter, replay adapter, resume schema, and GPU backend gates, ending with `PEG_HANDOFF_VERIFY: PASS`. The independent environment smoke passed reset and one step with `CONTROL_DT_SECONDS: 0.02` and step API length 5. No learner/actor loop, W&B run, checkpoint write, formal training, or formal evaluation was started.

### Rollback method

Restore the backed-up upstream BC file, replace the two project-relative symlinks with their preserved `.pointer_legacy` files if returning to the old machine layout, and stop using the `serl_peg_gpu` editable installs. Preserve all delivered Demo and checkpoint artifacts.

## 2026-09-03 — Peg bounded GPU smoke to fresh-checkpoint gate

### Files

- `training/peg_insertion/checkpoint.py`
- `training/peg_insertion/checkpoint_before_fix_global_monkeypatch_20260903.py`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md` and their `before_peg_gpu_smoke_checkpoint_20260903` backups

### Original behavior

The first learner launch required the complete SERL `franka_sim` path. After that path was supplied, learner/actor reached the real update path, but the project launcher replaced the shared Flax `checkpoints.save_checkpoint` module attribute and the project checkpoint helper called the replaced attribute again, causing recursive bundle saves at step 0.

### Modification and reason

Kept stable references to the original legacy-Flax save and restore functions when `training.peg_insertion.checkpoint` is imported. This preserves the validated legacy checkpoint implementation while allowing the launcher to intercept only the delivered learner's save call. No external SERL/package source, algorithm, Demo, reward, success logic, frozen config, dependency, or legacy checkpoint was changed. Runtime launches used the restored SERL `franka_sim` path and, for the successful retry, `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid dual-process VRAM preallocation pressure on the 8 GiB RTX 4060.

### Tests and result

Focused monkeypatch regression saved `checkpoint_0` and restored `[1, 2]` even while the public Flax save/restore module attributes were replaced: PASS. W&B online run `2snda1i7` was created; actor completed 150/150; learner received 121 online transitions, verified `(32,20)` online, `(32,20)` Demo and `(64,20)` mixed batches, completed GPU update step 0, and saved `/home/makoto/flexible_line_project/results/checkpoints/peg_rlpd_gpu_smoke_seed0/checkpoint_0`. The checkpoint root also contains `checkpoint_0_online_replay.pkl` and `checkpoint_0_resume_metadata.json`. Per operator instruction, both processes were terminated immediately after checkpoint discovery; no restore or bundle-content validation was run. No residual process remained and GPU memory returned to 36 MiB. Result: checkpoint gate PASS; full 10-update bounded smoke PARTIAL; formal training/evaluation NOT STARTED.

### Rollback method

Restore `training/peg_insertion/checkpoint.py` from `checkpoint_before_fix_global_monkeypatch_20260903.py` and restore the three documentation files from their dated backups. Preserve the new checkpoint, replay sidecar, metadata sidecar, W&B runs and logs as experimental evidence.

## 2026-09-03 — Peg bounded-smoke Resume Gate follow-up

### Files

- `training/peg_insertion/run_peg_training.py`
- `training/peg_insertion/run_peg_training_before_resume_profile_inference_20260903.py`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md` and their `before_peg_resume_gate_20260903` backups

### Original behavior

The generic learner/actor resume scripts do not pass `--smoke`. Resuming the smoke checkpoint therefore selected formal limits, W&B name and output paths even though the checkpoint metadata was bound to `peg_rlpd_gpu_smoke_seed0`. `checkpoint_0` existed, but its metadata/replay hashes, real restore, same-run W&B continuation and a later checkpoint had not been validated.

### Modification and reason

The project launcher now derives the resume profile from the checkpoint metadata's W&B name when `--smoke` is not explicit. Known smoke metadata selects the frozen smoke limits/paths; known formal metadata selects formal limits/paths; an unknown identity fails closed. No delivered learner/actor loop, SERL/package source, optimizer, algorithm, Demo, reward, success logic, dependency, legacy checkpoint or credential was changed.

### Tests and result

`checkpoint_0` metadata and replay sidecar were parsed; the replay SHA-256 matched metadata and contained 121 transitions with actor progress 121. Learner and actor resume dry-runs reported `JAX_BACKEND: gpu` and `PEG_TRAINING_DRY_RUN: PASS`. A real learner resume restored the legacy-Flax model and replay under W&B `resume="must"`, completed one new GPU optimizer update, and created `checkpoint_1`, `checkpoint_1_online_replay.pkl`, and `checkpoint_1_resume_metadata.json`. W&B run `2snda1i7` contains one history row at step 1 with finite `critic.critic_loss=140.46263122558594`, `actor.actor_loss=0.1619262993335724`, entropy, temperature, learning-rate and timer metrics. `checkpoint_1` SHA-256 is `27101212bd805c539b7bbcf14ade00b3cde62610f8d1277aaf16beea1ba522be`; its replay SHA-256 `45cb8b5452228bdb8f1bdf5a7efcd322d88641115e178fda2d710b23785e36bb` matches metadata. A GPU dry-run from `checkpoint_1` passed. Actor continuation from metadata step 121 completed 29/29 bounded steps.

A subsequent learner resume from `checkpoint_1` did not complete its next optimizer update: one JAX/XLA worker remained at approximately 100% of one CPU core for more than ten minutes, no `checkpoint_2` appeared, and the process was terminated. No residual learner/actor process remains. Result: resume restore, same-run W&B metrics, original-checkpoint preservation, new bundle save and new-bundle load are PASS; complete 10-update bounded smoke is PARTIAL; formal training/evaluation are NOT STARTED.

### Rollback method

Restore `training/peg_insertion/run_peg_training.py` from `run_peg_training_before_resume_profile_inference_20260903.py` and the three documentation files from their `before_peg_resume_gate_20260903` backups. Preserve `checkpoint_0`, `checkpoint_1`, sidecars, W&B run/logs and terminal evidence unless a separate cleanup is authorized.

## 2026-09-03 — Peg bounded GPU smoke completion

### Files

- `training/peg_insertion/run_peg_training.py` and dated pre-diagnostic/pre-workaround backups
- `scripts/continue_peg_bounded_smoke.sh`
- `scripts/verify_peg_bounded_smoke_result.py`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md` and their `before_peg_bounded_smoke_complete_20260903` backups

### Original behavior

The actor gate and checkpoint/resume gate passed, but a resumed learner remained in one GPU update for more than ten minutes. Batch signatures were correct and stable, checkpoint states were finite, independent GPU matrix computation passed, and no restore, replay, W&B, or checkpoint operation occurred inside the blocked interval.

### Modification and reason

Added opt-in update-boundary diagnostics and strictly bounded resume controls to the project launcher. `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` avoids the identified Triton GEMM path. Each resume process performs one optimizer update, synchronizes its outputs, and writes the normal Flax/replay/metadata bundle immediately afterward. An explicit recorded safe PRNG key is accepted only under diagnostic learner-resume/no-periodic-save mode and is persisted in `source_snapshot`. The continuation script enforces timeouts, bounded retries, and three-file existence checks. No external SERL/package source, dependency, frozen RLPD architecture, loss, optimizer, learning rate, batch size, replay ratio, Demo, environment, reward, success logic, credential, or legacy checkpoint was changed.

### Tests and result

`checkpoint_2` through `checkpoint_9` were generated by synchronized real GPU optimizer updates. Final verification reports `CHECKPOINT_STEP: 9`, `LEARNER_UPDATES: 10`, `ONLINE_REPLAY_SIZE: 121`, replay SHA-256 PASS, 198 finite array leaves, and `PEG_BOUNDED_SMOKE_RESULT_VERIFY: PASS`. A GPU resume dry-run from `checkpoint_9` reports `[CudaDevice(id=0)]`, backend `gpu`, and `PEG_TRAINING_DRY_RUN: PASS`. W&B run `2snda1i7` is `finished`, with online actor/critic/temperature/learning-rate/timer metrics at steps 1–9. Actor evidence remains 150/150 (initial 121 plus resumed 29). No residual learner/actor process remains. Result: `PEG_BOUNDED_GPU_SMOKE = PASS_WITH_DIAGNOSTIC_GPU_WORKAROUND`; nominal uninterrupted/formal training and evaluation remain unvalidated and were not started.

### Rollback method

Restore `training/peg_insertion/run_peg_training.py` from the appropriate dated backup, stop using the two bounded-smoke helper scripts, and restore the three documentation files from their `before_peg_bounded_smoke_complete_20260903` backups. Preserve `checkpoint_9`, its sidecars, W&B run/logs, and earlier diagnostic evidence unless separate cleanup is authorized.

## 2026-09-03 — Peg formal-readiness profile isolation and candidate Gate

### Files

- `training/peg_insertion/run_peg_training.py`
- `configs/peg_insertion/peg_rlpd_baseline_seed0.yaml`
- `configs/peg_insertion/runtime_gpu_formal_candidate.env`
- `scripts/run_peg_formal_candidate_smoke.sh`
- `scripts/verify_peg_formal_candidate.py`
- `scripts/continue_peg_bounded_smoke.sh`, `scripts/verify_peg_handoff.py`
- Peg handoff learner/actor fresh/resume scripts and operator README
- `docs/peg_training_launcher.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`
- corresponding `before_formal_readiness_hardening_20260903` backups for every pre-existing modified file

### Modification and reason

Added required `formal`, `smoke`, and `diagnostic` profiles. `diagnose-updates`, `split-utd1`, `diagnostic-rng-from-step`, `diagnostic-rng-key`, `diagnostic-no-save`, `diagnostic-save-after-update`, `step-indexed-rng`, and `max-resume-updates` are diagnostic-only; formal and smoke reject them before runtime setup, and diagnostic rejects an empty diagnostic request. Formal and smoke continue to use the existing checkpoint RNG without override. Existing bounded-smoke continuation is explicitly diagnostic; formal launch scripts explicitly select formal.

Added an isolated formal-candidate configuration and runner. The runtime candidate file contains exactly `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, and `MUJOCO_GL=egl`. The candidate builds a fresh SAC agent, restores only the hash-validated bounded-smoke online replay, launches one learner process, calls the original `SACAgent.update_high_utd`, records sample/update/total time plus finite losses and RNG chaining for every completed update, rejects diagnostic controls, and terminates safely when one update exceeds 120 seconds. There is no process restart or retry. Candidate checkpoints, W&B files and result evidence use independent paths. No algorithm, network, optimizer, learning rate, batch, Demo, reward, success logic, external dependency or formal runtime was changed.

### Tests and result

Python compilation and shell syntax checks passed. Formal plus `--diagnose-updates`, smoke plus `--max-resume-updates`, and diagnostic without a diagnostic control all failed closed as required. The candidate launched W&B run `tzv5bsvz`, reported JAX backend `gpu` on `CudaDevice(id=0)`, loaded 5,512 Demo transitions and the integrity-checked 121-transition online replay, and used exact 32 online + 32 Demo batches. Update 1/20 completed in 4.928487 seconds with finite actor/critic/temperature losses; sample and total times were 0.086712 and 5.016801 seconds. Normal RNG advanced from `[1278412471, 2182328957]` to `[595175366, 1025945940]`. Normal periodic saving created independent `checkpoint_0`, replay and metadata files with SHA-256 values `7b95c0a47c46363ebeb416ffa92e6447698963466a09fd3ee6ccedf3d0fd9029`, `21fe7eeb0885b0ba3f75d1daa3633499620b7e2f6051beabeca5f5c5fa6f9561`, and `037fc25cb4485f70a96c13122f72d5e868182408b2a73c3c3223d9ba76c80e45`.

The same process did not return from update 2/20 before the 120-second hard limit. The watchdog wrote `reason=update_timeout`, `failed_update=1`, and exited 124. The wrapper printed `PEG_FORMAL_CANDIDATE_GPU_GATE=FAIL`; no retry occurred and no residual learner process remains. The independent `checkpoint_0` bundle passes a JAX GPU resume dry-run. W&B currently reports the forcibly terminated run as `running` with no synced history. Result: `PEG_FORMAL_CANDIDATE_GPU_GATE = FAIL`; formal remains blocked and was not started. The fact that update 2 began with the same `[595175366, 1025945940]` key used by the successful per-process diagnostic workaround strengthens the conclusion that process continuity/JAX execution state, rather than merely selection of that RNG key, is the unresolved risk.

### Rollback method

Restore every pre-existing modified file from its adjacent `before_formal_readiness_hardening_20260903` backup. Remove the three newly added candidate files only if rollback is explicitly authorized. Preserve candidate result evidence, checkpoint bundle and W&B identity unless a separate cleanup is authorized.

## 2026-09-03 — Peg candidate checkpoint-0 next-update isolation

### Files

- `training/peg_insertion/run_peg_training.py`
- `scripts/diagnose_peg_candidate_checkpoint0_update2.sh`
- `docs/peg_candidate_update2_diagnostic_20260903.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`
- adjacent `before_continuous_update2_diagnostic_20260903` backups for every modified pre-existing file

### Original behavior

The formal-candidate run completed update 1, wrote its normal step-0 checkpoint, and then timed out in update 2. That observation mixed same-process continuity, W&B activity and a preceding checkpoint save, so it did not identify which condition was necessary for the stall. Diagnostic resume did not accept candidate-bound checkpoint metadata, and diagnostic stack/W&B isolation controls were unavailable.

### Modification and reason

Candidate-bound checkpoints can now be inspected only under the explicit diagnostic profile while retaining their candidate config identity and output binding. Added diagnostic-only W&B disable and timed Python stack-dump controls; formal and smoke reject them through the existing fail-closed diagnostic argument gate. Added a one-shot script that restores candidate `checkpoint_0`, preserves its RNG, disables W&B and checkpoint writes, executes at most one original learner update, samples GPU status and terminates after 130 seconds without retry. Normal formal/smoke/candidate RNG and training behavior were not changed.

### Tests and result

Python compilation, shell syntax, formal fail-closed and candidate-checkpoint diagnostic GPU dry-run passed. The bounded run restored 121 online transitions, rebuilt 5,512 Demo transitions and formed the exact 32+32 batch. Sampling and concatenation completed. The original update returned asynchronous results, after which the main thread remained at `jax.block_until_ready(new_agent)` in both the 30-second and 90-second stack dumps. The process timed out with exit 124. GPU memory stabilized at 196 MiB; 10-second samples showed mostly 4–8% utilization and approximately 2.3 W. W&B was not initialized, no checkpoint was written, source checkpoint timestamps remained unchanged and no learner process remained.

Result: `PEG_CANDIDATE_CHECKPOINT0_NEXT_UPDATE_FRESH_PROCESS = FAIL_TIMEOUT`. The failure is reproducible outside the original continuous process and without W&B/checkpoint writes. Same-process continuity, W&B logging and the step-0 checkpoint save are therefore not sufficient explanations. The blocking location is JAX device-result synchronization, but compilation/runtime/driver attribution remains UNRESOLVED. Formal training remains BLOCKED.

### Rollback method

Restore `run_peg_training.py`, `PROJECT_STATUS.md`, `TASKS.md` and `docs/change_log.md` from their `before_continuous_update2_diagnostic_20260903` backups. Stop using the new diagnostic script/report. Preserve runtime diagnostic evidence unless a separate cleanup is authorized.

## 2026-09-03 — Peg JAX/XLA execution-layer A/B differential

### Files

- `scripts/peg_jax_xla_differential_probe.py`
- `scripts/run_peg_jax_xla_differential_diagnostic.sh`
- `scripts/analyze_peg_jax_xla_differential.py`
- `docs/peg_jax_xla_execution_differential_20260903.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`
- `before_peg_jax_xla_differential_20260903` backups for every pre-existing documentation file
- `scripts/peg_jax_xla_differential_probe_before_project_import_fix_20260903.py` for the setup-only import correction

### Scope and method

Added isolated diagnostic tooling that freezes one exact 32-online + 32-Demo batch, runs one fresh-initialized original update and one fresh-process candidate-checkpoint restored original update, explicitly synchronizes all returned leaves, and records state/batch/optimizer signatures, timing, compile logs, independent XLA dumps, GPU dmon, CPU-thread snapshots, timed stacks and immutable hashes. Both cases use JAX GPU, normal RNG, disabled Triton GEMM, no W&B, no checkpoint writes, no retry and a 130-second hard timeout. No formal runtime was created and no algorithm, dependency or training configuration was changed.

### Tests and result

The setup-only first invocation failed before agent construction because the standalone probe did not put the project root on `sys.path`; its output was preserved separately, the new probe was backed up, and the import path was corrected before the single actual A/B update trial. A and B then both exited 0. A completed in 5.189362 seconds and B in 5.301514 seconds; explicit synchronization was about 0.012 seconds in both, all losses were finite, and all 99 observed compilations completed in each process. Both cases used the same batch hash `854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c`.

All 198 fresh state leaves are GPU-backed before update, while all 198 restored leaves—including all 150 optimizer leaves—are host NumPy values at the same boundary. Shapes, dtypes and paths are compatible. The A/B `update_high_utd` before/after-optimization HLO, PTX and thunk hashes match exactly. The earlier timeout therefore did not reproduce and no XLA executable or synchronization stall was observed in this trial. Classification: `RESTORE_DEVICE_PLACEMENT_DIFFERENCE`. This is an observed lifecycle difference, not proof of causality for the intermittent timeout. Formal readiness remains FAIL/BLOCKED.

### Rollback method

Restore the three documentation files from their adjacent `before_peg_jax_xla_differential_20260903` backups and stop using the three new diagnostic tools. Preserve the diagnostic result directory unless a separate cleanup is authorized.

## 2026-09-03 — Peg checkpoint restore device-placement causal diagnostic

### Files

- `scripts/peg_restore_placement_causal_probe.py`
- `scripts/run_peg_restore_placement_causal_diagnostic.sh`
- `scripts/analyze_peg_restore_placement_causal.py`
- `docs/peg_restore_placement_causal_diagnostic_20260903.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`
- corresponding `before_peg_restore_placement_causal_20260903` backups for every pre-existing modified documentation file

### Scope and method

Added isolated diagnostic-only tooling without modifying the formal launcher. D1 restores candidate `checkpoint_0` and leaves all train-state leaves on host for the first update. D2 restores the same checkpoint, explicitly device-puts and synchronizes the entire train state, and requires all 198 leaves on `cuda:0`. Each fresh process runs the original update at most five consecutive times, with normal checkpoint RNG, the same frozen 32+32 batch, disabled Triton GEMM and a 120-second per-update hard timeout. W&B, checkpoint writes, fixed diagnostic RNG, retry, formal launch and dependency/algorithm/config changes are absent.

### Tests and result

Both arms were comparable and exited 0 after 5/5 updates. D1 confirmed 198 host NumPy leaves; D2 confirmed 198 GPU JAX leaves after 0.028590 seconds of placement/synchronization. Update 0 completed in 5.061709/5.135037 seconds; subsequent updates completed in 0.013–0.027 seconds. Both recorded 99 compilation starts and 99 finishes, finite losses, and the exact same RNG and loss sequence through final RNG `[3350198582, 799687021]`. No timeout or residual process occurred. Source checkpoint hashes remained unchanged, no W&B/checkpoint diagnostic output exists, and `runtime_gpu_formal.env` remains absent.

Result: `PEG_RESTORE_PLACEMENT_CAUSAL_DIAGNOSTIC = NO_EFFECT`. Explicit pre-placement did not improve or regress the tested five-update stability, so host restore placement is not a sufficient explanation for the earlier intermittent stall. Since neither arm reproduced the failure and the existing 20-update Gate is still failed, formal remains BLOCKED and was not started.

### Rollback method

Restore the three documentation files from their adjacent `before_peg_restore_placement_causal_20260903` backups and stop using the three new causal-diagnostic tools. Preserve the diagnostic result directory unless a separate cleanup is authorized.

## 2026-09-04 — Peg runtime-only continuous update reliability

Added `scripts/peg_runtime_continuous_update_probe.py`, `scripts/run_peg_runtime_reliability_diagnostic.sh`, and `scripts/analyze_peg_runtime_reliability.py`; generated `docs/peg_runtime_reliability_diagnostic_20260904.md` and updated status/task documentation. Existing documentation was backed up as `before_peg_runtime_reliability_20260903` before modification. The formal launcher and existing smoke/candidate artifacts were not modified.

R1 used one fixed 32+32 batch and R2 used the delivered replay iterators and concatenation to resample 32 online + 32 Demo each update. Each fresh process ran 20 original SAC updates without agent resume, RNG override, W&B, checkpoint writes, process restart or retry. Both completed 20/20 with finite losses and continuous normal RNG. Both had one initial update compilation and one index-1 recompilation; the latter took 4.276215/4.306559 seconds (WARNING), followed by 18 NORMAL updates. No update exceeded 30 seconds, no timeout occurred, and resource logs and checkpoint hash preservation were verified.

Final outputs: `PEG_RUNTIME_FIXED_BATCH_20=PASS`, `PEG_RUNTIME_DYNAMIC_BATCH_20=PASS`, `PEG_RUNTIME_RELIABILITY_CLASSIFICATION=RECOMPILE`. An analysis-only branch omitted the case of successful completion plus recompilation, initially yielding UNRESOLVED. The analyzer and original JSON summary were backed up before correcting this branch and reanalyzing existing logs; GPU experiments were not repeated. Formal candidate integration remains FAIL/BLOCKED, with no formal runtime file or formal launch.

Rollback: restore documentation from the adjacent `before_peg_runtime_reliability_20260903` backups; the analysis-only correction has `analyze_peg_runtime_reliability_before_classification_fix_20260904.py`. Stop using the new diagnostic tools if reverting. Preserve evidence unless cleanup is separately authorized.
## 2026-09-05 — Peg F3 lateral -Y command plan freeze

### Files

- `scripts/prepare_peg_lateral_minus_plan.py`
- `scripts/verify_peg_lateral_minus_plan_binding.py`
- `results/validation/peg/f3_lateral_minus_command_plan_20260905/command_plan.json`
- `results/validation/peg/f3_lateral_minus_command_plan_20260905/manifest.json`
- `results/validation/peg/f3_lateral_minus_command_plan_20260905/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Modification and reason

Created a frozen lateral -Y pilot plan from the accepted +Y canonical plan. Commands 107-109 mirror the challenge and 110-112 mirror the correction by negating only `action[1]`; all other commands and the entire phase order are preserved. Added a hash-bound read-only verifier.

### Tests and result

Local and WSL `py_compile` passed. The builder verified the accepted +Y SHA256 and wrote a new exclusive artifact. The dedicated verifier passed the 176-command count, legal phases, 7D finite bounded actions, exact six-command symmetry set, source binding, manifest binding, and frozen artifact hash. No simulator or training ran. Result: `PEG_F3_LATERAL_MINUS_COMMAND_PLAN=PASS`.

### Rollback method

Restore the three documentation files from their adjacent `before_f3_lateral_minus_plan_freeze_20260905` backups and stop using the new plan/scripts. Preserve validation evidence unless cleanup is separately authorized.
## 2026-09-05 — Peg F3 V2 lateral -Y pilot

### Files

- `results/demos/peg_v2_pilot/lateral_minus_y_post_exclusion_20260905_001/`
- `results/diagnostics/peg/f3_v2_lateral_minus_contact_20260905_001/`
- `results/validation/peg/f3_v2_lateral_minus_pilot_20260905/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Tests and result

The frozen-plan verifier and recorder preflight passed before simulator start. One authorized episode consumed the canonical 176-command plan and reached `COMPLETED_SUCCESS`. Existing V2 transition validation passed every row, including continuity, raw sensor reconstruction, binding, reward/success semantics, shape, finiteness and action bounds. The challenge peaked at 5.036 mm, correction reduced lateral error below 0.039 mm, 29 rows met near-hole coverage, final axial error was 2.850 mm, axial tracking ratio was 0.99745, and all contact rows were empty. Result: `PEG_F3_V2_LATERAL_MINUS=PASS`.

No production code or canonical plan changed. The pilot was not promoted to training data, and roll/pitch/training/resume did not start.

### Rollback method

Restore the documentation files from their adjacent `before_f3_v2_lateral_minus_pilot_20260905` backups and stop using the new episode/evidence. Preserve recorded evidence unless cleanup is separately authorized.
## 2026-09-05 — Peg F3 rotation-roll command plan freeze

### Files

- `scripts/prepare_peg_rotation_roll_plan.py`
- `scripts/verify_peg_rotation_roll_plan_binding.py`
- `results/validation/peg/f3_rotation_roll_command_plan_20260905/command_plan.json`
- `results/validation/peg/f3_rotation_roll_command_plan_20260905/manifest.json`
- `results/validation/peg/f3_rotation_roll_command_plan_20260905/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Modification and reason

Built a frozen local-roll plan from the accepted standard V2 trajectory. The standard prefix and insertion/success suffix remain unchanged. Four normal 7D roll actions plan a 0.06981317-rad challenge; five active opposite roll actions exactly cancel it, followed by five settle actions. Added a hash-bound read-only verifier.

### Tests and result

Local/WSL compilation passed. The exclusive builder verified the standard trace SHA256 and frozen protocol. The canonical verifier passed all 179 commands for legal phases, 7D shape, finite bounded actions, exact unchanged prefix/suffix, challenge range, five-transition correction, source binding, manifest binding and artifact hash. Result: `PEG_F3_ROTATION_ROLL_COMMAND_PLAN=PASS`. No simulator or training ran.

### Rollback method

Restore documentation from the adjacent `before_f3_rotation_roll_plan_freeze_20260905` backups and stop using the new scripts/artifacts. Preserve validation evidence unless cleanup is separately authorized.
## 2026-09-05 — Peg F3 V2 rotation-roll pilot

### Files

- `results/demos/peg_v2_pilot/rotation_roll_post_exclusion_20260905_001/`
- `results/diagnostics/peg/f3_v2_rotation_roll_contact_20260905_001/`
- `results/validation/peg/f3_v2_rotation_roll_pilot_20260905/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Tests and result

The canonical roll verifier and recorder preflight passed before simulator start. One authorized episode consumed the frozen 179-command plan and reached `COMPLETED_SUCCESS`. Existing V2 validation passed every transition, including continuity, raw sensor reconstruction, bindings, reward/success semantics, shape, finiteness, and action bounds. The runtime challenge peaked at 0.07496409 rad and correction reduced orientation error to 0.01331020 rad. Twenty-nine rows met near-hole coverage, final orientation was 0.01283377 rad, axial tracking ratio was 0.99568, and all contact rows were empty. Result: `PEG_F3_V2_ROTATION_ROLL=PASS`.

No production code or canonical plan changed. The pilot was not promoted to training data, and pitch/training/resume did not start.

### Rollback method

Restore the documentation files from their adjacent `before_f3_v2_rotation_roll_pilot_20260905` backups and stop using the new episode/evidence. Preserve recorded evidence unless cleanup is separately authorized.
## 2026-09-05 — Peg F3 rotation-pitch command plan freeze

### Files

- `scripts/prepare_peg_rotation_pitch_plan.py`
- `scripts/verify_peg_rotation_pitch_plan_binding.py`
- `results/validation/peg/f3_rotation_pitch_command_plan_20260905/command_plan.json`
- `results/validation/peg/f3_rotation_pitch_command_plan_20260905/manifest.json`
- `results/validation/peg/f3_rotation_pitch_command_plan_20260905/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Modification and reason

Built a frozen local-pitch plan using the accepted standard trace and verified roll plan as hash-bound structural inputs. The standard prefix and insertion/success suffix remain exact. Four normal 7D pitch actions plan a 0.06981317-rad challenge; five opposite pitch actions exactly cancel it, followed by five settle actions. Added a hash-bound read-only verifier.

### Tests and result

Local and WSL compilation passed. The exclusive builder verified the source trace, roll structure, and frozen protocol. The canonical verifier passed all 179 commands for legal phases, 7D shape, finite bounded actions, exact unchanged prefix/suffix, roll phase-structure equivalence, challenge range, five-transition correction, source/manifest binding, and artifact hash. Result: `PEG_F3_ROTATION_PITCH_COMMAND_PLAN=PASS`. No simulator, recording, training, or resume ran.

### Rollback method

Restore documentation from the adjacent `before_f3_rotation_pitch_plan_freeze_20260905` backups and stop using the new scripts/artifacts. Preserve validation evidence unless cleanup is separately authorized.
## 2026-09-05 — Peg F3 V2 rotation-pitch pilot

### Files

- `results/demos/peg_v2_pilot/rotation_pitch_post_exclusion_20260905_001/`
- `results/diagnostics/peg/f3_v2_rotation_pitch_contact_20260905_001/`
- `results/validation/peg/f3_v2_rotation_pitch_pilot_20260905/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Tests and result

The canonical pitch verifier and recorder preflight passed before simulator start. One authorized episode consumed the frozen 179-command plan and reached recorder status `COMPLETED_SUCCESS`. Existing V2 validation passed every transition, including continuity, raw sensor reconstruction, bindings, reward/success semantics, shape, finiteness, and action bounds. The correction, near-hole, final success, truncation, contact, and axial tracking Gates passed. However, the realized pitch challenge peaked at 0.05148208 rad, below the frozen 0.06-rad lower bound. Result: `PEG_F3_V2_ROTATION_PITCH=FAIL` and `FIVE_PILOT_AUDIT=BLOCKED`.

No retry, production/code/plan change, training, resume, or dataset promotion occurred.

### Rollback method

Restore the documentation files from their adjacent `before_f3_v2_rotation_pitch_pilot_20260905` backups and stop using the new episode/evidence. Preserve recorded evidence unless cleanup is separately authorized.
## 2026-09-06 — Peg F3 pitch challenge calibration and revised plan freeze

### Files

- `scripts/prepare_peg_rotation_pitch_plan_v2.py`
- `scripts/verify_peg_rotation_pitch_plan_v2_binding.py`
- `results/validation/peg/f3_rotation_pitch_revised_plan_20260906/command_plan_v2.json`
- `results/validation/peg/f3_rotation_pitch_revised_plan_20260906/manifest.json`
- `results/validation/peg/f3_rotation_pitch_revised_plan_20260906/REPORT.md`
- `PROJECT_STATUS.md`, `TASKS.md`, `docs/change_log.md`

### Modification and reason

Read-only calibration of the completed pitch `_001` episode showed a finite, monotonic response but a 0.73742645 realized/planned ratio. Five challenge steps are the smallest integer revision whose evidence-scaled peak enters the frozen 0.06-0.08-rad interval. The revised plan uses five `+1.0` challenge actions and seven correction actions (`6×-0.8`, `1×-0.2`) whose cumulative pitch command exactly cancels the challenge.

### Tests and result

The calibration binds exact hashes for the old plan, transition trace, contact rows, and observer telemetry. All 179 observer rows were complete and error-free; challenge response was strictly monotonic, finite, contact-free, and below 1.462 mm lateral error. Local/WSL compilation and the canonical 182-command verifier passed phase, 7D/finite/bounds, prefix/suffix preservation, non-pitch component preservation, exact cancellation, calibration, manifest, and artifact-hash Gates. Result: `PEG_F3_PITCH_CHALLENGE_CALIBRATION=PASS` and `ROTATION_PITCH_RERECORD=READY`.

No old artifact, production semantics, simulator, training, resume, promotion, or Five-Pilot Audit changed or ran.

### Rollback method

Restore documentation from the adjacent `before_f3_pitch_calibration_revised_plan_20260906` backups and stop using the new scripts/artifacts. Preserve calibration evidence unless cleanup is separately authorized.
