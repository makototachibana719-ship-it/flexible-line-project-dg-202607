# Project Tasks

Last updated: 2026-08-23

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `VALIDATED`.

`DONE` means implementation or documentation is complete. `VALIDATED` requires relevant testing and evidence.

## P0 — Blocker

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
| Peg remote GPU gates/training/evaluation return | TODO | Operator C may proceed independently; fresh RLPD training only and never load legacy checkpoints. |
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
