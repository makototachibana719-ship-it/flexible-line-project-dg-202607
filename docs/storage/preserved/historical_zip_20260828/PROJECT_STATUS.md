# Project Status

Last updated: 2026-08-23

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
