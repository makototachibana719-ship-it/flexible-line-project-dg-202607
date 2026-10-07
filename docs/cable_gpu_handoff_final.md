# Cable GPU Handoff Finalization

Date: 2026-08-22  
Milestone: `M1D-A — CABLE GPU HANDOFF FINALIZE`

## Frozen facts

```text
ENVIRONMENT_SMOKE = VALIDATED
FORMAL_DEMO_COUNT = 5
VALIDATED_FORMAL_DEMO_COUNT = 5
DEMO_TRANSITIONS = 10521
DEMO_BATCH_VALIDATION = VALIDATED
REPLAY_BUFFER_COMPATIBILITY = VALIDATED
MERGE_COMPATIBLE = YES
WANDB_ONLINE = VALIDATED
ACTOR_UPDATE_PATH = VALIDATED
LEARNER_UPDATE_PATH = VALIDATED
CHECKPOINT_ROOT_CAUSE = ORBAX_TENSORSTORE_MISMATCH
LEGACY_FLAX_SAVE_UNIT_TEST = VALIDATED
LEGACY_FLAX_RESTORE_UNIT_TEST = VALIDATED
FORMAL_TRAINING = NOT_STARTED
FORMAL_EVALUATION = NOT_STARTED
```

## Real-agent checkpoint roundtrip

Bounded output: `runtime/cable_handoff/checkpoint_roundtrip_final/roundtrip_report.json`.

The test constructed the same project training environment, observation/action schema, dual-camera ResNet-pretrained DrQ architecture, optimizer/state tree, and sharding path as the launcher. It saved step 0 with project-side legacy Flax, destroyed the original agent, rebuilt it, restored, and compared all 598 serialized leaf key paths, shapes, dtypes, and hashes. All leaf values matched; restored deterministic inference returned a finite `(3,)` action.

Raw `PyTreeDef` equality is false because its auxiliary data includes newly constructed apply-function/optimizer object identities. This diagnostic does not change the stable serialized key paths or values.

```text
CHECKPOINT_SAVE = PASS
CHECKPOINT_RESTORE = PASS
REAL_AGENT_CHECKPOINT_ROUNDTRIP = VALIDATED
```

## Gate matrix

| Gate | Status | Evidence / boundary |
|---|---|---|
| Environment | VALIDATED | Cable V2 lifecycle in `serl`/EGL. GPU runtime itself remains operator-validated. |
| Data | VALIDATED | Five files, 10,521 transitions, 5 successes, schema/replay/hash manifest. |
| Training | PREPARED | Frozen config and existing launcher; GPU smoke not started. |
| Monitoring | VALIDATED/PREPARED | Online W&B path validated; formal/smoke identity explicit; GPU runs not started. |
| Checkpoint | VALIDATED | Full real-agent legacy save/rebuild/restore/inference roundtrip. |
| GPU runtime specification | PREPARED | Portable base environment and device gate; target GPU runtime unverified. |
| Evaluation | PREPARED | Independent CLI, schema, metrics, and one-step dry run pass; trained checkpoint evaluation unverified. |
| Portability | VALIDATED for prepared files | Scripts derive `PROJECT_ROOT`; handoff config uses relative paths; no training dependency on `/home/ubuntu`. |

## W&B contract

- Project: `flexible-line-cable`
- Formal: `cable_baseline_5demo_seed0`
- GPU smoke: `cable_gpu_smoke_5demo_seed0`
- Mode: online
- Authentication: operator-owned `wandb login`; no API key is stored or inspected.

## Smoke and formal bounds

GPU smoke uses 10 learner updates and 400 actor steps. The actor bound must exceed the unchanged `training_starts=300`; 400 is therefore the smallest practical frozen allowance with margin. Smoke checkpoint/log directories are isolated from formal and historical local evidence. Formal settings retain the existing baseline hyperparameters and 1,000,000 maximum steps.

## Checkpoint and resume policy

Periodic, final, and best definitions are frozen in `expected_outputs.md`. `BEST_CHECKPOINT = UNDEFINED_UNTIL_EVALUATION`. The launcher has no correct update-step/W&B/actor resume contract:

M1D-A2 added fail-closed three-file checkpoint bundles, complete model/online-replay restoration, Demo replay rebuild, learner/W&B step offsets, actor progress validation, collision protection, and W&B `resume="must"`. A cross-process real-agent test resumed step 123 and safely committed step 125 without modifying step 123.

```text
TRAINING_RESUME = VALIDATED
ONLINE_REPLAY_RESUME = RESTORED
DEMO_REPLAY_RESUME = REBUILT
```

## Evaluation boundary

`evaluation.cable.evaluate_policy` defines success as terminal `info["success"]`, return as the sum of environment rewards, and final task metric as terminal `distance_to_slot`. It outputs raw episode CSV and summary JSON. Reset + one zero-action step passed. No trained policy episode ran.

```text
EVALUATION_ENTRY = PREPARED
EVALUATION_RUNTIME = PREPARED / TRAINED_CHECKPOINT_UNVERIFIED
```

## Decision

```text
CABLE_GPU_TRAINING_HANDOFF_READY = READY
REMAINING_BLOCKERS = NONE
```

The target GPU smoke and its checkpoint restore remain mandatory operator acceptance steps after handoff; they are `NOT_STARTED` and do not constitute local policy/training validation. READY means the package is prepared for that GPU-side acceptance, not that training or policy quality is validated.
