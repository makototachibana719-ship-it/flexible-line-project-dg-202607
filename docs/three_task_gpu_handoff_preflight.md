# Three-Task GPU Handoff Preflight

Date: 2026-08-22  
Milestone: `MILESTONE 1D — THREE TASK GPU TRAINING PREPARATION`

## Strategy

This machine is for development, Demo collection, data validation, training/checkpoint preflight, evaluation preparation, and GPU handoff preparation. It must not run formal Cable, Object Grasping, or Peg Insertion training. The retired local Cable 100-update retry must not be launched.

No task may be labelled `GPU_TRAINING_HANDOFF_READY` until every gate below has direct repository evidence. `PREPARED` means a concrete, reviewable entry/config/command exists; `VALIDATED` means an appropriate bounded test passed. Presence alone is insufficient.

## GPU_TRAINING_HANDOFF_READY gate

| Gate | Required evidence |
|---|---|
| A. Environment | `ENVIRONMENT_RUNTIME = VALIDATED` on the selected process runtime. |
| B. Data | Demo present; validation, schema, success semantics, and replay compatibility validated. |
| C. Training | Training entry/config validated; launcher prepared; output isolation validated. |
| D. Monitoring | W&B configuration validated; project and run name explicit. |
| E. Checkpoint | Save path and restore path validated by bounded project-side tests using the intended architecture/format. |
| F. GPU runtime | GPU environment specification and bounded smoke/formal commands prepared. |
| G. Evaluation | Evaluation entry prepared; success/return metrics and final artifact requirements defined. |

## Current readiness

| Process | Current status | Readiness |
|---|---|---|
| Cable Routing | Environment, five Demos, schema/replay, launcher/config, monitoring, full-agent checkpoint roundtrip, safe resume, evaluation entry, portable scripts, and artifact contract have evidence. | `CABLE_GPU_TRAINING_HANDOFF_READY = READY` |
| Object Grasping | Environment lifecycle validated; supplied Demo/checkpoint and DPRL identity remain unvalidated. | `OBJECT_HANDOFF_READINESS = NOT_READY` |
| Peg Insertion | Environment lifecycle validated with project XML staging; Demo/checkpoint/algorithm/reward semantics remain unvalidated. | `PEG_HANDOFF_READINESS = NOT_READY` |

## Cable handoff gap audit

Confirmed evidence:

- Five successful Demo files, 10,521 transitions, SHA256 values, common schema, and replay size 10,526 are validated.
- The project launcher has explicit W&B identity, sequential five-file loading, the `state/front/wrist` adapter, and isolated output paths.
- Online W&B and actor 500/500 completed; learner entered real update code.
- Orbax/TensorStore root cause is validated. Legacy Flax save/restore passed for an equivalent pytree and through the project wrapper.

Closed preparation gaps:

1. Full real Cable agent legacy save/restore, all-leaf equality, and restored policy inference passed in bounded tests.
2. The portable GPU environment specification records the required runtime and source identities.
3. GPU smoke/formal learner and actor commands freeze the config, output paths, seed, resources, and termination procedure.
4. The five-Demo manifest references canonical files and hashes without duplicating Demo data.
5. The independent evaluation entry and checkpoint-selection rules are documented and bounded-tested at the preparation level.
6. Returned-artifact requirements are defined in `docs/gpu_return_artifact_spec.md`.
7. Safe training resume restores the full agent, learner/actor progress, online replay, and W&B identity; rebuilds the validated Demo replay; and protects committed checkpoints from collision.

```text
CABLE_GPU_TRAINING_HANDOFF_READY = READY
CABLE_REMAINING_BLOCKERS = NONE
FULL_GPU_SMOKE = NOT_STARTED
FORMAL_TRAINING = NOT_STARTED
FORMAL_EVALUATION = NOT_STARTED
```

## Object preparation sequence

1. Audit the existing 20-trajectory pickle without mutation.
2. Validate keys, episode/transition counts, shapes, dtypes, finite values, terminal boundaries, and hashes.
3. Trace success from environment source into Demo terminal metadata; do not infer success from reward alone.
4. Insert through the intended visual replay path and record exact size/initialization behavior.
5. Reconcile the project DPRL label with delivered `DrQAgent` and 50/50 demo/online mixing using team/source evidence.
6. Map `checkpoint_4500` to the exact agent architecture and save target.
7. Select and bounded-test the checkpoint save/restore strategy.
8. Build a project-side `images/{front,wrist}` adapter; preserve package source.
9. Prepare launcher/config, explicit W&B identity, isolated outputs, GPU smoke/formal commands, and independent evaluation command.

New Demo recording is considered only if the delivered Demo fails documented validation.

## Peg preparation sequence

1. Audit the 20 NPZ episodes and hash/inventory them.
2. Map NPZ fields to transition PKL and BC PKL schemas.
3. Verify successful-Demo semantics from termination/success/error evidence.
4. Map SAC, RLPD, and BC entries and their replay/data dependencies.
5. Map BC and RLPD checkpoint candidates to architecture/config/step.
6. Bounded-test selected restore paths.
7. Record the effective source reward and README discrepancy without changing reward.
8. Reuse the project XML-output isolation adapter.
9. Prepare launcher/config, explicit W&B identity, isolated outputs, GPU smoke/formal commands, and evaluation command.

## Handoff directory specification

This round defines structure only. It does not copy packages, Demos, or checkpoints.

```text
deliverables/gpu_training_handoff/
├── README.md
├── TRAINING_ORDER.md
├── common/
│   ├── environment_check.md
│   └── returned_artifacts_checklist.md
├── cable/
│   ├── README.md
│   ├── config/
│   ├── demo_manifest.json
│   ├── environment_spec.yaml
│   ├── smoke_command.sh
│   ├── train_command.sh
│   ├── eval_command.sh
│   └── expected_outputs.md
├── object_grasping/
│   └── <same contract>
└── peg_insertion/
    └── <same contract>
```

Manifests reference canonical artifact locations and SHA256 values. Command files must be reviewed text artifacts before executable packaging; placeholders prohibit READY status.
