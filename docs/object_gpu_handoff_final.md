# Object GPU Handoff Final Status

Date: 2026-08-23

## Strategy

- `FRESH_INITIALIZATION = TRUE`
- `LOAD_LEGACY_CHECKPOINT = FALSE`
- `OBJECT_ALGORITHM = DRQ_PLUS_DEMO_REPLAY`
- `TEAM_REPORTED_NAME = DPRL`
- `OBJECT_DEMO_DATA = ACCEPTED_FOR_BASELINE`
- `OBJECT_DEMO_SUCCESS_PROVENANCE = UNVERIFIED`

Legacy `checkpoint_4500` is preserved without deletion or conversion. Its architecture is MATCH, native restore unresolved, and it is not required or a blocker for new training. No further old-checkpoint restore work is planned.

The historical approximately 70% success rate is `SOURCE = TEAM_REPORTED`; `CURRENT_REPOSITORY_REVALIDATION = NOT_REQUIRED`. It is not a new repository experiment.

## Frozen baseline

The delivered configuration evidence freezes seed 0, ResNet-pretrained dual visual encoder, batch 8 split 4 online/4 Demo, replay 10,000, starts/random steps 500, ratio/UTD 1, maximum 5,000, checkpoint 500 and log 50. Factory-default learning rates are preserved. Reset behavior remains fixed TCP with randomized block XY and scene-dependent camera variation; evaluation must retain the real reset distribution.

## Gate table

| Gate | Status | Evidence |
|---|---|---|
| OBJECT_ENVIRONMENT_GATE | VALIDATED | lifecycle plus adapter dry run |
| OBJECT_DATA_GATE | VALIDATED_ACCEPTED | SHA, 2,000 schema/replay; success provenance unverified but accepted |
| OBJECT_TRAINING_GATE | PREPARED | launcher dry run PASS; GPU actor/learner not run locally |
| OBJECT_MONITORING_GATE | PREPARED | explicit online project/names; GPU W&B not run |
| OBJECT_NEW_CHECKPOINT_GATE | VALIDATED | fresh 598-leaf save/rebuild/restore/inference PASS |
| OBJECT_RESUME_GATE | VALIDATED | cross-process step-123→125 full bundle, replay/RNG/Demo/W&B/failures/inference PASS |
| OBJECT_GPU_SPEC_GATE | PREPARED | WSL2 reference spec and GPU stop gate |
| OBJECT_EVALUATION_GATE | PREPARED | reset/one-step PASS; formal checkpoint evaluation not run |
| OBJECT_PORTABILITY_GATE | VALIDATED | project-relative handoff scripts and references |

`OBJECT_GPU_TRAINING_HANDOFF_READY = READY`

All preparation gates are satisfied. READY does not validate target WSL2 GPU runtime, GPU smoke, formal training, policy quality, or formal evaluation; those remain mandatory operator steps.
