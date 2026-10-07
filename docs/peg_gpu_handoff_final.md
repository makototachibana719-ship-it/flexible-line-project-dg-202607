# Peg GPU Training Handoff Final

Date: 2026-08-23  
`PEG_GPU_TRAINING_HANDOFF_READY = READY`

## Frozen policy

Fresh initialization is mandatory and delivered RLPD checkpoint_9000 plus BC checkpoints are retained legacy artifacts only. Their restore timeout is nonblocking; they are never auto-loaded or used for resume.

Training uses state RLPD/SAC, `(20,) float32` observations, `(7,) float32` actions, 20 successful NPZ provenance episodes and their exact 5,512-transition PKL. Reward is `-80 lateral -20 axial -2 orientation +2[lateral<0.01] +25[success]`. Success is lateral `<=0.002`, axial `<=0.003`, orientation `<=0.05`; force is excluded and success terminates.

## Prepared components

- single project launcher with runtime/replay adapters and fail-fast assertions;
- SHA-pinned 22-asset Demo manifest without copying data;
- explicit online W&B project and smoke/formal names; credentials remain operator-owned;
- WSL2 GPU environment specification with backend gate;
- fresh checkpoint plus full replay/metadata safe resume and cross-process evidence;
- CPU/GPU handoff verifier;
- bounded smoke, formal, resume and evaluation scripts;
- independent evaluation schema with 0.02-second control timing;
- periodic/final/best selection and returned-artifact contract;
- 13-step operator README with stop conditions.

## Gate matrix

| Gate | Status |
|---|---|
| `PEG_ENVIRONMENT_GATE` | VALIDATED |
| `PEG_DATA_GATE` | VALIDATED |
| `PEG_TRAINING_GATE` | PREPARED / CPU dry-run PASS |
| `PEG_MONITORING_GATE` | PREPARED; live online run is GPU-operator evidence |
| `PEG_NEW_CHECKPOINT_GATE` | VALIDATED |
| `PEG_RESUME_GATE` | VALIDATED |
| `PEG_GPU_SPEC_GATE` | PREPARED; target `--require-gpu` acceptance mandatory |
| `PEG_EVALUATION_GATE` | PREPARED / one-step dry-run PASS; formal result pending trained policy |
| `PEG_PORTABILITY_GATE` | VALIDATED |

All pre-training preparation gates are closed. READY does not claim GPU smoke, training, learned-policy success, or formal evaluation.

## Next milestone

M1D-E is complete. Exact next milestone is M1D-F — three-model GPU handoff package freeze, but it was not entered here.

