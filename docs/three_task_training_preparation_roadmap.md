# Three-Task Training Preparation Roadmap

Date: 2026-08-22  
Current milestone: `MILESTONE 1D`

## Ordered milestones

| Milestone | Outcome | Exit criterion |
|---|---|---|
| M1D-A | Cable GPU handoff finalize | All Cable gate items have evidence; no unresolved placeholder or restore/evaluation ambiguity. |
| M1D-B | Object Demo + algorithm validation | Existing Demo schema/semantics/replay and DPRL/DrQ identity are resolved truthfully. |
| M1D-C | Object training handoff preparation | Object launcher/config/W&B/checkpoint/GPU/evaluation contract passes the gate. |
| M1D-D | Peg Demo + algorithm validation | NPZ/PKL semantics, SAC/RLPD/BC mapping, candidates, and actual reward are documented/validated. |
| M1D-E | Peg training handoff preparation | Peg launcher/config/W&B/checkpoint/GPU/evaluation contract passes the gate. |
| M1D-F | Three-model handoff package freeze | All three independently read `GPU_TRAINING_HANDOFF_READY`; manifests and small text/config artifacts are frozen. |

Stages must not be skipped. Readiness for one model does not promote another.

## Training order on the GPU host

1. Cable Routing
2. Object Grasping
3. Peg Insertion

Cable is first because its validated Demo, replay, adapter, launcher, and W&B path provide the strongest GPU-host integration check. This ordering does not authorize Cable training before M1D-A acceptance or package freeze.

## Parallel local work during GPU training

- Unified three-task evaluation harness and report schema.
- Policy rollout video exporter and success/failure extraction.
- Failure taxonomy and robustness test plan.
- W&B result parser and model-comparison table generator.
- Randomized-reset ablation design only; no implementation in M1D.
- Unified Skill API, Skill Manager, and Order Manager design.
- Final competition deliverable structure.

All reported results must originate from returned artifacts under the common contract.
