# GPU Return Artifact Specification

Date: 2026-08-22

Each process must return a `training_handoff_report.md` plus the following evidence. Missing items remain `UNVERIFIED`; they must not be reconstructed from screenshots or memory.

## Model artifacts

- Final checkpoint: last successfully committed scheduled checkpoint at the declared training boundary.
- Best checkpoint: selected by the declared independent evaluation metric and deterministic tie-break.
- Agreed periodic checkpoints and a manifest containing paths, steps, sizes, and SHA256 values.
- Restore verification record for final and best checkpoints.

## Training record

- W&B project, run URL, run ID, and exported history CSV or JSON.
- Complete resolved configuration and exact commands.
- Complete stdout/stderr logs.
- Start/end timestamps, wall-clock runtime, termination reason, environment steps, and learner updates.
- For every interruption/resume: interruption time, resume checkpoint, old/resumed W&B run IDs, learner and actor steps before/after, replay restore strategy, and wall-clock downtime.

## Environment record

- GPU model, driver, CUDA, and cuDNN.
- Python, JAX, JAXlib, Flax, Orbax, TensorStore, Gym, Gymnasium, MuJoCo, W&B, and algorithm-specific versions.
- Conda export or equivalent lock record.
- Source snapshot or Git commit plus package/launcher/config hashes.

## Evaluation record

- Process, checkpoint path/step, seed, and number of episodes.
- Success count and success rate.
- Mean and standard deviation of return.
- Mean episode length and mean execution time.
- Task-specific final metrics and failure categories/cases.
- Raw per-episode results and summary outputs; videos are supplemental, not metric evidence.

## Reproducibility record

- Demo paths and SHA256 values.
- Seed, total environment steps, total learner updates, and checkpoint step.
- Exact evaluation command and success/return definitions.
- Any deviation from the reviewed handoff package, explicitly identified and justified.
- Matching resume metadata and online-replay sidecars for each returned resumable checkpoint.

## Suggested returned layout

```text
<process_return>/
├── training_handoff_report.md
├── checkpoints/
├── checkpoint_manifest.json
├── config/
├── logs/
├── wandb_export/
├── environment/
├── evaluation/
└── videos/
```
