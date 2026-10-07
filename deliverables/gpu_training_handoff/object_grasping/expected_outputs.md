# Returned Object Artifacts

Return final and periodic checkpoints, evaluation-defined best checkpoint, checkpoint metadata/hashes, W&B URL/run ID/history export, frozen config, exact commands, complete logs, GPU/CUDA/JAX environment record, evaluation summary and per-episode CSV, failure cases, wall-clock runtime, and `training_handoff_report.md`. Missing evidence remains UNVERIFIED.

If resume occurs, the report must also include interruption/resume timestamps, resume checkpoint, learner and actor steps before/after, W&B run ID, replay restore result, downtime, and interruption reason.
