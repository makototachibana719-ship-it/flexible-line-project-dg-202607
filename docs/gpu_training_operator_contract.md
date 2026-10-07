# GPU Training Operator Contract

Date: 2026-08-22

## Scope and authority

The GPU operator may execute only the reviewed per-process environment, smoke, training, checkpoint, and evaluation commands. They must not change reward, success logic, algorithm, Demo contents, checkpoint contents, or package source without a separately reviewed change request.

## Mandatory workflow

1. Record `nvidia-smi`, GPU model, driver, CUDA, and cuDNN information.
2. Create the specified Conda environment and export the resolved package inventory.
3. Record Python and JAX/JAXlib versions and confirm `jax.devices()` reports the intended GPU.
4. Run the process environment smoke test.
5. Verify every Demo SHA256 and run its schema/semantic validator.
6. Run only the bounded GPU smoke command.
7. Confirm explicit W&B project/run identity and online logging.
8. Confirm checkpoint save and restore using the intended full architecture.
9. Start formal training only after smoke acceptance is recorded.
10. Retain scheduled checkpoints under the agreed policy.
11. Run independent evaluation; training metrics do not substitute for evaluation.
12. Return all artifacts specified in `docs/gpu_return_artifact_spec.md`.

If training is interrupted, resume only from a complete process-specific bundle that restores model, progress, required replay state, actor progress, and the same W&B run. Model-only restoration with reset replay/steps/W&B is prohibited.

## Stop conditions

Stop and report `BLOCKED` if any hash differs, an output directory is unexpectedly non-empty, JAX does not select the intended GPU, observation/action schema differs, W&B identity is wrong, save/restore fails, success semantics are unclear, or the reviewed command/config has changed. Preserve logs and partial outputs; do not delete or overwrite them.

## Acceptance boundary

Smoke acceptance requires environment, Demo, replay, real update path, online monitoring, checkpoint save, and checkpoint restore evidence. Formal training acceptance requires returned checkpoints plus independent evaluation metrics. A checkpoint directory or rising reward alone is not acceptance.
