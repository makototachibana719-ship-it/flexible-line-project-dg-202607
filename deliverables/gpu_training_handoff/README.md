# Three-Model Parallel GPU Training Handoff

`THREE_MODEL_PARALLEL_GPU_HANDOFF_READY = READY`

This freeze assigns three independent tracks. Operator A runs only [Cable](cable/README.md), Operator B only [Object Grasping](object_grasping/README.md), and Operator C only [Peg Insertion](peg_insertion/README.md). A task may begin formal training as soon as its own verifier, GPU backend, environment smoke, bounded GPU smoke, checkpoint save/restore, and online W&B gates pass. It never waits for another task.

Start with [operator assignment](PARALLEL_OPERATOR_ASSIGNMENT.md), use the [parallel timeline](PARALLEL_TRAINING_TIMELINE.md), and copy the [tracker](GPU_TRAINING_TRACKER_TEMPLATE.md) for project-side monitoring. Each operator needs only their task directory plus the files listed by its `TRANSFER_CHECKLIST.md`.

Do not run a three-model verifier as an operator gate. Do not share W&B API-key files. Do not change algorithms, rewards, success logic, Demos, frozen configs, external SERL, or external environments. On any verifier, hash, environment, replay, checkpoint, W&B, or resume failure: stop and return the command, complete error/log, and runtime/GPU inventory.

Returns may arrive independently as `cable_training_return/`, `object_training_return/`, and `peg_training_return/`; the project lead later assembles `gpu_training_return/{cable,object_grasping,peg_insertion}/`.
