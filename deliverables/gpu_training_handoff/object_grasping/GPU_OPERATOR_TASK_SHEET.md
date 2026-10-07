# Operator B — Object Commands

Run from repository root with the Object environment active.

| Action | Exact command |
|---|---|
| Verify | `python scripts/verify_object_handoff.py --require-gpu` |
| Smoke learner | `PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_gpu_smoke_learner.sh` |
| Smoke actor | `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/object_grasping/run_gpu_smoke_actor.sh` |
| Formal learner | `PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_train_learner.sh` |
| Formal actor | `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/object_grasping/run_train_actor.sh` |
| Resume learner | `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<id> PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_resume_learner.sh` |
| Resume actor | `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<id> ACTOR_STEP=<metadata-step> PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/object_grasping/run_resume_actor.sh` |
| Evaluate | `CHECKPOINT_PATH=/abs/checkpoint_N EPISODES=50 OUTPUT_DIR=results/single_skill/object/formal_seed0 PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_eval.sh` |

W&B: `flexible-line-object-grasp`; smoke `object_gpu_smoke_seed0`; formal `object_baseline_seed0`. Formal outputs: `results/checkpoints/object_baseline_seed0/`, `results/logs/object_baseline_seed0/`. Return: `object_training_return/`. Never load legacy `checkpoint_4500`.
