# Parallel Operator Assignment

| Operator | Task | Own environment | W&B project | Independent return |
|---|---|---|---|---|
| GPU colleague A | Cable Routing | `serl_cable_gpu` (equivalent unique name allowed) | `flexible-line-cable` | `cable_training_return/` |
| GPU colleague B | Object Grasping | `serl_object_gpu` | `flexible-line-object-grasp` | `object_training_return/` |
| GPU colleague C | Peg Insertion | `serl_peg_gpu` | `flexible-line-peg-insertion` | `peg_training_return/` |

Each operator owns only transfer verification, environment/GPU smoke, bounded learner/actor smoke, fresh formal training, safe resume, independent evaluation, and return packaging for their assigned task. They do not run or debug the other two tasks, alter project algorithms/data/configs, select cross-task hyperparameters, or perform project-wide integration.

Start only when the task transfer checklist is complete and its data hashes match. Formal training starts only when `TASK_HANDOFF_VERIFY`, `GPU_BACKEND`, `ENV_SMOKE`, `GPU_SMOKE`, `CHECKPOINT_SAVE_RESTORE`, and `WANDB_ONLINE` are all PASS. Stop immediately on any failed gate, hash/schema mismatch, output collision, missing resume sidecar, W&B identity mismatch, non-finite value, or unexpected process exit.

Every return includes `environment/`, `configs/`, `logs/`, `wandb/`, `checkpoints/`, `evaluation/`, and `training_handoff_report.md`. The report records GPU/VRAM/driver/CUDA, Python/JAX/JAXlib/MuJoCo, commands, start/end and wall-clock time, environment and learner steps, W&B URL/ID, periodic/final/best checkpoint evidence, evaluation summary/episodes/failures, and every resume.
