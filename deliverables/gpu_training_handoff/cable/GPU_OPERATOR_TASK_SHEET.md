# Operator A — Cable Commands

Run from repository root with `PYTHON_BIN=python` after activating the task environment.

| Action | Exact command |
|---|---|
| Verify | `python scripts/verify_cable_handoff.py --require-gpu` |
| Smoke learner | `PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_gpu_smoke_learner.sh` |
| Smoke actor | `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/cable/run_gpu_smoke_actor.sh` |
| Formal learner | `PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_train_learner.sh` |
| Formal actor | `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/cable/run_train_actor.sh` |
| Resume learner | `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<id> PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_resume_learner.sh` |
| Resume actor | `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<id> ACTOR_STEP=<metadata-step> PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/cable/run_resume_actor.sh` |
| Evaluate | `CHECKPOINT_PATH=/abs/checkpoint_N EPISODES=50 OUTPUT_DIR=results/single_skill/cable/formal_seed0 PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_eval.sh` |

W&B: `flexible-line-cable`; smoke `cable_gpu_smoke_5demo_seed0`; formal `cable_baseline_5demo_seed0`. Formal outputs: `results/checkpoints/cable_baseline_5demo_seed0/`, `results/logs/cable_baseline_5demo_seed0/`. Return: `cable_training_return/`.
