# Operator C — Peg Commands

Run from repository root after setting `SERL_LAUNCHER_PATH="$PWD/external/serl/serl_launcher"` and `PYTHON_BIN=python`.

| Action | Exact command |
|---|---|
| Verify | `MUJOCO_GL=egl PYTHONPATH="$SERL_LAUNCHER_PATH" python scripts/verify_peg_handoff.py --require-gpu` |
| Smoke learner | `SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python deliverables/gpu_training_handoff/peg_insertion/run_gpu_smoke_learner.sh` |
| Smoke actor | `SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/peg_insertion/run_gpu_smoke_actor.sh` |
| Formal learner | `SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python deliverables/gpu_training_handoff/peg_insertion/run_train_learner.sh` |
| Formal actor | `SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/peg_insertion/run_train_actor.sh` |
| Resume learner | `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<id> SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python deliverables/gpu_training_handoff/peg_insertion/run_resume_learner.sh` |
| Resume actor | `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<id> ACTOR_STEP=<metadata-step> SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python deliverables/gpu_training_handoff/peg_insertion/run_resume_actor.sh` |
| Evaluate | `CHECKPOINT_PATH=/abs/checkpoint_N EPISODES=50 OUTPUT_DIR=results/single_skill/peg/formal_seed0 SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN=python deliverables/gpu_training_handoff/peg_insertion/run_eval.sh` |

W&B: `flexible-line-peg-insertion`; smoke `peg_rlpd_gpu_smoke_seed0`; formal `peg_rlpd_baseline_seed0`. Formal outputs: `results/checkpoints/peg_rlpd_baseline_seed0/`, `results/logs/peg_rlpd_baseline_seed0/`, actor log sibling. Return: `peg_training_return/`.
