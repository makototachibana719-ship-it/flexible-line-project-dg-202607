#!/usr/bin/env bash
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05

# W&B 在线记录
export WANDB_MODE=online
export WANDB_SHOW_RUN=true
export WANDB_DIR="$HOME/serl_wandb"

mkdir -p "$WANDB_DIR"

python async_sac_state_sim.py "$@" --learner --critic_actor_ratio 8 --env=AuboPegInsert-State-v0 --exp_name=serl_peg_state_smoke --seed=0 --max_steps=2000 --training_starts=300 --utd_ratio=1 --batch_size=64 --checkpoint_period=500 --checkpoint_path=/home/makoto/serl_checkpoints/peg_insert_sac_smoke
