#!/usr/bin/env bash
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05

# W&B 在线记录
export WANDB_MODE=online
export WANDB_SHOW_RUN=true
export WANDB_DIR="$HOME/serl_wandb"

mkdir -p "$WANDB_DIR"

python async_sac_state_sim.py "$@" --learner --critic_actor_ratio=1 --env=AuboPegInsert-State-v0 --exp_name=serl_peg_insert_rlpd20_v3_10k_utd1 --seed=0 --max_steps=10000 --training_starts=100 --utd_ratio=1 --batch_size=64 --checkpoint_period=1000 --checkpoint_path=/home/makoto/serl_checkpoints/peg_insert_rlpd_v3_10k_utd1 --demo_path=/home/makoto/serl_data/peg_insert_20_demos_state.pkl
