#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export TF_ENABLE_ONEDNN_OPTS=0

python async_drq_sim.py \
  --actor \
  --exp_name=serl_dev_drq_sim_test_resnet \
  --seed=0 \
  --random_steps=100 \
  --encoder_type=resnet-pretrained \
  --debug \
  "$@"
