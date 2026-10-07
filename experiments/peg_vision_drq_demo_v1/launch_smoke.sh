#!/usr/bin/env bash
set -u

project=/home/makoto/flexible_line_project
python=/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python
output="$1"
updates="$2"
port="$3"

mkdir -p "$project/$output"
cd "$project" || exit 2
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
"$python" -X faulthandler -u -m experiments.peg_vision_drq_demo_v1.smoke_train \
  --demo results/datasets/peg_vision_demo_v1_20260910_170000_pilot5/peg_vision_demo_v1.pkl \
  --online-data results/datasets/peg_vision_safe_online_20260910_180500/safe_online_visual.pkl \
  --output "$output" \
  --updates "$updates" \
  --batch-size 256 \
  --port "$port" \
  >"$output/stdout.log" 2>&1
code=$?
printf '%s\n' "$code" >"$output/process_exit_code"
exit "$code"
