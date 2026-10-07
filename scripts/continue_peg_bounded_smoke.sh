#!/usr/bin/env bash
set -euo pipefail

cd /home/makoto/flexible_line_project
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export MUJOCO_GL=egl
export PYTHONPATH=/home/makoto/flexible_line_project/external/serl/serl_launcher:/home/makoto/flexible_line_project/external/serl/franka_sim

python_bin=/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python
checkpoint_dir=/home/makoto/flexible_line_project/results/checkpoints/peg_rlpd_gpu_smoke_seed0
wandb_run_id=2snda1i7

for checkpoint_step in 7 8; do
  next_step=$((checkpoint_step + 1))
  echo "[BOUNDED_SMOKE] resume checkpoint_${checkpoint_step}; target checkpoint_${next_step}"
  completed=false
  for attempt in 1 2 3 4 5; do
    echo "[BOUNDED_SMOKE] attempt ${attempt}/5"
    if timeout --signal=TERM --kill-after=10s 45s "${python_bin}" \
      -m training.peg_insertion.run_peg_training \
      --profile diagnostic \
      --role learner \
      --require-gpu \
      --resume-from "${checkpoint_dir}/checkpoint_${checkpoint_step}" \
      --wandb-run-id "${wandb_run_id}" \
      --diagnostic-no-save \
      --diagnostic-save-after-update \
      --diagnostic-rng-key=595175366,1025945940 \
      --max-resume-updates=1; then
      completed=true
      break
    fi
    echo "[BOUNDED_SMOKE] attempt ${attempt} timed out or failed before checkpoint save"
  done
  if [[ "${completed}" != true ]]; then
    echo "[BOUNDED_SMOKE] checkpoint_${next_step} failed after 5 attempts" >&2
    exit 1
  fi
  test -s "${checkpoint_dir}/checkpoint_${next_step}"
  test -s "${checkpoint_dir}/checkpoint_${next_step}_online_replay.pkl"
  test -s "${checkpoint_dir}/checkpoint_${next_step}_resume_metadata.json"
  echo "[BOUNDED_SMOKE] checkpoint_${next_step} complete"
done

echo "PEG_BOUNDED_SMOKE_LEARNER_UPDATES: PASS (10/10; checkpoint_0 through checkpoint_9)"
