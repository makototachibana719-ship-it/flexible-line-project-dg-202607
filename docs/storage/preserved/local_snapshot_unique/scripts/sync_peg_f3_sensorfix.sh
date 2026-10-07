#!/usr/bin/env bash
set -euo pipefail
src='/mnt/d/桌面/小挑/flexible_line_project/flexible_line_project'
dst='/home/makoto/flexible_line_project'
files=(
external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/utils/sensor_utils.py
external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim/peg_env_adapter.py
configs/peg_insertion/peg_rlpd_baseline_seed0.yaml
training/peg_insertion/checkpoint.py
training/peg_insertion/resume.py
training/peg_insertion/run_peg_training.py
training/peg_insertion/formal_preflight.py
training/peg_insertion/formal_policy.py
evaluation/peg_insertion/evaluate_policy.py
scripts/preflight_peg_formal.py
)
for f in "${files[@]}"; do
  cmp "$dst/$f" "$src/$f.before_f3_sensorfix_20260904"
done
for f in "${files[@]}"; do
  cp -n "$dst/$f" "$dst/$f.before_f3_sensorfix_20260904"
  cp "$src/$f" "$dst/$f"
done
for f in training/peg_insertion/observation_contract.py configs/peg_insertion/legacy_observation_contracts.json scripts/test_peg_f3_sensorfix.py scripts/audit_peg_f3_sensorfix.py; do
  test ! -e "$dst/$f"
  cp "$src/$f" "$dst/$f"
done
