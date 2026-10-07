#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a committed checkpoint_N file}"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
args=(
  --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml
  --role learner
  --resume-from "$CHECKPOINT_PATH"
)
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  args+=(--wandb-run-id "$WANDB_RUN_ID")
fi
"$PYTHON_BIN" -m training.cable.run_cable_training "${args[@]}"
