#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LEARNER_IP="${LEARNER_IP:-localhost}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the learner's committed checkpoint_N file}"
: "${ACTOR_STEP:?Set ACTOR_STEP to actor_steps from the matching resume metadata}"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
args=(
  --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml
  --role actor
  --ip "$LEARNER_IP"
  --resume-from "$CHECKPOINT_PATH"
  --resume-actor-step "$ACTOR_STEP"
)
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  args+=(--wandb-run-id "$WANDB_RUN_ID")
fi
"$PYTHON_BIN" -m training.cable.run_cable_training "${args[@]}"
