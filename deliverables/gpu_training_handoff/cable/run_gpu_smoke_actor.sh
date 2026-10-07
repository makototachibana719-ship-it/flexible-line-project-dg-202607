#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LEARNER_IP="${LEARNER_IP:-localhost}"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
timeout --signal=TERM 30m "$PYTHON_BIN" -m training.cable.run_cable_training \
  --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml \
  --role actor --smoke --ip "$LEARNER_IP"
