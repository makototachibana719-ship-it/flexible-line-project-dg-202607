#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
"$PYTHON_BIN" scripts/verify_cable_handoff.py --require-gpu
"$PYTHON_BIN" -m training.cable.run_cable_training \
  --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml \
  --role learner
