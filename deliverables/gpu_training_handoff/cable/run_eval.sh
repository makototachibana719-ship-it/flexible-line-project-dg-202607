#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a legacy Flax checkpoint directory}"
EPISODES="${EPISODES:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-results/single_skill/cable/evaluation_seed0}"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
"$PYTHON_BIN" -m evaluation.cable.evaluate_policy \
  --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml \
  --checkpoint "$CHECKPOINT_PATH" --episodes "$EPISODES" --seed 0 \
  --output-dir "$OUTPUT_DIR"
