#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT_PATH:?}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
env MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher "$PY" -m evaluation.object_grasping.evaluate_policy --checkpoint "$CHECKPOINT_PATH" --episodes "${EPISODES:-50}" --output-dir "${OUTPUT_DIR:-results/single_skill/object/formal_seed0}"
