#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT_PATH:?}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
env MUJOCO_GL=egl PYTHONPATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}" "$PY" -m evaluation.peg_insertion.evaluate_policy --checkpoint "$CHECKPOINT_PATH" --episodes "${EPISODES:-50}" --output-dir "${OUTPUT_DIR:-results/single_skill/peg/formal_seed0}"
