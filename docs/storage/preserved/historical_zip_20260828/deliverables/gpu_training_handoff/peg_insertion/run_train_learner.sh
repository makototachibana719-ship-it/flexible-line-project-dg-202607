#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
env MUJOCO_GL=egl PYTHONPATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}" "$PY" -m training.peg_insertion.run_peg_training --role learner --require-gpu
