#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"
cd "$ROOT"; timeout 1800s env MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher "$PY" -m training.object_grasping.run_object_training --role learner --smoke
