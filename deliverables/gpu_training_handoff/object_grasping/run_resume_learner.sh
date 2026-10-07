#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT_PATH:?}"; : "${WANDB_RUN_ID:?}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
env MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher "$PY" -m training.object_grasping.run_object_training --role learner --resume-from "$CHECKPOINT_PATH" --wandb-run-id "$WANDB_RUN_ID"
