#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT_PATH:?}"; : "${WANDB_RUN_ID:?}"; : "${ACTOR_STEP:?}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
env MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher "$PY" -m training.object_grasping.run_object_training --role actor --ip "${LEARNER_IP:-localhost}" --resume-from "$CHECKPOINT_PATH" --resume-actor-step "$ACTOR_STEP" --wandb-run-id "$WANDB_RUN_ID"
