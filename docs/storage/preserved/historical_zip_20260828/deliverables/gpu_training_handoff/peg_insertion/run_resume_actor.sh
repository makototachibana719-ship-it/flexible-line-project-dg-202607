#!/usr/bin/env bash
set -euo pipefail
: "${CHECKPOINT_PATH:?}"; : "${WANDB_RUN_ID:?}"; : "${ACTOR_STEP:?}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
env MUJOCO_GL=egl PYTHONPATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}" "$PY" -m training.peg_insertion.run_peg_training --role actor --require-gpu --ip "${LEARNER_IP:-localhost}" --resume-from "$CHECKPOINT_PATH" --resume-actor-step "$ACTOR_STEP" --wandb-run-id "$WANDB_RUN_ID"
