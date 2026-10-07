#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PY="${PYTHON_BIN:-python}"; cd "$ROOT"
source "$ROOT/scripts/peg_runtime_policy.sh"
set -a
source "$ROOT/configs/peg_insertion/runtime_gpu_formal.env"
set +a
: "${WANDB_RUN_ID:?required learner run binding}"
"$PY" scripts/preflight_peg_formal.py --role actor_join --run-id "$WANDB_RUN_ID" --ip "${LEARNER_IP:-localhost}"
exec "$PY" -m training.peg_insertion.run_peg_training --profile formal --role actor --require-gpu --ip "${LEARNER_IP:-localhost}" --wandb-run-id "$WANDB_RUN_ID"
