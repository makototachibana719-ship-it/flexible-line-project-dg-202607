#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"
source "$ROOT/scripts/peg_runtime_policy.sh"
set -a
source "$ROOT/configs/peg_insertion/runtime_gpu_formal.env"
set +a
"$PY" scripts/preflight_peg_formal.py --role fresh_learner
exec "$PY" -m training.peg_insertion.run_peg_training --profile formal --role learner --require-gpu
