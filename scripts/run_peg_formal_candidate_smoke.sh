#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "PEG_FORMAL_CANDIDATE_GPU_GATE=FAIL: candidate command exited with status ${status}" >&2; exit "$status"' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
ENV_FILE="$ROOT/configs/peg_insertion/runtime_gpu_formal_candidate.env"
SERL_PATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}"
FRANKA_PATH="${SERL_FRANKA_SIM_PATH:-$ROOT/external/serl/franka_sim}"

cd "$ROOT"
if pgrep -af "training.peg_insertion.run_peg_training" >/dev/null; then
  echo "PEG_FORMAL_CANDIDATE_GPU_GATE=FAIL: existing Peg training process" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "FORMAL_CANDIDATE_PROFILE: smoke"
echo "FORMAL_CANDIDATE_PROCESS_POLICY: single learner / no retry"
echo "FORMAL_CANDIDATE_RNG_MODE: normal fresh-state progression"
echo "FORMAL_CANDIDATE_XLA_FLAGS: $XLA_FLAGS"

timeout --signal=TERM --kill-after=15s 2700s env \
  PYTHONPATH="$SERL_PATH:$FRANKA_PATH" \
  "$PY" -m training.peg_insertion.run_peg_training \
  --profile smoke \
  --formal-candidate \
  --role learner \
  --require-gpu

PYTHONPATH="$SERL_PATH:$FRANKA_PATH" "$PY" scripts/verify_peg_formal_candidate.py
