#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
SERL_PATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}"
FRANKA_PATH="${SERL_FRANKA_SIM_PATH:-$ROOT/external/serl/franka_sim}"
CHECKPOINT="$ROOT/results/checkpoints/peg_rlpd_gpu_formal_candidate_seed0/checkpoint_0"
RESULT_DIR="$ROOT/results/diagnostics/peg_candidate_checkpoint0_update2_fresh_process_20260903"
RUN_ID="tzv5bsvz"

cd "$ROOT"
if pgrep -af "[t]raining.peg_insertion.run_peg_training" >/dev/null; then
  echo "PEG_CANDIDATE_UPDATE2_DIAGNOSTIC=FAIL: existing Peg training process" >&2
  exit 2
fi
for required in "$CHECKPOINT" "${CHECKPOINT}_online_replay.pkl" "${CHECKPOINT}_resume_metadata.json"; do
  [[ -s "$required" ]] || { echo "missing candidate checkpoint component: $required" >&2; exit 2; }
done
if [[ -e "$RESULT_DIR" ]]; then
  echo "PEG_CANDIDATE_UPDATE2_DIAGNOSTIC=FAIL: result directory already exists" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR"

printf 'unix_seconds,index,name,memory_used_mib,memory_total_mib,gpu_util_percent,power_watts\n' > "$RESULT_DIR/gpu.csv"
monitor_gpu() {
  while true; do
    timestamp="$(date +%s)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits |
      sed "s/^/${timestamp},/" >> "$RESULT_DIR/gpu.csv" || true
    sleep 10
  done
}
monitor_gpu &
monitor_pid=$!

set +e
timeout --signal=TERM --kill-after=15s 130s env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  MUJOCO_GL=egl \
  PYTHONPATH="$SERL_PATH:$FRANKA_PATH" \
  "$PY" -m training.peg_insertion.run_peg_training \
  --profile diagnostic \
  --role learner \
  --require-gpu \
  --resume-from "$CHECKPOINT" \
  --wandb-run-id "$RUN_ID" \
  --diagnose-updates \
  --diagnostic-no-save \
  --max-resume-updates 1 \
  --diagnostic-disable-wandb \
  --diagnostic-stack-dumps \
  > "$RESULT_DIR/learner.log" 2>&1
status=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
printf '%s\n' "$status" > "$RESULT_DIR/exit_status.txt"
cat "$RESULT_DIR/learner.log"

if pgrep -af "[t]raining.peg_insertion.run_peg_training" >/dev/null; then
  echo "PEG_CANDIDATE_UPDATE2_DIAGNOSTIC=FAIL: residual training process" >&2
  exit 3
fi
if [[ "$status" -eq 0 ]]; then
  echo "PEG_CANDIDATE_CHECKPOINT0_NEXT_UPDATE_FRESH_PROCESS=PASS"
  exit 0
fi
if [[ "$status" -eq 124 ]]; then
  echo "PEG_CANDIDATE_CHECKPOINT0_NEXT_UPDATE_FRESH_PROCESS=FAIL_TIMEOUT"
  exit 124
fi
echo "PEG_CANDIDATE_CHECKPOINT0_NEXT_UPDATE_FRESH_PROCESS=FAIL_EXIT_${status}"
exit "$status"
