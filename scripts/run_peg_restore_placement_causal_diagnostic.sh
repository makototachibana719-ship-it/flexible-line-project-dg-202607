#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
SERL_PATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}"
FRANKA_PATH="${SERL_FRANKA_SIM_PATH:-$ROOT/external/serl/franka_sim}"
RESULT="$ROOT/results/diagnostics/peg_restore_placement_causal_20260903"
BATCH="$ROOT/results/diagnostics/peg_jax_xla_differential_20260903/fixed_batch.npz"
EXPECTED_BATCH_SHA256="854aa9aa22aa2f6cbd4c7b81905c7bf7a2a32e7303676e3e0420d1686630f38c"

cd "$ROOT"
if pgrep -af "[t]raining.peg_insertion.run_peg_training" >/dev/null; then
  echo "PEG_RESTORE_PLACEMENT_CAUSAL=FAIL: existing Peg training process" >&2
  exit 2
fi
if [[ -e "$RESULT" ]]; then
  echo "PEG_RESTORE_PLACEMENT_CAUSAL=FAIL: existing result directory" >&2
  exit 2
fi
if [[ ! -f "$BATCH" ]] || [[ "$(sha256sum "$BATCH" | awk '{print $1}')" != "$EXPECTED_BATCH_SHA256" ]]; then
  echo "PEG_RESTORE_PLACEMENT_CAUSAL=FAIL: frozen batch missing or changed" >&2
  exit 2
fi
mkdir -p "$RESULT"

run_case() {
  local mode="$1"
  local case_dir="$RESULT/$mode"
  set +e
  env \
    JAX_LOG_COMPILES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    MUJOCO_GL=egl \
    PYTHONPATH="$SERL_PATH:$FRANKA_PATH" \
    "$PY" -u scripts/peg_restore_placement_causal_probe.py \
    --mode "$mode" --batch "$BATCH" --result-dir "$case_dir" \
    --timeout-seconds 120 --max-updates 5 \
    > "$RESULT/$mode.probe.log" 2>&1 &
  local probe_pid=$!
  nvidia-smi dmon -s pucvmet -d 1 > "$RESULT/$mode.dmon.log" 2>&1 &
  local dmon_pid=$!
  (
    while kill -0 "$probe_pid" 2>/dev/null; do
      date --iso-8601=ns
      ps -L -p "$probe_pid" -o pid,tid,psr,pcpu,stat,wchan:32,comm
      sleep 1
    done
  ) > "$RESULT/$mode.cpu_threads.log" 2>&1 &
  local cpu_pid=$!
  wait "$probe_pid"
  local status=$?
  kill "$dmon_pid" "$cpu_pid" 2>/dev/null || true
  wait "$dmon_pid" "$cpu_pid" 2>/dev/null || true
  set -e
  if [[ ! -d "$case_dir" ]]; then
    mkdir -p "$case_dir"
  fi
  mv "$RESULT/$mode.probe.log" "$case_dir/probe.log"
  mv "$RESULT/$mode.dmon.log" "$case_dir/nvidia_smi_dmon.log"
  mv "$RESULT/$mode.cpu_threads.log" "$case_dir/cpu_threads.log"
  printf '%s\n' "$status" > "$case_dir/exit_status.txt"
  echo "RESTORE_PLACEMENT_CASE_${mode}_EXIT=$status"
}

# Each case is executed exactly once. D2 is a separate causal arm, not a retry of D1.
run_case D1
run_case D2

PYTHONPATH="$SERL_PATH:$FRANKA_PATH" "$PY" scripts/analyze_peg_restore_placement_causal.py \
  --result-dir "$RESULT"

pgrep -af "[t]raining.peg_insertion.run_peg_training" > "$RESULT/residual_processes.txt" || true
if [[ -s "$RESULT/residual_processes.txt" ]]; then
  echo "PEG_RESTORE_PLACEMENT_CAUSAL=FAIL: residual training process" >&2
  exit 3
fi
