#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
SERL_PATH="${SERL_LAUNCHER_PATH:?set SERL_LAUNCHER_PATH}"
FRANKA_PATH="${SERL_FRANKA_SIM_PATH:-$ROOT/external/serl/franka_sim}"
RESULT="$ROOT/results/diagnostics/peg_jax_xla_differential_20260903"
BATCH="$RESULT/fixed_batch.npz"

cd "$ROOT"
if pgrep -af "[t]raining.peg_insertion.run_peg_training" >/dev/null; then
  echo "PEG_JAX_XLA_DIFFERENTIAL=FAIL: existing Peg training process" >&2
  exit 2
fi
if [[ -e "$RESULT" ]]; then
  echo "PEG_JAX_XLA_DIFFERENTIAL=FAIL existing result directory" >&2
  exit 2
fi
mkdir -p "$RESULT"

PYTHONPATH="$SERL_PATH:$FRANKA_PATH" "$PY" scripts/peg_jax_xla_differential_probe.py \
  --mode prepare --batch "$BATCH" --result-dir "$RESULT"

run_case() {
  local mode="$1"
  local run_dir="$RESULT/$mode"
  local dump_dir="$RESULT/${mode}_xla_dump"
  set +e
  env \
    JAX_LOG_COMPILES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_dump_to=$dump_dir --xla_dump_hlo_as_text --xla_dump_hlo_as_proto" \
    MUJOCO_GL=egl \
    PYTHONPATH="$SERL_PATH:$FRANKA_PATH" \
    "$PY" scripts/peg_jax_xla_differential_probe.py \
    --mode "$mode" --batch "$BATCH" --result-dir "$RESULT" --timeout-seconds 130 \
    > "$run_dir.probe.log" 2>&1 &
  local probe_pid=$!
  nvidia-smi dmon -s pucvmet -d 1 > "$run_dir.dmon.log" 2>&1 &
  local dmon_pid=$!
  (
    while kill -0 "$probe_pid" 2>/dev/null; do
      date --iso-8601=ns
      ps -L -p "$probe_pid" -o pid,tid,psr,pcpu,stat,wchan:32,comm
      sleep 1
    done
  ) > "$run_dir.cpu_threads.log" 2>&1 &
  local cpu_pid=$!
  wait "$probe_pid"
  local status=$?
  kill "$dmon_pid" "$cpu_pid" 2>/dev/null || true
  wait "$dmon_pid" "$cpu_pid" 2>/dev/null || true
  set -e
  mv "$run_dir.probe.log" "$run_dir/probe.log"
  mv "$run_dir.dmon.log" "$run_dir/nvidia_smi_dmon.log"
  mv "$run_dir.cpu_threads.log" "$run_dir/cpu_threads.log"
  mv "$dump_dir" "$run_dir/xla_dump"
  printf '%s\n' "$status" > "$run_dir/exit_status.txt"
  echo "DIFFERENTIAL_CASE_${mode}_EXIT=$status"
  if [[ "$mode" == A && "$status" -ne 0 ]]; then
    echo "PEG_JAX_XLA_DIFFERENTIAL=FAIL successful baseline A did not complete" >&2
    exit "$status"
  fi
}

run_case A
run_case B

PYTHONPATH="$SERL_PATH:$FRANKA_PATH" "$PY" scripts/analyze_peg_jax_xla_differential.py --result-dir "$RESULT"
