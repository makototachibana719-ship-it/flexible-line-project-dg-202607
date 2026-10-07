#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/makoto/flexible_line_project
PYTHON=/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/external/serl/serl_launcher:$PROJECT_ROOT/external/serl/franka_sim"
cd "$PROJECT_ROOT"

mode="${1:-}"
case "$mode" in
  imports)
    exec "$PYTHON" -c 'import scripts, training, training.peg_insertion, training.peg_insertion.actor_shutdown; print("PEG_F3_CANONICAL_IMPORTS=PASS")'
    ;;
  help)
    exec "$PYTHON" "$PROJECT_ROOT/scripts/run_peg_f3_v2_rlpd_smoke.py" --help
    ;;
  short-preflight)
    export WANDB_MODE=disabled
    output="${2:?fresh output path required}"
    exec "$PYTHON" "$PROJECT_ROOT/scripts/run_peg_f3_v2_rlpd_smoke.py" \
      --role preflight --short-shutdown-acceptance --output "$output"
    ;;
  short-run)
    export WANDB_MODE=disabled
    output="${2:?fresh output path required}"
    exec "$PYTHON" -u "$PROJECT_ROOT/scripts/run_peg_f3_v2_rlpd_smoke.py" \
      --role supervise --short-shutdown-acceptance --output "$output"
    ;;
  formal-preflight)
    evidence="${2:?fresh evidence path required}"
    exec "$PYTHON" "$PROJECT_ROOT/scripts/validate_peg_v2_formal_5k_preflight.py" \
      --evidence "$evidence"
    ;;
  formal-run)
    config="${2:?frozen config path required}"
    output="${3:?fresh output path required}"
    authorization="${4:-}"
    [[ "$authorization" == "--authorize-formal-5k" ]] || {
      echo "formal 5K requires --authorize-formal-5k" >&2
      exit 2
    }
    export WANDB_MODE=disabled
    exec "$PYTHON" -u "$PROJECT_ROOT/scripts/run_peg_f3_v2_rlpd_smoke.py" \
      --role supervise --formal-5k-config "$config" --output "$output" \
      --authorize-formal-5k
    ;;
  *)
    echo "usage: $0 {imports|help|short-preflight|short-run|formal-preflight|formal-run}" >&2
    exit 2
    ;;
esac
