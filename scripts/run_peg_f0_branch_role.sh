#!/usr/bin/env bash
# One bounded branch role through the production project resume entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/peg_runtime_policy.sh"
export JAX_LOG_COMPILES=1
case "${1:-}" in learner|actor) ROLE="$1";; *) exit 2;; esac
ARGS=(--profile acceptance --role "$ROLE" --require-gpu --observe-update1 --per-leaf-sync
  --resume-branch-acceptance
  --resume-from "$ROOT/results/validation/peg/f0_sync_20260904/checkpoints/checkpoint_25"
  --wandb-run-id hr88a6u1)
if [[ "$ROLE" == actor ]]; then ARGS+=(--resume-actor-step 1261); fi
exec "$PY" -u -m training.peg_insertion.run_peg_training "${ARGS[@]}"
