#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/peg_runtime_policy.sh"
exec "$PY" -u scripts/verify_peg_f0_branch.py
