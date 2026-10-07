#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/peg_runtime_policy.sh"
"$PY" scripts/preflight_peg_formal.py --candidate
exec "$PY" scripts/verify_peg_f0.py --run-once
