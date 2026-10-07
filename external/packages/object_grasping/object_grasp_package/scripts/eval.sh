#!/usr/bin/env bash
set -euo pipefail
LOG="${1:-logs/actor_5000_seed0_run2.log}"
[[ -f "$LOG" ]] || { echo "log not found: $LOG" >&2; exit 1; }
grep -E '\[EVAL\]|\[SUCCESS\]' "$LOG" || true
echo "注意：这里只提取已有日志，不是独立 checkpoint restore/evaluation。"
