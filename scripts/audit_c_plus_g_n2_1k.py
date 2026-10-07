#!/usr/bin/env python3
"""Bind the existing N=2 audit protocol to the C+G candidate replicas."""
from __future__ import annotations

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_multi_replica_baseline_protocol as proto
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent

proto.CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
proto.BASE = ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907"
proto.RUNS = {
    "R1": proto.BASE / "replica1",
    "R2": proto.BASE / "replica2_retry001",
}
proto.make_early_pretanh_a0_anchor_agent = make_combined_q_anchor_agent

_json_dumps = proto.json.dumps


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _dumps_no_nonfinite(value, *args, **kwargs):
    return _json_dumps(_finite_json(value), *args, **kwargs)


proto.json.dumps = _dumps_no_nonfinite


def unique_out():
    stem = "f3_v2_c_plus_g_n2_audit_20260907"
    for index in range(1, 100):
        out = proto.OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


proto.unique_out = unique_out


if __name__ == "__main__":
    proto.main()
