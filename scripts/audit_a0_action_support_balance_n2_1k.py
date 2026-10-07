#!/usr/bin/env python3
"""Offline N=2 audit for the a0-action-support balance candidate replicas."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_multi_replica_baseline_protocol as proto
from training.peg_insertion.a0_action_support_balance_ablation import make_a0_action_support_balance_agent

proto.CONFIG = ROOT / "configs/peg_insertion/peg_v2_a0_action_support_balance_ablation_1k_seed0.yaml"
proto.BASE = ROOT / "results/ablation/peg_f3_v2_a0_action_support_balance_n2_20260906"
proto.RUNS = {"R1": proto.BASE / "replica1", "R2": proto.BASE / "replica2_retry001"}
proto.make_early_pretanh_a0_anchor_agent = make_a0_action_support_balance_agent


def unique_out():
    stem = "f3_v2_a0_action_support_balance_n2_audit_20260906"
    for index in range(1, 100):
        out = proto.OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


proto.unique_out = unique_out

if __name__ == "__main__":
    proto.main()
