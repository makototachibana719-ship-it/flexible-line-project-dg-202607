#!/usr/bin/env python3
"""Short-horizon comparison: joint-support candidate vs prior a0-support candidate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.short_horizon_promotion_gate_audit as sh
from training.peg_insertion.a0_action_support_balance_ablation import make_a0_action_support_balance_agent
from training.peg_insertion.joint_action_support_balance_ablation import make_joint_action_support_balance_agent

joint_cfg = ROOT / "configs/peg_insertion/peg_v2_joint_action_support_balance_ablation_1k_seed0.yaml"
sh.RUNS = {
    "baseline_R1": (sh.CANDIDATE_CONFIG, ROOT / "results/ablation/peg_f3_v2_a0_action_support_balance_n2_20260906/replica1", make_a0_action_support_balance_agent),
    "baseline_R2": (sh.CANDIDATE_CONFIG, ROOT / "results/ablation/peg_f3_v2_a0_action_support_balance_n2_20260906/replica2_retry001", make_a0_action_support_balance_agent),
    "candidate_R1": (joint_cfg, ROOT / "results/ablation/peg_f3_v2_joint_action_support_balance_n2_20260907/replica1", make_joint_action_support_balance_agent),
    "candidate_R2": (joint_cfg, ROOT / "results/ablation/peg_f3_v2_joint_action_support_balance_n2_20260907/replica2_retry001", make_joint_action_support_balance_agent),
}


def unique_out():
    stem = "f3_v2_joint_support_vs_a0_short_horizon_20260907"
    for index in range(1, 100):
        out = sh.OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


sh.unique_out = unique_out

if __name__ == "__main__":
    sh.main()
