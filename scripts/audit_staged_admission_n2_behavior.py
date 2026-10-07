#!/usr/bin/env python3
"""Read-only matched behavior audit for staged-admission N=2."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [str(ROOT), str(ROOT / "external/serl/serl_launcher"),
                str(ROOT / "external/serl/franka_sim"),
                str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
                str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env")]

from peg_env_adapter import make_peg_state_env
from scripts.recover_r1_fixed_prefill_checkpoint import eval_rollout
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
RUNS = {
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica2",
}
BASELINE = {
    "R1": ROOT / "results/diagnostics/peg/f3_v2_r1_fixed_prefill_checkpoint_recovery_20260907_001/result.json",
    "R2": ROOT / "results/diagnostics/peg/f3_v2_r2_fixed_prefill_checkpoint_recovery_20260907_001/result.json",
}
OUT_ROOT = ROOT / "results/diagnostics/peg"
METRICS = ("SUCCESS", "NEAR_HOLE", "FIRST_DIVERGENCE", "CORRIDOR", "ORIENTATION", "AXIAL", "FORCE", "CONTACT")


def unique_out():
    for index in range(1, 100):
        out = OUT_ROOT / f"f3_v2_staged_admission_n2_behavior_20260907_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def load_agent(cfg, run):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return fresh.replace(state=state)


def mean(rows, key):
    return sum(float(row[key]) for row in rows.values()) / len(rows)


def trend(current, baseline, higher_is_better=False):
    tolerance = max(abs(baseline) * 0.01, 1e-9)
    delta = current - baseline
    if abs(delta) <= tolerance:
        return "FLAT"
    improving = delta > 0 if higher_is_better else delta < 0
    return "IMPROVING" if improving else "WORSENING"


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    staged = {name: eval_rollout(cfg, load_agent(cfg, run)) for name, run in RUNS.items()}
    baseline = {name: json.loads(path.read_text()) for name, path in BASELINE.items()}
    staged_mean = {key: mean(staged, key) for key in METRICS}
    baseline_mean = {key: mean(baseline, key) for key in METRICS}
    trends = {
        key: trend(staged_mean[key], baseline_mean[key], key in {"SUCCESS", "NEAR_HOLE", "FIRST_DIVERGENCE"})
        for key in METRICS
    }
    noncatastrophic = (
        staged_mean["CORRIDOR"] <= 2.0 * baseline_mean["CORRIDOR"]
        and staged_mean["FORCE"] <= 2.0 * baseline_mean["FORCE"]
    )
    improvements = sum(trends[key] == "IMPROVING" for key in METRICS)
    clearly_better = (
        staged_mean["SUCCESS"] > baseline_mean["SUCCESS"]
        or staged_mean["NEAR_HOLE"] > baseline_mean["NEAR_HOLE"]
        or staged_mean["FIRST_DIVERGENCE"] > baseline_mean["FIRST_DIVERGENCE"]
        or improvements >= 3
    )
    result = {
        "STAGED_BEHAVIOR_AUDIT": "PASS" if noncatastrophic else "FAIL",
        "BEHAVIOR_CLEARLY_BETTER": "YES" if clearly_better else "NO",
        "BEHAVIOR_NONCATASTROPHIC": "YES" if noncatastrophic else "NO",
        "STAGED": {name: {key: row[key] for key in METRICS} | {
            "QPOLICY_MINUS_QDEMO": row["QPOLICY_MINUS_QDEMO"], "A0_1000": row["A0_1000"]}
            for name, row in staged.items()},
        "BASELINE_MEAN": baseline_mean,
        "STAGED_MEAN": staged_mean,
        "TRENDS": trends,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "TRAINING_STARTED": False,
    }
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "STAGED"}, sort_keys=True))
    print(json.dumps({"R1": result["STAGED"]["R1"], "R2": result["STAGED"]["R2"]}, sort_keys=True))
    print(str(out / "result.json"))


if __name__ == "__main__":
    main()
