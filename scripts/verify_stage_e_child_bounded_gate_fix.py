#!/usr/bin/env python3
"""Static regression matrix for Stage-E bounded child config validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

import scripts.run_peg_f3_v2_rlpd_smoke as smoke
import scripts.run_peg_f3_v2_combined_q_anchor_ablation_3k as stage_e

OUT = ROOT / "results/validation/peg/f3_v2_stage_e_child_gate_fix_20260906_001"


def expect_pass(fn):
    fn()
    return "PASS"


def expect_fail(fn):
    try:
        fn()
    except Exception:
        return "PASS"
    raise RuntimeError("expected failure did not occur")


def with_config(module, cfg, fn, **kwargs):
    old = module.config
    module.config = lambda: cfg
    try:
        return fn(check_gpu=False, **kwargs)
    finally:
        module.config = old


def main():
    smoke_cfg = yaml.safe_load((ROOT / "configs/peg_insertion/peg_v2_rlpd_smoke_1000_seed0.yaml").read_text())
    stage_cfg = stage_e.VirtualConfig(yaml.safe_load(stage_e.CONFIG_PATH.read_text()))
    results = {}

    results["SMOKE_1000_500_PASS"] = expect_pass(
        lambda: with_config(smoke, copy.deepcopy(smoke_cfg), smoke.validate_config_and_dataset))
    results["STAGE_E_3000_1000_PASS"] = expect_pass(
        lambda: with_config(smoke, copy.deepcopy(stage_cfg), smoke.validate_config_and_dataset,
                            expected_learner_updates=3000, expected_checkpoint_period=1000))
    wrong_updates = stage_e.VirtualConfig(copy.deepcopy(dict(stage_cfg)))
    wrong_updates["smoke"] = dict(stage_cfg["smoke"])
    wrong_updates["smoke"]["learner_updates"] = 2999
    results["STAGE_E_WRONG_LEARNER_UPDATES_FAIL"] = expect_fail(
        lambda: with_config(smoke, wrong_updates, smoke.validate_config_and_dataset,
                            expected_learner_updates=3000, expected_checkpoint_period=1000))
    wrong_period = stage_e.VirtualConfig(copy.deepcopy(dict(stage_cfg)))
    wrong_period["smoke"] = dict(stage_cfg["smoke"])
    wrong_period["smoke"]["checkpoint_period"] = 500
    results["STAGE_E_WRONG_CHECKPOINT_PERIOD_FAIL"] = expect_fail(
        lambda: with_config(smoke, wrong_period, smoke.validate_config_and_dataset,
                            expected_learner_updates=3000, expected_checkpoint_period=1000))
    bad_hash = stage_e.VirtualConfig(copy.deepcopy(dict(stage_cfg)))
    bad_hash["demo"]["sha256"] = "0" * 64
    results["DATASET_HASH_MISMATCH_FAIL"] = expect_fail(
        lambda: with_config(smoke, bad_hash, smoke.validate_config_and_dataset,
                            expected_learner_updates=3000, expected_checkpoint_period=1000))
    drift = stage_e.VirtualConfig(copy.deepcopy(dict(stage_cfg)))
    drift["training"]["utd_ratio"] = 2
    results["CONFIG_SEMANTIC_DRIFT_FAIL"] = expect_fail(
        lambda: with_config(smoke, drift, smoke.validate_config_and_dataset,
                            expected_learner_updates=3000, expected_checkpoint_period=1000))

    original_smoke_validate = smoke.validate_config_and_dataset
    base, cfg, _ = stage_e.install_bindings("learner")
    child_cfg, _, _ = base.validate_config_and_dataset(check_gpu=False)
    smoke.validate_config_and_dataset = original_smoke_validate
    results["CHILD_LEARNER_RECEIVES_BOUNDED_VALUES_PASS"] = (
        "PASS" if child_cfg["smoke"]["learner_updates"] == 3000 and
        child_cfg["smoke"]["checkpoint_period"] == 1000 else "FAIL")
    results["ORIGINAL_SMOKE_RUNNER_SEMANTICS_UNCHANGED_PASS"] = expect_pass(
        lambda: with_config(smoke, copy.deepcopy(smoke_cfg), smoke.validate_config_and_dataset))

    gate = "PASS" if all(value == "PASS" for value in results.values()) else "FAIL"
    result = {
        "STAGE_E_CHILD_GATE_FIX": gate,
        "TRAINING_STARTED": False,
        "MATRIX": results,
        "STAGE_E_RUN_DIR": str(stage_e.OUTPUT),
        "STAGE_E_CONFIG_SHA256": stage_e.sha(stage_e.CONFIG_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
