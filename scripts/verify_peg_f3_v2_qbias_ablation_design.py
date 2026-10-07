#!/usr/bin/env python3
"""Static/read-only readiness verifier for the config-only Q-bias ablation."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import yaml

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_agent
from training.peg_insertion.observation_contract import V2, validate_config
from training.peg_insertion.qbias_ablation import make_qbias_ablation_agent

BASE = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
ABLATION = ROOT / "configs/peg_insertion/peg_v2_qbias_ablation_1k_seed0.yaml"
EXPECTED_BASE = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def tree_equal(a, b):
    aa = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(a)]
    bb = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(b)]
    return len(aa) == len(bb) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(aa, bb))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(); out = args.output.resolve()
    require(ROOT in out.parents and not out.exists(), "fresh output required")
    out.mkdir(parents=True, exist_ok=False)
    result = {"PEG_F3_V2_QBIAS_MINIMAL_FIX_DESIGN": "FAIL", "Q_BIAS_ABLATION": "BLOCKED",
              "TRAINING_STARTED": False, "NEXT": "BLOCKED"}
    try:
        require(sha(BASE) == EXPECTED_BASE, "baseline config hash")
        base, cfg = yaml.safe_load(BASE.read_text()), yaml.safe_load(ABLATION.read_text())
        require(validate_config(base, for_training=True) == V2 and validate_config(cfg, for_training=True) == V2, "V2 contract")
        require(cfg["ablation"]["baseline_config_sha256"] == EXPECTED_BASE, "baseline binding")
        baseline = {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4,
                    "temperature_learning_rate": 3e-4, "actor_warmup_steps": 2000,
                    "critic_warmup_steps": 2000, "discount": .99,
                    "soft_target_update_rate": .005, "target_entropy": -3.5,
                    "backup_entropy": False, "critic_ensemble_size": 10,
                    "critic_subsample_size": 2}
        require("discount=0.99" in inspect.getsource(make_sac_agent), "production discount source")
        create_sig = inspect.signature(SACAgent.create)
        require(create_sig.parameters["soft_target_update_rate"].default == .005, "target rate source")
        require(create_sig.parameters["backup_entropy"].default is False, "entropy source")
        require(cfg["optimizer"]["actor_learning_rate"] == baseline["actor_learning_rate"], "actor LR changed")
        require(cfg["optimizer"]["temperature_learning_rate"] == baseline["temperature_learning_rate"], "temperature LR changed")
        require(cfg["optimizer"]["actor_warmup_steps"] == baseline["actor_warmup_steps"], "actor warmup changed")
        require(cfg["optimizer"]["critic_learning_rate"] == 1e-4 and cfg["optimizer"]["critic_warmup_steps"] == 3000, "candidate")
        require(cfg["ablation"]["config_diff_fields"] == ["critic_learning_rate", "critic_warmup_steps"], "diff fields")
        for key in ("environment", "algorithm", "seed", "runtime", "demo", "observation", "action", "contracts"):
            require(cfg[key] == base[key], f"protected config section changed: {key}")
        for key in ("online_batch_size", "demo_batch_size", "total_batch_size", "demo_fraction",
                    "critic_actor_ratio", "utd_ratio", "training_starts", "random_steps", "replay_capacity"):
            require(cfg["training"][key] == base["training"][key], f"training semantic changed: {key}")
        require(cfg["fresh_initialization"] is True and cfg["load_legacy_checkpoint"] is False, "fresh init")
        require(cfg["ablation"]["target_updates"] == 1000 and cfg["ablation"]["checkpoint_targets"] == [500, 1000] and
                cfg["training"]["checkpoint_period"] == 500, "bounded envelope")
        require(not (ROOT / cfg["outputs"]["run_dir"]).exists(), "ablation output is not fresh")
        sample_obs, sample_action = np.zeros(20, np.float32), np.zeros(7, np.float32)
        prod = make_sac_agent(0, sample_obs, sample_action)
        candidate = make_qbias_ablation_agent(0, sample_obs, sample_action, cfg)
        require(tree_equal(prod.state.params, candidate.state.params), "architecture/initial params changed")
        require(prod.config == candidate.config, "SAC algorithm config changed")
        require(cfg["ablation"]["q_bias_improved_below"] == 1.578540 and
                cfg["ablation"]["q_bias_strong_pass_at_or_below"] == .5, "Q gate")
        result.update(PEG_F3_V2_QBIAS_MINIMAL_FIX_DESIGN="PASS",
                      BASELINE_CRITIC_LR=baseline["critic_learning_rate"], NEW_CRITIC_LR=cfg["optimizer"]["critic_learning_rate"],
                      BASELINE_CRITIC_WARMUP=baseline["critic_warmup_steps"], NEW_CRITIC_WARMUP=cfg["optimizer"]["critic_warmup_steps"],
                      CONFIG_DIFF_FIELDS=cfg["ablation"]["config_diff_fields"], ABLATION_CONFIG_PATH=str(ABLATION.relative_to(ROOT)),
                      ABLATION_CONFIG_SHA256=sha(ABLATION), TARGET_UPDATES=1000, CHECKPOINTS=[500, 1000],
                      Q_BIAS_MATCHED_BASELINE=cfg["ablation"]["q_bias_matched_baseline"],
                      POLICY_ALIGNMENT_BASELINE=cfg["ablation"]["policy_alignment_baseline_mae"],
                      LATERAL_ALIGNMENT_BASELINE={"mae": cfg["ablation"]["lateral_correction_baseline_mae"],
                                                  "action_1_mae": cfg["ablation"]["lateral_correction_action1_baseline_mae"]},
                      BASELINE_DEFAULTS=baseline, EXECUTION_BOUNDARY_FIELDS=["target_updates", "checkpoint_period", "checkpoint_targets", "fresh_output_paths"],
                      PARAMETER_INITIALIZATION_IDENTICAL="PASS", SAC_OBJECTIVE_IDENTICAL="PASS",
                      Q_BIAS_ABLATION="READY", TRAINING_STARTED=False,
                      NEXT="USER_AUTHORIZATION_REQUIRED_FOR_1K_ABLATION")
    except Exception as exc:
        result["error"] = repr(exc); result["traceback"] = traceback.format_exc(); traceback.print_exc()
    (out / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "REPORT.md").write_text("# Peg F3 V2 Minimal Q-Bias Fix Design\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["PEG_F3_V2_QBIAS_MINIMAL_FIX_DESIGN"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
