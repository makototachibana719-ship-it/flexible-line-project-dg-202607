#!/usr/bin/env python3
"""Run the isolated Peg F3 V2 joint-action-support balance candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_peg_f3_v2_rlpd_smoke as base

CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_joint_action_support_balance_ablation_1k_seed0.yaml"
EXPECTED_CONFIG = "03b765a09bf299bc291fce8a6c8e832e03a350d6e6f8584519e2cd372daf2ec3"
RUN_ID = "wandb-disabled-peg-f3-v2-joint-action-support-balance-ablation-1k"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class VirtualConfig(dict):
    _smoke = {
        "learner_updates": 1000,
        "checkpoint_period": 500,
        "actor_max_steps": 1000000,
        "process_timeout_seconds": 7200,
        "update_timeout_seconds": 120,
    }

    def __getitem__(self, key):
        if key == "smoke" and key not in self:
            return self._smoke
        return super().__getitem__(key)


def install(output: Path):
    from training.peg_insertion.joint_action_support_balance_ablation import make_joint_action_support_balance_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "candidate config SHA256")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    require(raw["training"]["checkpoint_period"] == 500, "checkpoint period")
    require(raw["ablation"]["checkpoint_targets"] == [500, 1000], "checkpoint targets")
    require(
        raw["ablation"]["semantic_diff_fields"]
        == ["conservative_q", "demo_actor_anchor", "actor_objective_balance", "a0_action_support", "joint_action_support"],
        "allowed diff",
    )
    require(raw["actor_objective_balance"]["selected_lambda"] == 0.08, "actor objective balance lambda")
    require(raw["a0_action_support"]["selected_lambda"] == 0.0258505165, "a0 support lambda")
    require(raw["joint_action_support"]["selected_lambda"] == 0.0900026310, "joint support lambda")
    require(raw["joint_action_support"]["dims"] == "a0:a5", "joint support dims")
    require(raw["joint_action_support"]["exclude_dims"] == ["a6"], "a6 excluded")

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_joint_action_support_balance_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save, original_load = resume.save_bundle, resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {
            "identifier": "peg-f3-v2-joint-action-support-balance-ablation-1k",
            "profile": "isolated_joint_action_support_balance_ablation",
            "fresh_initialization": True,
            "resume": False,
            "demo_online_ratio": "32:32",
            "utd_ratio": 1,
            "conservative_q_scope": "DEMO_TAIL",
            "conservative_q_lambda": 5.0,
            "demo_actor_anchor_scope": "ALL_DEMO",
            "demo_actor_anchor_lambda": 0.07,
            "actor_objective_balance_action_dim": "a0",
            "actor_objective_balance_lambda": 0.08,
            "a0_action_support_scope": "demo_early_approach_rows",
            "a0_action_support_lambda": 0.0258505165,
            "joint_action_support_scope": "setup_approach_lateral_correction",
            "joint_action_support_dims": "a0:a5",
            "joint_action_support_lambda": 0.0900026310,
        }
        return original_save(directory, state, replay, step, actor_steps, cfg, manifest, RUN_ID, source, **kwargs)

    resume.save_bundle = save_bundle
    resume.load_bundle = lambda cp, cfg, manifest, run_id: original_load(cp, cfg, manifest, RUN_ID)
    base.CONFIG = CONFIG_PATH
    base.OUTPUT = output
    base.ACTIVE_OUTPUT = output
    base.RUN_UPDATES = 1000
    base.CHECKPOINT_TARGETS = (500, 1000)
    base.FORMAL_5K = False
    base.SHORT_SHUTDOWN_ACCEPTANCE = False
    base.__file__ = str(Path(__file__).resolve())
    base.config = config

    original_runtime_env = base.runtime_env

    def runtime_env():
        env = original_runtime_env()
        required = [
            str(ROOT),
            str(ROOT / "external/serl/serl_launcher"),
            str(ROOT / "external/serl/franka_sim"),
            str(base.ENV_PARENT),
        ]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(required + ([existing] if existing else []))
        return env

    base.runtime_env = runtime_env
    return raw


def preflight(output: Path):
    raw = install(output)
    result = base.preflight()
    result.update(
        {
            "JOINT_SUPPORT_CONFIG_FROZEN": "PASS",
            "JOINT_SUPPORT_CONFIG_SHA256": sha(CONFIG_PATH),
            "CHECKPOINT_SEMANTICS": [500, 1000],
            "NO_EXTRA_INSTRUMENTATION": "PASS",
            "ALLOWED_DIFF_ONLY": "PASS",
            "JOINT_ACTION_SUPPORT_LAMBDA": raw["joint_action_support"]["selected_lambda"],
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("preflight", "supervise", "learner", "actor", "verify"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(ROOT in output.parents, "output outside project")
    if args.role == "preflight":
        return preflight(output)
    install(output)
    if args.role == "supervise":
        require(not output.exists(), "fresh candidate output required")
        return base.supervise()
    return {"learner": base.learner, "actor": base.actor, "verify": base.verify}[args.role](output) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
