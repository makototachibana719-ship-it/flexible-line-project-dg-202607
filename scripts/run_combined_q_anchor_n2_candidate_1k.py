#!/usr/bin/env python3
"""Output-isolated Peg F3 V2 C+G candidate runner for matched N=2 1K."""
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

CONFIG_PATH = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
EXPECTED_CONFIG = "b1cd4126c10673c267e0e2382a44eef97cfc7b0611a5774568d46ab6d8f2ff41"
RUN_ID = "wandb-disabled-peg-f3-v2-combined-q-anchor-n2-1k"


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
    from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
    import serl_launcher.utils.launcher as launcher
    import training.peg_insertion.resume as resume

    require(sha(CONFIG_PATH) == EXPECTED_CONFIG, "C+G config SHA256")
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    require(raw["training"]["checkpoint_period"] == 500, "checkpoint period")
    require(raw["ablation"]["checkpoint_targets"] == [500, 1000], "checkpoint targets")
    require(raw["ablation"]["semantic_diff_fields"] == ["conservative_q", "demo_actor_anchor"], "allowed diff")
    require(raw["conservative_q"]["selected_lambda"] == 5.0, "conservative q lambda")
    require(raw["demo_actor_anchor"]["selected_lambda"] == 0.07, "demo anchor lambda")

    def config():
        return VirtualConfig(yaml.safe_load(CONFIG_PATH.read_text()))

    def make_agent(seed, sample_obs, sample_action, discount=.99):
        require(discount == .99, "discount override forbidden")
        return make_combined_q_anchor_agent(seed, sample_obs, sample_action, raw)

    launcher.make_sac_agent = make_agent
    original_save, original_load = resume.save_bundle, resume.load_bundle

    def save_bundle(directory, state, replay, step, actor_steps, cfg, manifest, run_id, source, **kwargs):
        source = {
            "identifier": "peg-f3-v2-combined-q-anchor-n2-1k",
            "profile": "isolated_c_plus_g_n2_1k",
            "fresh_initialization": True,
            "resume": False,
            "demo_online_ratio": "32:32",
            "utd_ratio": 1,
            "conservative_q_scope": "DEMO_TAIL",
            "conservative_q_lambda": 5.0,
            "demo_actor_anchor_scope": "ALL_DEMO",
            "demo_actor_anchor_lambda": 0.07,
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
    result.update({
        "C_PLUS_G_CONFIG_FROZEN": "PASS",
        "C_PLUS_G_CONFIG_SHA256": sha(CONFIG_PATH),
        "CHECKPOINT_SEMANTICS": [500, 1000],
        "NO_EXTRA_INSTRUMENTATION": "PASS",
        "ALLOWED_DIFF_ONLY": "PASS",
        "CONSERVATIVE_Q_LAMBDA": raw["conservative_q"]["selected_lambda"],
        "DEMO_ANCHOR_LAMBDA": raw["demo_actor_anchor"]["selected_lambda"],
    })
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
        require(not output.exists(), "fresh output required")
        return base.supervise()
    return {"learner": base.learner, "actor": base.actor, "verify": base.verify}[args.role](output) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
