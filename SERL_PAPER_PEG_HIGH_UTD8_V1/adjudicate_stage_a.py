#!/usr/bin/env python3
"""Verify preserved Stage-A bundles and adjudicate the post-result ZMQ abort."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import jax
import numpy as np
import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
RUN = BRANCH / "runs/stage_a_seed0_attempt2"
STEPS = (250, 500, 750, 1000)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    cfg = yaml.safe_load((BRANCH / "config.yaml").read_text())
    sys_paths = [
        ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim",
        ROOT / "external/packages/peg_insertion/peg_insertion/code/env",
    ]
    import sys
    sys.path[:0] = [str(path) for path in sys_paths] + [str(ROOT)]
    from peg_env_adapter import make_peg_state_env
    from serl_launcher.utils.launcher import make_sac_agent
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import load_bundle

    learner = json.loads((RUN / "learner_result.json").read_text())
    actor = json.loads((RUN / "actor_result.json").read_text())
    cleanup = json.loads((RUN / "actor_cleanup_complete.json").read_text())
    before = json.loads((RUN / "protected_before.json").read_text())
    after = json.loads((RUN / "protected_after.json").read_text())
    run_id = "wandb-disabled-serl-paper-peg-high-utd8-v1"
    checkpoints = {}
    for step in STEPS:
        meta = load_bundle(RUN / "checkpoints" / f"checkpoint_{step}", cfg,
                           ROOT / cfg["demo"]["manifest"], run_id)
        evaluation = json.loads((RUN / f"eval_{step}_primary.json").read_text())
        checks = {
            "step": meta["checkpoint_step"] == meta["learner_update_steps"] == step,
            "utd8": meta["source_snapshot"]["utd_ratio"] == 8,
            "critic_updates": meta["source_snapshot"]["critic_updates"] == step * 8,
            "demo_sha": meta["demo_sha256"] == cfg["demo"]["sha256"],
            "replay": meta["online_replay_size"] > 0,
            "eval_3": evaluation["aggregate"]["episodes"] == 3,
        }
        require(all(checks.values()), f"checkpoint {step} metadata/eval")
        checkpoints[str(step)] = checks

    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_sac_agent(seed=cfg["seed"], sample_obs=np.zeros(20, np.float32),
                               sample_action=np.zeros(7, np.float32))
        restore = {}
        for step in STEPS:
            state = restore_fresh_state(RUN / "checkpoints", fresh.state, step)
            arrays = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(state)
                      if hasattr(x, "dtype")]
            restore[str(step)] = bool(arrays and all(np.isfinite(x).all() for x in arrays))
        env.close()
    residual = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                              text=True, check=True).stdout
    residual = [line.strip() for line in residual.splitlines()
                if "SERL_PAPER_PEG_HIGH_UTD8_V1" in line and
                ("run_high_utd8.py" in line or "run_stage_a.py" in line)]
    checks = {
        "LEARNER_RESULT_COMPLETE": learner["status"] == "PASS" and
                                   learner["completed_updates"] == 1000 and
                                   learner["critic_updates"] == 8000,
        "ACTOR_RESULT_COMPLETE": actor["status"] == "PASS" and
                                 actor["received_networks"] == 1001,
        "ACTOR_EXPLICIT_CLEANUP": cleanup["status"] == "PASS" and
                                  cleanup["shutdown"]["status"] == "PASS" and
                                  not cleanup["shutdown"]["after"]["broadcast"]["alive"],
        "ALL_CHECKPOINTS_RESTORE_FINITE": all(restore.values()),
        "PROTECTED_UNCHANGED": before == after,
        "RESIDUAL_PROCESSES_NONE": not residual,
    }
    require(all(checks.values()), "Stage-A validity checks")
    result = {
        "STAGE_A_EVIDENCE_VALID": "YES",
        "POST_RESULT_EXIT_ADJUDICATION": "NON_SCIENTIFIC_ZMQ_INTERPRETER_TEARDOWN_ASSERTION",
        "TRAINING_AND_EVALUATION_ACCEPTED": "YES",
        "checks": checks,
        "checkpoint_checks": checkpoints,
        "restore_finite": restore,
        "raw_stage_a_result_preserved": str(RUN / "STAGE_A_RESULT.json"),
        "UTD8_CAUSAL_COMPARISON": "FAIL",
        "PAPER_FIDELITY_REBUILD_REQUIRED": "YES",
    }
    (RUN / "STAGE_A_ADJUDICATION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
