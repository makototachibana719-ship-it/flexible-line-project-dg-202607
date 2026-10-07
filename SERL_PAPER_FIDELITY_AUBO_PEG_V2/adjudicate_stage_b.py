#!/usr/bin/env python3
"""Restore-check and close the paper-fidelity Stage-B evidence."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import yaml

BRANCH = Path(__file__).resolve().parent
ROOT = BRANCH.parent
RUN = BRANCH / "runs/stage_b_seed0"
STEPS = (250, 500, 750, 1000, 1500, 2000)
EXAMPLE = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
ENV_PARENT = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
sys.path[:0] = [str(BRANCH), str(ROOT), str(EXAMPLE), str(ENV_PARENT)]


def require(value, message):
    if not value: raise RuntimeError(message)


def main():
    from paper_fidelity_env import make_paper_fidelity_env
    from serl_launcher.utils.launcher import make_sac_agent
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    from training.peg_insertion.resume import load_bundle

    cfg = yaml.safe_load((BRANCH / "config.yaml").read_text())
    learner = json.loads((RUN / "learner_result.json").read_text())
    actor = json.loads((RUN / "actor_result.json").read_text())
    cleanup = json.loads((RUN / "actor_cleanup_complete.json").read_text())
    result = json.loads((RUN / "STAGE_B_RESULT.json").read_text())
    before = json.loads((RUN / "protected_before.json").read_text())
    after = json.loads((RUN / "protected_after.json").read_text())
    run_id = "wandb-disabled-serl-paper-fidelity-aubo-peg-v2"
    metadata_checks = {}
    for step in STEPS:
        meta = load_bundle(RUN / "checkpoints" / f"checkpoint_{step}", cfg,
                           ROOT / cfg["demo"]["manifest"], run_id)
        evaluation = json.loads((RUN / f"eval_{step}_primary.json").read_text())
        checks = {
            "step": meta["checkpoint_step"] == meta["learner_update_steps"] == step,
            "utd8": meta["source_snapshot"]["utd_ratio"] == 8,
            "critic_updates": meta["source_snapshot"]["critic_updates"] == step * 8,
            "demo20": meta["source_snapshot"]["demo_episodes"] == 20 and
                      meta["demo_transition_count"] == 3627,
            "demo_sha": meta["demo_sha256"] == cfg["demo"]["sha256"],
            "randomized_train_reset": meta["source_snapshot"]["training_reset_randomization"] is True,
            "relative_action": meta["source_snapshot"]["relative_action"] ==
                               "AUBO_COMPATIBLE_RELATIVE_INCREMENTAL",
            "eval_3_canonical": evaluation["aggregate"]["episodes"] == 3 and
                                evaluation["canonical_reset"] is True,
        }
        require(all(checks.values()), f"checkpoint {step}")
        metadata_checks[str(step)] = checks
    with portable_peg_runtime(ROOT):
        env = make_paper_fidelity_env(render_mode="rgb_array", max_episode_steps=500,
                                      training_reset_randomization=False)
        fresh = make_sac_agent(seed=cfg["seed"], sample_obs=np.zeros(20, np.float32),
                               sample_action=np.zeros(7, np.float32))
        restored = {}
        for step in STEPS:
            state = restore_fresh_state(RUN / "checkpoints", fresh.state, step)
            arrays = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(state)
                      if hasattr(x, "dtype")]
            restored[str(step)] = bool(arrays and all(np.isfinite(x).all() for x in arrays))
        env.close()
    final5 = json.loads((RUN / "eval_250_final5.json").read_text())
    processes = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                               text=True, check=True).stdout
    residual = [line.strip() for line in processes.splitlines()
                if "SERL_PAPER_FIDELITY_AUBO_PEG_V2" in line and
                ("run_paper_fidelity.py" in line or "run_stage_b.py" in line)]
    checks = {
        "LEARNER_COMPLETE": learner["status"] == "PASS" and
                            learner["completed_updates"] == 2000 and
                            learner["critic_updates"] == 16000 and
                            learner["demo_size"] == 3627,
        "ACTOR_COMPLETE": actor["status"] == "PASS" and actor["received_networks"] == 2001,
        "EXPLICIT_CLEANUP": cleanup["status"] == "PASS" and
                            cleanup["shutdown"]["status"] == "PASS" and
                            not cleanup["shutdown"]["after"]["broadcast"]["alive"],
        "ALL_RESTORES_FINITE": all(restored.values()),
        "FINAL5_CANONICAL": final5["aggregate"]["episodes"] == 5 and
                            final5["canonical_reset"] is True,
        "FINAL5_ZERO_SUCCESS": final5["aggregate"]["success_count"] == 0,
        "FINAL5_CONTACT_FAIL": final5["aggregate"]["contact_health"] is False,
        "PROTECTED_UNCHANGED": before == after and result["PROTECTED_ARTIFACTS_UNCHANGED"] is True,
        "NO_RESIDUAL_PROCESSES": not residual,
    }
    require(all(checks.values()), "Stage-B closure")
    adjudication = {
        "STAGE_B_EVIDENCE_VALID": "YES", "checks": checks,
        "checkpoint_checks": metadata_checks, "restore_finite": restored,
        "PAPER_FIDELITY_CONVERGED": "NO", "FINAL_CANONICAL_SUCCESS": "0/5",
        "CONTACT_HEALTH": "FAIL", "PURE_SERL_RLPD_AUBO_PEG_CONVERGENCE": "NO",
        "NEXT": "HIL_SERL_INTERVENTION",
    }
    (RUN / "STAGE_B_ADJUDICATION.json").write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
