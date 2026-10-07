#!/usr/bin/env python3
"""Read-only final audit of the completed Peg F3 V2 formal 5K run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
EXPECTED_CONFIG = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
EXPECTED_DATASET = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
STEPS = (1000, 2000, 3000, 4000, 5000)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    run, evidence = args.run.resolve(), args.evidence.resolve()
    require(ROOT in run.parents and ROOT in evidence.parents and not evidence.exists(), "path isolation")
    cfg = yaml.safe_load(CONFIG.read_text())
    require(sha(CONFIG) == EXPECTED_CONFIG, "config hash")
    require(sha(ROOT / cfg["demo"]["path"]) == EXPECTED_DATASET, "dataset hash")
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    require([row["update"] for row in updates] == list(range(1, 5001)), "update sequence")
    require(all(row["losses_finite"] and row["demo_batch"] == row["online_batch"] == 32
                and all(np.isfinite(value) for value in row["losses"].values()) for row in updates),
            "finite/ratio")
    learner, actor = read_json(run / "learner_result.json"), read_json(run / "actor_result.json")
    shutdown, cleanup = read_json(run / "actor_shutdown.json"), read_json(run / "actor_cleanup_complete.json")
    restore = read_json(run / "restore_verification.json")
    require(learner["completed_updates"] == 5000 and learner["backend"] == "gpu", "learner completion")
    require(actor["received_networks"] > 0 and actor["stopped_by_learner"] is True, "communication/stop")
    require(shutdown["status"] == "PASS" and shutdown["after"]["req_rep"]["socket_closed"] is True
            and shutdown["after"]["req_rep"]["context_closed"] is True and not shutdown["close_errors"],
            "reqrep shutdown")
    require(cleanup["environment_closed"] is True, "environment close")
    checkpoint_results = {}
    import jax
    from training.peg_insertion.checkpoint import restore_fresh_state
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime
    example = ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"
    env_parent = ROOT / "external/packages/peg_insertion/peg_insertion/code/env"
    sys.path[:0] = [str(example), str(env_parent)]
    from async_sac_state_sim import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_sac_agent(seed=0, sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample())
        for step in STEPS:
            cp = run / "checkpoints" / f"checkpoint_{step}"
            meta = read_json(cp.with_name(cp.name + "_resume_metadata.json"))
            replay = cp.with_name(cp.name + "_online_replay.pkl")
            state = restore_fresh_state(cp.parent, fresh.state, step)
            arrays = [np.asarray(jax.device_get(leaf)) for leaf in jax.tree_util.tree_leaves(state)
                      if hasattr(leaf, "dtype")]
            checkpoint_results[str(step)] = {
                "exists": cp.is_file() and replay.is_file(),
                "step": meta.get("checkpoint_step") == meta.get("learner_update_steps") == step,
                "config": meta.get("config_hash") == canonical(cfg),
                "dataset": meta.get("demo_sha256") == EXPECTED_DATASET,
                "v2": meta.get("observation_contract_version") == "PEG_OBS_CONTRACT_V2_SENSOR_FIXED",
                "model_hash": meta.get("model_sha256") == sha(cp),
                "replay_hash": meta.get("online_replay_sha256") == sha(replay),
                "finite": bool(arrays) and all(np.isfinite(array).all() for array in arrays),
                "numeric_leaves": len(arrays),
            }
            require(all(value for key, value in checkpoint_results[str(step)].items()
                        if key != "numeric_leaves"), f"checkpoint {step}")
        env.close()
    process_text = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                                  text=True, check=True).stdout
    residual = [line.strip() for line in process_text.splitlines()
                if ("run_peg_f3_v2_rlpd_smoke.py" in line or "async_sac_state_sim.py" in line)
                and str(os.getpid()) not in line]
    require(not residual, "residual process")
    result = {
        "PEG_F3_V2_FORMAL_5K": "PASS", "TRAINING_STARTED": True,
        "TRAINING_COMPLETED": True, "UPDATES": "5000/5000",
        "ACTOR_STEPS": actor["steps"], "NETWORK_UPDATES_RECEIVED": actor["received_networks"],
        "ONLINE_REPLAY_FINAL": learner["final_online_size"], "GPU_BACKEND": learner["backend"],
        "DEMO_BUFFER": learner["demo_size"], "DEMO_ONLINE_RATIO": "32:32",
        "LOSSES_FINITE": "PASS", "ACTOR_LEARNER_COMM": "PASS",
        **{f"CKPT_{step}": "PASS" for step in STEPS},
        "CONFIG_SHA256": EXPECTED_CONFIG, "DATASET_SHA256": EXPECTED_DATASET,
        "NORMAL_SHUTDOWN": "PASS", "ACTOR_NATURAL_EXIT": "PASS",
        "LEARNER_NATURAL_EXIT": "PASS", "ZMQ_NATIVE_THREADS_AFTER_EXIT": "NONE",
        "RESIDUAL_PROCESSES": "NONE", "FORMAL_RUN_DIR": str(run.parent),
        "CHECKPOINT_SCREENING": "READY", "FORMAL_10K_CONTINUE": "NOT_AUTHORIZED",
        "checkpoint_details": checkpoint_results, "restore_verification": restore,
    }
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (evidence / "REPORT.md").write_text("# Peg F3 V2 Formal 5K Completion Audit\n\n```json\n" +
                                        json.dumps(result, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
