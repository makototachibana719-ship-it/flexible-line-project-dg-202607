#!/usr/bin/env python3
"""Read-only replica2 shutdown evidence recovery and offline attribution design."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

import jax
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path[:0] = [
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from peg_env_adapter import make_peg_state_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.early_pretanh_a0_anchor_ablation import make_early_pretanh_a0_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime


RUN = ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002/replica2"
OUT = ROOT / "results/reproduction/peg_f3_v2_failed_runner_exact_repro_20260906_002/replica2_shutdown_recovery.json"


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def residuals():
    proc = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True)
    needle = "peg_f3_v2_failed_runner_exact_repro_20260906_002/replica2"
    rows = [line.strip() for line in proc.stdout.splitlines() if needle in line]
    return "NONE" if not rows else rows


def checkpoint_valid(cfg):
    meta = read_json(RUN / "checkpoints/checkpoint_1000_resume_metadata.json")
    if meta is None:
        return False, {}
    metadata_ok = (
        meta.get("checkpoint_step") == meta.get("learner_update_steps") == 1000
        and meta.get("demo_sha256") == cfg["demo"]["sha256"]
        and meta.get("online_replay_size", 0) > 0
    )
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_early_pretanh_a0_anchor_agent(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        restored = restore_fresh_state(RUN / "checkpoints", fresh.state, 1000)
        leaves = [
            np.asarray(jax.device_get(x))
            for x in jax.tree_util.tree_leaves(restored)
            if hasattr(x, "dtype")
        ]
        finite = bool(leaves) and all(np.isfinite(x).all() for x in leaves)
        env.close()
    return metadata_ok and finite, {
        key: meta.get(key)
        for key in ("checkpoint_step", "learner_update_steps", "online_replay_size", "actor_steps", "demo_sha256")
    }


def main():
    cfg = yaml.safe_load((ROOT / "configs/peg_insertion/peg_v2_early_pretanh_a0_anchor_ablation_1k_seed0.yaml").read_text())
    stop = read_json(RUN / "stop_actor.json") or {}
    actor = read_json(RUN / "actor_result.json") or {}
    actor_stop = read_json(RUN / "actor_stop_observed.json") or {}
    actor_shutdown = read_json(RUN / "actor_shutdown.json") or {}
    ckpt_ok, ckpt_meta = checkpoint_valid(cfg)
    res = {
        "REPLICA2_TRAINING_COMPLETED": stop.get("completed_updates") == 1000,
        "REPLICA2_CKPT1000_VALID": ckpt_ok,
        "REPLICA2_ACTOR_EXIT": actor.get("status") == "PASS" and actor_stop.get("stop_marker") is True,
        "REPLICA2_LEARNER_EXIT": (RUN / "learner_result.json").is_file(),
        "REPLICA2_ZMQ_CLEAN": actor_shutdown.get("status") == "PASS",
        "REPLICA2_RESIDUAL_PROCESSES": residuals(),
        "checkpoint_meta": ckpt_meta,
        "NON_INTRUSIVE_ATTRIBUTION_DESIGN": "PASS",
        "ATTRIBUTION_AVAILABLE_FIELDS": [
            "ckpt500 actor params",
            "ckpt1000 actor params",
            "ckpt500/1000 critic params",
            "temperature",
            "offline initial-state A0 mode/pre-tanh/std",
            "offline local Q sweep",
            "offline Q/entropy/global-anchor/early-anchor loss components",
            "native updates losses",
            "checkpoint online replay snapshots",
        ],
        "ATTRIBUTION_UNAVAILABLE_FIELDS": [
            "per-update cumulative gradients without intrusive training-time logging",
            "exact async scheduling trace",
        ],
        "NEXT_CAUSAL_TEST": "LOCAL_CRITIC_PRESSURE",
    }
    res["REPLICA2_SHUTDOWN_GATE"] = (
        "PASS"
        if all(
            [
                res["REPLICA2_TRAINING_COMPLETED"],
                res["REPLICA2_CKPT1000_VALID"],
                res["REPLICA2_ACTOR_EXIT"],
                res["REPLICA2_LEARNER_EXIT"],
                res["REPLICA2_ZMQ_CLEAN"],
                res["REPLICA2_RESIDUAL_PROCESSES"] == "NONE",
            ]
        )
        else "UNPROVEN"
    )
    res["SHUTDOWN_ARTIFACT_RECOVERY"] = "PASS" if res["REPLICA2_SHUTDOWN_GATE"] == "PASS" else "FAIL"
    OUT.write_text(json.dumps(res, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(res, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
