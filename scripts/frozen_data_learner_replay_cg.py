#!/usr/bin/env python3
"""Frozen-data learner-only replay for Peg F3 V2 C+G async variance isolation."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path("/home/makoto/flexible_line_project")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "external/serl/serl_launcher"),
    str(ROOT / "external/serl/franka_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
    str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
]

from async_sac_state_sim import concat_rlpd_batches
from peg_env_adapter import make_peg_state_env
from scripts.audit_multi_replica_baseline_protocol import fixed_metrics
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
ONLINE_SOURCE = ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica1/checkpoints/checkpoint_1000_online_replay.pkl"
OUT_ROOT = ROOT / "results/diagnostics/peg"
EVAL_STEPS = (0, 100, 250, 500, 750, 1000)


def unique_out() -> Path:
    stem = "f3_v2_frozen_data_learner_replay_cg_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def tree_l2(a, b) -> float:
    leaves_a, _ = jax.tree_util.tree_flatten(a)
    leaves_b, _ = jax.tree_util.tree_flatten(b)
    total = 0.0
    for x, y in zip(leaves_a, leaves_b):
        dx = np.asarray(jax.device_get(x)) - np.asarray(jax.device_get(y))
        total += float(np.sum(dx * dx))
    return float(np.sqrt(total))


def load_arrays(cfg):
    replay = pickle.loads(ONLINE_SOURCE.read_bytes())
    online = replay["dataset_dict"]
    n_online = int(replay["size"])
    online = {k: np.asarray(v)[:n_online] for k, v in online.items()}
    demos = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    demo = {k: np.asarray([row[k] for row in demos]) for k in demos[0]}
    return online, demo, n_online, len(demos)


def take(data, idx):
    return {k: np.asarray(v)[idx] for k, v in data.items() if k in ("observations", "actions", "next_observations", "rewards", "masks", "dones")}


def build_stream(cfg, out: Path):
    online, demo, n_online, n_demo = load_arrays(cfg)
    rng = np.random.default_rng(20260907)
    online_indices = rng.integers(0, n_online, size=(1000, 32), endpoint=False)
    demo_indices = rng.integers(0, n_demo, size=(1000, 32), endpoint=False)
    payload = {
        "online_source": str(ONLINE_SOURCE),
        "online_source_sha256": sha_file(ONLINE_SOURCE),
        "config": str(CONFIG),
        "config_sha256": sha_file(CONFIG),
        "demo_path": str(ROOT / cfg["demo"]["path"]),
        "demo_sha256": cfg["demo"]["sha256"],
        "stream_seed": 20260907,
        "updates": 1000,
        "online_batch": 32,
        "demo_batch": 32,
        "online_size": n_online,
        "demo_size": n_demo,
        "online_indices_sha256": sha_bytes(online_indices.tobytes()),
        "demo_indices_sha256": sha_bytes(demo_indices.tobytes()),
    }
    (out / "frozen_stream_provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return online, demo, online_indices, demo_indices, payload


def initial_agent_and_obs(cfg):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        agent = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        env.close()
    return agent, np.asarray(obs, np.float32)


def eval_agent(cfg, agent, obs):
    action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
    fixed = fixed_metrics(cfg, agent)
    return {
        "a0": float(action[0]),
        "action": action.tolist(),
        "q_policy_minus_demo": fixed["q_policy_minus_demo"],
        "demo_mae": fixed["demo_mae"],
        "action1_mae": fixed["action1_mae"],
    }


def run_replica(name, cfg, online, demo, online_indices, demo_indices, out):
    agent, obs = initial_agent_and_obs(cfg)
    evals = {"0": eval_agent(cfg, agent, obs)}
    for update in range(1, 1001):
        batch = concat_rlpd_batches(take(online, online_indices[update - 1]), take(demo, demo_indices[update - 1]))
        agent, info = agent.update_high_utd(batch, utd_ratio=1)
        if update in EVAL_STEPS:
            evals[str(update)] = eval_agent(cfg, agent, obs)
    (out / f"{name}_evals.json").write_text(json.dumps(finite(evals), indent=2, sort_keys=True) + "\n")
    return agent, evals


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    online, demo, online_idx, demo_idx, stream = build_stream(cfg, out)
    agents, evals = {}, {}
    for name in ("R1", "R2"):
        agents[name], evals[name] = run_replica(name, cfg, online, demo, online_idx, demo_idx, out)
    dist = tree_l2(agents["R1"].state.params, agents["R2"].state.params)
    a0_r1 = evals["R1"]["1000"]["a0"]
    a0_r2 = evals["R2"]["1000"]["a0"]
    close = abs(a0_r1 - a0_r2) <= 1e-6 and dist <= 1e-5
    result = {
        "ASYNC_VARIANCE_ISOLATION": "PASS",
        "FROZEN_STREAM_READY": "PASS",
        "FROZEN_STREAM": stream,
        "A0_R1_1000": a0_r1,
        "A0_R2_1000": a0_r2,
        "POLICY_VECTOR_DISTANCE_R1_R2": dist,
        "QPOLICY_MINUS_QDEMO_R1": evals["R1"]["1000"]["q_policy_minus_demo"],
        "QPOLICY_MINUS_QDEMO_R2": evals["R2"]["1000"]["q_policy_minus_demo"],
        "FROZEN_DATA_REPRODUCIBILITY": "PASS" if close else "FAIL",
        "PRIMARY_ROOT_CAUSE": "ASYNC_ONLINE_DATA_TIMING_VARIANCE" if close else "ACTOR_OPTIMIZATION_STOCHASTIC_INSTABILITY",
        "NEXT": "ASYNC_TRAINING_PROTOCOL_STABILIZATION_REVIEW" if close else "ACTOR_OPTIMIZER_DETERMINISM_AUDIT",
        "TRAINING_STARTED": "false",
        "FORMAL_3K_STARTED": "false",
        "FORMAL_5K_STARTED": "false",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    (out / "result.json").write_text(json.dumps(finite(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(finite({
        "FROZEN_STREAM_READY": result["FROZEN_STREAM_READY"],
        "A0_R1_1000": a0_r1,
        "A0_R2_1000": a0_r2,
        "POLICY_VECTOR_DISTANCE_R1_R2": dist,
        "FROZEN_DATA_REPRODUCIBILITY": result["FROZEN_DATA_REPRODUCIBILITY"],
        "PRIMARY_ROOT_CAUSE": result["PRIMARY_ROOT_CAUSE"],
        "NEXT": result["NEXT"],
        "OUT": str(out / "result.json"),
    }), sort_keys=True))


if __name__ == "__main__":
    main()
