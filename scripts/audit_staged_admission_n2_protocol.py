#!/usr/bin/env python3
"""Matched N=2 protocol audit for deterministic staged admission."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import jax
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

from peg_env_adapter import make_peg_state_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_1k_seed0.yaml"
OUT_ROOT = ROOT / "results/diagnostics/peg"
STABLE = {
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_deterministic_staged_admission_n2_20260907/replica2",
}
PRIOR = {
    "R1": ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica1",
    "R2": ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica2_retry001",
}
def unique_out() -> Path:
    stem = "f3_v2_staged_admission_n2_protocol_20260907"
    for idx in range(1, 100):
        out = OUT_ROOT / f"{stem}_{idx:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def read(path: Path):
    return json.loads(path.read_text())


def obs_mean(run: Path) -> np.ndarray:
    payload = pickle.loads((run / "checkpoints/checkpoint_1000_online_replay.pkl").read_bytes())
    data = payload["dataset_dict"]
    size = int(payload["size"])
    return np.asarray(data["observations"][:size], np.float64).mean(axis=0)


def replay_size(run: Path) -> int:
    return int(read(run / "learner_result.json")["final_online_size"])


def load_agent_params(cfg, run: Path):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_combined_q_anchor_agent(cfg["seed"], env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(run / "checkpoints", fresh.state, 1000)
        env.close()
    return state.params


def tree_l2(a, b) -> float:
    la, _ = jax.tree_util.tree_flatten(a)
    lb, _ = jax.tree_util.tree_flatten(b)
    total = 0.0
    for x, y in zip(la, lb):
        d = np.asarray(jax.device_get(x)) - np.asarray(jax.device_get(y))
        total += float(np.sum(d * d))
    return float(np.sqrt(total))


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    sizes = {k: replay_size(v) for k, v in STABLE.items()}
    replay_delta_pct = 100.0 * abs(sizes["R1"] - sizes["R2"]) / max(sizes.values())
    obs_distance = float(np.linalg.norm(obs_mean(STABLE["R1"]) - obs_mean(STABLE["R2"])))
    params_stable = {k: load_agent_params(cfg, v) for k, v in STABLE.items()}
    params_prior = {k: load_agent_params(cfg, v) for k, v in PRIOR.items()}
    stable_policy_distance = tree_l2(params_stable["R1"], params_stable["R2"])
    prior_policy_distance = tree_l2(params_prior["R1"], params_prior["R2"])
    learner = {k: read(v / "learner_result.json") for k, v in STABLE.items()}
    protocol_pass = (
        replay_delta_pct <= 2.0
        and obs_distance <= 1e-9
        and stable_policy_distance < prior_policy_distance
        and all(learner[k]["completed_updates"] == 1000 and learner[k]["final_online_size"] == 604 for k in learner)
    )
    result = {
        "STAGED_PROTOCOL_REPRODUCIBILITY": "PASS" if protocol_pass else "FAIL",
        "REPLAY_SIZE_R1": sizes["R1"],
        "REPLAY_SIZE_R2": sizes["R2"],
        "REPLAY_SIZE_DELTA_PCT": replay_delta_pct,
        "OBS_OCCUPANCY_DISTANCE": obs_distance,
        "POLICY_VECTOR_DISTANCE_R1_R2": stable_policy_distance,
        "PRIOR_POLICY_VECTOR_DISTANCE_R1_R2": prior_policy_distance,
        "FIRST_BAD_GATE": "NONE" if protocol_pass else "STAGED_PROTOCOL_REPRODUCIBILITY",
        "NEXT": "SCIENTIFIC_GEOMETRY_AND_BEHAVIOR_AUDIT" if protocol_pass else "STOP_PROTOCOL_TOO_LOOSE",
        "TRAINING_RUNNING": "false",
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    keys = [k for k in result if k not in ("PRIOR_POLICY_VECTOR_DISTANCE_R1_R2",)]
    print(json.dumps({k: result[k] for k in keys}, sort_keys=True))
    print(str(out / "result.json"))


if __name__ == "__main__":
    main()
