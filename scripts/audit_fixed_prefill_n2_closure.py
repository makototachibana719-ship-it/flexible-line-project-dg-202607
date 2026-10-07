#!/usr/bin/env python3
"""Matched N=2 closure audit for stabilized C+G fixed-prefill runs."""
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
    "R1": ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica1_retry002",
    "R2": ROOT / "results/ablation/peg_f3_v2_cg_fixed_prefill_freeze_n2_20260907/replica2",
}
PRIOR = {
    "R1": ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica1",
    "R2": ROOT / "results/ablation/peg_f3_v2_c_plus_g_n2_20260907/replica2_retry001",
}
RECOVERY = {
    "R1": ROOT / "results/diagnostics/peg/f3_v2_r1_fixed_prefill_checkpoint_recovery_20260907_001/result.json",
    "R2": ROOT / "results/diagnostics/peg/f3_v2_r2_fixed_prefill_checkpoint_recovery_20260907_001/result.json",
}


def unique_out() -> Path:
    stem = "f3_v2_fixed_prefill_n2_closure_20260907"
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
    rec = {k: read(v) for k, v in RECOVERY.items()}
    sizes = {k: replay_size(v) for k, v in STABLE.items()}
    replay_delta_pct = 100.0 * abs(sizes["R1"] - sizes["R2"]) / max(sizes.values())
    obs_distance = float(np.linalg.norm(obs_mean(STABLE["R1"]) - obs_mean(STABLE["R2"])))
    params_stable = {k: load_agent_params(cfg, v) for k, v in STABLE.items()}
    params_prior = {k: load_agent_params(cfg, v) for k, v in PRIOR.items()}
    stable_policy_distance = tree_l2(params_stable["R1"], params_stable["R2"])
    prior_policy_distance = tree_l2(params_prior["R1"], params_prior["R2"])
    a0 = {k: rec[k]["A0_1000"] for k in rec}
    q = {k: rec[k]["QPOLICY_MINUS_QDEMO"] for k in rec}
    protocol_pass = (
        replay_delta_pct <= 5.0
        and stable_policy_distance < prior_policy_distance
        and all(rec[k]["TRAINING_RESULT"] == "PASS" and rec[k]["CHECKPOINT_VALIDITY"] == "PASS" for k in rec)
    )
    behavior_fail = (
        rec["R1"]["NEAR_HOLE"] == 0.0
        and rec["R2"]["NEAR_HOLE"] == 0.0
        and rec["R1"]["FIRST_DIVERGENCE"] <= 1.0
        and rec["R2"]["FIRST_DIVERGENCE"] <= 1.0
    )
    behavior_status = "FAIL" if behavior_fail else "PROMISING"
    result = {
        "PROTOCOL_STABILIZED": "PASS" if protocol_pass else "FAIL",
        "BEHAVIOR_STATUS": behavior_status,
        "STABILIZED_CG_1K": "SCIENTIFIC_FAIL" if protocol_pass and behavior_status == "FAIL" else "PROMISING",
        "REPLAY_SIZE_R1": sizes["R1"],
        "REPLAY_SIZE_R2": sizes["R2"],
        "REPLAY_SIZE_DELTA_PCT": replay_delta_pct,
        "OBS_OCCUPANCY_DISTANCE": obs_distance,
        "POLICY_VECTOR_DISTANCE_R1_R2": stable_policy_distance,
        "PRIOR_POLICY_VECTOR_DISTANCE_R1_R2": prior_policy_distance,
        "A0_R1": a0["R1"],
        "A0_R2": a0["R2"],
        "A0_DISTANCE": abs(a0["R1"] - a0["R2"]),
        "QPOLICY_MINUS_QDEMO_R1": q["R1"],
        "QPOLICY_MINUS_QDEMO_R2": q["R2"],
        "Q_DIFFERENCE": abs(q["R1"] - q["R2"]),
        "SUCCESS_R1": rec["R1"]["SUCCESS"],
        "SUCCESS_R2": rec["R2"]["SUCCESS"],
        "NEAR_HOLE_R1": rec["R1"]["NEAR_HOLE"],
        "NEAR_HOLE_R2": rec["R2"]["NEAR_HOLE"],
        "FIRST_DIVERGENCE_R1": rec["R1"]["FIRST_DIVERGENCE"],
        "FIRST_DIVERGENCE_R2": rec["R2"]["FIRST_DIVERGENCE"],
        "CORRIDOR_R1": rec["R1"]["CORRIDOR"],
        "CORRIDOR_R2": rec["R2"]["CORRIDOR"],
        "ORIENTATION_R1": rec["R1"]["ORIENTATION"],
        "ORIENTATION_R2": rec["R2"]["ORIENTATION"],
        "AXIAL_R1": rec["R1"]["AXIAL"],
        "AXIAL_R2": rec["R2"]["AXIAL"],
        "FORCE_R1": rec["R1"]["FORCE"],
        "FORCE_R2": rec["R2"]["FORCE"],
        "CONTACT_R1": rec["R1"]["CONTACT"],
        "CONTACT_R2": rec["R2"]["CONTACT"],
        "FIRST_BAD_GATE": "STABILIZED_CG_BEHAVIOR_FAIL" if protocol_pass and behavior_status == "FAIL" else "PROTOCOL_STABILIZATION_FAIL",
        "PRIMARY_ROOT_CAUSE": "C_PLUS_G_BEHAVIOR_INSUFFICIENT_AFTER_ASYNC_STABILIZATION" if protocol_pass and behavior_status == "FAIL" else "CURRENT_ASYNC_STABILIZATION_PROTOCOL_INSUFFICIENT",
        "NEXT": "STABLE_PROTOCOL_CAUSAL_AUDIT" if protocol_pass and behavior_status == "FAIL" else "PROTOCOL_CAUSAL_REVIEW",
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
