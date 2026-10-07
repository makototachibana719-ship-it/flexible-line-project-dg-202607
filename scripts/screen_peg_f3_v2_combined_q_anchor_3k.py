#!/usr/bin/env python3
"""Inference-only fixed screening for Stage-E combined 3K checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np
import yaml

from scripts.screen_peg_f3_v2_formal_5k import aggregate, contact_flags, raw_env
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.combined_q_anchor_ablation import make_combined_q_anchor_agent
from training.peg_insertion.observation_contract import V2, validate_config
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_combined_q_anchor_ablation_3k_seed0.yaml"
RUN = ROOT / "results/ablation/peg_f3_v2_combined_q_anchor_ablation_3k_seed0_run_002"
OUT = ROOT / "results/validation/peg/f3_v2_combined_q_anchor_3k_behavior_20260906_001"
STEPS = (1000, 2000, 3000)
SEEDS = (0, 1, 2)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    require(not OUT.exists(), "fresh screening output required")
    OUT.mkdir(parents=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    require(validate_config(cfg, for_training=True) == V2, "V2 config")
    require(jax.default_backend() == "gpu", "GPU inference backend")
    sys.path[:0] = [
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/example/async_rlpd_peg_sim"),
        str(ROOT / "external/packages/peg_insertion/peg_insertion/code/env"),
    ]
    from peg_env_adapter import make_peg_state_env

    rows = []
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        fresh = make_combined_q_anchor_agent(
            0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
        for step in STEPS:
            state = restore_fresh_state(RUN / "checkpoints", fresh.state, step)
            agent = fresh.replace(state=state)
            for seed in SEEDS:
                obs, _ = env.reset(seed=seed)
                first = float(np.linalg.norm(obs[13:16]))
                mins = {"position": first, "lateral": float("inf"), "axial": float("inf"),
                        "orientation": float("inf")}
                ret = 0.0
                near = sat = 0
                max_force = 0.0
                link01 = unexpected = success = terminated = truncated = False
                t0 = time.monotonic()
                for index in range(500):
                    action = np.asarray(jax.device_get(agent.sample_actions(observations=obs, argmax=True)))
                    require(action.shape == (7,) and np.isfinite(action).all(), "action contract")
                    sat += int((np.abs(action) >= .99).sum())
                    obs, reward, terminated, truncated, info = env.step(action)
                    expected_success = (info["lateral_error"] <= .002 and
                                        info["axial_error"] <= .003 and
                                        info["orientation_error"] <= .05)
                    require(bool(info["is_success"]) == expected_success, "success semantics")
                    mins["position"] = min(mins["position"], float(info["position_error"]))
                    mins["lateral"] = min(mins["lateral"], float(info["lateral_error"]))
                    mins["axial"] = min(mins["axial"], float(info["axial_error"]))
                    mins["orientation"] = min(mins["orientation"], float(info["orientation_error"]))
                    near += int(info["lateral_error"] <= .010 and info["axial_error"] <= .020)
                    max_force = max(max_force, float(info["force_norm"]))
                    l, u, _ = contact_flags(env)
                    link01 |= l
                    unexpected |= u
                    ret += float(reward)
                    success = bool(info["is_success"])
                    if terminated or truncated:
                        break
                row = {
                    "checkpoint": step, "episode": seed, "seed": seed,
                    "success": success, "steps": index + 1, "return": ret,
                    "first_position_error": first, "min_position_error": mins["position"],
                    "approach_reduction": first - mins["position"],
                    "min_lateral_error": mins["lateral"], "min_axial_error": mins["axial"],
                    "min_orientation_error": mins["orientation"],
                    "near_hole_transition_count": near, "near_hole_reached": near > 0,
                    "max_force": max_force, "link01_contact": link01,
                    "unexpected_contact": unexpected, "contact_health": not (link01 or unexpected),
                    "action_saturation": sat / (7 * (index + 1)),
                    "termination_reason": "SUCCESS" if success else ("TRUNCATED" if truncated else
                                          ("TERMINATED" if terminated else "EVALUATOR_HORIZON")),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "wall_seconds": time.monotonic() - t0,
                }
                rows.append(row)
                with (OUT / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
        env.close()

    by_step = {str(step): aggregate([r for r in rows if r["checkpoint"] == step])
               for step in STEPS}
    near = [by_step[str(step)]["near_hole_episode_rate"] for step in STEPS]
    contact = [by_step[str(step)]["contact_failures"] for step in STEPS]
    approach = [by_step[str(step)]["mean_approach_reduction"] for step in STEPS]
    lateral = [by_step[str(step)]["median_min_lateral"] for step in STEPS]
    behavior_improved = max(near) > 0 or approach[-1] > approach[0] or lateral[-1] < lateral[0] or contact[-1] < contact[0]
    result = {
        "STAGE_E_BEHAVIOR_SCREENING": "PASS",
        "TRAINING_STARTED": False,
        "CHECKPOINTS": list(STEPS),
        "SEEDS": list(SEEDS),
        "summary": by_step,
        "BEHAVIOR_TREND": "IMPROVED" if behavior_improved else "NO_CLEAR_IMPROVEMENT",
        "NEAR_HOLE_TREND": near,
        "CONTACT_TREND": contact,
        "APPROACH_TREND": approach,
        "LATERAL_TREND": lateral,
    }
    dump(OUT / "validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
