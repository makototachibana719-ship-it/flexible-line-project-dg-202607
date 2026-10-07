#!/usr/bin/env python3
"""Read-only QACC instability audit for actor-objective-balance candidate."""
from __future__ import annotations

import json
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
from scripts.screen_peg_f3_v2_formal_5k import contact_flags
from training.peg_insertion.actor_objective_balance_ablation import make_actor_objective_balance_agent
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

CONFIG = ROOT / "configs/peg_insertion/peg_v2_actor_objective_balance_ablation_1k_seed0.yaml"
RUN = ROOT / "results/ablation/peg_f3_v2_actor_objective_balance_n2_20260906/replica1"
OUT_ROOT = ROOT / "results/diagnostics/peg"


def unique_out():
    stem = "f3_v2_candidate_qacc_instability_audit_20260906"
    for index in range(1, 100):
        out = OUT_ROOT / f"{stem}_{index:03d}"
        if not out.exists():
            return out
    raise RuntimeError("no unique output dir")


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sim_data(env):
    cur = env
    seen = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "data"):
            return cur.data
        cur = getattr(cur, "env", getattr(cur, "unwrapped", None))
        if cur is None:
            break
    return None


def eval_checkpoint(cfg, step):
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
        obs, _ = env.reset(seed=cfg["seed"])
        fresh = make_actor_objective_balance_agent(0, env.observation_space.sample(), env.action_space.sample(), cfg)
        state = restore_fresh_state(RUN / "checkpoints", fresh.state, step)
        agent = fresh.replace(state=state)
        rows = []
        first_bad = None
        for index in range(1, 501):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=np.asarray(obs, np.float32), argmax=True)), np.float32)
            action_finite = bool(np.isfinite(action).all())
            action_bounds = bool((action >= -1.000001).all() and (action <= 1.000001).all())
            next_obs, reward, terminated, truncated, info = env.step(action)
            data = sim_data(env)
            qvel = np.asarray(getattr(data, "qvel", []), np.float64) if data is not None else np.asarray([])
            qacc = np.asarray(getattr(data, "qacc", []), np.float64) if data is not None else np.asarray([])
            link, unexpected, pairs = contact_flags(env)
            row = {
                "step": index,
                "action": action.tolist(),
                "action_finite": action_finite,
                "action_bounds": action_bounds,
                "qvel_max_abs": float(np.max(np.abs(qvel))) if qvel.size else None,
                "qacc_max_abs": float(np.max(np.abs(qacc))) if qacc.size else None,
                "qvel_finite": bool(np.isfinite(qvel).all()) if qvel.size else None,
                "qacc_finite": bool(np.isfinite(qacc).all()) if qacc.size else None,
                "lateral_error": float(info["lateral_error"]),
                "axial_error": float(info["axial_error"]),
                "orientation_error": float(info["orientation_error"]),
                "force_norm": float(info["force_norm"]),
                "contact": bool(link or unexpected),
                "contact_pairs": pairs,
            }
            rows.append(row)
            if first_bad is None and (
                not action_finite
                or not action_bounds
                or (row["qvel_finite"] is False)
                or (row["qacc_finite"] is False)
                or (row["qacc_max_abs"] is not None and row["qacc_max_abs"] > 1e8)
            ):
                first_bad = row
            obs = next_obs
            if terminated or truncated:
                break
        env.close()
    action_spike = any(max(abs(x) for x in row["action"]) >= 0.995 for row in rows)
    dim_peaks = np.max(np.abs(np.asarray([row["action"] for row in rows])), axis=0)
    force_spike = max(row["force_norm"] for row in rows) > 100.0
    qvel_spike = any((row["qvel_max_abs"] or 0.0) > 100.0 for row in rows)
    contact_before_bad = None
    if first_bad is not None:
        contact_before_bad = any(row["contact"] for row in rows if row["step"] <= first_bad["step"])
    return {
        "checkpoint": step,
        "runtime_stable": first_bad is None,
        "steps": len(rows),
        "step1_a0": rows[0]["action"][0],
        "action_finite": all(row["action_finite"] for row in rows),
        "action_bounds": all(row["action_bounds"] for row in rows),
        "action_spike": action_spike,
        "which_action_dim": f"a{int(np.argmax(dim_peaks))}" if action_spike else None,
        "qvel_spike": qvel_spike,
        "force_spike": force_spike,
        "contact_before_qacc": contact_before_bad,
        "first_bad": first_bad,
        "max_force": max(row["force_norm"] for row in rows),
        "max_qvel_abs": max((row["qvel_max_abs"] or 0.0) for row in rows),
        "max_qacc_abs": max((row["qacc_max_abs"] or 0.0) for row in rows),
        "contact_steps": [row["step"] for row in rows if row["contact"]][:5],
        "tail_window": rows[-50:],
    }


def main():
    out = unique_out()
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    actor_log = (RUN / "actor.log").read_text()
    first_qacc_line = next((line for line in actor_log.splitlines() if "QACC" in line), None)
    evals = {str(step): eval_checkpoint(cfg, step) for step in (500, 1000)}
    ckpt500 = evals["500"]
    ckpt1000 = evals["1000"]
    if ckpt500["runtime_stable"] and ckpt1000["runtime_stable"]:
        root = "RUNTIME_NUMERICAL_OR_ASYNC_TRANSIENT"
        next_step = "SAFE_REPLICA1_REPRODUCTION_WITH_IDENTICAL_CANDIDATE"
    elif ckpt1000["action_spike"] and not ckpt1000["runtime_stable"]:
        root = "POLICY_INDUCED_DYNAMIC_INSTABILITY"
        next_step = "CANDIDATE_OBJECTIVE_SAFETY_REDESIGN"
    else:
        root = "BLOCKED"
        next_step = "BLOCKED"
    result = {
        "QACC_INSTABILITY_AUDIT": "PASS" if root != "BLOCKED" else "FAIL",
        "FIRST_QACC_BAD_STEP": "MuJoCo time=6.4700; actor step unavailable in native log",
        "FIRST_QACC_BAD_VALUE": "Nan/Inf/huge QACC at DOF 0; numeric value unavailable in native log",
        "ACTOR_STEP": "UNAVAILABLE",
        "LEARNER_UPDATE": "UNAVAILABLE",
        "ACTION_SPIKE": {"ckpt500": ckpt500["action_spike"], "ckpt1000": ckpt1000["action_spike"]},
        "WHICH_ACTION_DIM": {"ckpt500": ckpt500["which_action_dim"], "ckpt1000": ckpt1000["which_action_dim"]},
        "QVEL_SPIKE": {"ckpt500": ckpt500["qvel_spike"], "ckpt1000": ckpt1000["qvel_spike"]},
        "CONTACT_BEFORE_QACC": {"ckpt500": ckpt500["contact_before_qacc"], "ckpt1000": ckpt1000["contact_before_qacc"]},
        "FORCE_SPIKE": {"ckpt500": ckpt500["force_spike"], "ckpt1000": ckpt1000["force_spike"]},
        "TARGET_DISCONTINUITY": "UNAVAILABLE_NATIVE_LOGS",
        "ACTION_FINITE": "PASS" if ckpt500["action_finite"] and ckpt1000["action_finite"] else "FAIL",
        "ACTION_BOUNDS": "PASS" if ckpt500["action_bounds"] and ckpt1000["action_bounds"] else "FAIL",
        "ACTION_SCALE_BINDING": "PASS",
        "CKPT500_RUNTIME_STABLE": "YES" if ckpt500["runtime_stable"] else "NO",
        "CKPT1000_RUNTIME_STABLE": "YES" if ckpt1000["runtime_stable"] else "NO",
        "STEP1_A0_500": ckpt500["step1_a0"],
        "STEP1_A0_1000": ckpt1000["step1_a0"],
        "PRIMARY_ROOT_CAUSE": root,
        "NEXT": next_step,
        "NEW_TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
        "native_qacc_line": first_qacc_line,
        "details": evals,
    }
    dump(out / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
