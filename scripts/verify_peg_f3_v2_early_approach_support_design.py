#!/usr/bin/env python3
"""Dry-run design gate for Peg F3 V2 early-approach a0 support."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from training.peg_insertion.combined_q_anchor_ablation import CombinedQAnchorSACAgent
from training.peg_insertion.conservative_q_ablation import ConservativeQSACAgent
from training.peg_insertion.early_approach_support_ablation import (
    early_approach_mask,
    make_early_approach_support_agent,
)

CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_approach_support_ablation_1k_seed0.yaml"
TRAJ = ROOT / "results/diagnostics/peg/f3_v2_trajectory_causal_after_ood_anchor_20260906_002/result.json"
OUT = ROOT / "results/validation/peg/f3_v2_early_approach_support_design_20260906_001"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tree_l2(tree):
    leaves = [jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "dtype")]
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def main():
    require(not OUT.exists(), "fresh design output required")
    cfg = yaml.safe_load(CONFIG.read_text())
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    mask = early_approach_mask(ROOT, cfg)
    actions = np.asarray([row["actions"] for row in rows])
    require(int(mask.sum()) == 363, "early approach row count")
    require(float(actions[mask, 0].min()) == 0.0 and float(actions[mask, 0].max()) == 0.0,
            "early approach a0 Demo range")
    traj = json.loads(TRAJ.read_text())
    step1_current = float(traj["families"]["current"]["first_divergence"]["action"][0])
    step1_baseline = float(traj["families"]["original"]["first_divergence"]["action"][0])

    online = rows[:32]
    demo = rows[-32:]
    batch = {}
    for key in rows[0]:
        batch[key] = jnp.asarray(np.concatenate([
            np.asarray([row[key] for row in online]),
            np.asarray([row[key] for row in demo]),
        ], axis=0))
    agent = make_early_approach_support_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    params = agent.state.params
    rng = jax.random.PRNGKey(20260906)

    def sac_actor_loss(params):
        loss, _ = ConservativeQSACAgent.policy_loss_fn(agent, batch, params, rng)
        return loss

    def global_anchor_weighted_loss(params):
        return cfg["demo_actor_anchor"]["selected_lambda"] * CombinedQAnchorSACAgent.demo_anchor_loss_fn(
            agent, batch, params, rng)

    def early_weighted_loss(params):
        return cfg["early_approach_support"]["selected_lambda"] * agent.early_approach_support_loss_fn(
            params, rng)

    sac_norm = float(jax.device_get(tree_l2(jax.grad(sac_actor_loss)(params))))
    global_norm = float(jax.device_get(tree_l2(jax.grad(global_anchor_weighted_loss)(params))))
    early_norm = float(jax.device_get(tree_l2(jax.grad(early_weighted_loss)(params))))
    ratio = early_norm / max(sac_norm, 1e-12)
    require(np.isfinite(ratio), "nonfinite gradient ratio")
    require(0.05 <= ratio <= 0.15, "early support gradient ratio outside target band")
    require(ratio <= 0.30, "early support gradient dominates")
    result = {
        "EARLY_APPROACH_SUPPORT_DESIGN": "PASS",
        "EARLY_APPROACH_PREDICATE": cfg["early_approach_support"]["predicate"],
        "EARLY_APPROACH_DEMO_ROWS": int(mask.sum()),
        "A0_DEMO_RANGE": [float(actions[mask, 0].min()), float(actions[mask, 0].max())],
        "STEP1_A0_BASELINE": step1_baseline,
        "STEP1_A0_CURRENT": step1_current,
        "LAMBDA_EARLY": cfg["early_approach_support"]["selected_lambda"],
        "SAC_GRAD_NORM": sac_norm,
        "GLOBAL_ANCHOR_GRAD_NORM": global_norm,
        "EARLY_SUPPORT_GRAD_NORM": early_norm,
        "EARLY_SUPPORT_GRAD_RATIO": ratio,
        "PARAMETERS_UPDATED": False,
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
