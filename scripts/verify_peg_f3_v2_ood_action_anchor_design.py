#!/usr/bin/env python3
"""Dry-run design gate for Peg F3 V2 OOD-action-anchor ablation."""
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

from training.peg_insertion.ood_action_anchor_ablation import make_ood_action_anchor_agent
from training.peg_insertion.conservative_q_ablation import ConservativeQSACAgent

CONFIG = ROOT / "configs/peg_insertion/peg_v2_ood_action_anchor_ablation_1k_seed0.yaml"
OUT = ROOT / "results/validation/peg/f3_v2_ood_action_anchor_design_20260906_001"


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
    require(len(rows) == 1921, "demo cardinality")
    online = rows[:32]
    demo = rows[-32:]
    batch = {}
    for key in rows[0]:
        batch[key] = jnp.asarray(np.concatenate([
            np.asarray([row[key] for row in online]),
            np.asarray([row[key] for row in demo]),
        ], axis=0))
    agent = make_ood_action_anchor_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    params = agent.state.params
    rng = jax.random.PRNGKey(20260906)

    def sac_actor_loss(params):
        loss, _ = ConservativeQSACAgent.policy_loss_fn(agent, batch, params, rng)
        return loss

    def ood_weighted_loss(params):
        raw = agent.ood_action_anchor_loss_fn(batch, params, rng)
        return cfg["ood_action_anchor"]["selected_lambda"] * raw

    sac_grad = jax.grad(sac_actor_loss)(params)
    ood_grad = jax.grad(ood_weighted_loss)(params)
    sac_norm = float(jax.device_get(tree_l2(sac_grad)))
    ood_norm = float(jax.device_get(tree_l2(ood_grad)))
    ratio = ood_norm / max(sac_norm, 1e-12)
    require(np.isfinite(ratio), "nonfinite gradient ratio")
    require(0.05 <= ratio <= 0.20, "OOD anchor gradient ratio outside target band")
    result = {
        "OOD_ACTION_ANCHOR_DESIGN": "PASS",
        "SCOPE": "ONLINE_ROWS",
        "TARGET": cfg["ood_action_anchor"]["target"],
        "OOD_ACTION_ANCHOR_LAMBDA": cfg["ood_action_anchor"]["selected_lambda"],
        "OOD_GRAD_RATIO": ratio,
        "OOD_GRAD_DOMINATES": "NO" if ratio < 0.5 else "YES",
        "TRAINING_STARTED": False,
        "PROTECTED_ARTIFACTS_CHANGED": "NO",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
