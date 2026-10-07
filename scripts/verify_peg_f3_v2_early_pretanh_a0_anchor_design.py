#!/usr/bin/env python3
"""Dry-run design gate for Peg F3 V2 early pre-tanh a0 anchor."""
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
from training.peg_insertion.early_pretanh_a0_anchor_ablation import (
    early_pretanh_mask,
    make_early_pretanh_a0_anchor_agent,
)

CONFIG = ROOT / "configs/peg_insertion/peg_v2_early_pretanh_a0_anchor_ablation_1k_seed0.yaml"
OUT = ROOT / "results/validation/peg/f3_v2_early_pretanh_a0_anchor_design_20260906_001"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tree_l2(tree):
    leaves = [jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "dtype")]
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def reject(candidate, message):
    try:
        validate_config(candidate)
    except RuntimeError:
        return True
    raise RuntimeError(message)


def validate_config(cfg):
    early = cfg.get("early_pretanh_a0_anchor", {})
    require(early.get("enabled") is True, "pretanh anchor enabled")
    require(early.get("scope") == "demo_early_approach_rows", "early rows only")
    require(early.get("action_dim") == "a0", "a0 only")
    require(early.get("pretanh_source") == "native_tanh_multivariate_normal_diag_distribution_loc",
            "native pretanh loc required")
    require(early.get("loss") == "mse_policy_pretanh_loc_a0_to_zero", "pretanh loss")
    require(cfg["demo_actor_anchor"]["selected_lambda"] == 0.07, "global anchor lambda changed")
    require(cfg["conservative_q"]["selected_lambda"] == 5.0 and cfg["conservative_q"]["scope"] == "demo_tail",
            "conservative-Q changed")
    require(cfg["ablation"]["semantic_diff_fields"] == ["conservative_q", "demo_actor_anchor", "early_pretanh_a0_anchor"],
            "semantic diff fields")


def main():
    require(not OUT.exists(), "fresh design output required")
    cfg = yaml.safe_load(CONFIG.read_text())
    validate_config(cfg)
    rows = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    mask = early_pretanh_mask(ROOT, cfg)
    actions = np.asarray([row["actions"] for row in rows])
    require(int(mask.sum()) == 363, "early approach row count")
    require(float(actions[mask, 0].min()) == 0.0 and float(actions[mask, 0].max()) == 0.0,
            "early approach a0 Demo range")
    online, demo = rows[:32], rows[-32:]
    batch = {}
    for key in rows[0]:
        batch[key] = jnp.asarray(np.concatenate([
            np.asarray([row[key] for row in online]),
            np.asarray([row[key] for row in demo]),
        ], axis=0))
    agent = make_early_pretanh_a0_anchor_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)
    dist = agent.forward_policy(jnp.asarray(agent.config["early_pretanh_a0_anchor"]["observations"][:1]),
                                rng=None, train=False)
    require(type(dist).__name__ == "TanhMultivariateNormalDiag", "unexpected actor distribution")
    require(hasattr(dist, "distribution") and hasattr(dist.distribution, "loc"), "native pretanh loc unavailable")
    mode = np.asarray(jax.device_get(dist.mode()))[0]
    loc = np.asarray(jax.device_get(dist.distribution.loc))[0]
    require(abs(np.tanh(loc[0]) - mode[0]) < 1e-5, "pretanh dimension binding")
    params = agent.state.params
    rng = jax.random.PRNGKey(20260906)

    def sac_actor_loss(params):
        loss, _ = ConservativeQSACAgent.policy_loss_fn(agent, batch, params, rng)
        return loss

    def global_anchor_weighted_loss(params):
        return cfg["demo_actor_anchor"]["selected_lambda"] * CombinedQAnchorSACAgent.demo_anchor_loss_fn(
            agent, batch, params, rng)

    def pretanh_raw_loss(params):
        return agent.early_pretanh_a0_anchor_loss_fn(params, rng)

    sac_norm = float(jax.device_get(tree_l2(jax.grad(sac_actor_loss)(params))))
    global_norm = float(jax.device_get(tree_l2(jax.grad(global_anchor_weighted_loss)(params))))
    raw_norm = float(jax.device_get(tree_l2(jax.grad(pretanh_raw_loss)(params))))
    lam = float(cfg["early_pretanh_a0_anchor"]["selected_lambda"])
    ratio = lam * raw_norm / max(sac_norm, 1e-12)
    require(np.isfinite(ratio), "nonfinite weighted pretanh ratio")
    require(0.05 <= ratio <= 0.15, "weighted pretanh ratio outside target band")
    require(ratio <= 0.20, "weighted pretanh ratio above hard max")
    negatives = {}
    for name, mutate in {
        "wrong_action_dimension": lambda x: x["early_pretanh_a0_anchor"].__setitem__("action_dim", "a1"),
        "post_tanh_used_as_z0": lambda x: x["early_pretanh_a0_anchor"].__setitem__("pretanh_source", "atanh_mode_or_post_tanh"),
        "non_early_demo_rows": lambda x: x["early_pretanh_a0_anchor"].__setitem__("scope", "all_demo"),
        "online_rows": lambda x: x["early_pretanh_a0_anchor"].__setitem__("scope", "online_rows"),
        "global_anchor_lambda_changed": lambda x: x["demo_actor_anchor"].__setitem__("selected_lambda", 0.08),
        "conservative_q_changed": lambda x: x["conservative_q"].__setitem__("selected_lambda", 4.0),
    }.items():
        c = json.loads(json.dumps(cfg))
        mutate(c)
        negatives[name] = reject(c, f"negative test accepted: {name}")
    result = {
        "PRETANH_A0_DESIGN": "PASS",
        "PRETANH_SOURCE": "distribution.distribution.loc",
        "PRETANH_DIMENSION_BINDING": "PASS",
        "EARLY_APPROACH_DEMO_ROWS": int(mask.sum()),
        "A0_DEMO_RANGE": [float(actions[mask, 0].min()), float(actions[mask, 0].max())],
        "LAMBDA_PRETANH": lam,
        "SAC_ACTOR_PARAM_GRAD_NORM": sac_norm,
        "GLOBAL_ANCHOR_PARAM_GRAD_NORM": global_norm,
        "PRETANH_RAW_PARAM_GRAD_NORM": raw_norm,
        "WEIGHTED_PRETANH_GRAD_RATIO": ratio,
        "STATIC_NEGATIVE_TESTS": negatives,
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
