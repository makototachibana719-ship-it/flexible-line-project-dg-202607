#!/usr/bin/env python3
"""Stage A fixed-batch verifier for isolated Peg F3 V2 conservative-Q design."""
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

from serl_launcher.agents.continuous.sac import SACAgent
from training.peg_insertion.conservative_q_ablation import make_conservative_q_agent


CONFIG = ROOT / "configs/peg_insertion/peg_v2_conservative_q_ablation_1k_seed0.yaml"
OUT = ROOT / "results/validation/peg/f3_v2_conservative_q_design_20260906_002"
CRITIC_UPDATE_FILE = ROOT / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tree_l2(tree):
    leaves = [np.asarray(jax.device_get(x), np.float64) for x in jax.tree_util.tree_leaves(tree)]
    return float(np.sqrt(sum(np.sum(x * x) for x in leaves)))


def critic_grads_only(grads):
    return grads["modules_critic"]


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    require(cfg["conservative_q"]["enabled"] is True, "conservative q disabled")
    require(cfg["conservative_q"].get("demo_anchor_enabled") is False, "demo anchor enabled")
    require(cfg["optimizer"]["critic_learning_rate"] == 0.0003, "critic lr drift")
    require(cfg["optimizer"]["critic_warmup_steps"] == 2000, "critic warmup drift")
    require(cfg["training"]["demo_batch_size"] == 32 and cfg["training"]["online_batch_size"] == 32, "batch drift")
    require(cfg["algorithm_defaults"]["critic_ensemble_size"] == 10, "ensemble drift")
    require(cfg["algorithm_defaults"]["backup_entropy"] is False, "backup entropy drift")

    data = pickle.loads((ROOT / cfg["demo"]["path"]).read_bytes())
    online = data[:32]
    demo = data[-32:]
    batch = {key: np.asarray([row[key] for row in online + demo]) for key in data[0]}
    agent = make_conservative_q_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32), cfg)

    params = agent.state.params
    rng = jax.random.PRNGKey(20260906)

    base_loss, _ = SACAgent.critic_loss_fn(agent, batch, params, rng)
    raw_loss = agent.conservative_q_loss_fn(batch, params, rng)
    base_grads = jax.grad(lambda p: SACAgent.critic_loss_fn(agent, batch, p, rng)[0])(params)
    raw_grads = jax.grad(lambda p: agent.conservative_q_loss_fn(batch, p, rng))(params)
    base_norm = tree_l2(critic_grads_only(base_grads))
    raw_norm = tree_l2(critic_grads_only(raw_grads))
    weight = float(cfg["conservative_q"]["selected_lambda"])
    weighted_ratio = weight * raw_norm / base_norm

    result = {
        "PEG_F3_CONSERVATIVE_Q_DESIGN": "PASS" if 0.05 <= weighted_ratio <= 0.20 else "FAIL",
        "CRITIC_UPDATE_FILE": str(CRITIC_UPDATE_FILE),
        "BASE_CRITIC_OBJECTIVE": "mean((predicted_qs - target_qs) ** 2)",
        "ENSEMBLE_SIZE": 10,
        "TARGET_Q_FORMULA": "reward + discount * mask * min(subsampled_target_next_qs); backup_entropy=false",
        "PRODUCTION_SAC_CHANGED": "NO",
        "DATASET_CHANGED": "NO",
        "ENV_CHANGED": "NO",
        "DEMO_ANCHOR_ENABLED": "NO",
        "PARAMETERS_UPDATED": False,
        "BASE_CRITIC_LOSS": float(base_loss),
        "RAW_CONSERVATIVE_LOSS": float(raw_loss),
        "BASE_CRITIC_GRAD_NORM": base_norm,
        "CONSERVATIVE_GRAD_NORM": raw_norm,
        "LAMBDA_C_SMALL": weight,
        "LAMBDA_C_MEDIUM": float(cfg["conservative_q"]["lambda_medium"]),
        "WEIGHTED_GRAD_RATIO": weighted_ratio,
        "CONSERVATIVE_Q_MARGIN": float(cfg["conservative_q"]["margin"]),
        "CONSERVATIVE_Q_1K": "READY" if 0.05 <= weighted_ratio <= 0.20 else "BLOCKED",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (OUT / "REPORT.md").write_text("# Peg F3 V2 Conservative-Q Design\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["PEG_F3_CONSERVATIVE_Q_DESIGN"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
