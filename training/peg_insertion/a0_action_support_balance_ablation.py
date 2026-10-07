"""Actor-objective-balance candidate plus isolated a0 Demo-support regularizer."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np

from training.peg_insertion.actor_objective_balance_ablation import (
    ActorObjectiveBalanceSACAgent,
    actor_balance_mask,
    make_actor_objective_balance_agent,
)


def a0_support_mask(root: Path, config):
    return actor_balance_mask(root, config)


class A0ActionSupportBalanceSACAgent(ActorObjectiveBalanceSACAgent):
    def a0_action_support_loss_fn(self, params, rng):
        cfg = self.config["a0_action_support"]
        observations = jnp.asarray(cfg["observations"])
        lower = jnp.asarray(cfg["lower"])
        upper = jnp.asarray(cfg["upper"])
        policy_a0 = self.forward_policy(observations, rng=rng, grad_params=params).mode()[:, 0]
        chex.assert_shape(policy_a0, (observations.shape[0],))
        below = jnp.maximum(lower - policy_a0, 0.0)
        above = jnp.maximum(policy_a0 - upper, 0.0)
        return jnp.mean(jnp.square(below) + jnp.square(above))

    def policy_loss_fn(self, batch, params, rng):
        base_loss, info = super().policy_loss_fn(batch, params, rng)
        support_loss = self.a0_action_support_loss_fn(params, rng)
        weighted_support = self.config["a0_action_support"]["lambda"] * support_loss
        total = base_loss + weighted_support
        info = dict(info)
        info.update(
            a0_action_support_loss=support_loss,
            weighted_a0_action_support_loss=weighted_support,
            actor_loss=total,
        )
        return total, info


def make_a0_action_support_balance_agent(seed, sample_obs, sample_action, config):
    agent = make_actor_objective_balance_agent(seed, sample_obs, sample_action, config)
    root = Path(__file__).resolve().parents[2]
    rows = pickle.loads((root / config["demo"]["path"]).read_bytes())
    mask = a0_support_mask(root, config)
    observations = jnp.asarray([row["observations"] for row, keep in zip(rows, mask) if keep], dtype=jnp.float32)
    actions = np.asarray([row["actions"] for row, keep in zip(rows, mask) if keep], dtype=np.float32)
    support = config["a0_action_support"]
    support_min = float(actions[:, 0].min())
    support_max = float(actions[:, 0].max())
    return A0ActionSupportBalanceSACAgent(
        state=agent.state,
        config={
            **agent.config,
            "a0_action_support": {
                "lambda": float(support["selected_lambda"]),
                "observations": observations,
                "lower": support_min,
                "upper": support_max,
                "row_count": int(observations.shape[0]),
                "range": [support_min, support_max],
            },
        },
    )
