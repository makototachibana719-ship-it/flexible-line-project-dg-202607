"""Isolated OOD-state action anchor for Peg F3 V2 behavior-transfer ablation."""
from __future__ import annotations

import pickle
from pathlib import Path

import chex
import jax.numpy as jnp

from training.peg_insertion.combined_q_anchor_ablation import (
    CombinedQAnchorSACAgent,
    make_combined_q_anchor_agent,
)


class OODActionAnchorSACAgent(CombinedQAnchorSACAgent):
    """Combined Q-anchor agent with an online-state Demo-support action constraint."""

    def ood_action_anchor_loss_fn(self, batch, params, rng):
        cfg = self.config["ood_action_anchor"]
        online_batch_size = int(cfg["online_batch_size"])
        observations = batch["observations"][:online_batch_size]
        demo_obs = jnp.asarray(cfg["demo_observations"])
        demo_actions = jnp.asarray(cfg["demo_actions"])
        scale = jnp.asarray(cfg["observation_scale"])
        scaled_delta = (observations[:, None, :] - demo_obs[None, :, :]) / scale
        nearest = jnp.argmin(jnp.sum(jnp.square(scaled_delta), axis=-1), axis=1)
        target_actions = demo_actions[nearest]
        policy_mean = self.forward_policy(observations, rng=rng, grad_params=params).mode()
        chex.assert_equal_shape([policy_mean, target_actions])
        return jnp.mean(jnp.square(policy_mean - target_actions))

    def policy_loss_fn(self, batch, params, rng):
        base_loss, info = super().policy_loss_fn(batch, params, rng)
        ood_loss = self.ood_action_anchor_loss_fn(batch, params, rng)
        weighted_ood = self.config["ood_action_anchor"]["lambda"] * ood_loss
        total = base_loss + weighted_ood
        info = dict(info)
        info.update(ood_action_anchor_loss=ood_loss,
                    weighted_ood_action_anchor_loss=weighted_ood,
                    actor_loss=total)
        return total, info


def make_ood_action_anchor_agent(seed, sample_obs, sample_action, config):
    agent = make_combined_q_anchor_agent(seed, sample_obs, sample_action, config)
    anchor = config.get("ood_action_anchor", {})
    root = Path(__file__).resolve().parents[2]
    rows = pickle.loads((root / config["demo"]["path"]).read_bytes())
    demo_obs = jnp.asarray([row["observations"] for row in rows], dtype=jnp.float32)
    demo_actions = jnp.asarray([row["actions"] for row in rows], dtype=jnp.float32)
    std = jnp.std(demo_obs, axis=0)
    scale = jnp.maximum(std, jnp.asarray(1e-3, dtype=jnp.float32))
    return OODActionAnchorSACAgent(
        state=agent.state,
        config={**agent.config,
                "ood_action_anchor": {
                    "lambda": float(anchor["selected_lambda"]),
                    "online_batch_size": int(anchor.get("online_batch_size", 32)),
                    "demo_observations": demo_obs,
                    "demo_actions": demo_actions,
                    "observation_scale": scale,
                }},
    )
