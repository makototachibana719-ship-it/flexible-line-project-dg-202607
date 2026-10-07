"""Isolated conservative-Q plus Demo-anchor agent for Peg F3 V2 ablation."""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from training.peg_insertion.conservative_q_ablation import (
    ConservativeQSACAgent,
    make_conservative_q_agent,
)


class CombinedQAnchorSACAgent(ConservativeQSACAgent):
    """Conservative-Q critic regularization plus Demo-tail actor anchor."""

    def demo_anchor_loss_fn(self, batch, params, rng):
        demo_batch_size = int(self.config["demo_anchor_batch_size"])
        observations = batch["observations"][-demo_batch_size:]
        actions = batch["actions"][-demo_batch_size:]
        distribution = self.forward_policy(observations, rng=rng, grad_params=params)
        policy_mean = distribution.mode()
        chex.assert_equal_shape([policy_mean, actions])
        return jnp.mean(jnp.square(policy_mean - actions))

    def policy_loss_fn(self, batch, params, rng):
        sac_loss, info = super().policy_loss_fn(batch, params, rng)
        anchor_loss = self.demo_anchor_loss_fn(batch, params, rng)
        weighted_anchor = self.config["demo_anchor_lambda"] * anchor_loss
        total = sac_loss + weighted_anchor
        info = dict(info)
        info.update(sac_actor_loss=sac_loss, demo_anchor_loss=anchor_loss,
                    weighted_demo_anchor_loss=weighted_anchor,
                    actor_loss=total)
        return total, info


def make_combined_q_anchor_agent(seed, sample_obs, sample_action, config):
    agent = make_conservative_q_agent(seed, sample_obs, sample_action, config)
    anchor = config.get("demo_actor_anchor", {})
    return CombinedQAnchorSACAgent(
        state=agent.state,
        config={**agent.config,
                "demo_anchor_lambda": float(anchor["selected_lambda"]),
                "demo_anchor_batch_size": int(anchor.get("demo_batch_size", 32))},
    )
