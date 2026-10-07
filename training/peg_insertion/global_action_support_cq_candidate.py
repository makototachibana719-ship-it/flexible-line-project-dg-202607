"""Experiment-local global action-support conservative critic candidate."""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from training.peg_insertion.combined_q_anchor_ablation import (
    CombinedQAnchorSACAgent,
    make_combined_q_anchor_agent,
)


class GlobalActionSupportCQAgent(CombinedQAnchorSACAgent):
    """C+G plus one Demo-state global counterfactual critic term."""

    def global_cq_comparisons(self, batch, params, rng):
        cfg = self.config["global_action_support_cq"]
        count = int(cfg["demo_batch_size"])
        obs = batch["observations"][-count:]
        demo = batch["actions"][-count:]
        policy = self.forward_policy(obs, rng=rng, train=False).mode()
        policy = jax.lax.stop_gradient(policy)[:, None, :]
        basis = jnp.concatenate(
            [jnp.eye(6, dtype=demo.dtype), jnp.zeros((6, 1), dtype=demo.dtype)], axis=1
        )

        def probes(values):
            values = jnp.asarray(values, dtype=demo.dtype)
            deltas = values[None, :, None] - demo[:, None, :6]
            actions = demo[:, None, None, :] + deltas[:, :, :, None] * basis[None, None, :, :]
            return actions.reshape((count, -1, demo.shape[-1]))

        boundary = probes(cfg["boundary_values"])
        interior = probes(cfg["interior_values"])
        counter = jnp.concatenate([policy, boundary, interior], axis=1)
        counter_obs = jnp.broadcast_to(obs[:, None, :], (count, counter.shape[1], obs.shape[-1]))
        q_counter = self.forward_critic(
            counter_obs.reshape((-1, obs.shape[-1])),
            counter.reshape((-1, demo.shape[-1])),
            rng=rng,
            grad_params=params,
        ).reshape((self.config["critic_ensemble_size"], count, counter.shape[1]))
        q_demo = self.forward_critic(obs, demo, rng=rng, grad_params=params)
        chex.assert_shape(q_demo, (self.config["critic_ensemble_size"], count))
        return q_counter - q_demo[:, :, None] + float(cfg["margin"])

    def global_cq_loss_fn(self, batch, params, rng):
        return jnp.mean(jax.nn.relu(self.global_cq_comparisons(batch, params, rng)))

    def critic_loss_fn(self, batch, params, rng):
        base_loss, info = super().critic_loss_fn(batch, params, rng)
        global_loss = self.global_cq_loss_fn(batch, params, rng)
        weighted = float(self.config["global_action_support_cq"]["lambda"]) * global_loss
        total = base_loss + weighted
        info = dict(info)
        info.update(
            critic_loss=total,
            base_cg_critic_loss=base_loss,
            global_action_support_cq_loss=global_loss,
            weighted_global_action_support_cq_loss=weighted,
        )
        return total, info


def make_global_action_support_cq_agent(seed, sample_obs, sample_action, config):
    agent = make_combined_q_anchor_agent(seed, sample_obs, sample_action, config)
    global_cq = config["global_action_support_cq"]
    return GlobalActionSupportCQAgent(
        state=agent.state,
        config={
            **agent.config,
            "global_action_support_cq": {
                "lambda": float(global_cq["selected_lambda"]),
                "margin": float(global_cq["margin"]),
                "demo_batch_size": int(global_cq.get("demo_batch_size", 32)),
                "boundary_values": tuple(float(x) for x in global_cq["boundary_values"]),
                "interior_values": tuple(float(x) for x in global_cq["interior_values"]),
            },
        },
    )
