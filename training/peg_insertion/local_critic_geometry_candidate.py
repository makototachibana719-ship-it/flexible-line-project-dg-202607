"""Experiment-local Demo-centered critic geometry correction."""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from training.peg_insertion.combined_q_anchor_ablation import (
    CombinedQAnchorSACAgent,
    make_combined_q_anchor_agent,
)


class LocalGeometryCombinedAgent(CombinedQAnchorSACAgent):
    """C+G with one auxiliary, Demo-only local critic ranking term."""

    def local_critic_comparisons(self, batch, params, rng):
        cfg = self.config["local_critic_geometry"]
        count = int(cfg["demo_batch_size"])
        obs = batch["observations"][-count:]
        demo = batch["actions"][-count:]
        scales = jnp.asarray(cfg["perturbation_scales"], dtype=demo.dtype)
        radii = jnp.minimum(scales[None, :], 1.0 - jnp.abs(demo[:, :6]))
        basis = jnp.concatenate(
            [jnp.eye(6, dtype=demo.dtype), jnp.zeros((6, 1), dtype=demo.dtype)], axis=1
        )
        signs = jnp.asarray([-1.0, 1.0], dtype=demo.dtype)
        delta = signs[None, :, None, None] * radii[:, None, :, None] * basis[None, None, :, :]
        local = demo[:, None, None, :] + delta
        local_obs = jnp.broadcast_to(obs[:, None, None, :], (count, 2, 6, obs.shape[-1]))
        q_local = self.forward_critic(
            local_obs.reshape((-1, obs.shape[-1])),
            local.reshape((-1, demo.shape[-1])),
            rng=rng,
            grad_params=params,
        ).reshape((self.config["critic_ensemble_size"], count, 2, 6))
        q_demo = self.forward_critic(obs, demo, rng=rng, grad_params=params)
        chex.assert_shape(q_demo, (self.config["critic_ensemble_size"], count))
        valid = radii > 1e-6
        gaps = q_local - q_demo[:, :, None, None] + float(cfg["margin"])
        return gaps, valid

    def local_critic_loss_fn(self, batch, params, rng):
        gaps, valid = self.local_critic_comparisons(batch, params, rng)
        violations = jax.nn.relu(gaps)
        mask = valid[None, :, None, :]
        denominator = self.config["critic_ensemble_size"] * 2 * jnp.maximum(jnp.sum(valid), 1)
        return jnp.sum(jnp.where(mask, violations, 0.0)) / denominator

    def critic_loss_fn(self, batch, params, rng):
        base_loss, info = super().critic_loss_fn(batch, params, rng)
        local_loss = self.local_critic_loss_fn(batch, params, rng)
        weighted = float(self.config["local_critic_geometry"]["lambda"]) * local_loss
        total = base_loss + weighted
        info = dict(info)
        info.update(
            critic_loss=total,
            base_cg_critic_loss=base_loss,
            local_critic_geometry_loss=local_loss,
            weighted_local_critic_geometry_loss=weighted,
        )
        return total, info


def make_local_geometry_agent(seed, sample_obs, sample_action, config):
    agent = make_combined_q_anchor_agent(seed, sample_obs, sample_action, config)
    local = config["local_critic_geometry"]
    return LocalGeometryCombinedAgent(
        state=agent.state,
        config={
            **agent.config,
            "local_critic_geometry": {
                "lambda": float(local["selected_lambda"]),
                "margin": float(local["margin"]),
                "demo_batch_size": int(local.get("demo_batch_size", 32)),
                "perturbation_scales": tuple(float(x) for x in local["perturbation_scales"]),
            },
        },
    )
