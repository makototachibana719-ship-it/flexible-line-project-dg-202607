"""Isolated actor Demo-action anchor for a bounded Peg F3 ablation."""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jax import nn

from serl_launcher.agents.continuous.sac import SACAgent


class DemoAnchoredSACAgent(SACAgent):
    """SAC with an additive BC auxiliary term on the Demo tail of an RLPD batch."""

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


def make_demo_anchor_agent(seed, sample_obs, sample_action, config, *, anchor_lambda=None):
    optimizer = config["optimizer"]
    defaults = config["algorithm_defaults"]
    anchor = config.get("demo_actor_anchor", {})
    value = float(anchor["lambda_small"] if anchor_lambda is None else anchor_lambda)
    agent = DemoAnchoredSACAgent.create_states(
        jax.random.PRNGKey(seed), sample_obs, sample_action,
        policy_kwargs={"tanh_squash_distribution": True, "std_parameterization": "exp",
                       "std_min": 1e-5, "std_max": 5},
        critic_network_kwargs={"activations": nn.tanh, "use_layer_norm": True,
                               "hidden_dims": [256, 256]},
        policy_network_kwargs={"activations": nn.tanh, "use_layer_norm": True,
                               "hidden_dims": [256, 256]},
        actor_optimizer_kwargs={"learning_rate": optimizer["actor_learning_rate"],
                                "warmup_steps": optimizer["actor_warmup_steps"]},
        critic_optimizer_kwargs={"learning_rate": optimizer["critic_learning_rate"],
                                 "warmup_steps": optimizer["critic_warmup_steps"]},
        temperature_optimizer_kwargs={"learning_rate": optimizer["temperature_learning_rate"]},
        temperature_init=1e-2, discount=defaults["discount"],
        soft_target_update_rate=defaults["soft_target_update_rate"],
        target_entropy=defaults["target_entropy"], backup_entropy=defaults["backup_entropy"],
        critic_ensemble_size=defaults["critic_ensemble_size"],
        critic_subsample_size=defaults["critic_subsample_size"],
    )
    return agent.replace(config={**agent.config, "demo_anchor_lambda": value,
                                 "demo_anchor_batch_size": int(anchor.get("demo_batch_size", 32))})
