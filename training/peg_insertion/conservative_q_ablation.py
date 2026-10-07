"""Isolated conservative-Q critic regularization for Peg F3 V2 ablation."""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from jax import nn

from serl_launcher.agents.continuous.sac import SACAgent


class ConservativeQSACAgent(SACAgent):
    """SAC with an additive Demo-state Q ranking penalty on the critic loss."""

    def conservative_q_loss_fn(self, batch, params, rng):
        cfg = self.config["conservative_q"]
        demo_batch_size = int(cfg["demo_batch_size"])
        margin = float(cfg["margin"])
        obs_demo = batch["observations"][-demo_batch_size:]
        action_demo = batch["actions"][-demo_batch_size:]
        policy_action = self.forward_policy(obs_demo, rng=rng, train=False).mode()
        policy_action = jax.lax.stop_gradient(policy_action)
        q_policy = self.forward_critic(obs_demo, policy_action, rng=rng, grad_params=params)
        q_demo = self.forward_critic(obs_demo, action_demo, rng=rng, grad_params=params)
        chex.assert_equal_shape([q_policy, q_demo])
        return jnp.mean(jax.nn.relu(q_policy - q_demo + margin))

    def critic_loss_fn(self, batch, params, rng):
        bellman_loss, info = super().critic_loss_fn(batch, params, rng)
        conservative_loss = self.conservative_q_loss_fn(batch, params, rng)
        weight = float(self.config["conservative_q"]["lambda"])
        weighted = weight * conservative_loss
        total = bellman_loss + weighted
        info = dict(info)
        info.update(base_critic_loss=bellman_loss,
                    conservative_q_loss=conservative_loss,
                    weighted_conservative_q_loss=weighted,
                    critic_loss=total)
        return total, info


def make_conservative_q_agent(seed, sample_obs, sample_action, config, *, conservative_lambda=None):
    optimizer = config["optimizer"]
    defaults = config["algorithm_defaults"]
    conservative = config.get("conservative_q", {})
    value = float(conservative["lambda_small"] if conservative_lambda is None else conservative_lambda)
    agent = ConservativeQSACAgent.create_states(
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
    return agent.replace(config={**agent.config,
                                 "conservative_q": {
                                     "lambda": value,
                                     "margin": float(conservative.get("margin", 0.0)),
                                     "demo_batch_size": int(conservative.get("demo_batch_size", 32)),
                                 }})
