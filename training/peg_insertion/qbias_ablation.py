"""Isolated config binding for the Peg F3 V2 Q-bias ablation.

This module does not change the production factory.  It reproduces that factory's
architecture and algorithm defaults while allowing only critic LR/warmup overrides.
"""
from __future__ import annotations

import jax
from jax import nn

from serl_launcher.agents.continuous.sac import SACAgent


def make_qbias_ablation_agent(seed, sample_obs, sample_action, config):
    optimizer = config["optimizer"]
    defaults = config["algorithm_defaults"]
    return SACAgent.create_states(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        actor_optimizer_kwargs={
            "learning_rate": optimizer["actor_learning_rate"],
            "warmup_steps": optimizer["actor_warmup_steps"],
        },
        critic_optimizer_kwargs={
            "learning_rate": optimizer["critic_learning_rate"],
            "warmup_steps": optimizer["critic_warmup_steps"],
        },
        temperature_optimizer_kwargs={
            "learning_rate": optimizer["temperature_learning_rate"],
        },
        temperature_init=1e-2,
        discount=defaults["discount"],
        soft_target_update_rate=defaults["soft_target_update_rate"],
        target_entropy=defaults["target_entropy"],
        backup_entropy=defaults["backup_entropy"],
        critic_ensemble_size=defaults["critic_ensemble_size"],
        critic_subsample_size=defaults["critic_subsample_size"],
    )
