"""Experiment-local Obs20 recovery head layered over a frozen global C2 policy."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate


class TrajectoryRecoveryHead(nn.Module):
    @nn.compact
    def __call__(self, observations):
        x = nn.relu(nn.Dense(256)(observations))
        x = nn.relu(nn.Dense(256)(x))
        return nn.Dense(6, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)


TRAJECTORY_RECOVERY_HEAD_DEF = TrajectoryRecoveryHead()


def recovery_distance(observations, recovery_support, normalization_scale):
    obs = jnp.asarray(observations)
    support = jnp.asarray(recovery_support)
    scale = jnp.asarray(normalization_scale)
    flat = obs.reshape((-1, obs.shape[-1]))
    delta = (flat[:, None, :] - support[None, :, :]) / scale[None, None, :]
    distance = jnp.sqrt(jnp.min(jnp.sum(jnp.square(delta), axis=-1), axis=1))
    return distance.reshape(obs.shape[:-1])


def recovery_gate(observations, recovery_support, normalization_scale, core_threshold, outer_threshold):
    distance = recovery_distance(observations, recovery_support, normalization_scale)
    x = jnp.clip((distance - core_threshold) / (outer_threshold - core_threshold), 0.0, 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)


def compose_recovery_action(
    definition,
    params,
    observations,
    global_actions,
    recovery_support,
    recovery_scale,
    recovery_core,
    recovery_outer,
    c1_support,
    c1_scale,
    c1_core,
    c1_outer,
):
    obs = jnp.asarray(observations)
    frozen_global = jax.lax.stop_gradient(jnp.asarray(global_actions))
    logits = jnp.arctanh(jnp.clip(frozen_global[..., :6], -0.999999, 0.999999))
    local_action = jnp.tanh(logits + definition.apply({"params": params}, obs))
    local_gate = recovery_gate(obs, recovery_support, recovery_scale, recovery_core, recovery_outer)
    c2_gate = progressive_gate(obs, c1_support, c1_scale, c1_core, c1_outer)
    gate = local_gate * c2_gate
    first6 = (1.0 - gate[..., None]) * frozen_global[..., :6] + gate[..., None] * local_action
    return frozen_global.at[..., :6].set(first6), gate
