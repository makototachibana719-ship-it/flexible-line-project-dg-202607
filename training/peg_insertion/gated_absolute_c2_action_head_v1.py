"""Experiment-local gated absolute C2 action head V1.

The C0 base actor and confirmed C1 residual define the frozen lower policy.
The new trainable head predicts an unconstrained *head-logit initialization
offset*.  Its legal absolute C2 action is obtained with the ordinary tanh,
then convexly mixed with the legal lower action by the existing g2 gate.

No physical action range, controller scaling, or gate threshold is changed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from training.peg_insertion.observation_gated_residual_v1 import (
    gate_value,
    residual_output,
)
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate


class AbsoluteC2Head(nn.Module):
    """Obs20 Dense256/ReLU head with an exactly zero initial logit offset."""

    @nn.compact
    def __call__(self, observations):
        x = nn.Dense(256)(observations)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return nn.Dense(
            6,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)


ABSOLUTE_C2_HEAD_DEF = AbsoluteC2Head()


def lower_policy_components(base_agent, c1_params, observations):
    """Return frozen lower pre-tanh location and deterministic legal action."""
    obs = jnp.asarray(observations)
    base = base_agent.forward_policy(obs, rng=None, train=False)
    base_loc = jax.lax.stop_gradient(base.distribution.loc)
    c1 = jax.lax.stop_gradient(residual_output(c1_params, obs))
    lower_loc = base_loc.at[..., :6].add(gate_value(obs)[..., None] * c1)
    return lower_loc, jnp.tanh(lower_loc)


def absolute_c2_head_action(head_params, observations, lower_loc):
    """Legal normalized absolute C2 action; zero params reproduce lower action."""
    delta_logits = ABSOLUTE_C2_HEAD_DEF.apply({"params": head_params}, observations)
    return jnp.tanh(jax.lax.stop_gradient(lower_loc[..., :6]) + delta_logits)


def gated_absolute_compose(lower_action, head_action, g2):
    """Convexly compose legal normalized actions without a residual scale."""
    mixed_first6 = (1.0 - g2[..., None]) * lower_action[..., :6] + g2[..., None] * head_action
    return lower_action.at[..., :6].set(mixed_first6)


def gated_absolute_mode(
    base_agent,
    c1_params,
    c2_head_params,
    observations,
    support_observations,
    normalization_scale,
    core_threshold,
    shell_outer_threshold,
):
    """Deterministic P2 policy using only runtime Obs20 and the existing gate."""
    obs = jnp.asarray(observations)
    lower_loc, lower_action = lower_policy_components(base_agent, c1_params, obs)
    head_action = absolute_c2_head_action(c2_head_params, obs, lower_loc)
    g2 = progressive_gate(
        obs,
        support_observations,
        normalization_scale,
        core_threshold,
        shell_outer_threshold,
    )
    return gated_absolute_compose(lower_action, head_action, g2)
