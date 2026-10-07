"""Experiment-local frozen-base Obs20-gated residual policy V1."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from serl_launcher.networks.actor_critic_nets import TanhMultivariateNormalDiag

INNER_RADIUS=1.0
OUTER_RADIUS=1.00497305393219

class ResidualMLP(nn.Module):
    """Same 256x256 ReLU MLP family; bounded six-axis location residual."""
    @nn.compact
    def __call__(self, observations):
        x=nn.Dense(256)(observations);x=nn.relu(x)
        x=nn.Dense(256)(x);x=nn.relu(x)
        x=nn.Dense(6,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x)
        return jnp.tanh(x)

RESIDUAL_DEF=ResidualMLP()

def normalized_radius(observations):
    observations=jnp.asarray(observations)
    lateral=jnp.linalg.norm(observations[...,14:16],axis=-1)
    axial=jnp.abs(observations[...,13])
    return jnp.maximum(lateral/0.010,axial/0.020)

def gate_value(observations):
    radius=normalized_radius(observations)
    x=jnp.clip((radius-INNER_RADIUS)/(OUTER_RADIUS-INNER_RADIUS),0.0,1.0)
    smooth=x*x*(3.0-2.0*x)
    return jnp.where(radius<=INNER_RADIUS,jnp.zeros_like(smooth),smooth)

def residual_output(params,observations):
    return RESIDUAL_DEF.apply({'params':params},observations)

def combined_distribution(base_agent,residual_params,observations,*,rng=None,train=False):
    base=base_agent.forward_policy(observations,rng=rng,train=train)
    base_loc=jax.lax.stop_gradient(base.distribution.loc)
    base_scale=jax.lax.stop_gradient(base.distribution.scale_diag)
    delta=residual_output(residual_params,observations)
    gated=gate_value(observations)[...,None]*delta
    loc=base_loc.at[...,:6].add(gated)
    return TanhMultivariateNormalDiag(loc=loc,scale_diag=base_scale)

def combined_mode(base_agent,residual_params,observations):
    return combined_distribution(base_agent,residual_params,observations,train=False).mode()
