"""Progressive frozen residual V2 with Obs20 nearest-support recovery shells."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from serl_launcher.networks.actor_critic_nets import TanhMultivariateNormalDiag
from training.peg_insertion.observation_gated_residual_v1 import gate_value,residual_output

NUMERICAL_EPS=1e-8

def support_distance(observations,support_observations,normalization_scale):
    obs=jnp.asarray(observations)
    support=jnp.asarray(support_observations)
    scale=jnp.asarray(normalization_scale)
    flat=obs.reshape((-1,obs.shape[-1]))
    distances=jnp.sqrt(jnp.sum(jnp.square((flat[:,None,:]-support[None,:,:])/scale),axis=-1))
    result=jnp.min(distances,axis=-1)
    return result.reshape(obs.shape[:-1])

def recovery_gate(observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold):
    distance=support_distance(observations,support_observations,normalization_scale)
    x=jnp.clip((distance-core_threshold)/(shell_outer_threshold-core_threshold),0.0,1.0)
    smooth=x*x*(3.0-2.0*x)
    return jnp.where(distance<=core_threshold,jnp.zeros_like(smooth),jnp.where(distance>=shell_outer_threshold,jnp.ones_like(smooth),smooth))

def progressive_gate(observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold):
    """Gate C2 outside C1's shell while preserving every lower protected core."""
    return gate_value(observations)*recovery_gate(
        observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold
    )

def progressive_distribution(base_agent,c1_params,c2_params,observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold,*,rng=None,train=False):
    base=base_agent.forward_policy(observations,rng=rng,train=train)
    base_loc=jax.lax.stop_gradient(base.distribution.loc)
    base_scale=jax.lax.stop_gradient(base.distribution.scale_diag)
    c1=jax.lax.stop_gradient(residual_output(c1_params,observations))
    c2=residual_output(c2_params,observations)
    g2=progressive_gate(observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold)
    loc=base_loc.at[...,:6].add(gate_value(observations)[...,None]*c1+g2[...,None]*c2)
    return TanhMultivariateNormalDiag(loc=loc,scale_diag=base_scale)

def progressive_mode(base_agent,c1_params,c2_params,observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold):
    return progressive_distribution(base_agent,c1_params,c2_params,observations,support_observations,normalization_scale,core_threshold,shell_outer_threshold,train=False).mode()
