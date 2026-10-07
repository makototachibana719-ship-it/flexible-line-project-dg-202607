"""Experiment-local observable recovery option for the frozen C2 stack.

The finite-state controller uses only runtime Obs20 geometry.  It does not
change any learned parameter, action scaling, protected threshold, or task
definition.
"""
from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np

from training.peg_insertion.c2_trajectory_recovery_head_v1 import (
    TRAJECTORY_RECOVERY_HEAD_DEF,
    compose_recovery_action,
    recovery_distance,
)
from training.peg_insertion.gated_absolute_c2_action_head_v1 import (
    lower_policy_components,
)
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import (
    progressive_gate,
    support_distance,
)


class RecoveryPhase(IntEnum):
    APPROACH = 0
    RECOVERY = 1
    LOWER_HANDOFF = 2


def lower_action(agent, c1_params, observation):
    """Return the deterministic frozen C0+C1 action."""
    _, action = lower_policy_components(agent, c1_params, jnp.asarray(observation))
    return np.asarray(jax.device_get(action), np.float32)


def recovery_option_step(
    phase,
    observation,
    *,
    global_action_fn,
    local_params,
    recovery_support,
    recovery_scale,
    recovery_outer,
    agent,
    c1_params,
    c1_support,
    c1_scale,
    c1_core,
    c1_outer,
):
    """Advance the observable FSM and return action plus diagnostics.

    Initiation: current Obs20 is inside the learned recovery support and the
    existing C2 gate is active.  Termination: current Obs20 enters the frozen
    C1 recovery shell or the existing C0 protection gate becomes zero.
    """
    obs = np.asarray(observation, np.float32)
    d_rec = float(np.asarray(jax.device_get(
        recovery_distance(obs, recovery_support, recovery_scale))))
    d_c1 = float(np.asarray(jax.device_get(
        support_distance(obs, c1_support, c1_scale))))
    g0 = float(np.asarray(jax.device_get(gate_value(obs))))
    g2 = float(np.asarray(jax.device_get(
        progressive_gate(obs, c1_support, c1_scale, c1_core, c1_outer))))

    next_phase = RecoveryPhase(phase)
    transition = None
    if next_phase == RecoveryPhase.APPROACH and d_rec <= recovery_outer and g2 > 0.0:
        next_phase = RecoveryPhase.RECOVERY
        transition = "APPROACH_TO_RECOVERY"
    if next_phase == RecoveryPhase.RECOVERY and (d_c1 <= c1_outer or g0 == 0.0):
        next_phase = RecoveryPhase.LOWER_HANDOFF
        transition = "RECOVERY_TO_LOWER_HANDOFF"

    if next_phase == RecoveryPhase.LOWER_HANDOFF:
        action = lower_action(agent, c1_params, obs)
    else:
        global_action = global_action_fn(obs)
        if next_phase == RecoveryPhase.RECOVERY:
            # Force only the local-support latch.  The pre-existing g2 still
            # gives exact zero recovery authority in the protected C1 region.
            action, _ = compose_recovery_action(
                TRAJECTORY_RECOVERY_HEAD_DEF,
                local_params,
                jnp.asarray(obs),
                jnp.asarray(global_action),
                recovery_support,
                recovery_scale,
                1.0e9,
                1.0e9 + 1.0,
                c1_support,
                c1_scale,
                c1_core,
                c1_outer,
            )
            action = np.asarray(jax.device_get(action), np.float32)
        else:
            action = np.asarray(global_action, np.float32)

    info = {
        "phase": RecoveryPhase(next_phase).name,
        "transition": transition,
        "D_REC": d_rec,
        "D_C1": d_c1,
        "g0": g0,
        "g2": g2,
    }
    return next_phase, action, info
