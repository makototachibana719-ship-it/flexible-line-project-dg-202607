"""Obs20-only empirical one-step planner inside the frozen C2 recovery option."""
from __future__ import annotations

import jax
import numpy as np

from training.peg_insertion.c2_observable_recovery_option_v1 import (
    RecoveryPhase, lower_action,
)
from training.peg_insertion.c2_trajectory_recovery_head_v1 import recovery_distance
from training.peg_insertion.observation_gated_residual_v1 import gate_value
from training.peg_insertion.progressive_frozen_residual_v2 import progressive_gate, support_distance


class EmpiricalOneStepPlanner:
    def __init__(self, observations, actions, next_observations, obs_scale, c1_support, c1_scale, candidate_count=64):
        self.observations = np.asarray(observations, np.float32)
        self.actions = np.asarray(actions, np.float32)
        self.deltas = np.asarray(next_observations, np.float32) - self.observations
        self.obs_scale = np.asarray(obs_scale, np.float32)
        self.c1_support = np.asarray(c1_support, np.float32)
        self.c1_scale = np.asarray(c1_scale, np.float32)
        self.candidate_count = int(candidate_count)

    def action(self, observation):
        obs = np.asarray(observation, np.float32)
        obs_distance = np.sum(np.square((self.observations - obs) / self.obs_scale), axis=1)
        candidates = np.argsort(obs_distance)[:self.candidate_count]
        reports = []
        for source_row in candidates:
            action_distance = np.sum(np.square(self.actions[:, :6] - self.actions[source_row, :6]), axis=1)
            transition_row = int(np.argmin(obs_distance + action_distance))
            predicted = obs + self.deltas[transition_row]
            d_c1 = float(np.min(np.sqrt(np.sum(np.square((self.c1_support - predicted) / self.c1_scale), axis=1))))
            reports.append((d_c1, int(source_row), transition_row))
        predicted_d_c1, source_row, transition_row = min(reports)
        return self.actions[source_row].copy(), {"selected_source_row": source_row,
            "selected_transition_row": transition_row, "predicted_D_C1": predicted_d_c1}


def empirical_option_step(phase, observation, *, planner, global_action_fn, recovery_support,
    recovery_scale, recovery_outer, agent, c1_params, c1_support, c1_scale, c1_core, c1_outer):
    obs = np.asarray(observation, np.float32)
    d_rec = float(np.asarray(jax.device_get(recovery_distance(obs, recovery_support, recovery_scale))))
    d_c1 = float(np.asarray(jax.device_get(support_distance(obs, c1_support, c1_scale))))
    g0 = float(np.asarray(jax.device_get(gate_value(obs))))
    g2 = float(np.asarray(jax.device_get(progressive_gate(obs, c1_support, c1_scale, c1_core, c1_outer))))
    phase = RecoveryPhase(phase); transition = None; planner_info = None
    if phase == RecoveryPhase.APPROACH and d_rec <= recovery_outer and g2 > 0.0:
        phase = RecoveryPhase.RECOVERY; transition = "APPROACH_TO_RECOVERY"
    if phase == RecoveryPhase.RECOVERY and (d_c1 <= c1_outer or g0 == 0.0):
        phase = RecoveryPhase.LOWER_HANDOFF; transition = "RECOVERY_TO_LOWER_HANDOFF"
    if phase == RecoveryPhase.LOWER_HANDOFF:
        action = lower_action(agent, c1_params, obs)
    elif phase == RecoveryPhase.RECOVERY:
        action, planner_info = planner.action(obs)
    else:
        action = np.asarray(global_action_fn(obs), np.float32)
    return phase, action, {"phase": phase.name, "transition": transition, "D_REC": d_rec,
        "D_C1": d_c1, "g0": g0, "g2": g2, "planner": planner_info}
