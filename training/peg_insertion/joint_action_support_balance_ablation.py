"""A0-support balance candidate plus one phase-selective joint-action support term."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np

from training.peg_insertion.a0_action_support_balance_ablation import (
    A0ActionSupportBalanceSACAgent,
    a0_support_mask,
    make_a0_action_support_balance_agent,
)


def joint_support_mask(root: Path, config):
    provenance = json.loads((root / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases = []
    for entry in provenance:
        for line in (root / entry["episode_path"] / "transitions.jsonl").open():
            phases.append(json.loads(line)["phase"])
    allowed = set(config["joint_action_support"]["phases"])
    base_mask = a0_support_mask(root, config)
    phase_mask = np.asarray(
        [phase in allowed or (phase == "lateral_alignment" and "lateral_correction" in allowed) for phase in phases],
        dtype=bool,
    )
    return base_mask & phase_mask


class JointActionSupportBalanceSACAgent(A0ActionSupportBalanceSACAgent):
    def joint_action_support_loss_fn(self, params, rng):
        cfg = self.config["joint_action_support"]
        observations = jnp.asarray(cfg["observations"])
        actions = jnp.asarray(cfg["actions"])
        policy = self.forward_policy(observations, rng=rng, grad_params=params).mode()
        chex.assert_shape(policy, (observations.shape[0], 7))
        return jnp.mean(jnp.square(policy[:, :6] - actions[:, :6]))

    def policy_loss_fn(self, batch, params, rng):
        base_loss, info = super().policy_loss_fn(batch, params, rng)
        joint_loss = self.joint_action_support_loss_fn(params, rng)
        weighted_joint = self.config["joint_action_support"]["lambda"] * joint_loss
        total = base_loss + weighted_joint
        info = dict(info)
        info.update(
            joint_action_support_loss=joint_loss,
            weighted_joint_action_support_loss=weighted_joint,
            actor_loss=total,
        )
        return total, info


def make_joint_action_support_balance_agent(seed, sample_obs, sample_action, config):
    agent = make_a0_action_support_balance_agent(seed, sample_obs, sample_action, config)
    root = Path(__file__).resolve().parents[2]
    rows = pickle.loads((root / config["demo"]["path"]).read_bytes())
    mask = joint_support_mask(root, config)
    observations = jnp.asarray([row["observations"] for row, keep in zip(rows, mask) if bool(keep)], dtype=jnp.float32)
    actions = jnp.asarray([row["actions"] for row, keep in zip(rows, mask) if bool(keep)], dtype=jnp.float32)
    support = config["joint_action_support"]
    return JointActionSupportBalanceSACAgent(
        state=agent.state,
        config={
            **agent.config,
            "joint_action_support": {
                "lambda": float(support["selected_lambda"]),
                "observations": observations,
                "actions": actions,
                "row_count": int(observations.shape[0]),
                "dims": "a0:a5",
                "a6_excluded": True,
            },
        },
    )
