"""Isolated early-approach a0 support for Peg F3 V2 ablation."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np

from training.peg_insertion.combined_q_anchor_ablation import (
    CombinedQAnchorSACAgent,
    make_combined_q_anchor_agent,
)


def early_approach_mask(root: Path, config):
    provenance = json.loads((root / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, metrics = [], []
    for entry in provenance:
        rows = [json.loads(line) for line in (root / entry["episode_path"] / "transitions.jsonl").open()]
        for row in rows:
            m = row["after"]["metrics"]
            phases.append(row["phase"])
            metrics.append([m["lateral_error"], m["axial_error"], m["orientation_error"]])
    errors = np.asarray(metrics, np.float64)
    cfg = config["early_approach_support"]["predicate"]
    return (
        np.isin(np.asarray(phases), cfg["phases"])
        & (errors[:, 0] >= float(cfg["lateral_min"]))
        & (errors[:, 1] >= float(cfg["axial_min"]))
        & (errors[:, 1] <= float(cfg["axial_max"]))
        & (errors[:, 2] <= float(cfg["orientation_max"]))
    )


class EarlyApproachSupportSACAgent(CombinedQAnchorSACAgent):
    """Combined Q-anchor agent with a Demo early-approach a0 actor constraint."""

    def early_approach_support_loss_fn(self, params, rng):
        cfg = self.config["early_approach_support"]
        observations = jnp.asarray(cfg["observations"])
        target_a0 = jnp.asarray(cfg["target_a0"])
        policy_mean = self.forward_policy(observations, rng=rng, grad_params=params).mode()
        chex.assert_shape(policy_mean, (observations.shape[0], 7))
        return jnp.mean(jnp.square(policy_mean[:, 0] - target_a0))

    def policy_loss_fn(self, batch, params, rng):
        base_loss, info = super().policy_loss_fn(batch, params, rng)
        early_loss = self.early_approach_support_loss_fn(params, rng)
        weighted_early = self.config["early_approach_support"]["lambda"] * early_loss
        total = base_loss + weighted_early
        info = dict(info)
        info.update(early_approach_support_loss=early_loss,
                    weighted_early_approach_support_loss=weighted_early,
                    actor_loss=total)
        return total, info


def make_early_approach_support_agent(seed, sample_obs, sample_action, config):
    agent = make_combined_q_anchor_agent(seed, sample_obs, sample_action, config)
    root = Path(__file__).resolve().parents[2]
    rows = pickle.loads((root / config["demo"]["path"]).read_bytes())
    mask = early_approach_mask(root, config)
    observations = jnp.asarray([row["observations"] for row, keep in zip(rows, mask) if keep], dtype=jnp.float32)
    actions = jnp.asarray([row["actions"] for row, keep in zip(rows, mask) if keep], dtype=jnp.float32)
    anchor = config["early_approach_support"]
    return EarlyApproachSupportSACAgent(
        state=agent.state,
        config={**agent.config,
                "early_approach_support": {
                    "lambda": float(anchor["selected_lambda"]),
                    "observations": observations,
                    "target_a0": actions[:, 0],
                    "row_count": int(observations.shape[0]),
                }},
    )
