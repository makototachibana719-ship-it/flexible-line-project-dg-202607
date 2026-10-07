"""Isolated early-approach pre-tanh a0 anchor for Peg F3 V2 ablation."""
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


def early_pretanh_mask(root: Path, config):
    provenance = json.loads((root / "results/datasets/peg_v2_accepted_11_20260906/episode_provenance.json").read_text())
    phases, metrics = [], []
    for entry in provenance:
        rows = [json.loads(line) for line in (root / entry["episode_path"] / "transitions.jsonl").open()]
        for row in rows:
            m = row["after"]["metrics"]
            phases.append(row["phase"])
            metrics.append([m["lateral_error"], m["axial_error"], m["orientation_error"]])
    errors = np.asarray(metrics, np.float64)
    cfg = config["early_pretanh_a0_anchor"]["predicate"]
    return (
        np.isin(np.asarray(phases), cfg["phases"])
        & (errors[:, 0] >= float(cfg["lateral_min"]))
        & (errors[:, 1] >= float(cfg["axial_min"]))
        & (errors[:, 1] <= float(cfg["axial_max"]))
        & (errors[:, 2] <= float(cfg["orientation_max"]))
    )


class EarlyPretanhA0AnchorSACAgent(CombinedQAnchorSACAgent):
    """Combined Q-anchor agent with a Demo early-approach pre-tanh a0 constraint."""

    def early_pretanh_a0_anchor_loss_fn(self, params, rng):
        cfg = self.config["early_pretanh_a0_anchor"]
        observations = jnp.asarray(cfg["observations"])
        distribution = self.forward_policy(observations, rng=rng, grad_params=params)
        loc = distribution.distribution.loc
        chex.assert_shape(loc, (observations.shape[0], 7))
        return jnp.mean(jnp.square(loc[:, 0]))

    def policy_loss_fn(self, batch, params, rng):
        base_loss, info = super().policy_loss_fn(batch, params, rng)
        pretanh_loss = self.early_pretanh_a0_anchor_loss_fn(params, rng)
        weighted_pretanh = self.config["early_pretanh_a0_anchor"]["lambda"] * pretanh_loss
        total = base_loss + weighted_pretanh
        info = dict(info)
        info.update(early_pretanh_a0_anchor_loss=pretanh_loss,
                    weighted_early_pretanh_a0_anchor_loss=weighted_pretanh,
                    actor_loss=total)
        return total, info


def make_early_pretanh_a0_anchor_agent(seed, sample_obs, sample_action, config):
    agent = make_combined_q_anchor_agent(seed, sample_obs, sample_action, config)
    root = Path(__file__).resolve().parents[2]
    rows = pickle.loads((root / config["demo"]["path"]).read_bytes())
    mask = early_pretanh_mask(root, config)
    observations = jnp.asarray([row["observations"] for row, keep in zip(rows, mask) if keep], dtype=jnp.float32)
    anchor = config["early_pretanh_a0_anchor"]
    return EarlyPretanhA0AnchorSACAgent(
        state=agent.state,
        config={**agent.config,
                "early_pretanh_a0_anchor": {
                    "lambda": float(anchor["selected_lambda"]),
                    "observations": observations,
                    "row_count": int(observations.shape[0]),
                }},
    )
