#!/usr/bin/env python3
"""Read-only loss/gradient scale measurement for the actor Demo anchor design."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.utils.launcher import make_sac_agent
from training.peg_insertion.checkpoint import restore_fresh_state

DATASET = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
CHECKPOINTS = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run/checkpoints"
UPDATES = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run/updates.jsonl"


def norm(tree):
    return float(np.sqrt(sum(np.sum(np.square(np.asarray(jax.device_get(x)), dtype=np.float64))
                             for x in jax.tree_util.tree_leaves(tree))))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() or ROOT not in out.parents:
        raise RuntimeError("fresh project output required")
    demo = pickle.loads(DATASET.read_bytes())
    demo_arrays = {key: np.asarray([row[key] for row in demo]) for key in demo[0]}
    payload = pickle.loads((CHECKPOINTS / "checkpoint_1000_online_replay.pkl").read_bytes())
    size = int(payload["size"])
    online = {key: np.asarray(value)[:size] for key, value in payload["dataset_dict"].items()}
    rng_np = np.random.default_rng(20260906)
    online_idx = rng_np.choice(size, 32, replace=False)
    demo_idx = rng_np.choice(len(demo), 32, replace=False)
    keys = ("observations", "actions", "next_observations", "rewards", "masks", "dones")
    batch = {key: jnp.asarray(np.concatenate([online[key][online_idx], demo_arrays[key][demo_idx]], axis=0))
             for key in keys}
    fresh = make_sac_agent(0, np.zeros(20, np.float32), np.zeros(7, np.float32))
    agent = fresh.replace(state=restore_fresh_state(CHECKPOINTS, fresh.state, 1000))
    key = jax.random.PRNGKey(20260906)

    def sac_fn(params):
        return agent.policy_loss_fn(batch, params, key)[0]

    def anchor_fn(params):
        dist = agent.forward_policy(batch["observations"][-32:], rng=key, grad_params=params)
        return jnp.mean(jnp.square(dist.mode() - batch["actions"][-32:]))

    sac_value, sac_grad = jax.value_and_grad(sac_fn)(agent.state.params)
    anchor_value, anchor_grad = jax.value_and_grad(anchor_fn)(agent.state.params)
    sac_norm, anchor_norm = norm(sac_grad), norm(anchor_grad)
    if not all(np.isfinite(x) and x > 0 for x in (sac_norm, anchor_norm)):
        raise RuntimeError("invalid gradient scale")
    small = 0.10 * sac_norm / anchor_norm
    medium = 0.20 * sac_norm / anchor_norm
    actor_losses = []
    for line in UPDATES.read_text().splitlines()[:1000]:
        row = json.loads(line)
        actor_losses += [abs(float(v)) for key_name, v in row["losses"].items()
                         if "actor_loss" in key_name and "sac_actor_loss" not in key_name]
    result = {
        "SAMPLE_CONTRACT": "fixed_seed_32_online_plus_32_demo_checkpoint_1000",
        "ACTOR_LOSS_TYPICAL_MAGNITUDE": float(np.median(actor_losses)),
        "SAC_ACTOR_LOSS": float(sac_value), "DEMO_ANCHOR_RAW_MAGNITUDE": float(anchor_value),
        "SAC_GRAD_NORM": sac_norm, "ANCHOR_GRAD_NORM": anchor_norm,
        "LAMBDA_SMALL_EXACT": small, "LAMBDA_MEDIUM_EXACT": medium,
        "TARGET_SMALL_GRAD_RATIO": 0.10, "TARGET_MEDIUM_GRAD_RATIO": 0.20,
        "PARAMETERS_UPDATED": False,
    }
    out.mkdir(parents=True)
    (out / "scale_measurement.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
