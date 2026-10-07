#!/usr/bin/env python3
"""Static and no-update single-batch verification for the Demo anchor design."""
from __future__ import annotations

import argparse
import hashlib
import inspect
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
import yaml

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_agent
from training.peg_insertion.actor_demo_anchor_ablation import make_demo_anchor_agent
from training.peg_insertion.checkpoint import restore_fresh_state
from training.peg_insertion.observation_contract import V2, validate_config

BASE = ROOT / "configs/peg_insertion/peg_v2_formal_5k_seed0.yaml"
CONFIG = ROOT / "configs/peg_insertion/peg_v2_demo_anchor_ablation_1k_seed0.yaml"
DATASET = ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
CHECKPOINTS = ROOT / "results/formal/peg_f3_v2_formal_5k_seed0_20260906/run/checkpoints"
MEASUREMENT = ROOT / "results/validation/peg/f3_v2_demo_anchor_scale_20260906/scale_measurement.json"
EXPECTED_BASE = "eee952a9e88f46cb173d9894d4ee7d90aa6155541fcc5f3cd5bdde64a99538de"
EXPECTED_DATASET = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tree_equal(a, b):
    aa = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(a)]
    bb = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(b)]
    return len(aa) == len(bb) and all(x.shape == y.shape and np.array_equal(x, y)
                                      for x, y in zip(aa, bb))


def tree_norm(tree):
    return float(np.sqrt(sum(np.sum(np.square(np.asarray(jax.device_get(x)), dtype=np.float64))
                             for x in jax.tree_util.tree_leaves(tree))))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    require(ROOT in out.parents and not out.exists(), "fresh output required")
    base = yaml.safe_load(BASE.read_text()); cfg = yaml.safe_load(CONFIG.read_text())
    require(sha(BASE) == EXPECTED_BASE and sha(DATASET) == EXPECTED_DATASET, "protected hashes")
    require(validate_config(base, for_training=True) == validate_config(cfg, for_training=True) == V2,
            "V2 config")
    for section in ("environment", "algorithm", "seed", "fresh_initialization", "load_legacy_checkpoint",
                    "runtime", "demo", "observation", "action", "contracts"):
        require(cfg[section] == base[section], f"protected section changed: {section}")
    expected_training = dict(base["training"]); expected_training["checkpoint_period"] = 500
    require(cfg["training"] == expected_training, "training semantics drift")
    require(cfg["optimizer"] == {"actor_learning_rate": .0003, "actor_warmup_steps": 2000,
                                  "critic_learning_rate": .0003, "critic_warmup_steps": 2000,
                                  "temperature_learning_rate": .0003}, "optimizer drift")
    require(cfg["algorithm_defaults"] == {"discount": .99, "soft_target_update_rate": .005,
                                           "target_entropy": -3.5, "backup_entropy": False,
                                           "critic_ensemble_size": 10, "critic_subsample_size": 2},
            "algorithm default drift")
    anchor = cfg["demo_actor_anchor"]
    require(anchor == {"enabled": True, "scope": "all_demo",
                        "scope_source": "fixed_rlpd_concat_demo_tail", "demo_batch_size": 32,
                        "loss": "mse_policy_mode_to_demo_action", "lambda_small": .07,
                        "lambda_medium": .14, "selected_lambda": .07}, "anchor contract")
    source = inspect.getsource(SACAgent.policy_loss_fn)
    require("predicted_q - temperature * log_probs" in source and "actor_loss = -jnp.mean" in source,
            "production actor objective drift")
    impl_source = inspect.getsource(make_demo_anchor_agent)
    class_source = inspect.getsource(__import__(
        "training.peg_insertion.actor_demo_anchor_ablation", fromlist=["DemoAnchoredSACAgent"]
    ).DemoAnchoredSACAgent)
    require("super().policy_loss_fn" in class_source and "[-demo_batch_size:]" in class_source,
            "additive/demo-only implementation")

    data = pickle.loads(DATASET.read_bytes())
    require(set(data[0]) == {"observations", "actions", "next_observations", "rewards", "masks", "dones"},
            "phase identity unexpectedly available in replay")
    arrays = {key: np.asarray([row[key] for row in data]) for key in data[0]}
    payload = pickle.loads((CHECKPOINTS / "checkpoint_1000_online_replay.pkl").read_bytes())
    online = {key: np.asarray(value)[:int(payload["size"])] for key, value in payload["dataset_dict"].items()}
    rng_np = np.random.default_rng(20260906)
    oi = rng_np.choice(len(online["actions"]), 32, replace=False)
    di = rng_np.choice(len(arrays["actions"]), 32, replace=False)
    keys = ("observations", "actions", "next_observations", "rewards", "masks", "dones")
    batch = {key: jnp.asarray(np.concatenate([online[key][oi], arrays[key][di]], axis=0)) for key in keys}

    sample_obs, sample_action = np.zeros(20, np.float32), np.zeros(7, np.float32)
    prod_fresh = make_sac_agent(0, sample_obs, sample_action)
    candidate_fresh = make_demo_anchor_agent(0, sample_obs, sample_action, cfg)
    require(tree_equal(prod_fresh.state.params, candidate_fresh.state.params), "architecture/init changed")
    state = restore_fresh_state(CHECKPOINTS, prod_fresh.state, 1000)
    candidate = candidate_fresh.replace(state=state)
    key = jax.random.PRNGKey(20260906)
    params_before = [np.asarray(jax.device_get(x)).copy() for x in jax.tree_util.tree_leaves(state.params)]

    def sac_fn(params): return SACAgent.policy_loss_fn(candidate, batch, params, key)[0]
    def anchor_fn(params): return candidate.demo_anchor_loss_fn(batch, params, key)
    def total_fn(params): return candidate.policy_loss_fn(batch, params, key)[0]
    sac_loss, sac_grad = jax.value_and_grad(sac_fn)(state.params)
    raw_anchor, anchor_grad = jax.value_and_grad(anchor_fn)(state.params)
    total_loss, total_grad = jax.value_and_grad(total_fn)(state.params)
    sac_norm, anchor_norm, total_norm = map(tree_norm, (sac_grad, anchor_grad, total_grad))
    weighted_ratio = anchor["selected_lambda"] * anchor_norm / sac_norm
    require(all(np.isfinite(x) for x in (sac_loss, raw_anchor, total_loss, sac_norm, anchor_norm, total_norm)),
            "nonfinite loss/gradient")
    require(np.isclose(float(total_loss), float(sac_loss + anchor["selected_lambda"] * raw_anchor),
                       rtol=1e-6, atol=1e-6), "non-additive objective")
    require(.05 <= weighted_ratio <= .20, "anchor gradient contribution out of range")
    params_after = [np.asarray(jax.device_get(x)) for x in jax.tree_util.tree_leaves(state.params)]
    require(all(np.array_equal(a, b) for a, b in zip(params_before, params_after)), "dry-run updated params")
    measurement = json.loads(MEASUREMENT.read_text())

    result = {
        "PEG_F3_V2_DEMO_ANCHOR_MINIMAL_FIX_DESIGN": "PASS",
        "ACTOR_UPDATE_FILE": str(Path(inspect.getsourcefile(SACAgent)).resolve()),
        "CURRENT_ACTOR_OBJECTIVE": "-mean(Q(s,a_sample)-temperature*log_prob(a_sample|s)) on all 64 concatenated states",
        "ACTOR_Q_TERM": "mean ensemble Q for sampled policy actions",
        "ENTROPY_TERM": "temperature*log_prob in actor objective",
        "DEMO_SPECIFIC_ACTOR_TERM_PRESENT": "NO",
        "ACTOR_BATCH": "32 Online followed by 32 Demo; actor objective currently averages all 64",
        "DEMO_IDENTITY_AT_UPDATE": "distinguishable only by fixed concat boundary; no replay metadata field",
        "ANCHOR_SCOPE": "ALL_DEMO", "ANCHOR_SCOPE_SOURCE": "fixed RLPD concat tail batch[-32:]",
        "ANCHOR_LOSS": "mean((mode(policy(obs_demo))-action_demo)^2)",
        "ACTOR_LOSS_TYPICAL_MAGNITUDE": measurement["ACTOR_LOSS_TYPICAL_MAGNITUDE"],
        "DEMO_ANCHOR_RAW_MAGNITUDE": float(raw_anchor),
        "LAMBDA_SMALL": .07, "LAMBDA_MEDIUM": .14,
        "SAC_ACTOR_LOSS_FINITE": "PASS", "DEMO_ANCHOR_LOSS_FINITE": "PASS",
        "TOTAL_ACTOR_LOSS_FINITE": "PASS", "SAC_GRAD_NORM": sac_norm,
        "ANCHOR_GRAD_NORM": anchor_norm, "TOTAL_GRAD_NORM": total_norm,
        "ANCHOR_GRAD_RATIO": weighted_ratio, "PARAMETERS_UPDATED": False,
        "FILES_EXPECTED_TO_CHANGE": [
            "training/peg_insertion/actor_demo_anchor_ablation.py",
            "configs/peg_insertion/peg_v2_demo_anchor_ablation_1k_seed0.yaml",
            "future isolated ablation runner only"],
        "BASELINE_TRAINING_SEMANTICS_PRESERVED": "PASS", "CRITIC_OBJECTIVE_CHANGED": "NO",
        "ENVIRONMENT_CHANGED": "NO", "DATASET_CHANGED": "NO",
        "CONFIG_PATH": str(CONFIG.relative_to(ROOT)), "CONFIG_SHA256": sha(CONFIG),
        "DEMO_ANCHOR_ABLATION_1K": "READY", "TRAINING_STARTED": False,
        "NEXT": "USER_AUTHORIZATION_REQUIRED_FOR_DEMO_ANCHOR_1K",
    }
    out.mkdir(parents=True)
    (out / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "REPORT.md").write_text("# Peg F3 V2 Minimal Actor Demo-Action Anchoring Design\n\n```json\n" +
                                    json.dumps(result, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
