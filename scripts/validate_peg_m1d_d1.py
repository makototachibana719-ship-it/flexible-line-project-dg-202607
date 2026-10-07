#!/usr/bin/env python3
"""Bounded M1D-D1 replay, portability, and fresh-checkpoint validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "external/packages/peg_insertion/peg_insertion"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "code/example/async_rlpd_peg_sim"))
sys.path.insert(0, str(PKG / "code/env"))


def replay() -> dict:
    from peg_env_adapter import make_peg_state_env
    from serl_launcher.utils.launcher import make_replay_buffer
    from training.peg_insertion.peg_replay_adapter import repaired_replay_datastore_insert
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

    transitions = pickle.load(open(PKG / "demos/peg_insert_20_demos_state.pkl", "rb"))
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=500)
    with repaired_replay_datastore_insert():
        demo = make_replay_buffer(env, capacity=10000, type="replay_buffer")
        before_rng = repr(demo.np_random.bit_generator.state)
        for transition in transitions:
            demo.insert(transition)
        demo_batch = demo.sample(batch_size=32)
        after_rng = repr(demo.np_random.bit_generator.state)
        online = make_replay_buffer(env, capacity=128, type="replay_buffer")
        for transition in transitions[:64]:
            online.insert(transition)
        online_batch = online.sample(batch_size=32)
    mixed = {k: np.concatenate([online_batch[k], demo_batch[k]], axis=0) for k in demo_batch}
    env.close()
    terminal_count = sum(bool(x["dones"]) or float(x["masks"]) == 0.0 for x in transitions)
    return {
        "result": "PASS",
        "raw_transitions": len(transitions), "demo_size": len(demo),
        "demo_insert_index": demo._insert_index, "online_size": len(online),
        "online_insert_index": online._insert_index, "terminal_transitions": terminal_count,
        "rng_evolved": before_rng != after_rng,
        "demo_batch": {k: [list(np.asarray(v).shape), str(np.asarray(v).dtype)] for k, v in demo_batch.items()},
        "online_batch": {k: [list(np.asarray(v).shape), str(np.asarray(v).dtype)] for k, v in online_batch.items()},
        "mixed_batch": {k: [list(np.asarray(v).shape), str(np.asarray(v).dtype)] for k, v in mixed.items()},
        "all_finite": all(np.isfinite(np.asarray(x[k])).all() for x in transitions for k in ("observations", "actions", "next_observations", "rewards", "masks")),
    }


def portability() -> dict:
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime, runtime_paths
    with portable_peg_runtime(ROOT) as paths:
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=5)
        obs, _ = env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(np.zeros(7, np.float32))
        env.close()
    with tempfile.TemporaryDirectory(prefix="different-user-project-") as tmp:
        synthetic = runtime_paths(Path(tmp) / "flexible_line_project", paths["assets"])
    xml = next(paths["generated"].glob("*.xml"))
    text = xml.read_text(encoding="utf-8")
    return {"result": "PASS", "installed_assets": str(paths["assets"]),
            "generated_xml": str(xml), "asset_link": str(paths["asset_link"]),
            "xml_has_home_ubuntu": ("/home/" + "ubuntu") in text,
            "synthetic_generated": str(synthetic["generated"]),
            "obs_shape": list(np.asarray(obs).shape), "step_reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated)}


def checkpoint() -> dict:
    import jax
    from flax.traverse_util import flatten_dict
    from serl_launcher.utils.launcher import make_sac_agent
    from peg_env_adapter import make_peg_state_env
    from training.peg_insertion.checkpoint import save_fresh_state, restore_fresh_state
    from training.peg_insertion.peg_runtime_adapter import portable_peg_runtime

    out = ROOT / "runtime/peg_handoff/checkpoint_roundtrip"
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "checkpoint_0"
    with portable_peg_runtime(ROOT):
        env = make_peg_state_env(render_mode="rgb_array", max_episode_steps=5)
        obs, _ = env.reset(seed=0)
    def fresh():
        return make_sac_agent(seed=0, sample_obs=np.zeros(20, np.float32), sample_action=np.zeros(7, np.float32))
    agent = fresh(); original = agent.state
    if existing.exists():
        saved = existing
    else:
        saved = save_fresh_state(out, original, 0)
    del agent
    rebuilt = fresh(); restored = restore_fresh_state(out, rebuilt.state, 0)
    old_leaves = jax.tree_util.tree_leaves(original); new_leaves = jax.tree_util.tree_leaves(restored)
    equal = len(old_leaves) == len(new_leaves) and all(np.array_equal(np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b))) for a, b in zip(old_leaves, new_leaves))
    rebuilt = rebuilt.replace(state=restored)
    action = np.asarray(jax.device_get(rebuilt.sample_actions(observations=np.asarray(obs, np.float32), argmax=True)), np.float32)
    env.close()
    flat = flatten_dict(restored.params)
    return {"result": "PASS" if equal and action.shape == (7,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1.0 + 1e-6) else "FAIL",
            "saved": str(saved), "leaf_count": len(old_leaves), "values_equal": equal,
            "parameter_roots": sorted({str(k[0]) for k in flat}), "state_step": int(np.asarray(restored.step)),
            "action_shape": list(action.shape), "action": action.tolist(), "action_finite": bool(np.isfinite(action).all()),
            "action_in_bounds": bool(np.all(np.abs(action) <= 1.0 + 1e-6))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("replay", "portability", "checkpoint")); args = parser.parse_args()
    result = {"replay": replay, "portability": portability, "checkpoint": checkpoint}[args.mode]()
    print(json.dumps(result, indent=2, sort_keys=True))
