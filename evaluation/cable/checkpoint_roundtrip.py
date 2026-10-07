"""Bounded full-architecture Cable agent legacy-checkpoint roundtrip."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import config as flax_config
from flax.training import checkpoints

from training.cable.run_cable_training import (
    DEFAULT_CONFIG,
    assert_observation,
    load_config,
    load_delivered_module,
)
from tools.cable_demo import make_cable_training_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime/cable_handoff/checkpoint_roundtrip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_agent(config: dict[str, Any], delivered: Any, env: Any) -> Any:
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    agent = delivered.make_drq_agent(
        seed=int(config["seed"]),
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=config["training"]["encoder_type"],
    )
    return jax.device_put(jax.tree_map(jnp.array, agent), delivered.sharding.replicate())


def leaf_digest(value: Any) -> str:
    array = np.asarray(jax.device_get(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tree_signature(tree: Any) -> tuple[Any, list[dict[str, Any]], list[Any]]:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    leaves = [leaf for _, leaf in path_leaves]
    signature = [
        {
            "index": index,
            "path": jax.tree_util.keystr(path),
            "shape": list(np.shape(leaf)),
            "dtype": str(np.asarray(jax.device_get(leaf)).dtype),
        }
        for index, (path, leaf) in enumerate(path_leaves)
    ]
    return treedef, signature, leaves


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if PROJECT_ROOT not in output_dir.parents:
        raise ValueError("roundtrip output must remain inside the project")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    delivered = load_delivered_module()
    env = make_cable_training_env()
    old_orbax = flax_config.flax_use_orbax_checkpointing
    try:
        observation, _ = env.reset(seed=int(config["seed"]))
        assert_observation(observation, config)
        original = make_agent(config, delivered, env)
        original_tree, original_signature, original_leaves = tree_signature(original.state)
        representative_indices = sorted({0, len(original_leaves) // 2, len(original_leaves) - 1})
        original_hashes = [leaf_digest(leaf) for leaf in original_leaves]

        flax_config.update("flax_use_orbax_checkpointing", False)
        saved_path = checkpoints.save_checkpoint(
            output_dir, original.state, step=0, keep=1, overwrite=False
        )
        del original, original_leaves
        gc.collect()

        rebuilt = make_agent(config, delivered, env)
        restored_state = checkpoints.restore_checkpoint(output_dir, rebuilt.state, step=0)
        restored = rebuilt.replace(state=restored_state)
        restored_tree, restored_signature, restored_leaves = tree_signature(restored.state)
        restored_hashes = [leaf_digest(leaf) for leaf in restored_leaves]

        # PyTreeDef auxiliary data contains newly constructed apply_fn/optimizer
        # objects whose identities differ across rebuilds. Stable checkpoint
        # structure is therefore compared by every serialized leaf key path,
        # shape, and dtype; raw PyTreeDef equality is reported diagnostically.
        treedef_pass = original_tree == restored_tree
        signature_pass = original_signature == restored_signature
        structure_pass = signature_pass
        values_pass = original_hashes == restored_hashes
        inference_action = np.asarray(
            jax.device_get(
                restored.sample_actions(
                    observations=jax.device_put(observation),
                    argmax=True,
                )
            )
        )
        inference_pass = (
            inference_action.shape == tuple(config["action"]["shape"])
            and np.isfinite(inference_action).all()
        )
        signature_mismatches = [
            {"before": before, "after": after}
            for before, after in zip(original_signature, restored_signature)
            if before != after
        ][:10]
        report = {
            "checkpoint_step": 0,
            "checkpoint_path": str(saved_path),
            "leaf_count": len(restored_leaves),
            "raw_pytree_auxiliary_metadata_match": treedef_pass,
            "serialized_key_path_shape_dtype_match": signature_pass,
            "signature_mismatches_first_10": signature_mismatches,
            "representative_indices": representative_indices,
            "representative_hashes_before": {
                str(index): original_hashes[index] for index in representative_indices
            },
            "representative_hashes_after": {
                str(index): restored_hashes[index] for index in representative_indices
            },
            "all_leaf_values_match": values_pass,
            "params_leaf_count": sum("params" in item["path"] for item in restored_signature),
            "optimizer_leaf_count": sum("opt_state" in item["path"] for item in restored_signature),
            "temperature_leaf_count": sum("temperature" in item["path"] for item in restored_signature),
            "restored_policy_inference": "PASS" if inference_pass else "FAIL",
            "restored_action_shape": list(inference_action.shape),
            "checkpoint_save": "PASS",
            "checkpoint_restore": "PASS" if structure_pass and values_pass and inference_pass else "FAIL",
            "real_agent_checkpoint_roundtrip": (
                "VALIDATED" if structure_pass and values_pass and inference_pass else "FAILED"
            ),
        }
        report_path = output_dir / "roundtrip_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for key, value in report.items():
            print(f"{key.upper()}: {value}")
        return 0 if structure_pass and values_pass and inference_pass else 1
    finally:
        flax_config.update("flax_use_orbax_checkpointing", old_orbax)
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
