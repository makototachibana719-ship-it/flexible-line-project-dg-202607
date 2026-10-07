"""Safe Cable resume sidecars for model progress and online replay state."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match or not path.is_file():
        raise ValueError(f"expected committed legacy Flax checkpoint_N file: {path}")
    return int(match.group(1))


def sidecar_paths(checkpoint_path: Path) -> tuple[Path, Path]:
    stem = checkpoint_path.name
    return (
        checkpoint_path.parent / f"{stem}_resume_metadata.json",
        checkpoint_path.parent / f"{stem}_online_replay.pkl",
    )


def _take(tree: Any, indices: np.ndarray) -> Any:
    if isinstance(tree, Mapping):
        return {key: _take(value, indices) for key, value in tree.items()}
    return np.asarray(tree)[indices].copy()


def _restore(tree: Any, saved: Any, size: int) -> None:
    if isinstance(tree, Mapping):
        if set(tree) != set(saved):
            raise ValueError("online replay tree keys mismatch")
        for key in tree:
            _restore(tree[key], saved[key], size)
        return
    target = np.asarray(tree)
    source = np.asarray(saved)
    if source.shape[0] != size or target.shape[1:] != source.shape[1:]:
        raise ValueError("online replay array shape mismatch")
    if target.dtype != source.dtype:
        raise ValueError("online replay array dtype mismatch")
    target[:size] = source


def _rng_state(buffer: Any) -> dict[str, Any]:
    rng = buffer.np_random
    if hasattr(rng, "bit_generator"):
        return {"kind": "generator", "state": rng.bit_generator.state}
    return {"kind": "random_state", "state": rng.get_state()}


def _set_rng_state(buffer: Any, state: dict[str, Any]) -> None:
    rng = buffer.np_random
    if state["kind"] == "generator" and hasattr(rng, "bit_generator"):
        rng.bit_generator.state = state["state"]
    elif state["kind"] == "random_state" and hasattr(rng, "set_state"):
        rng.set_state(state["state"])
    else:
        raise ValueError("online replay RNG implementation mismatch")


def install_actor_step_tracking(buffer: Any, initial_actor_steps: int = 0) -> None:
    if hasattr(buffer, "project_original_insert"):
        raise RuntimeError("online replay tracking already installed")
    buffer.project_actor_steps = int(initial_actor_steps)
    buffer.project_original_insert = buffer.insert

    def tracked_insert(self, data):
        self.project_original_insert(data)
        self.project_actor_steps += 1

    buffer.insert = MethodType(tracked_insert, buffer)


def snapshot_online_replay(buffer: Any, output_path: Path) -> dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite replay snapshot: {output_path}")
    size = len(buffer)
    capacity = int(buffer._capacity)
    insert_index = int(buffer._insert_index)
    if size < capacity:
        indices = np.arange(size, dtype=np.int64)
    else:
        indices = np.concatenate(
            (np.arange(insert_index, capacity), np.arange(0, insert_index))
        )
    with buffer._lock:
        payload = {
            "format_version": 1,
            "capacity": capacity,
            "size": size,
            "dataset_dict": _take(buffer.dataset_dict, indices),
            "is_correct_index": np.asarray(buffer._is_correct_index)[indices].copy(),
            "first": bool(buffer._first),
            "pixel_keys": list(buffer.pixel_keys),
            "rng_state": _rng_state(buffer),
            "actor_steps": int(buffer.project_actor_steps),
        }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "size": size,
        "capacity": capacity,
        "actor_steps": int(buffer.project_actor_steps),
    }


def restore_online_replay(buffer: Any, snapshot_path: Path) -> dict[str, Any]:
    with snapshot_path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported online replay snapshot format")
    if int(payload["capacity"]) != int(buffer._capacity):
        raise ValueError("online replay capacity mismatch")
    if tuple(payload["pixel_keys"]) != tuple(buffer.pixel_keys):
        raise ValueError("online replay pixel-key mismatch")
    size = int(payload["size"])
    if not 0 <= size <= int(buffer._capacity):
        raise ValueError("invalid online replay size")
    with buffer._lock:
        _restore(buffer.dataset_dict, payload["dataset_dict"], size)
        buffer._is_correct_index[:] = False
        buffer._is_correct_index[:size] = np.asarray(payload["is_correct_index"])
        buffer._size = size
        buffer._insert_index = size % int(buffer._capacity)
        buffer._first = bool(payload["first"])
        _set_rng_state(buffer, payload["rng_state"])
    install_actor_step_tracking(buffer, int(payload["actor_steps"]))
    return {
        "size": size,
        "actor_steps": int(payload["actor_steps"]),
        "sha256": sha256_file(snapshot_path),
    }


def write_resume_metadata(path: Path, metadata: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite resume metadata: {path}")
    payload = dict(metadata)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_resume_bundle(
    checkpoint_path: Path,
    *,
    expected_environment: str,
    expected_algorithm: str,
    expected_seed: int,
    expected_config_hash: str,
    expected_demo_manifest_hash: str,
    expected_observation: dict[str, Any],
    expected_action: dict[str, Any],
    expected_wandb_project: str,
    expected_wandb_name: str,
    explicit_wandb_run_id: str | None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    step = parse_checkpoint_step(checkpoint_path)
    metadata_path, replay_path = sidecar_paths(checkpoint_path)
    if not metadata_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError("resume requires checkpoint, metadata, and online replay sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = {
        "checkpoint_step": step,
        "learner_update_steps": step,
        "environment": expected_environment,
        "algorithm": expected_algorithm,
        "seed": expected_seed,
        "config_hash": expected_config_hash,
        "demo_manifest_hash": expected_demo_manifest_hash,
        "observation": expected_observation,
        "action": expected_action,
        "wandb_project": expected_wandb_project,
        "wandb_name": expected_wandb_name,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"resume {key} mismatch: {metadata.get(key)!r} != {expected!r}")
    if not metadata.get("wandb_run_id"):
        raise ValueError("resume metadata has no W&B run ID")
    if explicit_wandb_run_id and explicit_wandb_run_id != metadata["wandb_run_id"]:
        raise ValueError("explicit W&B run ID mismatch")
    if sha256_file(replay_path) != metadata.get("online_replay_sha256"):
        raise ValueError("online replay SHA256 mismatch")
    metadata.update(
        {
            "checkpoint_path": checkpoint_path,
            "metadata_path": metadata_path,
            "online_replay_path": replay_path,
        }
    )
    return metadata
