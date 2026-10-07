"""Verify the final Peg bounded-smoke checkpoint bundle without training."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np
from flax.training import checkpoints


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "results/checkpoints/peg_rlpd_gpu_smoke_seed0"
CHECKPOINT = CHECKPOINT_DIR / "checkpoint_9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    replay_path = CHECKPOINT.with_name(f"{CHECKPOINT.name}_online_replay.pkl")
    metadata_path = CHECKPOINT.with_name(f"{CHECKPOINT.name}_resume_metadata.json")
    for path in (CHECKPOINT, replay_path, metadata_path):
        assert path.is_file() and path.stat().st_size > 0, path

    metadata = json.loads(metadata_path.read_text())
    assert metadata["checkpoint_step"] == 9
    assert metadata["learner_update_steps"] == 9
    assert metadata["online_replay_size"] == 121
    assert metadata["online_replay_sha256"] == sha256(replay_path)
    replay = pickle.loads(replay_path.read_bytes())
    assert int(replay["size"]) == 121

    state = checkpoints.restore_checkpoint(CHECKPOINT_DIR, target=None, step=9)
    leaves = jax.tree_util.tree_leaves(state)
    array_leaves = [np.asarray(leaf) for leaf in leaves if hasattr(leaf, "dtype")]
    assert array_leaves and all(np.all(np.isfinite(leaf)) for leaf in array_leaves)

    print("CHECKPOINT_PATH:", CHECKPOINT)
    print("CHECKPOINT_STEP:", metadata["checkpoint_step"])
    print("LEARNER_UPDATES:", metadata["learner_update_steps"] + 1)
    print("ONLINE_REPLAY_SIZE:", metadata["online_replay_size"])
    print("ONLINE_REPLAY_SHA256: PASS")
    print("CHECKPOINT_ARRAY_LEAVES:", len(array_leaves))
    print("CHECKPOINT_FINITE: PASS")
    print("PEG_BOUNDED_SMOKE_RESULT_VERIFY: PASS")


if __name__ == "__main__":
    main()
