"""Reconstruct synchronized visual Peg demos by replaying accepted Action7 streams."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from experiments.peg_vision_drq_demo_v1.peg_vision_env import (
    FRONT_CAMERA_NAME,
    PROJECT_ROOT,
    WRIST_CAMERA_NAME,
    experiment_peg_runtime,
    make_object_contract_env,
)
PRODUCTION_DEMO = (
    PROJECT_ROOT / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl"
)
PRODUCTION_DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"

ALL11 = (
    ("standard", "results/demos/peg_v2_pilot/standard_post_exclusion_20260905_001"),
    ("standard", "results/demos/peg_v2_pilot/standard_post_exclusion_20260906_002"),
    ("standard", "results/demos/peg_v2_pilot/standard_post_exclusion_20260906_003"),
    ("lateral_plus", "results/demos/peg_v2_pilot/lateral_plus_y_post_exclusion_20260905_003"),
    ("lateral_plus", "results/demos/peg_v2_pilot/lateral_plus_y_post_exclusion_20260906_004"),
    ("lateral_minus", "results/demos/peg_v2_pilot/lateral_minus_y_post_exclusion_20260905_001"),
    ("lateral_minus", "results/demos/peg_v2_pilot/lateral_minus_y_post_exclusion_20260906_002"),
    ("rotation_roll", "results/demos/peg_v2_pilot/rotation_roll_post_exclusion_20260905_001"),
    ("rotation_roll", "results/demos/peg_v2_pilot/rotation_roll_post_exclusion_20260906_002"),
    ("rotation_pitch", "results/demos/peg_v2_pilot/rotation_pitch_post_exclusion_20260906_002"),
    ("rotation_pitch", "results/demos/peg_v2_pilot/rotation_pitch_post_exclusion_20260906_003"),
)
PILOT5 = tuple(
    ALL11[index] for index in (0, 3, 5, 7, 9)
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_sha(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        action = np.asarray(row["action"], dtype=np.float32)
        digest.update(action.tobytes())
    return digest.hexdigest()


def clone_obs(obs: dict) -> dict:
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def raw_env(env):
    """Unwrap without importing the GPU screening entry point.

    That entry point mutates ``MUJOCO_GL`` to EGL at import time.  The target
    WSL host requires OSMesa for stable unattended visual reconstruction.
    """
    seen = set()
    while hasattr(env, "env") and id(env) not in seen:
        seen.add(id(env))
        env = env.env
    return env


def contact_flags(env):
    """Exact contact-health rule from screen_peg_f3_v2_formal_5k.py."""
    raw = raw_env(env)
    model, data = raw.model, raw.data
    link01 = unexpected = False
    pairs = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        tokens, bodies = [], []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            body_id = int(model.geom_bodyid[geom_id])
            data_id = int(model.geom_dataid[geom_id])
            geom_name = model.geom(geom_id).name
            body_name = model.body(body_id).name
            mesh_name = (
                model.mesh(data_id).name
                if int(model.geom_type[geom_id]) == 7 and data_id >= 0
                else None
            )
            tokens.extend((geom_name, body_name, mesh_name))
            bodies.append(body_name)
        is_link01 = frozenset(bodies) == frozenset(("base_link", "shoulder_Link"))
        expected = any(
            "hole" in str(token).lower() for token in tokens if token is not None
        )
        link01 |= is_link01
        unexpected |= not expected
        pairs.append(bodies)
    return link01, unexpected, pairs


def table_ee_contact(env) -> bool:
    model, data = env.unwrapped.model, env.unwrapped.data

    def descendant(child: int, ancestor: int) -> bool:
        while child > 0:
            if child == ancestor:
                return True
            child = int(model.body_parentid[child])
        return child == ancestor

    tables = {
        idx for idx in range(model.nbody) if "table" in (model.body(idx).name or "").lower()
    }
    anchors = {
        idx
        for idx in range(model.nbody)
        if any(
            token in (model.body(idx).name or "").lower()
            for token in ("ft_sensor_mount", "gripper", "assemble_axle", "end_effector")
        )
    }
    ee_bodies = {
        idx
        for idx in range(model.nbody)
        if any(descendant(idx, anchor) for anchor in anchors)
    }
    for idx in range(int(data.ncon)):
        contact = data.contact[idx]
        bodies = {
            int(model.geom_bodyid[int(contact.geom1)]),
            int(model.geom_bodyid[int(contact.geom2)]),
        }
        if bodies & tables and bodies & ee_bodies:
            return True
    return False


def image_ok(obs: dict) -> bool:
    return all(
        value.shape == (1, 128, 128, 3)
        and value.dtype == np.uint8
        and float(value.std()) > 1.0
        for value in (np.asarray(obs["front"]), np.asarray(obs["wrist"]))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("pilot5", "all11", "one"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    specs = {"pilot5": PILOT5, "all11": ALL11, "one": PILOT5[:1]}[args.set]
    output = args.output_dir.resolve()
    if PROJECT_ROOT not in output.parents:
        raise ValueError("dataset output must be below project root")
    output.mkdir(parents=True, exist_ok=False)
    if sha_file(PRODUCTION_DEMO) != PRODUCTION_DEMO_SHA:
        raise RuntimeError("production Demo SHA mismatch before reconstruction")

    dataset: list[dict] = []
    provenance: list[dict] = []
    camera_config = Path(__file__).with_name("peg_vision_env.py")
    camera_sha = sha_file(camera_config)
    with experiment_peg_runtime():
        env = make_object_contract_env(max_episode_steps=500)
        for episode_index, (episode_type, relative) in enumerate(specs):
            source_dir = PROJECT_ROOT / relative
            source_stream = source_dir / "transitions.jsonl"
            rows = [json.loads(line) for line in source_stream.open() if line.strip()]
            status = json.loads((source_dir / "status.json").read_text())
            if status.get("status") != "COMPLETED_SUCCESS":
                raise RuntimeError(f"source not accepted success: {relative}")
            if status.get("transitions") != len(rows):
                raise RuntimeError(f"source transition count mismatch: {relative}")
            if status.get("transitions_sha256") != sha_file(source_stream):
                raise RuntimeError(f"source stream SHA mismatch: {relative}")

            obs, _ = env.reset(seed=0)
            state_max_abs = reward_max_abs = 0.0
            max_force = 0.0
            link01_seen = unexpected_seen = table_seen = False
            front_deltas: list[float] = []
            wrist_deltas: list[float] = []
            episode_start = len(dataset)
            for step, row in enumerate(rows):
                source_obs = np.asarray(row["obs"], dtype=np.float32)
                state_delta = float(np.max(np.abs(np.asarray(obs["state"])[0] - source_obs)))
                state_max_abs = max(state_max_abs, state_delta)
                if state_delta != 0.0 or not image_ok(obs):
                    raise RuntimeError(f"observation mismatch/camera failure {relative} step {step}")
                action = np.asarray(row["action"], dtype=np.float32)
                if action.shape != (7,) or not env.action_space.contains(action):
                    raise RuntimeError(f"illegal Action7 {relative} step {step}")
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_delta = float(
                    np.max(
                        np.abs(
                            np.asarray(next_obs["state"])[0]
                            - np.asarray(row["next_obs"], dtype=np.float32)
                        )
                    )
                )
                state_max_abs = max(state_max_abs, next_delta)
                reward_delta = abs(float(reward) - float(row["reward"]))
                reward_max_abs = max(reward_max_abs, reward_delta)
                if next_delta != 0.0 or reward_delta != 0.0:
                    raise RuntimeError(f"physics/reward mismatch {relative} step {step}")
                if bool(terminated) != bool(row["terminated"]) or bool(truncated) != bool(row["truncated"]):
                    raise RuntimeError(f"termination mismatch {relative} step {step}")
                if not image_ok(next_obs):
                    raise RuntimeError(f"next camera failure {relative} step {step}")
                front_deltas.append(
                    float(np.mean(np.abs(next_obs["front"].astype(np.int16) - obs["front"].astype(np.int16))))
                )
                wrist_deltas.append(
                    float(np.mean(np.abs(next_obs["wrist"].astype(np.int16) - obs["wrist"].astype(np.int16))))
                )
                link01, unexpected, _pairs = contact_flags(env)
                link01_seen |= bool(link01)
                unexpected_seen |= bool(unexpected)
                table_seen |= table_ee_contact(env)
                force = float(info.get("force_norm", 0.0))
                max_force = max(max_force, force if np.isfinite(force) else float("inf"))
                dataset.append(
                    {
                        "observations": clone_obs(obs),
                        "actions": action.copy(),
                        "next_observations": clone_obs(next_obs),
                        "rewards": np.float32(reward),
                        "masks": np.float32(1.0 - float(terminated)),
                        "dones": np.bool_(terminated or truncated),
                    }
                )
                obs = next_obs

            success = bool(rows[-1]["success"] and rows[-1]["terminated"] and not rows[-1]["truncated"])
            contact_health = bool(
                not link01_seen and not unexpected_seen and not table_seen and np.isfinite(max_force)
            )
            if not success or not contact_health:
                raise RuntimeError(f"semantic/contact gate failed: {relative}")
            source_hashes = rows[0].get("hashes", {})
            provenance.append(
                {
                    "episode_index": episode_index,
                    "episode_type": episode_type,
                    "source_episode": relative,
                    "source_stream_sha256": sha_file(source_stream),
                    "source_action_sequence_sha256": action_sha(rows),
                    "camera_config_sha256": camera_sha,
                    "environment_sha256": source_hashes.get("env_hash"),
                    "controller_sha256": source_hashes.get("files", {}).get(
                        "external/packages/peg_insertion/peg_insertion/code/env/mujoco_env/mujoco_env/controllers/operational_space.py"
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "transition_start": episode_start,
                    "transition_count": len(rows),
                    "state_max_abs_error": state_max_abs,
                    "reward_max_abs_error": reward_max_abs,
                    "camera_continuity": {
                        "front_mean_step_mae": float(np.mean(front_deltas)),
                        "front_max_step_mae": float(np.max(front_deltas)),
                        "wrist_mean_step_mae": float(np.mean(wrist_deltas)),
                        "wrist_max_step_mae": float(np.max(wrist_deltas)),
                    },
                    "success": success,
                    "truncated": False,
                    "contact_health": contact_health,
                    "max_force": max_force,
                }
            )
            print(f"RECONSTRUCTED {episode_index + 1}/{len(specs)} {relative}", flush=True)
        env.close()

    dataset_path = output / "peg_vision_demo_v1.pkl"
    with dataset_path.open("xb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": "PEG_VISION_DEMO_V1_OBJECT_REPLAY_SCHEMA",
        "status": "PASS",
        "set": args.set,
        "episodes": len(specs),
        "transitions": len(dataset),
        "episode_types": {name: sum(row["episode_type"] == name for row in provenance) for name in sorted({row["episode_type"] for row in provenance})},
        "state_shape": [1, 20],
        "front_shape": [1, 128, 128, 3],
        "wrist_shape": [1, 128, 128, 3],
        "action_shape": [7],
        "front_camera": FRONT_CAMERA_NAME,
        "wrist_camera": WRIST_CAMERA_NAME,
        "dataset_file": dataset_path.name,
        "dataset_sha256": sha_file(dataset_path),
        "production_demo_sha_unchanged": sha_file(PRODUCTION_DEMO) == PRODUCTION_DEMO_SHA,
        "provenance": provenance,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "set", "episodes", "transitions", "dataset_sha256", "production_demo_sha_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
