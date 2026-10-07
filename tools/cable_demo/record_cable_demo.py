"""Keyboard recorder for authentic Cable Routing V2 demonstrations."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tools.cable_demo import (
    ENVIRONMENT_ID,
    PROJECT_ROOT,
    make_cable_training_env,
    make_transition,
    schema_metadata,
)


KEY_ACTIONS = {
    ord("w"): np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ord("s"): np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    ord("a"): np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ord("d"): np.array([0.0, -1.0, 0.0], dtype=np.float32),
    ord("q"): np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ord("e"): np.array([0.0, 0.0, -1.0], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-magnitude", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/cable_demo",
    )
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=PROJECT_ROOT / "runtime/cable_demo/TEST_ONLY",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write exactly one non-demo TEST_ONLY transition and verify loader insertion.",
    )
    return parser.parse_args()


def output_paths(directory: Path, episode_count: int, test_only: bool) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "TEST_ONLY_" if test_only else ""
    stem = f"{prefix}cable_routing_{episode_count:02d}_demos_{timestamp}"
    return directory / f"{stem}.pkl", directory / f"{stem}_metadata.json"


def build_metadata(
    episodes: list[dict[str, Any]],
    transition_count: int,
    test_only: bool,
) -> dict[str, Any]:
    metadata = {
        "format_version": 1,
        "test_only": test_only,
        "environment": ENVIRONMENT_ID,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "mujoco_gl": "egl",
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "episode_count": len(episodes),
        "transition_count": transition_count,
        "success_count": sum(bool(item["success"]) for item in episodes),
        "success_only_policy": not test_only,
        "episodes": episodes,
    }
    metadata.update(schema_metadata())
    return metadata


def save_dataset(
    transitions: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    directory: Path,
    test_only: bool,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    pickle_path, metadata_path = output_paths(directory, len(episodes), test_only)
    with pickle_path.open("xb") as stream:
        pickle.dump(transitions, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = build_metadata(episodes, len(transitions), test_only)
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return pickle_path, metadata_path


def verify_demo_consumer(pickle_path: Path, env) -> int:
    """Use SERL's actual pickle population helper and replay implementation."""
    from serl_launcher.data.data_store import populate_data_store
    from serl_launcher.utils.launcher import make_replay_buffer

    data_store = make_replay_buffer(
        env,
        capacity=16,
        type="memory_efficient_replay_buffer",
        image_keys=["front", "wrist"],
    )
    populate_data_store(data_store, [str(pickle_path)])
    return len(data_store)


def run_dry_run(args: argparse.Namespace) -> int:
    env = make_cable_training_env()
    try:
        observation, _ = env.reset()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        transition = make_transition(
            observation,
            action,
            next_observation,
            reward,
            terminated,
            truncated,
        )
        episode = {
            "episode_id": "TEST_ONLY_0001",
            "start_index": 0,
            "end_index_exclusive": 1,
            "steps": 1,
            "return": float(reward),
            "success": bool(info.get("success", False)),
            "duration_seconds": None,
            "final_distance_to_slot": float(info["distance_to_slot"]),
            "complete": bool(terminated or truncated),
            "discard_reason": "dry_run_not_human_demonstration",
        }
        pickle_path, metadata_path = save_dataset(
            [transition], [episode], args.test_output_dir, test_only=True
        )
        replay_size = verify_demo_consumer(pickle_path, env)
        print(f"DRY_RUN: PASS")
        print(f"TEST_ONLY_PKL: {pickle_path}")
        print(f"TEST_ONLY_METADATA: {metadata_path}")
        print(f"CONSUMER_INSERT: PASS (replay size {replay_size})")
        print("FORMAL_DEMO_SAVED: NO")
        return 0
    finally:
        env.close()


def display_and_read_key(observation: dict[str, np.ndarray], episode: int, steps: int) -> int:
    import cv2

    front = observation["front"][0]
    wrist = observation["wrist"][0]
    frame = np.concatenate([front, wrist], axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(
        frame,
        f"episode={episode} steps={steps}  W/S X  A/D Y  Q/E Z",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow("Cable Routing Demo Recorder", frame)
    return cv2.waitKey(0) & 0xFF


def run_interactive(args: argparse.Namespace) -> int:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not 0.0 < args.action_magnitude <= 1.0:
        raise ValueError("--action-magnitude must be in (0, 1]")

    env = make_cable_training_env()
    accepted: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    try:
        while len(episode_records) < args.episodes:
            observation, _ = env.reset()
            episode_transitions: list[dict[str, Any]] = []
            episode_return = 0.0
            started = time.monotonic()
            last_info: dict[str, Any] = {}
            discard_reason: str | None = None

            while len(episode_transitions) < args.max_steps:
                key = display_and_read_key(
                    observation, len(episode_records) + 1, len(episode_transitions)
                )
                if key == 27:
                    print("Exit requested; current episode discarded.")
                    return 0
                if key in (ord("r"), ord("R")):
                    discard_reason = "manual_reset"
                    break
                if key in (10, 13):
                    discard_reason = "manual_end_without_success"
                    break

                direction = KEY_ACTIONS.get(ord(chr(key).lower()) if key < 128 else key)
                if direction is None:
                    print("Ignored key. Use W/S, A/D, Q/E, R, Enter, or ESC.")
                    continue

                action = np.clip(
                    direction * args.action_magnitude,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)
                episode_transitions.append(
                    make_transition(
                        observation,
                        action,
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                    )
                )
                episode_return += float(reward)
                observation = next_observation
                last_info = dict(info)

                if bool(info.get("success", False)):
                    start_index = len(accepted)
                    accepted.extend(episode_transitions)
                    episode_records.append(
                        {
                            "episode_id": f"cable_success_{len(episode_records) + 1:04d}",
                            "start_index": start_index,
                            "end_index_exclusive": len(accepted),
                            "steps": len(episode_transitions),
                            "return": episode_return,
                            "success": True,
                            "duration_seconds": time.monotonic() - started,
                            "final_distance_to_slot": float(info["distance_to_slot"]),
                            "complete": True,
                        }
                    )
                    print(f"Accepted successful episode {len(episode_records)}/{args.episodes}.")
                    break
                if terminated or truncated:
                    discard_reason = "terminated_without_success" if terminated else "timeout"
                    break
            else:
                discard_reason = "recorder_max_steps"

            if discard_reason is not None:
                distance = last_info.get("distance_to_slot", "UNVERIFIED")
                print(
                    f"Discarded episode: {discard_reason}; "
                    f"steps={len(episode_transitions)} distance={distance}"
                )

        pickle_path, metadata_path = save_dataset(
            accepted, episode_records, args.output_dir, test_only=False
        )
        print(f"FORMAL_PKL: {pickle_path}")
        print(f"FORMAL_METADATA: {metadata_path}")
        return 0
    finally:
        try:
            import cv2

            cv2.destroyAllWindows()
        finally:
            env.close()


def main() -> int:
    args = parse_args()
    return run_dry_run(args) if args.dry_run else run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
