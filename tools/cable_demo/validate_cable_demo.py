"""Static validator for project-owned Cable Routing demo pickle files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from tools.cable_demo import OBSERVATION_KEYS, REQUIRED_TRANSITION_KEYS


EXPECTED_OBSERVATIONS = {
    "state": ((1, 7), np.dtype("float32")),
    "front": ((1, 128, 128, 3), np.dtype("uint8")),
    "wrist": ((1, 128, 128, 3), np.dtype("uint8")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_path", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--allow-test-only",
        action="store_true",
        help="Permit a TEST_ONLY incomplete/non-success dry-run artifact.",
    )
    return parser.parse_args()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_observation(obs: Any, label: str, errors: list[str]) -> None:
    require(isinstance(obs, dict), f"{label}: expected dict", errors)
    if not isinstance(obs, dict):
        return
    require(tuple(obs.keys()) == OBSERVATION_KEYS, f"{label}: wrong/order keys {tuple(obs.keys())}", errors)
    for key, (shape, dtype) in EXPECTED_OBSERVATIONS.items():
        if key not in obs:
            continue
        array = np.asarray(obs[key])
        require(array.shape == shape, f"{label}/{key}: shape {array.shape} != {shape}", errors)
        require(array.dtype == dtype, f"{label}/{key}: dtype {array.dtype} != {dtype}", errors)
        if key == "state":
            require(np.isfinite(array).all(), f"{label}/{key}: NaN/Inf", errors)


def validate_transition(transition: Any, index: int, errors: list[str]) -> None:
    label = f"transition[{index}]"
    require(isinstance(transition, dict), f"{label}: expected dict", errors)
    if not isinstance(transition, dict):
        return
    require(tuple(transition.keys()) == REQUIRED_TRANSITION_KEYS, f"{label}: wrong/order keys {tuple(transition.keys())}", errors)
    if "observations" in transition:
        validate_observation(transition["observations"], f"{label}/observations", errors)
    if "next_observations" in transition:
        validate_observation(transition["next_observations"], f"{label}/next_observations", errors)
    if "actions" in transition:
        action = np.asarray(transition["actions"])
        require(action.shape == (3,), f"{label}/actions: shape {action.shape}", errors)
        require(action.dtype == np.dtype("float32"), f"{label}/actions: dtype {action.dtype}", errors)
        require(np.isfinite(action).all(), f"{label}/actions: NaN/Inf", errors)
        require(np.logical_and(action >= -1.0, action <= 1.0).all(), f"{label}/actions: outside [-1,1]", errors)
    for key, dtype in (("rewards", "float32"), ("masks", "float32"), ("dones", "bool")):
        if key not in transition:
            continue
        value = np.asarray(transition[key])
        require(value.shape == (), f"{label}/{key}: expected scalar", errors)
        require(value.dtype == np.dtype(dtype), f"{label}/{key}: dtype {value.dtype} != {dtype}", errors)
        if key != "dones":
            require(np.isfinite(value).all(), f"{label}/{key}: NaN/Inf", errors)
    if "masks" in transition:
        mask = float(np.asarray(transition["masks"]))
        require(mask in (0.0, 1.0), f"{label}/masks: must be 0 or 1", errors)


def validate_episodes(
    transitions: list[Any], metadata: dict[str, Any], allow_test_only: bool, errors: list[str]
) -> None:
    episodes = metadata.get("episodes")
    require(isinstance(episodes, list) and len(episodes) > 0, "metadata: empty/missing episodes", errors)
    if not isinstance(episodes, list):
        return
    cursor = 0
    test_only = bool(metadata.get("test_only", False))
    require(not test_only or allow_test_only, "TEST_ONLY artifact requires --allow-test-only", errors)
    for episode_index, episode in enumerate(episodes):
        label = f"episode[{episode_index}]"
        require(isinstance(episode, dict), f"{label}: expected dict", errors)
        if not isinstance(episode, dict):
            continue
        start = episode.get("start_index")
        end = episode.get("end_index_exclusive")
        require(start == cursor, f"{label}: non-contiguous start {start} != {cursor}", errors)
        require(isinstance(end, int) and isinstance(start, int) and end > start, f"{label}: empty/invalid range", errors)
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(transitions):
            continue
        require(episode.get("steps") == end - start, f"{label}: steps/range mismatch", errors)
        success = episode.get("success") is True
        if not test_only:
            require(success, f"{label}: formal dataset contains non-success episode", errors)
            for idx in range(start, end - 1):
                require(not bool(transitions[idx]["dones"]), f"{label}: early done at {idx}", errors)
                require(float(transitions[idx]["masks"]) == 1.0, f"{label}: early mask at {idx}", errors)
            require(bool(transitions[end - 1]["dones"]), f"{label}: final done is false", errors)
            require(float(transitions[end - 1]["masks"]) == 0.0, f"{label}: final success mask is not 0", errors)
        cursor = end
    require(cursor == len(transitions), f"metadata: episode ranges cover {cursor}/{len(transitions)}", errors)
    require(metadata.get("episode_count") == len(episodes), "metadata: episode_count mismatch", errors)
    require(metadata.get("transition_count") == len(transitions), "metadata: transition_count mismatch", errors)
    require(metadata.get("success_count") == sum(ep.get("success") is True for ep in episodes if isinstance(ep, dict)), "metadata: success_count mismatch", errors)


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata or args.pickle_path.with_name(
        f"{args.pickle_path.stem}_metadata.json"
    )
    errors: list[str] = []
    try:
        with args.pickle_path.open("rb") as stream:
            transitions = pickle.load(stream)
    except Exception as exc:
        print(f"RESULT: FAIL\nERROR: pickle unreadable: {type(exc).__name__}: {exc}")
        return 1
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except Exception as exc:
        print(f"RESULT: FAIL\nERROR: metadata unreadable: {type(exc).__name__}: {exc}")
        return 1

    require(isinstance(transitions, list), "pickle top level must be list", errors)
    require(isinstance(transitions, list) and len(transitions) > 0, "empty transition list", errors)
    if isinstance(transitions, list):
        for index, transition in enumerate(transitions):
            validate_transition(transition, index, errors)
        validate_episodes(transitions, metadata, args.allow_test_only, errors)

    if errors:
        print("RESULT: FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("RESULT: PASS")
    print(f"TEST_ONLY: {bool(metadata.get('test_only', False))}")
    print(f"EPISODES: {metadata['episode_count']}")
    print(f"TRANSITIONS: {len(transitions)}")
    print(f"SUCCESSES: {metadata['success_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
