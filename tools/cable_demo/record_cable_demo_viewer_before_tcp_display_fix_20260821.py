"""MuJoCo Viewer teleoperation recorder for authentic Cable Routing demos."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tools.cable_demo import ENVIRONMENT_ID, PROJECT_ROOT, make_cable_training_env, make_transition
from tools.cable_demo.record_cable_demo import save_dataset


MOVEMENT_KEYS = frozenset("wasdqe")


def action_from_pressed_keys(pressed: set[str], magnitude: float) -> np.ndarray:
    """Build one control-period action independently of keyboard autorepeat."""
    action = np.zeros(3, dtype=np.float32)
    action[0] = magnitude * (float("w" in pressed) - float("s" in pressed))
    action[1] = magnitude * (float("a" in pressed) - float("d" in pressed))
    action[2] = magnitude * (float("q" in pressed) - float("e" in pressed))
    return action


class PressedKeyTeleop:
    """Thread-safe pynput pressed-key state plus one-shot control requests."""

    def __init__(self, keyboard_module: Any) -> None:
        self._keyboard = keyboard_module
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._reset_requested = False
        self._print_requested = False
        self._end_requested = False
        self._exit_requested = False

    @staticmethod
    def _character(key: Any) -> str | None:
        character = getattr(key, "char", None)
        return character.lower() if isinstance(character, str) else None

    def on_press(self, key: Any) -> None:
        character = self._character(key)
        with self._lock:
            changed = False
            if character in MOVEMENT_KEYS and character not in self._pressed:
                self._pressed.add(character)
                changed = True
            elif character == "r":
                self._reset_requested = True
            elif character == "p":
                self._print_requested = True
            elif key == self._keyboard.Key.enter:
                self._end_requested = True
            elif key == self._keyboard.Key.esc:
                self._exit_requested = True
            if changed:
                print(f"KEY STATE: {sorted(self._pressed)}", flush=True)

    def on_release(self, key: Any) -> None:
        character = self._character(key)
        with self._lock:
            if character in self._pressed:
                self._pressed.remove(character)
                print(f"KEY STATE: {sorted(self._pressed)}", flush=True)

    def snapshot(self) -> tuple[set[str], bool, bool, bool, bool]:
        """Return pressed keys and consume one-shot reset/print/end/exit flags."""
        with self._lock:
            result = (
                set(self._pressed),
                self._reset_requested,
                self._print_requested,
                self._end_requested,
                self._exit_requested,
            )
            self._reset_requested = False
            self._print_requested = False
            self._end_requested = False
            return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="Required successful episode count.")
    parser.add_argument("--action-magnitude", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/cable_demo"
    )
    parser.add_argument(
        "--discard-log-dir",
        type=Path,
        default=PROJECT_ROOT / "runtime/cable_demo/discarded_logs",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run reset/one zero-action step/close without saving any demo; open Viewer only when DISPLAY exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not 0.0 < args.action_magnitude <= 1.0:
        raise ValueError("--action-magnitude must be in (0, 1]")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.print_every < 1:
        raise ValueError("--print-every must be at least 1")


def native_env(env):
    base = env.unwrapped
    if not hasattr(base, "model") or not hasattr(base, "data"):
        raise RuntimeError("Cable environment does not expose MuJoCo model/data")
    return base


def printable_vector(value: Any) -> str:
    return np.array2string(np.asarray(value), precision=4, suppress_small=True)


def print_state(
    episode_number: int,
    step: int,
    action: np.ndarray,
    observation: dict[str, np.ndarray],
    info: dict[str, Any],
    reward: float,
    episode_return: float,
) -> None:
    state = np.asarray(observation["state"])[0]
    print(
        f"Episode {episode_number} | Step {step} | Dist "
        f"{info.get('distance_to_slot', 'UNVERIFIED')} m | "
        f"Return {episode_return:.6f} | Success {bool(info.get('success', False))}"
    )
    print(f"  ACTION: {printable_vector(action)}")
    print(f"  TCP_POS: {printable_vector(state[:3])}")
    print(f"  CABLE_TIP_POS: {printable_vector(info.get('cable_tip_pos', []))}")
    print(f"  SLOT_POS: {printable_vector(info.get('slot_pos', []))}")
    print(f"  REWARD: {float(reward):.6f}")


def write_discard_log(
    directory: Path,
    episode_number: int,
    reason: str,
    steps: int,
    episode_return: float,
    duration: float,
    info: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = directory / f"discarded_cable_episode_{episode_number:04d}_{timestamp}.json"
    payload = {
        "test_only": True,
        "learner_dataset": False,
        "environment": ENVIRONMENT_ID,
        "episode_attempt": episode_number,
        "discard_reason": reason,
        "steps": steps,
        "return": episode_return,
        "duration_seconds": duration,
        "success": bool(info.get("success", False)),
        "final_distance_to_slot": info.get("distance_to_slot"),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return path


def save_success_episode(
    transitions: list[dict[str, Any]],
    episode_number: int,
    episode_return: float,
    duration: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    episode = {
        "episode_id": f"cable_success_{episode_number:04d}",
        "start_index": 0,
        "end_index_exclusive": len(transitions),
        "steps": len(transitions),
        "return": episode_return,
        "success": True,
        "duration_seconds": duration,
        "final_distance_to_slot": float(info["distance_to_slot"]),
        "complete": bool(terminated or truncated),
        "environment_terminated": bool(terminated),
        "environment_truncated": bool(truncated),
        "dataset_boundary_reason": "info_success",
    }
    # Each success is saved immediately as its own one-episode formal dataset.
    # This prevents Viewer closure from losing an already completed success and
    # never overwrites an earlier demo.
    pickle_path, metadata_path = save_dataset(
        transitions, [episode], output_dir, test_only=False
    )
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata["runtime"] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
        "viewer": "mujoco.viewer.launch_passive",
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return pickle_path, metadata_path


def run_preflight(args: argparse.Namespace) -> int:
    import importlib.util
    import inspect
    import mujoco.viewer

    env = make_cable_training_env()
    viewer = None
    try:
        base = native_env(env)
        print(f"VIEWER_API: {inspect.signature(mujoco.viewer.launch_passive)}")
        print("VIEWER_TELEOP_INPUT: DISABLED")
        print(
            "PYNPUT_IMPORT: "
            f"{'PRESENT' if importlib.util.find_spec('pynput') is not None else 'MISSING'}"
        )
        print(f"MODEL_DATA_EXPOSED: {base.model is not None and base.data is not None}")
        observation, reset_info = env.reset()
        display_available = bool(os.environ.get("DISPLAY"))
        if display_available:
            viewer = mujoco.viewer.launch_passive(base.model, base.data)
            viewer.sync()
            viewer_status = "PASS"
        else:
            viewer_status = "OPERATOR_VIEWER_PREFLIGHT_REQUIRED"
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if viewer is not None and viewer.is_running():
            viewer.sync()
        print("PREFLIGHT_ENV_CREATE: PASS")
        print(f"PREFLIGHT_RESET: PASS ({type(observation).__name__}, info={sorted(reset_info)})")
        print(
            "PREFLIGHT_STEP: PASS "
            f"(reward={float(reward)}, terminated={terminated}, truncated={truncated}, "
            f"success={bool(info.get('success', False))})"
        )
        print(f"PREFLIGHT_NEXT_OBS_KEYS: {list(next_observation)}")
        print(f"VIEWER_CREATE: {viewer_status}")
        print("FORMAL_DEMO_SAVED: NO")
        return 0
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        print("PREFLIGHT_CLOSE: PASS")


def run_interactive(args: argparse.Namespace) -> int:
    import mujoco.viewer

    if not os.environ.get("DISPLAY"):
        print("RESULT: OPERATOR_KEYBOARD_VALIDATION_REQUIRED")
        print("ERROR: Viewer teleoperation requires a graphical X11 desktop session.")
        return 2

    try:
        from pynput import keyboard
    except Exception as exc:
        print("RESULT: PYNPUT_TELEOP_UNAVAILABLE")
        print(f"ERROR: pynput import failed: {type(exc).__name__}: {exc}")
        print("FORMAL_DEMO_SAVED: NO")
        return 3

    teleop = PressedKeyTeleop(keyboard)
    listener = keyboard.Listener(
        on_press=teleop.on_press,
        on_release=teleop.on_release,
    )
    try:
        listener.start()
        listener.wait()
        if not listener.running:
            raise RuntimeError("pynput listener stopped during startup")
    except Exception as exc:
        print("RESULT: PYNPUT_TELEOP_UNAVAILABLE")
        print(f"ERROR: pynput listener failed to start: {type(exc).__name__}: {exc}")
        print("FORMAL_DEMO_SAVED: NO")
        listener.stop()
        return 3

    env = None
    viewer = None
    successes = 0
    attempts = 0
    try:
        env = make_cable_training_env()
        base = native_env(env)
        viewer = mujoco.viewer.launch_passive(base.model, base.data)
        print("PYNPUT_LISTENER: RUNNING")
        print("IMPORTANT: Keep the terminal window focused while teleoperating.")
        print("IMPORTANT: Do not click the MuJoCo Viewer before using W/A/S/D/Q/E.")
        while successes < args.episodes and viewer.is_running():
            attempts += 1
            observation, reset_info = env.reset()
            viewer.sync()
            transitions: list[dict[str, Any]] = []
            episode_return = 0.0
            started = time.monotonic()
            last_info: dict[str, Any] = dict(reset_info)
            last_reward = 0.0
            discard_reason: str | None = None

            while len(transitions) < args.max_steps and viewer.is_running():
                loop_started = time.monotonic()
                pressed, reset_requested, force_print, end_requested, exit_requested = (
                    teleop.snapshot()
                )
                if exit_requested:
                    discard_reason = "operator_exit"
                    break
                if end_requested:
                    discard_reason = "manual_end_without_success"
                    break
                if reset_requested:
                    discard_reason = "manual_reset"
                    break
                action = action_from_pressed_keys(pressed, args.action_magnitude)
                action = np.clip(
                    action, env.action_space.low, env.action_space.high
                ).astype(np.float32)

                next_observation, reward, terminated, truncated, info = env.step(action)
                transitions.append(
                    make_transition(
                        observation, action, next_observation, reward, terminated, truncated
                    )
                )
                observation = next_observation
                last_info = dict(info)
                last_reward = float(reward)
                episode_return += last_reward
                step_number = len(transitions)
                viewer.sync()

                if force_print or step_number % args.print_every == 0:
                    print_state(
                        attempts,
                        step_number,
                        action,
                        observation,
                        last_info,
                        last_reward,
                        episode_return,
                    )

                if bool(info.get("success", False)):
                    duration = time.monotonic() - started
                    pickle_path, metadata_path = save_success_episode(
                        transitions,
                        successes + 1,
                        episode_return,
                        duration,
                        bool(terminated),
                        bool(truncated),
                        last_info,
                        args.output_dir,
                    )
                    successes += 1
                    print("SUCCESS EPISODE SAVED")
                    print(f"episode_id: cable_success_{successes:04d}")
                    print(f"steps: {len(transitions)}")
                    print(f"return: {episode_return}")
                    print(f"duration_seconds: {duration}")
                    print(f"final_distance_to_slot: {info['distance_to_slot']}")
                    print(f"environment_terminated: {bool(terminated)}")
                    print(f"environment_truncated: {bool(truncated)}")
                    print(f"pickle_path: {pickle_path}")
                    print(f"metadata_path: {metadata_path}")
                    break
                if terminated or truncated:
                    discard_reason = "terminated_without_success" if terminated else "timeout"
                    break

                remaining = float(base.control_dt) - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
            else:
                if not viewer.is_running():
                    discard_reason = "viewer_closed"
                elif len(transitions) >= args.max_steps:
                    discard_reason = "recorder_max_steps"

            if discard_reason is not None:
                log_path = write_discard_log(
                    args.discard_log_dir,
                    attempts,
                    discard_reason,
                    len(transitions),
                    episode_return,
                    time.monotonic() - started,
                    last_info,
                )
                print(f"DISCARDED: {discard_reason}; log={log_path}")
                if discard_reason in {"operator_exit", "viewer_closed"}:
                    break

        print(f"RESULT: {'PASS' if successes == args.episodes else 'STOPPED'}")
        print(f"SUCCESSFUL_EPISODES: {successes}/{args.episodes}")
        return 0 if successes == args.episodes else 1
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        listener.stop()
        listener.join(timeout=2.0)


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.preflight:
        return run_preflight(args)
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
