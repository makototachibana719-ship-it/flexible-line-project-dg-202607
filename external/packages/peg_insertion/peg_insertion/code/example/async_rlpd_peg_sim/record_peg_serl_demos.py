from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco.viewer as mj_viewer
from pynput import keyboard

from peg_env_adapter import make_peg_state_env


class KeyboardTeleop:
    """
    直接生成 PegStateSERLEnv 使用的 7D 归一化动作：

    [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """

    def __init__(
        self,
        translation_step_normalized: float,
        rotation_step_normalized: float,
    ):
        self.translation_step = float(
            translation_step_normalized
        )
        self.rotation_step = float(
            rotation_step_normalized
        )

        # 1=粗调，2=正常，3=精调
        self.speed_scale = 1.0

        self.pending = np.zeros(
            6,
            dtype=np.float32,
        )

        self.ctrl = False
        self.shift = False

        # -1 = open
        # +1 = closed
        self.gripper = -1.0
        self.gripper_dirty = False

        self.abort_episode = False
        self.exit_requested = False

        self.lock = threading.Lock()
        self.listener = None

    def _add(self, index, value):
        with self.lock:
            self.pending[index] += (
                value * self.speed_scale
            )

    def on_press(self, key):
        try:
            ctrl_keys = {
                keyboard.Key.ctrl,
                keyboard.Key.ctrl_l,
                keyboard.Key.ctrl_r,
            }

            shift_keys = {
                keyboard.Key.shift,
                keyboard.Key.shift_l,
                keyboard.Key.shift_r,
            }

            if key in ctrl_keys:
                self.ctrl = True
                return

            if key in shift_keys:
                self.shift = True
                return

            # ---------- 平移 / 旋转 ----------
            if key == keyboard.Key.up:
                if self.ctrl and self.shift:
                    self._add(
                        5,
                        +self.rotation_step,
                    )
                elif self.ctrl:
                    self._add(
                        2,
                        +self.translation_step,
                    )
                elif self.shift:
                    self._add(
                        4,
                        -self.rotation_step,
                    )
                else:
                    self._add(
                        0,
                        +self.translation_step,
                    )

            elif key == keyboard.Key.down:
                if self.ctrl and self.shift:
                    self._add(
                        5,
                        -self.rotation_step,
                    )
                elif self.ctrl:
                    self._add(
                        2,
                        -self.translation_step,
                    )
                elif self.shift:
                    self._add(
                        4,
                        +self.rotation_step,
                    )
                else:
                    self._add(
                        0,
                        -self.translation_step,
                    )

            elif key == keyboard.Key.left:
                if self.shift:
                    self._add(
                        3,
                        +self.rotation_step,
                    )
                else:
                    self._add(
                        1,
                        +self.translation_step,
                    )

            elif key == keyboard.Key.right:
                if self.shift:
                    self._add(
                        3,
                        -self.rotation_step,
                    )
                else:
                    self._add(
                        1,
                        -self.translation_step,
                    )

            char = getattr(
                key,
                "char",
                None,
            )

            if char:
                char = char.lower()

                if char == "1":
                    self.speed_scale = 5.0
                    print(
                        "[SPEED] 粗调 x5"
                    )

                elif char == "2":
                    self.speed_scale = 1.0
                    print(
                        "[SPEED] 中调 x1"
                    )

                elif char == "3":
                    self.speed_scale = 0.25
                    print(
                        "[SPEED] 精调 x0.25"
                    )

                elif char == "o":
                    with self.lock:
                        self.gripper = -1.0
                        self.gripper_dirty = True
                    print("[KEY] gripper OPEN")

                elif char == "p":
                    with self.lock:
                        self.gripper = +1.0
                        self.gripper_dirty = True
                    print("[KEY] gripper CLOSE")

                elif char == "r":
                    self.abort_episode = True
                    print(
                        "[KEY] 当前 Episode 作废并重新 reset"
                    )

        except Exception as exc:
            print("[WARN] keyboard:", exc)

    def on_release(self, key):
        ctrl_keys = {
            keyboard.Key.ctrl,
            keyboard.Key.ctrl_l,
            keyboard.Key.ctrl_r,
        }

        shift_keys = {
            keyboard.Key.shift,
            keyboard.Key.shift_l,
            keyboard.Key.shift_r,
        }

        if key in ctrl_keys:
            self.ctrl = False

        if key in shift_keys:
            self.shift = False

        if key == keyboard.Key.esc:
            self.exit_requested = True
            print("[KEY] ESC：结束 Demo 录制")
            return False

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()

    def clear_abort(self):
        self.abort_episode = False

    def consume_action(self):
        with self.lock:
            movement = self.pending.copy()
            self.pending[:] = 0.0

            gripper_changed = (
                self.gripper_dirty
            )
            self.gripper_dirty = False

            gripper = float(
                self.gripper
            )

        active = bool(
            np.any(
                np.abs(movement) > 1e-8
            )
            or gripper_changed
        )

        action = np.zeros(
            7,
            dtype=np.float32,
        )

        action[:6] = movement
        action[6] = gripper

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        return action, active


def save_demo(
    save_dir: Path,
    demo_index: int,
    data: dict,
    metadata: dict,
):
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        save_dir
        / f"demo_{demo_index:03d}.npz"
    )

    arrays = {}

    for key in [
        "observations",
        "actions",
        "next_observations",
    ]:
        arrays[key] = np.asarray(
            data[key],
            dtype=np.float32,
        )

    arrays["rewards"] = np.asarray(
        data["rewards"],
        dtype=np.float32,
    )

    arrays["masks"] = np.asarray(
        data["masks"],
        dtype=np.float32,
    )

    arrays["dones"] = np.asarray(
        data["dones"],
        dtype=np.bool_,
    )

    arrays["terminated"] = np.asarray(
        data["terminated"],
        dtype=np.bool_,
    )

    arrays["truncated"] = np.asarray(
        data["truncated"],
        dtype=np.bool_,
    )

    for key in [
        "position_error",
        "lateral_error",
        "axial_error",
        "orientation_error",
        "force_norm",
    ]:
        arrays[key] = np.asarray(
            data[key],
            dtype=np.float32,
        )

    arrays["initial_observation"] = np.asarray(
        metadata["initial_observation"],
        dtype=np.float32,
    )

    metadata_to_save = dict(metadata)
    metadata_to_save.pop(
        "initial_observation"
    )

    arrays["metadata_json"] = np.array(
        json.dumps(
            metadata_to_save,
            ensure_ascii=False,
        )
    )

    np.savez_compressed(
        path,
        **arrays,
    )

    index_path = (
        save_dir
        / "index.jsonl"
    )

    with index_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                {
                    "file": path.name,
                    **metadata_to_save,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    return path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-demos",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(
            Path.home()
            / "serl_data"
            / "peg_demos_test"
        ),
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--translation-step-mm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=0.057,
    )

    args = parser.parse_args()

    env = make_peg_state_env(
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )

    raw_env = env.env

    translation_step_normalized = (
        args.translation_step_mm
        / 1000.0
        / env.translation_scale
    )

    rotation_step_normalized = (
        np.deg2rad(
            args.rotation_step_deg
        )
        / env.rotation_scale
    )

    print(
        "translation normalized step:",
        translation_step_normalized,
    )

    print(
        "rotation normalized step:",
        rotation_step_normalized,
    )

    print()
    print("控制方式：")
    print("  ↑ ↓       : X +/-")
    print("  ← →       : Y +/-")
    print("  Ctrl+↑/↓  : Z +/-")
    print("  Shift+方向: 姿态")
    print("  1          : 粗调 x5")
    print("  2          : 中调 x1")
    print("  3          : 精调 x0.25")
    print("  O/P        : 夹爪")
    print("  R          : 当前 Episode 作废重来")
    print("  ESC        : 结束录制")
    print()

    teleop = KeyboardTeleop(
        translation_step_normalized,
        rotation_step_normalized,
    )

    teleop.start()

    save_dir = Path(
        args.save_dir
    ).expanduser()

    saved_count = 0
    attempt = 0
    stop_all = False

    try:
        with mj_viewer.launch_passive(
            raw_env.model,
            raw_env.data,
        ) as viewer:

            while (
                saved_count < args.num_demos
                and viewer.is_running()
                and not stop_all
            ):
                attempt += 1

                obs, info = env.reset()

                initial_obs = (
                    np.asarray(
                        obs,
                        dtype=np.float32,
                    ).copy()
                )

                try:
                    hole_center = (
                        raw_env.data
                        .site("hole_center")
                        .xpos
                        .copy()
                        .tolist()
                    )
                except Exception:
                    hole_center = None

                print()
                print(
                    "=" * 60
                )
                print(
                    f"Attempt {attempt}"
                    f" | 已保存 "
                    f"{saved_count}/{args.num_demos}"
                )
                print(
                    "请完成一次正确插入。"
                )
                print(
                    "若操作失误按 R 作废重来。"
                )

                data = {
                    "observations": [],
                    "actions": [],
                    "next_observations": [],
                    "rewards": [],
                    "masks": [],
                    "dones": [],
                    "terminated": [],
                    "truncated": [],
                    "position_error": [],
                    "lateral_error": [],
                    "axial_error": [],
                    "orientation_error": [],
                    "force_norm": [],
                }

                full_return = 0.0
                success = False
                aborted = False

                # SERL_DEMO_RECORDER_V2_ACTIVE_ONLY
                # env_step 现在代表真实控制动作数量，
                # 人工观察/思考时间不计入 Episode 步数。
                env_step = 0

                while env_step < args.max_episode_steps:
                    if teleop.exit_requested:
                        stop_all = True
                        break

                    if teleop.abort_episode:
                        teleop.clear_abort()
                        aborted = True
                        print(
                            "[REJECT] 当前 Episode 已作废"
                        )
                        break

                    action, active = (
                        teleop.consume_action()
                    )

                    # 人工正在观察、思考时：
                    # 不执行环境动作，也不消耗 Episode 步数。
                    if not active:
                        viewer.sync()
                        time.sleep(0.02)
                        continue

                    current_obs = (
                        np.asarray(
                            obs,
                            dtype=np.float32,
                        ).copy()
                    )

                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    next_obs = np.asarray(
                        next_obs,
                        dtype=np.float32,
                    )

                    terminated = bool(
                        terminated
                    )
                    truncated = bool(
                        truncated
                    )

                    episode_done = (
                        terminated
                        or truncated
                    )

                    full_return += float(
                        reward
                    )

                    # 不保存长时间停顿产生的 idle transition。
                    # 但终止 transition 必须保存。
                    if active or episode_done:
                        data[
                            "observations"
                        ].append(
                            current_obs
                        )

                        data["actions"].append(
                            action.copy()
                        )

                        data[
                            "next_observations"
                        ].append(
                            next_obs.copy()
                        )

                        data["rewards"].append(
                            float(reward)
                        )

                        data["masks"].append(
                            0.0
                            if terminated
                            else 1.0
                        )

                        data["dones"].append(
                            episode_done
                        )

                        data[
                            "terminated"
                        ].append(
                            terminated
                        )

                        data[
                            "truncated"
                        ].append(
                            truncated
                        )

                        for key in [
                            "position_error",
                            "lateral_error",
                            "axial_error",
                            "orientation_error",
                            "force_norm",
                        ]:
                            value = info.get(
                                key,
                                np.nan,
                            )

                            data[key].append(
                                float(value)
                            )

                    obs = next_obs

                    # 只有真正执行一次动作才 +1
                    env_step += 1

                    # 每 10 个真实动作输出一次诊断数据
                    if (
                        env_step == 1
                        or env_step % 10 == 0
                        or terminated
                        or truncated
                    ):
                        print(
                            "[METRIC] "
                            f"step={env_step} "
                            f"pos={info.get('position_error')} "
                            f"lat={info.get('lateral_error')} "
                            f"axial={info.get('axial_error')} "
                            f"ori={info.get('orientation_error')} "
                            f"force={info.get('force_norm')}"
                        )

                    viewer.sync()

                    time.sleep(
                        raw_env.control_dt
                    )

                    if terminated:
                        success = True

                        print(
                            "[SUCCESS] "
                            f"环境在 step={env_step} "
                            "判定插入成功"
                        )

                        break

                    if truncated:
                        print(
                            "[REJECT] "
                            "达到最大 Episode 步数，"
                            "本条不保存"
                        )
                        break

                if stop_all:
                    break

                if aborted:
                    continue

                if not success:
                    continue

                if len(
                    data["observations"]
                ) == 0:
                    print(
                        "[REJECT] 没有有效 transition"
                    )
                    continue

                final_info = dict(info)

                metadata = {
                    "created_at": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                    "task": (
                        "AuboPegInsert-State-v0"
                    ),
                    "attempt": attempt,
                    "success": True,
                    "full_env_steps": (
                        env_step
                    ),
                    "recorded_transitions": len(
                        data["actions"]
                    ),
                    "episode_return": (
                        float(full_return)
                    ),
                    "observation_shape": [20],
                    "action_shape": [7],
                    "translation_scale": float(
                        env.translation_scale
                    ),
                    "rotation_scale": float(
                        env.rotation_scale
                    ),
                    "control_dt": float(
                        raw_env.control_dt
                    ),
                    "position_tolerance": float(
                        raw_env.task
                        .position_tolerance
                    ),
                    "orientation_tolerance": float(
                        raw_env.task
                        .orientation_tolerance
                    ),
                    "hole_center": hole_center,
                    "final_position_error": float(
                        final_info.get(
                            "position_error",
                            np.nan,
                        )
                    ),
                    "final_lateral_error": float(
                        final_info.get(
                            "lateral_error",
                            np.nan,
                        )
                    ),
                    "final_axial_error": float(
                        final_info.get(
                            "axial_error",
                            np.nan,
                        )
                    ),
                    "final_orientation_error": float(
                        final_info.get(
                            "orientation_error",
                            np.nan,
                        )
                    ),
                    "final_force_norm": float(
                        final_info.get(
                            "force_norm",
                            np.nan,
                        )
                    ),
                    "initial_observation": (
                        initial_obs
                    ),
                }

                output = save_demo(
                    save_dir,
                    saved_count,
                    data,
                    metadata,
                )

                print(
                    "[SAVED]",
                    output,
                )

                print(
                    "transitions:",
                    len(data["actions"]),
                )

                print(
                    "return:",
                    full_return,
                )

                print(
                    "final metrics:",
                    {
                        key: final_info.get(
                            key
                        )
                        for key in [
                            "lateral_error",
                            "axial_error",
                            "orientation_error",
                            "force_norm",
                        ]
                    },
                )

                saved_count += 1

    finally:
        teleop.stop()
        env.close()

    print()
    print(
        f"[DONE] 成功保存 "
        f"{saved_count}/{args.num_demos} 条 Demo"
    )

    print(
        "目录:",
        save_dir,
    )


if __name__ == "__main__":
    main()
