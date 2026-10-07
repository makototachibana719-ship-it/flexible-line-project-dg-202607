"""SERL 状态版 PegInsertion 环境适配器。"""

from __future__ import annotations

import numpy as np

try:
    import gym
except ImportError:
    import gymnasium as gym

from mujoco_env.mujoco_env.envs.env_instance.aubo_i5_assemble_hole import (
    make_aubo_i5_assemble_hole_env,
)
from mujoco_env.mujoco_env.utils import transform as T


class PegStateSERLEnv(gym.Env):
    """
    将 PegInsertion 的 8 维绝对位姿动作，
    转换为适合 SERL 的 7 维归一化增量动作。
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ]
    }

    def __init__(
        self,
        render_mode: str = "rgb_array",
        max_episode_steps: int = 1000,
        translation_scale: float = 0.002,
        rotation_scale: float = np.deg2rad(1.0),
    ) -> None:
        super().__init__()

        self.env = (
            make_aubo_i5_assemble_hole_env(
                scene_name="assemble_hole",
                include_image=False,
                include_depth=False,
                control_dt=0.02,
                physics_dt=0.002,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
            )
        )

        self.translation_scale = float(
            translation_scale
        )
        self.rotation_scale = float(
            rotation_scale
        )

        # dx,dy,dz,droll,dpitch,dyaw,gripper
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # tcp_pos 3
        # tcp_quat 4
        # wrench 6
        # relative_goal_pos 3
        # goal_quat 4
        # 总计 20
        self.observation_space = (
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20,),
                dtype=np.float32,
            )
        )

        self._target_pose = np.zeros(
            7,
            dtype=np.float32,
        )

        self._workspace_low = np.full(
            3,
            -np.inf,
            dtype=np.float32,
        )

        self._workspace_high = np.full(
            3,
            np.inf,
            dtype=np.float32,
        )

    @staticmethod
    def _state(obs: dict) -> np.ndarray:
        tcp_pos = np.asarray(
            obs["tcp_pos"],
            dtype=np.float32,
        )[:3]

        tcp_quat = np.asarray(
            obs["tcp_quat"],
            dtype=np.float32,
        )[:4]

        wrench = np.asarray(
            obs.get(
                "wrench",
                np.zeros(6),
            ),
            dtype=np.float32,
        )[:6]

        goal = np.asarray(
            obs["desired_goal"],
            dtype=np.float32,
        )

        relative_goal_pos = (
            goal[:3] - tcp_pos
        )

        state = np.concatenate(
            [
                tcp_pos,
                tcp_quat,
                wrench,
                relative_goal_pos,
                goal[3:7],
            ]
        )

        return state.astype(np.float32)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        self._target_pose = np.concatenate(
            [
                np.asarray(
                    obs["tcp_pos"]
                )[:3],
                np.asarray(
                    obs["tcp_quat"]
                )[:4],
            ]
        ).astype(np.float32)

        start = np.asarray(
            obs["tcp_pos"],
            dtype=np.float32,
        )[:3]

        goal = np.asarray(
            obs["desired_goal"],
            dtype=np.float32,
        )[:3]

        margin = np.array(
            [0.05, 0.05, 0.05],
            dtype=np.float32,
        )

        self._workspace_low = (
            np.minimum(start, goal)
            - margin
        )

        self._workspace_high = (
            np.maximum(start, goal)
            + margin
        )

        return self._state(obs), info

    def step(self, action):
        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

        self._target_pose[:3] += (
            action[:3]
            * self.translation_scale
        )

        self._target_pose[:3] = np.clip(
            self._target_pose[:3],
            self._workspace_low,
            self._workspace_high,
        )

        current_matrix = T.quat_2_mat(
            self._target_pose[3:7]
        )

        delta_matrix = T.euler_2_mat(
            action[3:6]
            * self.rotation_scale
        )

        self._target_pose[3:7] = (
            T.mat_2_quat(
                current_matrix
                @ delta_matrix
            )
        )

        gripper = (
            1.0
            if action[6] > 0.0
            else 0.0
        )

        absolute_action = np.concatenate(
            [
                self._target_pose,
                np.array(
                    [gripper],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            absolute_action
        )

        info = dict(info)
        info["terminated"] = bool(
            terminated
        )
        info["truncated"] = bool(
            truncated
        )
        info["is_success"] = bool(
            terminated and not truncated
        )

        metrics = getattr(
            self.env.task,
            "last_metrics",
            None,
        )

        if isinstance(metrics, dict):
            info.update(metrics)

        return (
            self._state(obs),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_peg_state_env(**kwargs):
    return PegStateSERLEnv(**kwargs)
