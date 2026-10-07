from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import mujoco
import numpy as np
from gym import spaces

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "cable_route.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])


class PandaCableRouteGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: float = 0.001,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = float(action_scale)

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_names = ("front", "handcam_rgb")
        self.image_obs = image_obs

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        # CableRoute task objects.
        self._base_body_id = self._model.body("base").id
        self._cable_root_body_id = self._model.body("cableB_first").id
        self._cable_last_body_id = self._model.body("cableB_last").id
        self._slot_site_id = self._model.site("slot_center").id

        # The first cable body has the free joint created by
        # <composite type="cable" initial="free">.
        self._cable_root_joint_id = self._model.body_jntadr[
            self._cable_root_body_id
        ]

        if (
            self._cable_root_joint_id < 0
            or self._model.jnt_type[self._cable_root_joint_id]
            != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError(
                "cableB_first does not have the expected free joint"
            )

        self._cable_root_qadr = self._model.jnt_qposadr[
            self._cable_root_joint_id
        ]
        self._cable_root_vadr = self._model.jnt_dofadr[
            self._cable_root_joint_id
        ]

        # V0 success threshold: cable tip close to slot center.
        self._success_threshold = 0.03

        state_space = gym.spaces.Dict(
            {
                "panda/tcp_pos": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "panda/tcp_vel": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "panda/gripper_pos": spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": state_space,
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(
                                    render_spec.height,
                                    render_spec.width,
                                    3,
                                ),
                                dtype=np.uint8,
                            ),
                            "wrist": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(
                                    render_spec.height,
                                    render_spec.width,
                                    3,
                                ),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {"state": state_space}
            )

        # CableRoute V0:
        # [dx, dy, dz]. Gripper is permanently closed.
        self.action_space = gym.spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
            width=640,
            height=480,
        )

        # Native MuJoCo renderer for RGB observations.
        # gymnasium 1.3.0 does not accept camera_id in
        # MujocoRenderer.render(), so camera switching is
        # handled with mujoco.Renderer.update_scene().
        self._rgb_renderer = mujoco.Renderer(
            self._model,
            height=render_spec.height,
            width=render_spec.width,
        )

      
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset Panda and attach the cable root to the gripper."""

        mujoco.mj_resetData(self._model, self._data)

        # ----------------------------------------------------
        # Panda -> HOME
        # ----------------------------------------------------
        for i, q in enumerate(_PANDA_HOME, start=1):
            jid = self._model.joint(f"joint{i}").id
            qadr = self._model.jnt_qposadr[jid]
            self._data.qpos[qadr] = q

        self._data.qvel[:] = 0.0

        mujoco.mj_forward(
            self._model,
            self._data,
        )

        # ----------------------------------------------------
        # cableB_first -> pinch
        # ----------------------------------------------------
        pinch_pos = self._data.site_xpos[
            self._pinch_site_id
        ].copy()

        base_quat = self._data.xquat[
            self._base_body_id
        ].copy()

        qadr = self._cable_root_qadr
        vadr = self._cable_root_vadr

        self._data.qpos[qadr:qadr + 3] = pinch_pos
        self._data.qpos[qadr + 3:qadr + 7] = base_quat
        self._data.qvel[vadr:vadr + 6] = 0.0

        mujoco.mj_forward(
            self._model,
            self._data,
        )

        # ----------------------------------------------------
        # Mocap target = current TCP
        # ----------------------------------------------------
        self._data.mocap_pos[0] = self._data.site_xpos[
            self._pinch_site_id
        ].copy()

        self._data.mocap_quat[0] = self._data.sensor(
            "2f85/pinch_quat"
        ).data.copy()

        # Robotiq control value 0 = closed in this setup.
        self._data.ctrl[self._gripper_ctrl_id] = 0.0

        mujoco.mj_forward(
            self._model,
            self._data,
        )

        # ----------------------------------------------------
        # Let the cable settle under gravity while holding TCP
        # ----------------------------------------------------
        for _ in range(1000):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )

            self._data.ctrl[self._panda_ctrl_ids] = tau
            self._data.ctrl[self._gripper_ctrl_id] = 0.0

            mujoco.mj_step(
                self._model,
                self._data,
            )

        # The settling phase is part of reset, not the episode.
        # Remove residual cable/robot velocity and restart the
        # episode clock from zero.
        self._data.qvel[:] = 0.0
        self._data.time = 0.0

        mujoco.mj_forward(
            self._model,
            self._data,
        )

        obs = self._compute_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[
        Dict[str, np.ndarray],
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:
        """Execute one CableRoute V0 control step."""

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (3,):
            raise ValueError(
                f"Expected action shape (3,), got {action.shape}"
            )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # ----------------------------------------------------
        # xyz delta -> mocap target
        # ----------------------------------------------------
        pos = self._data.mocap_pos[0].copy()

        dpos = action * self._action_scale

        npos = np.clip(
            pos + dpos,
            *_CARTESIAN_BOUNDS,
        )

        self._data.mocap_pos[0] = npos

        # Gripper stays closed for the entire CableRoute task.
        self._data.ctrl[self._gripper_ctrl_id] = 0.0

        # ----------------------------------------------------
        # Operational-space control
        # ----------------------------------------------------
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )

            self._data.ctrl[self._panda_ctrl_ids] = tau
            self._data.ctrl[self._gripper_ctrl_id] = 0.0

            mujoco.mj_step(
                self._model,
                self._data,
            )

        obs = self._compute_observation()
        info = self._get_info()
        reward = self._compute_reward()

        # Successful routing ends the episode.
        terminated = bool(info["success"])

        # Time limit is a truncation, not task success.
        truncated = bool(
            self.time_limit_exceeded()
        )

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        rendered_frames = []

        for camera_name in self.camera_names:
            self._rgb_renderer.update_scene(
                self._data,
                camera=camera_name,
            )

            # copy() is important because MuJoCo may reuse
            # its internal render buffer.
            frame = self._rgb_renderer.render().copy()
            rendered_frames.append(frame)

        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {
            "state": {}
        }

        tcp_pos = self._data.sensor(
            "2f85/pinch_pos"
        ).data

        obs["state"]["panda/tcp_pos"] = (
            tcp_pos.astype(np.float32)
        )

        tcp_vel = self._data.sensor(
            "2f85/pinch_vel"
        ).data

        obs["state"]["panda/tcp_vel"] = (
            tcp_vel.astype(np.float32)
        )

        gripper_pos = np.array(
            [
                self._data.ctrl[
                    self._gripper_ctrl_id
                ] / 255.0
            ],
            dtype=np.float32,
        )

        obs["state"]["panda/gripper_pos"] = (
            gripper_pos
        )

        if self.image_obs:
            obs["images"] = {}

            (
                obs["images"]["front"],
                obs["images"]["wrist"],
            ) = self.render()

        if self.render_mode == "human":
            self._viewer.render(
                self.render_mode
            )

        return obs

    def _check_success(self) -> bool:
        cable_tip = self._data.xpos[
            self._cable_last_body_id
        ]

        slot_pos = self._data.site_xpos[
            self._slot_site_id
        ]

        distance = np.linalg.norm(
            cable_tip - slot_pos
        )

        return bool(
            distance < self._success_threshold
        )

    def _get_info(self) -> Dict[str, Any]:
        cable_tip = self._data.xpos[
            self._cable_last_body_id
        ].copy()

        slot_pos = self._data.site_xpos[
            self._slot_site_id
        ].copy()

        pinch_pos = self._data.site_xpos[
            self._pinch_site_id
        ]

        cable_root = self._data.xpos[
            self._cable_root_body_id
        ]

        distance = float(
            np.linalg.norm(
                cable_tip - slot_pos
            )
        )

        root_error = float(
            np.linalg.norm(
                pinch_pos - cable_root
            )
        )

        return {
            "success": bool(
                distance < self._success_threshold
            ),
            "distance_to_slot": distance,
            "root_error": root_error,
            "cable_tip_pos": cable_tip.astype(
                np.float32
            ),
            "slot_pos": slot_pos.astype(
                np.float32
            ),
        }

    def _compute_reward(self) -> float:
        info = self._get_info()

        distance = info[
            "distance_to_slot"
        ]

        # Dense debugging reward for the first simulation
        # version. Later we can replace this with the
        # SERL binary reward classifier if desired.
        reward = np.exp(
            -10.0 * distance
        )

        if info["success"]:
            reward += 5.0

        return float(reward)

if __name__ == "__main__":
    env = PandaCableRouteGymEnv(
        render_mode="human"
    )

    obs, info = env.reset()

    print("reset info:", info)
    print("action space:", env.action_space)

    for _ in range(100):
        action = (
            env.action_space.sample()
            * 0.1
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            obs, info = env.reset()
