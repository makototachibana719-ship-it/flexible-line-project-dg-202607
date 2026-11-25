"""
轴孔装配任务（Peg-in-Hole）

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Dict, Any, Optional
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from gymnasium import spaces
from mujoco_env.tasks.base_task import BaseTask
from mujoco_env.robot_config.aubo_i5 import AuboI5Robot

import pdb

class PegInsertionTask(BaseTask):
    """
    轴孔装配任务（Peg-in-Hole）
    
    目标：将圆柱形的轴（peg）精确插入到孔（hole）中
    成功条件：轴完全插入孔中，位置和姿态都在容差范围内
    
    这是一个经典的机器人装配任务，需要高精度的位置和力控制。
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        success_threshold: float = 0.002,  # 2mm
        position_tolerance: float = 0.002,
        orientation_tolerance: float = 0.05,  # 约2.9度
        peg_radius: float = 0.015,  # 轴半径 15mm（根据实际装配需求调整）
        hole_radius: float = 0.016,  # 孔半径 16mm（与STL检测结果匹配）
        insertion_depth: float = 0.05,  # 插入深度 50mm
        ee_site_name: str = "grip_site",  # 末端执行器 site 名称
        render_dt: float = 0.016, 
        **kwargs
    ):
        """
        初始化轴孔装配任务
        
        Args:
            max_episode_steps: 最大步数
            success_threshold: 成功阈值
            position_tolerance: 位置容差
            orientation_tolerance: 姿态容差（弧度）
            peg_radius: 轴的半径
            hole_radius: 孔的半径
            insertion_depth: 插入深度
            **kwargs: 额外参数
        """
        super().__init__(
            name="PegInsertion",
            max_episode_steps=max_episode_steps,
            success_threshold=success_threshold,
            **kwargs
        )
        
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.peg_radius = peg_radius
        self.hole_radius = hole_radius
        self.insertion_depth = insertion_depth
        self.clearance = hole_radius - peg_radius
        self.ee_site_name = ee_site_name
        self.render_dt = render_dt
        self.arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        self.init_qpos = AuboI5Robot.INIT_QPOS
        
        # 孔的位置范围（用于动态场景）
        self.hole_position_range = np.array([
            [0.3, 0.6],   # x: 300-600mm
            [-0.3, 0.3],  # y: -300-300mm
            [0.05, 0.2],  # z: 50-200mm
        ])

        # 从场景文件中读取孔信息
        self.hole_position = None
        self.hole_orientation = None
        if hasattr(self, 'scene_name') and self.scene_name:
            self._load_hole_info_from_scene()

    def _load_hole_info_from_scene(self):
        """
        从场景XML文件中读取孔的位置和方向信息
        """
        try:
            # 构建场景文件路径
            scene_file = Path(__file__).parent.parent.parent / "assets" / "scenes" / f"{self.scene_name}.xml"

            if not scene_file.exists():
                print(f"Warning: Scene file {scene_file} not found, using default hole info")
                return

            # 解析XML文件
            tree = ET.parse(scene_file)
            root = tree.getroot()

            # 查找孔body
            hole_body = None
            for body in root.findall(".//body"):
                if body.get('name') == 'hole':
                    hole_body = body
                    break

            if hole_body is None:
                print("Warning: Hole body not found in scene file, using default hole info")
                return

            # 读取孔位置
            pos_str = hole_body.get('pos')
            if pos_str:
                pos_values = [float(x) for x in pos_str.split()]
                self.hole_position = np.array(pos_values)
                print(f"Loaded hole position from scene: {self.hole_position}")

            # 读取孔方向（euler角度）
            euler_str = hole_body.get('euler')
            if euler_str:
                euler_values = [float(x) for x in euler_str.split()]
                # 将euler角度转换为四元数
                # 这里使用简化的转换，实际应该使用更精确的方法
                if len(euler_values) >= 3:
                    # 绕x轴旋转theta的四元数: [cos(theta/2), sin(theta/2), 0, 0]
                    theta = euler_values[0]  # x轴旋转
                    qw = np.cos(theta / 2)
                    qx = np.sin(theta / 2)
                    qy = 0.0
                    qz = 0.0
                    self.hole_orientation = np.array([qw, qx, qy, qz])
                    print(f"Loaded hole orientation from scene: {self.hole_orientation} (from euler: {euler_values})")

        except Exception as e:
            print(f"Warning: Failed to load hole info from scene file: {e}")
            self.hole_position = None
            self.hole_orientation = None
    
    def obs_finalize(self, obs_dict): 
        keys_to_remove = ["qpos", "qvel", "achieved_goal"]
        for key in keys_to_remove: 
            if key in obs_dict.keys():
                del obs_dict[key]
        return obs_dict

    def set_initial_state(self, data, model, options: Optional[Dict] = None): 
        random = False
        if options and options.get("random", False):
            random = True

        if random: 
            # 在初始位置附近添加随机噪声
            # init_qpos[2] 是 foreArm_joint (肘关节)，需要保证其角度合理（例如防止自碰撞或奇异点）
            # 这里硬编码了 < 1.0 的检查，建议根据实际机器人关节限制调整
            while True:
                noise = np.random.normal(loc=0, scale=0.1, size=self.init_qpos.shape)
                init_qpos_rand = self.init_qpos + noise
                if init_qpos_rand[2] >= 1.0: # 假设 1.0 是一个安全阈值
                     break
                # 或者如果原本就是为了 < 1.0 (原代码逻辑)，则保持原样:
                # 原代码: while init_qpos_rand[2] < 1.0: ...
                # 这意味着必须 >= 1.0 才能跳出循环？
                # 让我们仔细看原代码：while init_qpos_rand[2] < 1.0: 重新采样
                # 这意味着它希望 init_qpos_rand[2] >= 1.0
        else:
            init_qpos_rand = self.init_qpos

        for i, name in enumerate(self.arm_joint_names):
            try:
                joint_id = model.joint(name).id
                qpos_adr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_adr] = init_qpos_rand[i]
                # 同时重置速度
                qvel_adr = model.jnt_dofadr[joint_id]
                data.qvel[qvel_adr] = 0
            except Exception as e:
                print(f"Warning: Could not set joint {name}: {e}")
        
        # 只有在位置控制模式下才需要设置 ctrl
        # 假设前6个 actuator 对应 arm joints
        if model.nu >= len(self.arm_joint_names):
             data.ctrl[:len(self.arm_joint_names)] = init_qpos_rand[:len(self.arm_joint_names)]
             
        data.qacc[:] = 0

    # SERL_PEG_TASK_PATCH_V1

    @staticmethod
    def _quat_angle_error(
        current_quat: np.ndarray,
        target_quat: np.ndarray,
    ) -> float:
        """返回两个四元数之间的最小夹角，单位为弧度。"""
        current = np.asarray(
            current_quat,
            dtype=np.float64,
        )[:4]
        target = np.asarray(
            target_quat,
            dtype=np.float64,
        )[:4]

        current_norm = np.linalg.norm(current)
        target_norm = np.linalg.norm(target)

        if current_norm < 1e-8 or target_norm < 1e-8:
            return float(np.pi)

        current = current / current_norm
        target = target / target_norm

        dot = np.clip(
            abs(float(np.dot(current, target))),
            0.0,
            1.0,
        )

        return float(2.0 * np.arccos(dot))

    def _get_site_pose(self, site_name: str):
        """读取运行时 site 的世界坐标位置与四元数。"""
        if (
            self.env is None
            or not hasattr(self.env, "model")
            or not hasattr(self.env, "data")
        ):
            return None, None

        site_id = mujoco.mj_name2id(
            self.env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id < 0:
            return None, None

        position = (
            self.env.data.site_xpos[site_id]
            .copy()
            .astype(np.float32)
        )

        quaternion = np.zeros(
            4,
            dtype=np.float64,
        )

        mujoco.mju_mat2Quat(
            quaternion,
            self.env.data.site_xmat[site_id],
        )

        return position, quaternion.astype(np.float32)

    def _update_insertion_axis(
        self,
        target_position: np.ndarray,
    ) -> None:
        """
        用 hole body 原点到 hole_center 的方向
        估计实际插入轴。
        """
        axis = None

        if (
            self.env is not None
            and hasattr(self.env, "model")
        ):
            body_id = mujoco.mj_name2id(
                self.env.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "hole",
            )

            if body_id >= 0:
                body_position = (
                    self.env.data.xpos[body_id]
                    .copy()
                )

                candidate = (
                    np.asarray(target_position)
                    - body_position
                )

                candidate_norm = np.linalg.norm(
                    candidate
                )

                if candidate_norm > 1e-8:
                    axis = (
                        candidate
                        / candidate_norm
                    )

        if axis is None:
            axis = np.array(
                [-1.0, 0.0, 0.0],
                dtype=np.float64,
            )

        self._insertion_axis = np.asarray(
            axis,
            dtype=np.float64,
        )

    def _goal_metrics(
        self,
        obs: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        desired_goal = obs.get("desired_goal")

        if desired_goal is None:
            desired_goal = self.sample_goal()

        tcp_pos = np.asarray(
            obs.get(
                "tcp_pos",
                np.zeros(3),
            ),
            dtype=np.float64,
        )[:3]

        tcp_quat = np.asarray(
            obs.get(
                "tcp_quat",
                np.array(
                    [1.0, 0.0, 0.0, 0.0]
                ),
            ),
            dtype=np.float64,
        )[:4]

        desired_goal = np.asarray(
            desired_goal,
            dtype=np.float64,
        )

        target_pos = desired_goal[:3]
        target_quat = desired_goal[3:7]

        axis = np.asarray(
            getattr(
                self,
                "_insertion_axis",
                np.array(
                    [-1.0, 0.0, 0.0]
                ),
            ),
            dtype=np.float64,
        )

        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-8:
            axis = axis / axis_norm
        else:
            axis = np.array(
                [-1.0, 0.0, 0.0],
                dtype=np.float64,
            )

        error_vector = tcp_pos - target_pos

        axial_signed = float(
            np.dot(error_vector, axis)
        )

        axial_error = abs(axial_signed)

        lateral_vector = (
            error_vector
            - axial_signed * axis
        )

        lateral_error = float(
            np.linalg.norm(lateral_vector)
        )

        position_error = float(
            np.linalg.norm(error_vector)
        )

        orientation_error = (
            self._quat_angle_error(
                tcp_quat,
                target_quat,
            )
        )

        wrench = obs.get("wrench")
        force_norm = 0.0

        if wrench is not None:
            force_norm = float(
                np.linalg.norm(
                    np.asarray(
                        wrench,
                        dtype=np.float64,
                    )[:3]
                )
            )

        return {
            "position_error": position_error,
            "lateral_error": lateral_error,
            "axial_error": axial_error,
            "orientation_error": orientation_error,
            "force_norm": force_norm,
        }

    def compute_reward_from_obs(
        self,
        obs: Dict[str, np.ndarray],
    ) -> float:
        """
        状态版第一阶段奖励：
        1. 横向对准；
        2. 沿插入轴接近；
        3. 保持目标姿态；
        4. 成功时给予额外奖励。
        """
        metrics = self._goal_metrics(obs)

        reward = (
            -80.0 * metrics["lateral_error"]
            -20.0 * metrics["axial_error"]
            -2.0 * metrics["orientation_error"]
        )

        # 横向基本对准后给予阶段奖励
        if metrics["lateral_error"] < 0.01:
            reward += 2.0

        success = (
            metrics["lateral_error"]
            <= self.position_tolerance
            and metrics["axial_error"]
            <= 1.5 * self.position_tolerance
            and metrics["orientation_error"]
            <= self.orientation_tolerance
        )

        if success:
            reward += 25.0

        self.last_metrics = dict(metrics)
        self.last_metrics["is_success"] = bool(
            success
        )

        return float(reward)

    def is_success_from_obs(
        self,
        obs: Dict[str, np.ndarray],
    ) -> bool:
        """
        hole_center 作为最终目标点，同时检查：
        横向误差、插入轴误差和姿态误差。
        """
        metrics = self._goal_metrics(obs)

        success = (
            metrics["lateral_error"]
            <= self.position_tolerance
            and metrics["axial_error"]
            <= 1.5 * self.position_tolerance
            and metrics["orientation_error"]
            <= self.orientation_tolerance
        )

        self.last_metrics = dict(metrics)
        self.last_metrics["is_success"] = bool(
            success
        )

        return bool(success)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算奖励（旧接口，基于 achieved/desired goal）
        """
        # 位置误差
        pos_error = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
        distance_reward = -10.0 * pos_error

        # 姿态误差
        quat_achieved = achieved_goal[3:7]
        quat_desired = desired_goal[3:7]
        quat_dot = np.abs(np.dot(quat_achieved, quat_desired))
        quat_error = 1.0 - quat_dot
        orientation_reward = -5.0 * quat_error

        # 插入深度误差
        depth_achieved = achieved_goal[7] if len(achieved_goal) > 7 else 0.0
        depth_desired = desired_goal[7] if len(desired_goal) > 7 else self.insertion_depth
        depth_error = abs(depth_achieved - depth_desired)
        depth_reward = -20.0 * depth_error

        # 接近奖励
        approach_bonus = 5.0 if pos_error < 0.05 else 0.0

        # 成功奖励
        success_bonus = 0.0
        if (pos_error < self.position_tolerance and
            quat_error < self.orientation_tolerance and
            depth_error < 0.005):
            success_bonus = 20.0

        reward = (distance_reward + orientation_reward + depth_reward +
                  approach_bonus + success_bonus)
        return float(reward)

    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功（旧接口）
        """
        pos_error = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])

        quat_achieved = achieved_goal[3:7]
        quat_desired = desired_goal[3:7]
        quat_dot = np.abs(np.dot(quat_achieved, quat_desired))
        quat_error = 1.0 - quat_dot

        depth_achieved = achieved_goal[7] if len(achieved_goal) > 7 else 0.0
        depth_desired = desired_goal[7] if len(desired_goal) > 7 else self.insertion_depth
        depth_error = abs(depth_achieved - depth_desired)

        position_ok = pos_error < self.position_tolerance
        orientation_ok = quat_error < self.orientation_tolerance
        depth_ok = depth_error < 0.005
        return bool(position_ok and orientation_ok and depth_ok)

    def sample_goal(self) -> np.ndarray:
        """
        使用运行时 hole_center，而不是 hole body 原点，
        生成目标位姿。
        """
        target_position, _ = self._get_site_pose(
            "hole_center"
        )

        if target_position is None:
            if self.hole_position is not None:
                target_position = (
                    self.hole_position[:3]
                    .copy()
                    .astype(np.float32)
                )
            else:
                target_position = np.array(
                    [
                        np.random.uniform(
                            self.hole_position_range[
                                0, 0
                            ],
                            self.hole_position_range[
                                0, 1
                            ],
                        ),
                        np.random.uniform(
                            self.hole_position_range[
                                1, 0
                            ],
                            self.hole_position_range[
                                1, 1
                            ],
                        ),
                        np.random.uniform(
                            self.hole_position_range[
                                2, 0
                            ],
                            self.hole_position_range[
                                2, 1
                            ],
                        ),
                    ],
                    dtype=np.float32,
                )

        # 第一阶段保持 reset 时的工具姿态。
        _, target_quaternion = (
            self._get_site_pose(
                self.ee_site_name
            )
        )

        if target_quaternion is None:
            if (
                self.hole_orientation is not None
                and len(self.hole_orientation) >= 4
            ):
                target_quaternion = (
                    self.hole_orientation[:4]
                    .copy()
                    .astype(np.float32)
                )
            else:
                target_quaternion = np.array(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                )

        self._update_insertion_axis(
            target_position
        )

        goal = np.concatenate(
            [
                np.asarray(
                    target_position,
                    dtype=np.float32,
                )[:3],
                np.asarray(
                    target_quaternion,
                    dtype=np.float32,
                )[:4],
                np.array(
                    [self.insertion_depth],
                    dtype=np.float32,
                ),
            ]
        )

        return goal.astype(np.float32)

    def get_action_space(self) -> spaces.Box:
        """
        获取动作空间

        根据控制器类型返回相应的动作空间：
        - cartesian_ik, operational_space, cartesian_impedance: 7维位姿 + 1维夹爪 = 8维
        - joint_position, joint_velocity, joint_torque: dof维关节 + 1维夹爪

        Returns:
            action_space: 动作空间 [-6.28, 16.28]
        """

        # 根据控制器类型确定动作维度
        controller_type = self.robot_config.controller_type
        if controller_type in ["cartesian_ik", "operational_space", "cartesian_impedance", "osc", "task_space"]:
            # 任务空间控制：7维位姿（位置+四元数）+ 1维夹爪
            dim = 7
            if self.robot_config.gripper_name:
                dim += 1

            # 设置动作范围
            low = np.full(dim, -6.28, dtype=np.float32)
            high = np.full(dim, 6.28, dtype=np.float32)

            # 位置限制放宽到 [-10, 10]
            low[:3] = -10.0
            high[:3] = 10.0

            # 夹爪动作范围 [0, 1] 或 [-1, 1]
            if self.robot_config.gripper_name:
                low[-1] = -1.0
                high[-1] = 1.0

        else:
            # 关节空间控制：关节角度 + 夹爪
            dim = self.robot_config.dof
            if self.robot_config.gripper_name:
                dim += 1

            low = np.full(dim, -6.28, dtype=np.float32)
            high = np.full(dim, 6.28, dtype=np.float32)

            # 夹爪动作范围
            if self.robot_config.gripper_name:
                low[-1] = -1.0
                high[-1] = 1.0

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def get_achieved_goal(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取当前轴的位姿

        Returns:
            achieved_goal: [x, y, z, qw, qx, qy, qz, depth]
        """
        pos = None
        quat = None
        depth = None

        # 优先使用环境中的末端 site 位姿
        if self.env is not None and hasattr(self.env, 'data') and hasattr(self.env, 'model'):
            try:
                import mujoco
                ee_site_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
                if ee_site_id != -1:
                    pos = self.env.data.site_xpos[ee_site_id].copy()
                    if hasattr(self.env.data, 'site_xquat'):
                        quat = self.env.data.site_xquat[ee_site_id].copy()
            except Exception:
                pass

        # 回退到观测中的位置/姿态
        if pos is None:
            if 'peg_pos' in obs:
                pos = obs['peg_pos'].copy()
            elif 'tcp_pos' in obs:
                pos = obs['tcp_pos'].copy()
            else:
                pos = np.zeros(3, dtype=np.float32)

        if quat is None:
            if 'peg_quat' in obs:
                quat = obs['peg_quat'].copy()
            elif 'tcp_quat' in obs:
                quat = obs['tcp_quat'].copy()
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if depth is None:
            if 'insertion_depth' in obs:
                depth = float(obs['insertion_depth'])
            else:
                depth = 0.0

        achieved = np.concatenate([pos[:3], quat[:4], [depth]]).astype(np.float32)
        return achieved

    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间
        
        Returns:
            obs_space: 观测空间
        """
        # 装配任务的观测：
        # - force: 力/力矩占位（3维）（fx, fy, fz）或仅使用 XY 平面的 fx,fy
        # - peg_pos: 轴端（末端）位置 (x, y, z)
        # - peg_vel: 轴端速度 (x, y, z)
        # - desired_goal: 孔心位置 (x, y, z)
        return spaces.Dict({
            "tcp_pos": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            "tcp_quat": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            ),
            "wrench": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(8,),
                dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(8,),
                dtype=np.float32
            ),
        })
