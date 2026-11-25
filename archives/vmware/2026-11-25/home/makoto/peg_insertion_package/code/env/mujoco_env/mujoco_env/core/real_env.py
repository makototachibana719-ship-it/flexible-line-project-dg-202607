"""
真机环境基类

作者: Liu Gang
日期: 2025-12-20
更新: 2025-12-24 - Phase 2: 完整实现真机环境
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from gymnasium import spaces
import time

from mujoco_env.mujoco_env.core.base_env import BaseRobotEnv
from mujoco_env.mujoco_env.tasks.base_task import BaseTask
from mujoco_env.mujoco_env.real.robot_interface import RobotInterface


class RealRobotEnv(BaseRobotEnv):
    """
    真机环境基类
    
    提供与真实机器人交互的完整Gymnasium环境接口。
    
    设计思路:
    - robot_interface: 负责与真实机器人的底层通信
    - task: 定义任务目标、奖励函数、成功判断
    - 环境本身: 实现Gymnasium标准接口
    
    使用示例:
        ```python
        from mujoco_env.mujoco_env.real import get_robot_interface
        from mujoco_env.mujoco_env.tasks import PickCubeTask
        from mujoco_env.mujoco_env.tasks.base_task import RobotConfig
        
        # 创建机器人接口
        robot_interface = get_robot_interface(
            robot_type="franka",
            robot_ip="192.168.1.1"
        )
        
        # 创建任务
        robot_config = RobotConfig(name="franka", dof=7, ...)
        task = PickCubeTask(robot_config=robot_config)
        
        # 创建环境
        env = RealRobotEnv(
            robot_interface=robot_interface,
            task=task
        )
        
        # 使用环境
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        ```
    """
    
    def __init__(
        self,
        robot_interface: RobotInterface,
        task: BaseTask,
        control_freq: float = 20.0,
        render_mode: Optional[str] = None,
    ):
        """
        初始化真机环境
        
        Args:
            robot_interface: 机器人接口实例（需要实现 RobotInterface）
            task: 任务实例（需要实现 BaseTask）
            control_freq: 控制频率 (Hz)
            render_mode: 渲染模式（真机环境通常不需要渲染）
        """
        self.robot_interface = robot_interface
        self.task = task
        self.control_freq = control_freq
        self.control_dt = 1.0 / control_freq
        
        # 从 task 中获取配置
        self.robot_config = task.robot_config
        self.obs_config = task.obs_config
        self.max_episode_steps = task.max_episode_steps
        
        # Episode 状态
        self.current_step = 0
        self._desired_goal = None
        
        # 连接机器人
        if not self.robot_interface.is_connected:
            if not self.robot_interface.connect():
                raise ConnectionError(
                    f"Failed to connect to robot at {self.robot_interface.robot_ip}"
                )
        
        # 初始化父类
        super().__init__(render_mode=render_mode)
        
        # 设置观测和动作空间（必须在 task 创建之后）
        self._setup_spaces()
    
    def _setup_spaces(self):
        """
        设置观测空间和动作空间
        
        观测空间由三部分组成：
        1. 机器人本体感知（qpos, qvel, tcp_pose, tcp_vel）
        2. 任务相关观测（从 task.get_obs_space() 获取）
        3. 传感器观测（force, torque 等）
        """
        # ========== Action Space ==========
        # 优先使用任务定义的动作空间，否则使用机器人DOF
        if hasattr(self.task, 'get_action_space'):
            self.action_space = self.task.get_action_space()
        else:
            # 默认使用机器人DOF（向后兼容）
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.robot_config.dof,),
                dtype=np.float32
            )
        
        # ========== Observation Space ==========
        obs_dict = {}
        
        # 1. 机器人本体感知（始终包含）
        obs_dict["qpos"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot_config.dof,),
            dtype=np.float32
        )
        obs_dict["qvel"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot_config.dof,),
            dtype=np.float32
        )
        obs_dict["tcp_pose"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # [x, y, z, qw, qx, qy, qz]
            dtype=np.float32
        )
        obs_dict["tcp_vel"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # [vx, vy, vz, wx, wy, wz]
            dtype=np.float32
        )
        
        # 2. 任务相关观测（从 task.get_obs_space() 获取）
        if self.task is not None:
            task_obs_space = self.task.get_obs_space()
            if isinstance(task_obs_space, spaces.Dict):
                obs_dict.update(task_obs_space.spaces)

        # 目标相关观测（补齐缺失的 achieved/desired）
        if self.task is not None and getattr(self.obs_config, 'include_goal', False):
            if 'desired_goal' not in obs_dict or 'achieved_goal' not in obs_dict:
                try:
                    goal_shape = tuple(self.task.sample_goal().shape)
                except Exception:
                    goal_shape = (3,)
                if 'desired_goal' not in obs_dict:
                    obs_dict['desired_goal'] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=goal_shape, dtype=np.float32
                    )
                if 'achieved_goal' not in obs_dict:
                    obs_dict['achieved_goal'] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=goal_shape, dtype=np.float32
                    )

        # 3. 传感器观测（力/力矩）
        if self.obs_config.include_proprioception:
            obs_dict["tcp_force"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            )
            obs_dict["tcp_torque"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(obs_dict)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置步数
        self.current_step = 0
        
        # 清除机器人错误
        self.robot_interface.clear_errors()
        
        # 重置任务
        if self.task is not None:
            task_info = self.task.reset()
            
            # 获取期望目标
            self._desired_goal = self.task.sample_goal()
        
        # 获取初始观测
        obs = self._get_obs()
        
        # 构建info
        info = {
            "is_success": False,
            "episode_step": self.current_step,
        }
        if self.task is not None:
            info.update(self.task.get_info())
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作，shape=(dof,)，范围 [-1, 1]
        
        Returns:
            observation: 新的观测
            reward: 奖励
            terminated: 是否达成任务目标
            truncated: 是否超时
            info: 额外信息
        """
        # 记录开始时间（用于控制频率）
        start_time = time.time()
        
        # 1. 发送动作到机器人
        self._set_action(action)
        
        # 2. 等待一个控制周期（保持控制频率）
        elapsed = time.time() - start_time
        if elapsed < self.control_dt:
            time.sleep(self.control_dt - elapsed)
        
        # 3. 获取新的观测
        obs = self._get_obs()
        
        # 4. 计算奖励
        reward = self._compute_reward(obs)
        
        # 5. 检查是否终止
        terminated = self._check_terminated(obs)
        
        # 6. 检查是否超时
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        
        # 7. 构建info
        info = {
            "is_success": terminated,
            "episode_step": self.current_step,
        }
        if self.task is not None:
            info.update(self.task.get_info())
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取当前观测
        
        从机器人接口获取实时状态，并组合任务相关观测
        
        Returns:
            observation: 观测字典
        """
        obs = {}
        
        # 1. 从机器人接口获取状态
        robot_state = self.robot_interface.get_robot_state()
        
        # 机器人本体感知
        obs["qpos"] = robot_state["joint_positions"].astype(np.float32)
        obs["qvel"] = robot_state["joint_velocities"].astype(np.float32)
        obs["tcp_pose"] = robot_state["tcp_pose"].astype(np.float32)
        obs["tcp_vel"] = robot_state["tcp_velocity"].astype(np.float32)
        
        # 力/力矩传感器
        if self.obs_config.include_proprioception:
            obs["tcp_force"] = robot_state["tcp_force"].astype(np.float32)
            obs["tcp_torque"] = robot_state["tcp_torque"].astype(np.float32)
        
        # 2. 任务相关观测
        if self.task is not None:
            # 获取 achieved_goal 和 desired_goal
            obs["achieved_goal"] = self.task.get_achieved_goal(obs).astype(np.float32)
            
            if self._desired_goal is not None:
                obs["desired_goal"] = self._desired_goal.astype(np.float32)
            else:
                # 如果 desired_goal 未设置，采样一个
                obs["desired_goal"] = self.task.sample_goal().astype(np.float32)
        
        return obs
    
    def _set_action(self, action: np.ndarray):
        """
        发送动作到机器人
        
        将归一化的动作转换为实际控制指令并发送到机器人
        
        Args:
            action: 归一化动作，shape=(dof,)，范围 [-1, 1]
        """
        # 根据 controller_type 处理动作
        controller_type = self.robot_config.controller_type
        
        if controller_type in ["joint_position", "position"]:
            # 关节位置控制
            # 将 [-1, 1] 映射到关节限位
            current_qpos = self.robot_interface.get_joint_positions()
            
            # 简单的增量控制: delta_qpos = action * max_delta
            max_delta = 0.1  # 最大增量（弧度）
            target_qpos = current_qpos + action * max_delta
            
            # 发送目标位置（非阻塞）
            self.robot_interface.move_to_joint_positions(
                positions=target_qpos,
                velocity=0.5,
                acceleration=0.5,
                blocking=False
            )
        
        elif controller_type in ["joint_velocity", "velocity"]:
            # 关节速度控制
            # action 直接作为速度指令
            # 注意：需要机器人接口支持速度控制
            raise NotImplementedError(
                "Joint velocity control not yet implemented for RealRobotEnv"
            )
        
        elif controller_type in ["joint_torque", "torque"]:
            # 关节力矩控制
            # action 直接作为力矩指令
            raise NotImplementedError(
                "Joint torque control not yet implemented for RealRobotEnv"
            )
        
        else:
            raise ValueError(
                f"Unknown controller type: {controller_type}. "
                f"Supported types: joint_position, joint_velocity, joint_torque"
            )
    
    def _compute_reward(self, obs: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        计算奖励

        委托给 task 进行奖励计算（支持直接基于obs）

        Returns:
            reward: 当前步的奖励
        """
        if self.task is None:
            return 0.0

        if obs is None:
            obs = self._get_obs()

        if hasattr(self.task, "compute_reward_from_obs"):
            return float(self.task.compute_reward_from_obs(obs))

        # 兼容旧接口
        achieved_goal = obs.get("achieved_goal")
        if achieved_goal is None:
            achieved_goal = self.task.get_achieved_goal(obs)
        desired_goal = obs.get("desired_goal", self._desired_goal)
        return self.task.compute_reward(achieved_goal, desired_goal, self.task.get_info())
    
    def _check_terminated(self, obs: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        检查任务是否完成

        委托给 task 进行成功判断（支持直接基于obs）

        Returns:
            terminated: 是否完成任务
        """
        if self.task is None:
            return False

        if obs is None:
            obs = self._get_obs()

        if hasattr(self.task, "is_success_from_obs"):
            return bool(self.task.is_success_from_obs(obs))

        # 兼容旧接口
        achieved_goal = obs.get("achieved_goal")
        if achieved_goal is None:
            achieved_goal = self.task.get_achieved_goal(obs)
        desired_goal = obs.get("desired_goal", self._desired_goal)
        return self.task.is_success_fn(achieved_goal, desired_goal)
    
    def close(self):
        """关闭环境并断开机器人连接"""
        if self.robot_interface.is_connected:
            # 停止机器人运动
            try:
                self.robot_interface.stop_motion()
            except Exception as e:
                print(f"Warning: Failed to stop robot motion: {e}")
            
            # 断开连接
            self.robot_interface.disconnect()
    
    def render(self):
        """
        渲染环境
        
        真机环境通常不需要渲染，可以返回相机图像（如果有）
        """
        # TODO: 如果有外部相机，可以在这里返回图像
        return None

