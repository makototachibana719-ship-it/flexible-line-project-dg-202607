"""
导纳控制器 (Admittance Controller)

导纳控制是一种力控制策略，通过测量外部力/力矩来调整末端执行器的位置/姿态。
适用于需要力反馈的接触任务，如装配、打磨、插拔等。

控制原理:
根据力误差调整位置: M·ẍ + D·ẋ + K·x = F_external - F_desired

其中:
- M: 虚拟质量
- D: 虚拟阻尼
- K: 虚拟刚度
- F_external: 外部力/力矩
- F_desired: 期望力/力矩

作者: Liu Gang
日期: 2025-12-21
"""

import numpy as np
import mujoco
from typing import Optional, Dict
from mujoco_env.mujoco_env.controllers.cartesian_ik import CartesianIKController
from mujoco_env.mujoco_env.utils import transform as T


class AdmittanceController(CartesianIKController):
    """
    导纳控制器
    
    基于CartesianIKController，添加力反馈的导纳控制。
    根据外部力/力矩调整目标位姿，实现柔顺的力控制。
    
    使用场景:
    - 接触任务（装配、插拔）
    - 力控制打磨/抛光
    - 拖动示教（力引导）
    - 柔顺操作
    
    Attributes:
        M: 虚拟质量 (6,)
        D: 虚拟阻尼 (6,)
        K: 虚拟刚度 (6,)
        selection_vector: 选择向量 (6,)，1表示启用导纳控制，0表示位置控制
        force_offset: 力传感器偏置 (6,)
        desired_force: 期望力/力矩 (6,)
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        ee_site_name: str = "pinch",
        mass: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        stiffness: Optional[np.ndarray] = None,
        selection_vector: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        初始化导纳控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            ee_site_name: 末端执行器site名称
            mass: 虚拟质量 (6,)，默认[20, 20, 20, 20, 20, 20]
            damping: 虚拟阻尼 (6,)，默认[200, 200, 200, 200, 200, 200]
            stiffness: 虚拟刚度 (6,)，默认[0, 0, 0, 0, 0, 0]
            selection_vector: 选择向量 (6,)，默认全0（纯位置控制）
            kp: 关节位置增益
            kd: 关节速度增益
            **kwargs: 额外参数
        """
        super().__init__(
            model, data, dof, control_freq, ee_site_name,
            kp=kp, kd=kd, **kwargs
        )
        
        # 导纳参数
        self.M = mass if mass is not None else 20.0 * np.ones(6)
        self.D = damping if damping is not None else 200.0 * np.ones(6)
        self.K = stiffness if stiffness is not None else 0.0 * np.ones(6)
        
        # 选择向量：1表示启用导纳控制，0表示位置控制
        self.selection_vector = (
            selection_vector if selection_vector is not None 
            else np.zeros(6)
        )
        
        # 力传感器偏置和期望力
        self.force_offset = np.zeros(6)
        self.desired_force = np.zeros(6)
        
        # 导纳状态
        self.adm_pos = np.zeros(6)  # [x, y, z, rx, ry, rz] (欧拉角)
        self.adm_vel = np.zeros(6)
        self.adm_acc = np.zeros(6)
        
        # 初始化导纳位置为当前位置
        current_pos, current_quat = self.forward_kinematics()
        self.adm_pos[:3] = current_pos
        self.adm_pos[3:] = T.quat_2_euler(current_quat)
        
        self.cycle_time = 1.0 / control_freq
    
    def update_admittance(
        self,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
        ref_acc: np.ndarray,
        external_force: np.ndarray
    ) -> np.ndarray:
        """
        更新导纳控制状态
        
        根据外部力和参考轨迹，计算新的导纳位置。
        
        Args:
            ref_pos: 参考位姿 (7,) [x, y, z, qw, qx, qy, qz]
            ref_vel: 参考速度 (6,) [vx, vy, vz, wx, wy, wz]
            ref_acc: 参考加速度 (6,)
            external_force: 外部力/力矩 (6,) [fx, fy, fz, tx, ty, tz]
        
        Returns:
            adm_pose (7,): 新的导纳位姿 [x, y, z, qw, qx, qy, qz]
        """
        # 转换参考位姿到欧拉角
        ref_pos_xyz = ref_pos[:3]
        ref_euler = T.quat_2_euler(ref_pos[3:])
        ref_pos_euler = np.concatenate([ref_pos_xyz, ref_euler])
        
        # 计算误差
        pos_error = ref_pos_euler - self.adm_pos
        vel_error = ref_vel - self.adm_vel
        
        # 力误差（去除偏置）
        force_error = self.desired_force - (external_force - self.force_offset)
        
        # 导纳控制更新
        delta_pos = np.zeros(6)
        
        for i in range(6):
            if self.selection_vector[i] == 1:
                # 启用导纳控制
                # M·ẍ + D·ẋ + K·x = F_ext - F_des
                self.adm_acc[i] = ref_acc[i] + (
                    -force_error[i] + self.D[i] * vel_error[i] + self.K[i] * pos_error[i]
                ) / self.M[i]
                
                # 速度积分
                self.adm_vel[i] = ref_vel[i] - vel_error[i] + self.adm_acc[i] * self.cycle_time
                
                # 位置积分
                delta_pos[i] = self.adm_vel[i] * self.cycle_time
            else:
                # 纯位置控制
                self.adm_acc[i] = ref_acc[i]
                self.adm_vel[i] = ref_vel[i]
                delta_pos[i] = 0.0
        
        # 更新位置（笛卡尔分量直接加）
        self.adm_pos[:3] += delta_pos[:3]
        
        # 更新姿态（欧拉角增量左乘）
        delta_mat = T.euler_2_mat(delta_pos[3:])
        current_mat = T.euler_2_mat(self.adm_pos[3:])
        new_mat = delta_mat @ current_mat
        self.adm_pos[3:] = T.mat_2_euler(new_mat)
        
        # 转换回四元数
        adm_quat = T.euler_2_quat(self.adm_pos[3:])
        return np.concatenate([self.adm_pos[:3], adm_quat])
    
    def compute_control(
        self,
        target: np.ndarray,
        current_state: Optional[Dict[str, np.ndarray]] = None,
        external_force: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算导纳控制力矩
        
        Args:
            target: 目标位姿 (7,) [x, y, z, qw, qx, qy, qz]
            current_state: 当前状态
            external_force: 外部力/力矩 (6,)，如果为None则使用零力
        
        Returns:
            tau (dof,): 关节力矩
        """
        # 如果没有提供外部力，使用零力
        if external_force is None:
            external_force = np.zeros(6)
        
        # 参考轨迹（简化：假设参考速度和加速度为0）
        ref_vel = np.zeros(6)
        ref_acc = np.zeros(6)
        
        # 如果选择向量全为0，直接使用位置控制
        if np.all(self.selection_vector == 0):
            return super().compute_control(target, current_state)
        
        # 更新导纳状态
        adm_pose = self.update_admittance(target, ref_vel, ref_acc, external_force)
        
        # 使用IK控制器跟踪导纳位姿
        return super().compute_control(adm_pose, current_state)
    
    def set_admittance_params(
        self,
        mass: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        stiffness: Optional[np.ndarray] = None
    ):
        """
        设置导纳参数
        
        Args:
            mass: 虚拟质量 (6,)
            damping: 虚拟阻尼 (6,)
            stiffness: 虚拟刚度 (6,)
        """
        if mass is not None:
            self.M = mass
        if damping is not None:
            self.D = damping
        if stiffness is not None:
            self.K = stiffness
    
    def set_selection_vector(self, selection: np.ndarray):
        """
        设置选择向量
        
        Args:
            selection (6,): 1表示启用导纳控制，0表示位置控制
                           例如: [1, 1, 0, 0, 0, 0] 表示只在x,y方向启用力控制
        """
        assert len(selection) == 6, "Selection vector must be length 6"
        self.selection_vector = selection
    
    def set_desired_force(self, force: np.ndarray):
        """
        设置期望力/力矩
        
        Args:
            force (6,): [fx, fy, fz, tx, ty, tz]
        """
        assert len(force) == 6, "Force vector must be length 6"
        self.desired_force = force
    
    def calibrate_force_sensor(self, force_reading: np.ndarray):
        """
        标定力传感器（设置偏置）
        
        Args:
            force_reading (6,): 当前力/力矩读数
        """
        self.force_offset = force_reading.copy()
    
    def reset(self):
        """重置导纳控制器"""
        super().reset()
        
        # 重置导纳状态
        self.adm_vel = np.zeros(6)
        self.adm_acc = np.zeros(6)
        
        # 重置导纳位置为当前位置
        current_pos, current_quat = self.forward_kinematics()
        self.adm_pos[:3] = current_pos
        self.adm_pos[3:] = T.quat_2_euler(current_quat)
    
    def __repr__(self) -> str:
        active_dims = np.sum(self.selection_vector)
        return (
            f"AdmittanceController("
            f"dof={self.dof}, "
            f"freq={self.control_freq}Hz, "
            f"active_dims={int(active_dims)})"
        )

