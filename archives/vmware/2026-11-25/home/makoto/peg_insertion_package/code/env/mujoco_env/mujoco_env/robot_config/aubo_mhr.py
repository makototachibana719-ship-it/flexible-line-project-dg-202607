"""
Aubo MHR 机器人配置

Aubo MHR 是一款双臂移动机器人，配备四轮转向底盘、双7-DOF机械臂和双夹爪。

作者: Liu Gang
日期: 2026-02-02
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import RobotConfig


class AuboMHRRobot(RobotConfig):
    """Aubo MHR 移动双臂机器人配置类
    
    MHR (Mobile Humanoid Robot) 是一款集成移动底盘和双臂操作的机器人平台。
    
    主要特性:
    - 四轮阿克曼转向底盘
    - 双 7-DOF 机械臂 (左臂和右臂)
    - 可升降躯干
    - 双夹爪末端执行器
    - 胸部和头部摄像头
    """
    
    # ============ 基本参数 ============
    NAME = "aubo_mhr"
    DOF = 31  # 4(转向) + 1(升降) + 7*2(双臂) + 6*2(双夹爪)
    
    # ============ 支持的底座类型 ============
    # 底座资源路径: assets/models/mobile_bases/aubo_mhr/
    AVAILABLE_MOUNTS = ["default", "aubo_mhr"]
    
    # 底座位置配置
    MOUNT_POS = {
        "default": [0, 0, 0.101],      # 默认位置 (轮子离地高度)
        "aubo_mhr": [0, 0, 0.101],     # 四轮底座位置
    }
    
    # 机器人安装偏移 (相对于底座)
    ROBOT_OFFSET = {
        "default": [0, 0, 0],
        "aubo_mhr": [0, 0, 0],
    }
    
    # ============ 支持的夹爪类型 ============
    # 夹爪资源路径: assets/models/grippers/aubo_mhr/
    AVAILABLE_GRIPPERS = ["aubo_mhr_gripper_left", "aubo_mhr_gripper_right"]
    
    # ============ 支持的传感器类型 ============
    # 传感器资源路径: assets/models/sensors/aubo_mhr/
    AVAILABLE_CAMERAS = ["camera_chest", "camera_head"]
    
    # 传感器配置
    SENSOR_CONFIG = {
        "camera_chest": {
            "model_path": "models/sensors/aubo_mhr/meshes/camera_chest.STL",
            "pos": [-0.00150228, -0.315718, 0.064177],
            "quat": [0.508498, 0.507122, -0.495193, -0.488918],
        },
        "camera_head": {
            "model_path": "models/sensors/aubo_mhr/meshes/camera_head.STL",
            "pos": [0.0217833, -0.357549, 0.601506],
            "quat": [0.641628, -0.291476, 0.646416, -0.292407],
        },
    }
    
    # ============ 关节名称 ============
    # 按顺序: 底盘转向 -> 升降 -> 左臂 -> 左夹爪 -> 右臂 -> 右夹爪
    JOINT_NAMES = [
        # 底盘转向关节 (4个)
        "front_right_steering_joint",
        "front_left_steering_joint",
        "rear_right_steering_joint",
        "rear_left_steering_joint",
        # 躯干升降关节 (1个)
        "robot_joint",
        # 左臂关节 (7个)
        "l_shoulder_roll",
        "l_shoulder_pitch",
        "l_shoulder_yaw",
        "l_elbow",
        "l_wrist_roll",
        "l_wrist_pitch",
        "l_wrist_yaw",
        # 左夹爪关节 (6个)
        "gripper_r_joint1_left",
        "gripper_r_joint2_left",
        "gripper_r_joint_finger_left",
        "gripper_l_joint1_left",
        "gripper_l_joint2_left",
        "gripper_l_joint_finger_left",
        # 右臂关节 (7个)
        "r_shoulder_roll",
        "r_shoulder_pitch",
        "r_shoulder_yaw",
        "r_elbow",
        "r_wrist_roll",
        "r_wrist_pitch",
        "r_wrist_yaw",
        # 右夹爪关节 (6个)
        "gripper_r_joint1_right",
        "gripper_r_joint2_right",
        "gripper_r_joint_finger_right",
        "gripper_l_joint1_right",
        "gripper_l_joint2_right",
        "gripper_l_joint_finger_right",
    ]
    
    # 左臂关节名称 (用于单臂控制)
    LEFT_ARM_JOINT_NAMES = [
        "l_shoulder_roll", "l_shoulder_pitch", "l_shoulder_yaw",
        "l_elbow", "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw"
    ]
    
    # 右臂关节名称 (用于单臂控制)
    RIGHT_ARM_JOINT_NAMES = [
        "r_shoulder_roll", "r_shoulder_pitch", "r_shoulder_yaw",
        "r_elbow", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"
    ]
    
    # ============ 关节限位 (弧度) ============
    JOINT_LIMITS = np.array([
        # 底盘转向关节
        [-2.1, 2.1],        # front_right_steering_joint
        [-2.1, 2.1],        # front_left_steering_joint
        [-2.1, 2.1],        # rear_right_steering_joint
        [-2.1, 2.1],        # rear_left_steering_joint
        # 躯干升降关节
        [0.2, 0.7],         # robot_joint (slide)
        # 左臂关节
        [-3.05433, 3.05433],  # l_shoulder_roll
        [0.0, 1.74533],       # l_shoulder_pitch
        [-3.05433, 3.05433],  # l_shoulder_yaw
        [-2.26893, 0.0],      # l_elbow
        [-3.05433, 3.05433],  # l_wrist_roll
        [-1.39626, 1.39626],  # l_wrist_pitch
        [-3.05433, 3.05433],  # l_wrist_yaw
        # 左夹爪关节
        [-10.0, 10.0],  # gripper_r_joint1_left
        [-10.0, 10.0],  # gripper_r_joint2_left
        [-10.0, 10.0],  # gripper_r_joint_finger_left
        [-10.0, 10.0],  # gripper_l_joint1_left
        [-10.0, 10.0],  # gripper_l_joint2_left
        [-10.0, 10.0],  # gripper_l_joint_finger_left
        # 右臂关节
        [-3.05433, 3.05433],  # r_shoulder_roll
        [-1.74533, 0.0],      # r_shoulder_pitch
        [-3.05433, 3.05433],  # r_shoulder_yaw
        [0.0, 2.26893],       # r_elbow
        [-3.05433, 3.05433],  # r_wrist_roll
        [-1.39626, 1.39626],  # r_wrist_pitch
        [-3.05433, 3.05433],  # r_wrist_yaw
        # 右夹爪关节
        [-10.0, 10.0],  # gripper_r_joint1_right
        [-10.0, 10.0],  # gripper_r_joint2_right
        [-10.0, 10.0],  # gripper_r_joint_finger_right
        [-10.0, 10.0],  # gripper_l_joint1_right
        [-10.0, 10.0],  # gripper_l_joint2_right
        [-10.0, 10.0],  # gripper_l_joint_finger_right
    ])
    
    # ============ 速度限位 (rad/s) ============
    VELOCITY_LIMITS = np.array([
        # 底盘转向
        2.0, 2.0, 2.0, 2.0,
        # 升降
        0.5,
        # 左臂
        3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
        # 左夹爪
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        # 右臂
        3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14,
        # 右夹爪
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    ])
    
    # ============ 力矩限位 (Nm) ============
    TORQUE_LIMITS = np.array([
        # 底盘转向
        5, 5, 5, 5,
        # 升降
        100,
        # 左臂
        40, 40, 8, 22, 8, 8, 8,
        # 左夹爪
        5, 5, 5, 5, 5, 5,
        # 右臂
        40, 40, 22, 22, 8, 8, 8,
        # 右夹爪
        5, 5, 5, 5, 5, 5,
    ])
    
    # ============ Home 位置 ============
    HOME_QPOS = np.array([
        # 底盘转向 (全部0)
        0.0, 0.0, 0.0, 0.0,
        # 升降 (中间位置)
        0.45,
        # 左臂 (折叠姿态)
        0.0, 0.872665, 0.0, -1.134465, 0.0, 0.0, 0.0,
        # 左夹爪 (默认打开)
        -0.8, 0.8, -0.8, -0.8, 0.8, -0.8,
        # 右臂 (折叠姿态, 镜像)
        0.0, -0.872665, 0.0, 1.134465, 0.0, 0.0, 0.0,
        # 右夹爪 (默认打开)
        -0.8, 0.8, -0.8, -0.8, 0.8, -0.8,
    ])
    
    # ============ 初始位置 (用于任务) ============
    INIT_QPOS = HOME_QPOS.copy()

    def __init__(
        self,
        controller_type: str = "joint_position",
        control_freq: int = 20,
        mount_name: str = "default",
        gripper_name: Optional[str] = None,
        use_ft_sensor: Optional[str] = None,
        camera_name: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 Aubo MHR 机器人配置
        
        Args:
            controller_type: 控制器类型 ("joint_position", "cartesian_ik", etc.)
            control_freq: 控制频率 (Hz)
            mount_name: 底座名称
            gripper_name: 夹爪名称 (如果为 None，使用内置夹爪)
            use_ft_sensor: 力矩传感器名称
            camera_name: 相机名称
            **kwargs: 其他配置参数
        """
        mount_pos = self.MOUNT_POS.get(mount_name, [0, 0, 0])
        robot_offset = self.ROBOT_OFFSET.get(mount_name, [0, 0, 0])

        super().__init__(
            name=self.NAME,
            robot_type="aubo_mhr",
            dof=self.DOF,
            controller_type=controller_type,
            control_freq=control_freq,
            mount_name=mount_name,
            mount_pos=mount_pos,
            robot_offset=robot_offset,
            gripper_name=gripper_name,
            use_ft_sensor=use_ft_sensor,
            joint_names=self.JOINT_NAMES,
            camera_name=camera_name,
            **kwargs
        )

    @classmethod
    def get_config(
        cls,
        controller_type: str = "joint_position",
        control_freq: int = 20,
        mount_name: str = "default",
        gripper_name: Optional[str] = None,
        use_ft_sensor: Optional[str] = None,
        camera_name: Optional[str] = None,
        **kwargs
    ) -> "AuboMHRRobot":
        """
        获取机器人配置实例
        
        Args:
            controller_type: 控制器类型
            control_freq: 控制频率
            mount_name: 底座名称
            gripper_name: 夹爪名称
            use_ft_sensor: 力矩传感器名称
            camera_name: 相机名称
            **kwargs: 额外配置
            
        Returns:
            AuboMHRRobot: 机器人配置对象
        """
        return cls(
            controller_type=controller_type,
            control_freq=control_freq,
            mount_name=mount_name,
            gripper_name=gripper_name,
            use_ft_sensor=use_ft_sensor,
            camera_name=camera_name,
            **kwargs
        )

    @staticmethod
    def get_xml_path(scene: str = "default") -> Optional[Path]:
        """
        获取机器人 XML 文件路径
        
        Args:
            scene: 场景名称
            
        Returns:
            Path 对象或 None
        """
        base_path = Path(__file__).parent.parent / "assets" / "models"
        xml_path = base_path / "manipulators" / "aubo_mhr" / "aubo_mhr.xml"
        if xml_path.exists():
            return xml_path

        fallback = base_path / "robots" / "aubo_mhr" / "aubo_mhr.xml"
        if fallback.exists():
            return fallback

        return None
    
    @classmethod
    def get_mount_pos(cls, mount_name: str) -> List:
        """获取底座位置"""
        return cls.MOUNT_POS.get(mount_name, [0, 0, 0])
    
    @classmethod
    def get_info(cls) -> Dict[str, any]:
        """获取机器人信息"""
        return {
            "name": cls.NAME,
            "dof": cls.DOF,
            "manufacturer": "Aubo Robotics",
            "robot_type": "Mobile Dual-Arm Humanoid",
            "left_arm_dof": 7,
            "right_arm_dof": 7,
            "joint_limits": cls.JOINT_LIMITS.tolist(),
            "velocity_limits": cls.VELOCITY_LIMITS.tolist(),
            "torque_limits": cls.TORQUE_LIMITS.tolist(),
            "available_mounts": cls.AVAILABLE_MOUNTS,
            "available_grippers": cls.AVAILABLE_GRIPPERS,
            "available_cameras": cls.AVAILABLE_CAMERAS,
        }
    
    @classmethod
    def list_cameras(cls) -> List[str]:
        """列出支持的相机类型"""
        return cls.AVAILABLE_CAMERAS.copy()
    
    @classmethod
    def list_mounts(cls) -> List[str]:
        """列出支持的底座类型"""
        return cls.AVAILABLE_MOUNTS.copy()
    
    @classmethod
    def list_grippers(cls) -> List[str]:
        """列出支持的夹爪类型"""
        return cls.AVAILABLE_GRIPPERS.copy()
