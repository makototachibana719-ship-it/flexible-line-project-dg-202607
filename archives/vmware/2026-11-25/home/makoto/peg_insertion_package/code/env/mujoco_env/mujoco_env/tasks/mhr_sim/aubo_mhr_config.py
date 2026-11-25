"""
Aubo MHR 移动双臂机器人仿真任务配置

通过 RobotAssembler 动态组装 XML 文件。

日期: 2026-02-02
"""

from typing import Optional
from ...robot_config.aubo_mhr import AuboMHRRobot


class AuboMHRConfig(AuboMHRRobot):
    """Aubo MHR 移动双臂机器人仿真任务配置类
    
    该配置类用于基础仿真任务，支持通过 RobotAssembler 动态组装场景：
    - 移动底座 (aubo_mhr_base)
    - 双7-DOF机械臂
    - 双夹爪 (左/右)
    - 胸部和头部摄像头
    """
    
    @classmethod
    def get_config(cls) -> "AuboMHRConfig":
        """
        获取 mhr_sim 任务专用的 Aubo MHR 机器人配置
        
        这是一个基础配置，用于仿真测试：
        - 使用默认底座配置
        - 使用内置夹爪 (无需额外指定)
        - 关节位置控制模式
        - 20Hz 控制频率
        - 不使用外部力传感器
        
        Returns:
            AuboMHRConfig: 配置实例
        """
        return cls(
            controller_type="joint_position",  # 基础仿真使用关节位置控制
            control_freq=20,                     # 标准控制频率
            mount_name=None,                     # MHR 是一体化设计，底盘已在 manipulator 中
            gripper_name=None,                   # 使用内置夹爪
            use_ft_sensor=None,                  # 不使用外部力传感器
            camera_name=None,                    # 暂不指定外部相机
        )
    
    @classmethod
    def get_left_arm_config(cls) -> "AuboMHRConfig":
        """
        获取仅使用左臂的配置
        
        Returns:
            AuboMHRConfig: 左臂配置实例
        """
        config = cls.get_config()
        # 可以在这里添加左臂特定的设置
        return config
    
    @classmethod
    def get_right_arm_config(cls) -> "AuboMHRConfig":
        """
        获取仅使用右臂的配置
        
        Returns:
            AuboMHRConfig: 右臂配置实例
        """
        config = cls.get_config()
        # 可以在这里添加右臂特定的设置
        return config

