"""
AGV Aubo i5机器人位置跟踪任务配置

作者: mafuyuan
日期: 2026-01-07
"""

from typing import Optional
from ...robot_config.aubo_i5 import AuboI5Robot


class AuboI5AGVConfig(AuboI5Robot):
    """Aubo i5 AGV移动机器人配置类"""
    
    # 扩展底座列表，添加 freight
    AVAILABLE_MOUNTS = AuboI5Robot.AVAILABLE_MOUNTS + ["freight"]
    
    # 扩展夹爪列表，添加 robotiq_realsense
    AVAILABLE_GRIPPERS = AuboI5Robot.AVAILABLE_GRIPPERS + ["robotiq_realsense"]
    
    # 扩展底座位置
    MOUNT_POS = {
        **AuboI5Robot.MOUNT_POS,
        "freight": [-0.2, 0, 0],  # freight_base 的世界坐标位置
    }
    
    # 扩展机器人安装偏移
    ROBOT_OFFSET = {
        **AuboI5Robot.ROBOT_OFFSET,
        "freight": [-0.08, 0, 0.43],  # 相对于 freight_base 的偏移
    }

    @classmethod
    def get_config(cls) -> "AuboI5AGVConfig":
        """
        Aubo i5 AGV移动机器人配置
        
        - 使用 freight 移动底座（6自由度AGV小车）
        - 使用 robotiq_realsense 夹爪（带RealSense相机）
        - 笛卡尔IK控制模式
        - 20Hz 控制频率（control_dt=0.05）
        - 不使用六维力传感器
        """
        return cls(
            controller_type="cartesian_ik",
            control_freq=20,
            mount_name="pedestal",              # 使用freight移动底座,pedestal固定底座
            gripper_name="robotiq_realsense",  # 使用带相机的夹爪
            use_ft_sensor=False,
        )