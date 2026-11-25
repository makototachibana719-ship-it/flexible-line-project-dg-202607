#!/usr/bin/env python3
"""Aubo MHR 双臂场景仿真 - ROS 2 控制预备版本

此文件已清理，移除了所有基于 Python 的控制逻辑（CartesianIKController, RRT 等）。
后续将通过 ROS 2 (mujoco_ros2_control + MoveIt 2) 实现机器人控制。

当前功能：
- 使用 RobotAssembler 组装场景 XML
- 加载 MuJoCo 模型并启动被动可视化
"""
import os
import sys
import time
from pathlib import Path

import mujoco

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from mujoco_env.utils.robot_assembler import RobotAssembler
from mujoco_env.tasks.mhr_sim import AuboMHRConfig


def main():
    """主函数：组装场景并启动被动仿真"""
    # 路径配置
    assets_dir = Path(script_dir) / ".." / "mujoco_env" / "assets"
    output_dir = Path(script_dir) / ".." / "mujoco_env" / "tasks" / "mhr_sim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 通过 RobotAssembler 组装场景 XML
    robot_config = AuboMHRConfig.get_config()
    assembler = RobotAssembler(assets_dir.resolve())
    model_path = assembler.build_robot_scene("mhr_sim", robot_config, output_dir.resolve())
    
    print(f"模型已生成: {model_path}")

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    sim_dt = float(model.opt.timestep)

    # 启动被动仿真可视化
    print("启动被动仿真... (按 Ctrl + C 退出，或者Ctrl + \强制退出)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(sim_dt)


if __name__ == "__main__":
    main()
