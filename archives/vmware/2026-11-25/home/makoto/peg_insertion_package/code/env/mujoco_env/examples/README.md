# examples

本目录提供可直接运行的入口脚本，用于快速验证 `mujoco-env` 的环境、控制、规划与任务链路。

## 你可以用这些示例做什么

- 快速验证仿真环境是否可运行。
- 对比不同机器人与任务配置的控制效果。
- 复用统一环境工厂（`make_sim_env` / `make_real_env`）搭建自己的实验脚本。
- 串联 OpenPI 数据采集、推理与转换工作流。

## 依赖安装

```bash
cd path/to/mujoco-env
pip install -r requirements.txt
```

OpenPI 相关示例有额外依赖，详见：`examples/demo_aubo_i5_openpi/README.md`。

## 快速开始（最小可复现）

```bash
cd path/to/mujoco-env

# Aubo MHR 被动场景仿真
python examples/demo_aubo_mhr_scene_sim.py

# Aubo i5 + RRT 示例
python examples/demo_aubo_i5_pick_cube_rrt.py

# Franka Panda 抓方块示例
python examples/demo_franka_panda_pick_cube.py

# 真机接口示例（需真机与网络可达）
python examples/demo_real_robot_interface.py
```

## 示例分类

- 基础运动：`*_simple_move.py`
- 任务执行：`pick_*`、`assemble_*`
- 规划控制：`*_rrt.py`、`demo_admittance_control.py`
- 数据与推理：`demo_aubo_i5_openpi/*`

## 目录结构

```text
examples/
├── demo_admittance_control.py               # 导纳控制示例
├── demo_aubo_i5_simple_move.py              # Aubo i5 基础运动示例
├── demo_aubo_i5_pick_cube_keyboard.py       # Aubo i5 键盘抓取示例
├── demo_aubo_i5_pick_cube_rrt.py            # Aubo i5 + RRT 路径规划示例
├── demo_aubo_i5_pick_up_object.py           # Aubo i5 抓取物体示例
├── demo_aubo_i5_pick_apple.py               # Aubo i5 fruit_pick 任务示例
├── demo_aubo_i5_assemble_hole.py            # Aubo i5 轴孔装配示例
├── demo_franka_fr3_simple_move.py           # Franka FR3 基础运动示例
├── demo_franka_panda_pick_cube.py           # Franka Panda 抓方块示例
├── demo_real_robot_interface.py             # 真机接口示例（Aubo/Franka）
├── demo_aubo_mhr_scene_sim.py               # Aubo MHR 场景装配 + 被动仿真入口
└── demo_aubo_i5_openpi/                     # OpenPI 数据采集/推理/转换示例
```

## 相关模块

示例通常复用以下模块：
- `mujoco_env.robot_config`：机器人配置
- `mujoco_env.tasks`：任务定义
- `mujoco_env.envs.env_factory`：环境创建
- `mujoco_env.controllers`：控制器
- `mujoco_env.planners`：规划与轨迹能力

## 常见问题

- 无图形界面环境建议设置：`MUJOCO_GL=egl`。
- 真机示例执行前，请先确认机器人 IP、SDK 与权限配置。
- 示例优先用于验证链路，不保证覆盖所有生产场景。
