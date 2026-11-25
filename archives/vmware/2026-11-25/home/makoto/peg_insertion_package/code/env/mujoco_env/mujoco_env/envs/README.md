# envs

`envs` 模块是环境创建入口，负责把任务配置转换为可运行的 `SimulationRobotEnv` 或 `RealRobotEnv`。

## 核心接口（技术向）

| 接口 | 作用 | 返回 |
|---|---|---|
| `make_sim_env(task, ...)` | 创建仿真环境 | `SimulationRobotEnv` |
| `make_real_env(task, robot_ip, ...)` | 创建真机环境 | `RealRobotEnv` |
| `make_env(task, mode, ...)` | 按模式统一分发创建 | `SimulationRobotEnv` / `RealRobotEnv` |
| `register_envs()` | 注册预置 Gymnasium 环境 ID | `None` |

## 目录结构

```text
mujoco_env/envs/
├── __init__.py                    # 对外导出 make_env/make_sim_env/make_real_env/register_envs
├── env_factory.py                 # 通用环境工厂（推荐主入口）
└── env_instance/                  # 预置“任务+机器人+默认参数”模板工厂
    ├── __init__.py
    ├── franka_panda_pick_cube.py
    ├── franka_fr3_simple_move.py
    ├── aubo_i5_simple_move.py
    ├── aubo_i5_pick_cube.py
    ├── aubo_i5_assemble_hole.py
    └── aubo_i5_pick_apple.py
```

## 调用链路

1. 创建 `task`（包含 `robot_config`、观测配置和任务逻辑）。
2. 调用 `make_sim_env` / `make_real_env`（或 `make_env`）。
3. 工厂函数构建 `SimulationRobotEnv` 或 `RealRobotEnv`。
4. 在环境中执行 `reset()` / `step()` 进入控制循环。

## 预置模板工厂（`env_instance`）

| 工厂函数 | 说明 | 常见示例 |
|---|---|---|
| `make_franka_panda_pick_cube_env()` | Franka Panda 抓方块模板 | `examples/demo_franka_panda_pick_cube.py` |
| `make_franka_fr3_simple_move_env()` | FR3 简单运动模板 | `examples/demo_franka_fr3_simple_move.py` |
| `make_aubo_i5_simple_move_env()` | Aubo i5 简单运动模板 | `examples/demo_aubo_i5_simple_move.py` |
| `make_aubo_i5_pick_cube_env()` | Aubo i5 抓方块模板 | `examples/demo_aubo_i5_pick_cube_rrt.py` |
| `make_aubo_i5_assemble_hole_env()` | Aubo i5 轴孔装配模板 | `examples/demo_aubo_i5_assemble_hole.py` |
| `make_aubo_i5_fruit_pick_env()` | Aubo i5 采果模板 | `examples/demo_aubo_i5_pick_apple.py` |

## 最小使用示例

### 通用工厂（推荐）

```python
from mujoco_env.mujoco_env.envs import make_env

env = make_env(task=task, mode="sim", render_mode="human")
obs, info = env.reset()
```

### 预置模板

```python
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_pick_cube_env

env = make_aubo_i5_pick_cube_env(render_mode="human")
obs, info = env.reset()
```

## 常见问题

- `mode='real'` 时必须提供 `robot_ip`，否则会抛出参数错误。
- 切换模板工厂后若控制异常，优先检查默认 `controller_type` 与动作语义是否匹配。
- 使用 `gym.make()` 前请先调用 `register_envs()`。
