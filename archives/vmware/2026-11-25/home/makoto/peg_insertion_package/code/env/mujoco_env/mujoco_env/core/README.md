# core

`core` 是环境层核心模块，提供统一的 Gymnasium 接口，屏蔽仿真与真机执行差异。

## 功能概览

- 抽象环境生命周期：`reset()`、`step()`、`render()`、`close()`。
- 提供仿真环境实现：`SimulationRobotEnv`（MuJoCo）。
- 提供真机环境实现：`RealRobotEnv`（机器人接口驱动）。
- 统一奖励与终止判断入口，任务逻辑由 `task` 注入。

## 目录结构

```text
mujoco_env/core/
├── __init__.py      # 导出核心环境类
├── base_env.py      # 抽象基类 BaseRobotEnv
├── sim_env.py       # 仿真环境 SimulationRobotEnv
└── real_env.py      # 真机环境 RealRobotEnv
```

## 快速开始

如果你只想快速验证环境链路，直接运行示例脚本：

```bash
cd path/to/mujoco-env
python examples/demo_aubo_mhr_scene_sim.py
```

如果你要在代码中使用统一工厂：

```python
from mujoco_env import make_sim_env
from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot

robot_config = AuboI5Robot.get_config(controller_type="joint_position")
task = PickCubeTask(robot_config=robot_config, scene_name="grasping")
env = make_sim_env(task=task, render_mode="human", max_episode_steps=500)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## API 参考

### `BaseRobotEnv`（`base_env.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `reset(seed=None, options=None)` | 重置环境状态 | `(obs, info)` |
| `step(action)` | 执行一步控制 | `(obs, reward, terminated, truncated, info)` |
| `render()` | 按渲染模式输出画面 | 图像数组或 `None` |
| `close()` | 释放资源 | `None` |
| `_get_obs()` | 构建当前观测 | `Dict[str, np.ndarray]` |
| `_compute_reward(obs=None)` | 计算当前奖励 | `float` |
| `_check_terminated(obs=None)` | 判断任务终止 | `bool` |
| `_check_truncated()` | 判断是否截断 | `bool` |

### `SimulationRobotEnv`（`sim_env.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `reset()` | 重置 MuJoCo 数据、任务状态与控制器 | `(obs, info)` |
| `step(action)` | 设置动作并执行多次物理子步 | `(obs, reward, terminated, truncated, info)` |
| `render()` | 支持 `human` / `rgb_array` / `depth` 渲染 | 图像数组或 `None` |
| `close()` | 关闭渲染器并释放仿真资源 | `None` |
| `_get_obs()` | 组合机器人状态、任务观测与可选图像/深度 | `Dict[str, np.ndarray]` |
| `_compute_reward()` | 委托 `task` 计算奖励（兼容旧接口） | `float` |
| `_check_terminated()` | 委托 `task` 判断成功/失败（兼容旧接口） | `bool` |
| `_check_truncated()` | 使用 `max_episode_steps` 在 `step()` 中判定 | `bool`（在 `step()` 中计算） |

扩展接口（仿真专属）：
- `save_state()`：保存 MuJoCo 积分状态。
- `load_state(state)`：恢复状态。
- `get_state()`：获取当前状态副本。
- `is_contact(geom1, geom2)`：检查接触关系。

### `RealRobotEnv`（`real_env.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `reset()` | 清理机器人错误、重置任务并读取初始状态 | `(obs, info)` |
| `step(action)` | 下发动作并按 `control_freq` 节拍执行 | `(obs, reward, terminated, truncated, info)` |
| `render()` | 默认无渲染（可扩展外部相机） | `None` |
| `close()` | 停止运动并断开机器人连接 | `None` |
| `_get_obs()` | 从机器人接口读取关节、TCP、力矩等观测 | `Dict[str, np.ndarray]` |
| `_compute_reward()` | 委托 `task` 计算奖励（兼容旧接口） | `float` |
| `_check_terminated()` | 委托 `task` 判断任务完成（兼容旧接口） | `bool` |
| `_check_truncated()` | 使用 `current_step` 与 `max_episode_steps` 判定 | `bool`（在 `step()` 中计算） |

## 执行流程

1. 创建 `task`（包含 `robot_config`、观测配置和任务逻辑）。
2. 通过工厂创建环境（`make_sim_env()` 或 `make_real_env()`）。
3. 调用 `reset()` 获得初始观测。
4. 在循环中调用 `step(action)` 获取反馈。
5. 完成后调用 `close()` 释放资源。

## 常见问题

### 仿真步进抖动或不稳定

优先检查 `control_dt` 与 `physics_dt` 的比例是否合理，通常应保证 `control_dt >= physics_dt` 且整除关系明确。

### 动作维度与控制器不匹配

检查 `task.robot_config.controller_type` 与 `action_space` 是否一致；不同控制器对动作含义不同。

### 真机连接失败

先验证机器人 IP、SDK 与权限；确认 `robot_interface.connect()` 成功后再执行 `reset()`/`step()`。
