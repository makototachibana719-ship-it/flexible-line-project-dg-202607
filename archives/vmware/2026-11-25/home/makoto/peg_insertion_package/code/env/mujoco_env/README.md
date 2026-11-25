# mujoco-env

面向机器人任务开发的 MuJoCo 仿真与控制框架，提供统一的 Gymnasium 风格接口，覆盖仿真环境、真机接口、控制器与规划器。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.1.5](https://img.shields.io/badge/MuJoCo-3.1.5-green.svg)](https://mujoco.org/)

## 你可以用它做什么

- 用同一套接口切换仿真（`sim`）与真机（`real`）工作流。
- 组合机器人配置（`robot_config`）、任务（`tasks`）、控制器（`controllers`）与规划器（`planners`）。
- 直接运行 Aubo、Franka 等示例脚本验证控制与规划链路。
- 使用 OpenPI 相关示例完成数据采集、推理与格式转换。

## 安装

### 环境要求

- Python `>=3.10`
- Linux（建议）
- MuJoCo 运行环境可用（无头环境建议使用 EGL）

### 安装步骤

```bash
cd path/to/mujoco-env
conda create -n mujoco-env python=3.10 -y
conda activate mujoco-env
pip install -r requirements.txt
```

## 快速开始

运行一个最小仿真示例（Aubo MHR 被动场景）：

```bash
cd path/to/mujoco-env
python examples/demo_aubo_mhr_scene_sim.py
```

预期结果：
- 正常弹出 MuJoCo 画面并加载场景模型。
- 控制台无导入错误或资源路径错误。

## 常用示例

```bash
# Aubo i5 + RRT 路径规划
python examples/demo_aubo_i5_pick_cube_rrt.py

# Franka Panda 抓方块
python examples/demo_franka_panda_pick_cube.py

# 真机接口（需要机器人、网络与 SDK）
python examples/demo_real_robot_interface.py
```

## 项目结构

- `mujoco_env/`：核心包
  - `core/`：环境基类、仿真环境、真机环境
  - `envs/`：环境工厂与预定义环境
  - `tasks/`：任务定义与任务配置
  - `controllers/`：关节/任务空间控制器
  - `planners/`：RRT 与轨迹插值
  - `robot_config/`：机器人配置注册
  - `real/`：真机接口适配
  - `assets/`：模型、场景、纹理、物体
  - `utils/`：装配、渲染、传感器、记录工具
- `examples/`：可直接运行的示例脚本
- `tests/`：测试代码
- `docs/`：补充文档

### 如何基于项目架构新建一个完整 Demo

下面是按本项目真实调用链整理的推荐流程（仿真场景）：

1. **在 `assets/` 准备资产与场景**
   - 场景放在 `mujoco_env/assets/scenes/<scene>.xml`。
   - 机器人/底座/夹爪/传感器放在 `mujoco_env/assets/models/` 对应目录。
   - 任务物体放在 `mujoco_env/assets/objects/`。

2. **在 `robot_config/` 定义或复用机器人配置**
   - 明确 `controller_type`、`mount_name`、`gripper_name`、`use_ft_sensor` 等关键字段。
   - 若是新机器人类型，在 `mujoco_env/robot_config/__init__.py` 注册到 `ROBOT_REGISTRY`。

3. **在 `tasks/` 定义任务逻辑**
   - 新建 `BaseTask` 子类，实现 `sample_goal`、`get_achieved_goal`、`compute_reward`、`is_success_fn`。
   - `BaseTask` 初始化时会调用 `utils/robot_assembler.py`，自动将场景与机器人资产拼接为完整 XML，并写入 `task.model_path`。

4. **通过 `envs` 创建环境（由 `core/sim_env` 加载 XML）**
   - 在 `mujoco_env/envs/env_factory.py` 中使用 `make_sim_env(task=...)`。
   - `SimulationRobotEnv` 会读取 `task.model_path`，并根据 `task.robot_config.controller_type` 创建控制器。

5. **在 `examples/` 编写 Demo 主循环**
   - 典型循环：`reset -> 生成动作 -> step -> 终止判断`。
   - 动作来源可选：直接控制器目标、策略网络输出，或 `planners` 生成路径后逐步执行。
   - 需要记录数据时可接入 `utils/recorder.py`、`utils/camera.py`。

6. **（可选）沉淀为可复用入口**
   - 在 `mujoco_env/envs/env_instance/` 增加 `make_xxx_env` 工厂函数，统一默认参数。
   - 在 `examples/` 增加对应脚本，作为可直接运行的标准演示。

最小模板（结构示意）：

```python
from mujoco_env.mujoco_env.envs import make_sim_env
from xxx.robot_config import MyRobotConfig
from xxx.tasks import MyTask

robot_config = MyRobotConfig.get_config(...)
task = MyTask(
    robot_config=robot_config,
    scene_name="my_scene",
    include_image=False,
    include_depth=False,
)

env = make_sim_env(
    task=task,
    render_mode="human",
    max_episode_steps=500,
)

obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()  # 或替换为规划器/策略输出
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## 文档导航

- [`tests/README.md`](tests/README.md)：测试运行说明
- [`examples/README.md`](examples/README.md)：示例脚本清单与入口
- [`examples/demo_aubo_i5_openpi/README.md`](examples/demo_aubo_i5_openpi/README.md)：OpenPi部署流程
- [`mujoco_env/core/README.md`](mujoco_env/core/README.md)：环境层说明
- [`mujoco_env/envs/README.md`](mujoco_env/envs/README.md)：环境工厂与预置实例说明
- [`mujoco_env/controllers/README.md`](mujoco_env/controllers/README.md)：控制器说明
- [`mujoco_env/planners/README.md`](mujoco_env/planners/README.md)：规划与插值说明
- [`mujoco_env/tasks/README.md`](mujoco_env/tasks/README.md)：任务模块说明
- [`mujoco_env/robot_config/README.md`](mujoco_env/robot_config/README.md)：机器人配置说明
- [`mujoco_env/real/README.md`](mujoco_env/real/README.md)：真机接口说明
- [`mujoco_env/assets/README.md`](mujoco_env/assets/README.md)：资产目录总览
- [`mujoco_env/utils/README.md`](mujoco_env/utils/README.md)：工具模块说明

##  贡献

欢迎贡献！请遵循：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Merge Request

---

##  许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

##  致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真引擎
- [Gymnasium](https://gymnasium.farama.org/) - 强化学习标准接口
- [Ruckig](https://github.com/pantor/ruckig) - 在线轨迹生成
- [SERL](https://github.com/rail-berkeley/serl) - 样本高效强化学习框架

### 算法参考论文

- Karaman, S., & Frazzoli, E. *Sampling-based Algorithms for Optimal Motion Planning*. The International Journal of Robotics Research (IJRR), 2011.


---

**注意**: 本项目专注于机器人强化学习训练环境，主要支持 **Franka Panda** 和 **Aubo i5** 机器人。真机部署功能已验证可用，请严格遵守安全规范。

**Note**: This project focuses on robot reinforcement learning training environments, with primary support for **Franka Panda** and **Aubo i5** robots. Real robot deployment has been validated, please strictly follow safety regulations.
