# robot_config

`robot_config` 模块定义机器人参数模板，是任务组装与环境创建的基础输入。

## 核心接口（技术向）

| 接口 | 作用 | 返回 |
|---|---|---|
| `get_robot(robot_name)` | 按名称获取机器人配置类 | `Type` |
| `list_robots()` | 列出注册机器人名称 | `list[str]` |
| `ROBOT_REGISTRY` | 名称到配置类映射表 | `Dict[str, Type]` |

## `RobotConfig` 关键字段（`base.py`）

| 字段 | 含义 |
|---|---|
| `robot_type` / `name` | 机器人资产类型与配置名称 |
| `dof` | 机械臂自由度 |
| `controller_type` | 控制器类型（与 `controllers` 模块对齐） |
| `mount_name` / `mount_pos` / `robot_offset` | 安装件与位姿偏移 |
| `gripper_name` | 夹爪类型 |
| `use_ft_sensor` | 力传感器开关或名称 |
| `control_freq` | 控制频率（Hz） |

说明：`action_dim` 会根据是否挂接夹爪自动增加维度。

## 目录结构

```text
mujoco_env/robot_config/
├── __init__.py      # 注册表与查询接口
├── base.py          # RobotConfig 基类
├── aubo_i5.py       # Aubo i5 配置
├── aubo_mhr.py      # Aubo MHR 配置
├── franka_panda.py  # Franka Panda 配置
├── franka_fr3.py    # Franka FR3 配置
├── ur5e.py          # UR5e 及其扩展配置
└── diana_med.py     # Diana 系列配置
```

## 常用配置类

| 类名 | 说明 |
|---|---|
| `AuboI5Robot` | Aubo i5 常用配置 |
| `AuboMHRRobot` | Aubo MHR 平台配置 |
| `FrankaPandaRobot` | Franka Panda 配置 |
| `FrankaFR3Robot` | Franka FR3 配置 |
| `UR5eRobot` | UR5e 基础配置 |
| `DianaMedRobot` | Diana 系列基础配置 |

## 新增机器人配置（最小实现）

如果你要接入一个新机器人，最小改动通常是两步：

1. 新建配置类文件（如 `mujoco_env/robot_config/my_robot.py`）。
2. 在 `mujoco_env/robot_config/__init__.py` 的 `ROBOT_REGISTRY` 注册名称。

最小模板：

```python
# mujoco_env/robot_config/my_robot.py
from .base import RobotConfig


class MyRobot(RobotConfig):
    @classmethod
    def get_config(cls, controller_type="joint_position"):
        return cls(
            name="my_robot",
            robot_type="my_robot",
            dof=6,
            controller_type=controller_type,
            mount_name="pedestal",
            gripper_name="robotiq_gripper",
            use_ft_sensor=None,
            control_freq=20,
        )
```

```python
# mujoco_env/robot_config/__init__.py
from .my_robot import MyRobot

ROBOT_REGISTRY["my_robot"] = MyRobot
```

说明：
- `mount_name/gripper_name/use_ft_sensor` 对应 `assets` 中的资产目录名。
- `controller_type` 需要与 `controllers` 里已注册的控制器名称一致。

## 最小使用示例

```python
from mujoco_env.mujoco_env.robot_config import get_robot

RobotCls = get_robot("aubo_i5")
robot_config = RobotCls.get_config(
    controller_type="joint_position",
    mount_name="pedestal",
    gripper_name="robotiq_gripper",
)
```

## 常见问题

- `mount_name/gripper_name/use_ft_sensor` 必须与资产目录名一致。
- 切换控制器类型后需同步检查动作维度与增益参数。
- 新机器人接入建议顺序：`robot_config` → `tasks` → `envs/env_instance`。
