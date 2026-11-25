# real

`real` 模块把不同厂商 SDK 封装为统一机器人接口，供 `RealRobotEnv` 直接调用。

## 核心接口（技术向）

### 工厂与注册（`__init__.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `get_robot_interface(robot_type, robot_ip, **kwargs)` | 按类型创建真机接口 | `RobotInterface` 子类实例 |
| `register_robot_interface(name, interface_class)` | 注册自定义真机接口 | `None` |
| `list_available_robots()` | 列出已注册机器人类型 | `list[str]` |

### 抽象协议（`robot_interface.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `connect()` / `disconnect()` | 连接与断开机器人 | `bool` / `None` |
| `get_robot_state()` | 获取完整状态（关节、TCP、力矩等） | `Dict` |
| `move_to_joint_positions(...)` | 关节空间动作下发 | `None` |
| `move_tcp_pose(...)` | TCP 位姿动作下发 | `None` |
| `stop_motion()` / `clear_errors()` | 急停与清错 | `None` |

## 支持类型速查

| `robot_type` | 对应实现 |
|---|---|
| `franka` / `franka_panda` / `panda` | `FrankaInterface` |
| `aubo` / `aubo_i5` | `AuboInterface` |

## 目录结构

```text
mujoco_env/real/
├── __init__.py          # 接口工厂与注册表
├── robot_interface.py   # 抽象协议
├── franka_interface.py  # Franka 适配实现
└── aubo_interface.py    # Aubo 适配实现
```

## 最小使用示例

```python
from mujoco_env.mujoco_env.real import get_robot_interface

robot = get_robot_interface("aubo_i5", robot_ip="192.168.1.10")
ok = robot.connect()
if ok:
    state = robot.get_robot_state()
    robot.disconnect()
```
## 调用链路

1. `make_real_env()` 根据 `task.robot_config.name` 选择 `robot_type`。
2. 通过 `get_robot_interface()` 创建具体接口实例。
3. `RealRobotEnv.step()` 调用接口下发动作并回读状态。

## 常见问题

- `robot_type` 不在注册表会直接抛错。
- 真机异常优先检查 SDK 连通性、错误清理和示教/伺服状态。
- 真机动作语义应与仿真保持一致，避免 sim2real 偏差。

## 参考

- ROS Noetic 安装指南：http://wiki.ros.org/noetic/Installation
- libfranka 文档：https://frankaemika.github.io/docs/

