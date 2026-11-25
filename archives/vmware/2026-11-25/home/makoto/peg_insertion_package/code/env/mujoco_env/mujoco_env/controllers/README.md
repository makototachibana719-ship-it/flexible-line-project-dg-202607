# controllers

`controllers` 模块负责将任务/策略输出的动作目标转换为可执行控制量，并在每个控制周期输出到执行端（仿真中通常为 `data.ctrl`）。

## 核心接口（技术约定）

### 工厂与注册（`__init__.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `get_controller(controller_type, **kwargs)` | 根据类型创建控制器实例 | `BaseController` 子类实例 |
| `register_controller(name, controller_class)` | 注册自定义控制器 | `None` |
| `CONTROLLER_REGISTRY` | 控制器类型到类的映射表 | `Dict[str, Type[BaseController]]` |

### 抽象基类（`base_controller.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `compute_control(target, current_state=None)` | 计算当前控制输出（子类必须实现） | `np.ndarray` |
| `reset()` | 重置内部状态（可选覆写） | `None` |
| `get_state()` | 读取当前关节状态（`qpos/qvel`） | `Dict[str, np.ndarray]` |
| `set_gains(**kwargs)` | 动态设置增益（可选覆写） | `None` |

## 控制器类型速查

### 关节空间

| `controller_type` | 对应类 | 目标输入约定 |
|---|---|---|
| `joint_position` / `position` | `JointPositionController` | 目标关节位置 `(dof,)` |
| `joint_velocity` / `velocity` | `JointVelocityController` | 目标关节速度 `(dof,)` |
| `joint_torque` / `torque` | `JointTorqueController` | 目标关节力矩 `(dof,)` |
| `joint_impedance` / `impedance` | `JointImpedanceController` | 目标关节位置 `(dof,)` |
| `joint_freedrive` / `freedrive` | `JointFreedriveController` | 自由驱动模式，目标通常忽略 |

### 任务空间

| `controller_type` | 对应类 | 目标输入约定 |
|---|---|---|
| `operational_space` / `osc` / `task_space` | `OperationalSpaceController` | 常用 `(7,)` 位姿或 `(3,)` 位置 |
| `cartesian_impedance` / `cart_imp` | `CartesianImpedanceController` | `(7,)` 位姿或 `(6,)` 位姿（欧拉） |
| `cartesian_ik` / `cart_ik` | `CartesianIKController` | `(7,)` 位姿或 `(3,)` 仅位置 |
| `admittance` / `cart_adm` | `AdmittanceController` | `(7,)` 位姿（可结合外力） |

## 调用链路

1. `SimulationRobotEnv` 初始化时读取 `robot_config.controller_type`。
2. 通过 `get_controller()` 创建控制器实例。
3. 每个控制周期调用 `compute_control()` 生成控制量。
4. 控制量写入执行端并推进仿真/硬件状态。

## 新增控制器（推荐流程）

1. 新建类并继承 `BaseController`（任务空间建议继承 `TaskSpaceController`）。
2. 至少实现 `compute_control()`，必要时实现 `reset()`/`set_gains()`。
3. 在 `controllers/__init__.py` 中注册到 `CONTROLLER_REGISTRY`。
4. 在任务配置里将 `controller_type` 切换为新名称并联调。

## 常见问题

- `ee_site_name` / `site_name` 不存在：优先检查 XML 中 site/body 命名是否一致。
- 动作维度不匹配：检查上层动作语义是否与控制器输入约定一致。
- 切换控制器后行为异常：同步检查增益参数与 `dof`、执行器映射。
