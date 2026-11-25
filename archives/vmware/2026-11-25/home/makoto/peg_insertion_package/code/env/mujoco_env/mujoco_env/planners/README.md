# planners

`planners` 模块提供“离散路径规划 + 连续轨迹插值”能力，用于把目标点转换为可执行轨迹。

## 核心接口（技术向）

| 接口 | 作用 | 返回 |
|---|---|---|
| `get_planner(planner_name)` | 按名称获取规划器类 | `Type` |
| `get_interpolator(interpolator_name)` | 按名称获取插值器类 | `Type` |
| `list_planners()` | 列出可用规划器名称 | `list[str]` |
| `list_interpolators()` | 列出可用插值器名称 | `list[str]` |

## 类型速查

### 规划器

| 名称 | 对应类 | 说明 |
|---|---|---|
| `rrt` / `rrt_star` | `RRT` | 3D 采样规划，支持可选碰撞检测 |

### 插值器

| 名称 | 对应类 | 说明 |
|---|---|---|
| `otg` / `joint` / `joint_space` | `OTG` | 关节空间在线轨迹生成（Ruckig） |
| `linear` / `cartesian` / `task_space` | `LinearInterpolator` | 笛卡尔空间线性插值（Ruckig） |

## 目录结构

```text
mujoco_env/planners/
├── __init__.py            # 注册表与工厂函数
├── rrt.py                 # RRT* 路径规划
├── otg_interpolator.py    # 关节空间插值（OTG）
└── linear_interpolator.py # 笛卡尔空间插值
```

## 最小使用示例

### RRT 路径规划

```python
from mujoco_env.mujoco_env.planners.rrt import RRT

planner = RRT(
    start=[0.3, 0.0, 0.2],
    goal=[0.5, 0.2, 0.2],
    play_area=[-1, 1, -1, 1, 0, 1.5],
    sim=None,  # None 表示不做碰撞检测
)
path = planner.planning(animation=False)
```

### OTG 关节插值

```python
from mujoco_env.mujoco_env.planners.otg_interpolator import OTG

otg = OTG(OTG_dim=7, control_cycle=0.005)
otg.set_params(qpos=current_q, qvel=current_dq)
otg.update_target_position(target_q)
q_next, dq_next = otg.update_state()
```

## 调用链路

1. 规划阶段生成离散路径（例如 `RRT.planning()`）。
2. 插值阶段将离散目标转为连续轨迹点。
3. 控制器按控制周期读取轨迹并执行。

## 常见问题

- `RRT` 传入 `sim=None` 时不会进行碰撞检测。
- 插值周期应与控制周期一致或为其整数倍。
- 路径规划失败时优先检查目标可达性、搜索边界和采样参数。

## 参考论文与资料

- Karaman, S., & Frazzoli, E. *Sampling-based Algorithms for Optimal Motion Planning*. IJRR, 2011.
- Ruckig 项目文档：https://github.com/pantor/ruckig
