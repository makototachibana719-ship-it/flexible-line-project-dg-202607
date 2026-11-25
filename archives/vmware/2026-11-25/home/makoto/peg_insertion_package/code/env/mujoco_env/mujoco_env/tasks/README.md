# tasks

`tasks` 模块定义任务语义：目标采样、奖励计算、成功判定、任务观测与场景组装。

## 核心接口（技术向）

### 注册与工厂（`__init__.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `get_task(task_name, **kwargs)` | 按名称创建任务实例 | `BaseTask` 子类实例 |
| `register_task(name, task_class)` | 注册自定义任务类 | `None` |
| `TASK_REGISTRY` | 任务名称映射表 | `Dict[str, Type[BaseTask]]` |

### 抽象基类（`base_task.py`）

| 接口 | 作用 | 返回 |
|---|---|---|
| `compute_reward(...)` | 奖励计算（必须实现） | `float` |
| `is_success_fn(...)` | 成功判定（必须实现） | `bool` |
| `sample_goal()` | 目标采样（必须实现） | `np.ndarray` |
| `get_achieved_goal(obs)` | 从观测提取 achieved goal（必须实现） | `np.ndarray` |
| `reset()` / `step(obs)` | 任务级周期管理 | `Dict` / `(reward, done, info)` |

## 任务速查

### 当前注册到 `TASK_REGISTRY` 的任务

| 名称 | 对应类 |
|---|---|
| `pick_cube` / `pick` / `cube` | `PickCubeTask` |
| `pcb_insertion` / `pcb` | `PCBInsertionTask` |
| `peg_insertion` / `peg` / `peg_in_hole` | `PegInsertionTask` |

### 目录中其他任务/配置模块

- `single_move/`、`fruit_pick/`、`mhr_sim/`、`auboi5_agv/`：更多任务实现或配置模板，可被示例和环境工厂直接使用。

## 目录结构

```text
mujoco_env/tasks/
├── __init__.py
├── base_task.py
├── single_move/
├── pick_cube/
├── peg_insertion/
├── pcb_insertion/
├── fruit_pick/
├── mhr_sim/
└── auboi5_agv/
```

## 调用链路

1. 创建任务实例（`get_task(...)` 或直接实例化任务类）。
2. `BaseTask.__init__` 通过 `RobotAssembler` 组装模型并生成 `task.model_path`。
3. 环境在 `step()` 周期中调用任务接口计算奖励与终止信息。

## 新增 Task

新增Task的核心流程可以概括为：
- `reset()`：初始化任务状态与目标（例如从场景解析目标位姿）。
- `step(obs)`：基于当前观测计算奖励、终止条件并维护 `info`。
- `compute_reward` / `is_success_fn`：定义任务语义。
- `sample_goal` / `get_achieved_goal`：定义 goal-based 接口。

最小实现建议按下面 3 步：

1. 新建任务类文件（如 `mujoco_env/tasks/my_task/my_task.py`），继承 `BaseTask`。
2. 实现抽象接口：`compute_reward`、`is_success_fn`、`sample_goal`、`get_achieved_goal`。
3. 在 `mujoco_env/tasks/__init__.py` 注册到 `TASK_REGISTRY`。

最小代码骨架：

```python
from typing import Dict, Any
import numpy as np
from gymnasium import spaces

from ..base_task import BaseTask
from ...robot_config.base import RobotConfig


class MyTask(BaseTask):
    DEFAULT_SCENE = "my_scene"

    def __init__(
        self,
        robot_config: RobotConfig,
        scene_name: str = "my_scene",
        include_image: bool = False,
        image_size: tuple = (128, 128),
        include_depth: bool = False,
        distance_threshold: float = 0.02,
        **kwargs,
    ):
        super().__init__(
            name="my_task",
            robot_config=robot_config,
            scene_name=scene_name,
            include_image=include_image,
            image_size=image_size,
            include_depth=include_depth,
            **kwargs,
        )
        self.distance_threshold = distance_threshold
        self._desired_goal = None

    def reset(self) -> Dict[str, Any]:
        super().reset()
        self._desired_goal = self.sample_goal()
        return {"desired_goal": self._desired_goal.copy()}

    def step(self, observation: Dict[str, np.ndarray]):
        achieved_goal = self.get_achieved_goal(observation)
        desired_goal = observation.get("desired_goal", self._desired_goal)
        reward = self.compute_reward(achieved_goal, desired_goal, {"obs": observation})
        terminated = self.is_success_fn(achieved_goal, desired_goal)
        self.current_step += 1
        info = {"is_success": terminated, "task_step": self.current_step}
        return reward, terminated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return float(np.exp(-10.0 * dist))

    def is_success_fn(self, achieved_goal, desired_goal) -> bool:
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return bool(dist < self.distance_threshold)

    def sample_goal(self) -> np.ndarray:
        return np.array([0.5, 0.0, 0.3], dtype=np.float32)

    def get_achieved_goal(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return observation.get("tcp_pos", np.zeros(3, dtype=np.float32)).copy()

    def get_obs_space(self) -> spaces.Dict:
        return spaces.Dict({
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })
```

注册示例：

```python
# mujoco_env/tasks/__init__.py
from .my_task.my_task import MyTask
TASK_REGISTRY["my_task"] = MyTask
```

## 常见问题

- `scene_name` 需与 `assets/scenes/*.xml` 文件名匹配。
- 新任务应完整实现 `BaseTask` 的抽象接口，避免运行期报错。
- 奖励函数与成功判定应语义一致，避免训练目标冲突。
