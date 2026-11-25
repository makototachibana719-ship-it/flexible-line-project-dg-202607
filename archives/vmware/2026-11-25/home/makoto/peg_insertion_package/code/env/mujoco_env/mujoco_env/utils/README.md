# utils

`utils` 模块提供通用工程能力：模型组装、渲染、传感、记录、坐标变换与交互调试。

## 核心组件速查

| 组件 | 作用 | 主要接口 |
|---|---|---|
| `robot_assembler.py` | 组装场景+机器人+挂件 XML（主链路） | `RobotAssembler.build_robot_scene(...)` |
| `renderers.py` | MuJoCo 渲染封装 | `MjRenderer` |
| `sensor_utils.py` | 力传感与碰撞辅助 | `ForceSensorMonitor` |
| `recorder.py` | 低维与图像数据记录 | `DataSaver` / `ImageSaver` / `LowDimSaver` |
| `transform.py` | 位姿/旋转变换工具 | `euler_2_quat` / `quat_2_mat` / `pose_inv` 等 |
| `keyboard.py` | 键盘交互输入 | `KeyboardIO` |
| `camera.py` | 双目相机封装 | `StereoCamera` |

## 目录结构

```text
mujoco_env/utils/
├── __init__.py          # 常用工具导出入口
├── robot_assembler.py   # XML 组装器（当前主实现）
├── xml_splice.py        # 历史组装实现（兼容保留）
├── renderers.py         # MuJoCo 渲染封装（human/rgb/depth）
├── sensor_utils.py      # 力/力矩传感器读取与监控
├── recorder.py          # 低维状态与图像数据保存
├── camera.py            # 双目相机与深度估计工具
├── transform.py         # 位姿、四元数、旋转矩阵变换
├── cv_utils.py          # OpenCV 显示/相机参数辅助函数
├── plot_utils.py        # 运行期曲线绘图与可视化辅助
├── keyboard.py          # 键盘遥操作输入封装
├── make_datasets.py     # 数据集整理与导出工具
└── utils.py             # 通用数学/数据处理小工具
```

## 最常用流程

### 1) 任务初始化时组装 XML

```python
from pathlib import Path
from mujoco_env.mujoco_env.utils.robot_assembler import RobotAssembler

assembler = RobotAssembler(Path("mujoco_env/assets"))
xml_path = assembler.build_robot_scene(scene_name, robot_config, output_dir)
```

### 2) 运行中记录低维/图像数据

```python
from mujoco_env.mujoco_env.utils.recorder import LowDimSaver, ImageSaver

lowdim = LowDimSaver(save_dir="./data")
images = ImageSaver(save_dir="./data")
```

## 使用建议

- `robot_assembler.py` 是当前主链路；`xml_splice.py` 仅用于兼容旧逻辑。
- 若使用无图形环境，渲染相关工具需配合离屏配置（如 EGL）。
- `__init__.py` 以 re-export 为主，`RobotAssembler` 建议显式从子模块导入。

## 常见问题

- 组装后 XML 路径异常：优先检查资产相对路径和输出目录。
- 渲染无画面：检查图形后端与 MuJoCo 渲染模式。
- 记录数据过大：长时间采集时请控制保存频率与图像分辨率。
