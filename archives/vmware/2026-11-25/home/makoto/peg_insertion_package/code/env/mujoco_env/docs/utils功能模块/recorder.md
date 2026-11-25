# Recorder 数据记录模块

## 概述

`recorder.py` 是一个用于保存机器人数据的工具模块，提供多种数据保存功能：

- **CSV文件保存** - 通用数据保存器
- **实时数据打印** - 线程安全的日志记录
- **图像保存** - 支持RGB和深度图像
- **机器人状态保存** - 关节位置、速度、末端位姿等

## DataSaver 类

通用CSV数据保存器，适用于保存简单的结构化数据。

### 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | str | 保存的文件路径 |
| `headers` | List[str] | CSV列名列表 |

### 方法

#### `save_data(*args)`

保存一行数据。

**参数**：可变参数，将自动按顺序写入CSV各列。

**示例**：
```python
saver = DataSaver("data.csv", ["timestamp", "joint1", "joint2"])
saver.save_data(1.234, 0.5, 0.3)  # 写入一行
saver.save_data(1.235, 0.6, 0.4)  # 写入另一行
```

#### `close()`

关闭文件，确保数据写入磁盘。

---

## DataLogger 类

实时数据记录和打印器，支持线程安全。

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `print_interval` | float | 0.1 | 打印间隔时间（秒） |
| `formatter` | Callable | None | 自定义格式化函数 |

### 方法

#### `log_data(data_dict)`

记录数据到队列。

```python
logger = DataLogger(print_interval=0.5)
logger.start()
logger.log_data({"joint1": 0.5, "joint2": 0.3})
```

#### `start()`

启动打印线程。

#### `stop()`

停止数据记录。

---

## ImageSaver 类

图像保存器，支持RGB图像和深度图像。

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | - | 输出目录路径 |
| `save_rgb` | bool | True | 是否保存RGB图像 |
| `save_depth` | bool | False | 是否保存深度图像 |
| `prefix` | str | "camera" | 文件名前缀 |

### 目录结构

```
output_dir/
├── rgb/           # RGB图像保存目录
│   └── camera_000001_1234567.890.jpg
└── depth/         # 深度图像保存目录（PNG格式，保留真实深度值：米）
    └── camera_000001_1234567.890.png
```

### 方法

#### `save_image(rgb_image=None, depth_image=None, timestamp=None)`

保存图像。

**参数**：
- `rgb_image`: RGB图像数据 (numpy array, BGR格式)
- `depth_image`: 深度图像数据 (numpy array, float64格式，单位：米)
- `timestamp`: 时间戳（可选，自动生成）

**返回值**：包含保存文件路径的字典
```python
{"rgb": "path/to/rgb.jpg", "depth": "path/to/depth.png"}
```

**注意**：深度图像保存为PNG格式，保留真实的深度值（单位：米），可使用 `cv2.imread` 读取后直接使用。

**示例**：
```python
saver = ImageSaver(output_dir="./output", save_rgb=True, save_depth=True)
filenames = saver.save_image(rgb_image=cv2.imread("test.jpg"), depth_image=depth_map)
```

---

## LowDimSaver 类

低维数据保存器，专门用于存储机械臂状态数据。

### 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | str | 保存的文件路径 |
| `joint_names` | List[str] | 机械臂关节名称列表 |
| `save_fields` | List[str] | 要保存的字段列表 |

#### 可选保存字段

| 字段名 | 说明 |
|--------|------|
| `joint_pos` | 关节位置 |
| `joint_vel` | 关节速度 |
| `joint_acc` | 关节加速度 |
| `ee_pos` | 末端位置 (x, y, z) |
| `ee_quat` | 末端姿态 (四元数 w, x, y, z) |
| `ee_vel` | 末端速度 |
| `gripper_opening` | 夹爪开合程度 |

### 方法

#### `save_data(timestamp, ...)`

保存一行机器人状态数据。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float | 时间戳 |
| `joint_qpos` | List[float] | 关节位置列表 |
| `joint_qvel` | List[float] | 关节速度列表 |
| `gripper_opening` | float | 夹爪开合程度 (0-1) |
| `ee_pos` | List[float] | 末端位置 [x, y, z] |
| `ee_quat` | List[float] | 末端四元数 [w, x, y, z] |
| `image_filenames` | Dict | 图像文件名字典 |

**示例**：
```python
saver = LowDimSaver(
    filename="robot_state.csv",
    joint_names=["j1", "j2", "j3"],
    save_fields=["joint_pos", "joint_vel", "ee_pos", "ee_quat"]
)
saver.save_data(
    timestamp=time.time(),
    joint_qpos=[0.1, 0.2, 0.3],
    joint_qvel=[0.01, 0.02, 0.03],
    ee_pos=[0.5, 0.3, 0.4],
    ee_quat=[1.0, 0.0, 0.0, 0.0]
)
```

#### `close()`

关闭文件。

---

## 完整使用示例

```python
import time
import cv2
import numpy as np
from pathlib import Path
from mujoco_env.utils.recorder import ImageSaver, LowDimSaver

# 初始化保存器
output_dir = Path("output_data")
output_dir.mkdir(exist_ok=True)

image_saver = ImageSaver(
    output_dir=str(output_dir),
    save_rgb=True,
    save_depth=True,
    prefix="camera"
)

lowdim_saver = LowDimSaver(
    filename=str(output_dir / "robot_state.csv"),
    joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
    save_fields=["joint_pos", "joint_vel", "ee_pos", "ee_quat", "gripper_opening"]
)

# 模拟数据记录循环
for i in range(100):
    timestamp = time.time()
    
    # 模拟获取图像和状态
    rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    depth_image = np.random.rand(224, 224) * 2.0  # 2米范围内
    
    # 保存图像
    image_filenames = image_saver.save_image(rgb_image, depth_image, timestamp)
    
    # 保存机器人状态
    lowdim_saver.save_data(
        timestamp=timestamp,
        joint_qpos=[0.1 * i, 0.2, 0.3, 0.4, 0.5, 0.6],
        joint_qvel=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        gripper_opening=0.5 if i < 50 else 0.1,
        ee_pos=[0.5, 0.3, 0.4 + i * 0.001],
        ee_quat=[1.0, 0.0, 0.0, 0.0],
        image_filenames=image_filenames
    )

# 关闭保存器
lowdim_saver.close()
```

## CSV文件格式示例

`robot_state.csv` 文件格式：

```csv
timestamp,j1_pos,j2_pos,j3_pos,j4_pos,j5_pos,j6_pos,j1_vel,j2_vel,j3_vel,j4_vel,j5_vel,j6_vel,ee_pos_x,ee_pos_y,ee_pos_z,ee_quat_w,ee_quat_x,ee_quat_y,ee_quat_z,rgb_filename,depth_filename
1737443123.456789,0.1,0.2,0.3,0.4,0.5,0.6,0.01,0.02,0.03,0.04,0.05,0.06,0.5,0.3,0.4,1.0,0.0,0.0,0.0,output_data/camera_000001_1737443123.456.jpg,output_data/camera_000001_1737443123.456.jpg
```

## 注意事项

1. **路径处理**：所有路径建议使用 `Path` 对象处理，兼容不同操作系统
2. **文件关闭**：使用完成后务必调用 `close()` 方法
3. **图像格式**：RGB图像使用OpenCV的BGR格式
4. **深度图像**：深度值以米为单位保存
5. **线程安全**：`DataLogger` 是线程安全的，可用于多线程场景
