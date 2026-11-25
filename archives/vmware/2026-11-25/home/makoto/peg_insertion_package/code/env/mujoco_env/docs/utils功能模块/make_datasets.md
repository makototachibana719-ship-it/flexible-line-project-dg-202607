# MakeDataset 数据集创建模块

## 概述

`make_datasets.py` 是一个用于创建符合 RoboMimic 格式数据集的工具模块，提供以下功能：

- **随机数据生成** - 生成测试用的随机数据集
- **真实数据转换** - 将录制的数据转换为 HDF5 格式
- **HDF5 文件创建** - 支持 gzip 和 blosc 压缩

## MakeDataset 类

数据集创建器，用于构建符合 RoboMimic 格式的 HDF5 数据集。

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `robot_name` | str | "Panda" | 机器人名称 |
| `camera_name` | str | "robot0_eye_in_hand" | 相机名称 |
| `compression` | str | "gzip" | 压缩方式（"gzip" 或 "blosc"） |

### 压缩方式说明

| 压缩方式 | 说明 | 依赖 |
|----------|------|------|
| `gzip` | 默认压缩方式，兼容性好 | 无需额外依赖 |
| `blosc` | 更高效的压缩方式 | 需要安装 `hdf5plugin` |

**安装 blosc 依赖**：
```bash
# 该依赖库仅在需要使用blosc压缩时安装
pip install hdf5plugin
```

### 方法

#### `create_episode(obs, actions_relevant, actions_abs, rewards)`

构建完整的 episode 数据。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `obs` | Dict | 观测数据字典 |
| `actions_relevant` | np.ndarray | 相关动作数据 |
| `actions_abs` | np.ndarray | 绝对动作数据 |
| `rewards` | np.ndarray | 奖励数据 |

**返回值**：包含完整 episode 数据的字典

**示例**：
```python
dataset_maker = MakeDataset(robot_name="Panda", compression="gzip")
episode = dataset_maker.create_episode(
    obs={'image': rgb_images, 'depth': depth_images},
    actions_relevant=actions_rel,
    actions_abs=actions_abs,
    rewards=rewards
)
```

#### `create_hdf5_file(episodes_data, output_path)`

创建 HDF5 文件并写入数据。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `episodes_data` | List[Dict] | 所有 episode 数据列表 |
| `output_path` | str | 输出文件路径 |

**HDF5 数据结构**：
```
dataset.hdf5
└── data/
    ├── episode_0/
    │   ├── obs/
    │   │   ├── robot0_eye_in_hand_image    # RGB图像 (T, H, W, 3)
    │   │   ├── robot0_eye_in_hand_depth_image  # 深度图像 (T, H, W, 1)
    │   │   ├── robot0_eef_pos              # 末端位置 (T, 3)
    │   │   ├── robot0_eef_quat             # 末端四元数 (T, 4)
    │   │   ├── robot0_gripper_qpos         # 夹爪位置 (T, 1)
    │   │   └── target_position             # 目标位置 (T, 3)
    │   ├── actions_relevant                # 相关动作 (T, 8)
    │   ├── actions_abs                     # 绝对动作 (T, 8)
    │   ├── rewards                         # 奖励 (T,)
    │   ├── dones                          # 终止标志 (T,)
    │   ├── num_samples                     # 时间步数量
    │   └── model_file                      # 模型文件
    └── episode_1/
        └── ...
```

**示例**：
```python
dataset_maker = MakeDataset(compression="blosc")
dataset_maker.create_hdf5_file(episodes_data, "output/dataset.hdf5")
```

---

## create_random_dataset 函数

使用随机数据创建符合标准 RoboMimic 格式的 HDF5 数据集，用于测试验证。

### 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | "./datasets" | 输出目录 |
| `num_demos` | int | 2 | 演示数量 |
| `image_height` | int | 224 | 图像高度 |
| `image_width` | int | 224 | 图像宽度 |
| `action_dim` | int | 8 | 动作维度 |
| `compression` | str | "gzip" | 压缩方式 |

### 返回值

生成的 HDF5 文件路径（str）

### 示例

```python
from mujoco_env.utils.make_datasets import create_random_dataset

# 创建随机数据集
output_path = create_random_dataset(
    output_dir="./datasets",
    num_demos=5,
    image_height=224,
    image_width=224,
    action_dim=8,
    compression="gzip"
)
print(f"数据集已创建: {output_path}")
```

---

## create_real_dataset 函数

从真实录制数据创建符合标准 RoboMimic 格式的 HDF5 数据集。

### 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_dir` | str | - | 输入数据目录 |
| `output_dir` | str | "./datasets" | 输出目录 |
| `sample_rate` | int | 1 | 采样率，每隔 N 个时间步采样一次 |
| `compression` | str | "gzip" | 压缩方式，有gzip和blosc两种可选 |

### 输入数据格式

`input_dir` 目录结构：
```
input_dir/
├── episode_0/
│   ├── robot_state.csv    # 机器人状态数据
│   ├── rgb/              # RGB图像目录
│   │   ├── rgb_000001_xxx.jpg
│   │   └── rgb_000002_xxx.jpg
│   └── depth/            # 深度图像目录
│       ├── depth_000001_xxx.png
│       └── depth_000002_xxx.png
└── episode_1/
    └── ...
```

`robot_state.csv` 文件格式：
```csv
timestamp,joint1_pos,joint2_pos,...,ee_pos_x,ee_pos_y,ee_pos_z,ee_quat_w,ee_quat_x,ee_quat_y,ee_quat_z,gripper_opening,rgb_filename,depth_filename
1737443123.456789,0.1,0.2,...,0.5,0.3,0.4,1.0,0.0,0.0,0.0,0.5,outputs/exp_0/rgb/xxx.jpg,outputs/exp_0/depth/xxx.png
```

### 返回值

生成的 HDF5 文件路径（str）

### 示例

```python
from mujoco_env.utils.make_datasets import create_real_dataset

# 创建真实数据集
output_path = create_real_dataset(
    input_dir="outputs/outputs_20260124_094640",
    output_dir="./datasets",
    sample_rate=1,
    compression="gzip"
)
print(f"数据集已创建: {output_path}")
```

### 采样率说明

- `sample_rate=1`：保留所有帧
- `sample_rate=2`：每隔一帧取一帧
- `sample_rate=5`：每隔 4 帧取一帧（数据量减少为 1/5）

---

## 命令行使用

### 基本用法

```bash
# 使用真实数据创建数据集
python mujoco_env/utils/make_datasets.py --mode real --input-dir outputs/outputs_20260124_094640

# 使用随机数据创建数据集
python mujoco_env/utils/make_datasets.py --mode random --num-demos 2
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | "random" | 数据集创建模式（"random" 或 "real"），**必填** |
| `--input-dir` | str | - | 输入数据目录（real 模式必填） |
| `--output-dir` | str | "datasets" | 输出数据集目录 |
| `--num-demos` | int | 2 | 演示数量（random 模式） |
| `--sample-rate` | int | 1 | 采样率（real 模式） |
| `--compression` | str | "gzip" | 压缩方式（"gzip" 或 "blosc"） |

### 完整示例

```bash
# 真实数据，带采样
python mujoco_env/utils/make_datasets.py \
    --mode real \
    --input-dir outputs/outputs_20260124_094640 \
    --output-dir ./my_datasets \
    --sample-rate 2 \
    --compression blosc

# 随机数据
python mujoco_env/utils/make_datasets.py \
    --mode random \
    --num-demos 5 \
    --output-dir ./test_datasets \
    --compression gzip
```

---

## 完整使用示例

### 示例1：使用随机数据生成数据集（了解数据结构）

此示例展示如何手动构建符合 RoboMimic 格式的数据集，用于理解数据结构和测试。

```python
import numpy as np
from mujoco_env.utils.make_datasets import MakeDataset

# 初始化数据集创建器
dataset_maker = MakeDataset(
    robot_name="Panda",
    camera_name="robot0_eye_in_hand",
    compression="gzip"
)

# 准备一个 episode 的随机数据
T = 100  # 时间步数

# 1. 构建观测数据 (obs)
obs = {
    'robot0_eye_in_hand_image': np.random.randint(0, 255, (T, 224, 224, 3), dtype=np.uint8),
    'robot0_eye_in_hand_depth_image': np.random.rand(T, 224, 224, 1).astype(np.float32),
    'robot0_eef_pos': np.random.randn(T, 3).astype(np.float64),
    'robot0_eef_quat': np.random.randn(T, 4).astype(np.float64),
    'robot0_gripper_qpos': np.random.randn(T, 1).astype(np.float64),
    'target_position': np.random.randn(T, 3).astype(np.float64)
}

# 2. 构建动作数据
actions_relevant = np.random.randn(T, 8).astype(np.float64)  # 相关动作
actions_abs = np.random.randn(T, 8).astype(np.float64)       # 绝对动作

# 3. 构建奖励数据
rewards = np.random.randn(T).astype(np.float64)

# 4. 创建 episode
episode = dataset_maker.create_episode(obs, actions_relevant, actions_abs, rewards)

# 5. 保存为 HDF5 文件
dataset_maker.create_hdf5_file([episode], "random_dataset.hdf5")

print("✅ 随机数据集创建完成！")
```

**obs 字段说明：**

| 字段名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `robot0_eye_in_hand_image` | (T, 224, 224, 3) | uint8 | RGB 图像 |
| `robot0_eye_in_hand_depth_image` | (T, 224, 224, 1) | float32 | 深度图像（米） |
| `robot0_eef_pos` | (T, 3) | float64 | 末端执行器位置 (x, y, z) |
| `robot0_eef_quat` | (T, 4) | float64 | 末端执行器姿态（四元数 w, x, y, z） |
| `robot0_gripper_qpos` | (T, 1) | float64 | 夹爪关节位置 |
| `target_position` | (T, 3) | float64 | 目标位置 |

### 示例2：使用真实录制数据创建数据集

此示例展示如何从实际录制的机器人数据创建数据集。

```python
from mujoco_env.utils.make_datasets import create_real_dataset

# 从录制数据创建 HDF5 数据集
output_path = create_real_dataset(
    input_dir="outputs/outputs_20260124_094640",  # 录制数据目录
    output_dir="./datasets",                       # 输出目录
    sample_rate=1,                                 # 采样率（1=保留所有帧）
    compression="gzip"                             # 压缩方式
)

print(f"✅ 数据集创建完成: {output_path}")
```

**命令行方式：**
```bash
python mujoco_env/utils/make_datasets.py \
    --mode real \
    --input-dir outputs/outputs_20260124_094640 \
    --output-dir ./datasets \
    --sample-rate 1 \
    --compression gzip
```

---

## 注意事项

1. **压缩方式选择**：
   - 小规模数据集可以使用 `gzip`，无需额外依赖
   - 数据量较大时建议使用 `blosc`，压缩速度和写入速度更快

2. **图像路径**：
   - CSV 中的图像路径应为相对于项目根目录的路径
   - 格式如：`outputs/exp_0/rgb/xxx.jpg`

3. **采样率**：
   - 较大的 `sample_rate` 可以减少数据量，但会降低数据精度
   - 根据实际需求选择合适的采样率

4. **内存使用**：
   - 处理大量数据时注意内存占用
   - HDF5 文件会自动分块存储以优化内存使用

5. **数据类型**：
   - RGB 图像：`uint8` 格式
   - 深度图像：`float32` 格式（单位：米，opencv的png编码器最高支持float32）
   - 机器人状态：`float64` 格式
