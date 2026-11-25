# StereoCamera 双目深度相机模块

## 概述

`StereoCamera` 是一个双目深度相机封装类，将两个相同的相机组合成双目视觉系统，使用 OpenCV 的立体匹配算法计算深度图。

### 核心功能

- **双目图像获取**：获取左右相机的 RGB 图像
- **立体匹配**：使用 BM 或 SGBM 算法计算视差图
- **深度计算**：从视差图转换为深度图
- **可视化转换**：将深度图转换为彩色热力图

### 支持的立体匹配算法

| 算法 | 特点 |
|------|------|
| `BM` | StereoBM，块匹配算法，速度快 |
| `SGBM` | StereoSGBM，半全局块匹配算法，精度高 |

---

## StereoCamera 类

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mj_model` | mujoco.MjModel | - | MuJoCo 模型对象 |
| `camera1_name` | str | - | 左相机名称 |
| `camera2_name` | str | - | 右相机名称 |
| `pic_width` | int | 640 | 图像宽度 |
| `pic_height` | int | 480 | 图像高度 |
| `stereo_algorithm` | str | 'BM' | 立体匹配算法 ('BM' 或 'SGBM') |

### 示例

```python
from mujoco_env.utils.camera import StereoCamera

stereo_cam = StereoCamera(
    mj_model=model,
    camera1_name='camera1_left',
    camera2_name='camera2_right',
    pic_width=640,
    pic_height=480,
    stereo_algorithm='BM'
)
```

---

## 相机参数属性

初始化后自动提取以下参数：

| 属性 | 类型 | 说明 |
|------|------|------|
| `camera_intrinsic_matrix` | np.ndarray | 相机内参矩阵 (3x3) |
| `fx`, `fy` | float | 水平和垂直焦距 |
| `cx`, `cy` | float | 主点坐标（图像中心） |
| `baseline` | float | 双目基线距离（米） |
| `fovx`, `fovy` | float | 水平和垂直视场角（度） |

### 内参矩阵格式

```
fx  0  cx
 0  fy  cy
 0   0   1
```

---

## 方法

### get_camera_image()

获取单个相机图像。

```python
image = stereo_cam.get_camera_image(mj_model, mj_data, camera_name)
```

**参数**：
- `mj_model`: MuJoCo 模型
- `mj_data`: MuJoCo 数据
- `camera_name`: 相机名称

**返回值**：BGR 格式的彩色图像 (numpy array, uint8)

---

### get_stereo_images()

获取双相机彩色图像。

```python
left_image, right_image = stereo_cam.get_stereo_images(mj_model, mj_data)
```

**参数**：
- `mj_model`: MuJoCo 模型
- `mj_data`: MuJoCo 数据

**返回值**：
- `left_image`: 左相机图像 (BGR)
- `right_image`: 右相机图像 (BGR)

---

### get_stereo_gray_images()

获取双相机灰度图像。

```python
left_gray, right_gray = stereo_cam.get_stereo_gray_images(mj_model, mj_data)
```

**参数**：
- `mj_model`: MuJoCo 模型
- `mj_data`: MuJoCo 数据

**返回值**：
- `left_gray`: 左相机灰度图像
- `right_gray`: 右相机灰度图像

---

### compute_disparity()

计算视差图。

```python
disparity = stereo_cam.compute_disparity(left_gray, right_gray)
```

**参数**：
- `left_gray`: 左相机灰度图像
- `right_gray`: 右相机灰度图像

**返回值**：视差图 (numpy array, float32)

**注意**：视差 = (fx × baseline) / depth，视差越大表示距离越近

---

### compute_depth_from_disparity()

从视差计算深度图。

```python
depth = stereo_cam.compute_depth_from_disparity(disparity)
```

**参数**：
- `disparity`: 视差图

**返回值**：深度图 (numpy array, float64, 单位：米)

**深度范围限制**：0.01m - 10m

---

### compute_depth_map()

计算完整的双相机深度图。

```python
depth_map, disparity, left_image, right_image = stereo_cam.compute_depth_map(mj_model, mj_data)
```

**参数**：
- `mj_model`: MuJoCo 模型
- `mj_data`: MuJoCo 数据

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `depth_map` | np.ndarray | 深度图 (float64, 米) |
| `disparity` | np.ndarray | 视差图 (float32) |
| `left_image` | np.ndarray | 左相机图像 (BGR) |
| `right_image` | np.ndarray | 右相机图像 (BGR) |

---

### get_depth_at_point()

获取 2D 点处的深度值。

```python
depth_value = stereo_cam.get_depth_at_point(depth_map, point_2d)
```

**参数**：
- `depth_map`: 深度图
- `point_2d`: 2D 坐标 (x, y)

**返回值**：该点的深度值（米），无效返回 None

---

### depth_to_colormap()

将深度图转换为彩色热力图用于可视化。

```python
color_map = stereo_cam.depth_to_colormap(depth_map, max_depth=2.0)
```

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `depth_map` | np.ndarray | - | 深度图 (float64, 单位：米) |
| `max_depth` | float | 2.0 | 可视化的最大深度值（米） |

**返回值**：彩色热力图 (numpy array, uint8, BGR 格式)

**颜色映射**：

```
距离近 ────────────────────────→ 距离远

蓝色 → 青色 → 绿色 → 黄色 → 红色
 0     64    128    192    255
```

**示例**：

```python
# 计算深度图
depth_map, _, left_image, _ = stereo_cam.compute_depth_map(model, data)

# 转换为彩色热力图
color_map = stereo_cam.depth_to_colormap(depth_map, max_depth=1.5)

# 保存可视化结果
cv2.imwrite("depth_visualization.png", color_map)

# 显示
cv2.imshow("Depth Colormap", color_map)
```

---

## 完整使用示例

```python
import cv2
from mujoco_env.utils.camera import StereoCamera

# 创建双目相机实例
stereo_cam = StereoCamera(
    mj_model=model,
    camera1_name='camera1_left',
    camera2_name='camera2_right',
    stereo_algorithm='BM'
)

# 计算深度图
depth_map, disparity, left_image, right_image = stereo_cam.compute_depth_map(model, data)

# 获取某点的深度值
point_depth = stereo_cam.get_depth_at_point(depth_map, (320, 240))
print(f"点(320,240)深度: {point_depth:.3f} 米")

# 转换为可视化颜色图
color_map = stereo_cam.depth_to_colormap(depth_map, max_depth=2.0)

# 保存结果
cv2.imwrite("left_image.jpg", left_image)
cv2.imwrite("depth_map.png", depth_map)  # 保留真实深度值
cv2.imwrite("depth_visualization.png", color_map)  # 可视化颜色图
```

---

## 数据存储说明

### 深度图像存储

```python
# 使用recorder.py保存深度图
from mujoco_env.utils.recorder import ImageSaver

image_saver = ImageSaver(output_dir="./outputs", save_depth=True)
filenames = image_saver.save_image(depth_image=depth_map)
# 深度图保存为 .png 格式（float32）

# 读取深度值
depth_png = cv2.imread(filenames['depth'], cv2.IMREAD_UNCHANGED)
print(f"类型: {depth_png.dtype}")   # float32
print(f"深度值: {depth_png[100, 100]:.4f} 米")  # 直接使用真实值
```

### 格式对比

| 格式 | 大小 (224×224) | 精度 | OpenCV支持 |
|------|----------------|------|------------|
| `.png` (float32) | ~80 KB | ~0.1μm | ✅ 原生支持 |
| `.png` (uint16) | ~40 KB | 1mm | ✅ 原生支持 |
| `.npy` (float64) | ~392 KB | 100% | ❌ 需NumPy |

**注意**：OpenCV PNG 编码器原生最高支持 float32 格式，无需转换。

---

## 注意事项

1. **立体匹配算法选择**：
   - `BM`：速度快，适合实时应用
   - `SGBM`：精度高，适合需要精确深度的场景

2. **深度范围**：
   - 有效深度范围：0.01m - 10m
   - 超出范围的像素会被裁剪

3. **基线距离**：
   - 双目基线越大，深度测量精度越高
   - 基线太小会导致深度计算不稳定

4. **无头环境**：
   - `cv2.imshow()` 和 `cv2.waitKey()` 在无头环境下会报错
   - 使用 `cv2.imwrite()` 保存图像代替显示
