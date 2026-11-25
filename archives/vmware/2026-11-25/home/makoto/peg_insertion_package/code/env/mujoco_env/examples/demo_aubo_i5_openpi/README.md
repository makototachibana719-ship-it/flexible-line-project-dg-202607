# Aubo i5 OpenPI 工作流教程

本教程演示 `mujoco-env` 中 Aubo i5 的数据闭环流程：采集数据、远程推理、转换为 LeRobot 格式。

## 目录内容

该目录包含脚本：
- `demo_aubo_i5_grasp_collect_data.py`: 采集 MuJoCo 数据（LeRobot 兼容格式）
- `demo_aubo_i5_pick_cube_openpi_inference.py`: 在 MuJoCo 中进行 OpenPI 远程推理
- `convert_mujoco_to_lerobot.py`: 将采集的 episode 转成本地 LeRobot 数据集缓存

## 前置条件

- 已安装项目基础依赖：
  ```bash
  cd path/to/mujoco-env
  pip install -r requirements.txt
  pip install openpi_client
  ```
- 无图形环境（服务器）建议设置：
  ```bash
  export MUJOCO_GL=egl
  ```

## 详细步骤

### 1. 采集训练数据 (`demo_aubo_i5_grasp_collect_data.py`)

RRT* 路径规划 + 数据采集（OpenPI/LeRobot 用）。
每个 episode 记录 `states.npy`、`actions.npy`、`images.npz`、`metadata.json`。

```bash
cd mujoco-env
python examples/demo_aubo_i5_openpi/demo_aubo_i5_grasp_collect_data.py \
  --object green_cube \
  --record \
  --record-every 1 \
  --img-size 224 \
  --image-format npz \
  --max-episode 100 \
  --no-render
```

**常用参数**:
- `--object`: `green_cube`, `red_cube`, `milk`, `soup`, `bbq`, `cookie`
- `--record`: 开启数据采集
- `--record-every`: 采样间隔
- `--img-size`: 图像边长（默认 224）
- `--image-format`: `npz`(推荐) / `npy`
- `--no-render`: 无渲染模式

### 2. 转换为 LeRobot 本地格式 (`convert_mujoco_to_lerobot.py`)

将采集的 `episode_XXXX` 转换为 LeRobot 数据集格式 (`~/.cache/huggingface/lerobot/<repo_id>/`)。

```bash
python convert_mujoco_to_lerobot.py \
  --data-dir path/to/your/data \
  --repo-id <repo-id>/mujoco_aubo_data \
  --robot-type aubo_i5 \
  --fps 50 \
  --image-size 224 \
  --chunk-size 1000 \
  --overwrite
```

### 3. OpenPI 远程推理 (`demo_aubo_i5_pick_cube_openpi_inference.py`)

通过 websocket 调用 OpenPI 推理并在 MuJoCo 中执行。
需先在 OpenPI 环境启动 policy server。

```bash
# 1. 启动 server (在 openpi 环境)
# uv run scripts/serve_policy.py ...

# 2. 运行推理 (在 mujoco-env 环境)
python examples/demo_aubo_i5_openpi/demo_aubo_i5_pick_cube_openpi_inference.py \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "pick up the green cube and put it into basket" \
  --infer-interval 10
```

**参数说明**:
- `--host/--port`: OpenPI websocket 地址
- `--prompt`: 任务指令
- `--infer-interval`: 推理频率
- `--record`: 可选，保存推理时的 episode 数据

## 常见问题

- **推理延迟**: 检查网络连通性和服务端负载。
- **找不到 episode**: 确认 `--data-dir` 路径正确。
- **OpenPI Client**: 需要 `openpi_client` 和 `opencv-python`。

## 参考
- 具体部署流程、网络配置和参数规范可参考公司内部 OpenPI 相关文档。
