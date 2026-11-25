# assets

`assets` 是 `mujoco-env` 的资产总入口。你通常只需要在这里完成三件事：预览资产、在任务里选资产、排查资产装配问题。

## 快速开始（最常用）

### 1) 直接预览场景

```bash
cd path/to/mujoco-env
python -m mujoco.viewer mujoco_env/assets/scenes/mhr_sim.xml
```

### 2) 运行完整示例（组装 + 仿真）

```bash
cd path/to/mujoco-env
python examples/demo_aubo_mhr_scene_sim.py
```

### 3) 单独验证某类资产

```bash
cd path/to/mujoco-env

# 机械臂（Manipulator）
python -m mujoco.viewer --mjcf mujoco_env/assets/models/manipulators/franka_fr3/scene.xml

# 移动底盘（Mobile Base）
python -m mujoco.viewer --mjcf mujoco_env/assets/models/mobile_bases/freight/freight_aubo.xml

# 夹爪（Gripper）
python -m mujoco.viewer --mjcf mujoco_env/assets/models/grippers/robotiq_2f85/scene.xml

# 传感器（Sensor）
python -m mujoco.viewer --mjcf mujoco_env/assets/models/sensors/realsense_d435i/d435i.xml

# 安装件（Mount）
python -m mujoco.viewer --mjcf mujoco_env/assets/models/mounts/cylinder/cylinder.xml
```

## 新增资产（最小实现）

如果你只做资产侧改动，建议按下面最小流程：

1. 把文件放到对应目录（如 `models/grippers/<name>/`、`models/manipulators/<name>/`、`scenes/`）。
2. 保持目录名与主 XML 命名清晰一致（便于按名称引用）。
3. 用 `mujoco.viewer` 单体预览确认模型可加载。
4. （可选）如果你会用 `mujoco_env.assets` 的查询接口，再在 `mujoco_env/assets/__init__.py` 的注册表补充名称映射。

示例（新增夹爪资产）：

```text
mujoco_env/assets/models/grippers/my_gripper/
├── my_gripper.xml
├── meshes/
└── textures/
```

## 常用资产名速查

- `scene_name`：`default`、`cubes`、`assemble_hole`、`fruitpick`、`mhr_sim`
- `gripper_name`：`robotiq_gripper`、`panda_hand`、`assemble_axle`、`umi_scissors`、`robotiq_realsense`
- `mount_name`：`pedestal`、`default`、`cylinder`、`top_point`
- `use_ft_sensor`：`False` 或具体传感器名（如 `KWR75B`）

## 资产目录

```text
mujoco_env/assets/
├── scenes/                        # 场景入口 XML
├── models/
│   ├── manipulators/              # 机械臂本体
│   ├── mobile_bases/              # 移动底盘
│   ├── grippers/                  # 夹爪与末端工具
│   ├── sensors/                   # 相机/力传感器
│   └── mounts/                    # 安装件
├── objects/                       # 任务物体
└── textures/                      # 材质纹理
```

## 夹爪控制实现

### 接入方式（推荐）

- **Task 中处理**：夹爪动作与任务语义强绑定时（抓取/放置）。
- **Controller 中处理**：希望统一控制接口、便于跨任务复用时。
- **Environment 中拆分动作**：常见做法是将最后一维作为夹爪动作。

```python
robot_action = action[:-1]
gripper_action = action[-1]  # 约定范围 [-1, 1]
```

### 动作约定与映射

- 建议统一夹爪动作范围为 `[-1, 1]`（`-1` 关闭，`1` 打开）。
- 位置映射（开闭线性插值）：

```python
gripper_pos = gripper_close_pos + (gripper_action + 1) / 2 * (gripper_open_pos - gripper_close_pos)
data.ctrl[gripper_actuator_id] = gripper_pos
```

- 若 `robot_config.gripper_name` 有值，动作维度通常为“机器人控制维度 + 1（夹爪）”。

### 观测、奖励与判定建议

- 观测可加入夹爪状态：位置、速度、力（可选）、目标值（可选）。
- 奖励中可加入“接近 + 闭合抓取 + 提升高度”等分量，常见门控条件是“夹爪已闭合”。
- 成功判定可包含“夹爪闭合且目标物体已抬离”。

### 接触/力控制（可选增强）

- 接触检测：通过 `contact` 遍历与夹爪 `geom` 集合匹配判断。
- 力控制：对支持力控的夹爪（如 Robotiq）可在控制器中加入力误差闭环。

### 常用夹爪资产与数据约定

- 常用夹爪：`panda_hand`、`robotiq_2f85`、`robotiq_gripper`。
- OpenPI 相关采集数据中通常包含 `gripper` 字段（状态与动作）。

### 最小模板（夹爪动作执行）

```python
def apply_gripper_action(data, actuator_id, gripper_action, open_pos=0.08, close_pos=0.0):
    # gripper_action in [-1, 1]
    pos = close_pos + (gripper_action + 1.0) * 0.5 * (open_pos - close_pos)
    data.ctrl[actuator_id] = pos
```

## 常见问题

- 修改资产后建议清理旧的拼装 XML，避免加载缓存结果。
- mesh 路径优先使用仓库内相对路径，避免跨机器失效。
- 切换资产后动作维度或控制异常，先检查控制器类型与关节/执行器映射。
- 上游同步模型请保留 `LICENSE` 文件。

## 参考资源

- [MuJoCo 文档](https://mujoco.readthedocs.io/)
- [Franka Emika](https://www.franka.de/)
- [Aubo Robotics](https://www.aubo-robotics.cn/)
- [Universal Robots](https://www.universal-robots.com/)

## 致谢

感谢以下开源与厂商资源对模型构建与仿真验证的支持：
- Franka Emika 官方模型
- MuJoCo Menagerie
- 各机器人厂商公开资产与技术文档
