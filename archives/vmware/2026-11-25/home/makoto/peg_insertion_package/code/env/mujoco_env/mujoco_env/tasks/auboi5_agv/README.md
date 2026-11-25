# tasks/auboi5_agv

## 目录结构（含文件说明）

```text
mujoco_env/tasks/auboi5_agv/
├── aubo_i5_agv_config.py   # Aubo i5 + AGV 机器人配置类
└── photos/                 # 场景截图与问题记录图
```

## 包功能简述

该目录用于 Aubo i5 与移动底盘组合场景的任务级配置。

## 具体实现说明（逐文件）

`aubo_i5_agv_config.py`:
功能：在 `AuboI5Robot` 基础上扩展 AGV 相关底座和夹爪配置。
核心函数定义：`AuboI5AGVConfig`、`get_config()`。
参考使用：`examples/demo_aubo_i5_openpi/demo_aubo_i5_grasp_collect_data.py`。

`photos/*`:
功能：记录 AGV 场景效果与历史问题截图，便于配置调试回溯。
核心函数定义：无。
参考使用：文档说明与调参讨论。

## 常见问题与注意事项

- 该目录只提供配置，不直接创建环境实例。
- 底盘是否可控依赖控制器、场景接触参数和驱动关节配置的整体一致性。

