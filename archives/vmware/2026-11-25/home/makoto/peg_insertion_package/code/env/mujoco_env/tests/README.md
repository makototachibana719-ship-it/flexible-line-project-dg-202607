# tests

本目录包含 `mujoco-env` 的自动化测试用例，使用 `pytest` 执行。

## 快速运行

```bash
cd path/to/mujoco-env
pytest tests/ -v
```

## 运行单个测试文件

```bash
cd path/to/mujoco-env
pytest tests/test_robot_env.py -v
```

## 真机集成测试

仅在真实机器人可用时执行：

```bash
cd path/to/mujoco-env
export REAL_FRANKA_IP=192.168.1.1
pytest tests/test_real_robot_integration.py -v
```

## 注意事项

- 建议先在仿真环境通过基础测试，再执行真机测试。
- 真机测试前请确认网络连通、SDK 可用和安全防护已就位。
