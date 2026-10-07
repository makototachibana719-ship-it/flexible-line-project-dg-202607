# Object Grasping Package

- 工艺：Object Grasping（Franka 方块抓取 / 抬升）
- Gym env：`PandaPickCube-v0`
- Conda：`serl`
- Python：3.10
- JAX/jaxlib：0.4.35 / 0.4.35
- Flax：0.8.5
- NumPy/SciPy：1.26.4 / 1.11.4
- 主训练文件：`code/example/async_drq_sim.py`

## Observation
- `front`: `(1,128,128,3)` uint8
- `wrist`: `(1,128,128,3)` uint8
- `state`: `(1,7)` float32

## Action
`Box(-1,1,(4,),float32)`，四维 `[x,y,z,grasp]`。

## Reward / success
Reward：`0.3*r_close + 0.7*r_lift`。`z_success = z_init + 0.2m`。
Success 判据：`block_pos[2] >= z_success`。
Episode 原始终止仍由 `time_limit_exceeded()` 控制。

## Demo
官方 `franka_lift_cube_image_20_trajs.pkl`，20 条示范轨迹；训练日志显示 demo buffer size 2001 transitions。

## 5000-step run2
见 `configs/train_seed0_5000_run2.txt` 与 `results/training_summary.md`。

## Formal success rate
**未评测。** 不用 reward 代替 success rate。

## Checkpoint
`checkpoint_4500` 已确认保存，但尚未通过正式 success-rate eval 选出 best checkpoint。

## 一键打包
```bash
bash scripts/build_object_grasp_package.sh
```
默认从 `~/serl` 收集实际文件并生成 `~/object_grasp_package.zip`。
