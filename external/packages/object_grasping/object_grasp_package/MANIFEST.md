# MANIFEST

1. `franka_sim/franka_sim/envs/panda_pick_gym_env.py`
   - 适配多相机 MujocoRenderer；图像 128×128；human render 调整。
   - success-rate 补丁加入/计划加入 `success`、`block_height`、`lift_height`。
2. `examples/async_drq_sim/async_drq_sim.py`
   - 加入本地 `[TRAIN]` / `[EVAL]` 日志与嵌套指标扁平化。
   - success-rate evaluation 加入/计划加入逐 episode success 与 max lift height。
3. `examples/async_drq_sim/run_learner.sh`
   - 允许通过 `"$@"` 覆盖实验参数。
4. `examples/async_drq_sim/run_actor.sh`
   - 允许通过 `"$@"` 覆盖 Actor/eval 参数。

提交前必须用 `git diff --name-only` / `git diff` 最终核对。打包脚本会自动收集 `serl_launcher/` 下实际修改的公共文件。
