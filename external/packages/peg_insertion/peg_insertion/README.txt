# Peg Insertion 单工艺复现包

## 1. 工艺信息

工艺名称：Peg Insertion  
机械臂：Aubo i5  
仿真平台：MuJoCo  
强化学习框架：SERL  
当前主要算法：RLPD（20 Demo + Online Replay Buffer）  
辅助基线：
- Online SAC
- Behavior Cloning（BC，当前仍在测试）

当前阶段：
- Peg MuJoCo 环境：已跑通
- SERL Adapter：已跑通
- Online SAC：已跑通
- 20 条正式 Demo：已完成
- RLPD Demo Buffer：已跑通
- RLPD 多轮训练：已完成
- RLPD 正确率优化：进行中
- BC consistent 数据训练与测试：进行中
- 正式统一 success-rate evaluation：未完成

## 2. Python 环境

Conda 环境：serl_peg

完整环境见：

configs/environment.yml
configs/requirements.txt
configs/runtime_versions.txt

请优先使用提供的 environment.yml 或 requirements.txt，
不要自行整体升级 NumPy/JAX/Flax/MuJoCo/Gym 等关键依赖。

## 3. 环境接口

训练环境名称：

AuboPegInsert-State-v0

注意：
该名称用于当前训练配置；
实际 MuJoCo 环境通过：

code/example/async_rlpd_peg_sim/peg_env_adapter.py

中的 make_peg_state_env() 创建并适配。

## 4. Observation Space

当前 SERL 状态版 observation：

shape = (20,)
dtype = float32

当前为 state-only，不使用图像观测。

具体 20 维拼接顺序必须以：

code/example/async_rlpd_peg_sim/peg_env_adapter.py

中的当前实现为准。

# 0-2:   tcp_pos (obs["tcp_pos"][:3])
# 3-6:   tcp_quat (obs["tcp_quat"][:4])
# 7-12:  wrench (obs.get("wrench", zeros(6)))
# 13-15: relative_goal_pos (obs["desired_goal"][:3] - tcp_pos)
# 16-19: goal_quat (obs["desired_goal"][3:7])
## 5. Action Space

shape = (7,)
dtype = float32
range = [-1, 1]

各维意义：

0: dx
1: dy
2: dz
3: droll
4: dpitch
5: dyaw
6: gripper

前 6 维为归一化末端增量动作，
由 peg_env_adapter.py 按 translation_scale / rotation_scale
转换成 MuJoCo 环境使用的目标位姿。

Demo 与在线 Actor 使用相同 7D action space。

## 6. Reward

主要定义位置：

code/env/mujoco_env/mujoco_env/tasks/peg_insertion/peg_insertion.py

当前 reward 为 Peg 插入任务的手工状态奖励。

基础奖励（负值惩罚）：
   -80.0 * lateral_error      # 横向误差惩罚（权重最高，引导对准）
   -20.0 * axial_error        # 轴向误差惩罚（引导插入）
   -2.0  * orientation_error  # 姿态误差惩罚（引导姿态对齐）

 阶段奖励（正向激励）：
   +2.0   当 lateral_error < 0.01 (1cm) 时
   +25.0  当 success 条件满足时（成功奖励）

## 7. Success Condition

主要定义位置：

code/env/mujoco_env/mujoco_env/tasks/peg_insertion/peg_insertion.py

训练过程中 success 与 terminated 逻辑由当前 PegInsertion task 判断。

正式 Demo 要求最后一步：

terminated = True
truncated = False

并满足当前 lateral / axial / orientation 等任务成功条件。

同时满足以下三个条件：
   1. lateral_error <= 0.002   (2mm)   # 横向对准
   2. axial_error   <= 0.003   (3mm)   # 轴向对准（1.5倍容差）
   3. orientation_error <= 0.05 (约2.9度) # 姿态对准
 
 注意：axial_error 的容差是 position_tolerance 的 1.5 倍，
       允许插入方向有稍大的余量。

## 8. RLPD 主训练代码

主文件：

code/example/async_rlpd_peg_sim/async_sac_state_sim.py

当前正式配置：

env = AuboPegInsert-State-v0
exp_name = serl_peg_insert_rlpd20_v3_10k_utd1
seed = 0
max_steps = 10000
random_steps = 0
training_starts = 100
batch_size = 64
critic_actor_ratio = 1
utd_ratio = 1
checkpoint_period = 1000

Demo：

demos/peg_insert_20_demos_state.pkl

RLPD batch：

32 Online + 32 Demo = 64

## 9. Checkpoint

当前 RLPD 候选模型：

checkpoints/rlpd_current_candidate/

来源：

peg_insert_rlpd_v3_10k_utd1

当前状态：
尚未经过统一正式 success-rate evaluation，
因此本包将其标记为 current candidate，而不是 final/best。

当前 BC checkpoint：

checkpoints/bc_current/

来源：

peg_insert_bc_consistent_v1

BC 仍处于测试阶段。

## 10. Demo

正式人工 Demo：

demos/peg_demos_v1/

数量：20 条
成功条数：20 条

SERL/RLPD 转换数据：

demos/peg_insert_20_demos_state.pkl

总 transition 数：

5512

BC consistent 数据：

demos/peg_insert_bc_consistent.pkl

详细格式见：

demos/demo_info.md

## 11. 当前正式评测结果

RLPD：
未完成统一正式 success-rate evaluation。

BC：
未完成统一正式 evaluation。

注意：
训练 reward、W&B reward 曲线和单次成功现象不得作为正式 success rate。

## 12. 新终端运行流程

首先安装/激活 SERL 主仓库和本包提供的依赖环境。

Conda：

conda activate serl_peg

MuJoCo 环境需要安装：

cd code/env/mujoco_env
python -m pip install -e . --no-deps

SERL 中需要应用：

code/modified_serl/

下的公共代码修改。

本包 code/example/ 下包含：
- Online SAC
- RLPD
- BC

实际训练参数见 configs/。

## 13. 启动

推荐首先使用：

scripts/run_tmux.sh

分别启动时：

scripts/run_learner.sh
scripts/run_actor.sh

Demo 录制：

scripts/record_demo.sh

正式统一 evaluation 目前仍在整理，
详见 scripts/eval.sh 与 results/training_summary.md。