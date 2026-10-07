# AGENTS.md
# DG-202607 多工艺柔性产线机械臂自主决策与协同效能优化项目
# Codex 项目级工作规范

---

# 0. 文件作用

本文件是本项目中 Codex / AI Agent 的最高级项目工作规范。

在执行任何代码分析、代码修改、测试、实验、系统集成、文档生成或比赛材料整理之前，必须首先阅读本文件。

若其他临时任务说明与本文件冲突，以本文件为准。

本项目不允许以“快速完成”为理由破坏已经验证可以运行的代码、模型、Demo、Checkpoint 或实验结果。

---

# 1. 项目背景

比赛题目：

DG-202607
《多工艺柔性产线中机械臂自主决策与协同效能优化》

项目目标不是单独训练一个机械臂策略，而是建立一个完整的多工艺柔性制造系统，使多个机械臂能够：

1. 感知生产任务变化；
2. 自主选择和执行不同工艺；
3. 根据动态订单进行任务分配；
4. 在多机械臂共享生产环境中进行协同；
5. 处理资源冲突与潜在碰撞；
6. 根据设备状态进行动态调度；
7. 最大化柔性产线整体生产效能。

---

# 2. 比赛最终目标

最终必须形成一套可以实际运行和演示的软件仿真系统。

系统至少包含：

- 3 台机械臂；
- 3 种不同工艺；
- 动态订单；
- 自主任务决策；
- 多机械臂任务分配；
- 多机械臂协同；
- 冲突处理；
- 动态调度；
- KPI 统计；
- Baseline 对比；
- Proposed Method；
- 完整技术报告；
- 完整演示材料。

当前确定的三个工艺为：

1. Cable Routing
   线缆铺设

2. Object Grasping / Pick and Place
   物体抓取 / 取放

3. Peg / Shaft Insertion
   叉轴 / 插入装配

---

# 3. 当前技术路线

当前低层机器人技能学习主要基于：

- SERL
- DrQ
- SAC
- JAX
- MuJoCo
- mujoco_env

三个工艺目前由不同小组分别开发。

每个工艺包含自己的：

- MuJoCo environment
- task
- reward
- success condition
- observation
- action
- Demo
- Replay Buffer
- Actor
- Learner
- Checkpoint
- Evaluation

部分模型仍可能处于训练阶段。

---

# 4. 当前项目阶段

当前阶段：

PHASE 1
三工艺成果汇总与系统集成准备。

目前三个工艺尚未全部汇总到当前主工作区。

队友将按照统一格式提交：

incoming/
├── cable_routing_package/
├── object_grasp_package/
└── peg_insertion_package/

每个 package 应尽量包含：

README.md
MANIFEST.md

code/
├── example/
├── env/
└── modified_serl/

scripts/

checkpoints/

demos/

results/

configs/

logs/

在三个 package 未全部收到之前：

禁止假设不存在的代码结构。

禁止自行虚构另外两个工艺的接口。

禁止根据当前单个工艺直接设计不可修改的统一接口。

允许提前建立：

- 项目基础设施；
- 通用数据结构；
- evaluation framework；
- interface proposal；
- scheduler mock；
- documentation；
- tests；
- integration skeleton。

---

# 5. 项目目录规划

项目主目录：

~/flexible_line_project/

推荐结构：

flexible_line_project/
│
├── AGENTS.md
├── README.md
├── PROJECT_STATUS.md
├── TASKS.md
│
├── incoming/
│   ├── cable_routing_package/
│   ├── object_grasp_package/
│   └── peg_insertion_package/
│
├── external/
│   ├── serl
│   └── mujoco_env
│
├── skills/
│   ├── base/
│   ├── cable_routing/
│   ├── object_grasp/
│   └── peg_insertion/
│
├── orchestration/
│   ├── orders/
│   ├── scheduler/
│   ├── robot_manager/
│   ├── resources/
│   ├── coordination/
│   └── safety/
│
├── simulation/
│   ├── production_line/
│   ├── multi_robot/
│   ├── scenes/
│   └── configs/
│
├── evaluation/
│   ├── single_skill/
│   ├── system/
│   ├── baselines/
│   ├── benchmarks/
│   └── metrics/
│
├── experiments/
│
├── scripts/
│
├── tests/
│
├── results/
│
├── docs/
│   ├── architecture/
│   ├── experiments/
│   ├── competition/
│   └── technical_report/
│
└── deliverables/

不得为了匹配该目录结构而立即移动已经可以运行的原始代码。

首先使用适配层、引用或 wrapper。

---

# 6. 外部代码仓库

当前可能存在：

external/serl
external/mujoco_env

它们可能通过 symbolic link 指向：

~/serl
~/mujoco_env

这两个目录包含当前已经验证运行过的重要代码。

默认原则：

DO NOT REFACTOR EXTERNAL REPOSITORIES UNNECESSARILY.

优先：

1. 阅读；
2. 分析；
3. wrapper；
4. adapter；
5. 新建项目层代码；

最后才考虑修改 external 中的公共代码。

---

# 7. 最高级安全原则

任何已经跑通的代码都视为：

KNOWN WORKING VERSION

不得直接破坏。

修改已有代码之前必须：

1. 确认文件路径；
2. 确认文件用途；
3. 检查是否已有备份；
4. 如果没有备份，先创建备份；
5. 再进行修改；
6. 修改后运行最小测试；
7. 记录修改内容。

不得覆盖唯一可运行版本。

---

# 8. 强制备份规则

BEFORE MODIFYING ANY EXISTING SOURCE FILE:

必须执行：

1. 判断文件是否为已有可运行代码；
2. 如果是，则创建备份；
3. 备份必须能够追踪到原文件；
4. 不允许只依赖编辑器 Undo。

推荐命名：

filename_before_<change>.py

例如：

async_drq_randomized_before_eval.py

或：

filename_backup_YYYYMMDD_HHMMSS.py

如果已有 Git 仓库，也应优先创建明确 commit。

但 Git 不替代关键文件备份。

---

# 9. 严格禁止的操作

未经用户明确授权，不允许执行：

sudo

rm -rf

git reset --hard

git clean -fd

conda remove

conda env remove

pip uninstall

大规模 pip upgrade

大规模 conda update

删除 checkpoint

覆盖 checkpoint

删除 Demo

覆盖 Demo

删除 W&B 数据

删除实验结果

删除训练日志

删除 MuJoCo assets

删除 XML/MJCF

修改系统 CUDA

修改 NVIDIA Driver

修改系统级 Python

破坏当前 X11 / Viewer 环境

不得为了“解决依赖问题”直接重新安装整个环境。

---

# 10. 不得伪造实验结果

严禁：

- 编造 success rate；
- 编造 reward；
- 编造训练曲线；
- 编造 checkpoint；
- 编造 Demo 数量；
- 编造 KPI；
- 编造实验结果；
- 编造 benchmark；
- 编造真实机器人实验；
- 根据截图猜测精确数据。

所有实验结论必须来自实际程序输出。

无法确认时必须标记：

UNVERIFIED

或：

BLOCKED

不得写成：

PASS

---

# 11. Checkpoint 验收原则

训练完成 ≠ 模型有效。

Checkpoint 存在 ≠ 模型有效。

Reward 上升 ≠ Success Rate 高。

模型必须经过独立 Evaluation 才能标记：

VALIDATED

每个模型至少记录：

- process
- checkpoint path
- checkpoint step
- seed
- evaluation episodes
- success count
- success rate
- mean return
- std return
- mean episode length
- mean execution time
- failure categories

如果环境没有明确 success condition：

不得自行猜测。

必须：

1. 找到 reward；
2. 找到 terminated；
3. 找到 info；
4. 找到 task success logic；
5. 报告缺失；
6. 等待确认或实现明确 success interface。

---

# 12. Demo 数据原则

Demo 必须明确：

- observations
- actions
- next_observations
- rewards
- masks
- dones

如果实际格式不同，必须记录真实格式。

不得假设三个工艺 Demo 接口一致。

每份 Demo 应检查：

- transition count
- episode count
- observation structure
- state shape
- image shape
- action shape
- dtype
- reward range
- done
- mask
- success episode count

禁止把无动作轨迹、失败轨迹或异常轨迹自动标记为成功 Demo。

---

# 13. 三工艺汇总规则

收到三个 package 后：

DO NOT MERGE THEM IMMEDIATELY.

必须首先完成：

PROCESS PACKAGE AUDIT

分别检查：

1. Environment
2. Gym registration
3. MuJoCo scene
4. XML / MJCF
5. assets
6. robot config
7. observation space
8. action space
9. reward
10. success condition
11. reset
12. step
13. wrappers
14. demo format
15. ReplayBuffer
16. Actor
17. Learner
18. checkpoint
19. evaluation
20. W&B
21. modified SERL files
22. modified mujoco_env files

然后生成：

docs/process_comparison.md

必须明确：

- 相同点；
- 不同点；
- 接口冲突；
- 公共修改冲突；
- 可以统一的部分；
- 不应统一的部分。

完成审计之前禁止直接覆盖公共代码。

---

# 14. Skill Layer

三个工艺最终必须封装成统一技能层。

目标：

CableRoutingSkill

ObjectGraspSkill

PegInsertionSkill

统一抽象：

RobotSkill

建议提供：

reset()

load_checkpoint()

execute(task, robot_state)

evaluate()

get_status()

close()

---

# 15. SkillTask

上层 Scheduler 不应该知道 DrQ 内部细节。

统一任务输入建议包含：

SkillTask

字段可包括：

task_id

order_id

process_type

robot_id

target

priority

release_time

deadline

parameters

metadata

具体字段应在三个工艺完成接口审计后最终确定。

不得在尚未看到三个工艺真实接口前过度固定设计。

---

# 16. SkillResult

统一输出建议：

SkillResult

包含：

success

reward

execution_time

episode_steps

final_state

failure_reason

metrics

metadata

上层系统只通过 Skill API 调用技能。

禁止 Scheduler 直接调用：

agent.sample_actions()

或直接依赖 SERL Agent 内部实现。

---

# 17. Mock Skill

在三个模型尚未全部训练完成时，必须支持：

MockSkill

例如：

MockCableRoutingSkill

MockObjectGraspSkill

MockPegInsertionSkill

Mock 模式用于开发：

- Scheduler
- Order Manager
- Robot Manager
- KPI system
- Production Line simulator

这样上层系统开发不得被长时间训练阻塞。

---

# 18. 系统总体架构

最终目标架构：

Dynamic Orders
        ↓
Order Manager
        ↓
Task Graph
        ↓
Scheduler / Decision Layer
        ↓
Robot Manager
        ↓
Multi-Robot Coordination
        ↓
Skill Layer
        ↓
DrQ Policies
        ↓
MuJoCo Environments

Evaluation 模块应独立监控整个过程。

---

# 19. Dynamic Order Manager

必须支持动态订单。

Order 至少应考虑：

order_id

product_type

priority

release_time

due_time

process_plan

status

工艺链例如：

Order A:

GRASP
  ↓
PEG_INSERT
  ↓
CABLE_ROUTE

系统后续应支持：

- 正常订单；
- 动态新订单；
- 高优先级插单；
- 工序变化；
- 订单取消；
- 机器人故障后的重新调度。

---

# 20. Robot Manager

必须维护每台机器人状态。

至少包括：

robot_id

status

available

current_task

current_skill

capabilities

workspace

busy_time

idle_time

failure_state

机器人状态建议包括：

IDLE

BUSY

WAITING

BLOCKED

FAILED

RECOVERING

---

# 21. Scheduler Baselines

在开发 Proposed Method 之前必须实现 baseline。

至少：

Baseline 1:
Fixed Assignment

Baseline 2:
FCFS

Baseline 3:
Greedy / Earliest Finish Time

如果条件允许，可增加：

Shortest Processing Time

Priority-based Scheduling

Load Balancing

所有 Proposed Method 必须和 baseline 使用相同场景进行比较。

---

# 22. Proposed Method

项目最终必须形成明确的自主决策与协同优化方法。

推荐整体思路：

Hierarchical Decision Architecture

High-level:

task allocation
robot selection
skill selection
scheduling

Low-level:

SERL / DrQ robot control

即：

High-Level Decision Agent
          ↓
Robot + Task + Skill
          ↓
Low-Level DrQ Skill

不得仅仅把三个 DrQ 模型并排运行并称为：

multi-agent autonomous decision system

必须存在明确的上层自主决策逻辑。

---

# 23. 多机械臂协同

至少考虑：

- robot availability；
- task dependency；
- shared workspace；
- shared resource；
- collision/conflict；
- waiting；
- priority；
- task handoff；
- failure；
- replanning。

第一版可以使用：

Resource Lock / Workspace Reservation

例如：

ASSEMBLY_ZONE

TRANSFER_ZONE

CABLE_ZONE

机器人进入共享区域之前：

acquire(resource)

完成后：

release(resource)

后续可以升级为：

prediction-based reservation

priority arbitration

dynamic replanning

---

# 24. 三机械臂仿真策略

最终系统必须能够表现至少三台机械臂和三种工艺。

允许分阶段开发。

Stage A:

逻辑级 Production Line Simulator
+
Mock Skills

Stage B:

逻辑调度
+
独立 MuJoCo Skill Execution

Stage C:

统一或可视化的 Multi-Robot Simulation

不要为了立即建立一个复杂的三机器人 MuJoCo Scene 而阻塞 Scheduler 和 Evaluation 的开发。

---

# 25. Evaluation Framework

必须建立统一 Evaluation。

Single Skill Metrics：

success_rate

mean_return

std_return

mean_episode_length

mean_execution_time

System Metrics：

makespan

throughput

order_completion_time

task_waiting_time

robot_utilization

robot_idle_rate

changeover_time

reschedule_latency

conflict_count

failure_count

success_rate

所有指标必须有明确数学定义。

---

# 26. Benchmark 原则

所有调度算法必须使用相同：

- order sequence
- random seed
- robot configuration
- skill performance configuration
- failure scenario
- simulation horizon

进行公平比较。

输出至少包括：

raw_results.csv

summary.csv

summary.json

必要图表。

---

# 27. 必须测试的场景

至少设计：

SCENARIO 1
Normal Production

SCENARIO 2
Dynamic Order Arrival

SCENARIO 3
High Priority Urgent Order

SCENARIO 4
Robot Failure

SCENARIO 5
Shared Workspace Conflict

SCENARIO 6
Process / Product Change

每种场景必须可以重复运行。

---

# 28. Ablation Study

Proposed Method 完成后，应考虑消融实验。

例如：

Without Dynamic Scheduling

Without Resource Coordination

Without Priority Mechanism

Without Failure Replanning

Without Learned Scheduler

用于分析每个模块的实际贡献。

---

# 29. 实验结果目录

所有正式实验结果统一存储：

results/

不得散落在源码目录。

推荐：

results/
├── single_skill/
├── baseline/
├── proposed/
├── ablation/
├── stress_test/
└── figures/

实验文件名应包含：

method
scenario
seed
timestamp

---

# 30. PROJECT_STATUS.md

Codex 必须持续维护：

PROJECT_STATUS.md

至少包括：

Current Phase

Completed

In Progress

Blocked

Validated Components

Unverified Components

Known Issues

Latest Experiments

Next Milestone

该文件必须反映真实状态。

---

# 31. TASKS.md

所有任务统一记录在：

TASKS.md

优先级：

P0 = blocker

P1 = critical

P2 = important

P3 = optional

每个任务状态：

TODO

IN_PROGRESS

BLOCKED

DONE

VALIDATED

只有通过测试才能从 DONE 改为 VALIDATED。

---

# 32. Change Log

任何重要修改必须记录：

docs/change_log.md

格式：

Date

File

Original behavior

Modification

Reason

Test command

Test result

Rollback method

---

# 33. Codex 每次开始任务的流程

每次执行任务之前：

STEP 1
阅读 AGENTS.md

STEP 2
阅读 PROJECT_STATUS.md

STEP 3
阅读 TASKS.md

STEP 4
检查相关代码

STEP 5
确认是否需要修改已有文件

STEP 6
如需修改，先备份

STEP 7
提出最小实施方案

STEP 8
实施

STEP 9
运行最小测试

STEP 10
运行相关 smoke test

STEP 11
记录结果

STEP 12
更新 PROJECT_STATUS.md

STEP 13
更新 TASKS.md

STEP 14
更新 change_log

STEP 15
向用户报告

---

# 34. Codex 每次任务结束必须报告

必须明确输出：

## Files Changed

列出所有修改文件。

## Reason

解释修改原因。

## Commands Executed

列出关键命令。

## Tests

说明执行了什么测试。

## Result

PASS / FAIL / PARTIAL / BLOCKED

## Remaining Issues

仍存在什么问题。

## Next Recommended Task

下一步建议。

---

# 35. 长时间任务限制

不得自行启动：

- 长时间强化学习训练；
- 数小时 benchmark；
- 大规模超参数搜索；
- 无终止条件的任务。

除非用户明确授权。

可以：

- smoke test；
- 1~5 episode quick evaluation；
- import test；
- syntax test；
- unit test；
- 短时间仿真。

需要长时间运行时，先给出：

预计命令

预计资源

输出位置

终止方式

再等待用户确认。

---

# 36. Debug 原则

发现问题时：

禁止一次修改多个可能原因。

必须：

1. 复现；
2. 定位；
3. 最小修改；
4. 测试；
5. 再继续。

不得使用大量无意义 print 污染正式训练输出。

临时 debug print 完成后必须清理或通过 debug flag 控制。

---

# 37. 环境兼容原则

当前项目可能存在：

gym

gymnasium

不同版本 API。

不得未经检查直接转换。

必须确认：

reset() 返回格式

step() 返回格式

observation_space

action_space

terminated

truncated

info

wrapper compatibility

---

# 38. MuJoCo 原则

不得仅因为 Viewer / OpenCV / X11 问题就修改强化学习逻辑。

必须区分：

Simulation

Rendering

Teleoperation

Training

Evaluation

Demo Recording

这些模块的问题。

Viewer 异常不能自动推断为训练环境异常。

---

# 39. 训练与评测分离

Training 和 Evaluation 必须分离。

Training metrics：

train reward
loss
Q value
actor statistics

不能替代：

Evaluation success rate

正式模型比较必须使用 Evaluation。

---

# 40. 文档原则

所有关键模块必须有：

README

运行命令

输入

输出

依赖

测试方式

不得让最终系统只能由原作者运行。

---

# 41. 最终比赛技术报告

最终需要生成：

docs/technical_report/

报告至少包括：

1. 摘要
2. 研究背景
3. 工业问题分析
4. 系统总体架构
5. MuJoCo 仿真环境
6. 三种机器人技能
7. SERL / DrQ 方法
8. Demo 与数据
9. Skill Layer
10. 动态订单建模
11. 多机械臂任务分配
12. 协同与冲突处理
13. Proposed Method
14. Baselines
15. Experimental Setup
16. KPI Definitions
17. Results
18. Ablation
19. Dynamic Scenario Tests
20. Industrial Applicability
21. Innovation
22. Limitations
23. Conclusion
24. Reproduction Guide

所有实验数据必须来自 results/。

---

# 42. 最终交付目录

最终必须形成：

deliverables/
│
├── technical_report/
│
├── source_code/
│
├── runnable_simulation/
│
├── checkpoints/
│
├── demo_data/
│
├── experiment_results/
│
├── figures/
│
├── videos/
│
├── README.md
│
└── reproduction_guide.md

---

# 43. 一键运行目标

最终系统应尽量支持：

./run_demo.sh

或：

python scripts/run_demo.py

启动：

Dynamic Orders
+
3 Robots
+
3 Skills
+
Scheduler
+
Coordination
+
Metrics

同时应提供：

python scripts/run_benchmark.py

用于运行正式 benchmark。

---

# 44. Submission Validator

项目后期应实现：

scripts/validate_submission.py

检查：

- 3 robots
- 3 processes
- checkpoints
- demo data
- dynamic orders
- scheduler
- coordination
- baseline
- proposed method
- benchmark
- KPI results
- technical report
- README
- reproduction guide
- demo material

验证失败不得输出：

READY FOR SUBMISSION

---

# 45. 当前开发路线

当前路线严格按照：

MILESTONE 0
Repository / Package Audit

↓

MILESTONE 1
Single Skill Validation

↓

MILESTONE 2
Unified Skill API

↓

MILESTONE 3
Production Line Core

↓

MILESTONE 4
Dynamic Order Manager

↓

MILESTONE 5
Baseline Scheduler

↓

MILESTONE 6
Multi-Robot Coordination

↓

MILESTONE 7
Proposed Autonomous Decision Method

↓

MILESTONE 8
Benchmark

↓

MILESTONE 9
Ablation + Stress Tests

↓

MILESTONE 10
Visualization / Demo

↓

MILESTONE 11
Technical Report

↓

MILESTONE 12
Submission Package

不得无理由跳过前置验收。

---

# 46. 当前立即执行的任务

当前三个工艺 package 尚在收集中。

因此现在：

DO NOT WAIT IDLY.

在 package 到齐之前，可以进行：

1. 项目目录整理；
2. PROJECT_STATUS.md；
3. TASKS.md；
4. change_log；
5. architecture skeleton；
6. Skill API draft；
7. Mock Skill；
8. Order data model；
9. Robot state model；
10. KPI definitions；
11. Evaluation framework skeleton；
12. Baseline scheduler interface；
13. tests；
14. competition deliverable checklist。

但：

不得假设三个真实 Skill 的 observation/action 完全一致。

不得在 package 到齐前强制实现具体 Adapter。

---

# 47. Package 到齐后的第一任务

当：

incoming/cable_routing_package

incoming/object_grasp_package

incoming/peg_insertion_package

全部存在后：

立即暂停其他非关键开发。

执行：

THREE PROCESS COMPATIBILITY AUDIT

生成：

docs/process_comparison.md

docs/integration_plan.md

docs/interface_matrix.md

然后才允许进入：

Unified Skill API Integration。

---

# 48. 最终评价目标

项目不能只证明：

“机器人可以完成三个动作。”

最终必须证明：

“系统面对多工艺、动态订单、多机器人资源约束和异常情况时，能够自主进行任务决策和协同，并通过量化 KPI 相比传统调度方法提升柔性产线整体效能。”

这是整个项目所有代码、实验和报告的最终评价标准。