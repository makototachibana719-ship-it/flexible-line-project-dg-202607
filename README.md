# DG-202607 多工艺柔性产线项目

本仓库是 `flexible_line_project` 的 GitHub 源码快照，面向代码审查、协作开发和系统复现。

## 项目组成

- `configs/`：Cable Routing、Object Grasping、Peg Insertion 及训练/评测配置。
- `training/`：三工艺训练、恢复、检查点适配和运行时封装。
- `evaluation/`：单技能验证与评测入口。
- `scripts/`：数据审计、训练、诊断、验收及交付脚本。
- `external/serl/`：项目当前使用的 SERL 源码快照。
- `external/packages/`：三个工艺包中的源码、MuJoCo XML 与运行必需资产。
- `experiments/`：当前实验实现。
- `docs/`：架构、实验、交付、变更记录及历史状态文档。
- `deliverables/`：GPU 训练交接说明和验证入口。
- `SERL_PAPER_*`、`HIL_SERL_*`：论文一致性与专家干预实验的轻量源码和报告。

## 未纳入 Git 的资产

Checkpoint、Demo、Replay Buffer、训练数据集、W&B 运行目录、视频、压缩归档和大型临时复现目录没有提交到 Git。原因包括 GitHub 单文件大小限制、仓库可维护性和实验资产的敏感性。

缺失资产的位置和恢复原则见 [`docs/storage/EXCLUDED_ASSETS.md`](docs/storage/EXCLUDED_ASSETS.md)。原始权威资产当前保留在 WSL：

```text
/home/makoto/flexible_line_project
```

## 当前状态

请先阅读：

1. `AGENTS.md`
2. `PROJECT_STATUS.md`
3. `TASKS.md`
4. `docs/change_log.md`

仓库中的训练完成状态不等价于模型通过独立评测。任何 Success Rate、Checkpoint 或 Demo 结论都应以相应报告和原始资产为准。

## 安全说明

- 不要将密钥、令牌、`.env` 或本机绝对路径提交到仓库。
- 不要覆盖唯一可运行的 Checkpoint、Demo 或 MuJoCo 资产。
- 大型资产后续应使用独立 Release、对象存储或经确认的 Git LFS 方案管理。

