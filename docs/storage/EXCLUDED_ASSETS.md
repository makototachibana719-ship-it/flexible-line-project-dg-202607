# GitHub 外置资产清单

本仓库是源码与轻量文档快照，不是完整实验数据归档。

环境补充：`archives/vmware/2026-11-25/environments/serl_peg/` 仅保存用户确认本项目独占的 VMware `serl_peg` 环境。其他 Conda 环境、共享包缓存、系统及其他项目仍不属于备份范围；恢复约束见该目录 README。

后续补充：用户又明确确认 `serl_clean` 仅本项目使用，其完整环境另存于 `archives/vmware/2026-11-25/environments/serl_clean/`。这一例外不扩大到 `serl`、base 或共享缓存。完整归档包含 pip 已安装文件，但不代表已测试跨机器或任意路径恢复。

2026-11-25 补充：`archives/vmware/2026-11-25/` 保存本次明确核验范围内的 VMware Peg 文件，含部分历史 checkpoint、Demo 和日志，并附逐文件 SHA-256 清单。以下排除规则仍适用于归档之外的资产；不能据此认为其他工艺、当前 WSL 实验数据或共享运行环境已完整上传。

## 权威存储位置

```text
/home/makoto/flexible_line_project
```

## 未提交类别

| 类别 | 主要位置 | 原因 |
|---|---|---|
| Checkpoint | `results/checkpoints/`、`runtime/*_handoff/`、各工艺包 `checkpoints/` | 单文件大、禁止覆盖、需要独立验收 |
| Demo | `results/cable_demo/`、各工艺包 `demos/` | 数据量大且包含正式采集资产 |
| Dataset / Replay | `results/datasets/`、训练输出目录 | 不适合普通 Git 历史 |
| 训练与消融结果 | `results/training/`、`results/ablation/` | 保留于实验存储，仓库只保留代码与轻量报告 |
| W&B | 所有 `wandb/` | 本地运行缓存，不作为源码 |
| 交付压缩包 | `results/final_packages/*.tar.gz` 及本机交付包 | 由校验和管理，不重复进入 Git |
| 视频 | `*.mp4` 等 | 使用发布附件或对象存储 |
| 临时复现沙盒 | `results/reproduction/` | 包含完整项目副本，体积过大 |

## 恢复约束

1. 根据 `PROJECT_STATUS.md`、`TASKS.md` 和相应报告确定需要的精确资产版本。
2. 校验原始 SHA-256 或包内 `SHA256SUMS`。
3. 不得用同名文件覆盖唯一可运行版本。
4. Checkpoint 存在不代表策略通过独立评测。

`EXCLUDED_LARGE_FILES.csv` 记录源工程中大于 10 MiB 的文件路径和大小，供迁移到 Release、对象存储或 Git LFS 前审查。

