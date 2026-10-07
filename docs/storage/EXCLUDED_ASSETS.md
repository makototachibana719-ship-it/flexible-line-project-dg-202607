# GitHub 外置资产清单

本仓库是源码与轻量文档快照，不是完整实验数据归档。

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

