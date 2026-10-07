# Cleanup audit — 2026-10-07

This record was created before removing redundant local and WSL artifacts.

## Protected assets

The cleanup explicitly preserves:

- the authoritative WSL project at `/home/makoto/flexible_line_project`;
- all formal assets under `results/checkpoints`, `results/cable_demo`, `results/datasets`, `results/training`, `results/ablation`, `results/validation`, and `results/closure`;
- WSL final return package `peg_training_return_20260909_005` and its verified archive;
- the Windows `PEG_Training_Return_Complete_20260910_006.tar.gz` archive and SHA-256 sidecar;
- the historical Windows `flexible_line_project.zip` archive, because it predates Git and may contain removed history;
- this private Git repository and its SHA-256 source manifest.

## Preserved before deletion

- `preserved/local_snapshot_unique/` contains every non-cache file found only in the older Windows project snapshot and not at the corresponding WSL path: 62 files, approximately 7.29 MiB.
- `preserved/runtime_roundtrip_evidence/` contains the lightweight JSON reports and metadata from disposable checkpoint/resume roundtrip directories: 23 files, approximately 93 KiB.

## Authorized removal classes

1. Six WSL archive-extraction and portability-test working copies.
2. WSL final return versions 001, 003, and 004 plus the verified 004 archive; version 005 remains as rollback.
3. WSL `runtime/*_handoff/*roundtrip*` checkpoint test outputs after retaining their reports and metadata.
4. Python bytecode/cache directories in the project and test copies.
5. The superseded Windows project working snapshot after its unique files were retained.
6. Four Windows code mirrors proven to be byte-identical subsets of the WSL copies.
7. The expanded Windows 006 delivery directory after the verified 006 archive was retained.
8. Empty or transient Windows staging directories and the root temporary directory.

No formal Demo, formal Checkpoint, accepted dataset, current source tree, or verified latest delivery archive is included in the removal set.

