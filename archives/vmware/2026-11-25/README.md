# VMware Peg asset snapshot — 2026-11-25

This directory preserves the scoped Peg files from VMware VM `run01-Ubuntu`, under their original `home/makoto/` hierarchy. This is a historical asset backup, not a new successful-training claim or a replacement for the current WSL project.

## Provenance and integrity

- Guest transfer archive: `peg_vmware_backup_20261125_140221.tar.gz`.
- Archive bytes: `765612019`.
- Archive SHA-256: `15a3ed3ac8d4418b9ee46a9235b8ef71e0dedfca0ebf0122fe4bab31530368d5`.
- Expected regular files: `1606`.
- Each imported file is checked against the previously received VMware path/size/SHA-256 inventory.
- `MANIFEST_SHA256.csv` records the original home-relative path, bytes, SHA-256 and original Unix mode.
- `.gitattributes` disables text conversion throughout `home/` to preserve exact source and checkpoint bytes.

## Included scope

- `home/makoto/peg_insertion_package/`: the scoped package source, assets, checkpoint and Demo files.
- `home/makoto/serl_checkpoints/peg_insert*/`: remaining Peg training checkpoint versions.
- `home/makoto/serl_results/peg_insert*/`: scoped Peg evaluation and training logs.
- `home/makoto/serl_data/peg_*/` and the explicitly reviewed Peg Demo archive.

The already removed duplicate ZIP, duplicate BC/RLPD checkpoint directories and temporary evaluation checkpoint are not recreated. Their retained counterparts remain in the package or remaining checkpoint tree.

Other projects, `panda_*` artifacts, shared Conda environments, global caches, `serl_wandb`, unreviewed backups and root-level logs of uncertain ownership are excluded. Symbolic links and empty directories were not included in the guest archive. This snapshot alone does not prove an end-to-end reproducible environment.

## Restore and verify

Clone the repository and use this directory's `home/makoto/` tree as a restoration source. Before restoring, verify every file using `MANIFEST_SHA256.csv`. Restore first into a new directory; compare existing working files rather than overwriting them. Reapply Unix modes from the manifest where scripts require executable permissions.

For any checkpoint, restore the entire checkpoint directory including its metadata, manifests and data chunks. Do not restore isolated chunks or treat the mere presence of a checkpoint as evidence of a validated policy.

No VMware source file was deleted as part of this upload.
