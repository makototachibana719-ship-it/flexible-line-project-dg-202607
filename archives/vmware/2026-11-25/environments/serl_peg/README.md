# VMware serl_peg environment backup — 2026-11-25

The user confirmed that `/home/makoto/miniconda3/envs/serl_peg` is dedicated to this project and authorized its removal only after preservation. This directory contains the full environment archive in 23 ordered parts, the Conda environment export, explicit Conda package URLs, a per-member manifest and installation-origin references.

## Transfer verification

- Source machine: `zuodequandui-vm` (VMware Ubuntu).
- Original prefix: `/home/makoto/miniconda3/envs/serl_peg`.
- Compressed archive bytes: `1053456392`.
- SHA-256: `261b877fbea4af46b59dce66fa79f04adfc5f7a1411c4a39038c881835e0f96e`.
- The sender-reported SHA-256 exactly matches the receiver's computed SHA-256.
- Members: 68,819, including 58,784 regular files with 3,459,660,544 bytes. Symlinks and archive hardlinks are represented in the manifest; directories are included.
- Each part is at most 45 MiB. `PARTS_SHA256.csv` records part order, length and SHA-256. `ARCHIVE_INFO.json` records the complete archive hash.
- The parts were successfully reassembled locally and produced the sender's exact archive hash. `LIVE_TREE_FINGERPRINT.json` records a normalized content fingerprint for a guarded pre-deletion comparison. `EXTERNAL_SYMLINKS.json` is empty: all 1,382 archived symlinks normalize to locations inside the original environment prefix.
- A basic credential-pattern scan found no matches; this is not a guarantee that every possible sensitive value can be detected.

## Reassemble

Download or clone this entire directory. In an empty restoration workspace, run:

```bash
python3 reassemble.py /path/to/new/serl_peg.tar.gz
```

The helper checks every part and the complete archive and refuses to overwrite an existing output. It does not extract files or execute the backed-up environment.

## Recovery constraints

This is a raw Linux environment snapshot, not a conda-pack relocation archive. Prefer restoration on the same compatible Linux system to the exact original prefix. Interpreter shebangs and compiled packages can contain that prefix; arbitrary relocation is not verified.

Inspect the archive and manifest before extraction. Restore into a staging directory first. Preserve symlinks, hardlinks and Unix modes, verify file hashes, and ensure that the target prefix does not contain an existing working environment before placing the restored `serl_peg/` tree there. Then independently test the interpreter and required imports before using the project.

`EXTERNAL_REFERENCES.json` includes local installation-origin URLs under `~/serl` and `~/projects/mujoco_env`, and a historical pip build path. These URLs are provenance records, not proof that all those directories are runtime dependencies. Editable installations, system libraries, drivers and external source trees may still be required. Those paths were neither removed nor included as separate source backups by this environment-only operation.

`environment.yml` and `conda-explicit.txt` are supplemental recreation recipes. Network availability and compatibility of their packages are not guaranteed. The full archive retains installed package files to reduce dependence on redownloading them.

This backup does not cover `serl`, `serl_clean`, the base environment, shared package caches, other projects, operating-system configuration or GPU drivers. Uploading it does not constitute a tested end-to-end restoration.
