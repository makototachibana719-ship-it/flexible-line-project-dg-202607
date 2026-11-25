# VMware serl_clean environment backup — 2026-11-25

The user confirmed that `/home/makoto/miniconda3/envs/serl_clean` is dedicated to this project. Preserve this full environment before any guarded deletion. The base environment, `serl`, shared package caches and other projects are outside this operation.

## Integrity and contents

- Source machine: `zuodequandui-vm` (VMware Ubuntu).
- Original prefix: `/home/makoto/miniconda3/envs/serl_clean`.
- Compressed archive bytes: `1051056689`.
- Sender and receiver SHA-256: `8839f6cbe62444046c6ab62fb62c586913e9862555ccb492b684cdb7eae37301`.
- Archive members: 68,498, including 58,490 regular files and 1,382 symlinks. All symlinks normalize inside the original environment prefix. No external symlink targets were found.
- All 23 ordered parts were successfully reassembled locally; the resulting archive matched the sender's full SHA-256.
- `ARCHIVE_INFO.json` records member/part counts and archive properties.
- `PARTS_SHA256.csv` records each ordered part's bytes and SHA-256.
- `ENVIRONMENT_MANIFEST_SHA256.csv` records archive member type, bytes, regular-file SHA-256, Unix mode and link targets.
- `LIVE_TREE_FINGERPRINT.json` supports a complete pre-deletion comparison against the live environment.
- `EXTERNAL_REFERENCES.json` and `EXTERNAL_SYMLINKS.json` distinguish recorded local installation sources and symbolic links outside the environment prefix.

Conda reported 92 pip-installed packages and warned that its export alone cannot reliably lock them for reproducibility. The full archive therefore preserves the installed environment files in addition to `environment.yml` and `conda-explicit.txt`; it is not only a package-name list.

Basic credential-pattern checks are performed before publication. They are not a guarantee that every possible sensitive value is detectable.

## Reassemble and restore

Clone or download this complete directory and run:

```bash
python3 reassemble.py /path/to/new/serl_clean.tar.gz
```

The helper verifies every part and the complete archive, refuses to overwrite an existing output, and does not extract or execute the environment. A local reassembly test must reproduce the sender's archive hash before remote publication.

This is a raw Linux environment snapshot, not a conda-pack relocation archive. Prefer the same compatible Linux system and the exact original prefix: interpreter shebangs and compiled libraries may contain that prefix. Arbitrary relocation and executable restoration have not been tested.

Inspect the manifest and links first, restore into a staging directory, preserve Unix permissions/links, and verify hashes. Do not overwrite an existing working environment. Before restoring the original prefix, ensure that it is absent. Then independently test the interpreter and required imports.

Local installation-origin URLs are provenance, not proof that each source directory is still needed at runtime. Editable source trees, system libraries and drivers may remain necessary; no other source tree, installed environment, operating-system component or shared cache was removed or separately backed up by this environment-only action.

Conda/pip recipes require package availability and compatibility; the full archive retains installed package files to reduce redownload dependence. Successful transfer and reassembly do not prove end-to-end scientific reproduction.
