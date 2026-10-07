# Peg Portable Runtime (M1D-D1)

Date: 2026-08-23

## Dependency classification

- Generated output: previously redirected to `runtime/peg_insertion/generated`, already project-owned.
- Asset dependency: installed `mujoco_env` assets; current machine resolves these to `/home/ubuntu/mujoco_env/mujoco_env/assets`.
- Python package dependency: imports `mujoco_env.tasks.base_task` and the delivered double namespace used by `peg_env_adapter`.
- Editable-install dependency: the current `serl_mujoco` environment points `mujoco_env` at `/home/ubuntu/mujoco_env`. That location is host evidence, not a portable configuration value.

The original generated XML carried relative paths whose meaning depended on its original source-tree output location. Moving the XML alone to project runtime therefore left implicit dependency on the editable source layout.

## Project adapter

`training/peg_insertion/peg_runtime_adapter.py` discovers the installed asset directory from `base_task.__file__`, creates `runtime/peg_insertion/assets` as a project-side symlink to that discovered directory, stages XML under `runtime/peg_insertion/generated`, and rewrites asset/include paths to `../assets/...`. Explicit mesh paths make `compiler.meshdir` redundant, so it is removed to prevent double prefixing.

There is no username literal or `/home/ubuntu` path in the adapter/config/generated XML. Each host regenerates its own runtime symlink from its imported package location. No external source or asset was copied or modified.

## Tests

- Current host create/reset/one-zero-step/close: PASS; observation `(20,)`, finite reward, no termination/truncation.
- Generated XML contains no `/home/ubuntu`: PASS.
- Required assets resolve through the runtime symlink: PASS, demonstrated by MuJoCo model construction.
- Synthetic different-user project-root calculation under `/tmp/different-user-project-*/flexible_line_project`: PASS without moving the repository.

`PEG_PORTABLE_RUNTIME = VALIDATED`

