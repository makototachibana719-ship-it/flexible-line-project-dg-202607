# Cable Transfer Checklist

Transfer and preserve repository-relative paths:

- [ ] `training/cable/`, `evaluation/cable/`, `evaluation/smoke/smoke_cable_routing_v2.py`, `scripts/verify_cable_handoff.py`
- [ ] `configs/cable/environment_gpu.yaml`, this directory including README/scripts/spec/snapshot/manifest
- [ ] `external/serl/serl_launcher/` and `external/packages/cable_routing_v2/.../franka_sim/` including all MuJoCo assets
- [ ] All five files in `demo_manifest.json`; exactly 10,521 transitions and every SHA-256 matches
- [ ] `AGENTS.md`, `docs/gpu_return_artifact_spec.md`, and required Python package metadata

Do not transfer `runtime/`, debug checkpoint reproductions, old/partial smoke directories, W&B credentials, caches, historical logs, legacy checkpoints, or unrelated Object/Peg data. The five canonical Demos are required; references alone are insufficient on the remote host.
