# Object Grasping Transfer Checklist

- [ ] `training/object_grasping/`, `evaluation/object_grasping/`, `evaluation/smoke/smoke_object_grasping.py`, `scripts/verify_object_handoff.py`
- [ ] Shared project resume primitives used by Object (`training/cable/resume.py`); this is code reuse, not a Cable runtime/training prerequisite
- [ ] `configs/object_grasping/environment_gpu.yaml`, this entire handoff directory, snapshot and manifest
- [ ] `external/serl/serl_launcher/` and the Object package/environment/assets needed by `PandaPickCubeVision-v0`
- [ ] Canonical `franka_lift_cube_image_20_trajs.pkl`: exactly 2,000 transitions and manifest SHA-256 matches
- [ ] `AGENTS.md` and `docs/gpu_return_artifact_spec.md`

Do not transfer `runtime/object_*` diagnostics, `checkpoint_4500` for training, old logs/smokes, caches, W&B credentials, or Cable/Peg data. The legacy checkpoint is never loaded. Demo success provenance remains UNVERIFIED but its unchanged use is project-approved.
