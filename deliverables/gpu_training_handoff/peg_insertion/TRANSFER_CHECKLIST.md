# Peg Insertion Transfer Checklist

- [ ] `training/peg_insertion/`, `evaluation/peg_insertion/`, `scripts/verify_peg_handoff.py`
- [ ] `configs/peg_insertion/environment_gpu.yaml`, baseline config, and this entire handoff directory
- [ ] `external/serl/serl_launcher/`, Peg package code/assets, and an importable editable `mujoco_env`
- [ ] Formal PKL `peg_insert_20_demos_state.pkl`: 5,512 transitions and manifest SHA-256 matches
- [ ] All 20 provenance NPZ files and hashes in `demo_manifest.json`
- [ ] Optional BC PKL `peg_insert_bc_consistent.pkl` only as auxiliary provenance/BC data; never as RLPD formal input
- [ ] `AGENTS.md` and `docs/gpu_return_artifact_spec.md`

Do not transfer runtime-generated XML, debug/roundtrip outputs, legacy RLPD/BC checkpoints for training, old logs, caches, W&B credentials, or Cable/Object data.
