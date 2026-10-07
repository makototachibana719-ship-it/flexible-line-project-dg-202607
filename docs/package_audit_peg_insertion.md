# Peg Insertion Package Audit

Overall status: UNVERIFIED. Static presence/absence findings are validated; runtime behavior, demo contents, and checkpoints are not.

## Inventory

| Component | Status | Evidence / finding |
|---|---|---|
| README/manifest | PRESENT | `README.txt`, `MANIFEST.txt`. |
| Python/environment | PRESENT | Full `mujoco_env` copy, Peg task, and `PegStateSERLEnv` adapter. |
| Registration | MISSING | Adapter directly calls a factory. `AuboPegInsert-State-v0` is a configuration label, not a delivered static Gym registration. |
| MuJoCo model/assets | PRESENT | Extensive scenes, robot/gripper models, meshes and textures; runtime resolution UNVERIFIED. |
| Observation | PRESENT | Adapter produces state-only float32 Box `(20,)`. No image observation is used in current training. |
| Action | PRESENT | float32 Box `[-1,1]`, `(7,)`; normalized xyz/RPY deltas plus binary gripper, converted to an 8D absolute action. |
| reset/step | PRESENT | Adapter exposes Gymnasium reset and five-value step. Runtime UNVERIFIED. |
| Reward | PRESENT | Weighted lateral/axial/orientation penalties, alignment bonus, and success bonus in Peg task; README/source coefficient mismatch is UNVERIFIED. |
| Success | PRESENT | lateral ≤ 0.002 m, axial ≤ 0.003 m, orientation ≤ 0.05 rad, as documented/implemented. |
| termination/truncation | PRESENT | Adapter preserves both and sets `is_success = terminated and not truncated`; runtime semantics UNVERIFIED. |
| Demo Recorder | PRESENT | SERL recorder and NPZ-to-PKL builder supplied. |
| Demo Data | UNVERIFIED | 20 NPZ files, state transition PKL, and BC PKL are PRESENT; no demo was created or rewritten. |
| Checkpoints | UNVERIFIED | BC checkpoints 500–5000 and RLPD candidate checkpoints 0–9000 are PRESENT; restore/performance not tested. |
| SERL training | PRESENT | Online SAC, RLPD and BC scripts; RLPD mixes 32 online + 32 demo transitions. |
| Actor/learner | PRESENT | Async SAC/RLPD actor/learner scripts. |
| Evaluation | UNVERIFIED | BC eval code and generic evaluation hooks exist; formal unified evaluation is explicitly MISSING. |
| W&B | PRESENT | Logger integration exists; historical results remain UNVERIFIED. |
| Requirements/environment | PRESENT | Environment YAML, requirements, runtime versions, configs. |
| Modified SERL | PRESENT | State-only BC implementation is supplied; compatibility with current mainline is UNVERIFIED. |

## Transition and demo schema evidence

- Runtime and builder schema: `observations`, `actions`, `next_observations`, `rewards`, `masks`, `dones`.
- Observation `(20,)` float32; action `(7,)` float32; reward/mask scalar float32; done boolean.
- Builder asserts 5,512 total transitions from 20 NPZ episodes. Supplied metadata says all 20 are successful, but actual semantic validation was not performed, so Demo Data remains UNVERIFIED.
- Raw NPZ metadata additionally documents `terminated`, `truncated`, error/force metrics, initial observation, and JSON metadata.

## Blocking findings

- No static Gym registration for the configured environment name.
- README states an axial penalty of `-20 * axial_error`, but current source contains that term twice (effective `-40 * axial_error`); intended reward is UNVERIFIED.
- Submitter-specific absolute paths are documented in `configs/absolute_paths_to_review.txt`.
- Formal unified checkpoint evaluation is MISSING.
- Package carries a broad `mujoco_env` copy and a modified SERL file, creating merge/version risk.
