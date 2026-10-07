# Object Grasping Package Audit

Overall status: UNVERIFIED. Static presence/absence findings are validated; runtime behavior, demo contents, and checkpoint quality are not.

## Inventory

| Component | Status | Evidence / finding |
|---|---|---|
| README/manifest | PRESENT | Both supplied. |
| Python/environment | PRESENT | `PandaPickCubeGymEnv` and supporting controller/render code. |
| Registration | PRESENT | `PandaPickCube-v0`, `PandaPickCubeVision-v0`, max 100 steps. |
| MuJoCo model/assets | PRESENT | Panda/Robotiq XML plus OBJ/STL assets; runtime resolution UNVERIFIED. |
| Observation | PRESENT | Native nested Dict; vision path has state plus `images/front,wrist`. |
| Action | PRESENT | float32 Box `[-1,1]`, shape `(4,)`: xyz delta and grasp. |
| reset/step | PRESENT | Gymnasium-style reset/five-value step. Runtime UNVERIFIED. |
| Reward | PRESENT | `0.3*r_close + 0.7*r_lift`. |
| Success | PRESENT | Block height at least initial height + 0.2 m; returned in `info`. |
| termination/truncation | INCOMPATIBLE | `terminated` is set by the environment time limit, success does not terminate, and `truncated` is always false; outer Gym time-limit behavior needs runtime reconciliation. |
| Demo Recorder | PRESENT | `scripts/record_demo.sh` references recording workflow. |
| Demo Data | UNVERIFIED | `demos/franka_lift_cube_image_20_trajs.pkl` is PRESENT; supplied inspection report claims 2,000 transitions, but this audit did not deserialize untrusted pickle data. |
| Checkpoint | UNVERIFIED | Orbax-style `checkpoint_4500` PRESENT; restore and performance not tested. |
| DRQ/SERL actor/learner | PRESENT | Async DrQ program and actor/learner scripts. No captured `code/modified_serl` files were found. |
| Evaluation | MISSING | `scripts/eval.sh` explicitly extracts existing logs and is not independent restore/evaluation; README says formal success rate not evaluated. |
| W&B | PRESENT | Logger code/log artifacts supplied; run claims remain UNVERIFIED. |
| Requirements/environment | PRESENT | Conda environment, pip freeze, versions, git diff/status. |

## Supplied demo metadata (UNVERIFIED)

The package-provided `demo_info_generated.txt` reports a list of 2,000 transition dictionaries with keys `observations`, `actions`, `next_observations`, `rewards`, `masks`, `dones`. It reports observations `{front,state,wrist}` with image `(1,128,128,3)` uint8, state `(1,7)` float32, and action `(4,)` float32. These claims are useful schema evidence but are not VALIDATED against the pickle in this static round.

## Blocking findings

- Current project SERL wrapper expects singular `image`, not nested `images`; direct use is INCOMPATIBLE.
- Native success and termination semantics differ.
- Formal independent evaluation is MISSING.
- Demo and checkpoint are PRESENT but UNVERIFIED.
