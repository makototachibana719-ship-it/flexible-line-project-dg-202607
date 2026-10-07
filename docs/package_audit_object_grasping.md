# Object Grasping Package Audit

Overall status: UNVERIFIED. Static presence/absence findings are validated; runtime behavior, demo contents, and checkpoint quality are not.

Algorithm mapping: DPRL is confirmed by a project member. The delivered package does not contain a `DPRL` symbol or dedicated DPRL class/entry point and instead explicitly constructs `DrQAgent`; therefore the exact DPRL-to-code mapping is NEEDS_VERIFICATION.

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
| Demo Recorder | PRESENT | `scripts/record_demo.sh` records that the official demo is used, but also says the local recording workflow is not confirmed. |
| Demo Data | UNVERIFIED | `demos/franka_lift_cube_image_20_trajs.pkl` is PRESENT; supplied inspection report claims 2,000 transitions, but this audit did not deserialize untrusted pickle data. |
| Checkpoint | UNVERIFIED | Orbax-style `checkpoint_4500` PRESENT; restore and performance not tested. |
| DPRL training entry | NEEDS_VERIFICATION | Human-confirmed algorithm is DPRL; delivered entry is `code/example/async_drq_sim.py`, launched by `run_actor.sh`/`run_learner.sh`, and named/configured as `async_drq_sim`. |
| Delivered policy structure | PRESENT | Code imports `DrQAgent`, calls `make_drq_agent`, and uses a visual SAC-style actor/critic with pretrained ResNet-10 encoders, proprioception, two-layer 256-unit policy/critic MLPs, and configured critic ensemble/subsampling. Whether this is the team's DPRL variant NEEDS_VERIFICATION. |
| Replay/demo dependency | PRESENT | Online and demo buffers are memory-efficient visual replay buffers. Demo loading is optional in code, but the delivered run configuration supplies the demo PKL; with demo present, learner samples 50% online and 50% demo and concatenates batches. |
| Evaluation | MISSING | `scripts/eval.sh` explicitly extracts existing logs and is not independent restore/evaluation; README says formal success rate not evaluated. |
| W&B | PRESENT | Logger code/log artifacts supplied; run claims remain UNVERIFIED. |
| Requirements/environment | PRESENT | Conda environment, pip freeze, versions, git diff/status. |

## Supplied demo metadata (UNVERIFIED)

The package-provided `demo_info_generated.txt` reports a list of 2,000 transition dictionaries with keys `observations`, `actions`, `next_observations`, `rewards`, `masks`, `dones`. It reports observations `{front,state,wrist}` with image `(1,128,128,3)` uint8, state `(1,7)` float32, and action `(4,)` float32. These claims are useful schema evidence but are not VALIDATED against the pickle in this static round.

## DPRL evidence reconciliation

- Human-confirmed final algorithm label: DPRL.
- Delivered training entry: `code/example/async_drq_sim.py`.
- Delivered class/factory: `DrQAgent` / `make_drq_agent`; no `DPRL` identifier was found by static search.
- Delivered update path: `update_critics` and `update_high_utd`; with demo configured, every learner batch is a 50/50 concatenation of online and demonstration samples.
- Delivered checkpoint: Orbax/Flax `agent.state` saved by `checkpoints.save_checkpoint`; package contains `checkpoint_4500`, originating from configured path `drq_seed0_5000_run2`.
- Delivered run dependency: configuration specifies `franka_lift_cube_image_20_trajs.pkl`; code can run without demo, but the submitted run evidence used a demo path.

Conclusion: record the project-level algorithm as DPRL, but mark implementation identity NEEDS_VERIFICATION until the team supplies the DPRL definition or confirms that this DrQAgent plus demo-replay implementation is its DPRL realization.

## Blocking findings

- Current project SERL wrapper expects singular `image`, not nested `images`; direct use is INCOMPATIBLE.
- Native success and termination semantics differ.
- Formal independent evaluation is MISSING.
- Demo and checkpoint are PRESENT but UNVERIFIED.
- DPRL label versus delivered DrQ-named implementation NEEDS_VERIFICATION.
