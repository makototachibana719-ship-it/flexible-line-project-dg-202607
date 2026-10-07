# Peg Algorithm and Training Entry Mapping (M1D-D)

Date: 2026-08-23

## Identity

`PEG_PRIMARY_ALGORITHM = RLPD_STYLE_SAC_WITH_50_50_DEMO_REPLAY`  
`PEG_AUXILIARY_ALGORITHMS = online SAC baseline; state-only BC auxiliary`  
No Peg DrQ/DPRL implementation was found. RLPD and SAC instantiate the same state-only `SACAgent`; RLPD differs by its fixed online/Demo batch mixture.

There is no implemented `BC pretrain -> RLPD fine-tune` chain. BC has a separate `BCAgent` and checkpoint tree; the RLPD entry always creates a fresh SAC agent and contains no BC restore/import path.

## Entries

| Purpose | Entry / launcher | Actual frozen configuration |
|---|---|---|
| RLPD actor+learner | `code/example/async_rlpd_peg_sim/async_sac_state_sim.py`; `run_actor.sh`, `run_learner.sh` | seed 0; 10,000 steps; actor random steps 0; learner starts 100; batch 64; ratio 1; UTD 1; 32 online + 32 Demo; checkpoint 1,000; W&B learner online, actor debug |
| SAC actor+learner | `code/example/async_sac_peg_sim/async_sac_state_sim.py`; launchers | smoke launcher: 2,000 steps; starts 300; batch 64; critic/actor ratio 8; UTD 1; checkpoint 500 |
| BC | `code/example/bc_peg_state/train_peg_bc.py` | 5,000 updates; batch 64; save 500; log 100; seed 0 |

SAC/RLPD actor, critic and temperature learning rates resolve through the current SERL `SACAgent.create_states` defaults to `3e-4`; BC explicitly passes `3e-4`. The primary source/config evidence is `configs/rlpd_config.txt` plus the RLPD launchers. The configured Gym id is a label; the delivered entries actually call `make_peg_state_env()` directly.

## Runtime contract

- factory: package `peg_env_adapter.make_peg_state_env`
- observation: state vector `(20,) float32` = TCP position 3, quaternion 4, wrench 6, relative goal position 3, goal quaternion 4
- action: normalized incremental `(7,) float32`, bounds `[-1,1]`; translation scale 0.002 m, rotation scale 1 degree, binary gripper
- training time limit: 500 control steps; `control_dt=0.02`, `physics_dt=0.002`
- termination: task success; truncation: elapsed steps reaches 500
- success: lateral `<=0.002`, axial `<=0.003`, orientation `<=0.05`; no force threshold
- executed reward: `-80*lateral -20*axial -2*orientation + 2[lateral<0.01] + 25[success]`

README describes the same axial coefficient. The separate legacy achieved/desired-goal method also contains a `-20*depth_error`, but is not added to the state reward and is bypassed by the current SimEnv preference for `compute_reward_from_obs`.

`REWARD_DISCREPANCY = NO`  
`PEG_TRAINING_PIPELINE = collect online transitions + load exact 5,512-transition Demo replay -> sample 32/32 -> SAC update_high_utd -> checkpoint agent.state`

## Runtime portability

The project adapter correctly stages generated XML under `runtime/peg_insertion/generated`, but the generated build still resolves robot/sensor/gripper XML/assets through the editable `/home/ubuntu/mujoco_env` installation. Therefore current host smoke is valid but the adapter is not independently portable.

Portable design for M1D-E: resolve the delivered package's `code/env` as the authoritative read-only asset root, stage generated XML in the project runtime/handoff root, and verify every XML include/mesh resolves without `/home/ubuntu` literals. Do not edit external `mujoco_env`.

