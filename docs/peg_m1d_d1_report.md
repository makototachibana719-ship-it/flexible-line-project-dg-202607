# M1D-D1 Peg Replay + Fresh Infrastructure Closure

Date: 2026-08-23  
`PEG_M1D_D_STATUS = PASS_FOR_FRESH_GPU_TRAINING`

## Frozen baseline

Environment `(20,) float32` / action `(7,) float32`; 20/20 successful NPZ episodes; exact 5,512-transition Demo PKL; primary `RLPD_STYLE_SAC_PLUS_50_50_DEMO_REPLAY`; 32 online + 32 Demo; UTD 1; starts 100; max 10,000; checkpoint 1,000; fixed initial state. These validated facts were not re-audited or changed.

Formal configuration is frozen in `configs/peg_insertion/peg_rlpd_baseline_seed0.yaml`, including all three `3e-4` learning rates, replay capacity 1,000,000, explicit W&B identity, fresh initialization, and no legacy load.

## Reward and success contracts

Reward is exactly `-80*lateral -20*axial -2*orientation +2[lateral<0.01] +25[success]`; `REWARD_DISCREPANCY = NO`. Success is lateral `<=0.002 m`, axial `<=0.003 m`, orientation `<=0.05 rad`; force is excluded and success terminates. GPU operators must not alter these contracts.

## Gate matrix

| Gate | Result |
|---|---|
| `PEG_ENVIRONMENT_GATE` | PASS |
| `PEG_DEMO_GATE` | PASS |
| `PEG_REPLAY_GATE` | PASS |
| `PEG_ALGORITHM_MAPPING_GATE` | PASS |
| `PEG_NEW_CHECKPOINT_GATE` | PASS |
| `PEG_PORTABILITY_GATE` | PASS |
| legacy checkpoint restore | `UNRESOLVED_TIMEOUT_NONBLOCKING` |

## Scope and next milestone

No training, learner update, GPU smoke, formal evaluation, dependency installation, external-source mutation, Demo/checkpoint conversion, or reward/success change occurred.

M1D-D is closed for the fresh-training strategy. Exact next milestone is **M1D-E — Peg GPU training handoff preparation**, but it was not entered in this task.

