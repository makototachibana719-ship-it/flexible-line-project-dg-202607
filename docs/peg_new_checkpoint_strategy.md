# Peg Fresh-Training Checkpoint Strategy (M1D-D1)

Date: 2026-08-23

## Policy

`FRESH_INITIALIZATION = TRUE`  
`LOAD_LEGACY_CHECKPOINT = FALSE`

Fresh Peg RLPD and SAC use the same `SACAgent` state architecture. `JaxRLTrainState` contains step, RNG, actor/critic/temperature parameters, target parameters (target critic), separate optimizers and optimizer states, plus non-serialized apply/optimizer definitions reconstructed by the fresh target.

The project checkpoint primitive temporarily selects Flax legacy checkpointing, performs save/restore, and restores the global Flax option in `finally`. This follows the already validated Cable/Object compatibility strategy and avoids the local Orbax/TensorStore stall without changing installed packages.

## Roundtrip

Location: `runtime/peg_handoff/checkpoint_roundtrip/checkpoint_0`

- fresh state save: PASS, 26,189,442-byte legacy Flax file
- destroy and rebuild same `(20,)/(7,)` SAC target: PASS
- restore: PASS
- tree leaves: 198 before and after
- every restored leaf value: exact equality
- restored step: 0
- parameter roots: actor, critic, temperature; target critic and all optimizer/RNG state are present in the full train-state tree
- one restored deterministic policy call on a real reset observation: `(7,) float32`, finite, all values in `[-1,1]`

`NEW_PEG_CHECKPOINT_SAVE = PASS`  
`NEW_PEG_CHECKPOINT_RESTORE = PASS`  
`NEW_PEG_RESTORED_POLICY_INFERENCE = PASS`

This validates serialization compatibility only, not learning or policy quality.

## Legacy policy

Delivered `rlpd_current_candidate/checkpoint_9000` remains preserved. Its `_METADATA` SHA256 is `5b48f93dd977244392405c3f4063b8ed32ac1769549c6c28e66c138f7404edb8`; `manifest.ocdbt` SHA256 is `7002e603b948d86c36025b807b70d4efa05cdccda9bda570761b20220ae28da9`.

`PEG_LEGACY_CHECKPOINT_RESTORE = UNRESOLVED_TIMEOUT_NONBLOCKING`. It is not a fresh-training prerequisite, handoff blocker, or resume source. No further long restore diagnosis is planned.

