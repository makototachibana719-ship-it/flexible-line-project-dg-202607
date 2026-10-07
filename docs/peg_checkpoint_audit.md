# Peg Checkpoint Audit (M1D-D)

Date: 2026-08-23

## Inventory and mapping

Both groups are committed Flax/Orbax OCDBT directories containing `_CHECKPOINT_METADATA`, `_METADATA`, `_sharding`, `manifest.ocdbt`, `d/` and OCDBT data files.

| Group | Steps | Count | Metadata paths | Size per checkpoint | Mapping |
|---|---|---:|---:|---:|---|
| `checkpoints/bc_current` | 500..5000 by 500 | 10 | 52 | 1,150,079–1,171,002 bytes | BC `BCAgent`; matches BC save cadence and actor-only tree |
| `checkpoints/rlpd_current_candidate` | 0..9000 by 1000 | 10 | 201 | 12,129,276–12,259,360 bytes | SAC agent state produced by RLPD entry; matches config/cadence and actor+critic+temperature tree |

No delivered SAC-only checkpoint group or other checkpoint group exists. RLPD step 9000 is the highest delivered primary candidate; absence of 10000 is an inventory fact, not evidence of corruption.

`PEG_CHECKPOINT_MAPPING = BC:VALIDATED; RLPD:VALIDATED; SAC:NONE_DELIVERED`

## Bounded restore

One corrected, actual restore attempt was made in `serl_mujoco`, CPU-only, using `/home/ubuntu/serl/serl_launcher`, a fresh `make_sac_agent(seed=0, obs=(20,), action=(7,))` target, and `checkpoint_9000`. It did not return before the 120-second process timeout. A preceding invocation failed before restore because the package's partial `modified_serl` shadows the full SERL namespace; it did not read the checkpoint.

- candidate: `rlpd_current_candidate/checkpoint_9000`
- structural metadata: 201 paths
- restore: `TIMEOUT_120S`
- deterministic inference: `NOT_RUN_RESTORE_GATED`
- policy quality: `UNVERIFIED`

`PEG_LEGACY_CHECKPOINT_RESTORE = UNRESOLVED_TIMEOUT_NONBLOCKING`. The project decision is fresh initialization; this delivered checkpoint is preserved but is no longer a training prerequisite, handoff blocker, or resume source. The checkpoint remains read-only and was neither converted nor rewritten.
