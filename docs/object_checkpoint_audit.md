# Object Grasp Checkpoint 4500 Audit

Date: 2026-08-22  
Scope: frozen inventory, structural mapping, one bounded read-only restore attempt, no rollout.

## Frozen checkpoint

- Path: `external/packages/object_grasping/object_grasp_package/checkpoints/checkpoint_4500`
- Step: 4500, confirmed by path, learner log save/finalization records, and serialized state metadata.
- Files/size: 15 files / 105,193,080 bytes.
- Deterministic tree-manifest hash: `f23f676447cfcefe6568e5946de48b3ffd08ce5177ce2220d9caa76e0b7e831c`.
- Format: Flax `checkpoints.save_checkpoint(agent.state)` using the Orbax OCDBT backend.
- `_CHECKPOINT_METADATA` has non-null commit timestamp `1786095020566027166`; this is a committed directory, not a tmp checkpoint.

## Architecture mapping

The target was constructed with seed 0, the actual Demo observation contract (`front/wrist (1,128,128,3)`, state `(1,7)`), action `(4,)`, two image keys, and `resnet-pretrained` through the delivered `make_drq_agent` path.

Checkpoint metadata contains 601 serialized paths. The fresh target contains 598. All 598 target paths exist in the checkpoint, including actor, critic, temperature, dual `encoder_front`/`encoder_wrist`, and pretrained ResNet trees. The checkpoint additionally contains three post-update optimizer inner-state paths for actor, critic, and temperature that are absent from a fresh target.

Under the required exact mapping rule this is a mismatch, even though the parameter architecture is strongly consistent:

```text
CHECKPOINT_ARCHITECTURE_MATCH = MISMATCH
MISMATCH_SCOPE = THREE_OPTIMIZER_INNER_STATE_PATHS
```

## Restore and inference

One read-only Flax/Orbax restore was attempted against the constructed state. It did not return within the bounded 120-second window on CPU and was interrupted. No checkpoint or package file was written. Because no restored state was returned, no policy action was sampled.

```text
CHECKPOINT_RESTORE = UNVERIFIED_TIMEOUT
POLICY_INFERENCE = UNVERIFIED
POLICY_QUALITY = UNVERIFIED
```

This timeout is not reported as a successful mapping or a failed policy.

## Save/restore risk

The delivered environment records Flax 0.8.5 and Orbax 0.6.4; the project runtime also has the previously observed Orbax/TensorStore `assume_config` incompatibility on save. The delivered saver invokes Orbax-backed Flax saving directly, with no legacy fallback or resume contract. Combined with the exact optimizer-state mismatch and bounded restore timeout:

```text
OBJECT_CHECKPOINT_SAVE_RISK = HIGH
```

A later project-side launcher may reuse the Cable legacy-Flax strategy, but this round does not implement or convert anything.

Raw evidence: `results/object_m1d_b/checkpoint_structure_audit.json`. Individual checkpoint-file hashes are retained in the command record and the aggregate tree hash above.

## M1D-B2 correction

The earlier 601-versus-598 mismatch was caused by default flattening dropping empty nodes. The three metadata-only paths are Optax `EmptyState` sentinels with `value_type=None` and `skip_deserialize=true`. A fresh target contains the same nodes with `keep_empty_nodes=True`; exact structure is 598 arrays/601 paths before and after a zero-gradient optimizer update.

```text
CHECKPOINT_ARCHITECTURE_MATCH = MATCH
CHECKPOINT_RESTORE = UNVERIFIED_TIMEOUT
```

The new matching-target restore also exceeded 300 seconds. See `docs/object_checkpoint_closure.md`.
