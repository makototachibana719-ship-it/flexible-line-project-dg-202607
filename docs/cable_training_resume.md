# Cable Safe Training Resume

Date: 2026-08-22  
Milestone: `M1D-A2 — CABLE SAFE TRAINING RESUME`

## Resume state matrix

| State | Decision | Evidence / behavior |
|---|---|---|
| Agent params and target params | MUST_RESTORE | Restored through the validated full DrQ legacy Flax state. |
| Optimizer and temperature state | MUST_RESTORE | Part of the same 598-leaf agent state. |
| Learner/checkpoint step | MUST_RESTORE | Metadata and filename must agree; resumed logging/checkpoint numbering starts after N. |
| Actor total environment transitions | MUST_RESTORE | Learner-side replay insertion counter is checkpointed; actor CLI must match metadata. |
| Random exploration phase | UNSAFE_TO_RESET | Resume uses `max(0, random_steps - actor_steps)`; after 300 it is zero. |
| Demo replay | CAN_REBUILD | Five immutable validated files rebuild 10,521 raw / 10,526 replay records. |
| Online replay | MUST_RESTORE | Learner blocks below 300 and uses online/demo 50/50 thereafter; silent reset changes training semantics. |
| Online replay sampling RNG | MUST_RESTORE | Stored with the ring snapshot. |
| W&B run ID and step | MUST_RESTORE | Same project/name/ID with `resume="must"`; log steps are offset by checkpoint N+1. |

## Bundle format

Every committed project checkpoint has three same-stem files:

```text
checkpoint_N
checkpoint_N_online_replay.pkl
checkpoint_N_resume_metadata.json
```

Metadata is written last and is the bundle commit marker. It records checkpoint/learner step, actor steps, seed, W&B identity, environment/algorithm/schema, Demo manifest hash, config hash, source hash, replay size/capacity/hash, and timestamp. No credential is stored.

The online replay snapshot stores only the active chronological ring region, not unused capacity. It includes nested arrays, valid-index flags, episode-boundary state, insertion position reconstructed from active size, pixel keys, sampling RNG, and cumulative actor transitions. Full-capacity visual replay snapshots may be large and must be included in GPU storage planning.

## Resume behavior

- `--resume-from` accepts only a committed legacy Flax `checkpoint_N` file inside the configured output directory.
- Metadata and online replay sidecars are mandatory.
- Environment, algorithm, seed, observation, action, config, Demo manifest, W&B project/name/run ID, replay hash/capacity, and checkpoint step are fail-closed.
- The original checkpoint and sidecars are never overwritten.
- The delivered learner loop is reused. Its local step is translated to the global step `N + 1 + local_step`; the immediate local-step-zero checkpoint callback is skipped. Later saves must be greater than N and collision checks apply to all three bundle files.
- Demo replay is rebuilt before learner entry. Online replay is restored before the server starts.
- Actor does not load a separate model checkpoint; the resumed learner publishes restored params through Agentlace. Actor total progress must be passed explicitly and must equal metadata.
- W&B reuses the metadata run ID with `resume="must"`. Failure to resume that exact run stops launch.

## Bounded validation

Command:

```bash
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python -m evaluation.cable.resume_roundtrip
```

The parent launched separate create/resume Python processes. It created a real-architecture checkpoint at step 123, a small actual memory-efficient online replay with four actor transitions/five stored records, metadata, and synthetic W&B identity. The second process rebuilt/restored the agent and replay, reconstructed W&B `id`/`resume="must"` without online initialization, rebuilt all five Demo files, skipped the immediate collision-prone save, committed checkpoint 125, verified checkpoint 123 unchanged, and rejected a repeated checkpoint 125 save.

```text
PROCESS_RECONSTRUCTION = PASS
RESTORED_LEARNER_STEP = 123
ONLINE_REPLAY_RESUME = RESTORED
RESTORED_ACTOR_STEPS = 4
DEMO_RAW_TRANSITIONS = 10521
DEMO_REPLAY_SIZE = 10526
DEMO_REPLAY_RESUME = REBUILT
NEXT_CHECKPOINT_STEP = 125
COLLISION_PROTECTION = PASS
WANDB_RESUME_POLICY = must
TRAINING_RESUME = VALIDATED
```

This validates resume mechanics only. It is not training, GPU smoke, W&B online resume evidence, policy quality, or formal evaluation.
