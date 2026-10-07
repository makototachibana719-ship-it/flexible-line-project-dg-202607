# Cable Routing Automatic Demo Preflight

Date: 2026-08-21

## Result

- Checkpoint inventory: PRESENT
- Checkpoint restore: UNVERIFIED
- DrQ architecture match: UNVERIFIED
- Policy inference: NOT RUN
- Bounded evaluation episodes: 0
- Automatic demo policy: UNVERIFIED
- Candidate demo: NOT GENERATED
- Schema validation: NOT APPLICABLE
- Replay-buffer validation: NOT APPLICABLE

The automatic-demo path stopped before policy inference because checkpoint restoration did not complete within the bounded diagnostic window. No action was sampled and no demonstration file was created.

## Checkpoint Inventory

Non-empty delivered checkpoint directories:

- `checkpoints/cable_route_shutdown_test/checkpoint_0`
- `checkpoints/cable_route_shutdown_test/checkpoint_10`
- `checkpoints/cable_route_shutdown_test/checkpoint_20`
- `checkpoints/cable_route_smoke/checkpoint_0`
- `checkpoints/cable_route_smoke/checkpoint_100`
- `checkpoints/cable_route_smoke/checkpoint_200`

Empty named directories:

- `checkpoints/cable_route_v0`
- `checkpoints/cable_route_v0_20k`

No delivered path is labelled final, production, or validated. The selected restore candidate was the highest numbered delivered checkpoint:

`external/packages/cable_routing_v2/CableRoute_最终交接包_V2_20260820_221714/checkpoints/cable_route_smoke/checkpoint_200`

Its `smoke` label and step 200 do not establish policy quality.

## Training and Restore Call Chain

The delivered training entry is `async_cable_route_sim/async_drq_sim.py`. It constructs the agent through `make_drq_agent` with `encoder_type="resnet-pretrained"`; the environment-dependent image keys are `front` and `wrist`. `make_drq_agent` constructs a `DrQAgent` with two pretrained ResNet encoders, proprioceptive state input, a tanh-squashed policy, two-layer 256-unit actor/critic MLPs, and a ten-member critic ensemble.

The training entry saves `agent.state` with `flax.training.checkpoints.save_checkpoint`. The corresponding SERL evaluation pattern is:

1. construct the same `DrQAgent` from the wrapped observation/action spaces;
2. call `flax.training.checkpoints.restore_checkpoint(checkpoint_path, agent.state, step=...)`;
3. replace the agent state with the restored state;
4. infer deterministically with `agent.sample_actions(observations=jax.device_put(obs), argmax=True)`.

The required learner-facing observation keys are `state`, `front`, and `wrist`, with shapes `(1, 7)`, `(1, 128, 128, 3)`, and `(1, 128, 128, 3)`. The action shape is `(3,)` float32.

## Restore Attempts

Attempt 1 used a relative checkpoint path. Orbax rejected the call before reading checkpoint content:

```text
ValueError: Checkpoint path should be absolute. Got external/.../checkpoint_200
```

This was a harness invocation error, not evidence of a bad checkpoint. The retry used the resolved absolute directory.

Attempt 2 constructed the exact environment-derived target state and used the absolute checkpoint directory. It did not complete or return a structural error within approximately 150 seconds and was interrupted without running inference.

Attempt 3 removed MuJoCo from the restore path, used exact validated sample shapes, forced JAX CPU, and constructed a 598-leaf target state. Orbax again did not return within a hard 180-second limit and the command exited with code 124.

The runtime also emitted an existing optional TensorFlow-Datasets/protobuf import warning. It did not prevent target-agent construction, and it is not proven to be the restore stall's cause.

Because no restore call returned a restored state, architecture compatibility cannot be marked VALIDATED even though checkpoint metadata names `modules_actor`, `encoder_front`, and `encoder_wrist` consistently with the current DrQ construction.

## Safety Outcome

- Cable V2 source: unchanged
- SERL source: unchanged
- checkpoints: unchanged
- reward and success logic: unchanged
- evaluation episodes: none
- candidate or formal demo pickle: none
- existing demo outputs: unchanged

## Required Follow-up

Profile the Orbax restore operation in an operator-approved longer diagnostic window, including process resource usage and checkpoint storage-read behavior. Only after restore returns successfully should a project-owned evaluation harness execute up to three deterministic policy episodes. Formal Cable demo acquisition remains a separate TODO.
