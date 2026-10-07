# Object Grasp Algorithm Mapping

Date: 2026-08-22  
Status: source/log/replay mapping validated; policy quality not evaluated.

## Identity conclusion

The delivered executable class is `serl_launcher.agents.continuous.drq.DrQAgent`, constructed only through `make_drq_agent`. The implementation uses a visual actor/critic, pretrained ResNet-10 encoders for `front` and `wrist`, proprioception, two 256-unit actor/critic layers, a 10-member critic ensemble with two critics subsampled, entropy temperature, `update_critics`, and `update_high_utd`.

When a Demo is supplied, the learner selects `batch_size // 2` from online replay and the same amount from Demo replay, concatenating them into every critic and actor update. The source itself describes this as “50/50 sampling from RLPD”; the supplied launcher is named `tmux_rlpd_launch.sh`. There is no `DPRL` symbol, class, loss, replay implementation, paper reference, or configuration. Repository search also found no reward relabeling or prioritized replay. The pretrained visual representation is the standard selected encoder path, not evidence of a distinct DPRL implementation.

Therefore the truthful code-level classification is:

```text
OBJECT_ALGORITHM = DRQ_PLUS_DEMO_REPLAY
DPRL_IS_TEAM_LABEL = UNVERIFIED
DPRL_IMPLEMENTATION_PRESENT = NO_EVIDENCE
```

If the team intends “DPRL” to denote this implementation, that naming relationship still requires an authoritative definition; it cannot change the code-level classification.

## Demo consumer and batches

- Loader: direct trusted pickle load in `async_drq_sim.py`, followed by one `demo_buffer.insert(traj)` per list element.
- Input: flat transition dictionaries with the six delivered fields.
- Buffer: `MemoryEfficientReplayBufferDataStore` with `front,wrist` image keys.
- Default capacity/batch: 200,000 / 256.
- Delivered run2 config: capacity 10,000; total batch 8.
- With Demo: online batch 4 + Demo batch 4 = 50/50.
- Without Demo: online batch equals the full configured batch.
- No transition conversion or reward relabeling occurs in the loader.

## Training entry mapping

```text
OBJECT_TRAINING_ENTRY = code/example/async_drq_sim.py
OBJECT_ACTOR_ENTRY = code/example/run_actor.sh -> async_drq_sim.py --actor
OBJECT_LEARNER_ENTRY = code/example/run_learner.sh -> async_drq_sim.py --learner
OBJECT_TMUX_ENTRY = code/example/tmux_rlpd_launch.sh
```

Run2 evidence records seed 0, `resnet-pretrained`, 5,000 actor/learner steps, replay capacity 10,000, training start 500, batch 8, critic/actor ratio 1, checkpoint period 500, actor random steps 500, and evaluation period 500. Learner W&B construction uses project `serl_dev` and the experiment description. Formal W&B run identity is not frozen.

The text config says `env=PandaPickCube-v0`, but the actual learner log prints the vision contract `{front,state,wrist}` and two ResNet encoders. The checkpoint is therefore mapped to the vision DrQ execution evidence, and the config's non-vision env line is recorded as provenance drift.

## Evaluation gap

The source contains actor-side periodic evaluation collecting return, length, any-step `info.success`, and maximum lift height. `scripts/eval.sh` only extracts existing log lines and is not an independent checkpoint restore/evaluation entry. No formal checkpoint evaluation or success rate exists.

Required later metrics are success count/rate, return mean/std, episode length, execution time, maximum/final lift height, final block height, and failure categories, using explicit outer truncation semantics.

```text
OBJECT_FORMAL_EVALUATION = MISSING
```
