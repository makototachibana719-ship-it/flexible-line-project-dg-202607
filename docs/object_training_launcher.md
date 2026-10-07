# Object Training Launcher

`training/object_grasping/run_object_training.py` is the project boundary around the unchanged delivered `async_drq_sim.py` implementation.

It freezes `DRQ_PLUS_DEMO_REPLAY`, fresh initialization, dual `front/wrist` encoders, flattened horizon-one `state`, a single canonical Demo, explicit W&B project/name, output paths, and original delivered hyperparameters. `--dry-run` validates Demo identity/schema/count and observation/action contracts without W&B or training. Real roles delegate actor/learner logic to the delivered source after injecting the project observation adapter and explicit W&B logger.

Commands:

```bash
MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher \
python -m training.object_grasping.run_object_training --role learner --dry-run
```

The legacy `checkpoint_4500` is never passed. Learning rates remain `make_drq_agent` defaults because the delivered entry did not explicitly override them.

Resume uses the same entry. It validates the complete bundle, restores the legacy-Flax agent and first/online `MemoryEfficientReplayBufferDataStore`, rebuilds Demo replay, offsets learner/W&B steps, subtracts restored actor progress from random exploration, requires the same W&B run with `resume="must"`, and commits future checkpoint bundles atomically through `training.object_grasping.resume`.
