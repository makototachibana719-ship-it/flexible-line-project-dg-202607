# Cable Checkpoint Save Diagnosis

Date: 2026-08-21  
Milestone: `MILESTONE 1C-CABLE-CHECKPOINT-SAVE-DIAGNOSIS`

## Smoke evidence boundary

User-confirmed real smoke results:

```text
WANDB_ONLINE = VALIDATED
WANDB_PROJECT = flexible-line-cable
WANDB_RUN = cable_smoke_5demo_seed0
ACTOR_SMOKE = PASS
ACTOR_STEPS = 500
ACTOR_SHUTDOWN = CLEAN
LEARNER_SMOKE = PARTIAL
SMOKE_TRAINING = PARTIAL
FORMAL_TRAINING = NOT_STARTED
FORMAL_EVALUATION = NOT_YET_VALIDATED
```

The learner reached the real update path but failed during checkpoint save. This is not formal training success, checkpoint validation, policy validation, success-rate validation, or evaluation validation. EGL destructor messages remain `KNOWN_NONBLOCKING_CLEANUP_WARNING`.

## Installed versions

Read from `/home/ubuntu/miniconda3/envs/serl/bin/python`:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| Flax | 0.8.5 |
| Orbax Checkpoint | 0.6.4 |
| TensorStore | 0.1.51 |
| JAX | 0.4.35 |
| JAXlib | 0.4.35 |

`flax_use_orbax_checkpointing=True` in this environment.

## Save call and trigger

Delivered learner source:

```python
if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
    checkpoints.save_checkpoint(
        FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
    )
```

Smoke config sets `checkpoint_period=99`. At the first loop iteration, `update_steps` is still zero, and `0 % 99 == 0`. The call therefore uses:

```text
step = 0
keep = 20
overwrite = False  # Flax default
orbax_checkpointer = None  # Flax creates its default PyTree checkpointer
```

It is not a pre-update save: the first iteration has already performed three explicit critic-only calls and one `update_high_utd(..., utd_ratio=1)` call, then W&B logging, before reaching checkpoint save. However, `update_steps += 1` occurs after saving, and tqdm has not completed/refreshed the first iteration. A save exception therefore leaves the display at `learner: 0% | 0/100`.

## Minimal reproduction

Project debug entry:

```text
evaluation/debug/debug_cable_checkpoint_save.py
```

Outputs are isolated under:

```text
runtime/checkpoint_debug/20260821_085913_311734/
```

Results:

```text
TENSORSTORE_ASSUME_CONFIG = FAIL
DEFAULT_ORBAX_SAVE = FAIL_TIMEOUT (15 seconds)
LEGACY_FLAX_SAVE = PASS
LEGACY_FLAX_RESTORE = PASS
ROOT_CAUSE = ORBAX_TENSORSTORE_MISMATCH
RESULT = PASS
```

The exact Orbax-emitted TensorStore member was independently reproduced:

```text
ValueError: Error parsing object member "kvstore":
Object includes extra members: "assume_config"
```

Orbax 0.6.4 source adds `assume_config=True` to the kvstore spec. TensorStore 0.1.51 rejects that member. The full default Flax/Orbax save did not return within a killable 15-second subprocess, while the same equivalent array pytree saved and restored immediately through Flax legacy serialization.

## Root cause

```text
ROOT_CAUSE = ORBAX_TENSORSTORE_MISMATCH
```

This is not caused by Demo contents, DrQ update semantics, W&B, the Cable environment, checkpoint directory selection, `keep`, or `overwrite`.

## Selected minimal fix

Selected priority solution: **B — Flax legacy checkpoint path**.

The project launcher now captures the original `flax.training.checkpoints.save_checkpoint` function and wraps only each project-launched save with:

1. remember current `flax_use_orbax_checkpointing`;
2. set it to `False`;
3. call the original Flax saver with unchanged directory/target/step/keep arguments;
4. restore the previous config value in `finally`.

No external source or installed package is changed. A direct project-wrapper save/restore test also passed. The output format for new project Cable checkpoints is therefore Flax legacy until an environment change is separately approved and validated.

## Existing and next outputs

The failed smoke evidence is preserved unchanged:

```text
results/checkpoints/cable_smoke_5demo/checkpoint_0.orbax-checkpoint-tmp-0/
results/logs/cable_smoke_5demo/wandb/run-20260821_084718-095qa15i/
```

No incomplete output was deleted or reused. The next bounded retry is configured separately:

```text
WANDB_RUN = cable_smoke_5demo_checkpoint_fix_seed0
CHECKPOINT_DIR = results/checkpoints/cable_smoke_5demo_checkpoint_fix/
LOG_DIR = results/logs/cable_smoke_5demo_checkpoint_fix/
```

## Exact next command — not executed

Terminal 1:

```bash
cd /home/ubuntu/flexible_line_project
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.2 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --role learner --smoke
```

Terminal 2, after learner server startup:

```bash
cd /home/ubuntu/flexible_line_project
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.1 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --role actor --smoke --ip localhost
```

The retry was not started in this milestone.

