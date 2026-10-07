# Cable Project Training Launcher

Date: 2026-08-21  
Milestone: `MILESTONE 1C-CABLE-PROJECT-LAUNCHER`

## Status

```text
CABLE_TRAINING_LAUNCHER = IMPLEMENTED / UNVERIFIED
DRY_RUN = PASS
SMOKE_TRAINING = PARTIAL
FORMAL_TRAINING = NOT_STARTED
FORMAL_EVALUATION = NOT_YET_VALIDATED
TRAINING_RESUME = VALIDATED
```

No Cable V2 source, SERL source, formal Demo, checkpoint, reward, success condition, controller, reset implementation, W&B credential, or Conda environment was modified. The two requested prior documents `docs/cable_baseline_training_preflight.md` and `docs/cable_baseline_training_plan.md` were absent and were not invented.

## Entry and configuration

- Launcher: `training/cable/run_cable_training.py`
- Baseline config: `configs/cable/cable_baseline_5demo.yaml`
- Runtime: `/home/ubuntu/miniconda3/envs/serl/bin/python`
- Environment: `PandaCableRouteVision-v0`
- Algorithm: DrQ
- Seed: 0

The launcher imports the delivered Cable entry as a module and reuses its `actor()` and `learner()` loops. Project-owned injection is limited to:

1. the previously validated Cable observation wrapper;
2. an explicit project W&B logger factory;
3. sequential five-file Demo population;
4. explicit runtime/config/output assertions.

The learner owns W&B and output directories. The actor does not create a W&B run or checkpoint and may connect after learner-owned directories exist.

## Safe resume extension

M1D-A2 adds only project-side resume boundaries. Fresh checkpoints now commit a matching online-replay snapshot and metadata sidecar. `--resume-from checkpoint_N` restores full agent state and online replay, rebuilds the five-Demo buffer, preserves learner/W&B step continuity, rejects collisions, and requires the same W&B run with `resume="must"`. Actor resume requires `--resume-actor-step` equal to metadata; completed random exploration is not repeated. See `docs/cable_training_resume.md`.

## W&B configuration

Formal baseline:

```text
WANDB_PROJECT = flexible-line-cable
WANDB_RUN_NAME = cable_baseline_5demo_seed0
WANDB_MODE = online
```

Smoke:

```text
WANDB_PROJECT = flexible-line-cable
WANDB_RUN_NAME = cable_smoke_5demo_checkpoint_fix_seed0
WANDB_MODE = online
```

The first real smoke run `cable_smoke_5demo_seed0` successfully created an online W&B run and completed 500 actor steps, but its learner failed during the step-0 Orbax checkpoint save. It remains preserved as PARTIAL evidence. The configured smoke name above is the isolated checkpoint-fix retry and has not been run.

The project logger calls `wandb.init(project=..., name=..., mode="online", ...)` directly. It does not use or inspect API keys or credential files. Recursive metric dictionaries are flattened with `/`, preserving the delivered logger behavior. The real online initialization remains UNVERIFIED until an explicitly authorized smoke run.

## Multi-Demo strategy

The YAML contains the five exact validated paths, expected per-file counts, and SHA256 values. No merged pickle is created.

The learner loads one pickle, validates/inserts every transition into the same memory-efficient demo buffer, deletes that top-level list, runs garbage collection, and only then opens the next file. This avoids simultaneously retaining all five pickle lists.

```text
DEMO_COUNT = 5
RAW_TRANSITIONS = 10521
EXPECTED_HORIZON_ONE_REPLAY_SIZE = 10526
DEMO_REPLAY_CAPACITY = 200000
PERMANENT_MERGED_COPY = NO
```

Dry run recomputed all five SHA256 hashes, loaded each file sequentially, asserted every transition schema, and obtained exactly 10,521 raw transitions. The real replay-size result 10,526 was already validated by the unchanged SERL buffer path in the five-demo milestone; actual training-path population will assert that same value before entering learner updates.

## Observation adapter

The launcher reuses `tools.cable_demo.CableTrainingObservationWrapper` and SERL horizon-one `ChunkingWrapper` through `make_cable_training_env()`.

Runtime assertions require exactly:

| Key | Shape | dtype |
|---|---:|---|
| `state` | `(1, 7)` | `float32` |
| `front` | `(1, 128, 128, 3)` | `uint8` |
| `wrist` | `(1, 128, 128, 3)` | `uint8` |

Action must be `(3,) float32`. The delivered singular-`image` `SERLObsWrapper` is not modified. The same project wrapper is injected into the delivered actor's evaluation-environment construction so actor, learner, DrQAgent, replay, and demonstrations share `state/front/wrist`.

## Original versus new hyperparameters

No algorithm hyperparameter was optimized or intentionally changed.

| Parameter | Original | New | Reason |
|---|---:|---:|---|
| batch size | 256 | 256 | unchanged |
| demo fraction | 50% when demo buffer exists | 50% | unchanged delivered learner behavior |
| critic/actor ratio | 4 | 4 | unchanged |
| high-UTD ratio | 1 | 1 | unchanged |
| training starts | 300 | 300 | unchanged flag default |
| random actor steps | 300 | 300 | unchanged flag default |
| encoder | `resnet-pretrained` | `resnet-pretrained` | unchanged |
| actor LR | 3e-4 | 3e-4 | unchanged DrQ default |
| critic LR | 3e-4 | 3e-4 | unchanged DrQ default |
| temperature LR | 3e-4 | 3e-4 | unchanged DrQ default |
| replay capacity | 200000 | 200000 | unchanged |
| formal max steps | 1000000 | 1000000 | unchanged |

Smoke-only bounds are not formal hyperparameter changes: learner updates are capped at 100, actor steps at 500, and checkpoint period is 99 so a bounded run can exercise checkpoint output near its end.

## Output isolation

Formal:

- `results/checkpoints/cable_baseline_5demo_seed0/`
- `results/logs/cable_baseline_5demo_seed0/`

Smoke:

- `results/checkpoints/cable_smoke_5demo_checkpoint_fix/`
- `results/logs/cable_smoke_5demo_checkpoint_fix/`

The learner refuses to start if its selected output directory is non-empty. Paths must remain inside the project and must not overlap the delivered `cable_route_smoke` or `cable_route_shutdown_test` names. Smoke never writes to the formal baseline directories.

The original partial smoke directories `results/checkpoints/cable_smoke_5demo/` and `results/logs/cable_smoke_5demo/` remain untouched. New project saves use the Flax legacy serializer because the installed Orbax 0.6.4 emits a TensorStore `assume_config` field rejected by TensorStore 0.1.51.

## Startup assertions

Before learner updates, the project path verifies:

1. all five Demo files exist;
2. configured SHA256 values;
3. raw transition count 10,521;
4. exact observation keys/shapes/dtypes;
5. `(3,)` action shape and dtype/range in Demo data;
6. replay capacity at least 10,526;
7. isolated, non-overwriting learner output paths;
8. W&B mode exactly `online`;
9. non-empty W&B project/name;
10. at least one JAX device.

Actual sequential replay population additionally asserts raw total 10,521 and final horizon-one demo-buffer size 10,526 before calling `learner()`.

## Dry-run evidence

Commands executed:

```bash
/home/ubuntu/miniconda3/envs/serl/bin/python -m py_compile training/cable/run_cable_training.py
MUJOCO_GL=egl XDG_CACHE_HOME=/tmp/cable_launcher_dryrun_cache PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --dry-run --smoke
```

Result:

```text
DELIVERED_IMPORT = PASS
DEMO_FILES = 5/5 PASS
DEMO_RAW_TRANSITIONS = 10521 PASS
OBSERVATION_ADAPTER = PASS
ACTION_SHAPE = (3,) PASS
REPLAY_CAPACITY = PASS
JAX_DEVICE = TFRT_CPU_0 PASS
WANDB_CONFIG_CONSTRUCTION = PASS
WANDB_INIT = NOT_CALLED
LEARNER_UPDATE = NOT_STARTED
DRY_RUN_RESULT = PASS
```

The known TFDS/protobuf optional-import traceback and EGL destructor warning appeared, but the delivered import and environment checks completed. No environment change was attempted.

An offline fake-W&B test also verified that the project logger supplies `project=flexible-line-cable`, `name=cable_smoke_5demo_seed0`, `mode=online`, and flattens `train/critic_loss`; it did not contact W&B.

## Exact smoke commands — not executed

Terminal 1, learner:

```bash
cd /home/ubuntu/flexible_line_project
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.2 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --role learner --smoke
```

Terminal 2, actor, after learner server startup:

```bash
cd /home/ubuntu/flexible_line_project
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.1 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --role actor --smoke --ip localhost
```

## Exact formal baseline commands — not executed

Terminal 1, learner:

```bash
cd /home/ubuntu/flexible_line_project
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.2 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --role learner
```

Terminal 2, actor, after learner server startup:

```bash
cd /home/ubuntu/flexible_line_project
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.1 /home/ubuntu/miniconda3/envs/serl/bin/python -m training.cable.run_cable_training --role actor --ip localhost
```

Do not execute formal training before bounded smoke acceptance. Smoke authorization must explicitly permit an online W&B run, 100 learner updates, 500 actor steps, and isolated checkpoint/log writes.

## Remaining risks

- Real DrQ construction/update, Agentlace two-process coordination, online metric upload, and checkpoint save are UNVERIFIED until smoke.
- A 10,526-record dual-camera replay buffer may require substantial RAM even though pickle lists are loaded sequentially.
- Delivered memory-efficient replay insertion prints per-transition buffer sizes, which may make smoke logs noisy; it was not modified.
- Formal evaluation and `eval/final.success` runtime semantics remain unvalidated.
- Fixed-reset baseline is in-distribution for current evaluation but OOD generalization risk remains HIGH.
