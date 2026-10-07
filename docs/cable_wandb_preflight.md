# Cable W&B Training Preflight

Date: 2026-08-21  
Scope: static inspection and local package/version discovery only. No learner, actor, training, checkpoint restore, evaluation episode, or W&B run was started.

## Runtime discovery

```text
PYTHON = /home/ubuntu/miniconda3/envs/serl/bin/python
WANDB_VERSION = 0.28.0
WANDB_BASE_URL = https://api.wandb.ai
RUNTIME_ENVIRONMENT_DISCOVERY = VALIDATED
WANDB_AUTHENTICATION = USER_CONFIRMED
```

No credential file or API key was read, printed, requested, or modified.

## Initialization path

The learner is the only process that creates a W&B run:

1. `async_cable_route_sim/async_drq_sim.py::learner()` calls `make_wandb_logger(...)`.
2. It currently passes `project="serl_cable_route"`, `description=FLAGS.exp_name or FLAGS.env`, and `debug=FLAGS.debug`.
3. `serl_launcher.utils.launcher.make_wandb_logger()` maps the description to `exp_descriptor` and `tag`.
4. `serl_launcher.common.wandb.WandBLogger.__init__()` calls `wandb.init(...)`.

`--debug=True` selects W&B mode `disabled`; the default `--debug=False` selects `online`. The actor does not initialize W&B. It sends timer/evaluation dictionaries to the learner through Agentlace, and the learner logs them.

## Requested project and run identity

Requested configuration:

```text
project = flexible-line-cable
run name = cable_baseline_5demo_seed0
mode = online
```

The current code does **not** satisfy that identity exactly:

- project is explicitly hard-coded as `serl_cable_route`;
- `--exp_name=cable_baseline_5demo_seed0` affects the generated run ID prefix and tag;
- `WandBLogger` does not pass `name=` to `wandb.init`, so the requested visible run name is not guaranteed by `--exp_name`.

Therefore `WANDB_TARGET_CONFIGURATION = NEEDS_CONFIGURATION`. An environment variable alone must not be treated as proof that it overrides the explicitly supplied project argument. Before formal or smoke training, add project-side/configurable logger inputs or obtain explicit authorization for the smallest training-entry change. Cable V2 and SERL were not modified in this preflight.

## Metrics actually logged

W&B recursively flattens nested dictionaries with `/`.

| Source | Current W&B keys | Cadence / notes |
|---|---|---|
| Learner `update_info` | `critic_loss`, `predicted_qs`, `target_qs`, `actor_loss`, `temperature`, `entropy`, `temperature_loss`, and any optimizer learning-rate keys returned by the train state | Every `log_period=10` learner outer updates; there is no `train/` prefix. |
| Learner timer | `timer/*` | Every 10 learner outer updates. |
| Actor timer | `timer/*` | Sent every 10 actor environment steps and logged by the learner at its then-current learner update step. This shares the same namespace as learner timer values. |
| Evaluation | `eval/*` derived from reset/step `info`, including success and distance fields when emitted; terminal fields are additionally under `eval/final.*` | Actor initiates 5 trajectories every 2000 actor steps. The evaluator discards reward and does not explicitly calculate return or episode length itself. `RecordEpisodeStatistics` may add terminal `episode.*` fields. |

### Requested metric checklist

| Metric | Status | Evidence / meaning |
|---|---|---|
| `train/episode_return` | MISSING | `running_return` is accumulated in the actor and then reset without logging. |
| critic loss | PRESENT | Uploaded as `critic_loss`, not `train/critic_loss`. |
| actor loss | PRESENT | Uploaded as `actor_loss`, not `train/actor_loss`. |
| buffer size | MISSING_FROM_WANDB | Replay/demo sizes are printed locally only. |
| training success | MISSING | Actor training episodes do not send success statistics. |
| `eval/*` | PRESENT_IN_CODE, UNVERIFIED_AT_RUNTIME | Evaluation sends averaged environment info. No evaluation was run in this preflight. |
| evaluation success | PRESENT_IN_CODE, semantics require care | `eval/final.success` is the mean terminal success flag across evaluation episodes and is the closest existing success-rate metric. `eval/success` averages success over every recorded step and must not be presented as episode success rate. |
| evaluation return | POTENTIALLY_PRESENT_VIA_WRAPPER, UNVERIFIED | If Gym `RecordEpisodeStatistics` injects terminal `episode.r`, flattening yields an `eval/episode.r`-style key. The evaluator itself discards reward, so this must be runtime-confirmed before use. |

No missing metric was synthesized.

## Global step semantics

The W&B step is learner-local `update_steps`, initialized to zero and incremented once per outer learner loop iteration. With the default `critic_actor_ratio=4`, one W&B step corresponds to three explicit critic-only updates followed by `update_high_utd(..., utd_ratio=1)`, which performs one more critic update and one actor/temperature update. It is **not** the actor environment-step count, transition count, episode count, or raw gradient-update count.

Actor timer/evaluation messages are asynchronous. When received, they are logged at the learner's current `update_steps`, not at the actor step that triggered them.

## Five-demo and training-entry gaps

The five validated formal files remain separate. The delivered learner accepts one `--demo_path`, opens one pickle, and inserts its transitions directly. It does not accept a list of paths. The previous audit established that in-memory concatenation is compatible, but deliberately did not write a merged file.

The delivered training entry also still applies SERL's singular-`image` observation wrapper to the native nested `images/{front,wrist}` environment. The validated demonstrations use the project-side learner-facing `state/front/wrist` structure. This wrapper mismatch must be resolved with a project-side adapter before training; W&B readiness does not resolve it.

```text
FIVE_DEMO_MERGED_ARTIFACT = MISSING
TRAINING_OBSERVATION_ADAPTER = TODO
SMOKE_TRAINING_READINESS = BLOCKED
```

## Intended bounded smoke configuration

After the W&B identity, observation adapter, and five-demo input gaps are resolved, use:

```text
W&B project: flexible-line-cable
W&B run name: cable_baseline_5demo_seed0
W&B mode: online
seed: 0
runtime: /home/ubuntu/miniconda3/envs/serl/bin/python
rendering: MUJOCO_GL=egl
debug: false
```

The eventual training entry is a two-process Agentlace job. The exact command template is:

```bash
cd /home/ubuntu/flexible_line_project/external/packages/cable_routing_v2/CableRoute_最终交接包_V2_20260820_221714/async_cable_route_sim
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python async_drq_sim.py --learner --seed=0 --debug=False --exp_name=cable_baseline_5demo_seed0 --demo_path=/ABSOLUTE/PATH/TO/VALIDATED_FIVE_DEMO_MERGED.pkl --max_steps=<BOUNDED_SMOKE_STEPS>
```

Then, in a second terminal after the learner server is ready:

```bash
cd /home/ubuntu/flexible_line_project/external/packages/cable_routing_v2/CableRoute_最终交接包_V2_20260820_221714/async_cable_route_sim
MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/serl/bin/python async_drq_sim.py --actor --ip=localhost --seed=0 --debug=False --exp_name=cable_baseline_5demo_seed0 --max_steps=<BOUNDED_SMOKE_STEPS>
```

These are exact entry-point/flag templates, but intentionally contain explicit placeholders and are **not launch-ready**. Supplying a made-up merged path or smoke-step budget would be misleading. The target W&B project/name must be made configurable first; the merged artifact must be created and validated without overwriting the five originals; and the project-side observation adapter must be selected. No training command was executed.

## Result

```text
WANDB_PREFLIGHT = COMPLETED
WANDB_RUNTIME = VALIDATED
WANDB_ONLINE_PATH = PRESENT
WANDB_TARGET_CONFIGURATION = NEEDS_CONFIGURATION
METRIC_INVENTORY = VALIDATED_BY_STATIC_INSPECTION
SMOKE_TRAINING = NOT_STARTED
TRAINING_RESULTS = UNVERIFIED
```
