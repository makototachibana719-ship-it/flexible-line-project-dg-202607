# Cable Routing — Operator A GPU Handoff

Run only this task. Stop on any failed command or unmet expected result; never modify algorithm/data/config to bypass a gate.

## STEP 0 — Project transfer
Command: `export PROJECT_ROOT=/absolute/path/to/flexible_line_project; cd "$PROJECT_ROOT"; test -f AGENTS.md; test -f deliverables/gpu_training_handoff/cable/demo_manifest.json`
Expected exit 0. Stop if absent; complete `TRANSFER_CHECKLIST.md` first.

## STEP 1 — WSL2 GPU check
Command: `nvidia-smi`
Expected intended NVIDIA GPU, VRAM and driver. Stop if unavailable.

## STEP 2 — Conda environment
Command: `conda env create -n serl_cable_gpu -f configs/cable/environment_gpu.yaml; conda activate serl_cable_gpu; python -m pip install -e external/serl/serl_launcher; python -m pip install -e 'external/packages/cable_routing_v2/CableRoute_最终交接包_V2_20260820_221714/franka_sim'`
Expected unique Python 3.10.20 environment and editable installs. If YAML name conflicts with `-n`, omit `-n`; install official JAX/JAXlib 0.4.35 CUDA build per `environment_spec.md`. Stop on dependency/import error; do not broad-upgrade.

## STEP 3 — JAX GPU verification
Command: `python -c 'import jax; print(jax.devices()); print(jax.default_backend())'`
Expected CUDA device and `gpu`. Stop on `cpu`.

## STEP 4 — Task verifier
Command: `python scripts/verify_cable_handoff.py --require-gpu`
Expected `CABLE_HANDOFF_VERIFY: PASS`. Stop on any hash/schema/path/output/W&B/backend failure.

## STEP 5 — W&B login
Command: `python -m wandb login`
Expected authenticated account without repository credentials. Stop if login fails. Identity: project `flexible-line-cable`, smoke `cable_gpu_smoke_5demo_seed0`, formal `cable_baseline_5demo_seed0`.

## STEP 6 — Environment smoke
Command: `MUJOCO_GL=egl python evaluation/smoke/smoke_cable_routing_v2.py`
Expected import/create/reset/one-step/close PASS. Stop on lifecycle/schema mismatch.

## STEP 7 — Bounded GPU smoke
Learner: `PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_gpu_smoke_learner.sh`
Actor after server startup: `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/cable/run_gpu_smoke_actor.sh`
Expected 10 learner updates, 400 actor steps, online smoke W&B, 10,526 replay records and complete checkpoint within 30 minutes. Stop on partial output.

## STEP 8 — Checkpoint restore acceptance
Command: `MUJOCO_GL=egl python -m evaluation.cable.checkpoint_roundtrip --config deliverables/gpu_training_handoff/cable/baseline_config_snapshot.yaml --output-dir runtime/cable_handoff/gpu_checkpoint_roundtrip`
Expected `REAL_AGENT_CHECKPOINT_ROUNDTRIP: VALIDATED`, inference PASS, plus smoke-checkpoint restore/evaluation. Stop otherwise.

## STEP 9 — Formal training
Precheck: `test ! -d results/checkpoints/cable_baseline_5demo_seed0 || test -z "$(ls -A results/checkpoints/cable_baseline_5demo_seed0)"`
Learner: `PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_train_learner.sh`
Actor after server startup: `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/cable/run_train_actor.sh`
Expected isolated outputs and declared W&B identity. Stop rather than overwrite or on non-finite/error state.

## STEP 10 — W&B monitoring
Command: `python -m wandb status`
Expected online status; confirm project/name/ID/URL, monotonic steps, metrics and checkpoint cadence. Stop on wrong identity/history. Loss is not success.

## STEP 11 — Resume after interruption
Learner: `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<metadata-id> PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_resume_learner.sh`
Actor: `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<metadata-id> ACTOR_STEP=<metadata-actor_steps> LEARNER_IP=localhost PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_resume_actor.sh`
Expected same W&B run (`resume=must`), replay/RNG/steps restored, next complete checkpoint > N. Stop on sidecar/hash/config/W&B/collision mismatch.

## STEP 12 — Independent evaluation
Command: `CHECKPOINT_PATH=/abs/checkpoint_N EPISODES=50 OUTPUT_DIR=results/single_skill/cable/formal_seed0 PYTHON_BIN=python deliverables/gpu_training_handoff/cable/run_eval.sh`
Expected `summary.json` and `episodes.csv`. Stop on restore/episode failure; best selection follows declared evaluation rule.

## STEP 13 — Return artifacts
Command: `test -f results/single_skill/cable/formal_seed0/summary.json; test -f results/single_skill/cable/formal_seed0/episodes.csv; test -f cable_training_return/training_handoff_report.md`
Expected exit 0 and complete `cable_training_return/{environment,configs,logs,wandb,checkpoints,evaluation}` per `expected_outputs.md`. Stop shipment on missing evidence/hash/report fields.
