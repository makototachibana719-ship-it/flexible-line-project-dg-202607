# Peg Insertion GPU Training Operator README

Stop on any failed command. Do not edit the algorithm, reward, success logic, Demo, external SERL, or external `mujoco_env`. Training is fresh; never load delivered RLPD/BC checkpoints.

## STEP 0 — Project path

```bash
cd /home/<your-user>/flexible_line_project
export PYTHON_BIN=/path/to/peg-gpu-env/bin/python
export SERL_LAUNCHER_PATH=/path/to/serl/serl_launcher
```

Expected: repository root contains `AGENTS.md`. Stop if the project or SERL launcher cannot be resolved.

## STEP 1 — NVIDIA

```bash
nvidia-smi
```

Expected: WSL2 GPU and driver displayed. Stop on failure.

## STEP 2 — Conda/runtime

```bash
conda env create -f configs/peg_insertion/environment_gpu.yaml
conda activate serl_peg_gpu
python -m pip install -e external/serl/serl_launcher
```

Expected: an independent Python 3.10.20 Peg environment. Install the official JAX/JAXlib 0.4.35 CUDA build for the actual driver and install the available `mujoco_env` source editable. Record every deviation; stop on resolution/import conflicts and do not broad-upgrade.

## STEP 3 — JAX GPU

```bash
$PYTHON_BIN -c 'import jax; print(jax.devices()); print(jax.default_backend())'
```

Expected backend: `gpu`. Stop if CPU.

## STEP 4 — Verify handoff

```bash
MUJOCO_GL=egl PYTHONPATH="$SERL_LAUNCHER_PATH" $PYTHON_BIN scripts/verify_peg_handoff.py --require-gpu
```

Expected: `PEG_HANDOFF_VERIFY: PASS`.

## STEP 5 — W&B login

```bash
$PYTHON_BIN -m wandb login
```

Never place the API key in files. Record project `flexible-line-peg-insertion`, run name, run ID and URL.

## STEP 6 — Environment smoke

```bash
MUJOCO_GL=egl PYTHONPATH="$SERL_LAUNCHER_PATH" $PYTHON_BIN -m evaluation.peg_insertion.evaluate_policy --dry-run-one-step
```

Expected reset/one-step PASS and `CONTROL_DT_SECONDS: 0.02`. Stop on asset/XML/API errors.

## STEP 7 — Bounded GPU smoke

Terminal A, then B:

```bash
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./deliverables/gpu_training_handoff/peg_insertion/run_gpu_smoke_learner.sh
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./deliverables/gpu_training_handoff/peg_insertion/run_gpu_smoke_actor.sh
```

Actor bound is 150 steps (> training start 100); learner bound is 10 updates. Validate GPU, Demo/replay adapters, communication, W&B and a complete checkpoint bundle. Success rate is not a smoke gate.

## STEP 8 — Checkpoint restore

Choose a complete smoke triple and run the resume scripts with its metadata values. Stop if any hash/identity check fails; never fall back to silent fresh initialization.

## STEP 9 — Formal-candidate continuous acceptance gate

Run exactly one fresh learner process with normal checkpoint RNG progression and no diagnostic flags:

```bash
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./scripts/run_peg_formal_candidate_smoke.sh
```

Expected: 20/20 uninterrupted original `SACAgent.update_high_utd` calls, finite losses, a verified checkpoint bundle, W&B steps 0..19, and `PEG_FORMAL_CANDIDATE_GPU_GATE=PASS`. An update over 120 seconds is a safe FAIL with no retry. Do not create `runtime_gpu_formal.env` and do not start formal training until this gate passes; PASS does not start formal automatically.

## STEP 10 — Formal training

Use empty formal output directories. Terminal A, then B:

```bash
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./deliverables/gpu_training_handoff/peg_insertion/run_train_learner.sh
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./deliverables/gpu_training_handoff/peg_insertion/run_train_actor.sh
```

Expected outputs are `results/checkpoints/peg_rlpd_baseline_seed0/` and isolated learner/actor logs. Stop rather than overwrite a non-empty output.

## STEP 11 — W&B monitoring

Confirm project/name, run ID/URL, monotonic global learner step, losses, timing and checkpoint cadence. Do not select best by loss.

## STEP 12 — Resume

```bash
export CHECKPOINT_PATH=/absolute/path/to/checkpoint_N
export WANDB_RUN_ID=<same-run-id>
export ACTOR_STEP=<metadata-actor-step>
export TRAINING_PROFILE=formal  # use smoke only for a smoke checkpoint
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./deliverables/gpu_training_handoff/peg_insertion/run_resume_learner.sh
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" ./deliverables/gpu_training_handoff/peg_insertion/run_resume_actor.sh
```

Expected: same W&B run with `resume=must`, restored online replay/RNG/index, Demo rebuilt from canonical 5,512 transitions, and next checkpoint step greater than N.

## STEP 13 — Independent evaluation

```bash
export CHECKPOINT_PATH=/absolute/path/to/checkpoint_N
SERL_LAUNCHER_PATH="$SERL_LAUNCHER_PATH" PYTHON_BIN="$PYTHON_BIN" EPISODES=50 ./deliverables/gpu_training_handoff/peg_insertion/run_eval.sh
```

Expected `summary.json` and `episodes.csv`. Evaluation uses source success thresholds and `control_dt=0.02`.

## STEP 14 — Artifact return

Populate the structure in `expected_outputs.md`, including environment, exact commands, W&B identity/URL, periodic/final/best bundles, resume evidence, raw evaluation and wall-clock runtime. Stop shipment if hashes or required files are missing.
