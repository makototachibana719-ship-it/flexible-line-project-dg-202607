# Object Grasping — Operator B GPU Handoff

Run only this task. Fresh training only; legacy `checkpoint_4500` is never loaded. Stop on every failed gate.

## STEP 0 — Project transfer
Command: `export PROJECT_ROOT=/absolute/path/to/flexible_line_project; cd "$PROJECT_ROOT"; test -f AGENTS.md; test -f deliverables/gpu_training_handoff/object_grasping/demo_manifest.json`
Expected exit 0. Stop and complete `TRANSFER_CHECKLIST.md` if absent.

## STEP 1 — WSL2 GPU check
Command: `nvidia-smi`
Expected intended NVIDIA GPU/driver/VRAM. Stop otherwise.

## STEP 2 — Conda environment
Command: `conda env create -n serl_object_gpu -f configs/object_grasping/environment_gpu.yaml; conda activate serl_object_gpu; python -m pip install -e external/serl/serl_launcher`
Expected unique Python 3.10.20 environment. If YAML name conflicts, omit `-n`; install matching JAX/JAXlib 0.4.35 CUDA build and local Object package per `environment_spec.md`. Stop on resolution/import error; do not broad-upgrade.

## STEP 3 — JAX GPU verification
Command: `python -c 'import jax; print(jax.devices()); print(jax.default_backend())'`
Expected `gpu`. Stop on CPU.

## STEP 4 — Task verifier
Command: `python scripts/verify_object_handoff.py --require-gpu`
Expected `OBJECT_HANDOFF_VERIFY: PASS`. Stop on failure.

## STEP 5 — W&B login
Command: `python -m wandb login`
Expected private credential login. Stop on failure; never share/store API keys. Identity: `flexible-line-object-grasp`, smoke `object_gpu_smoke_seed0`, formal `object_baseline_seed0`.

## STEP 6 — Environment smoke
Command: `MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher python evaluation/smoke/smoke_object_grasping.py`
Expected lifecycle PASS. Stop on API/schema/assets failure.

## STEP 7 — Bounded GPU smoke
Learner: `PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_gpu_smoke_learner.sh`
Actor: `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/object_grasping/run_gpu_smoke_actor.sh`
Expected Demo load, actor beyond 500, 10 learner updates, online W&B and complete checkpoint within 30 minutes. Stop on partial result.

## STEP 8 — Checkpoint restore acceptance
Command: `MUJOCO_GL=egl PYTHONPATH=external/serl/serl_launcher python -m evaluation.object_grasping.checkpoint_roundtrip --output-dir runtime/object_handoff/gpu_roundtrip`
Expected save/rebuild/restore/inference PASS and smoke-checkpoint restore. Stop otherwise.

## STEP 9 — Formal training
Precheck: `test ! -d results/checkpoints/object_baseline_seed0 || test -z "$(ls -A results/checkpoints/object_baseline_seed0)"`
Learner: `PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_train_learner.sh`
Actor: `PYTHON_BIN=python LEARNER_IP=localhost deliverables/gpu_training_handoff/object_grasping/run_train_actor.sh`
Expected fresh initialization, isolated outputs, correct W&B identity. Stop on collision/error/non-finite values.

## STEP 10 — W&B monitoring
Command: `python -m wandb status`
Expected online status and correct name/ID/URL, monotonic metrics/checkpoints. Stop on wrong identity/history; loss is not success.

## STEP 11 — Resume after interruption
Learner: `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<metadata-id> PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_resume_learner.sh`
Actor: `CHECKPOINT_PATH=/abs/checkpoint_N WANDB_RUN_ID=<metadata-id> ACTOR_STEP=<metadata-actor_steps> LEARNER_IP=localhost PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_resume_actor.sh`
Expected same W&B run, restored replay/RNG/steps, rebuilt Demo, next complete checkpoint > N. Stop on sidecar/hash/config/Demo/W&B/collision mismatch.

## STEP 12 — Independent evaluation
Command: `CHECKPOINT_PATH=/abs/checkpoint_N EPISODES=50 OUTPUT_DIR=results/single_skill/object/formal_seed0 PYTHON_BIN=python deliverables/gpu_training_handoff/object_grasping/run_eval.sh`
Expected `summary.json`, `episodes.csv`, success and failure records. Stop on restore/episode failure.

## STEP 13 — Return artifacts
Command: `test -f results/single_skill/object/formal_seed0/summary.json; test -f results/single_skill/object/formal_seed0/episodes.csv; test -f object_training_return/training_handoff_report.md`
Expected complete `object_training_return/{environment,configs,logs,wandb,checkpoints,evaluation}` plus report. Stop shipment on missing hashes/evidence. Historical ~70% is not repository validation.
