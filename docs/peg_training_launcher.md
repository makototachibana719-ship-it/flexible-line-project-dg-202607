# Peg GPU Training Launcher

Date: 2026-09-03

The single entry is `python -m training.peg_insertion.run_peg_training`. It delegates unchanged RLPD/SAC construction and actor/learner loops to the delivered RLPD entry while automatically scoping the portable runtime, repaired SERL replay method, explicit W&B logger, and three-file checkpoint bundles.

Every invocation must select `--profile formal`, `--profile smoke`, or `--profile diagnostic`. The eight update/RNG/save controls prefixed or classified as diagnostic are rejected by formal and smoke profiles; diagnostic profile also fails closed unless at least one such control is explicit. Normal checkpoint RNG progression is untouched outside diagnostic code.

Supported common arguments are `--dry-run`, `--role learner|actor`, `--ip`, `--resume-from`, `--resume-actor-step`, `--wandb-run-id`, and `--require-gpu`. Non-dry execution refuses CPU. Before delegation it verifies `(20,) float32`, `(7,) float32 [-1,1]`, Demo SHA/count and 20/20 provenance, 5,512 replay population, fixed config/W&B identity, safe empty output, and JAX devices.

The frozen formal contract is `RLPD_STYLE_SAC_PLUS_50_50_DEMO_REPLAY`, seed 0, 32 online + 32 Demo, UTD 1, starts 100, max 10,000, replay 1,000,000, checkpoint 1,000 and all learning rates `3e-4`. Smoke uses actor 150 and learner 10 updates. No hyperparameter selection was performed.

The formal-candidate acceptance run is a smoke-profile submode, not formal training. It launches one fresh learner process, consumes the integrity-checked bounded-smoke replay, calls the original `SACAgent.update_high_utd` 20 times continuously with normal RNG, records per-update sample/update/total timing, saves to independent output paths, and permits no restart or retry. Its only runtime override is `runtime_gpu_formal_candidate.env`; an update over 120 seconds is a safe Gate FAIL. A PASS is necessary but does not authorize or start formal training.

CPU dry-run populated all Demo transitions and returned `PEG_TRAINING_DRY_RUN: PASS`; no learner update or W&B run occurred.
