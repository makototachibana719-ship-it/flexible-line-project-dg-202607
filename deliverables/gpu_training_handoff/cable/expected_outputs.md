# Cable Expected Outputs

## Checkpoint policy

- Periodic checkpoint: legacy Flax checkpoint saved every 10,000 learner outer updates during formal training. GPU smoke uses interval 4 only to exercise persistence within 10 updates.
- Final checkpoint: the highest successfully committed periodic checkpoint at normal formal-training completion. If the loop ends between periods, the current launcher does not emit an additional forced-final save; report that limitation explicitly.
- Best checkpoint: `BEST_CHECKPOINT = UNDEFINED_UNTIL_EVALUATION`. Select only after independent evaluation using highest success rate, then lowest mean final distance, then highest checkpoint step as deterministic tie-break. Training loss/reward cannot define best.

## Resume contract

```text
TRAINING_RESUME = VALIDATED
```

Resume requires a complete three-file bundle: `checkpoint_N`, `checkpoint_N_online_replay.pkl`, and `checkpoint_N_resume_metadata.json`. The launcher restores full agent state and online replay, rebuilds Demo replay, offsets learner/W&B/checkpoint steps, requires actor progress, and requests the exact W&B run with `resume="must"`. Any missing or mismatching component fails closed. See `docs/cable_training_resume.md`.

## Formal outputs

- Checkpoints: `results/checkpoints/cable_baseline_5demo_seed0/`
- Logs/W&B local files: `results/logs/cable_baseline_5demo_seed0/`
- Evaluation: operator-selected empty directory below `results/single_skill/cable/`

All output directories must be empty before their owning command begins. Existing directories are evidence and must not be overwritten.

## Returned layout

```text
cable_training_return/
├── README.md
├── environment/
├── configs/
├── logs/
├── wandb/
├── checkpoints/
│   ├── periodic/
│   ├── final/
│   └── best/
├── evaluation/
│   ├── summary.json
│   ├── episodes.csv
│   └── report.md
└── training_handoff_report.md
```

The `best/` directory remains empty with an explanation until the checkpoint-selection evaluation is complete.
