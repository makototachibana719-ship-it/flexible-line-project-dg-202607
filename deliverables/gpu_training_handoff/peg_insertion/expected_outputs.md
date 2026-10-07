# Peg Expected Outputs and Return Contract

Formal checkpoints are complete triples: `checkpoint_N`, `checkpoint_N_online_replay.pkl`, and `checkpoint_N_resume_metadata.json`. Metadata is committed last. Periodic cadence is 1,000 learner steps. Final is the last complete bundle after normal completion. Best is selected only by independent evaluation: highest success rate, then lower insertion errors, then higher checkpoint step; never by training loss.

Return exactly:

```text
peg_training_return/
├── README.md
├── environment/
├── configs/
├── logs/
├── wandb/
├── checkpoints/{periodic,final,best}/
├── evaluation/{summary.json,episodes.csv,report.md}
└── training_handoff_report.md
```

Record GPU/driver/CUDA, Python/JAX/JAXlib/MuJoCo, frozen config, exact commands, W&B URL/run ID, logs, all checkpoint classes, resume records, raw evaluation, success/errors/failures, and wall-clock runtime.
