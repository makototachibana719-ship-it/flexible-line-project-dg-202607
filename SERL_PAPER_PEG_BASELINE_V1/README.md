# SERL_PAPER_PEG_BASELINE_V1

Isolated paper-style SERL/RLPD baseline for the existing Aubo Peg environment.
It does not modify the official/PASS pipeline.

Runtime invariants: one standard SERL SAC policy, 32 Demo + 32 Online replay,
Obs20/Action7 only, UTD=1, no hierarchy, no custom loss, no HIL.

Stages are intentionally fresh, same-seed runs. This avoids mutating or
resuming any PASS checkpoint/replay bundle and makes each budget independently
auditable. Promotion is gated by policy-only evaluation.

Commands (inside `/home/makoto/flexible_line_project`):

```bash
/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python \
  SERL_PAPER_PEG_BASELINE_V1/run_baseline.py --role preflight \
  --updates 1000 --output SERL_PAPER_PEG_BASELINE_V1/runs/stage_1000

/home/makoto/miniconda3/envs/serl_peg_gpu/bin/python \
  SERL_PAPER_PEG_BASELINE_V1/run_baseline.py --role supervise \
  --updates 1000 --output SERL_PAPER_PEG_BASELINE_V1/runs/stage_1000
```

