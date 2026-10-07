# Object Training Resume Contract

Required classification:

| State | Policy |
|---|---|
| Agent state | MUST_RESTORE |
| Learner update step | MUST_RESTORE |
| Actor cumulative step | MUST_RESTORE |
| Online replay contents/index/RNG | MUST_RESTORE |
| Demo replay | CAN_REBUILD from canonical immutable Demo |
| W&B run identity | MUST_RESTORE with `resume="must"` |

A complete bundle must contain a legacy-Flax new-training checkpoint, metadata, online replay snapshot and hash, learner/actor steps, config/manifest hashes, and W&B run ID. Resume must reject partial bundles, mismatches, and step regression.

Metadata is committed last and is the complete-bundle marker. Every target checkpoint/replay/metadata path rejects collisions.

The cross-process bounded test restored learner step 123, actor step 17, five online replay records with index/RNG equality, rebuilt 2,000 raw Demo transitions into a 2,001-record horizon-one replay, reconstructed strict W&B identity, sampled one finite replay batch, ran one restored deterministic inference, committed step 125, and preserved the step-123 hash. Missing replay, bad replay hash, bad Demo hash, config mismatch, W&B mismatch, and collision all failed closed.

`OBJECT_RESUME_GATE = VALIDATED`
