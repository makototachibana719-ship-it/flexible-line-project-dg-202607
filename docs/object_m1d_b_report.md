# M1D-B Object Grasp Demo + Algorithm Validation Report

Date: 2026-08-22

## Outcome

M1D-B is `PARTIAL`. Environment, schema, replay compatibility, and code-level algorithm identity have direct evidence. Demo success/episode boundaries cannot be validated from the delivered pickle. Checkpoint parameters map strongly to the constructed visual DrQ target, but exact state paths mismatch by three optimizer entries and the single bounded restore timed out; inference remains unverified.

## Gate matrix

| Gate | Result | Evidence |
|---|---|---|
| `OBJECT_ENVIRONMENT_GATE` | `VALIDATED` | Existing lifecycle PASS plus source/runtime contract and 20 reset audit. |
| `OBJECT_DEMO_SCHEMA_GATE` | `VALIDATED` | 2,000 finite six-field transitions; actual shapes/dtypes verified. |
| `OBJECT_DEMO_SUCCESS_GATE` | `UNVERIFIED` | No boundary, success, block position/height, or lift-height data. |
| `OBJECT_REPLAY_GATE` | `VALIDATED` | 2,000 inserts, replay 2,001, packed batch sample PASS. |
| `OBJECT_ALGORITHM_MAPPING_GATE` | `VALIDATED` | `DRQ_PLUS_DEMO_REPLAY`; no repository evidence for distinct DPRL implementation. |
| `OBJECT_CHECKPOINT_MAPPING_GATE` | `MISMATCH` | Checkpoint has three extra post-update optimizer paths. |
| `OBJECT_CHECKPOINT_RESTORE_GATE` | `UNVERIFIED_TIMEOUT` | One bounded 120-second read-only attempt did not return. |

```text
OBJECT_DEMO_VALIDATED = NO
OBJECT_ALGORITHM_MAPPING_VALIDATED = YES
OBJECT_REPLAY_COMPATIBILITY_VALIDATED = YES
OBJECT_CHECKPOINT_MAPPING_VALIDATED = NO
OBJECT_M1D_B_STATUS = PARTIAL
```

## Blocking evidence and next action

Before M1D-C, obtain authoritative Demo episode-boundary/success metadata or provenance sufficient to reproduce the 20 claimed successful trajectories. If unavailable, only then propose newly recorded success-labelled demonstrations. Separately, select a project-side checkpoint restore strategy that handles the three optimizer-state paths and passes a bounded full restore plus one action inference without changing the delivered checkpoint.

Formal evaluation remains missing. No training, GPU smoke, Demo conversion, checkpoint conversion, or package/SERL/environment mutation occurred.

## M1D-B2 addendum

The prior checkpoint mapping `MISMATCH` is superseded: the three extra metadata entries are non-array Optax EmptyState sentinels, and exact structural mapping is `VALIDATED_MATCH`. A new matching-target restore still timed out beyond 300 seconds, so restore/inference remain unresolved. Demo provenance search found no delivered generator or success-only guarantee; 20×100 is strongly supported but not authoritative. Overall status remains `PARTIAL`. See `docs/object_m1d_b2_report.md`.
