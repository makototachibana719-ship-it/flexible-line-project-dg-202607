# M1D-B2 Object Demo Provenance + Checkpoint Closure Report

Date: 2026-08-22

## Outcome

M1D-B2 closes the checkpoint architecture-mapping diagnosis but does not close Demo success provenance or checkpoint runtime restoration. The earlier 601-versus-598 mismatch was a flattening artifact: the three additional metadata entries are Optax EmptyState sentinels, and exact structural architecture is `MATCH`.

No Object Demo generator or success-only persistence guarantee is delivered. The 20×100 boundary hypothesis is strongly supported but not authoritative, so it was not used to manufacture episode statistics. The new 300-second restore retry still did not return, and no inference ran.

## Gate matrix

| Gate | Result | Evidence |
|---|---|---|
| `OBJECT_DEMO_SCHEMA_GATE` | `VALIDATED` | Unchanged M1D-B evidence. |
| `OBJECT_DEMO_BOUNDARY_PROVENANCE` | `STRONGLY_SUPPORTED_NOT_AUTHORITATIVE` | 2,000 transitions, 20-demo claims, 100-step limit; no generator/boundaries. |
| `OBJECT_DEMO_SUCCESS_GATE` | `UNVERIFIED` | No direct data or generator guarantee. |
| `OBJECT_REPLAY_GATE` | `VALIDATED` | Unchanged M1D-B evidence. |
| `OBJECT_ALGORITHM_MAPPING_GATE` | `VALIDATED` | Unchanged `DRQ_PLUS_DEMO_REPLAY` evidence. |
| `OBJECT_CHECKPOINT_MAPPING_GATE` | `VALIDATED_MATCH` | 598 arrays + 3 EmptyState nodes exactly match 601 metadata paths. |
| `OBJECT_CHECKPOINT_RESTORE_GATE` | `UNVERIFIED_TIMEOUT` | Matching-target restore exceeded 300 seconds. |
| `OBJECT_POLICY_INFERENCE_GATE` | `NOT_RUN` | Restore returned no state. |

```text
OBJECT_M1D_B_STATUS = PARTIAL
```

## Remaining closure requirements

1. Obtain authoritative generator/sidecar evidence proving real episode boundaries and success-only persistence, or keep Demo success unverified and plan success-aware re-recording in a separately authorized task.
2. Diagnose the Orbax/TensorStore read stall using a project-side, non-converting restore diagnostic or the exact producing runtime, then obtain a returned state and run one bounded `(4,)` inference.

Do not enter M1D-C until the required M1D-B gates are accepted or explicitly re-scoped by the user.
