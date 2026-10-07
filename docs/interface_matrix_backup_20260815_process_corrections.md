# Interface Matrix

This matrix is a compact companion to `interface_compatibility_report.md`. All runtime cells remain UNVERIFIED.

| Process | Env construction | Policy input | Policy output | Replay | Success channel | Adapter required |
|---|---|---|---|---|---|---|
| Cable Routing | registered vision env | `{front,state,wrist}` intended | float32 `(3,)` | memory-efficient visual | `info.success`, termination | multi-camera observation + common result |
| Object Grasping | registered vision env | `{front,state,wrist}` supplied metadata | float32 `(4,)` | memory-efficient visual | `info.success`, not native termination | multi-camera observation + termination normalization |
| Peg Insertion | direct factory via adapter | float32 `(20,)` | float32 `(7,)` | standard state replay | termination and `info.is_success` | registration/config + common result |

The common Skill API may normalize task invocation and results only; it must not expose a single observation/action schema.
