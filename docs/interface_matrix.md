# Interface Matrix

This matrix is a compact companion to `interface_compatibility_report.md`. All runtime cells remain UNVERIFIED.

| Process | Algorithm mapping | Env construction | Policy input | Policy output | Replay/demo | Success channel | Adapter required |
|---|---|---|---|---|---|---|---|
| Cable Routing | DrQ | registered vision env | `{front,state,wrist}` intended | float32 `(3,)` | memory-efficient visual; Demo Data MISSING; acquisition TODO | `info.success`, termination | multi-camera observation + common result |
| Object Grasping | DPRL human-confirmed; delivered DrQ mapping NEEDS_VERIFICATION | registered vision env | `{front,state,wrist}` supplied metadata | float32 `(4,)` | memory-efficient online + optional demo; delivered config uses 50/50 mixing | `info.success`, not native termination | multi-camera observation + termination normalization |
| Peg Insertion | SAC/RLPD/BC | direct factory via adapter | float32 `(20,)` | float32 `(7,)` | standard state replay; RLPD uses demo mixing | termination and `info.is_success` | registration/config + common result |

The common Skill API may normalize task invocation and results only; it must not expose a single observation/action schema.
