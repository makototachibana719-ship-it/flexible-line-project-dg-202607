# Interface Compatibility Report

Static audit date: 2026-08-15. Compatibility is source-based and UNVERIFIED at runtime.

| Interface | Cable Routing | Object Grasping | Peg Insertion | Compatibility |
|---|---|---|---|---|
| Learning path | DrQ, visual | DrQ, visual | SAC/RLPD/BC, state-only | INCOMPATIBLE without per-skill adapters |
| Native observation keys | `state/{tcp_pos,tcp_vel,gripper_pos}`, `images/{front,wrist}` | same vision state keys and images; non-vision adds `block_pos` | raw task dict; adapter emits flat state | INCOMPATIBLE |
| Learner observation keys | intended `front,state,wrist` after wrapper | supplied demo: `front,state,wrist` | ndarray, no keys | INCOMPATIBLE |
| State dtype/shape | float32, intended `(1,7)` after chunking | float32, supplied demo `(1,7)` | float32 `(20,)` | INCOMPATIBLE |
| Image keys | `front`, `wrist` | `front`, `wrist` | MISSING | INCOMPATIBLE |
| Image dtype/shape | uint8 `(1,128,128,3)` intended | uint8 `(1,128,128,3)` supplied metadata | MISSING | Cable/Object structurally compatible; Peg differs |
| Action | float32 `(3,)` | float32 `(4,)` | float32 `(7,)` | INCOMPATIBLE |
| Reward | dense distance + success bonus | close/lift mixture | shaped error penalties/bonuses | INCOMPATIBLE semantics |
| `terminated` | task success | internal time limit | task termination/success through adapter | INCOMPATIBLE semantics |
| `truncated` | time limit | always false natively | task time limit through adapter | INCOMPATIBLE semantics |
| `mask` | `1 - terminated` | `1 - terminated` in actor loop | 0 only on termination | PRESENT common convention after adapter |
| `done` | terminated OR truncated | terminated OR truncated in actor loop | terminated OR truncated | PRESENT common convention |
| Transition schema | six standard fields | six standard fields | six standard fields | PRESENT structural commonality |
| Demo PKL | MISSING | list-of-transition metadata, UNVERIFIED | list-of-transition builder/metadata, UNVERIFIED | INCOMPATIBLE availability and observations |
| Replay input | memory-efficient image buffer | memory-efficient image buffer | standard state replay buffer | INCOMPATIBLE buffer type |
| Agent input | DrQ dict + image keys | DrQ dict + image keys | SAC flat state / BC state | INCOMPATIBLE |

## Current SERL mainline comparison

The current `external/serl` `SERLObsWrapper` statically expects `observation_space["image"]` and `obs["image"]`. Both visual packages expose nested `images` with two cameras. Therefore direct wrapper construction is INCOMPATIBLE before any environment step. Their training scripts imply a different multi-camera wrapper revision, but neither visual package delivers a matching modified wrapper file.

The current replay API can represent both dict visual and flat state observations, but separate buffer configuration is required: memory-efficient replay with explicit image keys for Cable/Object, regular replay for Peg. A single hard-coded DrQ input contract is not viable.

## Integration boundary

Only the high-level transition concept and Gymnasium five-value step can be normalized. Observation/action tensors, success semantics, model constructors, checkpoint loaders, and replay buffers must remain process-specific behind adapters. No component in this report is promoted to runtime VALIDATED.
