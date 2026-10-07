# Three-Process Comparison

| Area | Cable Routing | Object Grasping | Peg Insertion |
|---|---|---|---|
| Robot/task | Panda cable tip routing | Panda cube grasp/lift | Aubo i5 peg insertion |
| Observation mode | Two-camera vision + 7D state | Two-camera vision + 7D state | 20D state-only |
| Action | 3D xyz delta | 4D xyz + grasp | 7D xyz/RPY + gripper |
| Algorithm | DrQ | DrQ | SAC/RLPD and BC |
| Success | distance < 3 cm | lift ≥ 20 cm | position/orientation tolerances |
| Demo | MISSING | PRESENT, UNVERIFIED | PRESENT, UNVERIFIED |
| Checkpoint | PRESENT, UNVERIFIED; two empty groups | PRESENT, UNVERIFIED | PRESENT, UNVERIFIED |
| Formal evaluation | MISSING | MISSING | MISSING |

Common elements are Gym-style spaces, five-value step intent, SERL transition field names, JAX agents, Orbax-style checkpoints, and W&B hooks. The observation tensors, action semantics/dimensions, success/termination semantics, replay-buffer variants, agent classes, robot models, and dependencies must not be unified internally.
