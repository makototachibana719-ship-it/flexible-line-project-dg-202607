# Parallel Training Timeline

| Phase | Operator A — Cable | Operator B — Object | Operator C — Peg | Exit rule |
|---|---|---|---|---|
| 1 | Transfer + WSL2/Conda/GPU | Transfer + WSL2/Conda/GPU | Transfer + WSL2/Conda/GPU | Own environment/JAX GPU PASS |
| 2 | Cable verifier + env smoke | Object verifier + env smoke | Peg verifier + env smoke | Own verify/smoke PASS |
| 3 | Bounded learner/actor smoke | Bounded learner/actor smoke | Bounded learner/actor smoke | Own W&B/checkpoint restore PASS |
| 4 | Formal training immediately | Formal training immediately | Formal training immediately | No cross-task wait |
| 5 | Independent evaluation | Independent evaluation | Independent evaluation | Raw and summary outputs complete |
| 6 | Return Cable artifacts | Return Object artifacts | Return Peg artifacts | Hashes/report complete |

The project lead accepts and analyzes each return as soon as it arrives; completion need not be synchronized.
