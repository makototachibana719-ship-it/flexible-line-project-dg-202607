# SERL paper to local Aubo Peg mapping

## Paper recipe already implemented

| Paper element | Local implementation | Classification |
|---|---|---|
| SAC/RLPD off-policy actor-critic | `external/serl/.../sac.py` and `make_sac_agent()` | Reused unchanged |
| LayerNorm actor and critic | `external/serl/.../launcher.py::make_sac_agent` | Reused unchanged |
| Critic ensemble 10, subsample 2 | same factory | Reused unchanged |
| Adaptive SAC temperature | `SACAgent` | Reused unchanged |
| Prior/Demo + Online symmetric replay | isolated `run_baseline.py::concat_rlpd_batches`; 32+32, equivalent to current async entry | Isolated clean copy |
| Async actor/learner transport | Agentlace `TrainerServer`/`TrainerClient` | Reused unchanged |
| One learned policy | single `SACAgent` | Reused unchanged |
| State-defined reward is allowed for PCB assembly | current Aubo task reward/success | Reused unchanged |

## Necessary Aubo/state-only adapters

| Difference from paper | Local choice | Reason |
|---|---|---|
| Franka + two wrist cameras | Aubo state-only Obs20 | User-authorized legal adapter; no image/classifier path |
| 6D EE delta twist in current EE frame | normalized Aubo Action7; world translation, local-target rotation, gripper | Existing validated controller contract; no frame conversion |
| Random initial EE pose | fixed canonical Aubo reset | Existing environment is frozen; result only supports this reset distribution |
| 20 demonstrations | immutable V2 dataset: 11 episodes, 1921 transitions | Only validated Obs20/Action7 dataset; no new demo collection authorized |
| Generic environment/replay runtime | `portable_peg_runtime` and scoped ReplayBuffer insert repair | Engineering compatibility only; learner math unchanged |
| Paper emphasizes high UTD | UTD=1 in V1 | Explicit prior plan and only locally validated stable GPU path; UTD=8 is a later controlled comparison |

## Historical custom changes excluded

No import or execution of Franka simulation, C0/C1/C2/C3, exact-state curriculum, staged admission,
residual/gate/option heads, monotonic cursor, BC/DAgger supervision, custom critic
objectives, trust regions, HIL intervention, reward changes, success changes, or
formal-evaluation modifications.

## Interpretation boundary

This is a pure local RLPD baseline, not an exact reproduction of the paper's
camera observations, 20-demo set, random-reset distribution, relative action
frame, impedance controller, or high-UTD setting. Therefore a negative result
answers the scoped Aubo/Obs20/Action7/canonical-reset question only.
