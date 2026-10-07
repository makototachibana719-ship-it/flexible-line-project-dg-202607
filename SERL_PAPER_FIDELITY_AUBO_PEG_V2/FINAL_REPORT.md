# SERL Paper Peg — High-UTD causal replication and Aubo paper-fidelity rebuild

## Decision

`PURE_SERL_RLPD_AUBO_PEG_CONVERGENCE=NO`

This is a scoped negative result for Aubo state-only Obs20, the legal
controller-compatible relative incremental Action7, seed 0, and the tested
UTD8 schedules. It does not disprove SERL/RLPD or the paper's camera-based
Franka result.

## Required outputs

```text
UTD8_CAUSAL_COMPARISON=FAIL
BEST_UTD8_CHECKPOINT=250
BEST_UTD8_SUCCESS=0/3
BEST_UTD8_MIN_POSITION_ERROR=0.04935561794661174
HIGH_UTD_SUPPORTS_PAPER_HYPOTHESIS=NO
PAPER_FIDELITY_REBUILD_REQUIRED=YES
PAPER_FIDELITY_20_DEMO_SHA=d4c625b243420460a48cad68fa6d75cd987f306099f0d4ad7c18cd1404135bc3
PAPER_FIDELITY_RESET_RANDOMIZATION=YES
PAPER_FIDELITY_RELATIVE_ACTION=YES
PAPER_FIDELITY_CONVERGED=NO
FINAL_CANONICAL_SUCCESS=0/5
CONTACT_HEALTH=FAIL
PURE_SERL_RLPD_AUBO_PEG_CONVERGENCE=NO
NEXT=HIL_SERL_INTERVENTION
```

## Stage A — UTD8 causal comparison

The local upstream `update_high_utd` semantics were resolved before training:
one 64-transition learner iteration reshapes the symmetric 32-online + 32-demo
batch into eight 8-transition minibatches, performs eight critic/target
updates, then performs one actor and one temperature update on the full batch.
`critic_actor_ratio` remained 1.

All other V1 scientific conditions remained fixed, including demo SHA
`4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878`.

| checkpoint | success | near-hole | median min position (m) | mean return | max force (N) | saturation | contact |
|---:|---:|---:|---:|---:|---:|---:|---|
| 250 | 0/3 | 0/3 | 0.049356 | -10460.4 | 86.60 | 3.03% | FAIL |
| 500 | 0/3 | 0/3 | 0.070361 | -9506.0 | 86.60 | 4.13% | FAIL |
| 750 | 0/3 | 0/3 | 0.054311 | -8136.7 | 86.60 | 6.80% | FAIL |
| 1000 | 0/3 | 0/3 | 0.082306 | -9820.0 | 86.60 | 5.43% | FAIL |

At 1000, behavior was no longer improving, so 1500/2000 were not authorized.
The best UTD8 position error was more than five times the UTD1 best
(0.009442 m), with no near-hole episode or success. High UTD alone therefore
did not support the paper hypothesis in this Aubo adapter.

All four checkpoints restore with finite parameters. The raw run recorded a
post-result ZeroMQ interpreter teardown assertion, but learner results,
checkpoints, evaluations, actor explicit socket/context close, broadcast join,
environment close, and protected hashes were already complete and PASS. The
separate adjudication accepts the scientific result and preserves the raw
error evidence.

## Stage B — paper-fidelity Aubo rebuild

The rebuild uses the clean local SERL SAC/RLPD implementation with:

- one policy and standard SAC objectives;
- UTD8, critic ensemble 10/subsample 2, actor and critic LayerNorm;
- async actor/learner and exact 32-demo + 32-online sampling;
- exactly 20 successful V2 demos;
- training-only small physical initial-EE randomization;
- unchanged canonical formal evaluation, goal, reward and success tolerances;
- Aubo-compatible relative incremental Action7;
- no HIL, hierarchy, C0/C1/C2/C3, residual, gate, option, vision, trust region,
  critic penalty or custom policy objective.

The new physical demo set contains 3627 transitions and exactly 20 terminal
episodes: 5 standard, 4 +Y, 4 -Y, 3 roll and 4 pitch. The accepted 11 remain
bit-exact; nine new scripted physical rollouts cover varied approach states and
correction timing. Every new rollout passed success=true, truncated=false, V2,
contact health, legal Action7 and continuity checks.

| checkpoint | success | near-hole | median min position (m) | mean return | max force (N) | saturation | contact |
|---:|---:|---:|---:|---:|---:|---:|---|
| 250 | 0/3 | 0/3 | 0.051287 | -8468.6 | 86.60 | 4.67% | FAIL |
| 500 | 0/3 | 0/3 | 0.068079 | -9046.3 | 86.60 | 1.00% | FAIL |
| 750 | 0/3 | 0/3 | 0.126190 | -9107.4 | 76.89 | 1.87% | FAIL |
| 1000 | 0/3 | 0/3 | 0.062614 | -5688.8 | 86.60 | 0.03% | FAIL |
| 1500 | 0/3 | 0/3 | 0.064331 | -4562.4 | 80.51 | 0.00% | FAIL |
| 2000 | 0/3 | 0/3 | 0.113337 | -9883.3 | 86.60 | 0.10% | FAIL |

The 1000 checkpoint materially improved against 750, so conditional
1500/2000 training was justified. It still produced no near-hole or success
episode, and 2000 regressed. The final five canonical episodes used the best
geometric checkpoint (250): 0/5 success, 0/5 near-hole, median minimum position
0.051287 m, and contact health FAIL. Each trajectory reached expected hole
contact but also registered unexpected contact; no base/shoulder link01 contact
was seen.

All six Stage-B checkpoints restore with finite parameters. Learner completed
2000 iterations/16000 critic updates, actor received 2001 network versions,
explicit cleanup passed, protected artifacts were unchanged, and no branch
process remains.

## Why Action7 was retained

Action7 is already relative incremental control: translation updates the
current Aubo controller target in fixed task axes, while rotation is a local
right-multiplied target-frame delta. Obs20 explicitly contains goal-minus-TCP
position. Reinterpreting translation in a Franka-style body frame would make
the accepted 11 demonstrations semantically incompatible and alter the proven
Aubo contact controller. The retained representation is the closest compatible
paper principle, not a claim of bit-exact Franka control.

## Evidence locations

- Stage A result: `SERL_PAPER_PEG_HIGH_UTD8_V1/runs/stage_a_seed0_attempt2/STAGE_A_RESULT.json`
- Stage A restore/exit adjudication: `SERL_PAPER_PEG_HIGH_UTD8_V1/runs/stage_a_seed0_attempt2/STAGE_A_ADJUDICATION.json`
- 20-demo manifest: `PAPER_FIDELITY_20_DEMO_V1/manifest.json`
- Stage B preflight: `preflight_audit.json`
- Stage B result: `runs/stage_b_seed0/STAGE_B_RESULT.json`
- Stage B restore adjudication: `runs/stage_b_seed0/STAGE_B_ADJUDICATION.json`
- Final five episodes: `runs/stage_b_seed0/eval_250_final5.json`

The next scientifically distinct experiment is HIL-SERL intervention; it is
not part of this pure-policy reconstruction and was not started.
