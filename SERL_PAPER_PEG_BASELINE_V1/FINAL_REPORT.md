# SERL paper-style Aubo Peg baseline - final report

`PURE_SERL_RLPD_AUBO_PEG_CONVERGENCE=NO`

This conclusion is scoped to `SERL_PAPER_PEG_BASELINE_V1`: one standard SERL
SAC policy, immutable V2 Demo replay, symmetric 32 Demo + 32 Online batches,
Obs20/Action7, fixed canonical reset, UTD=1, seed 0, no hierarchy, no custom
loss, and no intervention.

## Mapping outcome

Already present and reused unchanged: SERL SAC/RLPD, LayerNorm actor/critic,
10-critic ensemble with two-critic subsampling, adaptive temperature,
`update_high_utd`, Agentlace async transport, and replay buffers.

Necessary local adapters: Aubo state-only Obs20, normalized Action7, portable
MuJoCo asset runtime, and the scoped installed-ReplayBuffer insert repair.

Excluded from the new baseline: Franka simulation import, C0/C1/C2/C3,
curriculum/staged admission, residual/gate/option, monotonic cursor, BC/DAgger,
custom critic/actor objectives, trust regions, and HIL intervention.

## Evidence

| Checkpoint | Success | Mean return | Median min position error | Max force |
|---:|---:|---:|---:|---:|
| Fresh | 0/3 | -4993.000 | 0.059454 | 77.383 N |
| 1K | 0/3 | -6892.680 | 0.009442 | 86.603 N |
| 3K | 0/3 | -8044.162 | 0.105320 | 86.603 N |

The 1K policy showed real geometric approach improvement and therefore passed
the bounded promotion gate to 3K. The fresh 3K run then regressed below both
1K and fresh initialization on position error and return, with 0/3 success.
Accordingly, 5K and 10K were not scientifically justified and were not run.

Both completed training stages passed GPU execution, finite-loss checks,
32:32 sampling, real actor/learner communication, checkpoint/replay restore,
normal shutdown, and no-residual-process checks. The first 1K supervisor result
recorded a post-training verifier failure caused only by an old run-id string;
the existing checkpoint was subsequently restored successfully without
retraining or modification, and the correction is preserved in
`adjudication.json`.

## Interpretation boundary

This is not an exact reproduction of the paper: the legal local dataset has 11
V2 demonstrations rather than 20; reset is fixed rather than randomized; the
policy is state-only rather than camera-based; Action7 differs from the paper's
relative 6D twist; and V1 uses the previously agreed stable UTD=1 path rather
than high UTD. Therefore `NO` answers the scoped Aubo V1 question. A separately
named UTD=8 ablation would be required before making the stronger claim that
the exact high-UTD paper recipe cannot converge.

## Protection

All recorded upstream SAC/launcher, async entry, Aubo adapter, Replay adapter,
Demo and manifest hashes match before/after. The production Demo SHA256 remains
`4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878`.
No official/PASS source or artifact was modified.

