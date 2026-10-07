# SERL paper-fidelity Aubo Peg V2

Isolated Stage-B reconstruction using the clean local SERL SAC/RLPD source and
only branch-local Aubo adapters. It does not modify the official/PASS project,
the closed V1 baseline, or the UTD8 causal branch.

Targets: one policy, high UTD, LayerNorm, symmetric 32 demo + 32 online replay,
async actor/learner, exactly 20 successful V2 demonstrations, legitimate
training-only initial EE randomization, and the existing controller-compatible
relative incremental Action7. Formal evaluation retains the frozen canonical
reset and success contract.

No HIL, C0/C1/C2/C3, residual, gate, option, vision, custom critic penalty, or
trust region is imported.
