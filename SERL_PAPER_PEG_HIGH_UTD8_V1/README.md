# SERL Paper Peg High-UTD8 causal comparison

This directory is an isolated Stage-A branch copied from the closed
`SERL_PAPER_PEG_BASELINE_V1` implementation.  The only scientific change is
`utd_ratio: 1 -> 8`; all Aubo, Obs20, Action7, demo, reset, reward, success,
optimizer, architecture, LayerNorm, batch, replay, and seed conditions remain
bound to V1.

The learner pauses the actor at every formal checkpoint before saving.  A
separate controller evaluates three deterministic canonical episodes and then
writes an immutable continue/stop decision.  This makes the early-stop rule
effective during (rather than after) training.

## Resolved local high-UTD semantics

For a 64-transition 32-demo + 32-online batch, local SERL
`SACAgent.update_high_utd(..., utd_ratio=8)` reshapes the batch into eight
8-transition minibatches, performs eight critic/target optimizer updates, then
performs one actor and one temperature optimizer update on the full 64-item
batch.  `critic_actor_ratio` remains 1.

No official, PASS, historical, or V1 file is modified by this branch.
