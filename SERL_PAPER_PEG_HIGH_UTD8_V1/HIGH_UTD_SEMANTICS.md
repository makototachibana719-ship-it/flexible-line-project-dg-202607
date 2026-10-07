# Local `update_high_utd` audit

Audited source:
`external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py`,
`SACAgent.update_high_utd`.

- batch size: 64 (32 online + 32 immutable demo)
- UTD ratio: 8
- minibatch size: 8
- critic/target updates per learner iteration: 8
- actor updates per learner iteration: 1, over the full 64-transition batch
- temperature updates per learner iteration: 1, over the full batch
- `critic_actor_ratio`: 1 (unchanged)

Therefore checkpoint 250 corresponds to 2,000 critic/target optimizer steps
and 250 actor/temperature steps. Checkpoints 500, 750, 1000, 1500 and 2000
correspond to 4,000, 6,000, 8,000, 12,000 and 16,000 critic steps.

Changing `critic_actor_ratio` is neither required by the called upstream API
nor authorized by the causal design, so it is not changed.
