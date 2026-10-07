# HIL-SERL Automated Expert Intervention V1

Isolated causal branch. It imports the frozen Stage-B environment/training utilities read-only and changes only the training-time action selection and dual routing of intervention transitions. Formal evaluation constructs the HIL adapter with `intervention_enabled=False` and `provider=None`.

The trigger uses the already validated normalized-Obs20 production-demo support threshold (`D_DEMO=7.41773975814`) with exact restored-state one-step policy lookahead. The expert is the already validated nearest-state production-demo Action7 provider. No threshold search or new expert is introduced.

