# Peg F3 V2 — HIL-SERL Automated Expert Intervention V1

## Outcome

The bounded causal pilot completed 500 learner updates on GPU. It did not meet Strong PASS or Partial PASS. All policy-only canonical evaluations remained 0-success and contact health failed. Checkpoint 250 entered the near-hole region in all 3 primary episodes and materially improved median minimum position error to `0.014854864563324775`, but the intervention rate stayed near 100% and did not decrease.

`HIL_SERL_AUBO_PEG_CONVERGED=NO`

## Frozen recipe and sole causal change

The run retained one standard SERL SAC policy, Obs20/Action7, Aubo relative incremental actions, Stage-B reset randomization, reward/success/controller/IK/geometry/collision behavior, critic ensemble and LayerNorm, UTD8, and symmetric 32 Demo + 32 Online batches. The immutable 20-demo Stage-B prior remained the starting demo replay (`d4c625b243420460a48cad68fa6d75cd987f306099f0d4ad7c18cd1404135bc3`).

The only scientific change was training-time intervention. The approved adapter selected either policy Action7 or expert Action7, called the control environment once, stored the executed action in online replay, and dual-routed intervention transitions into the augmented demo replay.

The provider reused the validated state-feedback binding: nearest normalized-Obs20 action from the immutable accepted production Demo (`4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878`). The trigger used an independent exact-state MuJoCo restoration and one-step policy lookahead against the pre-registered `D_DEMO=7.41773975814` support boundary. No threshold sweep or new expert was introduced. Active intervention released immediately when the current state re-entered that validated support.

## Checkpoint evidence

| Checkpoint | Success | Near-hole | Median min position | Median min lateral | Median min axial | Median min orientation | Mean return | Contact | Max force |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 100 | 0/3 | 0/3 | 0.1631238717 | 0.1428323706 | 0.0001418569 | 0.0327616533 | -12129.8955 | FAIL | 86.6025404 |
| 250 | 0/3 | 3/3 | 0.0148548646 | 0.0079768417 | 0.0000460018 | 0.0128532633 | -10730.4540 | FAIL | 86.6025404 |
| 500 | 0/3 | 0/3 | 0.0459639725 | 0.0281342856 | 0.0004326297 | 0.0201498623 | -7026.1745 | FAIL | 86.6025404 |

The best checkpoint was 250. Its independent 5-episode canonical evaluation was `0/5`; all five episodes reached near-hole but all had unexpected contact. Intervention was disabled, the provider was `None`, and expert call count was zero in every formal evaluation.

## Intervention and replay audit

- Environment steps / online replay: `3648 / 3648`.
- Intervention steps / intervention-demo additions: `3526 / 3526`.
- Intervention fraction: `0.9665570175438597`.
- Segment rates at checkpoints 100/250/500: `0.9456919060`, `0.9824561404`, `0.9942802669` (classified `STABLE` by the pre-implemented slope tolerance; critically, not decreasing).
- Mean / median intervention duration: `66.5283018868 / 2.0` steps.
- Completed episodes with intervention: `100%`.
- Recovery restoration rate: `0.8490566038`.

The intervention data path was active and influential in sampling: the intervention fraction of the augmented demo store grew from `0.18695` to `0.49237`; over 500 updates, the expected intervention content of each 32-row demo half averaged `12.79` rows (range `5.98–15.76`). The maximum probability of a demo half containing zero intervention rows was about `0.00133`. All 500 actor/critic/temperature updates had finite losses, and the actor received 501 networks. Thus this is not a missing-buffer or missing-update failure.

## Adjudication

Strong PASS failed because no checkpoint reached 2/3 success and contact health never passed. Partial PASS failed because, although checkpoint 250 had near-hole > 0 and position error materially improved over Stage B, intervention rate did not decrease.

The pre-registered near-100% rule applies:

`PRIMARY_ROOT_CAUSE=POLICY_NOT_LEARNING_FROM_INTERVENTION`

No matched confirmation or 1K extension was authorized. The next bounded task is an intervention-retention review; pure RLPD tuning remains closed.

## Required fields

```text
HIL_SERL_V1_PREFLIGHT=PASS
HIL_SERL_V1_TRAINING=PASS
TOTAL_ENV_STEPS=3648
TOTAL_INTERVENTIONS=3526
INTERVENTION_FRACTION=0.9665570175438597
MEAN_INTERVENTION_DURATION=66.52830188679245
RECOVERY_RESTORATION_RATE=0.8490566037735849
CHECKPOINT_100_SUCCESS=0/3
CHECKPOINT_250_SUCCESS=0/3
CHECKPOINT_500_SUCCESS=0/3
BEST_CHECKPOINT=250
BEST_SUCCESS=0/3
BEST_NEAR_HOLE=3
BEST_MIN_POSITION_ERROR=0.014854864563324775
BEST_CONTACT_HEALTH=FAIL
INTERVENTION_RATE_TREND=STABLE
HIL_SERL_V1_RESULT=FAIL
MATCHED_CONFIRMATION=NOT_RUN
FINAL_CANONICAL_SUCCESS=0/5
HIL_SERL_AUBO_PEG_CONVERGED=NO
PURE_SERL_RLPD_ARTIFACTS_CHANGED=NO
PRODUCTION_DEMO_SHA_UNCHANGED=YES
INTERVENTION_ENABLED_IN_FORMAL_EVAL=NO
RESIDUAL_PROCESSES=NONE
NEXT=INTERVENTION_RETENTION_REVIEW
```

