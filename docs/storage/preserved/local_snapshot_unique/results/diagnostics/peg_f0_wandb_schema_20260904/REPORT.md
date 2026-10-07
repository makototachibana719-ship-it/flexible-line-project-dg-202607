# F0-W&B-Schema — read-only investigation

Run: hr88a6u1, 2026-09-04. No learner/actor, resume, cloud writes, runtime/sync changes or Gate changes.

```
PEG_F0_WANDB_SCHEMA_INVESTIGATION=COMPLETE
PEG_F0_WANDB_SCHEMA=VERIFIER_FIELD_SCHEMA_MISMATCH_CONFIRMED
PEG_F0_WANDB_HISTORY_50=PASS
PEG_FORMAL_TRAINING=BLOCKED
```

## Evidence

Actual producer passes nested update_info directly to wandb.log: critic -> critic_loss, actor -> actor_loss, temperature -> temperature_loss. Logger does not convert separators. Local SDK history and cloud schema expose dotted fields: critic.critic_loss, actor.actor_loss, temperature.temperature_loss.

Verifier `scripts/verify_peg_f0.py` instead requests slash-separated fields and also transforms local comparison keys using replace('.', '/'). Both assumptions disagree with the observed schema.

| Read-only source/query | Rows | Steps | Loss agreement with local updates |
|---|---:|---|---|
| Local SDK .wandb history |50|0–49 unique|all three losses match|
| Cloud unfiltered scan |50|0–49 unique|all three losses match|
| Cloud dotted-key projection |50|0–49 unique|all three losses match|
| Cloud slash-key projection, current query |50|only _step returned|no requested losses present|
| Original cached slash-key query |0|empty|not available|

Run is finished. Checked finite losses and math.isclose(rel_tol=1e-7,abs_tol=1e-9) against all50 local records, across all three loss fields. No evidence of missing training history in the correct-field queries.

Important: wrong-key query does not consistently return an empty list. Current read-only slash projection returned50 step-only objects. Exact reason for this presentation difference from the original empty response remains unresolved; field mismatch itself is confirmed and neither response supplies the requested loss data.

## Raw test preservation

`projection_audit.json` preserves status FAIL because its exploratory hypothesis `slash_empty` was false. Its other seven checks (three unique-complete, three loss-match, plus finished) pass. This is not evidence of lost cloud history. The final investigation conclusion uses the observed step-only slash records, rather than silently overwriting that raw hypothesis check. Original B audit and formal_readiness.json are untouched.

## Suggested repair, not implemented

Use the observed dotted fields for projection and retain the dotted local loss keys; validate actual field presence, finite values, complete unique steps and local/cloud correspondence. Do not accept step-only rows, substitute zeros, or remove checks merely to obtain PASS. Separately authorize verifier changes/tests and an audit of the existing B artifacts; no training rerun is required to address this schema mismatch. This investigation does not promote formal readiness or resolve native synchronization root cause.

## Files

Only this new investigation directory was added: two read-only query scripts, raw_evidence.json, projection_audit.json and this report. No existing project source, checkpoint, historical results, state documents, Gate or cloud history was modified.
