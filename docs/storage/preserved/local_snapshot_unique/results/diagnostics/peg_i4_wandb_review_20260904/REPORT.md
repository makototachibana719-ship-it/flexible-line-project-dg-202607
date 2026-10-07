# Peg I4-W&B read-only investigation and formal readiness review

Date: 2026-09-04. Run: `44yrxufd`, `peg_rlpd_gpu_i3_snapshot_fix_seed0`.

## Judgment

```
PEG_I4_WANDB_DUPLICATE_SOURCE=A_QUERY_API_REPRESENTATION_ISSUE_SUPPORTED
PEG_I4_WANDB_EXACT_ROOT_CAUSE=UNRESOLVED
PEG_I4_ACTUAL_DUPLICATE_TRAINING_LOG_CALLS=NOT_OBSERVED
PEG_I4_WANDB_UNIQUE_STEPS_0_24=PASS
PEG_FORMAL_READINESS=BLOCKED
PEG_FORMAL_TRAINING=BLOCKED
```

The observable source range is downstream of the application's local history
records, in W&B history serving/SDK-core/query representation. Evidence supports
A, but does **not** isolate a particular SDK defect, cache, backend export/live-tail
merge, or ingestion mechanism. This is not an unconditional exact-root-cause claim.

## Evidence collected

Read the existing SDK0.28.0 source and decoded both `.wandb` event files through
an explicitly **rb** stream (SDK's ordinary `open_for_scan` uses r+b and was not
called). Event-file hashes unchanged after decoding. No learner/actor was started;
no wandb.init/log/finish/sync/upload/delete operation was used. Public API queries
read the same existing run only. Scripts and output were created only in this new
report directory; no existing project/status/launcher/runtime file was modified.

| Evidence | Rows | Steps | Duplicate/conflict finding |
|---|---:|---|---|
| I3 local application log calls | 20 | 0–19 | Each once |
| I3 raw SDK history records | 20 | 0–19 | Each once |
| I4 local application log calls | 5 | 20–24 | Each once |
| I4 raw SDK history records | 5 | 20–24 | Each once; no old history replay records |
| Previously cached I4 cloud query | 45 | 25 unique | Old0–19 duplicated identically |
| Investigation scan, cache disabled, page100 | 45 | 25 unique | Same identical old duplicates |
| Next scan, cache disabled, page7 | 25 | 0–24 | Unique |
| GraphQL sampledHistory, samples1000 | 25 | 0–24 | Unique, all losses match local |
| Fresh query process, reversed order: page7 | 25 | 0–24 | Unique |
| Same reversed-order process: page100 | 25 | 0–24 | Unique |

GraphQL is a sampled query, not by itself proof of storage uniqueness; here it
returned all25 expected steps and corroborates the complete scan and local records.
The page7 complete scan and GraphQL values for learner_update/critic/actor/temperature
loss match all25 local raw SDK records exactly. Run state remains `finished`.

Important: page size and elapsed time/query ordering are confounded. The later
page100 scan also returned25, so do not declare a deterministic page-size bug.
`use_cache=False` rules against a simple stale **Python caller** cache explanation
but does not rule out backend caches or wandb-core internal/export state.

SDK source inspected on WSL:

- `.../site-packages/wandb/apis/public/history.py`: `HistoryScan` calls core
  ScanRunHistoryInit/ScanRunHistory; `_load_next` converts each returned row once,
  then increments the step window. The Python wrapper does not explicitly duplicate
  old rows. Actual core/backend internals were not instrumented.
- `.../site-packages/wandb/apis/public/runs.py`: scan_history exposes use_cache;
  `_sampled_history` sends read-only GraphQL sampledHistory.
- I4 debug.log reports resumed, and its raw file contains history20–24 only.

Local file SHA256:

- I3 `run-44yrxufd.wandb`: `89d0c32068534b8b181d7d30145f4931ccff682e74439f33a78dbeaf8c11a36b`
- I4 `run-44yrxufd.wandb`: `08f761025816d6844aaa1065e98fff133c522b634f94a72a768afecb82a29074`

## A–E attribution

| Option | Judgment | Reason / limit |
|---|---|---|
| A Query/API representation | **Best-supported source range** | Read views vary45→25 without training or run mutation; local records unique and full values agree |
| B Prior lifecycle/logging duplication | Not supported by inspected sessions | One20-record I3 session and one5-record I4 session; transport/backend resubmission cannot be excluded solely from client records |
| C SDK resume semantics | Possible context, not proven cause | resumed=true is confirmed; no evidence that resume legitimately mandates duplicate history; no controlled new-run experiment permitted/performed |
| D Actual duplicate training log calls | Not observed; strongly contradicted for these two sessions | Unique application calls, completed updates and raw SDK history records all agree |
| E Unresolved | **Exact mechanism remains unresolved** | No server storage audit/core trace, no causal cache/export instrumentation |

No cloud repair is justified. Preserve the original45-row evidence and original
strict I4 FAIL, alongside the conditional lifecycle PASS and this later read-only
observation. Do not rewrite past results as though the warning never occurred.

## Formal readiness: actual source path and risks

WSL source hashes match Windows mirror. Formal entry:

`deliverables/gpu_training_handoff/peg_insertion/run_train_learner.sh:4`
→ `python -m training.peg_insertion.run_peg_training --profile formal --role learner --require-gpu`
→ original delivered `async_sac_state_sim.py` learner, project W&B Logger and
project `save_bundle` adapter. It does not call I3/I4 harnesses.

| Area | Evidence / current behavior | Readiness judgment |
|---|---|---|
| GPU, normal RNG, original update | R1/R2/I1/I2/I3/I4 bounded evidence passed with Triton disabled | Component evidence PASS, not long-run guarantee |
| Snapshot consistency | Production resume.py repair tested concurrently and I3/I4 counters agree | PASS within tested boundaries |
| Same-run W&B resume | I4 original5 new records, later unique0–24 cloud view, finished | Lifecycle supported; query robustness warning remains |
| Formal runtime reproducibility | Formal shell sets EGL/PYTHONPATH only; no candidate env sourcing, no explicit Triton/preallocation flags | **Blocker:** actual flags depend on inherited shell environment; cannot assume tested runtime |
| Diagnostic isolation | run_peg_training.py:47–51 rejects diagnostic parameters in formal/smoke; diagnostics guarded | Static PASS; do not import diagnostic RNG/split/one-update policies into formal |
| Gate enforcement | Formal shell/main do not require a verified candidate PASS artifact | **Blocker:** documented BLOCKED is not an enforced launch guard; don't execute it |
| Production entry parity | I3/I4 are separate harnesses; main launcher candidate remains historical FAIL | **Blocker:** integrations don't prove the exact intended production launch path |
| Cadence and limits | Formal max_steps10000, save1000, delivered default log_period10; I4 logs every update and terminal save24 | Test differences must be reviewed, not silently changed |
| Resume compatibility | Main launcher binds only configured smoke/formal/candidate run names at lines55–58; I3/I4 name is different | I4 checkpoint24 is not directly accepted by default formal launcher; don't rename metadata or force it |
| Dry-run evidence | Main returns at line103 before make_agent restoration at line152 | A dry-run label alone does not prove actual model/optimizer restore; I4 actual restore provides stronger but harness-specific evidence |
| Long-run / replay wrap | Tiny bounded tests and no million-entry wrap integration | Unverified; no formal stability guarantee |
| Policy quality | Finite losses and valid checkpoints, no independent formal evaluation | Unverified policy competence; not a demonstrated success rate |

`runtime_gpu_formal.env` is absent. No Peg learner/actor process observed.
No production source/config changes were made. Existing formal candidate FAIL and
formal BLOCKED remain unchanged.

Recommended next scope (not executed): separately authorized readiness hardening
and bounded acceptance of the exact intended launch path, with explicit approved
runtime, enforced gate, documented logging/save semantics and robust read-only
history validation. History validation may recognize exact duplicates with a
warning, but must reject conflicting values, missing steps, or duplicate new
application records. This is an audit policy, not permission to mutate cloud data.

## Reproduction and evidence

In this directory, `investigate.py` performs rb local decoding and three read-only
views; `query_order_control.py` performs the fresh-process reverse-order control.
Both use the installed locked Python/SDK without upgrading packages. Output files
are exclusive-create to preserve evidence. Do not run the I4 training harness for
this investigation.

Primary evidence: `local.json`, `I3_event_history.json`, `I4_event_history.json`,
`investigation.json`, `scan_uncached_100.json`, `scan_uncached_7.json`,
`graphql_sampled.json`, `reversed_order.json`.
