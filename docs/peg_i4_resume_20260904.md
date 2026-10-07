# Peg I4 — Fresh checkpoint resume lifecycle

Date: 2026-09-04. One experiment; no retry, formal run, dependency or algorithm changes.

## Result

```
PEG_I4_RESUME_LIFECYCLE_5=PASS_WITH_WANDB_HISTORY_WARNING
CHECKPOINT_RESUME_AND_COUNTER_INVARIANT=PASS
WANDB_NEW_STEPS_20_24=PASS
WANDB_HISTORY_UNIQUENESS=FAIL
PEG_FORMAL_TRAINING=BLOCKED
```

The original strict verifier result remains FAIL in `i4_result.json` and
`verification.json`; it required exactly 25 history rows. Offline adjudication
in `i4_cached_history_audit.json` passes the requested resume lifecycle with an
explicit history warning, not an unconditional full-integration/formal PASS.

## Source and isolation

Only source: `/home/makoto/flexible_line_project/results/diagnostics/peg_i3_checkpoint_20260904_132008/checkpoints/checkpoint_19`.
The three source hashes were checked before and after and remain unchanged:

- model: `aacc7116b24fc162b57701bc9f1c7b8403d2042647084ea1d0183e481c7b0579`
- replay: `73bb5d9f915b92f146d1d712a5a6817faf927455b0c4815d4058a9f9f9f52ace`
- metadata: `cbe97c02b793b41894cd936228b039cc4f153936632587dd045115fc4f3f99e6`

Output: `/home/makoto/flexible_line_project/results/diagnostics/peg_i4_resume_20260904_133856`.
New isolated scripts only; no production launcher, resume implementation, or save cadence changed.
Legacy failed I3 artifacts were not loaded. No fixed/diagnostic RNG or split update.

## Restore and execution evidence

- Fresh learner PID 410; actual actor PID 661; both exit 0.
- JAX/JAXlib 0.4.35, GPU, Triton GEMM disabled.
- All 198 serialized state leaves match source exactly, including optimizer state and learner RNG.
- Complete replay content/size/index/counter/RNG state equality checked **before** live communication and sampling.
- Restored replay size/index 1292; cumulative accepted actor steps 1171.
- Learner RNG starts `[279913639, 4177495616]`, progresses normally through all five updates, and ends `[3228677402, 4064400818]`.
- Original `SACAgent.update_high_utd`, UTD=1, dynamic 32 online + 32 Demo, Demo 5512 unchanged.
- Live actor generated 947 local steps; learner accepted 931 additional transitions; actor received 6 parameter publications. Live uploads overlapped updates.
- Actor begins from restored source policy and fresh environment/action-sampling RNG. The bundle has no actor environment state or actor action RNG: **not bitwise trajectory-exact resume**. `actor_steps` denotes cumulative transitions accepted by learner, not actor's unacknowledged local steps.

| Global update | Dispatch seconds | Full sync seconds | Total update seconds |
|---|---:|---:|---:|
| 20 | 5.578723 | 0.024890 | 5.603618 |
| 21 | 3.923042 | 0.021273 | 3.944320 |
| 22 | 0.003348 | 0.012556 | 0.015908 |
| 23 | 0.002976 | 0.012095 | 0.015075 |
| 24 | 0.003885 | 0.019914 | 0.023804 |

All critic/actor/temperature losses finite. Two `Compiling update_high_utd` entries (initial and next-call compilation). No timeout, process restart, retry, or remaining learner/actor process observed.

## New checkpoint

`/home/makoto/flexible_line_project/results/diagnostics/peg_i4_resume_20260904_133856/checkpoints/checkpoint_24`

Saved once after all five completed updates, W&B logs and parameter publications. Terminal I4 save explicitly requested by user; not diagnostic early save and not a production cadence change. Save time 0.220255 s. Saved full model/optimizer/RNG equals final in-memory state.

Snapshot audit:

```
replay size = insert_index = 2223
initial non-live replay = 121
cumulative actor_steps = 2102
2223 - 121 = 2102 = metadata.actor_steps = replay.actor_steps
1171 inherited + 931 newly accepted = 2102
```

Original 1292 replay entries unchanged; replay RNG advanced; model finite198 leaves; bundle metadata/config/demo/run/replay hash validation passed.

New SHA256:

- model: `7e687783861b64740e1dd6d63b2587d19815981393c5982bc24c5090e8abb628`
- replay: `e3e98b501fc66148a12b99dcbbf412b662bf281e66bf80b68f7ca83b66b278dc`
- metadata: `df251ea12478c380c8a81f98a05db5573cf4010d7cf7a3c0efc297196ea1bfa6`

The new bundle was decoded and compared, not used for a second resumed training run.

## W&B warning and offline adjudication

Same run `44yrxufd`, `resume=must`, SDK reported `resumed=true`; final cloud state finished.
Run: https://wandb.ai/makototachibana719-beihang-university/flexible-line-peg-insertion/runs/44yrxufd

The one cloud history query returned **45 rows / 25 unique steps**. Historical steps0–19 each appeared twice with identical values matching the cached pre-resume I3 history. New steps20–24 each appeared exactly once; all three losses match the five local log calls. No conflicting records or missing steps were found. The reason for duplicate historical rows is **unresolved**; this report does not attribute it to training, SDK, or server without evidence.

`scripts/audit_peg_i4_cached_history.py` compared saved JSON only, with no training or network calls. It retained raw FAIL, rejected conflicting historical duplicates, required every new step unique, and produced the conditional result above. Do not rerun the training script to repair the history warning.

## Commands and remaining work

With locked Python and existing SERL/franka_sim PYTHONPATH:

```
python -m py_compile scripts/peg_i4_resume_validation.py
python -u scripts/peg_i4_resume_validation.py --role supervise
python scripts/audit_peg_i4_cached_history.py results/diagnostics/peg_i4_resume_20260904_133856
```

The first supervised command was executed exactly once. Raw verifier exit1 reflects the history row-count condition, not a learner/actor failure.

Next: read-only investigation of W&B duplicate historical rows and evidence-based full readiness review; do not automatically rerun I4, resume checkpoint24, or start formal. `runtime_gpu_formal.env` is absent. Formal remains BLOCKED. Tracking docs have `.before_i4_resume_20260904` backups; rollback concerns only scripts/docs, never deletion of experiment evidence or checkpoints.
