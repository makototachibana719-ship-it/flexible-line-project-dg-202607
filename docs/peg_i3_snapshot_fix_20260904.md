# Peg I3-Fix — snapshot/counter consistency repair

Date: 2026-09-04. **Repair regressions PASS (5/5); authorized post-fix bounded I3 PASS.**

`PEG_I3_CHECKPOINT_IO_20=PASS` for new run `peg_i3_checkpoint_20260904_132008` only. Formal training remains BLOCKED. No resume executed. Old I3 FAIL and its artifacts remain unchanged and must not be used as validated resume artifacts.

## Minimal repair and snapshot boundary

Only production source changed: `training/peg_insertion/resume.py`.

1. `install_actor_step_tracking` is called before exposing the buffer to producer/consumer threads. It installs an RLock shared with SERL's existing insert/sample methods. The tracked insert holds that lock over both the original insert and counter increment. Reentrancy avoids deadlock when the original insert acquires the same lock internally. Installation on an already-active buffer is not supported.
2. `capture_replay_snapshot` holds that same lock while copying contents, size, insert index, capacity, actor count, and a deep copy of replay RNG state. This is the declared snapshot boundary, before model serialization. The learner's immutable train state remains fixed at the completed update while save_bundle runs.
3. Model serialization and replay file writing take place after releasing the lock. Live actor inserts and ordinary sampling can continue. Detached arrays and RNG state prevent subsequent changes from changing the captured snapshot.
4. The stale caller actor_steps argument is retained for API compatibility but never written back to the live replay. Bundle metadata gets its count, size, capacity and replay hash from the detached snapshot, not a later live read.

No algorithm, network, optimizer, LR, batch, Demo, reward, success, learner RNG, dependencies/JAX runtime, checkpoint cadence or formal launcher changes. Existing schemas remain compatible; no claim of exact full-training deterministic resume is made (e.g. actor environment state or replay iterator prefetch lifecycle is outside this repair).

Backup: `training/peg_insertion/resume.py.before_i3_snapshot_fix_20260904`.

Old SHA256: `50987ca2f9f68c5759fdd58f3ebf0e816230f30a552dfbdc940fc4c0be91fa38`.
New SHA256: `23f589189efcddd3242a6e3cbb21bcb27294dc46747bc6e8eb941d34fec881d2`.

## Regression tests — completed before GPU run

New `scripts/test_peg_snapshot_consistency.py` uses the actual SERL ReplayBufferDataStore and existing insert adapter, on CPU. Model I/O is stubbed only for the deterministic concurrent-insert unit case; the subsequent GPU run exercises real model I/O.

All five tests passed in 0.214 seconds (excluding imports):

- Insertion and counter increment are atomic: block after original insertion and assert a snapshot cannot escape before counter completion.
- Detached arrays/RNG and wrapped replay index: later insert/sample does not alter the captured payload; capacity 4 with 6 inserts retains size 4, index 2 and actor count 6.
- Sampling and snapshot use the same lock: captured RNG equals the post-sample state, without deadlock.
- Inserts during model serialization: a producer inserts 30 entries during each of two saves; it is not blocked by disk/model I/O, live counts never rewind, and metadata/replay reflect the earlier boundary even when the passed actor_steps is stale.
- Duplicate replay file fails without mutating live counter.

Command: locked Python with `JAX_PLATFORMS=cpu`, existing SERL PYTHONPATH, `python scripts/test_peg_snapshot_consistency.py`. No W&B, GPU updates or dependency changes during regression tests.

## Single authorized post-fix I3

New `scripts/peg_i3_fixed_acceptance.py` reuses the unchanged I3 supervisor/learner/actor. Only the run name and stricter verifier dispatch differ. Period remains 19 (saves at steps 0 and 19). Verifier failure now fails the subprocess and prevents the top-level gate from returning PASS. Original I3/R1/R2/I1/I2 scripts remain unmodified.

Executed once after all regressions passed:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/makoto/flexible_line_project/external/serl/serl_launcher:/home/makoto/flexible_line_project/external/serl/franka_sim \
  /home/makoto/miniconda3/envs/serl_peg_gpu/bin/python -u \
  /home/makoto/flexible_line_project/scripts/peg_i3_fixed_acceptance.py --role supervise
```

Fresh initialization, original update_high_utd, normal RNG, dynamic 32 online + 32 Demo, live communication, online W&B, Triton GEMM disabled. No resume, diagnostic early save, process restart or automatic retry.

| Check | Result |
|---|---|
| Learner | 20/20 continuous updates, one PID 367, exit 0 |
| Actor | PID 624, 1,228 steps, exit 0 |
| Live entries received | 1,201; 35 uploads overlap updates |
| Parameter callbacks | 21 |
| Update 0/1 | 5.523500 / 4.533597 seconds |
| Updates 2–19 | 0.011737–0.030522 seconds, mean 0.020502 |
| Compile count | One initial plus one index-1 recompile, none later |
| Save 0/19 | 0.211825 / 0.159239 seconds |
| Losses / normal RNG | All finite, uninterrupted progression |
| W&B | 20 cloud steps/losses match local; finished |
| Checkpoint decode | 198 finite leaves each, RNG matches corresponding update |
| Residual / timeout | None |

[W&B run 44yrxufd](https://wandb.ai/makototachibana719-beihang-university/flexible-line-peg-insertion/runs/44yrxufd), name `peg_rlpd_gpu_i3_snapshot_fix_seed0`.

### Counter invariants now pass

| Bundle | Replay snapshot size | Initial entries | Live actor_steps | Undercount |
|---|---:|---:|---:|---:|
| checkpoint_0 | 812 | 121 | 691 | 0 |
| checkpoint_19 | 1292 | 121 | 1171 | 0 |

Final live replay received 1,201 transitions, whereas checkpoint_19 captures 1,171. This is expected: the last 30 arrived after its snapshot boundary and are intentionally excluded from both saved contents and saved counter. Unlike old I3, later inserts do not rewind the live counter.

Model SHA256:

- checkpoint_0: `3ebb140f37ff7249c1d73716e13b36dc0258bdd460469bbb2b25e66a62bd40d1`
- checkpoint_19: `aacc7116b24fc162b57701bc9f1c7b8403d2042647084ea1d0183e481c7b0579`

Full bundle hashes are in `checkpoint_verification.json`; the stricter counter results are in `i3_counter_audit.json`.

## Evidence, preservation and limits

WSL: `/home/makoto/flexible_line_project/results/diagnostics/peg_i3_checkpoint_20260904_132008/`.

Windows mirror: `D:/桌面/小挑/flexible_line_project/flexible_line_project/results/diagnostics/peg_i3_checkpoint_20260904_132008/`.

Old failed I3 model hashes remain `9fab84d7...7daf21` and `2d32de62...96967f`; original failure audit is preserved. No old bundle was restored. No runtime_gpu_formal.env exists. This validates the tested snapshot fix and a short fresh integration run, not formal readiness, model quality, or actual resume lifecycle. It does not prove the historic intermittent stall was caused by this separate counter race.

Tracking docs were backed up with `.before_i3_snapshot_fix_20260904` before modification. Any rollback should preserve both failed and successful experiment evidence. Next possible task requires separate authorization: review/validate a fresh checkpoint's resume lifecycle, never use the old failed I3 artifacts, and do not launch formal automatically.
