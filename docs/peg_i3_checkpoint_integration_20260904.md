# Peg I3 — Normal Checkpoint I/O Integration Test

Date: 2026-09-04. Final audited gate: **`PEG_I3_CHECKPOINT_IO_20=FAIL`**.

Continuous updates / live communication / W&B / file saving succeeded. Checkpoint live actor-counter consistency failed. No stall was reproduced. Formal remains BLOCKED; no resume was executed.

## Method

One fresh learner PID 364 and real actor PID 623; 20 continuous original SACAgent.update_high_utd updates, normal RNG, dynamic 32 online + 32 Demo sampling, Triton GEMM disabled, GPU, online W&B. One run, no restart or retry. Verifier PID 1017 ran after the learner and actor; it did not create or restore an agent.

New harness `scripts/peg_i3_checkpoint_integration.py` reuses unmodified I1/I2. Only periodic checkpoint bundle saving and its read-only validation were added. It calls the existing `training/peg_insertion/resume.py:save_bundle`, which uses the existing legacy Flax `save_fresh_state`, then replay and metadata serialization. No existing training source or formal launcher changed.

The existing 20-update candidate cadence is period 19, giving ordinary saves at step 0 and 19. Saving occurs after completed update synchronization, W&B log and parameter publication, never inside the update/block wrapper. This preserves I2's log-then-publication order; the delivered learner orders publication then logging. Both precede saving. Therefore this is the I2 integration ladder with the normal periodic save function/position, not a byte-for-byte execution of the formal launcher. No diagnostic early-save branch was used.

Initial online replay retains the R2/I1 snapshot of 121 transitions; 5,512 Demo transitions unchanged. Live insertion tracking starts at zero for this fresh actor. Source metadata explicitly defines actor_steps as new accepted live transitions in this I3 run. No ring-buffer wrap occurred (capacity 1,000,000).

## Runtime results

| Item | Result |
|---|---|
| Learner | 20/20, single PID, exit 0 |
| Actor | 1,228 environment steps, exit 0 |
| Accepted live transitions | 1,201; remaining 27 not flushed after stop |
| Uploads overlapping learner updates | 35 |
| Parameter callbacks | 21 (initial + 20 updates) |
| Update 0 | 5.478903 s, initial compile |
| Update 1 | 4.608983 s, one recompile / WARNING |
| Updates 2–19 | 0.011746–0.032156 s; average 0.019847 s; NORMAL |
| Compile count | Initial 1, post-initial 1; none after index 1 |
| checkpoint_0 save | 0.198993 s |
| checkpoint_19 save | 0.189496 s |
| Losses / learner RNG | All finite, uninterrupted normal progression |
| Timeout / residual | None; learner/actor/verifier all exit 0; no trainer/wandb-core residual |

W&B run: [b6ajsw8v](https://wandb.ai/makototachibana719-beihang-university/flexible-line-peg-insertion/runs/b6ajsw8v), name `peg_rlpd_gpu_i3_checkpoint_seed0`, state `finished`. Cloud steps 0–19 and all loss triples match local data. Initialization 3.385258 s; mean log enqueue 0.000477 s; finish 2.419565 s. No checkpoint was uploaded as a W&B artifact.

## Checkpoint results and failed invariant

Both bundles contain model, online replay and metadata. Model size is 26,189,458 bytes each. All 198 decoded numeric leaves are finite; saved RNG matches the corresponding completed update. Step 0 RNG `[595175366,1025945940]`; step 19 RNG `[279913639,4177495616]`.

The initial verifier checked sidecar/metadata agreement, file hashes, model finite values, step, config, Demo and run identity. It emitted PASS. A subsequent independent invariant audit found that two agreeing counters can both be stale:

| Bundle | Replay size | Initial snapshot | Actual live entries | Recorded actor_steps | Undercount |
|---|---:|---:|---:|---:|---:|
| checkpoint_0 | 842 | 121 | 721 | 691 | 30 |
| checkpoint_19 | 1322 | 121 | 1201 | 1141 | 60 |

Thus the raw PASS in `i3_result.json` and `checkpoint_verification.json` is **superseded**, not erased. `i3_counter_audit.json` records the final FAIL. This is a metadata/counter failure, not a non-finite model or update timeout. It does not establish model quality or full replay snapshot atomicity.

### Source and event evidence

`training/peg_insertion/resume.py:52` performs:

```python
cp=save_fresh_state(directory,state,step)
replay.project_actor_steps=int(actor_steps)
ri=snapshot_replay(replay,rp)
```

The passed actor_steps value is captured before model serialization. Live actor inserts continue during that serialization, but the second statement resets their advanced counter to the old captured value. At step 0, `communication.jsonl` records a 30-entry upload to replay size 842 between save_start and save_complete. At step 19 another 30-entry upload to size 1322 occurs inside the save interval. The observed 30 then 60 undercount is consistent with the repeated stale-counter overwrite. The current save path also lacks a single coherent snapshot boundary across replay contents and counters.

This issue exists in the reused normal bundle function, not in a diagnostic early-save path. No repair was attempted in this test, and no resume was run against either bundle. These bundles must not be treated as validated resume artifacts.

## Paths and hashes

Output: `/home/makoto/flexible_line_project/results/diagnostics/peg_i3_checkpoint_20260904_121013/`.

Bundles under its `checkpoints/` directory:

- `checkpoint_0` SHA256 `9fab84d7da6568105336aacf1ead5208e9e3a66b83d6c4130a0c47f9897daf21`
- `checkpoint_19` SHA256 `2d32de62178763eae536c29a3937e3f89c0eb30ed0da9277a2efacad1896967f`

Full sidecar hashes/sizes are in `checkpoint_verification.json`. Original candidate checkpoint_0 hashes remain unchanged. Existing save function hashes remain unchanged. No formal runtime file was created.

Windows mirror: `D:/桌面/小挑/flexible_line_project/flexible_line_project/results/diagnostics/peg_i3_checkpoint_20260904_121013/`.

Evidence includes updates, communication, checkpoint events, W&B cloud verification, CPU thread/GPU logs and the counter audit. No prior smoke/candidate evidence was overwritten.

## Commands and boundaries

Executed once with the locked GPU environment and existing SERL/franka_sim PYTHONPATH:

```bash
python scripts/peg_i3_checkpoint_integration.py --role supervise
python scripts/verify_peg_i3_counter_consistency.py results/diagnostics/peg_i3_checkpoint_20260904_121013
```

The second command only reads saved metadata and writes a new audit JSON, without importing JAX, rebuilding an agent or calling resume. It returned exit 1 / FAIL. File decoding in the first verifier likewise did not inject state into any agent.

## Next work, not executed

Separately authorize a minimal normal bundle snapshot/counter consistency repair: do not overwrite an advanced live counter from a stale save-call argument, and define a coherent snapshot boundary for replay data, replay RNG, size/index and actor count. Add a concurrent-insert regression test and rerun the bounded I3 only after explicit authorization. Do not infer that this race caused the historic intermittent stall. Do not start formal or resume from these I3 artifacts.

Tracking docs were backed up with `.before_peg_i3_20260904` before updating. New harness/report/auditor only; no dependencies or algorithm settings changed.
