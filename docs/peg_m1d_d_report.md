# M1D-D Peg Demo + Algorithm Validation Report

Date: 2026-08-23  
Historical result before M1D-D1: `PEG_M1D_D_STATUS = PARTIAL`. This report is superseded for current strategy/status by `docs/peg_m1d_d1_report.md`, which closes the fresh-training gates without restoring the legacy checkpoint.

## Outcome

The package/archive is frozen; the 20 NPZ demos, 5,512-transition RLPD PKL, and 2,388-transition BC subset are structurally and numerically mapped. All 20 demos satisfy source-derived success semantics. Primary algorithm is RLPD-style SAC with a 50/50 online/Demo batch. BC and RLPD checkpoint groups are mapped.

The round cannot pass because the sole bounded RLPD restore timed out and the exact current `make_replay_buffer(..., type="replay_buffer")` path fails on first insert in `/home/ubuntu/serl/.../data_store.py`: `ReplayBufferDataStore.insert` calls `super(MemoryEfficientReplayBufferDataStore, self).insert(data)`, producing `TypeError`. BC stacking `(2388,20)/(2388,7)/(2388,20)` passes, but that does not substitute for the failed primary replay path.

## Source freeze hashes

| Source | SHA256 |
|---|---|
| RLPD entry | `514b08065c0b46c7652c6b65a9af9c63206d3380584b0d20b90a10058ec248c7` |
| SAC entry | `5b04f54fb4f5649a1ee8db560f5787882388caca34fda1a29280b1d76a343f83` |
| BC trainer | `3ed4f875a1b5a3cfbf7da25a4cd72fdb528edcc8509782abaea2bcefbefa74fa` |
| Peg env adapter | `da2ab725a464d59689c85a747974efbe6e48de9299249ed633d6a62659fe1573` |
| Peg task/reward | `a75071fe53f57eef205caf0b075aa13e41805eb621713cd7e759f569c7281bf7` |

## Gate matrix

| Gate | Result | Evidence |
|---|---|---|
| `PEG_ENVIRONMENT_GATE` | PASS | prior lifecycle plus source contract and 20 reset audit |
| `PEG_DEMO_SCHEMA_GATE` | PASS | 20 independent NPZ and both PKLs |
| `PEG_DEMO_SUCCESS_GATE` | PASS | 20/20 source-threshold + terminal-array validation |
| `PEG_REPLAY_GATE` | FAIL | official primary datastore first insert TypeError |
| `PEG_ALGORITHM_MAPPING_GATE` | PASS | RLPD/SAC/BC entries and loaders mapped |
| `PEG_CHECKPOINT_MAPPING_GATE` | PASS | two groups mapped by tree/entry/config/cadence |
| `PEG_CHECKPOINT_RESTORE_GATE` | FAIL | primary checkpoint_9000 timeout at 120 s |

## Exact next milestone

Remain in `M1D-D`; do not enter M1D-E. Exact next task is **M1D-D1 — project-side Peg replay compatibility repair and bounded checkpoint restore diagnosis**: reproduce/fix the datastore call only through a project-side compatibility wrapper (no SERL/package mutation), validate 5,512 inserts and a 64-sample 32/32 mixed batch, then isolate the checkpoint read stall. Only after both gates pass may M1D-D close and M1D-E be proposed.
