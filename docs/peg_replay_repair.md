# Peg Replay Repair (M1D-D1)

Date: 2026-08-23

## Root cause

`PEG_REPLAY_ROOT_CAUSE = SERL_PACKAGE_SOURCE_BUG`

The failing file is `/home/ubuntu/serl/serl_launcher/serl_launcher/data/data_store.py`, class `ReplayBufferDataStore(ReplayBuffer, DataStoreBase)`, method `insert`, line 49 in the inspected runtime. It calls:

```python
super(MemoryEfficientReplayBufferDataStore, self).insert(data)
```

`self` is a `ReplayBufferDataStore`, not an instance/subclass of `MemoryEfficientReplayBufferDataStore`, so Python rejects the `super(type, obj)` pair before any data write. The intended parent implementation is `ReplayBuffer.insert(self, data)`. This is neither a project call error nor a runtime-version API difference; it is a wrong sibling-class name in the installed SERL source.

## Project-side repair

`training/peg_insertion/peg_replay_adapter.py` provides a scoped context manager. It temporarily replaces only `ReplayBufferDataStore.insert`, calls the existing `ReplayBuffer.insert` parent explicitly under the existing datastore lock, preserves the optional RLDS logging transition logic, then restores the original method in `finally`.

This is not a second replay buffer. Capacity, allocated arrays, `_size`, `_insert_index`, sampling, RNG, locking, and schema remain the original SERL implementations. Peg package and SERL were not modified.

## Bounded validation

- 5,512/5,512 formal transitions inserted; buffer size and insert index both 5,512.
- 20 terminal transitions retained.
- All state/action/next-state/reward/mask values finite; no transition or field was rewritten.
- Demo sample 32 and bounded online sample 32 both produced the exact six-field schema.
- Concatenated batch is 64: observations `(64,20)`, actions `(64,7)`, next observations `(64,20)`, scalar arrays `(64,)` with original dtypes.
- Online buffer size/index: 64/64. Sampling changed the original NumPy RNG state.

`PEG_REPLAY_COMPATIBILITY = PASS`

