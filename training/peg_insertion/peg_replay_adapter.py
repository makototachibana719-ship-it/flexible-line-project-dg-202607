"""Scoped repair for SERL's state ReplayBufferDataStore insert typo."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def repaired_replay_datastore_insert() -> Iterator[None]:
    """Temporarily route inserts to the datastore's real ReplayBuffer parent.

    This preserves the instantiated SERL datastore, its lock, capacity, storage,
    RNG, indices, sampling implementation, and optional RLDS logging semantics.
    """
    from serl_launcher.data import data_store
    from serl_launcher.data.replay_buffer import ReplayBuffer

    cls = data_store.ReplayBufferDataStore
    original_insert = cls.insert

    def insert(self, transition):
        with self._lock:
            ReplayBuffer.insert(self, transition)
            if self._logger:
                if self.step_type in {
                    data_store.RLDSStepType.TERMINATION,
                    data_store.RLDSStepType.TRUNCATION,
                }:
                    self.step_type = data_store.RLDSStepType.RESTART
                elif not transition["masks"]:
                    self.step_type = data_store.RLDSStepType.TERMINATION
                elif transition["dones"]:
                    self.step_type = data_store.RLDSStepType.TRUNCATION
                else:
                    self.step_type = data_store.RLDSStepType.TRANSITION
                self._logger(
                    action=transition["actions"],
                    obs=transition["next_observations"],
                    reward=transition["rewards"],
                    step_type=self.step_type,
                )

    cls.insert = insert
    try:
        yield
    finally:
        cls.insert = original_insert

