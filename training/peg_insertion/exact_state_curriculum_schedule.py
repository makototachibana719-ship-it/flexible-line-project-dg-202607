"""Deterministic TRAIN_ENV-only scheduling for exact-state curriculum resets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.peg_insertion.exact_state_curriculum import (
    LEVELS,
    TrainingOnlyExactStateCurriculum,
)


class DeterministicExactStateCurriculumSchedule:
    """A fixed cyclic snapshot schedule; never used by evaluation."""

    def __init__(self, canonical_env, snapshot_directory: str | Path, level: str):
        normalized = str(level).upper()
        if normalized not in LEVELS:
            raise ValueError(f"unknown curriculum level {level!r}")
        self.level = normalized
        self.snapshot_directory = Path(snapshot_directory)
        self.schedule = self._build_schedule()
        self.schedule_sha256 = hashlib.sha256(
            json.dumps(self.schedule, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.reset_index = 0
        self.env = TrainingOnlyExactStateCurriculum(
            canonical_env, self.snapshot_directory, self.level
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata

    def _build_schedule(self) -> list[dict]:
        if self.level == "C3":
            return [{"level": "C3", "source": "ORIGINAL_CANONICAL_RESET"}]
        stem = self.level.lower() + "_exact_state"
        manifest = self.snapshot_directory / f"{stem}_manifest.json"
        snapshot = self.snapshot_directory / f"{stem}.npz"
        if not manifest.is_file() or not snapshot.is_file():
            raise FileNotFoundError(f"missing exact-state pair for {self.level}")
        return [{"level": self.level, "manifest": manifest.name, "snapshot": snapshot.name}]

    def reset(self, *, seed=None, options=None):
        entry = self.schedule[self.reset_index % len(self.schedule)]
        self.reset_index += 1
        observation, info = self.env.reset(seed=seed, options=options)
        result = dict(info)
        result.update(
            curriculum_schedule_index=self.reset_index - 1,
            curriculum_schedule_entry=entry,
            curriculum_schedule_sha256=self.schedule_sha256,
        )
        return observation, result

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

