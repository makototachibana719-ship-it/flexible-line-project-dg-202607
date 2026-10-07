"""Minimal branch-local Aubo adapter: optional training-only reset variation."""
from __future__ import annotations

import numpy as np

from peg_env_adapter import PegStateSERLEnv


class PaperFidelityPegEnv(PegStateSERLEnv):
    """Keep Obs20/Action7 and add a small legitimate initial-EE distribution."""

    def __init__(self, *args, training_reset_randomization=False,
                 randomization_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_reset_randomization = bool(training_reset_randomization)
        self._reset_rng = np.random.default_rng(randomization_seed)
        self.last_reset_randomization = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if not self.training_reset_randomization:
            self.last_reset_randomization = {
                "enabled": False, "canonical": True,
                "translation_action": [0.0, 0.0, 0.0],
                "rotation_action": [0.0, 0.0, 0.0],
            }
            return obs, info

        # At the canonical start the tool is far from contact. Two bounded
        # incremental commands followed by settling produce a small physical EE
        # distribution while leaving the hole/goal and success tolerances fixed.
        translation = self._reset_rng.uniform(
            low=np.array([-0.20, -0.35, -0.20]),
            high=np.array([0.20, 0.35, 0.20]),
        ).astype(np.float32)
        rotation = self._reset_rng.uniform(
            low=np.array([-0.15, -0.15, -0.05]),
            high=np.array([0.15, 0.15, 0.05]),
        ).astype(np.float32)
        action = np.concatenate((translation, rotation, np.array([-1.0], np.float32)))
        terminated = truncated = False
        for _ in range(2):
            obs, _, terminated, truncated, info = super().step(action)
            if terminated or truncated:
                raise RuntimeError("training reset randomization ended the episode")
        settle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], np.float32)
        for _ in range(6):
            obs, _, terminated, truncated, info = super().step(settle)
            if terminated or truncated:
                raise RuntimeError("training reset settling ended the episode")
        self.last_reset_randomization = {
            "enabled": True, "canonical": False,
            "translation_action": translation.tolist(),
            "rotation_action": rotation.tolist(),
            "commands": 2, "settle_commands": 6,
        }
        info = dict(info)
        info["paper_fidelity_reset_randomization"] = self.last_reset_randomization
        return np.asarray(obs, np.float32), info


def make_paper_fidelity_env(**kwargs):
    return PaperFidelityPegEnv(**kwargs)
