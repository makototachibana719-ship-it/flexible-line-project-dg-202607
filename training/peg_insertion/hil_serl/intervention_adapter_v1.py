"""Training-only HIL-SERL intervention adapter for Peg F3 V2.

This module deliberately does not import an expert implementation or transform
frames. Providers must emit the project's normalized Aubo Action7 directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


ACTION_CONTRACT_VERSION = "PEG_F3_V2_AUBO_NORMALIZED_ACTION7_V1"
REPLAY_KEYS = frozenset(
    {
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "masks",
        "dones",
    }
)


@dataclass(frozen=True)
class InterventionDecision:
    """One provider decision made before the environment is stepped."""

    intervened: bool
    expert_action: Optional[np.ndarray] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExpertInterventionProviderProtocol(Protocol):
    """Protocol for a training-only provider that emits Aubo Action7."""

    def decide(
        self,
        *,
        obs20: np.ndarray,
        policy_action: np.ndarray,
    ) -> InterventionDecision:
        ...


def assert_legal_aubo_action7(
    action: Any,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Return a float32 Aubo Action7 or fail closed.

    No frame transform is performed. Values outside the range by no more than
    floating-point tolerance are clamped to the exact normalized boundary;
    grossly illegal values are never silently clipped.
    """

    raw = np.asarray(action)
    if raw.shape != (7,):
        raise ValueError(f"Aubo Action7 requires shape (7,), got {raw.shape}")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"Aubo Action7 requires a numeric dtype, got {raw.dtype}")
    try:
        normalized = raw.astype(np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Aubo Action7 is not convertible to float32") from exc
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Aubo Action7 contains NaN or Inf")
    max_abs = float(np.max(np.abs(normalized)))
    if max_abs > 1.0 + float(tolerance):
        raise ValueError(f"Aubo Action7 is outside [-1, 1]: max_abs={max_abs}")
    if max_abs > 1.0:
        normalized = np.clip(normalized, -1.0, 1.0)
    return normalized


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_value(value.item())
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported provenance type: {type(value).__name__}")


def transition_identity(transition: Mapping[str, Any]) -> str:
    if set(transition) != REPLAY_KEYS:
        raise ValueError("transition must have exactly the canonical six keys")
    payload = json.dumps(
        _json_value(dict(transition)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class HILSERLInterventionAdapterV1:
    """C3-authority-scoped, training-only intervention decision adapter."""

    def __init__(
        self,
        *,
        mode: str,
        intervention_enabled: bool,
        provider: Optional[ExpertInterventionProviderProtocol],
        provenance_path: Optional[Path | str] = None,
        checkpoint_identity: Optional[Mapping[str, Any]] = None,
        expert_identity: Optional[Mapping[str, Any]] = None,
    ) -> None:
        normalized_mode = str(mode).upper()
        if normalized_mode not in {"TRAINING", "FORMAL_EVAL"}:
            raise ValueError("mode must be TRAINING or FORMAL_EVAL")
        if normalized_mode == "FORMAL_EVAL":
            if intervention_enabled or provider is not None:
                raise ValueError(
                    "FORMAL_EVAL requires intervention_enabled=False and provider=None"
                )
        elif intervention_enabled and provider is None:
            raise ValueError("enabled TRAINING intervention requires a provider")

        self.mode = normalized_mode
        self._enabled = bool(intervention_enabled)
        self._provider = provider
        self.provenance_path = (
            Path(provenance_path) if provenance_path is not None else None
        )
        self.checkpoint_identity = dict(checkpoint_identity or {})
        self.expert_identity = dict(expert_identity or {})
        self.intervention_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        """Permanently disable this adapter instance."""

        self._enabled = False
        self._provider = None

    def decide(
        self,
        *,
        obs20: Any,
        policy_action: Any,
        c3_has_authority: bool,
    ) -> InterventionDecision:
        policy = assert_legal_aubo_action7(policy_action)
        observation = np.asarray(obs20, dtype=np.float32)
        if observation.shape != (20,) or not np.all(np.isfinite(observation)):
            raise ValueError("Peg F3 V2 intervention requires finite Obs20")
        if not bool(c3_has_authority):
            self.disable()
            return InterventionDecision(False)
        if not self._enabled:
            return InterventionDecision(False)
        if self.mode != "TRAINING" or self._provider is None:
            raise RuntimeError("intervention provider is unavailable")

        decision = self._provider.decide(obs20=observation, policy_action=policy)
        if not isinstance(decision, InterventionDecision):
            raise TypeError("provider must return InterventionDecision")
        if not isinstance(decision.intervened, (bool, np.bool_)):
            raise TypeError("decision.intervened must be boolean")
        if not bool(decision.intervened):
            if decision.expert_action is not None:
                raise ValueError("non-intervention decision must not carry expert_action")
            return InterventionDecision(False, provenance=dict(decision.provenance))
        expert = assert_legal_aubo_action7(decision.expert_action)
        return InterventionDecision(True, expert, dict(decision.provenance))

    def append_provenance_sidecar(
        self,
        *,
        episode: int,
        step: int,
        policy_action: Any,
        executed_action: Any,
        intervention_action: Any,
        observations: Any,
        next_observations: Any,
        reward: Any,
        terminated: bool,
        truncated: bool,
        transition: Mapping[str, Any],
        provider_provenance: Optional[Mapping[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Append one intervention-only JSON record; never alter replay data."""

        if self.provenance_path is None:
            return None
        if set(transition) != REPLAY_KEYS:
            raise ValueError("replay transition schema changed")
        record = {
            "episode": int(episode),
            "step": int(step),
            "policy_action": assert_legal_aubo_action7(policy_action),
            "executed_action": assert_legal_aubo_action7(executed_action),
            "intervention_action": assert_legal_aubo_action7(intervention_action),
            "intervention_flag": True,
            "observations": np.asarray(observations, dtype=np.float32),
            "next_observations": np.asarray(next_observations, dtype=np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "mask": np.float32(transition["masks"]),
            "done": np.bool_(transition["dones"]),
            "terminated": np.bool_(terminated),
            "truncated": np.bool_(truncated),
            "checkpoint_identity": self.checkpoint_identity,
            "expert_identity": self.expert_identity,
            "provider_provenance": dict(provider_provenance or {}),
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "transition_identity": transition_identity(transition),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        normalized = _json_value(record)
        self.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with self.provenance_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(normalized, sort_keys=True, allow_nan=False) + "\n"
            )
        return normalized


@dataclass(frozen=True)
class ControlStepResult:
    next_obs: np.ndarray
    reward: np.ndarray
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    transition: Mapping[str, Any]
    decision: InterventionDecision
    executed_action: np.ndarray


def execute_intervention_control_step(
    *,
    env: Any,
    online_store: Any,
    intervention_store: Any,
    adapter: HILSERLInterventionAdapterV1,
    obs20: Any,
    policy_action: Any,
    c3_has_authority: bool,
    episode: int,
    step: int,
) -> ControlStepResult:
    """Decide, execute exactly once, and route one canonical transition."""

    policy = assert_legal_aubo_action7(policy_action)
    decision = adapter.decide(
        obs20=obs20,
        policy_action=policy,
        c3_has_authority=c3_has_authority,
    )
    executed = assert_legal_aubo_action7(
        decision.expert_action if decision.intervened else policy
    )
    next_obs, reward, terminated, truncated, info = env.step(executed)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    reward = np.asarray(reward, dtype=np.float32)
    terminated = bool(terminated)
    truncated = bool(truncated)
    info = dict(info or {})
    if decision.intervened:
        info["intervene_action"] = executed.copy()

    transition = {
        "observations": np.asarray(obs20, dtype=np.float32),
        "actions": executed.copy(),
        "next_observations": next_obs,
        "rewards": reward,
        "masks": np.float32(1.0 - float(terminated)),
        "dones": terminated,
    }
    if set(transition) != REPLAY_KEYS:
        raise RuntimeError("internal replay schema violation")
    online_store.insert(transition)
    if decision.intervened:
        if intervention_store is None:
            raise RuntimeError("intervention store is required for dual routing")
        intervention_store.insert(transition)
        adapter.intervention_count += 1
        adapter.append_provenance_sidecar(
            episode=episode,
            step=step,
            policy_action=policy,
            executed_action=executed,
            intervention_action=executed,
            observations=obs20,
            next_observations=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            transition=transition,
            provider_provenance=decision.provenance,
        )
    return ControlStepResult(
        next_obs=next_obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        transition=transition,
        decision=decision,
        executed_action=executed,
    )
