"""Read-only, policy-free environment smoke-test helpers."""

from __future__ import annotations

import importlib
import importlib.metadata
import errno
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np


FIELDS = (
    "PROCESS",
    "PACKAGE",
    "PYTHON_PATH",
    "PYTHON",
    "GYM",
    "GYMNASIUM",
    "MUJOCO",
    "JAX",
    "FLAX",
    "IMPORT",
    "ENV_CREATE",
    "RESET",
    "RESET_RETURN_TYPE",
    "OBS_TYPE",
    "OBS_KEYS",
    "OBS_SHAPES",
    "OBS_DTYPES",
    "ACTION_SPACE",
    "ACTION_SHAPE",
    "STEP",
    "STEP_RETURN_LENGTH",
    "REWARD_TYPE",
    "TERMINATED",
    "TRUNCATED",
    "INFO_KEYS",
    "CLOSE",
    "RESULT",
)


def package_version(import_name: str, distribution_name: str | None = None) -> str:
    """Return an installed version without making installation changes."""
    try:
        module = importlib.import_module(import_name)
    except Exception:
        try:
            return importlib.metadata.version(distribution_name or import_name)
        except importlib.metadata.PackageNotFoundError:
            return "MISSING"
    return str(getattr(module, "__version__", "UNKNOWN"))


def base_report(process: str, package: str) -> dict[str, str]:
    report = {field: "UNVERIFIED" for field in FIELDS}
    report.update(
        {
            "PROCESS": process,
            "PACKAGE": package,
            "PYTHON_PATH": sys.executable,
            "PYTHON": sys.version.replace("\n", " "),
            "GYM": package_version("gym"),
            "GYMNASIUM": package_version("gymnasium"),
            "MUJOCO": package_version("mujoco"),
            "JAX": package_version("jax"),
            "FLAX": package_version("flax"),
        }
    )
    return report


def observation_facts(obs: Any) -> tuple[str, str, str, str]:
    keys: list[str] = []
    shapes: dict[str, str] = {}
    dtypes: dict[str, str] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}/{key}" if path else str(key)
                keys.append(child_path)
                visit(child, child_path)
            return
        array = np.asarray(value)
        label = path or "<root>"
        shapes[label] = str(tuple(array.shape))
        dtypes[label] = str(array.dtype)

    visit(obs, "")
    return type(obs).__name__, repr(keys), repr(shapes), repr(dtypes)


def run_lifecycle(report: dict[str, str], create_env, steps: int = 1) -> None:
    env = None
    try:
        env = create_env()
        report["ENV_CREATE"] = "PASS"
        report["ACTION_SPACE"] = repr(env.action_space)
        report["ACTION_SHAPE"] = str(getattr(env.action_space, "shape", None))

        reset_result = env.reset()
        report["RESET"] = "PASS"
        report["RESET_RETURN_TYPE"] = (
            f"{type(reset_result).__name__}(len={len(reset_result)})"
            if isinstance(reset_result, tuple)
            else type(reset_result).__name__
        )
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs_type, keys, shapes, dtypes = observation_facts(obs)
        report["OBS_TYPE"] = obs_type
        report["OBS_KEYS"] = keys
        report["OBS_SHAPES"] = shapes
        report["OBS_DTYPES"] = dtypes

        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        if not env.action_space.contains(action):
            action = env.action_space.sample()

        last_result = None
        for _ in range(steps):
            last_result = env.step(action)
        report["STEP"] = f"PASS ({steps})"
        report["STEP_RETURN_LENGTH"] = (
            str(len(last_result)) if isinstance(last_result, tuple) else "NOT_TUPLE"
        )
        if isinstance(last_result, tuple) and len(last_result) == 5:
            _, reward, terminated, truncated, info = last_result
            report["REWARD_TYPE"] = f"{type(reward).__name__}: {reward!r}"
            report["TERMINATED"] = repr(terminated)
            report["TRUNCATED"] = repr(truncated)
            report["INFO_KEYS"] = repr(sorted(info.keys()))
    finally:
        if env is not None:
            env.close()
            report["CLOSE"] = "PASS"


def classify_exception(report: dict[str, str], exc: BaseException) -> None:
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        target = "IMPORT" if report["IMPORT"] == "UNVERIFIED" else "ENV_CREATE"
        report[target] = detail
        report["RESULT"] = "BLOCKED_DEPENDENCY"
    elif isinstance(exc, OSError) and getattr(exc, "errno", None) in {
        errno.EROFS,
        errno.EACCES,
        errno.EPERM,
    }:
        target = "ENV_CREATE" if report["ENV_CREATE"] == "UNVERIFIED" else "STEP"
        report[target] = detail
        report["RESULT"] = "BLOCKED_CODE"
    else:
        target = "ENV_CREATE" if report["ENV_CREATE"] == "UNVERIFIED" else "STEP"
        report[target] = detail
        report["RESULT"] = "FAIL"


def finish(report: dict[str, str]) -> None:
    if report["RESULT"] == "UNVERIFIED":
        required = ("IMPORT", "ENV_CREATE", "RESET", "STEP", "CLOSE")
        report["RESULT"] = (
            "PASS" if all(report[key].startswith("PASS") for key in required) else "FAIL"
        )
    for field in FIELDS:
        print(f"{field}: {report[field]}")
