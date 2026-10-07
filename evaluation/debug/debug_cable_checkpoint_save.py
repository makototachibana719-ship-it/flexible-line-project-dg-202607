"""Isolated reproduction for the Cable checkpoint backend compatibility error."""

from __future__ import annotations

import importlib.metadata
import multiprocessing
import platform
from datetime import datetime
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import tensorstore as ts
from flax import config as flax_config
from flax.training import checkpoints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "runtime/checkpoint_debug"


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "flax": flax.__version__,
        "orbax-checkpoint": importlib.metadata.version("orbax-checkpoint"),
        "tensorstore": importlib.metadata.version("tensorstore"),
        "jax": jax.__version__,
    }


def equivalent_train_state_pytree() -> dict:
    """Small array pytree exercising the same PyTree/TensorStore save route."""
    return {
        "step": jnp.asarray(1, dtype=jnp.int32),
        "params": {
            "actor": {"kernel": jnp.arange(12, dtype=jnp.float32).reshape(3, 4)},
            "critic": {"bias": jnp.zeros((4,), dtype=jnp.float32)},
        },
        "opt_state": {"count": jnp.asarray(1, dtype=jnp.int32)},
    }


def save_with_legacy_flax(directory: Path, target: dict) -> str:
    old_value = flax_config.flax_use_orbax_checkpointing
    flax_config.update("flax_use_orbax_checkpointing", False)
    try:
        return checkpoints.save_checkpoint(
            directory, target, step=0, keep=20, overwrite=False
        )
    finally:
        flax_config.update("flax_use_orbax_checkpointing", old_value)


def default_orbax_worker(directory: str, queue) -> None:
    """Run the potentially hanging default backend in a killable child."""
    try:
        checkpoints.save_checkpoint(
            directory, equivalent_train_state_pytree(), step=0, keep=20
        )
        queue.put(("PASS", ""))
    except Exception as exc:
        queue.put(("FAIL", f"{type(exc).__name__}: {exc}"))


def reproduce_tensorstore_spec_error(directory: Path) -> str:
    """Exercise the exact Orbax-emitted kvstore member rejected by TensorStore."""
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(directory) + "/",
            "assume_config": True,
        },
        "metadata": {"shape": [1], "chunks": [1], "dtype": "<f4"},
        "create": True,
    }
    try:
        ts.open(spec).result()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError("TensorStore unexpectedly accepted assume_config")


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = OUTPUT_ROOT / stamp
    orbax_dir = run_root / "default_orbax"
    legacy_dir = run_root / "legacy_flax"
    run_root.mkdir(parents=True, exist_ok=False)
    target = equivalent_train_state_pytree()

    for key, value in versions().items():
        print(f"{key.upper()}: {value}")
    print(f"DEBUG_ROOT: {run_root}")

    direct_error = reproduce_tensorstore_spec_error(run_root / "tensorstore_direct")
    print(f"TENSORSTORE_ASSUME_CONFIG: FAIL ({direct_error})", flush=True)

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=default_orbax_worker, args=(str(orbax_dir), queue)
    )
    process.start()
    process.join(timeout=15.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        default_status = "FAIL_TIMEOUT"
        default_detail = "did not return within 15.0s"
    elif queue.empty():
        default_status = "FAIL"
        default_detail = f"child exited {process.exitcode} without result"
    else:
        default_status, default_detail = queue.get_nowait()
    print(
        f"DEFAULT_ORBAX_SAVE: {default_status}"
        + (f" ({default_detail})" if default_detail else ""),
        flush=True,
    )

    legacy_path = save_with_legacy_flax(legacy_dir, target)
    restored = checkpoints.restore_checkpoint(legacy_path, target=None)
    restored_kernel = jnp.asarray(restored["params"]["actor"]["kernel"])
    if not jnp.array_equal(restored_kernel, target["params"]["actor"]["kernel"]):
        raise AssertionError("legacy checkpoint restore content mismatch")
    print(f"LEGACY_FLAX_SAVE: PASS ({legacy_path})")
    print("LEGACY_FLAX_RESTORE: PASS")

    if default_status != "PASS" and "assume_config" in direct_error:
        print("ROOT_CAUSE: ORBAX_TENSORSTORE_MISMATCH")
        print("RESULT: PASS")
        return 0
    print("ROOT_CAUSE: UNVERIFIED (default Orbax save did not reproduce failure)")
    print("RESULT: UNVERIFIED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
