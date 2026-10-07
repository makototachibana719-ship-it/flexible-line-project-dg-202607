#!/usr/bin/env python3
"""Read-only OCDBT/TensorStore diagnostics for Object checkpoint 4500."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "external/packages/object_grasping/object_grasp_package/checkpoints/checkpoint_4500"
)


def ocdbt_spec():
    return {
        "driver": "ocdbt",
        "base": {"driver": "file", "path": str(CHECKPOINT)},
    }


def zarr_spec(path: str):
    return {
        "driver": "zarr",
        "kvstore": {**ocdbt_spec(), "path": path + "/"},
    }


def kv_inventory():
    import tensorstore as ts

    started = time.monotonic()
    store = ts.KvStore.open(ocdbt_spec()).result()
    open_elapsed = time.monotonic() - started
    started = time.monotonic()
    keys = list(store.list().result())
    list_elapsed = time.monotonic() - started
    arrays = []
    for key in keys:
        name = key.decode()
        if not name.endswith("/.zarray"):
            continue
        metadata = json.loads(store.read(key).result().value.decode())
        logical_bytes = int(np.prod(metadata["shape"])) * np.dtype(
            metadata["dtype"]
        ).itemsize
        arrays.append(
            {
                "path": name[: -len("/.zarray")],
                "shape": metadata["shape"],
                "dtype": metadata["dtype"],
                "chunks": metadata["chunks"],
                "logical_bytes": logical_bytes,
            }
        )
    arrays.sort(key=lambda item: (item["logical_bytes"], item["path"]))
    return {
        "open": "PASS",
        "open_elapsed_seconds": open_elapsed,
        "list": "PASS",
        "list_elapsed_seconds": list_elapsed,
        "kv_key_count": len(keys),
        "array_count": len(arrays),
        "smallest_arrays": arrays[:20],
        "largest_arrays": arrays[-20:],
    }


def read_leaf(path: str):
    import tensorstore as ts

    started = time.monotonic()
    array = ts.open(zarr_spec(path), open=True).result()
    open_elapsed = time.monotonic() - started
    started = time.monotonic()
    value = array.read().result()
    read_elapsed = time.monotonic() - started
    numpy_value = np.asarray(value)
    return {
        "path": path,
        "open": "PASS",
        "open_elapsed_seconds": open_elapsed,
        "read": "PASS",
        "read_elapsed_seconds": read_elapsed,
        "shape": list(numpy_value.shape),
        "dtype": str(numpy_value.dtype),
        "nbytes": int(numpy_value.nbytes),
        "finite": bool(np.isfinite(numpy_value).all()),
    }


def orbax_metadata():
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    started = time.monotonic()
    metadata = checkpointer.metadata(str(CHECKPOINT))
    elapsed = time.monotonic() - started
    leaves = __import__("jax").tree_util.tree_leaves(metadata)
    return {
        "inspection": "PASS",
        "elapsed_seconds": elapsed,
        "metadata_leaf_count": len(leaves),
        "metadata_type": type(metadata).__name__,
    }


def no_target_worker():
    from flax.training import checkpoints

    started = time.monotonic()
    restored = checkpoints.restore_checkpoint(
        str(CHECKPOINT.parent), target=None, step=4500
    )
    elapsed = time.monotonic() - started
    leaves = __import__("jax").tree_util.tree_leaves(restored)
    return {
        "restore": "PASS",
        "elapsed_seconds": elapsed,
        "leaf_count": len(leaves),
        "result_type": type(restored).__name__,
    }


def proc_sample(pid: int):
    sample = {"elapsed_seconds": None, "pid": pid}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        status_lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        io_lines = Path(f"/proc/{pid}/io").read_text().splitlines()
        status = dict(
            line.split(":", 1) for line in status_lines if ":" in line
        )
        io = dict(line.split(":", 1) for line in io_lines if ":" in line)
        sample.update(
            {
                "state": stat[2],
                "utime_ticks": int(stat[13]),
                "stime_ticks": int(stat[14]),
                "rss_kb": int(status.get("VmRSS", "0 kB").split()[0]),
                "read_bytes": int(io.get("read_bytes", "0")),
                "rchar": int(io.get("rchar", "0")),
            }
        )
    except (FileNotFoundError, ProcessLookupError):
        sample["state"] = "EXITED"
    return sample


def monitor_no_target(timeout_seconds: int):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "no-target-worker",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    samples = []
    while process.poll() is None and time.monotonic() - started < timeout_seconds:
        sample = proc_sample(process.pid)
        sample["elapsed_seconds"] = time.monotonic() - started
        samples.append(sample)
        time.sleep(1)
    timed_out = process.poll() is None
    if timed_out:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stdout, stderr = process.communicate()
    return {
        "result": "TIMEOUT" if timed_out else "COMPLETED",
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "samples": samples,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("inventory", "leaf", "orbax-metadata", "no-target-worker", "monitor-no-target"),
    )
    parser.add_argument("--path")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "inventory":
        report = kv_inventory()
    elif args.mode == "leaf":
        if not args.path:
            parser.error("--path is required for leaf mode")
        report = read_leaf(args.path)
    elif args.mode == "orbax-metadata":
        report = orbax_metadata()
    elif args.mode == "no-target-worker":
        report = no_target_worker()
    else:
        report = monitor_no_target(args.timeout_seconds)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
