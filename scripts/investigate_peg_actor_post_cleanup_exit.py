#!/usr/bin/env python3
"""Actor-only post-cleanup process-exit investigation; never trains."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "scripts/run_peg_f3_v2_rlpd_smoke.py"
OUTPUT = ROOT / "results/validation/peg/f3_actor_post_cleanup_exit_investigation_20260906"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def native_snapshot(pid: int, delay: int) -> dict:
    task_root = Path(f"/proc/{pid}/task")
    threads = []
    if task_root.is_dir():
        for task in sorted(task_root.iterdir(), key=lambda item: int(item.name)):
            record = {"tid": int(task.name)}
            for name in ("comm", "wchan", "stack"):
                try:
                    record[name] = (task / name).read_text(errors="replace").strip()
                except (OSError, PermissionError) as exc:
                    record[name] = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
            try:
                status = (task / "status").read_text(errors="replace").splitlines()
                record["status"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                                    for line in status if ":" in line and
                                    line.split(":", 1)[0] in {"Name", "State", "Pid", "Tgid"}}
            except (OSError, PermissionError) as exc:
                record["status"] = {"error": f"{type(exc).__name__}:{exc}"}
            threads.append(record)
    return {"delay_seconds": delay, "timestamp_ns": time.time_ns(),
            "process_alive": task_root.is_dir(), "native_threads": threads}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"exclusive output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "stop_actor.json", {
        "completed_updates": 0, "timestamp_ns": time.time_ns(),
        "diagnostic_preexisting_stop": True})
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl",
               WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    log = (OUTPUT / "actor.log").open("w")
    command = [sys.executable, "-u", str(ACTOR), "--role", "actor", "--output", str(OUTPUT),
               "--short-shutdown-acceptance", "--post-cleanup-exit-diagnostics"]
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    cleanup = OUTPUT / "actor_cleanup_complete.json"
    deadline = time.monotonic() + 180
    while not cleanup.is_file():
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"actor exited before cleanup evidence: {process.returncode}")
        if time.monotonic() > deadline:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            log.close()
            raise TimeoutError("actor cleanup startup deadline")
        time.sleep(.1)
    started = time.monotonic()
    snapshots = []
    for delay in (0, 2, 5, 10):
        remaining = delay - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        snapshot = native_snapshot(process.pid, delay)
        snapshots.append(snapshot)
        write_json(OUTPUT / f"native_state_t{delay}.json", snapshot)
    alive_after_10 = process.poll() is None
    natural_returncode = process.poll()
    cleanup_method = "NONE"
    if alive_after_10:
        cleanup_method = "SIGTERM"
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cleanup_method = "SIGKILL"
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    log.close()
    result = {"status": "EVIDENCE_CAPTURED", "actor_pid": process.pid,
              "actor_alive_after_cleanup_t10": alive_after_10,
              "natural_returncode": natural_returncode,
              "cleanup_method": cleanup_method,
              "final_returncode": process.returncode,
              "actor_steps": json.loads((OUTPUT / "actor_result.json").read_text()).get("steps"),
              "training_started": False, "learner_started": False}
    write_json(OUTPUT / "investigation_result.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
