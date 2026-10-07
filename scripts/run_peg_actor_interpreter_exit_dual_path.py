#!/usr/bin/env python3
"""Dual-path Actor interpreter-exit evidence supervisor; never formal training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ACTOR_SCRIPT = ROOT / "scripts/run_peg_f3_v2_rlpd_smoke.py"
OUTPUT = ROOT / "results/validation/peg/f3_actor_interpreter_exit_dual_path_20260906"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def runtime_env():
    return dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                XLA_FLAGS="--xla_gpu_enable_triton_gemm=false", MUJOCO_GL="egl",
                WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")


def launch(role: str, output: Path, log, path_label: str | None = None):
    command = [sys.executable, "-u", str(ACTOR_SCRIPT), "--role", role,
               "--output", str(output), "--short-shutdown-acceptance"]
    if path_label is not None:
        command += ["--interpreter-exit-diagnostics", "--interpreter-exit-path", path_label]
    return subprocess.Popen(command, env=runtime_env(), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def wait_for(path: Path, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if process.poll() is not None:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(.1)
    return True


def read_optional(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def native_snapshot(pid: int, output: Path, label: str) -> dict:
    process_root = Path(f"/proc/{pid}")
    result = {"pid": pid, "label": label, "timestamp_ns": time.time_ns(),
              "process_alive": process_root.is_dir(), "tasks": []}
    for name in ("status", "wchan", "stack"):
        try:
            result[name] = (process_root / name).read_text(errors="replace").strip()
        except OSError as exc:
            result[name] = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
    task_root = process_root / "task"
    if task_root.is_dir():
        for task in sorted(task_root.iterdir(), key=lambda item: int(item.name)):
            item = {"tid": int(task.name)}
            for name in ("comm", "wchan", "stack", "status"):
                try:
                    item[name] = (task / name).read_text(errors="replace").strip()
                except OSError as exc:
                    item[name] = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
            result["tasks"].append(item)
    write_json(output / f"native_state_{label}.json", result)
    return result


def sample_after_finally(process, output: Path) -> list[dict]:
    started = time.monotonic()
    snapshots = []
    for delay in (0, 2, 5, 10):
        remaining = delay - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        snapshots.append(native_snapshot(process.pid, output, f"t{delay}"))
    return snapshots


def terminate_if_alive(process) -> str:
    if process.poll() is not None:
        return "NONE"
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
        return "SIGTERM"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return "SIGKILL"


def run_exception_path(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "stop_actor.json", {"completed_updates": 0,
               "timestamp_ns": time.time_ns(), "diagnostic_preexisting_stop": True})
    log = (output / "actor.log").open("x")
    process = launch("actor", output, log, "exception")
    reached = wait_for(output / "toplevel_finally_exit.json", process, 240)
    snapshots = sample_after_finally(process, output) if reached else []
    alive_t10 = bool(snapshots and snapshots[-1]["process_alive"])
    natural_returncode = process.poll()
    cleanup = terminate_if_alive(process)
    log.close()
    result = {"path": "exception", "finally_reached": reached,
              "atexit_enter": (output / "atexit_enter.json").is_file(),
              "atexit_exit": (output / "atexit_exit.json").is_file(),
              "alive_t10": alive_t10, "natural_returncode": natural_returncode,
              "cleanup_method": cleanup, "final_returncode": process.returncode,
              "learner_started": False, "learner_updates": 0}
    write_json(output / "path_result.json", result)
    return result


def run_normal_path(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    learner_log = (output / "learner.log").open("x")
    actor_log = (output / "actor.log").open("x")
    learner = launch("learner", output, learner_log)
    actor = None
    try:
        if not wait_for(output / "server_ready.json", learner, 300):
            raise RuntimeError("normal learner server readiness failed")
        actor = launch("actor", output, actor_log, "normal")
        learner.wait(timeout=300)
        finally_reached = wait_for(output / "toplevel_finally_exit.json", actor, 60)
        snapshots = sample_after_finally(actor, output) if finally_reached else []
        alive_t10 = bool(snapshots and snapshots[-1]["process_alive"])
        natural_returncode = actor.poll()
        cleanup = terminate_if_alive(actor)
        result = {"path": "normal", "finally_reached": finally_reached,
                  "atexit_enter": (output / "atexit_enter.json").is_file(),
                  "atexit_exit": (output / "atexit_exit.json").is_file(),
                  "alive_t10": alive_t10, "natural_returncode": natural_returncode,
                  "cleanup_method": cleanup, "final_returncode": actor.returncode,
                  "learner_returncode": learner.returncode,
                  "learner_updates": len((output / "updates.jsonl").read_text().splitlines())
                                     if (output / "updates.jsonl").is_file() else 0}
        write_json(output / "path_result.json", result)
        return result
    finally:
        terminate_if_alive(learner)
        if actor is not None:
            terminate_if_alive(actor)
        learner_log.close()
        actor_log.close()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"exclusive output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "attach_tools.json", {"py_spy": False, "gdb": False,
               "reason": "not installed; no installation authorized"})
    exception = run_exception_path(OUTPUT / "exception")
    normal = run_normal_path(OUTPUT / "normal")
    result = {"status": "EVIDENCE_CAPTURED", "exception": exception, "normal": normal,
              "production_modified": False, "formal_training_started": False,
              "smoke_1000_rerun": False}
    write_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
