#!/usr/bin/env python3
"""One-shot no-server ReqRep timeout phase localization."""
from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


MARKER_NAMES = [
    "M00_PROCESS_ENTRY", "M01_IMPORTS_BEGIN", "M02_IMPORTS_DONE",
    "M03_CLIENT_CONSTRUCT_BEGIN", "M04_CLIENT_CONSTRUCT_EXCEPTION",
    "M05_EXCEPTION_HANDLER_ENTER", "M06_EXPLICIT_CLEANUP_BEGIN",
    "M07_SOCKET_LINGER_SET", "M08_SOCKET_CLOSE_BEGIN", "M09_SOCKET_CLOSE_END",
    "M10_CONTEXT_CLOSE_BEGIN", "M11_CONTEXT_CLOSE_END",
    "M12_EXPLICIT_CLEANUP_END", "M13_TOPLEVEL_FINALLY_ENTER",
    "M14_TOPLEVEL_FINALLY_EXIT", "M15_ATEXIT_ENTER", "M16_ATEXIT_EXIT",
]


def _durable_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _child(output: Path) -> None:
    started = time.monotonic()
    markers = output / "markers"

    def mark(name: str) -> None:
        index = MARKER_NAMES.index(name)
        _durable_json(markers / f"{index:02d}_{name}.json", {
            "marker": name, "pid": os.getpid(),
            "elapsed_seconds": time.monotonic() - started,
            "time_ns": time.time_ns(),
        })

    mark("M00_PROCESS_ENTRY")
    diagnostic_stream = (output / "faulthandler.log").open("w", encoding="utf-8")
    faulthandler.enable(file=diagnostic_stream, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=diagnostic_stream, all_threads=True)

    def exit_probe() -> None:
        mark("M15_ATEXIT_ENTER")
        mark("M16_ATEXIT_EXIT")
        diagnostic_stream.flush()
        os.fsync(diagnostic_stream.fileno())

    atexit.register(exit_probe)
    mark("M01_IMPORTS_BEGIN")
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from training.peg_insertion.actor_shutdown import (  # noqa: PLC0415
        BoundedTrainerClient, set_shutdown_diagnostic_marker)
    from serl_launcher.utils.launcher import make_trainer_config  # noqa: PLC0415
    import dataclasses  # noqa: PLC0415
    mark("M02_IMPORTS_DONE")
    set_shutdown_diagnostic_marker(mark)
    config = dataclasses.replace(make_trainer_config(), port_number=5635,
                                 broadcast_port=5636)
    try:
        try:
            mark("M03_CLIENT_CONSTRUCT_BEGIN")
            BoundedTrainerClient(
                "peg_reqrep_timeout_localization", "127.0.0.1", config,
                wait_for_server=False, timeout_ms=100,
                shutdown_timeout_seconds=2.0,
                shutdown_evidence_path=output / "constructor_cleanup.json")
        except BaseException:
            mark("M04_CLIENT_CONSTRUCT_EXCEPTION")
            mark("M05_EXCEPTION_HANDLER_ENTER")
            raise
    finally:
        mark("M13_TOPLEVEL_FINALLY_ENTER")
        mark("M14_TOPLEVEL_FINALLY_EXIT")


def _last_marker(markers: Path) -> dict | None:
    files = sorted(markers.glob("*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _proc_snapshot(pid: int) -> dict:
    base = Path(f"/proc/{pid}")
    result = {"pid": pid, "exists": base.exists()}
    if not base.exists():
        return result
    for name in ("status", "wchan"):
        try:
            result[name] = (base / name).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result[name] = f"UNAVAILABLE: {exc}"
    tasks = sorted((base / "task").iterdir(), key=lambda path: int(path.name))
    result["task_count"] = len(tasks)
    result["tasks"] = []
    for task in tasks:
        record = {"tid": int(task.name)}
        for name in ("comm", "wchan", "stack"):
            try:
                record[name] = (task / name).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                record[name] = f"UNAVAILABLE: {exc}"
        result["tasks"].append(record)
    ps = subprocess.run(
        ["ps", "-L", "-p", str(pid), "-o", "pid,tid,stat,comm,wchan:40"],
        capture_output=True, text=True, timeout=3, check=False)
    result["ps_L"] = ps.stdout + ps.stderr
    return result


def _classify(last: str | None) -> tuple[str, str]:
    if last in (None, "M00_PROCESS_ENTRY", "M01_IMPORTS_BEGIN"):
        return "PRE_CONSTRUCTION_TIMEOUT", "PYTHON_IMPORT_BOOTSTRAP"
    if last in ("M02_IMPORTS_DONE", "M03_CLIENT_CONSTRUCT_BEGIN"):
        return "CONSTRUCTOR_TIMEOUT", "TRAINERCLIENT_CONSTRUCTION"
    if last in ("M04_CLIENT_CONSTRUCT_EXCEPTION", "M05_EXCEPTION_HANDLER_ENTER"):
        return "CONSTRUCTOR_TIMEOUT", "CONSTRUCTOR_EXCEPTION_HANDLING"
    if last and last.startswith(tuple(f"M{i:02d}_" for i in range(6, 13))):
        return "CLEANUP_TIMEOUT", "EXPLICIT_REQREP_CLEANUP"
    if last in ("M13_TOPLEVEL_FINALLY_ENTER", "M14_TOPLEVEL_FINALLY_EXIT", "M15_ATEXIT_ENTER"):
        return "POST_CLEANUP_INTERPRETER_TIMEOUT", "TOPLEVEL_RETURN_OR_ATEXIT"
    if last == "M16_ATEXIT_EXIT":
        return "POST_CLEANUP_INTERPRETER_TIMEOUT", "POST_ATEXIT_INTERPRETER_EXIT"
    return "UNRESOLVED", "UNRESOLVED"


def _parent(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "child.log"
    snapshots = []
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", "--output", str(output)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for deadline in (2.0, 5.0, 10.0):
            delay = deadline - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            last = _last_marker(output / "markers")
            snapshots.append({"sample_time_seconds": deadline,
                              "process_alive": child.poll() is None,
                              "last_marker": last})
        timed_out = child.poll() is None
        proc = None
        if timed_out:
            proc = _proc_snapshot(child.pid)
            os.kill(child.pid, signal.SIGUSR1)
            time.sleep(0.5)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
        else:
            child.wait(timeout=1)
    final_marker = _last_marker(output / "markers")
    boundary_marker = snapshots[-1]["last_marker"]
    if timed_out:
        classification, phase = _classify(
            boundary_marker["marker"] if boundary_marker else None)
    else:
        classification, phase = "EXIT_WITHIN_BOUND", "COMPLETED"
    result = {
        "timed_out_at_10_seconds": timed_out,
        "child_returncode": child.returncode,
        "snapshots": snapshots,
        "last_marker_at_10_seconds": boundary_marker,
        "final_marker_after_timeout_sampling": final_marker,
        "classification": classification,
        "timeout_phase": phase,
        "proc_snapshot": proc,
    }
    _durable_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.child:
        _child(args.output)
        return 0
    return _parent(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
