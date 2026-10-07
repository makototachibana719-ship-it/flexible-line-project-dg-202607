"""Early, diagnostic-only probes for Actor interpreter exit paths."""
from __future__ import annotations

import atexit
import asyncio
from concurrent.futures import thread as futures_thread
import faulthandler
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import traceback


class InterpreterExitProbe:
    def __init__(self, output: Path, path_label: str):
        self.output = Path(output)
        self.path_label = path_label
        self.first_snapshot = threading.Event()
        self.fault_stream = None

    def _write(self, name: str, payload: dict) -> None:
        path = self.output / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)

    def _marker(self, name: str) -> None:
        self._write(name, {"path": self.path_label, "pid": os.getpid(),
                           "timestamp_ns": time.time_ns()})

    def _snapshot(self, label: str, delay: int | None = None) -> None:
        frames = sys._current_frames()
        threads = []
        for thread in threading.enumerate():
            frame = frames.get(thread.ident)
            threads.append({"name": thread.name, "ident": thread.ident,
                            "daemon": thread.daemon, "alive": thread.is_alive(),
                            "stack": traceback.format_stack(frame) if frame is not None else []})
        children = [{"pid": child.pid, "name": child.name, "alive": child.is_alive()}
                    for child in multiprocessing.active_children()]
        try:
            loop = asyncio.get_running_loop()
            pending = [repr(task) for task in asyncio.all_tasks(loop) if not task.done()]
            asyncio_state = {"running_loop": True, "pending": pending}
        except RuntimeError:
            asyncio_state = {"running_loop": False, "pending": []}
        executor_threads = []
        queues = getattr(futures_thread, "_threads_queues", {})
        for thread in list(queues):
            executor_threads.append({"name": thread.name, "ident": thread.ident,
                                     "daemon": thread.daemon, "alive": thread.is_alive()})
        payload = {"path": self.path_label, "pid": os.getpid(), "label": label,
                   "delay_seconds": delay, "timestamp_ns": time.time_ns(),
                   "threads": threads,
                   "non_daemon_threads": [item for item in threads if not item["daemon"]],
                   "multiprocessing_children": children, "asyncio": asyncio_state,
                   "executor_threads": executor_threads}
        self._write(f"python_state_{label}.json", payload)
        with (self.output / f"faulthandler_{label}.txt").open("x") as stream:
            faulthandler.dump_traceback(file=stream, all_threads=True)
            stream.flush()
            os.fsync(stream.fileno())

    def install(self) -> None:
        self.fault_stream = (self.output / "faulthandler_early.log").open("x")
        faulthandler.enable(file=self.fault_stream, all_threads=True)
        atexit.register(self._atexit)
        self._marker("early_probe_installed.json")

    def _atexit(self) -> None:
        self._marker("atexit_enter.json")
        self._snapshot("atexit")
        self._marker("atexit_exit.json")
        if self.fault_stream is not None:
            self.fault_stream.flush()
            os.fsync(self.fault_stream.fileno())

    def _delayed_samples(self) -> None:
        started = time.monotonic()
        for delay in (0, 2, 5, 10):
            remaining = delay - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            self._snapshot(f"t{delay}", delay)
            if delay == 0:
                self.first_snapshot.set()

    def top_level_finally(self) -> None:
        self._marker("toplevel_finally_enter.json")
        self._snapshot("toplevel_finally")
        sampler = threading.Thread(target=self._delayed_samples,
                                   name="interpreter-exit-diagnostic-sampler", daemon=True)
        sampler.start()
        if not self.first_snapshot.wait(timeout=2):
            raise RuntimeError("T+0 interpreter-exit snapshot timeout")
        self._marker("toplevel_finally_exit.json")
