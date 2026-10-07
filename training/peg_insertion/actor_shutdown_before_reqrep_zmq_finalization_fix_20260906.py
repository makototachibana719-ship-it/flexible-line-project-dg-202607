"""Project-owned bounded shutdown adapter for Agentlace actor clients."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import zmq
from agentlace.trainer import TrainerClient
from agentlace.zmq_wrapper.broadcast import BroadcastClient


class ShutdownTimeout(RuntimeError):
    """Raised after durable evidence when a client thread exceeds its stop bound."""


def _write_evidence(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _thread_record(thread) -> dict:
    return {
        "present": thread is not None,
        "alive": bool(thread and thread.is_alive()),
        "name": getattr(thread, "name", None),
        "daemon": getattr(thread, "daemon", None),
        "ident": getattr(thread, "ident", None),
    }


def _close_broadcast(broadcast) -> list[str]:
    errors = []
    socket = getattr(broadcast, "socket", None)
    context = getattr(broadcast, "context", None)
    if socket is not None:
        try:
            socket.close(linger=0)
        except TypeError:
            try:
                socket.close()
            except BaseException as exc:  # recorded and propagated by caller
                errors.append(f"socket.close: {type(exc).__name__}: {exc}")
        except BaseException as exc:
            errors.append(f"socket.close: {type(exc).__name__}: {exc}")
    if context is not None:
        try:
            context.destroy(linger=0)
        except TypeError:
            try:
                context.destroy()
            except BaseException as exc:
                errors.append(f"context.destroy: {type(exc).__name__}: {exc}")
        except BaseException as exc:
            errors.append(f"context.destroy: {type(exc).__name__}: {exc}")
    return errors


def bounded_stop_trainer_client(client, *, timeout_seconds: float = 10.0,
                                evidence_path: Path | None = None) -> dict:
    """Signal, bounded-join, close, and fail closed without an unbounded wait."""
    if timeout_seconds <= 0:
        raise ValueError("positive shutdown timeout required")
    lock = getattr(client, "_peg_shutdown_lock", None)
    if lock is None:
        lock = threading.RLock()
        client._peg_shutdown_lock = lock
    with lock:
        previous = getattr(client, "_peg_shutdown_result", None)
        if previous is not None:
            if previous["status"] != "PASS":
                raise ShutdownTimeout(previous["reason"])
            return previous

        started = time.monotonic()
        deadline = started + timeout_seconds
        broadcast = getattr(client, "broadcast_client", None)
        broadcast_thread = getattr(broadcast, "thread", None) if broadcast else None
        update_thread = getattr(client, "update_thread", None)
        before = {"broadcast": _thread_record(broadcast_thread),
                  "update": _thread_record(update_thread)}

        if broadcast is not None:
            broadcast.is_kill = True
        stop_update = getattr(client, "stop_update_flag", None)
        if stop_update is not None:
            stop_update.set()

        for thread in (broadcast_thread, update_thread):
            if thread is not None and thread.is_alive():
                thread.join(max(0.0, deadline - time.monotonic()))

        timed_out = [name for name, thread in (("broadcast", broadcast_thread),
                     ("update", update_thread)) if thread is not None and thread.is_alive()]
        close_errors = _close_broadcast(broadcast) if broadcast is not None else []
        # Closing the socket/context wakes a blocked recv. The second join is also bounded.
        if timed_out:
            grace_deadline = time.monotonic() + min(1.0, timeout_seconds)
            for thread in (broadcast_thread, update_thread):
                if thread is not None and thread.is_alive():
                    thread.join(max(0.0, grace_deadline - time.monotonic()))
            timed_out = [name for name, thread in (("broadcast", broadcast_thread),
                         ("update", update_thread)) if thread is not None and thread.is_alive()]

        elapsed = time.monotonic() - started
        after = {"broadcast": _thread_record(broadcast_thread),
                 "update": _thread_record(update_thread)}
        if timed_out or close_errors:
            reason = "shutdown threads still alive: " + ",".join(timed_out) if timed_out else "; ".join(close_errors)
            result = {"status": "FAIL", "reason": reason, "elapsed_seconds": elapsed,
                      "timeout_seconds": timeout_seconds, "before": before, "after": after,
                      "close_errors": close_errors, "bounded": True}
            client._peg_shutdown_result = result
            _write_evidence(evidence_path, result)
            raise ShutdownTimeout(reason)

        result = {"status": "PASS", "reason": None, "elapsed_seconds": elapsed,
                  "timeout_seconds": timeout_seconds, "before": before, "after": after,
                  "close_errors": [], "bounded": True}
        client._peg_shutdown_result = result
        _write_evidence(evidence_path, result)
        return result


class BoundedBroadcastClient(BroadcastClient):
    """Vendor-compatible receiver with a daemon thread and shutdown-aware errors."""

    def async_start(self, callback):
        def async_listen():
            while not self.is_kill:
                try:
                    serialized = self.socket.recv()
                    callback(self.decompress(serialized))
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    if self.is_kill:
                        return
                    raise
        self.thread = threading.Thread(target=async_listen,
                                       name="agentlace-broadcast-recv", daemon=True)
        self.thread.start()


class BoundedTrainerClient(TrainerClient):
    """TrainerClient whose stop cannot indefinitely join a network thread."""

    def __init__(self, *args, shutdown_timeout_seconds=10.0,
                 shutdown_evidence_path=None, **kwargs):
        self._peg_shutdown_timeout = float(shutdown_timeout_seconds)
        self._peg_shutdown_evidence = (Path(shutdown_evidence_path)
                                       if shutdown_evidence_path is not None else None)
        self._peg_shutdown_lock = threading.RLock()
        self._peg_shutdown_result = None
        super().__init__(*args, **kwargs)

    def recv_network_callback(self, callback):
        self.broadcast_client = BoundedBroadcastClient(
            self.server_ip, self.config.broadcast_port, log_level=self.log_level)
        self.broadcast_client.async_start(callback)

    def stop(self):
        return bounded_stop_trainer_client(
            self, timeout_seconds=self._peg_shutdown_timeout,
            evidence_path=self._peg_shutdown_evidence)


def install_bounded_actor_shutdown(module, evidence_path: Path, *, timeout_seconds=10.0):
    """Install a lifecycle-only wrapper around a delivered actor module."""
    clients = []

    class TrackingTrainerClient(BoundedTrainerClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("shutdown_timeout_seconds", timeout_seconds)
            kwargs.setdefault("shutdown_evidence_path", evidence_path)
            super().__init__(*args, **kwargs)
            clients.append(self)

    module.TrainerClient = TrackingTrainerClient
    original_actor = module.actor

    def actor_with_bounded_shutdown(*args, **kwargs):
        try:
            result = original_actor(*args, **kwargs)
        except BaseException as primary:
            try:
                for client in clients:
                    client.stop()
            except BaseException as cleanup:
                primary.add_note(f"actor shutdown also failed: {cleanup!r}")
            raise
        for client in clients:
            client.stop()
        return result

    module.actor = actor_with_bounded_shutdown
    return clients
