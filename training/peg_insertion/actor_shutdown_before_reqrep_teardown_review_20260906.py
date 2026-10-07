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


_shutdown_diagnostic_marker = None


def set_shutdown_diagnostic_marker(callback) -> None:
    """Install an optional diagnostics-only marker callback."""
    global _shutdown_diagnostic_marker
    _shutdown_diagnostic_marker = callback


def _mark_shutdown_diagnostic(name: str) -> None:
    callback = _shutdown_diagnostic_marker
    if callback is not None:
        callback(name)


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


def _reqrep_record(req_rep) -> dict:
    socket = getattr(req_rep, "socket", None) if req_rep is not None else None
    context = getattr(req_rep, "context", None) if req_rep is not None else None
    return {
        "present": req_rep is not None,
        "socket_present": socket is not None,
        "socket_closed": bool(getattr(socket, "closed", False)) if socket is not None else True,
        "context_present": context is not None,
        "context_closed": bool(getattr(context, "closed", False)) if context is not None else True,
        "explicitly_closed": bool(getattr(req_rep, "_peg_explicitly_closed", False))
                            if req_rep is not None else True,
    }


def _close_req_rep_client(req_rep) -> list[str]:
    """Close Agentlace request-side ZMQ resources with linger=0 and idempotence."""
    if req_rep is None or getattr(req_rep, "_peg_explicitly_closed", False):
        return []
    _mark_shutdown_diagnostic("M06_EXPLICIT_CLEANUP_BEGIN")
    errors = []
    socket = getattr(req_rep, "socket", None)
    context = getattr(req_rep, "context", None)
    if socket is not None and not getattr(socket, "closed", False):
        try:
            socket.setsockopt(zmq.LINGER, 0)
            _mark_shutdown_diagnostic("M07_SOCKET_LINGER_SET")
        except BaseException as exc:
            errors.append(f"req.socket.linger: {type(exc).__name__}: {exc}")
        try:
            _mark_shutdown_diagnostic("M08_SOCKET_CLOSE_BEGIN")
            socket.close(linger=0)
            _mark_shutdown_diagnostic("M09_SOCKET_CLOSE_END")
        except TypeError:
            try:
                socket.close()
                _mark_shutdown_diagnostic("M09_SOCKET_CLOSE_END")
            except BaseException as exc:
                errors.append(f"req.socket.close: {type(exc).__name__}: {exc}")
        except BaseException as exc:
            errors.append(f"req.socket.close: {type(exc).__name__}: {exc}")
    if context is not None and not getattr(context, "closed", False):
        try:
            # destroy(linger=0) closes any context-owned reset socket before term.
            _mark_shutdown_diagnostic("M10_CONTEXT_CLOSE_BEGIN")
            context.destroy(linger=0)
            _mark_shutdown_diagnostic("M11_CONTEXT_CLOSE_END")
        except TypeError:
            try:
                context.destroy()
                _mark_shutdown_diagnostic("M11_CONTEXT_CLOSE_END")
            except BaseException as exc:
                errors.append(f"req.context.destroy: {type(exc).__name__}: {exc}")
        except BaseException as exc:
            errors.append(f"req.context.destroy: {type(exc).__name__}: {exc}")
    if not errors:
        req_rep._peg_explicitly_closed = True
    _mark_shutdown_diagnostic("M12_EXPLICIT_CLEANUP_END")
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
        req_rep = getattr(client, "req_rep_client", None)
        before = {"broadcast": _thread_record(broadcast_thread),
                  "update": _thread_record(update_thread),
                  "req_rep": _reqrep_record(req_rep)}

        if broadcast is not None:
            broadcast.is_kill = True
        stop_update = getattr(client, "stop_update_flag", None)
        if stop_update is not None:
            stop_update.set()

        # Close network resources before joining so blocked receive/request calls wake.
        close_errors = _close_broadcast(broadcast) if broadcast is not None else []
        close_errors.extend(_close_req_rep_client(req_rep))
        for thread in (broadcast_thread, update_thread):
            if thread is not None and thread.is_alive():
                thread.join(max(0.0, deadline - time.monotonic()))
        timed_out = [name for name, thread in (("broadcast", broadcast_thread),
                     ("update", update_thread)) if thread is not None and thread.is_alive()]

        elapsed = time.monotonic() - started
        after = {"broadcast": _thread_record(broadcast_thread),
                 "update": _thread_record(update_thread),
                 "req_rep": _reqrep_record(req_rep)}
        if timed_out or close_errors:
            reason = (("shutdown threads still alive: " + ",".join(timed_out))
                      if timed_out else "; ".join(close_errors))
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
        try:
            super().__init__(*args, **kwargs)
        except BaseException as primary:
            req_rep = getattr(self, "req_rep_client", None)
            before = _reqrep_record(req_rep)
            errors = _close_req_rep_client(req_rep)
            payload = {"status": "PASS" if not errors else "FAIL",
                       "stage": "TRAINERCLIENT_CONSTRUCTOR_EXCEPTION",
                       "req_rep_before": before, "req_rep_after": _reqrep_record(req_rep),
                       "close_errors": errors, "exception_type": type(primary).__name__,
                       "exception_message": str(primary)}
            _write_evidence(self._peg_shutdown_evidence, payload)
            if errors:
                primary.add_note("request-side ZeroMQ cleanup also failed: " + "; ".join(errors))
            raise

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
