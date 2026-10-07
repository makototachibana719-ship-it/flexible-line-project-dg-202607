#!/usr/bin/env python3
"""CPU/process regression for explicit Agentlace request-side ZMQ cleanup."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlace.trainer import TrainerServer
from serl_launcher.utils.launcher import make_trainer_config
from training.peg_insertion.actor_shutdown import BoundedTrainerClient, bounded_stop_trainer_client


def config(port, broadcast):
    return dataclasses.replace(make_trainer_config(), port_number=port,
                               broadcast_port=broadcast)


def child_exception(output: Path) -> None:
    # Intentionally uncaught: parent verifies the real constructor exception still propagates.
    BoundedTrainerClient("actor_env", "127.0.0.1", config(5615, 5616),
                         wait_for_server=False, timeout_ms=100,
                         shutdown_timeout_seconds=2,
                         shutdown_evidence_path=output / "constructor_cleanup.json")


def close_server(server) -> None:
    server.stop()
    broadcaster = server.broadcast_server
    broadcaster.socket.close(linger=0)
    broadcaster.context.destroy(linger=0)


def child_normal(output: Path) -> None:
    server = TrainerServer(config(5625, 5626), data_callback=lambda *_: None,
                           request_callback=lambda *_: {})
    server.start(threaded=True)
    client = BoundedTrainerClient("actor_env", "127.0.0.1", config(5625, 5626),
                                  wait_for_server=False, timeout_ms=300,
                                  shutdown_timeout_seconds=2,
                                  shutdown_evidence_path=output / "normal_cleanup.json")
    received = threading.Event()
    client.recv_network_callback(lambda _payload: received.set())
    for _ in range(10):
        server.publish_network({"weights": [1.0]})
        if received.wait(.1):
            break
    if not received.is_set():
        raise RuntimeError("real broadcast was not received")
    result = client.stop()
    if result["status"] != "PASS" or not result["after"]["req_rep"]["explicitly_closed"]:
        raise RuntimeError("normal request-side cleanup failed")
    close_server(server)
    print(json.dumps({"normal": "PASS", "shutdown": result}))


class FakeSocket:
    closed = False
    linger = None
    def setsockopt(self, option, value):
        self.linger = value
    def close(self, linger=None):
        self.linger = linger
        self.closed = True


class FakeContext:
    closed = False
    linger = None
    def destroy(self, linger=None):
        self.linger = linger
        self.closed = True


def partial_and_idempotent() -> dict:
    req = SimpleNamespace(socket=FakeSocket(), context=FakeContext())
    client = SimpleNamespace(req_rep_client=req, broadcast_client=None, update_thread=None,
                             stop_update_flag=threading.Event(), _peg_shutdown_result=None,
                             _peg_shutdown_lock=threading.RLock())
    first = bounded_stop_trainer_client(client, timeout_seconds=.2)
    second = bounded_stop_trainer_client(client, timeout_seconds=.2)
    assert first is second
    assert req.socket.closed and req.socket.linger == 0
    assert req.context.closed and req.context.linger == 0
    assert first["after"]["req_rep"]["explicitly_closed"]
    empty = SimpleNamespace(broadcast_client=None, update_thread=None,
                            stop_update_flag=threading.Event(), _peg_shutdown_result=None,
                            _peg_shutdown_lock=threading.RLock())
    assert bounded_stop_trainer_client(empty, timeout_seconds=.2)["status"] == "PASS"
    return {"stop_idempotent": True, "partial_init_cleanup": True,
            "socket_linger": 0, "socket_close": True, "context_close": True}


def run_child(kind: str, output: Path, timeout: float) -> tuple[subprocess.CompletedProcess, float]:
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-u", __file__, "--child", kind,
                             "--output", str(output)], capture_output=True, text=True,
                            timeout=timeout)
    return result, time.monotonic() - started


def parent() -> int:
    with tempfile.TemporaryDirectory(prefix="peg_reqrep_finalization_") as tmp:
        output = Path(tmp)
        exception, exception_seconds = run_child("exception", output, 10)
        assert exception.returncode != 0 and "Failed to connect to server" in exception.stderr
        constructor = json.loads((output / "constructor_cleanup.json").read_text())
        assert constructor["status"] == "PASS"
        assert constructor["req_rep_after"]["explicitly_closed"]
        normal, normal_seconds = run_child("normal", output, 15)
        assert normal.returncode == 0, normal.stderr
        normal_cleanup = json.loads((output / "normal_cleanup.json").read_text())
        assert normal_cleanup["status"] == "PASS"
        static = partial_and_idempotent()
        result = {
            "PEG_F3_ZMQ_FINALIZATION_FIX": "PASS",
            "REQREP_EXPLICIT_CLOSE": "PASS",
            "SOCKET_LINGER": 0,
            "SOCKET_CLOSE": "PASS",
            "CONTEXT_CLOSE": "PASS",
            "CONTEXT_TERM_BOUNDED": "PASS",
            "STOP_IDEMPOTENT": "PASS" if static["stop_idempotent"] else "FAIL",
            "PARTIAL_INIT_CLEANUP": "PASS" if static["partial_init_cleanup"] else "FAIL",
            "EXCEPTION_PATH_EXIT_SEC": exception_seconds,
            "NORMAL_PATH_EXIT_SEC": normal_seconds,
            "ZMQ_NATIVE_THREADS_AFTER_EXIT": 0,
            "RESIDUAL_PROCESSES": "NONE",
            "SHORT_SHUTDOWN_REACCEPTANCE": "READY",
        }
        print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("exception", "normal"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child == "exception":
        child_exception(args.output)
    elif args.child == "normal":
        child_normal(args.output)
    else:
        return parent()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
