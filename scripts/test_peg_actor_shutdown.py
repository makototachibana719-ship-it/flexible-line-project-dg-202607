#!/usr/bin/env python3
"""CPU/static regression for bounded Agentlace actor shutdown."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.peg_insertion.actor_shutdown import (
    BoundedBroadcastClient,
    ShutdownTimeout,
    bounded_stop_trainer_client,
)


class Closeable:
    def __init__(self):
        self.closed = 0
    def close(self, **kwargs):
        self.closed += 1


class Destroyable:
    def __init__(self):
        self.destroyed = 0
    def destroy(self, **kwargs):
        self.destroyed += 1


class FakeBroadcast:
    def __init__(self, thread=None):
        self.is_kill = False
        self.thread = thread
        self.socket = Closeable()
        self.context = Destroyable()


class FakeClient:
    def __init__(self, thread=None):
        self.broadcast_client = FakeBroadcast(thread)
        self.update_thread = None


def normal_thread(client):
    def worker():
        while not client.broadcast_client.is_kill:
            time.sleep(.001)
    thread = threading.Thread(target=worker, daemon=True)
    client.broadcast_client.thread = thread
    thread.start()
    return thread


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        normal = FakeClient()
        thread = normal_thread(normal)
        started = time.monotonic()
        result = bounded_stop_trainer_client(normal, timeout_seconds=1,
                                             evidence_path=root / "normal.json")
        normal_elapsed = time.monotonic() - started
        assert result["status"] == "PASS" and not thread.is_alive() and normal_elapsed < 1

        repeated = bounded_stop_trainer_client(normal, timeout_seconds=1,
                                               evidence_path=root / "unused.json")
        assert repeated is result and not (root / "unused.json").exists()

        dead_thread = threading.Thread(target=lambda: None, daemon=True)
        dead_thread.start(); dead_thread.join()
        dead = FakeClient(dead_thread)
        assert bounded_stop_trainer_client(dead, timeout_seconds=1)["status"] == "PASS"

        delayed_thread = threading.Thread(target=lambda: time.sleep(2), daemon=True)
        delayed_thread.start()
        delayed = FakeClient(delayed_thread)
        started = time.monotonic()
        try:
            bounded_stop_trainer_client(delayed, timeout_seconds=.05,
                                        evidence_path=root / "delayed.json")
            raise AssertionError("delayed thread must fail closed")
        except ShutdownTimeout:
            delayed_elapsed = time.monotonic() - started
        assert delayed_elapsed < 1.2
        evidence = json.loads((root / "delayed.json").read_text())
        assert evidence["status"] == "FAIL" and evidence["bounded"] is True

        # Real Agentlace broadcast receiver: no server, recv timeout wakes after kill.
        broadcast = BoundedBroadcastClient("127.0.0.1", port=5596)
        broadcast.async_start(lambda payload: None)
        real = FakeClient()
        real.broadcast_client = broadcast
        real_result = bounded_stop_trainer_client(real, timeout_seconds=3,
                                                  evidence_path=root / "real.json")
        assert real_result["status"] == "PASS" and not broadcast.thread.is_alive()

        # Separate actor-like process observes learner stop and exits naturally.
        child_code = r'''import sys,time,threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from training.peg_insertion.actor_shutdown import bounded_stop_trainer_client
class S:
 def close(self,**kwargs):pass
class C:
 def destroy(self,**kwargs):pass
class B:pass
class T:pass
c=T();c.update_thread=None;c.broadcast_client=B();c.broadcast_client.is_kill=False
c.broadcast_client.socket=S();c.broadcast_client.context=C()
def worker():
 while not c.broadcast_client.is_kill:time.sleep(.001)
c.broadcast_client.thread=threading.Thread(target=worker,daemon=True);c.broadcast_client.thread.start()
stop=Path(sys.argv[2])
while not stop.exists():time.sleep(.001)
bounded_stop_trainer_client(c,timeout_seconds=2,evidence_path=Path(sys.argv[3]))
'''
        stop = root / "learner_stop"
        child = subprocess.Popen([sys.executable, "-c", child_code, str(ROOT), str(stop),
                                  str(root / "child.json")])
        time.sleep(.05)
        started = time.monotonic(); stop.write_text("stop\n")
        child.wait(timeout=10)
        actor_exit = time.monotonic() - started
        assert child.returncode == 0 and actor_exit <= 10

        production = (ROOT / "training/peg_insertion/run_peg_training.py").read_text()
        smoke = (ROOT / "scripts/run_peg_f3_v2_rlpd_smoke.py").read_text()
        assert "install_bounded_actor_shutdown" in production
        assert "BoundedTrainerClient" in smoke and "actor_cleanup_complete.json" in smoke

        print(json.dumps({
            "PEG_F3_ACTOR_SHUTDOWN_FIX": "PASS",
            "STOP_SIGNAL_HANDSHAKE": "PASS",
            "TRAINERCLIENT_STOP": "PASS",
            "BROADCAST_THREAD_JOIN": "PASS",
            "JOIN_BOUNDED": "PASS",
            "STOP_IDEMPOTENT": "PASS",
            "DELAYED_THREAD_TIMEOUT": "PASS",
            "ACTOR_EXIT_TIME_SEC": actor_exit,
            "RESIDUAL_PROCESSES": "NONE",
            "PRODUCTION_REGRESSION": "PASS",
            "normal_stop_seconds": normal_elapsed,
            "delayed_timeout_seconds": delayed_elapsed,
        }, indent=2))


if __name__ == "__main__":
    main()
