"""Formal-only local guards. No JAX initialization, training math or W&B calls."""
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time

TIMEOUT_SECONDS = 120
SNAPSHOT_SECONDS = (30, 60, 110)
LOSS_KEYS = ('critic.critic_loss', 'actor.actor_loss', 'temperature.temperature_loss')


class MonitorFailure(RuntimeError):
    pass


class FormalMonitor:
    def __init__(self, directory, start_update, *, rng, sync, replay, observer,
                 clock=time.perf_counter, timer=threading.Timer, hard_exit=os._exit):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index = start_update
        self.previous_rng = None
        self.rng, self.sync, self.replay, self.observer = rng, sync, replay, observer
        self.clock, self.timer, self.hard_exit = clock, timer, hard_exit
        self.failed = False
        self.failure_lock = threading.Lock()

    def fail(self, reason):
        payload = dict(formal_status='FAIL', reason=reason, failed_update=self.index,
                       pid=os.getpid(), timestamp=datetime.now(timezone.utc).isoformat())
        # First failure wins; a later cleanup exception cannot erase its cause.
        with self.failure_lock:
            if self.failed:
                return
            self.failed = True
            with (self.directory/'formal_status.json').open('x') as stream:
                json.dump(payload, stream); stream.flush(); os.fsync(stream.fileno())

    def update(self, call, agent):
        if self.failed:
            raise MonitorFailure('monitor already failed; retry prohibited')
        started = self.clock()
        guard = threading.Lock()
        done = False
        observation = self.observer(self.directory, self.index)

        def timeout():
            nonlocal done
            with guard:
                if done:
                    return
                done = True
            try:
                self.fail('update_timeout')
            finally:
                # Native device waits cannot be interrupted with a Python exception.
                # No checkpoint is attempted while update state is incomplete.
                self.hard_exit(124)

        watchdog = self.timer(TIMEOUT_SECONDS, timeout)
        watchdog.daemon = True
        watchdog.start()
        try:
            observation.arm(delays=SNAPSHOT_SECONDS)
            before = self.rng(agent)
            if self.previous_rng is not None and before != self.previous_rng:
                raise MonitorFailure('rng_continuity_failure')
            observation.phase = 'before_update_high_utd'
            dispatch = self.clock()
            new_agent, info = call()
            returned = self.clock()
            observation.phase = 'full_synchronization'
            self.sync(new_agent)
            self.sync(info)
            synchronized = self.clock()
            observation.phase = 'after_update_checks'
            after = self.rng(new_agent)
            if before == after:
                raise MonitorFailure('rng_continuity_failure')
            flat = {}
            def flatten(prefix, value):
                if isinstance(value, Mapping):
                    for key, item in value.items():
                        flatten(prefix+'.'+str(key) if prefix else str(key), item)
                else:
                    flat[prefix] = value
            flatten('', info)
            try:
                losses = [float(flat[key]) for key in LOSS_KEYS]
                if not all(math.isfinite(value) for value in losses):
                    raise ValueError('non-finite')
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise MonitorFailure('non_finite_loss') from exc
            size, actor_steps = self.replay()
            record = dict(global_update=self.index, timestamp=datetime.now(timezone.utc).isoformat(),
                          dispatch_seconds=returned-dispatch, sync_seconds=synchronized-returned,
                          true_update_seconds=synchronized-started, rng_before=before, rng_after=after,
                          critic_loss=losses[0], actor_loss=losses[1], temperature_loss=losses[2],
                          finite=True, replay_size=size, actor_steps=actor_steps,
                          startup_window=0 <= self.index < 10, pid=os.getpid())
            # The timer also covers observation/validation, so no silent hang after sync.
            if self.clock()-started >= TIMEOUT_SECONDS:
                raise MonitorFailure('update_timeout')
            with guard:
                if done:
                    raise MonitorFailure('update_timeout')
                done = True
            with (self.directory/'formal_monitor_updates.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
            self.previous_rng = after
            self.index += 1
            return new_agent, info
        except BaseException as exc:
            with guard:
                done = True
            if not self.failed:
                reason = str(exc) if isinstance(exc, MonitorFailure) else 'update_exception'
                self.fail(reason)
            raise
        finally:
            watchdog.cancel()
            observation.close()
